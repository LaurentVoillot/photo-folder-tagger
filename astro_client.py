"""
Client Astro pour le mode Astronomie.
Deux niveaux d'analyse selon les métadonnées disponibles :

  1. Toujours : estimation du FOV depuis les EXIF (focal length + capteur)
  2. CLIP zero-shot sur vocabulaire d'objets du ciel (nébuleuses, galaxies,
     amas, planètes, types de photos astro)
  3. Si RA/Dec présents dans les EXIF (boîtier goto / monture) : croisement
     avec le catalogue OpenNGC pour identification précise Messier / NGC / IC

~50ms/photo sur M1 Max (CLIP) + 2ms calcul EXIF + ~5ms OpenNGC si dispo.
"""

import logging
import math
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

try:
    import torch
    import open_clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    logger.warning("open_clip ou torch non installé — mode Astro indisponible")

try:
    import opennugc
    _OPENNUGC_AVAILABLE = True
except ImportError:
    _OPENNUGC_AVAILABLE = False
    logger.info("opennugc non installé — identification Messier/NGC désactivée (pip install opennugc)")

# ---------------------------------------------------------------------------
# Vocabulaire CLIP astro (zero-shot)
# ---------------------------------------------------------------------------

DEFAULT_ASTRO_VOCABULARY = [
    # ── Objets deep-sky ──────────────────────────────────────────────────────
    "nébuleuse",
    "nébuleuse planétaire",
    "nébuleuse d'émission",
    "nébuleuse par réflexion",
    "nébuleuse obscure",
    "galaxie spirale",
    "galaxie elliptique",
    "galaxie irrégulière",
    "galaxie naine",
    "amas globulaire",
    "amas ouvert",
    "amas de galaxies",
    "rémanent de supernova",
    "région HII",
    # ── Objets du système solaire ─────────────────────────────────────────────
    "lune",
    "lune croissant",
    "pleine lune",
    "surface lunaire",
    "cratères lunaires",
    "soleil",
    "éclipse solaire",
    "éclipse lunaire",
    "jupiter",
    "saturne",
    "saturne et ses anneaux",
    "mars",
    "vénus",
    "comète",
    "astéroïde",
    "météore",
    "pluie de météores",
    # ── Étoiles ───────────────────────────────────────────────────────────────
    "étoile brillante",
    "étoile double",
    "étoile variable",
    "constellation",
    "voie lactée",
    "centre galactique",
    "nuages de magelllan",
    "trou noir",
    # ── Types de clichés ──────────────────────────────────────────────────────
    "photo grand champ",
    "photo longue pose",
    "photo en narrowband",
    "photo en Ha",
    "photo en OIII",
    "photo en SII",
    "photo RGB",
    "photo L-RVBI",
    "image composée",
    "mosaïque",
    # ── Contexte / conditions ─────────────────────────────────────────────────
    "ciel étoilé",
    "ciel laiteux",
    "ciel noir profond",
    "horizon",
    "pollution lumineuse",
    "aurore boréale",
    "nuit",
    # ── Composition ───────────────────────────────────────────────────────────
    "paysage nocturne",
    "filés d'étoiles",
    "panorama astro",
    "astropaysage",
]

# Taille capteur par défaut (plein format 35mm)
DEFAULT_SENSOR_WIDTH_MM  = 36.0
DEFAULT_SENSOR_HEIGHT_MM = 24.0

# Rayon de recherche OpenNGC (degrés) autour du centre de l'image
NGC_SEARCH_RADIUS_DEG = 5.0


class AstroClient:
    """Client IA local pour taguage de photos d'astronomie."""

    def __init__(self, config: dict):
        astro_cfg = config.get("astro", {})
        self.model_name  = astro_cfg.get("model", "ViT-L-14")
        self.pretrained  = astro_cfg.get("pretrained", "openai")
        self.top_k       = astro_cfg.get("top_k", 8)
        self.vocabulary  = astro_cfg.get("vocabulary", DEFAULT_ASTRO_VOCABULARY)

        self.sensor_width_mm  = astro_cfg.get("sensor_width_mm",  DEFAULT_SENSOR_WIDTH_MM)
        self.sensor_height_mm = astro_cfg.get("sensor_height_mm", DEFAULT_SENSOR_HEIGHT_MM)
        self.use_ngc_catalog  = astro_cfg.get("use_ngc_catalog", True)
        self.ngc_search_radius = astro_cfg.get("ngc_search_radius_deg", NGC_SEARCH_RADIUS_DEG)

        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features = None
        self._device = None
        self._lock = threading.Lock()
        self._loaded = False

        self._ngc_db = None   # Cache OpenNGC

    # -------------------------------------------------------------------------
    # Chargement CLIP (lazy)
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if not _CLIP_AVAILABLE:
            return False
        with self._lock:
            if self._loaded:
                return True
            try:
                logger.info(f"Chargement CLIP {self.model_name} ({self.pretrained}) pour mode Astro…")
                t0 = time.perf_counter()

                if torch.backends.mps.is_available():
                    self._device = "mps"
                elif torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"

                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    self.model_name, pretrained=self.pretrained
                )
                self._model = self._model.to(self._device).eval()
                self._tokenizer = open_clip.get_tokenizer(self.model_name)
                self._encode_vocabulary()
                self._warmup()

                elapsed = time.perf_counter() - t0
                logger.info(
                    f"CLIP Astro prêt sur {self._device} en {elapsed:.1f}s "
                    f"({len(self.vocabulary)} tags encodés)"
                )
                self._loaded = True
                return True
            except Exception as e:
                logger.error(f"Impossible de charger CLIP Astro : {e}")
                return False

    def _encode_vocabulary(self):
        import torch
        with torch.no_grad():
            tokens = self._tokenizer(self.vocabulary).to(self._device)
            self._text_features = self._model.encode_text(tokens)
            self._text_features /= self._text_features.norm(dim=-1, keepdim=True)

    def _warmup(self):
        import torch
        dummy = Image.new("RGB", (224, 224), color=(0, 0, 20))
        img_t = self._preprocess(dummy).unsqueeze(0).to(self._device)
        with torch.no_grad():
            _ = self._model.encode_image(img_t)

    # -------------------------------------------------------------------------
    # Interface publique
    # -------------------------------------------------------------------------

    def check_available(self) -> tuple[bool, str]:
        if not _CLIP_AVAILABLE:
            return False, "open_clip non installé (pip install open_clip_torch)"
        ok = self._ensure_loaded()
        if ok:
            ngc_status = " + OpenNGC" if _OPENNUGC_AVAILABLE else ""
            return True, f"CLIP {self.model_name} sur {self._device}{ngc_status}"
        return False, "Impossible de charger le modèle CLIP Astro"

    def generate_tags(self, image_path: Path) -> list[str]:
        """
        Génère les tags astro pour une image :
          - Tags CLIP zero-shot (vocabulaire astro)
          - FOV estimé depuis les EXIF (si focal length disponible)
          - Objets Messier/NGC identifiés via OpenNGC (si RA/Dec dans EXIF)
        """
        if not self._ensure_loaded():
            return []

        import torch
        tags = []
        t0 = time.perf_counter()

        try:
            # ── 1. Tags CLIP zero-shot ────────────────────────────────────────
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                img_features = self._model.encode_image(img_tensor)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                similarities = (img_features @ self._text_features.T)[0]

            top_k = min(self.top_k, len(self.vocabulary))
            top_indices = similarities.topk(top_k).indices.cpu().tolist()
            tags = [self.vocabulary[i] for i in top_indices]

            # ── 2. FOV depuis EXIF ────────────────────────────────────────────
            fov_tags = self._fov_tags_from_exif(image_path)
            tags.extend(fov_tags)

            # ── 3. Objets NGC/Messier via RA/Dec EXIF ─────────────────────────
            ngc_tags = self._ngc_tags_from_exif(image_path)
            tags.extend(ngc_tags)

            elapsed = time.perf_counter() - t0
            logger.debug(f"Astro {image_path.name}: {elapsed*1000:.0f}ms → {tags}")
            return tags

        except Exception as e:
            logger.error(f"Erreur AstroClient sur {image_path.name}: {e}")
            return tags  # retourne ce qu'on a déjà calculé

    # -------------------------------------------------------------------------
    # FOV depuis EXIF
    # -------------------------------------------------------------------------

    def _fov_tags_from_exif(self, image_path: Path) -> list[str]:
        """
        Calcule le FOV (angle de champ) depuis la focale EXIF.
        Priorité : FocalLengthIn35mmFilm (équivalent plein format) >
                   FocalLength + dimensions capteur configurées.
        Retourne des tags du type : 'FOV 2.1° × 1.4°' ou 'FOV 45° × 30°'.
        """
        try:
            with Image.open(image_path) as img:
                exif_raw = img._getexif()

            if not exif_raw:
                return []

            # Construire un dict lisible
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}

            focal_mm: Optional[float] = None
            sensor_w = self.sensor_width_mm
            sensor_h = self.sensor_height_mm

            # Méthode 1 : FocalLengthIn35mmFilm → focale équivalente 35mm
            # → on peut calculer le FOV directement avec les dims 35mm standard
            fl35 = exif.get("FocalLengthIn35mmFilm")
            if fl35 and float(fl35) > 0:
                focal_mm = float(fl35)
                sensor_w = DEFAULT_SENSOR_WIDTH_MM
                sensor_h = DEFAULT_SENSOR_HEIGHT_MM
            else:
                # Méthode 2 : FocalLength réelle + dims capteur configurées
                fl = exif.get("FocalLength")
                if fl is not None:
                    # Peut être un IFDRational ou un tuple (num, den)
                    if hasattr(fl, "numerator"):
                        focal_mm = fl.numerator / fl.denominator if fl.denominator else None
                    elif isinstance(fl, tuple):
                        focal_mm = fl[0] / fl[1] if fl[1] else None
                    else:
                        focal_mm = float(fl)

            if not focal_mm or focal_mm <= 0:
                return []

            fov_h = math.degrees(2 * math.atan(sensor_w / (2 * focal_mm)))
            fov_v = math.degrees(2 * math.atan(sensor_h / (2 * focal_mm)))

            # Format lisible selon la taille du champ
            def _fmt(deg: float) -> str:
                if deg < 1.0:
                    return f"{deg * 60:.1f}'"      # minutes d'arc
                return f"{deg:.1f}°"

            fov_tag = f"FOV {_fmt(fov_h)} × {_fmt(fov_v)}"
            logger.debug(f"FOV calculé : {fov_tag} (f={focal_mm:.0f}mm)")
            return [fov_tag]

        except Exception as e:
            logger.debug(f"Calcul FOV impossible pour {image_path.name}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Identification NGC / Messier depuis RA/Dec EXIF
    # -------------------------------------------------------------------------

    def _ngc_tags_from_exif(self, image_path: Path) -> list[str]:
        """
        Cherche les objets Messier/NGC/IC dans le champ de l'image,
        en utilisant les coordonnées RA/Dec si présentes dans les EXIF.
        Certains boîtiers astro (ZWO, QHYCCD) ou logiciels (SGP, N.I.N.A.)
        écrivent ces coordonnées dans des champs EXIF/IPTC personnalisés.
        """
        if not self.use_ngc_catalog or not _OPENNUGC_AVAILABLE:
            return []

        try:
            with Image.open(image_path) as img:
                exif_raw = img._getexif()

            if not exif_raw:
                return []

            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}

            # GPSInfo contient parfois RA/Dec pour les photos astro
            # (certains softs écrivent le centrage en GPS !)
            ra_deg, dec_deg = self._extract_ra_dec(exif)
            if ra_deg is None:
                return []

            return self._query_ngc(ra_deg, dec_deg)

        except Exception as e:
            logger.debug(f"Recherche NGC impossible pour {image_path.name}: {e}")
            return []

    def _extract_ra_dec(self, exif: dict) -> tuple[Optional[float], Optional[float]]:
        """
        Tente d'extraire RA/Dec depuis les champs GPS de l'EXIF.
        Convention utilisée par certains logiciels d'acquisition astro :
        - GPS Latitude  → Déclinaison (Dec)
        - GPS Longitude → Ascension droite (RA en degrés)
        """
        try:
            gps = exif.get("GPSInfo")
            if not gps:
                return None, None

            gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}

            def _dms_to_deg(dms) -> Optional[float]:
                if dms is None:
                    return None
                try:
                    d, m, s = dms
                    d = d.numerator / d.denominator if hasattr(d, "numerator") else float(d)
                    m = m.numerator / m.denominator if hasattr(m, "numerator") else float(m)
                    s = s.numerator / s.denominator if hasattr(s, "numerator") else float(s)
                    return d + m / 60 + s / 3600
                except Exception:
                    return None

            lat = _dms_to_deg(gps_tags.get("GPSLatitude"))
            lat_ref = gps_tags.get("GPSLatitudeRef", "N")
            lon = _dms_to_deg(gps_tags.get("GPSLongitude"))
            lon_ref = gps_tags.get("GPSLongitudeRef", "E")

            if lat is None or lon is None:
                return None, None

            dec_deg = lat if lat_ref == "N" else -lat
            # RA en heures → degrés
            ra_h = lon if lon_ref == "E" else 360 - lon
            ra_deg = ra_h  # déjà en degrés si lon ∈ [0, 360]

            return ra_deg, dec_deg

        except Exception:
            return None, None

    def _query_ngc(self, ra_deg: float, dec_deg: float) -> list[str]:
        """Interroge le catalogue OpenNGC autour de (ra_deg, dec_deg)."""
        if self._ngc_db is None:
            try:
                self._ngc_db = opennugc.OpenNGC()
                logger.info("Catalogue OpenNGC chargé")
            except Exception as e:
                logger.warning(f"Impossible de charger OpenNGC : {e}")
                return []

        tags = []
        try:
            objects = self._ngc_db.get_objects()
            for obj in objects:
                try:
                    obj_ra  = float(obj.ra)   * 15  # heures → degrés
                    obj_dec = float(obj.dec)
                except (TypeError, ValueError):
                    continue

                # Distance angulaire approchée (valid pour petits angles)
                d_ra  = abs(ra_deg  - obj_ra)  * math.cos(math.radians(dec_deg))
                d_dec = abs(dec_deg - obj_dec)
                dist  = math.sqrt(d_ra**2 + d_dec**2)

                if dist <= self.ngc_search_radius:
                    name = obj.name or ""
                    obj_type = getattr(obj, "type", "") or ""
                    common = getattr(obj, "commonname", "") or ""

                    label = name
                    if common:
                        label = f"{name} ({common})"
                    if obj_type:
                        label = f"{label} [{obj_type}]"

                    tags.append(label)
                    logger.debug(f"NGC trouvé : {label} à {dist:.2f}° du centre")

            if tags:
                logger.info(f"OpenNGC : {len(tags)} objet(s) identifié(s) dans le champ")

        except Exception as e:
            logger.warning(f"Erreur requête OpenNGC : {e}")

        return tags

    # -------------------------------------------------------------------------
    # Rechargement du vocabulaire
    # -------------------------------------------------------------------------

    def reload_vocabulary(self, new_vocabulary: list[str]):
        if not self._loaded:
            self.vocabulary = new_vocabulary
            return
        with self._lock:
            self.vocabulary = new_vocabulary
            self._encode_vocabulary()
            logger.info(f"Vocabulaire Astro rechargé : {len(new_vocabulary)} tags")

    def unload(self):
        if not self._loaded:
            return
        with self._lock:
            import torch
            del self._model
            del self._text_features
            self._model = None
            self._text_features = None
            self._loaded = False
            if self._device == "mps":
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Modèle CLIP Astro déchargé")
