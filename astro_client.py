"""
Client Astro — pipeline plate solving + SIMBAD + fallback Ollama.

Pipeline par ordre de priorité :
  1. Détection d'étoiles (sep / photutils) + plate solving offline (pkg `astrometry`)
     → coordonnées RA/Dec exactes → FOV précis
  2. Cone search SIMBAD autour du centre de l'image
     → identification des objets deep-sky (NGC, IC, Messier, nébuleuses, amas…)
  3. Fallback : RA/Dec depuis les EXIF (N.I.N.A., SGP) + OpenNGC offline
     → si plate solving impossible (image sans étoiles visibles)
  4. Fallback final : Ollama qwen2.5vl avec prompt spécialisé astro
     → planètes, nébuleuses diffuses plein champ, images saturées

FOV calculé depuis FocalLengthIn35mmFilm EXIF (priorité) ou FocalLength + capteur.

Dépendances optionnelles :
  pip install astrometry    # plate solving offline (~5 Go d'index Tycho-2 auto-dl)
  pip install astroquery    # SIMBAD cone search (internet requis)
  pip install sep           # détection d'étoiles rapide (C)
  pip install photutils     # détection d'étoiles pure Python (fallback sep)
  pip install pyongc        # catalogue NGC/IC/Messier offline (fallback SIMBAD)
"""

import logging
import math
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ExifTags
import numpy as np

logger = logging.getLogger(__name__)

# ── Disponibilité des dépendances optionnelles ────────────────────────────────

try:
    import astrometry
    _ASTROMETRY_AVAILABLE = True
except ImportError:
    _ASTROMETRY_AVAILABLE = False
    logger.info("astrometry non installé — plate solving désactivé (pip install astrometry)")

try:
    import sep
    _SEP_AVAILABLE = True
except ImportError:
    _SEP_AVAILABLE = False

try:
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    _PHOTUTILS_AVAILABLE = True
except ImportError:
    _PHOTUTILS_AVAILABLE = False

try:
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    _SIMBAD_AVAILABLE = True
except ImportError:
    _SIMBAD_AVAILABLE = False
    logger.info("astroquery non installé — SIMBAD désactivé (pip install astroquery)")

try:
    import pyongc
    _OPENNUGC_AVAILABLE = True
except ImportError:
    _OPENNUGC_AVAILABLE = False

# Taille capteur par défaut (plein format 35mm)
DEFAULT_SENSOR_WIDTH_MM  = 36.0
DEFAULT_SENSOR_HEIGHT_MM = 24.0

# Types d'objets SIMBAD intéressants pour l'astrophoto (filtre les étoiles isolées)
SIMBAD_INTERESTING_TYPES = {
    # Nébuleuses
    "ISM", "Neb", "HII", "PN", "SNR", "MoC", "RNe", "DNe", "bub",
    # Galaxies
    "G", "GiG", "GiC", "GiP", "BiC", "ClG", "PaG", "IG",
    # Amas
    "GlC", "OpC", "Cl*", "As*",
    # Divers
    "AGN", "QSO", "BLL", "SyG",
}

# Noms Messier pour enrichir les tags (M42 → "M42 (Grande Nébuleuse d'Orion)")
MESSIER_NAMES = {
    "M1": "Nébuleuse du Crabe", "M2": "Amas M2", "M3": "Amas M3",
    "M4": "Amas M4", "M5": "Amas M5", "M6": "Amas du Papillon",
    "M7": "Amas de Ptolémée", "M8": "Nébuleuse de la Lagune",
    "M11": "Amas du Canard Sauvage", "M13": "Grand Amas d'Hercule",
    "M16": "Nébuleuse de l'Aigle", "M17": "Nébuleuse Oméga",
    "M20": "Nébuleuse Trifide", "M22": "Amas M22",
    "M27": "Nébuleuse de l'Haltère", "M31": "Galaxie d'Andromède",
    "M32": "M32 (satellite d'Andromède)", "M33": "Galaxie du Triangle",
    "M42": "Grande Nébuleuse d'Orion", "M43": "Nébuleuse De Mairan",
    "M44": "Amas de la Crèche", "M45": "Pléiades",
    "M51": "Galaxie du Tourbillon", "M57": "Nébuleuse de l'Anneau",
    "M63": "Galaxie du Tournesol", "M64": "Galaxie de l'Œil Noir",
    "M74": "Galaxie M74", "M76": "Petite Nébuleuse Haltère",
    "M77": "Galaxie M77", "M78": "Nébuleuse M78",
    "M81": "Galaxie de Bode", "M82": "Galaxie du Cigare",
    "M83": "Galaxie du Moulinet du Sud", "M87": "Galaxie M87",
    "M97": "Nébuleuse du Hibou", "M101": "Galaxie du Moulinet",
    "M104": "Galaxie du Sombrero", "M106": "Galaxie M106",
    "M110": "M110 (satellite d'Andromède)",
}


class AstroClient:
    """
    Client IA pour taguage de photos d'astronomie.
    Pipeline : plate solving → SIMBAD → OpenNGC → Ollama.
    """

    def __init__(self, config: dict):
        astro_cfg = config.get("astro", {})
        self.sensor_width_mm   = astro_cfg.get("sensor_width_mm",  DEFAULT_SENSOR_WIDTH_MM)
        self.sensor_height_mm  = astro_cfg.get("sensor_height_mm", DEFAULT_SENSOR_HEIGHT_MM)
        self.use_plate_solving = astro_cfg.get("use_plate_solving", True)
        self.use_simbad        = astro_cfg.get("use_simbad", True)
        self.use_ngc_catalog   = astro_cfg.get("use_ngc_catalog", True)
        self.simbad_radius_deg = astro_cfg.get("simbad_radius_deg", 1.0)
        self.ngc_search_radius = astro_cfg.get("ngc_search_radius_deg", 5.0)
        self.ollama_fallback   = astro_cfg.get("ollama_fallback", True)
        # Index astrometry : si None → télécharge automatiquement serie 5200
        self.astrometry_index_dir = astro_cfg.get("astrometry_index_dir", None)

        self._ngc_db = None
        self._simbad_client = None
        self._ollama_client = None
        self._config = config
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Interface publique
    # -------------------------------------------------------------------------

    def check_available(self) -> tuple[bool, str]:
        parts = []
        if _ASTROMETRY_AVAILABLE:
            parts.append("plate solving offline")
        if _SIMBAD_AVAILABLE:
            parts.append("SIMBAD")
        if _OPENNUGC_AVAILABLE:
            parts.append("OpenNGC")
        if self.ollama_fallback:
            parts.append("Ollama fallback")

        if not parts:
            return False, (
                "Aucune dépendance astro disponible. "
                "pip install astrometry astroquery sep"
            )
        return True, "Astro : " + " + ".join(parts)

    def generate_tags(self, image_path: Path) -> list[str]:
        """
        Génère les tags pour une photo astro :
          1. FOV depuis EXIF
          2. Plate solving → SIMBAD (si étoiles détectables)
          3. RA/Dec EXIF → OpenNGC (si plate solving impossible)
          4. Ollama fallback (si tout échoue)
        """
        tags = []
        t0 = time.perf_counter()

        try:
            # ── 1. FOV depuis EXIF ────────────────────────────────────────────
            fov_info = self._fov_from_exif(image_path)
            if fov_info:
                fov_h, fov_v, focal_mm = fov_info
                tags.append(self._fmt_fov(fov_h, fov_v))

            # ── 2. Plate solving + SIMBAD ─────────────────────────────────────
            solved_ra, solved_dec = None, None
            if self.use_plate_solving and _ASTROMETRY_AVAILABLE:
                fov_hint_deg = fov_info[0] if fov_info else None
                solved_ra, solved_dec = self._plate_solve(image_path, fov_hint_deg)

            if solved_ra is not None:
                logger.info(
                    f"Plate solving OK pour {image_path.name}: "
                    f"RA={solved_ra:.4f}° Dec={solved_dec:.4f}°"
                )
                if self.use_simbad and _SIMBAD_AVAILABLE:
                    simbad_tags = self._query_simbad(solved_ra, solved_dec)
                    tags.extend(simbad_tags)
                elif self.use_ngc_catalog and _OPENNUGC_AVAILABLE:
                    tags.extend(self._query_ngc(solved_ra, solved_dec))

            else:
                # ── 3. Fallback RA/Dec EXIF → OpenNGC ────────────────────────
                exif_ra, exif_dec = self._ra_dec_from_exif(image_path)
                if exif_ra is not None and self.use_ngc_catalog and _OPENNUGC_AVAILABLE:
                    logger.debug(f"Utilisation RA/Dec EXIF pour {image_path.name}")
                    tags.extend(self._query_ngc(exif_ra, exif_dec))

                # ── 4. Fallback Ollama ────────────────────────────────────────
                if not tags and self.ollama_fallback:
                    logger.debug(f"Fallback Ollama pour {image_path.name}")
                    ollama_tags = self._do_ollama_fallback(image_path)
                    tags.extend(ollama_tags)

            elapsed = time.perf_counter() - t0
            logger.debug(f"Astro {image_path.name}: {elapsed*1000:.0f}ms → {tags}")
            return tags

        except Exception as e:
            logger.error(f"Erreur AstroClient sur {image_path.name}: {e}")
            if not tags and self.ollama_fallback:
                return self._do_ollama_fallback(image_path)
            return tags

    # -------------------------------------------------------------------------
    # Plate solving (astrometry Python package, offline)
    # -------------------------------------------------------------------------

    def _plate_solve(
        self, image_path: Path, fov_hint_deg: Optional[float] = None
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Tente d'identifier les étoiles et de faire le plate solving offline.
        Retourne (ra_deg, dec_deg) du centre, ou (None, None) si échec.
        """
        try:
            # Chargement image en niveaux de gris numpy
            with Image.open(image_path) as img:
                gray = np.array(img.convert("L"), dtype=np.float32)

            # Détection des étoiles
            stars = self._detect_stars(gray)
            if len(stars) < 6:
                logger.debug(
                    f"Plate solving abandonné pour {image_path.name}: "
                    f"seulement {len(stars)} étoiles détectées (min 6)"
                )
                return None, None

            logger.debug(f"Plate solving {image_path.name}: {len(stars)} étoiles détectées")

            # Hints de taille depuis FOV ou hint large par défaut
            if fov_hint_deg and fov_hint_deg > 0:
                # FOV en degrés → arcsec/pixel approximatif
                h, w = gray.shape
                scale_arcsec = fov_hint_deg * 3600 / max(w, h)
                lower = scale_arcsec * 0.5
                upper = scale_arcsec * 2.0
            else:
                lower = 0.1   # arcsec/pixel
                upper = 180.0  # très grand champ

            # Hints de position depuis EXIF si disponibles
            position_hint = None
            exif_ra, exif_dec = self._ra_dec_from_exif(image_path)
            if exif_ra is not None:
                position_hint = astrometry.PositionHint(
                    ra_deg=exif_ra, dec_deg=exif_dec, radius_deg=5.0
                )

            # Chargement de l'index (série 5200 = Tycho-2 + Gaia, multi-échelle)
            if self.astrometry_index_dir:
                index_dir = Path(self.astrometry_index_dir)
                index_files = list(index_dir.glob("*.fits"))
            else:
                # Téléchargement automatique (~5 Go, une seule fois)
                try:
                    index_files = astrometry.series_5200.index_files(
                        cache_directory=str(Path.home() / ".cache" / "astrometry"),
                        scales={6, 7, 8},   # couvre ~0.3° à ~10°
                    )
                except Exception as e:
                    logger.warning(f"Impossible de charger l'index astrometry: {e}")
                    return None, None

            solver = astrometry.Solver(index_files)

            # Positions des étoiles triées par flux décroissant
            star_positions = [(float(s[0]), float(s[1])) for s in stars[:50]]

            solution = solver.solve(
                stars=star_positions,
                size_hint=astrometry.SizeHint(
                    lower_arcsec_per_pixel=lower,
                    upper_arcsec_per_pixel=upper,
                ),
                position_hint=position_hint,
                solution_parameters=astrometry.SolutionParameters(
                    logodds_callback=lambda logodds: (
                        astrometry.Action.STOP if logodds > 30 else astrometry.Action.CONTINUE
                    ),
                ),
            )

            if solution.has_match():
                match = solution.best_match()
                # Centre de l'image en RA/Dec
                h, w = gray.shape
                wcs = match.astropy_wcs()
                from astropy.wcs import WCS
                ra, dec = wcs.all_pix2world([[w / 2, h / 2]], 0)[0]
                return float(ra), float(dec)

            return None, None

        except Exception as e:
            logger.debug(f"Plate solving échoué pour {image_path.name}: {e}")
            return None, None

    def _detect_stars(self, gray: "np.ndarray") -> list:
        """
        Détecte les étoiles dans l'image en niveaux de gris.
        Essaie sep (rapide, C) puis photutils (pure Python).
        Retourne une liste de (x, y, flux) triés par flux décroissant.
        """
        if _SEP_AVAILABLE:
            return self._detect_stars_sep(gray)
        elif _PHOTUTILS_AVAILABLE:
            return self._detect_stars_photutils(gray)
        else:
            # Fallback très simple : détection de maxima locaux
            return self._detect_stars_simple(gray)

    def _detect_stars_sep(self, gray: "np.ndarray") -> list:
        try:
            data = gray.astype(np.float64)
            bkg = sep.Background(data)
            data_sub = data - bkg
            objects = sep.extract(data_sub, 1.5, err=bkg.globalrms, minarea=5)
            stars = sorted(
                [(float(o["x"]), float(o["y"]), float(o["flux"])) for o in objects],
                key=lambda s: -s[2]
            )
            return stars
        except Exception as e:
            logger.debug(f"sep échoué: {e}")
            return self._detect_stars_simple(gray)

    def _detect_stars_photutils(self, gray: "np.ndarray") -> list:
        try:
            mean, median, std = sigma_clipped_stats(gray)
            finder = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
            sources = finder(gray - median)
            if sources is None:
                return []
            stars = sorted(
                [(float(s["xcentroid"]), float(s["ycentroid"]), float(s["peak"])) for s in sources],
                key=lambda s: -s[2]
            )
            return stars
        except Exception as e:
            logger.debug(f"photutils échoué: {e}")
            return self._detect_stars_simple(gray)

    def _detect_stars_simple(self, gray: "np.ndarray") -> list:
        """Détection naïve : pixels très brillants dans une image redimensionnée."""
        try:
            # Réduire pour accélérer
            small = gray[::4, ::4]
            threshold = np.percentile(small, 99.5)
            ys, xs = np.where(small > threshold)
            # Regrouper les voisins proches
            stars = [(float(x * 4), float(y * 4), float(gray[y * 4, x * 4]))
                     for x, y in zip(xs, ys)]
            return sorted(stars, key=lambda s: -s[2])[:100]
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # SIMBAD cone search
    # -------------------------------------------------------------------------

    def _query_simbad(self, ra_deg: float, dec_deg: float) -> list[str]:
        """
        Interroge SIMBAD autour de (ra_deg, dec_deg) et retourne les tags
        des objets intéressants (nébuleuses, galaxies, amas — pas les étoiles).
        """
        try:
            if self._simbad_client is None:
                self._simbad_client = Simbad()
                self._simbad_client.add_votable_fields("otype", "ids")
                self._simbad_client.TIMEOUT = 10

            center = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg")
            results = self._simbad_client.query_region(
                center, radius=self.simbad_radius_deg * u.deg
            )

            if results is None:
                return []

            tags = []
            for row in results:
                otype = str(row["OTYPE"]).strip()
                main_id = str(row["MAIN_ID"]).strip()

                # Filtrer : garder seulement les objets non-stellaires
                if not any(otype.startswith(t) for t in SIMBAD_INTERESTING_TYPES):
                    continue

                # Chercher un identifiant Messier dans les IDs alternatifs
                ids_str = str(row.get("IDS", ""))
                messier_id = None
                for part in ids_str.split("|"):
                    part = part.strip()
                    if part.startswith("M ") or part.startswith("M\xa0"):
                        m_num = part.replace("M ", "M").replace("M\xa0", "M").strip()
                        if m_num in MESSIER_NAMES:
                            messier_id = m_num
                            break

                # Construire le tag
                label = main_id
                if messier_id:
                    common = MESSIER_NAMES.get(messier_id, "")
                    label = f"{messier_id} / {main_id} — {common}" if common else f"{messier_id} / {main_id}"
                elif otype:
                    label = f"{main_id} [{otype}]"

                tags.append(label)
                logger.debug(f"SIMBAD: {label}")

            if tags:
                logger.info(f"SIMBAD: {len(tags)} objet(s) identifié(s)")
            return tags

        except Exception as e:
            logger.warning(f"Erreur SIMBAD: {e}")
            return []

    # -------------------------------------------------------------------------
    # OpenNGC (fallback offline)
    # -------------------------------------------------------------------------

    def _query_ngc(self, ra_deg: float, dec_deg: float) -> list[str]:
        """Interroge le catalogue PyONGC autour de (ra_deg, dec_deg)."""
        if self._ngc_db is None:
            try:
                # listObjects() retourne tous les objets du catalogue NGC/IC/Messier
                self._ngc_db = pyongc.listObjects(printOut=False)
            except Exception as e:
                logger.warning(f"Impossible de charger PyONGC: {e}")
                return []

        tags = []
        try:
            for obj in self._ngc_db:
                try:
                    coords = obj.getCoords()
                    if coords is None or coords[0] is None:
                        continue
                    ra_hms, dec_dms = coords
                    # RA : (heures, min, sec) → degrés
                    obj_ra = (ra_hms[0] + ra_hms[1] / 60 + ra_hms[2] / 3600) * 15
                    # Dec : ('+'/'-', deg, min, sec) → degrés
                    sign = 1 if dec_dms[0] == '+' else -1
                    obj_dec = sign * (dec_dms[1] + dec_dms[2] / 60 + dec_dms[3] / 3600)
                except (TypeError, ValueError, IndexError):
                    continue

                d_ra  = abs(ra_deg - obj_ra) * math.cos(math.radians(dec_deg))
                d_dec = abs(dec_deg - obj_dec)
                if math.sqrt(d_ra**2 + d_dec**2) <= self.ngc_search_radius:
                    name   = obj.name or ""
                    common = obj.commonnames or ""
                    if isinstance(common, list):
                        common = ", ".join(common)
                    otype  = getattr(obj, "type", "") or ""
                    label  = f"{name} ({common})" if common else name
                    if otype:
                        label = f"{label} [{otype}]"
                    tags.append(label)

            if tags:
                logger.info(f"PyONGC: {len(tags)} objet(s) identifié(s)")
        except Exception as e:
            logger.warning(f"Erreur PyONGC: {e}")

        return tags

    # -------------------------------------------------------------------------
    # FOV depuis EXIF
    # -------------------------------------------------------------------------

    def _fov_from_exif(
        self, image_path: Path
    ) -> Optional[tuple[float, float, float]]:
        """
        Retourne (fov_h_deg, fov_v_deg, focal_mm) ou None.
        Priorité : FocalLengthIn35mmFilm > FocalLength + capteur config.
        """
        try:
            with Image.open(image_path) as img:
                exif_raw = img._getexif()
            if not exif_raw:
                return None
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}

            sensor_w = self.sensor_width_mm
            sensor_h = self.sensor_height_mm
            focal_mm: Optional[float] = None

            fl35 = exif.get("FocalLengthIn35mmFilm")
            if fl35 and float(fl35) > 0:
                focal_mm = float(fl35)
                sensor_w = DEFAULT_SENSOR_WIDTH_MM
                sensor_h = DEFAULT_SENSOR_HEIGHT_MM
            else:
                fl = exif.get("FocalLength")
                if fl is not None:
                    if hasattr(fl, "numerator"):
                        focal_mm = fl.numerator / fl.denominator if fl.denominator else None
                    elif isinstance(fl, tuple):
                        focal_mm = fl[0] / fl[1] if fl[1] else None
                    else:
                        focal_mm = float(fl)

            if not focal_mm or focal_mm <= 0:
                return None

            fov_h = math.degrees(2 * math.atan(sensor_w / (2 * focal_mm)))
            fov_v = math.degrees(2 * math.atan(sensor_h / (2 * focal_mm)))
            return fov_h, fov_v, focal_mm

        except Exception:
            return None

    def _fmt_fov(self, fov_h: float, fov_v: float) -> str:
        def _f(d: float) -> str:
            return f"{d * 60:.1f}'" if d < 1.0 else f"{d:.1f}°"
        return f"FOV {_f(fov_h)} × {_f(fov_v)}"

    # -------------------------------------------------------------------------
    # RA/Dec depuis EXIF (N.I.N.A., SGP via GPS)
    # -------------------------------------------------------------------------

    def _ra_dec_from_exif(
        self, image_path: Path
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Extrait RA/Dec depuis les champs GPS EXIF (convention N.I.N.A./SGP).
        GPS Latitude → Déclinaison, GPS Longitude → RA (degrés).
        """
        try:
            with Image.open(image_path) as img:
                exif_raw = img._getexif()
            if not exif_raw:
                return None, None
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
            gps = exif.get("GPSInfo")
            if not gps:
                return None, None

            gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}

            def _dms(val) -> Optional[float]:
                try:
                    d, m, s = val
                    to_f = lambda x: x.numerator / x.denominator if hasattr(x, "numerator") else float(x)
                    return to_f(d) + to_f(m) / 60 + to_f(s) / 3600
                except Exception:
                    return None

            lat = _dms(gps_tags.get("GPSLatitude"))
            lon = _dms(gps_tags.get("GPSLongitude"))
            if lat is None or lon is None:
                return None, None

            lat_ref = gps_tags.get("GPSLatitudeRef", "N")
            lon_ref = gps_tags.get("GPSLongitudeRef", "E")
            dec_deg = lat if lat_ref == "N" else -lat
            ra_deg  = lon if lon_ref == "E" else 360.0 - lon
            return ra_deg, dec_deg

        except Exception:
            return None, None

    # -------------------------------------------------------------------------
    # Fallback Ollama
    # -------------------------------------------------------------------------

    def _do_ollama_fallback(self, image_path: Path) -> list[str]:
        """Fallback Ollama avec prompt spécialisé astrophotographie."""
        if not self.ollama_fallback:
            return []
        try:
            if self._ollama_client is None:
                from ollama_client import OllamaClient
                cfg = dict(self._config)
                cfg.setdefault("prompt", {})["auto_prompt"] = (
                    "You are an expert astrophotographer and astronomer. "
                    "Analyze this astronomical image.\n"
                    "1. Identify the type of object: nebula, galaxy, star cluster, "
                    "   planet, comet, Milky Way, auroras, star trails, etc.\n"
                    "2. If recognizable, name the specific object (e.g. M42, NGC 891, "
                    "   Orion Nebula, Andromeda Galaxy, Saturn, Pleiades).\n"
                    "3. Describe key visual features: shape, color, structure.\n"
                    "4. Add technical tags: long exposure, narrowband, Ha, OIII, RGB, "
                    "   wide field, deep sky, mosaic, etc. if applicable.\n"
                    "Return only comma-separated tags, nothing else. "
                    "Use French for common names."
                )
                self._ollama_client = OllamaClient(cfg)
            return self._ollama_client.generate_tags(image_path)
        except Exception as e:
            logger.error(f"Ollama fallback astro échoué pour {image_path.name}: {e}")
            return []

    def unload(self):
        pass  # Pas de modèle en mémoire permanente dans ce client
