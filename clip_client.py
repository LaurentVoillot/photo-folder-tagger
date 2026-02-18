"""
Client CLIP local pour le mode Balade.
Utilise open_clip (ViT-B-16) sur MPS/GPU Apple Silicon.
Reconnaissance zero-shot via vocabulaire de tags prédéfini.
~46ms/photo sur M1 Max — aucun serveur externe requis.
"""

import logging
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Import conditionnel — pas obligatoire si mode Balade non utilisé
try:
    import torch
    import open_clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    logger.warning("open_clip ou torch non installé — mode Balade indisponible")

# Vocabulaire par défaut (enrichissable via config.yaml)
DEFAULT_VOCABULARY = [
    # Nature - végétaux
    "rose", "fleur", "arbre", "forêt", "herbe", "feuilles", "branches", "végétation",
    "champignon", "mousse", "fougère", "cactus", "lavande", "tournesol", "blé",
    # Nature - lieux
    "montagne", "plage", "mer", "lac", "rivière", "cascade", "falaise",
    "champ", "désert", "colline", "vallée", "prairie", "marais", "dune",
    # Ciel / météo
    "ciel bleu", "nuages", "orage", "brume", "brouillard", "arc-en-ciel",
    "lever de soleil", "coucher de soleil", "nuit étoilée", "pleine lune",
    # Lumière
    "lumière dorée", "lumière naturelle", "contre-jour", "ombre portée",
    "reflet dans l'eau", "rayons de soleil",
    # Saison
    "printemps", "été", "automne", "hiver", "neige", "givre", "feuilles mortes",
    # Couleurs dominantes
    "rouge", "rose pâle", "orange", "jaune", "vert", "bleu", "violet",
    "blanc", "noir", "gris", "beige", "doré",
    # Composition / style
    "macro", "gros plan", "grand angle", "panoramique", "bokeh",
    "symétrie", "reflet", "silhouette", "texture", "motif répété",
    # Contexte balade
    "chemin", "sentier", "pont", "clôture", "mur en pierre", "bord de route",
    "parc", "jardin", "verger", "vignoble",
    # Animaux balade
    "oiseau", "insecte", "papillon", "abeille", "libellule", "escargot",
]


class ClipClient:
    """Client CLIP local pour taguage rapide mode Balade."""

    def __init__(self, config: dict):
        clip_cfg = config.get("clip", {})
        self.model_name = clip_cfg.get("model", "ViT-B-16")
        self.pretrained = clip_cfg.get("pretrained", "openai")
        self.top_k = clip_cfg.get("top_k", 8)
        self.vocabulary = clip_cfg.get("vocabulary", DEFAULT_VOCABULARY)

        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features = None
        self._device = None
        self._lock = threading.Lock()
        self._loaded = False

    def _ensure_loaded(self):
        """Charge le modèle CLIP au premier appel (lazy loading)."""
        if self._loaded:
            return True
        if not _CLIP_AVAILABLE:
            return False

        with self._lock:
            if self._loaded:
                return True
            try:
                logger.info(f"Chargement CLIP {self.model_name} ({self.pretrained})…")
                t0 = time.perf_counter()

                # Choisir le device : MPS > CUDA > CPU
                import torch
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

                # Pré-encoder tout le vocabulaire (fait une seule fois)
                self._encode_vocabulary()

                # Warm-up MPS (compile les shaders Metal au 1er appel)
                self._warmup()

                elapsed = time.perf_counter() - t0
                logger.info(
                    f"CLIP prêt sur {self._device} en {elapsed:.1f}s "
                    f"({len(self.vocabulary)} tags encodés)"
                )
                self._loaded = True
                return True

            except Exception as e:
                logger.error(f"Impossible de charger CLIP : {e}")
                return False

    def _encode_vocabulary(self):
        """Encode tous les tags du vocabulaire en vecteurs texte."""
        import torch
        with torch.no_grad():
            tokens = self._tokenizer(self.vocabulary).to(self._device)
            self._text_features = self._model.encode_text(tokens)
            self._text_features /= self._text_features.norm(dim=-1, keepdim=True)

    def _warmup(self):
        """Un passage à vide pour compiler les shaders MPS."""
        import torch
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_t = self._preprocess(dummy).unsqueeze(0).to(self._device)
        with torch.no_grad():
            _ = self._model.encode_image(img_t)

    def reload_vocabulary(self, new_vocabulary: list[str]):
        """Recharge le vocabulaire (après modification dans les paramètres)."""
        if not self._loaded:
            self.vocabulary = new_vocabulary
            return
        with self._lock:
            self.vocabulary = new_vocabulary
            self._encode_vocabulary()
            logger.info(f"Vocabulaire CLIP rechargé : {len(new_vocabulary)} tags")

    def check_available(self) -> tuple[bool, str]:
        """Vérifie si CLIP est disponible. Retourne (ok, message)."""
        if not _CLIP_AVAILABLE:
            return False, "open_clip non installé (pip install open_clip_torch)"
        ok = self._ensure_loaded()
        if ok:
            return True, f"CLIP {self.model_name} sur {self._device}"
        return False, "Impossible de charger le modèle CLIP"

    def generate_tags(self, image_path: Path) -> list[str]:
        """
        Génère des tags pour une image via CLIP zero-shot.

        Returns:
            Liste de tags (avec suffixe si configuré), vide en cas d'erreur.
        """
        if not self._ensure_loaded():
            return []

        import torch
        try:
            t0 = time.perf_counter()

            # Ouvrir et prétraiter l'image
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

            elapsed = time.perf_counter() - t0
            logger.debug(f"CLIP {image_path.name}: {elapsed*1000:.0f}ms → {tags}")
            return tags

        except Exception as e:
            logger.error(f"Erreur CLIP sur {image_path.name}: {e}")
            return []

    def unload(self):
        """Libère le modèle de la mémoire GPU."""
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
            logger.info("Modèle CLIP déchargé")
