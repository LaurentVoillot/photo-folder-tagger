"""
Client BioCLIP pour le mode Animaux.
Utilise imageomics/bioclip (spécialisé taxonomie biologique, 454k espèces).
~40ms/photo sur M1 Max pour l'identification d'espèce.

Architecture :
  1. BioCLIP identifie l'espèce la plus probable parmi le vocabulaire taxonomique
  2. Si score de confiance trop bas → fallback Ollama pour description libre
  3. Tags contextuels (lieu, lumière, saison) ajoutés via CLIP standard
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

try:
    import torch
    import open_clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

# Taxonomie par défaut — espèces européennes courantes
# Format : "Nom scientifique": "Nom commun français"
DEFAULT_SPECIES = {
    # ── Insectes ────────────────────────────────────────────────────────────────
    "Apis mellifera": "abeille domestique",
    "Bombus terrestris": "bourdon terrestre",
    "Bombus lapidarius": "bourdon des pierres",
    "Bombus pascuorum": "bourdon des champs",
    "Xylocopa violacea": "abeille charpentière violette",
    "Vespa crabro": "frelon européen",
    "Vespa velutina": "frelon asiatique",
    "Vespula vulgaris": "guêpe commune",
    "Vanessa atalanta": "papillon vulcain",
    "Vanessa cardui": "belle-dame",
    "Papilio machaon": "machaon",
    "Pieris brassicae": "piéride du chou",
    "Pieris rapae": "piéride de la rave",
    "Aglais io": "paon du jour",
    "Aglais urticae": "petite tortue",
    "Gonepteryx rhamni": "citron",
    "Coccinella septempunctata": "coccinelle à 7 points",
    "Harmonia axyridis": "coccinelle asiatique",
    "Mantis religiosa": "mante religieuse",
    "Libellula depressa": "libellule déprimée",
    "Calopteryx splendens": "agrion splendide",
    "Formica rufa": "fourmi rousse des bois",
    "Gryllus campestris": "grillon des champs",
    "Tettigonia viridissima": "grande sauterelle verte",
    # ── Oiseaux ────────────────────────────────────────────────────────────────
    "Parus major": "mésange charbonnière",
    "Cyanistes caeruleus": "mésange bleue",
    "Erithacus rubecula": "rouge-gorge familier",
    "Turdus merula": "merle noir",
    "Turdus philomelos": "grive musicienne",
    "Corvus corax": "grand corbeau",
    "Corvus corone": "corneille noire",
    "Pica pica": "pie bavarde",
    "Garrulus glandarius": "geai des chênes",
    "Ardea cinerea": "héron cendré",
    "Ardea alba": "grande aigrette",
    "Hirundo rustica": "hirondelle rustique",
    "Delichon urbicum": "hirondelle de fenêtre",
    "Falco tinnunculus": "faucon crécerelle",
    "Falco peregrinus": "faucon pèlerin",
    "Ciconia ciconia": "cigogne blanche",
    "Columba palumbus": "pigeon ramier",
    "Columba livia": "pigeon biset",
    "Anas platyrhynchos": "canard colvert",
    "Cygnus olor": "cygne tuberculé",
    "Accipiter nisus": "épervier d'Europe",
    "Buteo buteo": "buse variable",
    "Milvus milvus": "milan royal",
    "Strix aluco": "chouette hulotte",
    "Tyto alba": "effraie des clochers",
    "Upupa epops": "huppe fasciée",
    "Alcedo atthis": "martin-pêcheur d'Europe",
    "Passer domesticus": "moineau domestique",
    "Fringilla coelebs": "pinson des arbres",
    "Carduelis carduelis": "chardonneret élégant",
    "Sitta europaea": "sittelle torchepot",
    "Troglodytes troglodytes": "troglodyte mignon",
    # ── Mammifères ─────────────────────────────────────────────────────────────
    "Vulpes vulpes": "renard roux",
    "Capreolus capreolus": "chevreuil européen",
    "Cervus elaphus": "cerf élaphe",
    "Dama dama": "daim",
    "Sus scrofa": "sanglier",
    "Lepus europaeus": "lièvre d'Europe",
    "Oryctolagus cuniculus": "lapin de garenne",
    "Erinaceus europaeus": "hérisson d'Europe",
    "Meles meles": "blaireau européen",
    "Martes foina": "fouine",
    "Mustela putorius": "putois d'Europe",
    "Mustela nivalis": "belette d'Europe",
    "Lutra lutra": "loutre d'Europe",
    "Castor fiber": "castor d'Europe",
    "Sciurus vulgaris": "écureuil roux",
    "Myocastor coypus": "ragondin",
    "Ursus arctos": "ours brun",
    "Lynx lynx": "lynx boréal",
    "Canis lupus": "loup gris",
    # ── Reptiles / Amphibiens ──────────────────────────────────────────────────
    "Lacerta viridis": "lézard vert occidental",
    "Lacerta agilis": "lézard agile",
    "Podarcis muralis": "lézard des murailles",
    "Natrix natrix": "couleuvre à collier",
    "Vipera berus": "vipère péliade",
    "Vipera aspis": "vipère aspic",
    "Rana temporaria": "grenouille rousse",
    "Rana dalmatina": "grenouille agile",
    "Bufo bufo": "crapaud commun",
    "Triturus cristatus": "triton crêté",
    "Salamandra salamandra": "salamandre tachetée",
    # ── Poissons ───────────────────────────────────────────────────────────────
    "Salmo trutta": "truite commune",
    "Salmo salar": "saumon atlantique",
    "Esox lucius": "brochet",
    "Perca fluviatilis": "perche commune",
    "Cyprinus carpio": "carpe commune",
}

# Tags contextuels pour enrichir (lieu, lumière)
CONTEXT_TAGS = [
    "forêt", "prairie", "bord de rivière", "bord de mer", "marais", "montagne",
    "jardin", "parc", "champ cultivé",
    "macro", "gros plan", "en vol", "au sol", "dans l'eau", "sur une fleur", "sur un arbre",
    "printemps", "été", "automne", "hiver",
    "lumière dorée", "lumière naturelle", "contre-jour", "ombre",
]


class BioclipClient:
    """
    Client BioCLIP pour identification d'espèces animales.
    Fallback sur Ollama si le score de confiance est trop bas.
    """

    def __init__(self, config: dict):
        bioclip_cfg = config.get("bioclip", {})
        self.confidence_threshold = bioclip_cfg.get("confidence_threshold", 0.26)
        self.top_k = bioclip_cfg.get("top_k", 3)
        self.use_context_tags = bioclip_cfg.get("use_context_tags", True)
        self.context_top_k = bioclip_cfg.get("context_top_k", 5)
        self.ollama_fallback = bioclip_cfg.get("ollama_fallback", True)
        self.species_dict = bioclip_cfg.get("species", DEFAULT_SPECIES)
        self.tag_suffix = config.get("xmp", {}).get("tag_suffix", "_ai")

        # Ollama fallback
        self._ollama_client = None
        self._config = config

        self._bio_model = None
        self._bio_preprocess = None
        self._bio_tokenizer = None
        self._bio_text_features = None

        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._clip_ctx_features = None

        self._device = None
        self._lock = threading.Lock()
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if not _CLIP_AVAILABLE:
            return False

        with self._lock:
            if self._loaded:
                return True
            try:
                import torch
                if torch.backends.mps.is_available():
                    self._device = "mps"
                elif torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"

                logger.info(f"Chargement BioCLIP sur {self._device}…")
                t0 = time.perf_counter()

                # Modèle BioCLIP (identification espèces)
                self._bio_model, _, self._bio_preprocess = open_clip.create_model_and_transforms(
                    "hf-hub:imageomics/bioclip"
                )
                self._bio_model = self._bio_model.to(self._device).eval()
                self._bio_tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

                # Modèle CLIP standard (tags contextuels)
                if self.use_context_tags:
                    self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                        "ViT-B-16", pretrained="openai"
                    )
                    self._clip_model = self._clip_model.to(self._device).eval()
                    self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

                self._encode_all()

                # Warm-up
                dummy = open_clip.transform.image_transform(
                    224, is_train=False
                )(Image.new("RGB", (224, 224)))
                import torch
                with torch.no_grad():
                    _ = self._bio_model.encode_image(dummy.unsqueeze(0).to(self._device))

                elapsed = time.perf_counter() - t0
                logger.info(
                    f"BioCLIP prêt en {elapsed:.1f}s "
                    f"({len(self.species_dict)} espèces + {len(CONTEXT_TAGS)} tags contextuels)"
                )
                self._loaded = True
                return True

            except Exception as e:
                logger.error(f"Impossible de charger BioCLIP : {e}")
                return False

    def _encode_all(self):
        import torch
        species_names = list(self.species_dict.keys())
        with torch.no_grad():
            # Encodage taxonomique BioCLIP
            sp_tok = self._bio_tokenizer(species_names).to(self._device)
            self._bio_text_features = self._bio_model.encode_text(sp_tok)
            self._bio_text_features /= self._bio_text_features.norm(dim=-1, keepdim=True)

            # Encodage contexte CLIP standard
            if self.use_context_tags and self._clip_model is not None:
                ctx_tok = self._clip_tokenizer(CONTEXT_TAGS).to(self._device)
                self._clip_ctx_features = self._clip_model.encode_text(ctx_tok)
                self._clip_ctx_features /= self._clip_ctx_features.norm(dim=-1, keepdim=True)

    def check_available(self) -> tuple[bool, str]:
        if not _CLIP_AVAILABLE:
            return False, "open_clip non installé (pip install open_clip_torch)"
        ok = self._ensure_loaded()
        if ok:
            return True, f"BioCLIP sur {self._device} ({len(self.species_dict)} espèces)"
        return False, "Impossible de charger BioCLIP"

    def generate_tags(self, image_path: Path) -> list[str]:
        """
        Identifie l'espèce animale et génère les tags.
        Fallback Ollama si la confiance BioCLIP est insuffisante.
        """
        if not self._ensure_loaded():
            # Fallback Ollama direct si BioCLIP non disponible
            return self._ollama_fallback(image_path)

        import torch
        try:
            t0 = time.perf_counter()
            species_names = list(self.species_dict.keys())
            tags = []

            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                bio_img = self._bio_preprocess(img).unsqueeze(0).to(self._device)
                if self.use_context_tags and self._clip_preprocess is not None:
                    clip_img = self._clip_preprocess(img).unsqueeze(0).to(self._device)
                else:
                    clip_img = None

            with torch.no_grad():
                # ── Identification espèce (BioCLIP) ─────────────────────────
                bio_f = self._bio_model.encode_image(bio_img)
                bio_f /= bio_f.norm(dim=-1, keepdim=True)
                bio_sims = (bio_f @ self._bio_text_features.T)[0]

                top_k = min(self.top_k, len(species_names))
                top = bio_sims.topk(top_k)
                best_score = top.values[0].item()

                if best_score >= self.confidence_threshold:
                    # Espèce(s) identifiée(s) avec confiance
                    for score, idx in zip(top.values.cpu().tolist(), top.indices.cpu().tolist()):
                        if score >= self.confidence_threshold:
                            sp_name = species_names[idx]
                            common = self.species_dict[sp_name]
                            tags.append(f"{sp_name} ({common})")
                    logger.debug(
                        f"BioCLIP {image_path.name}: confiance={best_score:.3f} "
                        f"→ {tags[0] if tags else 'aucun'}"
                    )
                else:
                    # Confiance trop faible
                    logger.debug(
                        f"BioCLIP {image_path.name}: confiance={best_score:.3f} < "
                        f"{self.confidence_threshold} — fallback Ollama"
                    )
                    if self.ollama_fallback:
                        ollama_tags = self._ollama_fallback(image_path)
                        if ollama_tags:
                            return ollama_tags
                    # Si pas de fallback ou Ollama échoue : meilleure espèce quand même
                    if species_names:
                        best_idx = top.indices[0].item()
                        sp_name = species_names[best_idx]
                        common = self.species_dict[sp_name]
                        tags.append(f"{sp_name} ({common}) [?]")

                # ── Tags contextuels (CLIP standard) ────────────────────────
                if self.use_context_tags and clip_img is not None and self._clip_model is not None:
                    clip_f = self._clip_model.encode_image(clip_img)
                    clip_f /= clip_f.norm(dim=-1, keepdim=True)
                    ctx_sims = (clip_f @ self._clip_ctx_features.T)[0]
                    ctx_top = ctx_sims.topk(min(self.context_top_k, len(CONTEXT_TAGS)))
                    ctx_tags = [CONTEXT_TAGS[i] for i in ctx_top.indices.cpu().tolist()]
                    tags.extend(ctx_tags)

            # Appliquer le suffixe
            if self.tag_suffix:
                tags = [f"{t}{self.tag_suffix}" for t in tags]

            elapsed = time.perf_counter() - t0
            logger.debug(f"BioCLIP total {image_path.name}: {elapsed*1000:.0f}ms")
            return tags

        except Exception as e:
            logger.error(f"Erreur BioCLIP sur {image_path.name}: {e}")
            return self._ollama_fallback(image_path)

    def _ollama_fallback(self, image_path: Path) -> list[str]:
        """Fallback vers Ollama pour les cas hors-vocabulaire ou sans animal."""
        if not self.ollama_fallback:
            return []
        try:
            if self._ollama_client is None:
                from ollama_client import OllamaClient
                # Prompt spécialisé faune pour le fallback
                cfg = dict(self._config)
                cfg.setdefault("prompt", {})["auto_prompt"] = (
                    "Identifie l'animal principal sur cette photo.\n"
                    "Donne : nom scientifique, nom commun en français, famille, habitat, comportement visible.\n"
                    "Si pas d'animal visible : décris le sujet principal.\n"
                    "Tags séparés par des virgules, rien d'autre."
                )
                self._ollama_client = OllamaClient(cfg)
            return self._ollama_client.generate_tags(image_path)
        except Exception as e:
            logger.error(f"Ollama fallback échoué pour {image_path.name}: {e}")
            return []

    def unload(self):
        """Libère les modèles de la mémoire."""
        if not self._loaded:
            return
        with self._lock:
            import torch
            for attr in ("_bio_model", "_clip_model", "_bio_text_features", "_clip_ctx_features"):
                if getattr(self, attr) is not None:
                    delattr(self, attr)
                    setattr(self, attr, None)
            self._loaded = False
            if self._device == "mps":
                torch.mps.empty_cache()
            logger.info("BioCLIP déchargé")
