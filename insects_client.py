"""
Client BioCLIP 2 pour le mode Insectes.
Utilise imageomics/bioclip-2 (ViT-L/14, TreeOfLife-200M + BIOSCAN-5M).
BIOSCAN-5M (5 millions d'images d'insectes) est inclus dans le dataset de BioCLIP 2,
ce qui lui confère une excellente couverture entomologique.

Architecture :
  1. BioCLIP 2 identifie l'espèce parmi le vocabulaire taxonomique (noms scientifiques)
  2. Si confiance suffisante : retourne "Nom scientifique (Nom français)"
  3. Si confiance trop basse : fallback Ollama avec prompt spécialisé entomologie
  4. Tags contextuels optionnels via CLIP standard (fleur, habitat, posture)
"""

import logging
import threading
import time
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

try:
    import torch
    import open_clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False


# ── Espèces d'insectes européens ──────────────────────────────────────────────
# Format : "Nom scientifique": "Nom commun français"
# ~130 espèces couvrant papillons, abeilles, libellules, coléoptères, etc.
DEFAULT_INSECT_SPECIES = {
    # ── Lépidoptères — Papillons de jour (Rhopalocères) ───────────────────────
    "Papilio machaon": "machaon",
    "Iphiclides podalirius": "flambé",
    "Pieris brassicae": "piéride du chou",
    "Pieris rapae": "piéride de la rave",
    "Pieris napi": "piéride du navet",
    "Gonepteryx rhamni": "citron",
    "Colias croceus": "souci",
    "Colias hyale": "soufré",
    "Anthocharis cardamines": "aurore",
    "Vanessa atalanta": "vulcain",
    "Vanessa cardui": "belle-dame",
    "Aglais io": "paon du jour",
    "Aglais urticae": "petite tortue",
    "Nymphalis polychloros": "grande tortue",
    "Nymphalis antiopa": "morio",
    "Polygonia c-album": "robert-le-diable",
    "Araschnia levana": "carte géographique",
    "Limenitis camilla": "petit sylvain",
    "Limenitis populi": "grand sylvain",
    "Apatura iris": "grand mars changeant",
    "Apatura ilia": "petit mars changeant",
    "Argynnis paphia": "tabac d'Espagne",
    "Argynnis aglaja": "grand nacré",
    "Boloria selene": "petit nacré",
    "Melitaea cinxia": "mélitée du plantain",
    "Melitaea athalia": "mélitée du mélampyre",
    "Euphydryas aurinia": "damier de la succise",
    "Erebia ligea": "moiré blanc-fascié",
    "Melanargia galathea": "demi-deuil",
    "Maniola jurtina": "myrtil",
    "Pyronia tithonus": "amaryllis",
    "Aphantopus hyperantus": "tristan",
    "Coenonympha pamphilus": "fadet commun",
    "Coenonympha arcania": "céphale",
    "Pararge aegeria": "tircis",
    "Lasiommata megera": "mégère",
    "Callophrys rubi": "thécla de la ronce",
    "Lycaena phlaeas": "cuivré commun",
    "Lycaena dispar": "cuivré des marais",
    "Polyommatus icarus": "azuré commun",
    "Polyommatus bellargus": "azuré bleu-céleste",
    "Plebejus argus": "azuré de l'ajonc",
    "Celastrina argiolus": "azuré des nerpruns",
    "Cupido minimus": "argus frêle",
    "Zygaena filipendulae": "zygène du trèfle",
    # ── Lépidoptères — Papillons de nuit (Hétérocères) ────────────────────────
    "Sphinx ligustri": "sphinx du troène",
    "Deilephila elpenor": "sphinx de la vigne",
    "Macroglossum stellatarum": "sphinx gazé",
    "Saturnia pavonia": "petit paon de nuit",
    "Saturnia pyri": "grand paon de nuit",
    "Arctia caja": "écaille martre",
    "Callimorpha dominula": "écaille martre des marais",
    "Tyria jacobaeae": "goutte de sang",
    # ── Hyménoptères — Abeilles et bourdons ───────────────────────────────────
    "Apis mellifera": "abeille domestique",
    "Bombus terrestris": "bourdon terrestre",
    "Bombus lapidarius": "bourdon des pierres",
    "Bombus pascuorum": "bourdon des champs",
    "Bombus lucorum": "bourdon des clairieres",
    "Bombus hortorum": "bourdon des jardins",
    "Bombus hypnorum": "bourdon des arbres",
    "Xylocopa violacea": "abeille charpentière violette",
    "Anthophora plumipes": "anthophore à pieds velus",
    "Osmia rufa": "osmie rousse",
    "Osmia bicornis": "osmie à deux cornes",
    "Halictus scabiosae": "halicte",
    "Andrena fulva": "andrène rousse",
    # ── Hyménoptères — Guêpes et frelons ──────────────────────────────────────
    "Vespa crabro": "frelon européen",
    "Vespa velutina": "frelon asiatique",
    "Vespula vulgaris": "guêpe commune",
    "Vespula germanica": "guêpe germanique",
    "Polistes dominula": "poliste dominule",
    # ── Hyménoptères — Fourmis ────────────────────────────────────────────────
    "Formica rufa": "fourmi rousse des bois",
    "Formica fusca": "fourmi noire",
    "Lasius niger": "fourmi noire de jardin",
    "Camponotus herculeanus": "grande fourmi charpentière",
    "Myrmica rubra": "fourmi rouge",
    # ── Odonates — Libellules et demoiselles ──────────────────────────────────
    "Libellula depressa": "libellule déprimée",
    "Libellula quadrimaculata": "libellule à quatre taches",
    "Orthetrum cancellatum": "orthétrum réticulé",
    "Sympetrum striolatum": "sympétrum fascié",
    "Sympetrum sanguineum": "sympétrum sanguin",
    "Anax imperator": "anax empereur",
    "Aeshna cyanea": "aeschne bleue",
    "Aeshna grandis": "aeschne grande",
    "Calopteryx splendens": "caloptéryx splendide",
    "Calopteryx virgo": "caloptéryx vierge",
    "Coenagrion puella": "agrion jouvencelle",
    "Ischnura elegans": "agrion élégant",
    "Pyrrhosoma nymphula": "petite nymphe au corps de feu",
    "Platycnemis pennipes": "agrion à larges pattes",
    "Lestes sponsa": "leste fiancé",
    # ── Coléoptères ───────────────────────────────────────────────────────────
    "Coccinella septempunctata": "coccinelle à 7 points",
    "Harmonia axyridis": "coccinelle asiatique",
    "Adalia bipunctata": "coccinelle à 2 points",
    "Lampyris noctiluca": "ver luisant",
    "Cantharis rustica": "cantharide fauve",
    "Lucanus cervus": "lucane cerf-volant",
    "Cerambyx cerdo": "grand capricorne",
    "Rosalia alpina": "rosalie des Alpes",
    "Aromia moschata": "capricorne musqué",
    "Chrysochroa fulminans": "bupreste",
    "Oxythyrea funesta": "cétoine funeste",
    "Cetonia aurata": "cétoine dorée",
    "Protaetia cuprea": "cétoine cuivrée",
    "Leptinotarsa decemlineata": "doryphore",
    "Geotrupes stercorarius": "géotrupe stercoraire",
    "Carabus auratus": "carabe doré",
    "Carabus violaceus": "carabe violet",
    "Dytiscus marginalis": "dytique bordé",
    "Gyrinus natator": "gyrinus nageur",
    # ── Hémiptères et Orthoptères ─────────────────────────────────────────────
    "Mantis religiosa": "mante religieuse",
    "Gryllus campestris": "grillon des champs",
    "Acheta domesticus": "grillon domestique",
    "Tettigonia viridissima": "grande sauterelle verte",
    "Decticus verrucivorus": "dectique verrucivore",
    "Chorthippus parallelus": "criquet des pâtures",
    "Locusta migratoria": "criquet migrateur",
    "Forficula auricularia": "forficule auriculaire",
    # ── Diptères ─────────────────────────────────────────────────────────────
    "Eristalis tenax": "éristale des abeilles",
    "Volucella zonaria": "volucelle zonée",
    "Volucella bombylans": "volucelle bourdon",
    "Syrphus ribesii": "syrphe du groseillier",
    "Tipula oleracea": "tipule des prairies",
    # ── Névroptères / Plécoptères ─────────────────────────────────────────────
    "Chrysopa perla": "chrysope perlée",
    "Chrysoperla carnea": "chrysope dorée",
}

# Tags contextuels pour enrichir
INSECT_CONTEXT_TAGS = [
    # Habitat / support
    "sur une fleur", "sur une feuille", "sur une tige", "sur une branche",
    "au sol", "sur l'eau", "sur l'écorce", "en vol",
    "prairie fleurie", "jardin", "forêt", "bord de rivière", "marais",
    "champ", "haie", "mare", "montagne",
    # Plante hôte / nectar
    "sur lavande", "sur chardon", "sur pissenlit", "sur origan",
    "sur trèfle", "sur buddleia", "sur ombellifère",
    # Comportement / technique
    "macro", "gros plan", "pollinisation", "ponte", "accouplement",
    "larve", "chrysalide", "nymphe", "imago", "adulte",
    # Lumière
    "lumière dorée", "lumière naturelle", "contre-jour", "fond vert", "fond flou",
]


class InsectsClient:
    """
    Client BioCLIP 2 spécialisé identification d'insectes.
    Utilise imageomics/bioclip-2 entraîné sur BIOSCAN-5M (5M images d'insectes).
    """

    def __init__(self, config: dict):
        insects_cfg = config.get("insects", {})
        self.confidence_threshold = insects_cfg.get("confidence_threshold", 0.22)
        self.top_k = insects_cfg.get("top_k", 3)
        self.use_context_tags = insects_cfg.get("use_context_tags", True)
        self.context_top_k = insects_cfg.get("context_top_k", 4)
        self.ollama_fallback = insects_cfg.get("ollama_fallback", True)
        self.species_dict = insects_cfg.get("species", DEFAULT_INSECT_SPECIES)

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

                logger.info(f"Chargement BioCLIP 2 (Insectes) sur {self._device}…")
                t0 = time.perf_counter()

                # BioCLIP 2 — ViT-L/14, inclut BIOSCAN-5M (5M images insectes)
                self._bio_model, _, self._bio_preprocess = open_clip.create_model_and_transforms(
                    "hf-hub:imageomics/bioclip-2"
                )
                self._bio_model = self._bio_model.to(self._device).eval()
                self._bio_tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")

                # CLIP standard pour les tags contextuels
                if self.use_context_tags:
                    self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                        "ViT-B-16", pretrained="openai"
                    )
                    self._clip_model = self._clip_model.to(self._device).eval()
                    self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

                self._encode_all()

                # Warm-up MPS
                import torch as _torch
                dummy = self._bio_preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(self._device)
                with _torch.no_grad():
                    _ = self._bio_model.encode_image(dummy)

                elapsed = time.perf_counter() - t0
                logger.info(
                    f"BioCLIP 2 Insectes prêt en {elapsed:.1f}s "
                    f"({len(self.species_dict)} espèces + {len(INSECT_CONTEXT_TAGS)} tags contextuels)"
                )
                self._loaded = True
                return True

            except Exception as e:
                logger.error(f"Impossible de charger BioCLIP 2 (Insectes) : {e}")
                return False

    def _encode_all(self):
        import torch
        species_names = list(self.species_dict.keys())
        with torch.no_grad():
            sp_tok = self._bio_tokenizer(species_names).to(self._device)
            self._bio_text_features = self._bio_model.encode_text(sp_tok)
            self._bio_text_features /= self._bio_text_features.norm(dim=-1, keepdim=True)

            if self.use_context_tags and self._clip_model is not None:
                ctx_tok = self._clip_tokenizer(INSECT_CONTEXT_TAGS).to(self._device)
                self._clip_ctx_features = self._clip_model.encode_text(ctx_tok)
                self._clip_ctx_features /= self._clip_ctx_features.norm(dim=-1, keepdim=True)

    def check_available(self) -> tuple[bool, str]:
        if not _CLIP_AVAILABLE:
            return False, "open_clip non installé (pip install open_clip_torch)"
        ok = self._ensure_loaded()
        if ok:
            return True, f"BioCLIP 2 Insectes sur {self._device} ({len(self.species_dict)} espèces)"
        return False, "Impossible de charger BioCLIP 2 (Insectes)"

    def generate_tags(self, image_path: Path) -> list[str]:
        """Identifie l'espèce d'insecte et génère les tags."""
        if not self._ensure_loaded():
            return self._do_ollama_fallback(image_path)

        import torch
        try:
            t0 = time.perf_counter()
            species_names = list(self.species_dict.keys())
            tags = []

            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                bio_img = self._bio_preprocess(img).unsqueeze(0).to(self._device)
                clip_img = (
                    self._clip_preprocess(img).unsqueeze(0).to(self._device)
                    if self.use_context_tags and self._clip_preprocess is not None
                    else None
                )

            with torch.no_grad():
                # ── Identification espèce (BioCLIP 2) ────────────────────────
                bio_f = self._bio_model.encode_image(bio_img)
                bio_f /= bio_f.norm(dim=-1, keepdim=True)
                bio_sims = (bio_f @ self._bio_text_features.T)[0]

                top_k = min(self.top_k, len(species_names))
                top = bio_sims.topk(top_k)
                best_score = top.values[0].item()

                if best_score >= self.confidence_threshold:
                    for score, idx in zip(top.values.cpu().tolist(), top.indices.cpu().tolist()):
                        if score >= self.confidence_threshold:
                            sp_name = species_names[idx]
                            common = self.species_dict[sp_name]
                            tags.append(f"{sp_name} ({common})")
                    logger.debug(
                        f"BioCLIP2 Insectes {image_path.name}: {best_score:.3f} → {tags[0] if tags else '–'}"
                    )
                else:
                    logger.debug(
                        f"BioCLIP2 Insectes {image_path.name}: {best_score:.3f} < "
                        f"{self.confidence_threshold} — fallback Ollama"
                    )
                    if self.ollama_fallback:
                        ollama_tags = self._do_ollama_fallback(image_path)
                        if ollama_tags:
                            return ollama_tags
                    if species_names:
                        best_idx = top.indices[0].item()
                        sp_name = species_names[best_idx]
                        common = self.species_dict[sp_name]
                        tags.append(f"{sp_name} ({common}) [?]")

                # ── Tags contextuels (CLIP standard) ─────────────────────────
                if self.use_context_tags and clip_img is not None and self._clip_model is not None:
                    clip_f = self._clip_model.encode_image(clip_img)
                    clip_f /= clip_f.norm(dim=-1, keepdim=True)
                    ctx_sims = (clip_f @ self._clip_ctx_features.T)[0]
                    ctx_top = ctx_sims.topk(min(self.context_top_k, len(INSECT_CONTEXT_TAGS)))
                    ctx_tags = [INSECT_CONTEXT_TAGS[i] for i in ctx_top.indices.cpu().tolist()]
                    tags.extend(ctx_tags)

            elapsed = time.perf_counter() - t0
            logger.debug(f"BioCLIP2 Insectes {image_path.name}: {elapsed*1000:.0f}ms")
            return tags

        except Exception as e:
            logger.error(f"Erreur BioCLIP2 Insectes sur {image_path.name}: {e}")
            return self._do_ollama_fallback(image_path)

    def _do_ollama_fallback(self, image_path: Path) -> list[str]:
        """Fallback Ollama avec prompt spécialisé entomologie."""
        if not self.ollama_fallback:
            return []
        try:
            if self._ollama_client is None:
                from ollama_client import OllamaClient
                cfg = dict(self._config)
                cfg.setdefault("prompt", {})["auto_prompt"] = (
                    "Identifie l'insecte sur cette photo.\n"
                    "Donne : nom scientifique, nom commun en français, ordre (Lépidoptère, Hyménoptère…), "
                    "famille, plante hôte ou fleur visitée si visible, comportement (pollinisation, ponte, vol…).\n"
                    "Si aucun insecte n'est visible : décris le sujet principal.\n"
                    "Tags séparés par des virgules, rien d'autre."
                )
                self._ollama_client = OllamaClient(cfg)
            return self._ollama_client.generate_tags(image_path)
        except Exception as e:
            logger.error(f"Ollama fallback Insectes échoué pour {image_path.name}: {e}")
            return []

    def unload(self):
        """Libère les modèles de la mémoire."""
        if not self._loaded:
            return
        with self._lock:
            import torch
            for attr in ("_bio_model", "_clip_model", "_bio_text_features", "_clip_ctx_features"):
                if getattr(self, attr) is not None:
                    setattr(self, attr, None)
            self._loaded = False
            if self._device == "mps":
                torch.mps.empty_cache()
            logger.info("BioCLIP 2 Insectes déchargé")
