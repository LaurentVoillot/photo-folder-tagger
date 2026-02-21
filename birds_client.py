"""
Client BioCLIP 2 pour le mode Oiseaux.
Utilise imageomics/bioclip-2 (ViT-L/14, TreeOfLife-200M + BIOSCAN-5M).
BioCLIP 2 surpasse BioCLIP v1 de +18% en zero-shot sur les taxa biologiques.

Architecture :
  1. BioCLIP 2 identifie l'espèce parmi le vocabulaire taxonomique (noms scientifiques)
  2. Si confiance suffisante : retourne "Nom scientifique (Nom français)"
  3. Si confiance trop basse : fallback Ollama avec prompt spécialisé ornithologie
  4. Tags contextuels optionnels via CLIP standard (habitat, comportement)
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


# ── Espèces d'oiseaux européens ───────────────────────────────────────────────
# Format : "Nom scientifique": "Nom commun français"
# ~120 espèces couvrant les oiseaux les plus photographiés en Europe
DEFAULT_BIRD_SPECIES = {
    # ── Passereaux ─────────────────────────────────────────────────────────────
    "Erithacus rubecula": "rouge-gorge familier",
    "Turdus merula": "merle noir",
    "Turdus philomelos": "grive musicienne",
    "Turdus viscivorus": "grive draine",
    "Turdus iliacus": "grive mauvis",
    "Turdus pilaris": "grive litorne",
    "Parus major": "mésange charbonnière",
    "Cyanistes caeruleus": "mésange bleue",
    "Lophophanes cristatus": "mésange huppée",
    "Periparus ater": "mésange noire",
    "Poecile palustris": "mésange nonnette",
    "Poecile montanus": "mésange boréale",
    "Aegithalos caudatus": "mésange à longue queue",
    "Fringilla coelebs": "pinson des arbres",
    "Fringilla montifringilla": "pinson du Nord",
    "Carduelis carduelis": "chardonneret élégant",
    "Chloris chloris": "verdier d'Europe",
    "Spinus spinus": "tarin des aulnes",
    "Linaria cannabina": "linotte mélodieuse",
    "Pyrrhula pyrrhula": "bouvreuil pivoine",
    "Coccothraustes coccothraustes": "grosbec casse-noyaux",
    "Loxia curvirostra": "bec-croisé des sapins",
    "Passer domesticus": "moineau domestique",
    "Passer montanus": "moineau friquet",
    "Sturnus vulgaris": "étourneau sansonnet",
    "Sitta europaea": "sittelle torchepot",
    "Certhia familiaris": "grimpereau des bois",
    "Certhia brachydactyla": "grimpereau des jardins",
    "Troglodytes troglodytes": "troglodyte mignon",
    "Regulus regulus": "roitelet huppé",
    "Regulus ignicapilla": "roitelet à triple bandeau",
    "Phylloscopus collybita": "pouillot véloce",
    "Phylloscopus trochilus": "pouillot fitis",
    "Sylvia atricapilla": "fauvette à tête noire",
    "Sylvia communis": "fauvette grisette",
    "Sylvia borin": "fauvette des jardins",
    "Acrocephalus scirpaceus": "rousserolle effarvatte",
    "Acrocephalus arundinaceus": "rousserolle turdoïde",
    "Hippolais icterina": "hypolaïs ictérine",
    "Ficedula hypoleuca": "gobemouche noir",
    "Muscicapa striata": "gobemouche gris",
    "Phoenicurus phoenicurus": "rougequeue à front blanc",
    "Phoenicurus ochruros": "rougequeue noir",
    "Saxicola rubicola": "traquet pâtre",
    "Oenanthe oenanthe": "traquet motteux",
    "Luscinia megarhynchos": "rossignol philomèle",
    "Luscinia svecica": "gorgebleue à miroir",
    "Motacilla alba": "bergeronnette grise",
    "Motacilla cinerea": "bergeronnette des ruisseaux",
    "Motacilla flava": "bergeronnette printanière",
    "Anthus pratensis": "pipit farlouse",
    "Anthus trivialis": "pipit des arbres",
    "Anthus spinoletta": "pipit spioncelle",
    "Alauda arvensis": "alouette des champs",
    "Galerida cristata": "cochevis huppé",
    "Eremophila alpestris": "alouette hausse-col",
    "Emberiza citrinella": "bruant jaune",
    "Emberiza calandra": "bruant proyer",
    "Emberiza schoeniclus": "bruant des roseaux",
    "Miliaria calandra": "bruant proyer",
    "Corvus corax": "grand corbeau",
    "Corvus corone": "corneille noire",
    "Corvus monedula": "choucas des tours",
    "Corvus frugilegus": "corbeau freux",
    "Pica pica": "pie bavarde",
    "Garrulus glandarius": "geai des chênes",
    "Nucifraga caryocatactes": "cassenoix moucheté",
    "Pyrrhocorax pyrrhocorax": "crave à bec rouge",
    "Oriolus oriolus": "loriot d'Europe",
    "Hirundo rustica": "hirondelle rustique",
    "Delichon urbicum": "hirondelle de fenêtre",
    "Riparia riparia": "hirondelle de rivage",
    "Apus apus": "martinet noir",
    # ── Rapaces diurnes ───────────────────────────────────────────────────────
    "Falco tinnunculus": "faucon crécerelle",
    "Falco peregrinus": "faucon pèlerin",
    "Falco subbuteo": "faucon hobereau",
    "Falco columbarius": "faucon émerillon",
    "Accipiter nisus": "épervier d'Europe",
    "Accipiter gentilis": "autour des palombes",
    "Buteo buteo": "buse variable",
    "Buteo lagopus": "buse pattue",
    "Circus aeruginosus": "busard des roseaux",
    "Circus cyaneus": "busard Saint-Martin",
    "Circus pygargus": "busard cendré",
    "Milvus milvus": "milan royal",
    "Milvus migrans": "milan noir",
    "Pernis apivorus": "bondrée apivore",
    "Aquila chrysaetos": "aigle royal",
    "Haliaeetus albicilla": "pygargue à queue blanche",
    "Gyps fulvus": "vautour fauve",
    "Neophron percnopterus": "vautour percnoptère",
    "Pandion haliaetus": "balbuzard pêcheur",
    # ── Rapaces nocturnes ─────────────────────────────────────────────────────
    "Tyto alba": "effraie des clochers",
    "Strix aluco": "chouette hulotte",
    "Strix uralensis": "chouette de l'Oural",
    "Asio otus": "hibou moyen-duc",
    "Asio flammeus": "hibou des marais",
    "Bubo bubo": "grand-duc d'Europe",
    "Athene noctua": "chevêche d'Athéna",
    "Glaucidium passerinum": "chevêchette d'Europe",
    "Aegolius funereus": "chouette de Tengmalm",
    # ── Échassiers et limicoles ───────────────────────────────────────────────
    "Ardea cinerea": "héron cendré",
    "Ardea alba": "grande aigrette",
    "Egretta garzetta": "aigrette garzette",
    "Nycticorax nycticorax": "héron bihoreau",
    "Ciconia ciconia": "cigogne blanche",
    "Ciconia nigra": "cigogne noire",
    "Recurvirostra avosetta": "avocette élégante",
    "Haematopus ostralegus": "huîtrier pie",
    "Vanellus vanellus": "vanneau huppé",
    "Pluvialis apricaria": "pluvier doré",
    "Charadrius hiaticula": "grand gravelot",
    "Tringa totanus": "chevalier gambette",
    "Actitis hypoleucos": "chevalier guignette",
    "Gallinago gallinago": "bécassine des marais",
    "Scolopax rusticola": "bécasse des bois",
    "Limosa limosa": "barge à queue noire",
    "Numenius arquata": "courlis cendré",
    # ── Anatidés (canards, oies, cygnes) ─────────────────────────────────────
    "Anas platyrhynchos": "canard colvert",
    "Anas crecca": "sarcelle d'hiver",
    "Anas penelope": "canard siffleur",
    "Anas clypeata": "canard souchet",
    "Aythya fuligula": "fuligule morillon",
    "Aythya ferina": "fuligule milouin",
    "Mergus merganser": "grand harle",
    "Cygnus olor": "cygne tuberculé",
    "Cygnus cygnus": "cygne chanteur",
    "Anser anser": "oie cendrée",
    "Branta canadensis": "bernache du Canada",
    "Branta leucopsis": "bernache nonnette",
    "Tadorna tadorna": "tadorne de Belon",
    # ── Laridés (mouettes, goélands) ─────────────────────────────────────────
    "Larus argentatus": "goéland argenté",
    "Larus michahellis": "goéland leucophée",
    "Larus canus": "goéland cendré",
    "Larus ridibundus": "mouette rieuse",
    "Chroicocephalus ridibundus": "mouette rieuse",
    "Sterna hirundo": "sterne pierregarin",
    # ── Autres ───────────────────────────────────────────────────────────────
    "Columba palumbus": "pigeon ramier",
    "Columba livia": "pigeon biset",
    "Columba oenas": "pigeon colombin",
    "Streptopelia decaocto": "tourterelle turque",
    "Streptopelia turtur": "tourterelle des bois",
    "Alcedo atthis": "martin-pêcheur d'Europe",
    "Upupa epops": "huppe fasciée",
    "Merops apiaster": "guêpier d'Europe",
    "Coracias garrulus": "rollier d'Europe",
    "Picus viridis": "pic vert",
    "Dendrocopos major": "pic épeiche",
    "Dendrocopos minor": "pic épeichette",
    "Dryocopus martius": "pic noir",
    "Jynx torquilla": "torcol fourmilier",
    "Cuculus canorus": "coucou gris",
    "Rallus aquaticus": "râle d'eau",
    "Gallinula chloropus": "poule d'eau",
    "Fulica atra": "foulque macroule",
    "Grus grus": "grue cendrée",
    "Tetrao urogallus": "grand tétras",
    "Lagopus muta": "lagopède alpin",
    "Perdix perdix": "perdrix grise",
    "Coturnix coturnix": "caille des blés",
    "Phasianus colchicus": "faisan de Colchide",
}

# Tags contextuels comportement / habitat pour enrichir
BIRD_CONTEXT_TAGS = [
    # Habitat
    "forêt", "lisière de forêt", "prairie", "marais", "bord de rivière",
    "bord de lac", "falaise", "montagne", "littoral", "plage", "mer",
    "jardin", "parc", "champ", "haie", "roselière", "lande",
    # Comportement / posture
    "en vol", "planant", "posé", "perché", "au sol", "nageant",
    "en train de chanter", "nourrissage", "parade nuptiale",
    "en migration", "en groupe", "nichoir",
    # Lumière / technique photo
    "lumière dorée", "contre-jour", "macro", "gros plan",
    "plumage nuptial", "juvénile", "mâle", "femelle",
]


class BirdsClient:
    """
    Client BioCLIP 2 spécialisé identification d'oiseaux.
    Utilise imageomics/bioclip-2 (ViT-L/14, NeurIPS 2025 Spotlight).
    +18% vs BioCLIP v1 en zero-shot sur les taxa biologiques.
    """

    def __init__(self, config: dict):
        birds_cfg = config.get("birds", {})
        self.confidence_threshold = birds_cfg.get("confidence_threshold", 0.24)
        self.top_k = birds_cfg.get("top_k", 3)
        self.use_context_tags = birds_cfg.get("use_context_tags", True)
        self.context_top_k = birds_cfg.get("context_top_k", 4)
        self.ollama_fallback = birds_cfg.get("ollama_fallback", True)
        self.species_dict = birds_cfg.get("species", DEFAULT_BIRD_SPECIES)

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

                logger.info(f"Chargement BioCLIP 2 (Oiseaux) sur {self._device}…")
                t0 = time.perf_counter()

                # BioCLIP 2 — ViT-L/14, TreeOfLife-200M + BIOSCAN-5M
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
                    f"BioCLIP 2 Oiseaux prêt en {elapsed:.1f}s "
                    f"({len(self.species_dict)} espèces + {len(BIRD_CONTEXT_TAGS)} tags contextuels)"
                )
                self._loaded = True
                return True

            except Exception as e:
                logger.error(f"Impossible de charger BioCLIP 2 (Oiseaux) : {e}")
                return False

    def _encode_all(self):
        import torch
        species_names = list(self.species_dict.keys())
        with torch.no_grad():
            sp_tok = self._bio_tokenizer(species_names).to(self._device)
            self._bio_text_features = self._bio_model.encode_text(sp_tok)
            self._bio_text_features /= self._bio_text_features.norm(dim=-1, keepdim=True)

            if self.use_context_tags and self._clip_model is not None:
                ctx_tok = self._clip_tokenizer(BIRD_CONTEXT_TAGS).to(self._device)
                self._clip_ctx_features = self._clip_model.encode_text(ctx_tok)
                self._clip_ctx_features /= self._clip_ctx_features.norm(dim=-1, keepdim=True)

    def check_available(self) -> tuple[bool, str]:
        if not _CLIP_AVAILABLE:
            return False, "open_clip non installé (pip install open_clip_torch)"
        ok = self._ensure_loaded()
        if ok:
            return True, f"BioCLIP 2 Oiseaux sur {self._device} ({len(self.species_dict)} espèces)"
        return False, "Impossible de charger BioCLIP 2 (Oiseaux)"

    def generate_tags(self, image_path: Path) -> list[str]:
        """Identifie l'espèce d'oiseau et génère les tags."""
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
                        f"BioCLIP2 Oiseaux {image_path.name}: {best_score:.3f} → {tags[0] if tags else '–'}"
                    )
                else:
                    logger.debug(
                        f"BioCLIP2 Oiseaux {image_path.name}: {best_score:.3f} < "
                        f"{self.confidence_threshold} — fallback Ollama"
                    )
                    if self.ollama_fallback:
                        ollama_tags = self._do_ollama_fallback(image_path)
                        if ollama_tags:
                            return ollama_tags
                    # Retourne la meilleure espèce avec indicateur d'incertitude
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
                    ctx_top = ctx_sims.topk(min(self.context_top_k, len(BIRD_CONTEXT_TAGS)))
                    ctx_tags = [BIRD_CONTEXT_TAGS[i] for i in ctx_top.indices.cpu().tolist()]
                    tags.extend(ctx_tags)

            elapsed = time.perf_counter() - t0
            logger.debug(f"BioCLIP2 Oiseaux {image_path.name}: {elapsed*1000:.0f}ms")
            return tags

        except Exception as e:
            logger.error(f"Erreur BioCLIP2 Oiseaux sur {image_path.name}: {e}")
            return self._do_ollama_fallback(image_path)

    def _do_ollama_fallback(self, image_path: Path) -> list[str]:
        """Fallback Ollama avec prompt spécialisé ornithologie."""
        if not self.ollama_fallback:
            return []
        try:
            if self._ollama_client is None:
                from ollama_client import OllamaClient
                cfg = dict(self._config)
                cfg.setdefault("prompt", {})["auto_prompt"] = (
                    "Identifie l'oiseau sur cette photo.\n"
                    "Donne : nom scientifique, nom commun en français, famille, ordre, habitat, "
                    "comportement visible (en vol, posé, chantant…), plumage (nuptial, hivernage, juvénile).\n"
                    "Si aucun oiseau n'est visible : décris le sujet principal.\n"
                    "Tags séparés par des virgules, rien d'autre."
                )
                self._ollama_client = OllamaClient(cfg)
            return self._ollama_client.generate_tags(image_path)
        except Exception as e:
            logger.error(f"Ollama fallback Oiseaux échoué pour {image_path.name}: {e}")
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
            logger.info("BioCLIP 2 Oiseaux déchargé")
