"""
Client Ollama pour la génération de tags via modèles vision locaux.
Lit sa configuration depuis config.yaml.
"""

import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Optional
import threading

import requests
from PIL import Image

# Extensions RAW nécessitant rawpy pour le décodage
_RAW_EXTENSIONS = {
    ".dng", ".cr2", ".cr3", ".nef", ".arw",
    ".orf", ".rw2", ".raf", ".pef", ".srw",
    ".x3f", ".3fr", ".mef", ".erf", ".kdc",
}

try:
    import rawpy
    _RAWPY_AVAILABLE = True
except ImportError:
    _RAWPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client pour l'API Ollama avec support vision."""

    def __init__(self, config: dict):
        """
        Initialise le client Ollama.

        Args:
            config: Dictionnaire de configuration complet (chargé depuis config.yaml)
        """
        ollama_cfg = config.get("ollama", {})
        model_cfg = config.get("model", {})
        perf_cfg = config.get("performance", {})
        prompt_cfg = config.get("prompt", {})

        self.base_url = ollama_cfg.get("url", "http://localhost:11434")
        self.timeout = ollama_cfg.get("timeout", 300)
        self.retry_attempts = ollama_cfg.get("retry_attempts", 2)

        self.model_name = model_cfg.get("name", "llava:13b")
        self.temperature = model_cfg.get("temperature", 0.1)
        self.max_tokens = model_cfg.get("max_tokens", 150)

        self.max_image_size = perf_cfg.get("max_image_size", 1024)
        self.jpeg_quality = perf_cfg.get("jpeg_quality", 70)
        self.request_delay = perf_cfg.get("request_delay", 0.0)

        self.language = prompt_cfg.get("language", "français")
        self.max_tags = prompt_cfg.get("max_tags", 15)
        self.auto_prompt_template = prompt_cfg.get("auto_prompt", "")

        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"

    def check_server(self) -> bool:
        """Vérifie que le serveur Ollama est accessible."""
        try:
            response = requests.get(self.tags_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Serveur Ollama inaccessible : {e}")
            return False

    def list_models(self) -> list[str]:
        """Retourne la liste des modèles disponibles sur le serveur."""
        try:
            response = requests.get(self.tags_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except requests.exceptions.RequestException as e:
            logger.error(f"Impossible de lister les modèles : {e}")
        return []

    def encode_image(self, image_path: Path) -> Optional[str]:
        """
        Encode une image en base64 pour l'API Ollama.
        Redimensionne et compresse pour optimiser la vitesse.

        Args:
            image_path: Chemin vers le fichier image

        Returns:
            Chaîne base64 ou None en cas d'erreur
        """
        ext = image_path.suffix.lower()
        try:
            if ext in _RAW_EXTENSIONS:
                img = self._open_raw(image_path)
            else:
                img = self._open_standard(image_path)

            if img is None:
                return None

            # Redimensionnement en conservant le ratio (BILINEAR plus rapide que LANCZOS)
            max_size = self.max_image_size
            w, h = img.size
            if max(w, h) > max_size:
                ratio = max_size / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.BILINEAR)

            # Encodage JPEG en mémoire (optimize=False plus rapide pour la compression)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=False)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded

        except Exception as e:
            logger.error(f"Erreur encodage image {image_path}: {e}")
            return None

    def _open_standard(self, image_path: Path) -> Optional[Image.Image]:
        """Ouvre une image standard via Pillow (copie en mémoire pour libérer le fichier)."""
        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            else:
                img = img.copy()
        return img

    def _open_raw(self, image_path: Path) -> Optional[Image.Image]:
        """
        Ouvre un fichier RAW via rawpy (DNG, RW2, NEF, CR2, ARW…).
        Fallback sur Pillow si rawpy n'est pas installé ou échoue.
        """
        if _RAWPY_AVAILABLE:
            try:
                with rawpy.imread(str(image_path)) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        half_size=True,          # 2x plus rapide, suffisant pour le taguage
                        no_auto_bright=False,
                        output_bps=8,
                    )
                return Image.fromarray(rgb)
            except Exception as e:
                logger.warning(
                    f"rawpy a échoué sur {image_path.name} ({e}), tentative via Pillow…"
                )

        # Fallback Pillow (fonctionne sur certains DNG simples)
        try:
            return self._open_standard(image_path)
        except Exception as e:
            logger.error(f"Impossible d'ouvrir le RAW {image_path.name}: {e}")
            return None

    # Compteurs de performance cumulés (thread-safe)
    _perf_lock = threading.Lock()
    _perf_total_encode = 0.0
    _perf_total_api = 0.0
    _perf_count = 0

    def generate_tags(self, image_path: Path) -> list[str]:
        """
        Génère des tags pour une image via le modèle vision.

        Args:
            image_path: Chemin vers le fichier image

        Returns:
            Liste de tags générés (peut être vide en cas d'erreur)
        """
        t0 = time.perf_counter()
        image_b64 = self.encode_image(image_path)
        t_encode = time.perf_counter() - t0

        if not image_b64:
            return []

        prompt = self.auto_prompt_template.format(
            language=self.language,
            max_tags=self.max_tags,
        )

        for attempt in range(1, self.retry_attempts + 1):
            try:
                # Délai entre requêtes si configuré
                if self.request_delay > 0 and attempt == 1:
                    time.sleep(self.request_delay)

                timeout = self.timeout * attempt  # Augmente le timeout à chaque tentative

                t1 = time.perf_counter()
                raw_response = self._call_api(prompt, image_b64, timeout)
                t_api = time.perf_counter() - t1

                if raw_response:
                    tags = self._parse_tags(raw_response)

                    # Accumuler les stats de perf
                    with OllamaClient._perf_lock:
                        OllamaClient._perf_total_encode += t_encode
                        OllamaClient._perf_total_api += t_api
                        OllamaClient._perf_count += 1
                        count = OllamaClient._perf_count
                        avg_enc = OllamaClient._perf_total_encode / count
                        avg_api = OllamaClient._perf_total_api / count

                    logger.debug(
                        f"{image_path.name}: encode={t_encode:.2f}s api={t_api:.2f}s "
                        f"total={t_encode+t_api:.2f}s | "
                        f"moy encode={avg_enc:.2f}s moy api={avg_api:.2f}s (n={count})"
                    )
                    return tags

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Timeout tentative {attempt}/{self.retry_attempts} pour {image_path.name}"
                )
            except requests.exceptions.RequestException as e:
                logger.error(f"Erreur réseau tentative {attempt}: {e}")
            except Exception as e:
                logger.error(f"Erreur inattendue tentative {attempt}: {e}")

        logger.error(f"Échec de la génération de tags pour {image_path.name}")
        return []

    def _call_api(self, prompt: str, image_b64: str, timeout: int) -> Optional[str]:
        """
        Appel direct à l'API Ollama.

        Args:
            prompt: Texte du prompt
            image_b64: Image encodée en base64
            timeout: Timeout en secondes

        Returns:
            Réponse textuelle du modèle ou None
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        response = requests.post(self.generate_url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def _parse_tags(self, raw_response: str) -> list[str]:
        """
        Nettoie et parse la réponse du modèle en liste de tags.

        Args:
            raw_response: Réponse brute du modèle

        Returns:
            Liste de tags nettoyés et dédupliqués
        """
        if not raw_response:
            return []

        # Supprimer les lignes vides et sauts de ligne
        text = raw_response.replace("\n", ", ").replace(";", ",")

        # Séparation par virgule
        raw_tags = [t.strip() for t in text.split(",")]

        cleaned = []
        seen = set()

        for tag in raw_tags:
            if not tag:
                continue

            # Suppression des numérotations (1. 2. etc.)
            import re
            tag = re.sub(r"^\d+[\.\)]\s*", "", tag)

            # Suppression des tirets en début
            tag = tag.lstrip("- •*")

            # Suppression des guillemets
            tag = tag.strip("\"'`")

            # Nettoyage des caractères spéciaux sauf tirets internes
            tag = re.sub(r"[^\w\s\-àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿœÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝÞŸŒ]", "", tag)
            tag = tag.strip()

            # Filtrage
            if not tag or len(tag) < 2:
                continue

            # Déduplication insensible à la casse
            tag_lower = tag.lower()
            if tag_lower in seen:
                continue

            seen.add(tag_lower)
            cleaned.append(tag)

            if len(cleaned) >= self.max_tags:
                break

        return cleaned
