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

import requests
from PIL import Image

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
        try:
            with Image.open(image_path) as img:
                # Conversion en RGB si nécessaire (ex: RGBA, palette)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                # Redimensionnement en conservant le ratio
                max_size = self.max_image_size
                if max(img.width, img.height) > max_size:
                    img.thumbnail((max_size, max_size), Image.LANCZOS)

                # Encodage JPEG en mémoire
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8")

        except Exception as e:
            logger.error(f"Erreur encodage image {image_path}: {e}")
            return None

    def generate_tags(self, image_path: Path) -> list[str]:
        """
        Génère des tags pour une image via le modèle vision.

        Args:
            image_path: Chemin vers le fichier image

        Returns:
            Liste de tags générés (peut être vide en cas d'erreur)
        """
        image_b64 = self.encode_image(image_path)
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
                raw_response = self._call_api(prompt, image_b64, timeout)

                if raw_response:
                    tags = self._parse_tags(raw_response)
                    logger.debug(f"Tags générés pour {image_path.name}: {tags}")
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
