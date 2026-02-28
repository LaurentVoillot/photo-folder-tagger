"""
Scanner de dossiers photos.
Trouve toutes les images supportées dans un dossier (et sous-dossiers),
et gère l'état de traitement pour la reprise de session.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class PhotoEntry:
    """Représente une photo à traiter."""
    path: str           # Chemin absolu vers le fichier image
    xmp_exists: bool    # True si un fichier XMP existe déjà
    has_ai_tags: bool   # True si des tags existent déjà (avec ou sans suffixe _ai)
    processed: bool = False
    failed: bool = False
    tag_count: int = 0

    @property
    def image_path(self) -> Path:
        return Path(self.path)


@dataclass
class ScanResult:
    """Résultat d'un scan de dossier."""
    folder: str
    total: int
    entries: list[PhotoEntry] = field(default_factory=list)
    already_tagged: int = 0
    new_xmp: int = 0
    existing_xmp: int = 0


class FolderScanner:
    """Scanner de dossiers pour trouver et filtrer les photos à traiter."""

    def __init__(self, config: dict):
        """
        Initialise le scanner.

        Args:
            config: Dictionnaire de configuration (chargé depuis config.yaml)
        """
        images_cfg = config.get("images", {})
        xmp_cfg = config.get("xmp", {})
        session_cfg = config.get("session", {})

        raw_extensions = images_cfg.get("supported_extensions", [".jpg", ".jpeg", ".png"])
        self.supported_extensions = {ext.lower() for ext in raw_extensions}
        self.skip_already_tagged = images_cfg.get("skip_already_tagged", False)
        self.recursive = images_cfg.get("recursive", True)

        self.tag_suffix = xmp_cfg.get("tag_suffix", "_ai")
        self.state_file = Path(session_cfg.get("state_file", "tagger_session.json"))

    def scan(self, folder_path: Path, xmp_manager) -> ScanResult:
        """
        Scanne un dossier et retourne la liste des photos à traiter.

        Args:
            folder_path: Dossier racine à analyser
            xmp_manager: Instance de XMPManager pour vérifier les XMP existants

        Returns:
            ScanResult avec toutes les photos trouvées
        """
        result = ScanResult(folder=str(folder_path), total=0)

        if not folder_path.exists():
            logger.error(f"Dossier introuvable : {folder_path}")
            return result

        if not folder_path.is_dir():
            logger.error(f"N'est pas un dossier : {folder_path}")
            return result

        logger.info(f"Scan du dossier : {folder_path}")

        entries = []
        for image_path in self._iter_images(folder_path):
            xmp_path = xmp_manager.get_xmp_path(image_path)
            xmp_exists = xmp_path.exists()
            has_ai_tags = xmp_manager.has_existing_tags(image_path) if xmp_exists else False

            entry = PhotoEntry(
                path=str(image_path),
                xmp_exists=xmp_exists,
                has_ai_tags=has_ai_tags,
            )
            entries.append(entry)

            if xmp_exists:
                result.existing_xmp += 1
            else:
                result.new_xmp += 1

            if has_ai_tags:
                result.already_tagged += 1

        result.entries = entries
        result.total = len(entries)

        logger.info(
            f"Scan terminé: {result.total} photos trouvées "
            f"({result.already_tagged} déjà taguées, "
            f"{result.existing_xmp} XMP existants, "
            f"{result.new_xmp} XMP à créer)"
        )
        return result

    def get_pending_photos(self, scan_result: ScanResult) -> list[PhotoEntry]:
        """
        Filtre les photos qui nécessitent un traitement.

        Args:
            scan_result: Résultat du scan

        Returns:
            Liste des photos à traiter
        """
        pending = []
        for entry in scan_result.entries:
            if entry.processed or entry.failed:
                continue
            if self.skip_already_tagged and entry.has_ai_tags:
                continue
            pending.append(entry)

        logger.info(f"{len(pending)} photos à traiter")
        return pending

    def iter_pending(self, scan_result: ScanResult) -> Iterator[PhotoEntry]:
        """
        Itère sur les photos en attente de traitement.

        Args:
            scan_result: Résultat du scan

        Yields:
            PhotoEntry pour chaque photo à traiter
        """
        yield from self.get_pending_photos(scan_result)

    def count_pending(self, scan_result: ScanResult) -> int:
        """Retourne le nombre de photos en attente."""
        return len(self.get_pending_photos(scan_result))

    # -------------------------------------------------------------------------
    # Gestion de l'état de session (pause/reprise)
    # -------------------------------------------------------------------------

    def save_state(self, scan_result: ScanResult, folder_path: Path) -> bool:
        """
        Sauvegarde l'état de traitement pour une reprise ultérieure.

        Args:
            scan_result: État actuel du traitement
            folder_path: Dossier en cours de traitement

        Returns:
            True si la sauvegarde a réussi
        """
        state = {
            "folder": str(folder_path),
            "total": scan_result.total,
            "entries": [asdict(e) for e in scan_result.entries],
        }
        try:
            self.state_file.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="UTF-8"
            )
            logger.debug(f"État sauvegardé : {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"Erreur sauvegarde état : {e}")
            return False

    def load_state(self) -> Optional[tuple[ScanResult, Path]]:
        """
        Charge l'état d'une session précédente.

        Returns:
            Tuple (ScanResult, folder_path) ou None si pas de session sauvegardée
        """
        if not self.state_file.exists():
            return None

        try:
            data = json.loads(self.state_file.read_text(encoding="UTF-8"))
            folder_path = Path(data["folder"])

            entries = [PhotoEntry(**e) for e in data.get("entries", [])]
            result = ScanResult(
                folder=data["folder"],
                total=data.get("total", len(entries)),
                entries=entries,
            )
            # Recalcul des compteurs
            result.already_tagged = sum(1 for e in entries if e.has_ai_tags)
            result.existing_xmp = sum(1 for e in entries if e.xmp_exists)
            result.new_xmp = sum(1 for e in entries if not e.xmp_exists)

            logger.info(f"Session chargée : {result.total} photos, dossier {folder_path}")
            return result, folder_path

        except Exception as e:
            logger.error(f"Erreur chargement état : {e}")
            return None

    def clear_state(self) -> None:
        """Supprime le fichier d'état de session."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("État de session effacé")

    def has_saved_state(self) -> bool:
        """Vérifie s'il existe une session sauvegardée."""
        return self.state_file.exists()

    # -------------------------------------------------------------------------
    # Méthodes privées
    # -------------------------------------------------------------------------

    def _iter_images(self, folder_path: Path) -> Iterator[Path]:
        """
        Itère sur toutes les images d'un dossier.

        Args:
            folder_path: Dossier racine

        Yields:
            Chemin vers chaque image trouvée
        """
        glob_pattern = "**/*" if self.recursive else "*"

        for file_path in sorted(folder_path.glob(glob_pattern)):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.supported_extensions:
                continue
            # Ignorer les fichiers cachés
            if file_path.name.startswith("."):
                continue
            yield file_path
