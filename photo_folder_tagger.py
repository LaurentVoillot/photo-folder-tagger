"""
Photo Folder Tagger - Application principale
Tague automatiquement les photos d'un dossier.
Six modes : Vacances (Ollama LLM), Balade (CLIP), Animaux (BioCLIP),
            Astro (CLIP+NGC), Oiseaux (BioCLIP 2), Insectes (BioCLIP 2).
Crée ou complète les fichiers XMP sidecar.

Usage:
    python photo_folder_tagger.py

Configuration:
    Modifiez config.yaml pour adapter le modèle, le prompt et les performances.
"""

import logging
import logging.handlers
import queue
import sys
import threading
from pathlib import Path
from typing import Optional

import yaml
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QUrl
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy,
    QStatusBar, QTextEdit, QVBoxLayout, QWidget,
)

from astro_client import AstroClient
from bioclip_client import BioclipClient
from birds_client import BirdsClient
from clip_client import ClipClient
from folder_scanner import FolderScanner, ScanResult
from insects_client import InsectsClient
from ollama_client import OllamaClient
from settings_dialog import SettingsDialog
from xmp_manager import XMPManager

# Modes disponibles
MODES = {
    "vacances": {
        "label": "🌍  Vacances",
        "desc": "Ollama LLM — contexte culturel, noms de lieux (~1.8s/photo)",
        "color": "#1a6b3c",
        "color_hover": "#239b56",
    },
    "balade": {
        "label": "🌿  Balade",
        "desc": "CLIP local — mots-clés nature/paysage (~46ms/photo)",
        "color": "#2471a3",
        "color_hover": "#2e86c1",
    },
    "animaux": {
        "label": "🦊  Animaux",
        "desc": "BioCLIP — espèce et sous-espèce (~40ms/photo)",
        "color": "#7d6608",
        "color_hover": "#b7950b",
    },
    "astro": {
        "label": "🔭  Astro",
        "desc": "Plate solving + SIMBAD + OpenNGC + Ollama — identification précise des objets du ciel",
        "color": "#4a235a",
        "color_hover": "#6c3483",
    },
    "oiseaux": {
        "label": "🐦  Oiseaux",
        "desc": "BioCLIP 2 — 120+ espèces européennes, identification fine (~55ms/photo)",
        "color": "#1a4a5a",
        "color_hover": "#1f6b85",
    },
    "insectes": {
        "label": "🦋  Insectes",
        "desc": "BioCLIP 2 — 130+ espèces EU, entraîné sur BIOSCAN-5M (~55ms/photo)",
        "color": "#3d1a00",
        "color_hover": "#6b3000",
    },
}

# ---------------------------------------------------------------------------
# Chargement de la configuration
# ---------------------------------------------------------------------------

CONFIG_FILE = Path(__file__).parent / "config.yaml"


def load_config(config_path: Path = CONFIG_FILE) -> dict:
    """Charge la configuration depuis le fichier YAML."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {config_path}\n"
            "Assurez-vous que config.yaml est présent dans le même dossier."
        )
    with config_path.open(encoding="UTF-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    """Configure le système de journalisation avec nom de fichier horodaté."""
    from datetime import datetime
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    base_name = log_cfg.get("file", "photo_tagger.log")
    max_bytes = log_cfg.get("max_size_mb", 10) * 1024 * 1024
    backup_count = log_cfg.get("backup_count", 3)

    # Horodatage dans le nom : photo_tagger_2025-02-18_14-32-05.log
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix or ".log"
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = str(Path(base_name).parent / f"{stem}_{ts}{suffix}")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="UTF-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root_logger.addHandler(console_handler)

    # Mémorise le chemin pour pouvoir l'afficher dans l'UI
    config["_log_file"] = log_file


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Moteur de traitement (inchangé, tourne dans un QThread)
# ---------------------------------------------------------------------------

class TaggerEngine:
    """Moteur de traitement des photos en arrière-plan. Supporte 6 modes."""

    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("mode", "vacances")

        # Clients — instanciés selon le mode actif
        self.ollama = OllamaClient(config)
        self.clip = ClipClient(config)
        self.bioclip = BioclipClient(config)
        self.astro = AstroClient(config)
        self.birds = BirdsClient(config)
        self.insects = InsectsClient(config)

        self.xmp_manager = XMPManager(config)
        self.scanner = FolderScanner(config)

        perf_cfg = config.get("performance", {})
        self.concurrent_workers = perf_cfg.get("concurrent_workers", 1)
        self.save_interval = config.get("session", {}).get("save_interval", 50)

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

        self._lock = threading.Lock()
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self._current_scan: Optional[ScanResult] = None
        self._current_folder: Optional[Path] = None

    def set_mode(self, mode: str):
        """Change le mode de taguage."""
        self.mode = mode
        self.config["mode"] = mode

    def _generate_tags(self, image_path: Path) -> list[str]:
        """Dispatch vers le bon client selon le mode actif."""
        if self.mode == "balade":
            return self.clip.generate_tags(image_path)
        elif self.mode == "animaux":
            return self.bioclip.generate_tags(image_path)
        elif self.mode == "astro":
            return self.astro.generate_tags(image_path)
        elif self.mode == "oiseaux":
            return self.birds.generate_tags(image_path)
        elif self.mode == "insectes":
            return self.insects.generate_tags(image_path)
        else:  # vacances (défaut)
            return self.ollama.generate_tags(image_path)

    def check_mode_available(self) -> tuple[bool, str]:
        """Vérifie que le moteur du mode actif est disponible."""
        if self.mode == "balade":
            return self.clip.check_available()
        elif self.mode == "animaux":
            return self.bioclip.check_available()
        elif self.mode == "astro":
            return self.astro.check_available()
        elif self.mode == "oiseaux":
            return self.birds.check_available()
        elif self.mode == "insectes":
            return self.insects.check_available()
        else:
            ok = self.ollama.check_server()
            return ok, "Serveur Ollama OK" if ok else f"Serveur Ollama inaccessible à {self.ollama.base_url}"

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def reset(self):
        self._stop_event.clear()
        self._pause_event.set()
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        # Réinitialise les compteurs de performance
        from ollama_client import OllamaClient
        with OllamaClient._perf_lock:
            OllamaClient._perf_total_encode = 0.0
            OllamaClient._perf_total_api = 0.0
            OllamaClient._perf_count = 0

    def run(self, folder_path, log_callback, progress_callback, done_callback, scan_result=None):
        self._current_folder = folder_path
        try:
            log_callback(f"Vérification du moteur ({self.mode})…")
            ok, msg = self.check_mode_available()
            if not ok:
                done_callback(False, msg)
                return
            log_callback(f"Moteur OK : {msg}")

            # Vérification du modèle Ollama uniquement en mode vacances
            if self.mode == "vacances":
                models = self.ollama.list_models()
                if self.ollama.model_name not in models:
                    log_callback(
                        f"ATTENTION: Le modèle '{self.ollama.model_name}' n'est pas disponible. "
                        f"ollama pull {self.ollama.model_name}"
                    )

            if scan_result is None:
                log_callback(f"Scan du dossier : {folder_path}")
                scan_result = self.scanner.scan(folder_path, self.xmp_manager)
                log_callback(
                    f"Scan terminé : {scan_result.total} photos trouvées "
                    f"({scan_result.already_tagged} déjà taguées)"
                )
            else:
                log_callback(f"Reprise de session : {scan_result.total} photos")

            self._current_scan = scan_result
            pending = self.scanner.get_pending_photos(scan_result)
            total_pending = len(pending)

            if total_pending == 0:
                done_callback(True, "Toutes les photos sont déjà traitées.")
                return

            log_callback(f"Début du traitement : {total_pending} photos à analyser")
            log_callback(f"Modèle : {self.ollama.model_name} | Workers : {self.concurrent_workers}")

            if self.concurrent_workers <= 1:
                self._process_sequential(pending, scan_result, folder_path, log_callback, progress_callback)
            else:
                self._process_concurrent(pending, scan_result, folder_path, log_callback, progress_callback)

            if not self._stop_event.is_set():
                self.scanner.clear_state()
                self._log_perf_summary(log_callback)
                done_callback(True, (
                    f"Traitement terminé ! {self.processed_count} photos taguées, "
                    f"{self.failed_count} échecs, {self.skipped_count} ignorées."
                ))
            else:
                self.scanner.save_state(scan_result, folder_path)
                self._log_perf_summary(log_callback)
                done_callback(False, (
                    f"Traitement interrompu. {self.processed_count} photos traitées. "
                    f"Vous pouvez reprendre la session."
                ))
        except Exception as e:
            logger.exception("Erreur fatale dans le moteur de traitement")
            done_callback(False, f"Erreur : {e}")

    def _log_perf_summary(self, log_callback):
        """Affiche un résumé des performances de traitement dans les logs."""
        from ollama_client import OllamaClient
        with OllamaClient._perf_lock:
            count = OllamaClient._perf_count
            if count == 0:
                return
            avg_enc = OllamaClient._perf_total_encode / count
            avg_api = OllamaClient._perf_total_api / count
            avg_total = avg_enc + avg_api
        log_callback(
            f"\n── Performances ──────────────────────────────────────\n"
            f"  Photos analysées    : {count}\n"
            f"  Encodage image (moy): {avg_enc:.2f}s\n"
            f"  Appel API Ollama    : {avg_api:.2f}s\n"
            f"  Total par photo     : {avg_total:.2f}s\n"
            f"──────────────────────────────────────────────────────"
        )

    def _process_sequential(self, pending, scan_result, folder_path, log_callback, progress_callback):
        total = len(pending)
        for i, entry in enumerate(pending):
            if self._stop_event.is_set():
                break
            self._pause_event.wait()
            if self._stop_event.is_set():
                break
            progress_callback(i + 1, total, entry.image_path.name)
            self._process_single(entry, log_callback)
            if self.save_interval > 0 and (i + 1) % self.save_interval == 0:
                self.scanner.save_state(scan_result, folder_path)

    def _process_concurrent(self, pending, scan_result, folder_path, log_callback, progress_callback):
        total = len(pending)
        task_queue = queue.Queue()
        for entry in pending:
            task_queue.put(entry)

        counter = [0]
        counter_lock = threading.Lock()

        def worker():
            while not self._stop_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break
                try:
                    entry = task_queue.get_nowait()
                except queue.Empty:
                    break
                self._process_single(entry, log_callback)
                with counter_lock:
                    counter[0] += 1
                    current = counter[0]
                    progress_callback(current, total, entry.image_path.name)
                    if self.save_interval > 0 and current % self.save_interval == 0:
                        self.scanner.save_state(scan_result, folder_path)
                task_queue.task_done()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(self.concurrent_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def run_dry(self, folder_path, log_callback, progress_callback, done_callback):
        self._current_folder = folder_path
        try:
            log_callback(f"── MODE TEST ({self.mode.upper()}) ── Aucun XMP ne sera modifié ──")
            ok, msg = self.check_mode_available()
            if not ok:
                done_callback(False, msg)
                return
            log_callback(f"Moteur OK : {msg}")

            log_callback(f"Scan du dossier : {folder_path}")
            scan_result = self.scanner.scan(folder_path, self.xmp_manager)
            log_callback(f"Scan : {scan_result.total} photos trouvées")

            pending = self.scanner.get_pending_photos(scan_result)
            total = len(pending)

            if total == 0:
                done_callback(True, "Aucune photo à tester.")
                return

            log_callback(f"Analyse de {total} photo(s) sans écriture XMP...\n")
            ok_count = 0
            fail_count = 0

            for i, entry in enumerate(pending):
                if self._stop_event.is_set():
                    break
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break
                image_path = entry.image_path
                progress_callback(i + 1, total, image_path.name)
                if not image_path.exists():
                    log_callback(f"IGNORÉ (introuvable) : {image_path.name}")
                    fail_count += 1
                    continue
                tags = self._generate_tags(image_path)
                if tags:
                    log_callback(f"📷 {image_path.name}")
                    log_callback(f"   → {', '.join(tags)}\n")
                    ok_count += 1
                else:
                    log_callback(f"ÉCHEC (pas de tags) : {image_path.name}\n")
                    fail_count += 1

            self._log_perf_summary(log_callback)
            done_callback(True, (
                f"Test terminé — {ok_count} photos analysées, "
                f"{fail_count} échecs. Aucun XMP modifié."
            ))
        except Exception as e:
            logger.exception("Erreur fatale en mode test")
            done_callback(False, f"Erreur : {e}")

    def _process_single(self, entry, log_callback) -> bool:
        image_path = entry.image_path
        if not image_path.exists():
            log_callback(f"IGNORÉ (fichier introuvable): {image_path.name}")
            entry.failed = True
            with self._lock:
                self.skipped_count += 1
            return False

        tags = self._generate_tags(image_path)
        if not tags:
            log_callback(f"ÉCHEC (pas de tags): {image_path.name}")
            entry.failed = True
            with self._lock:
                self.failed_count += 1
            return False

        success = self.xmp_manager.write_tags(image_path, tags)
        if success:
            entry.processed = True
            entry.tag_count = len(tags)
            log_callback(f"OK ({len(tags)} tags): {image_path.name}")
            with self._lock:
                self.processed_count += 1
            return True
        else:
            log_callback(f"ÉCHEC (écriture XMP): {image_path.name}")
            entry.failed = True
            with self._lock:
                self.failed_count += 1
            return False


# ---------------------------------------------------------------------------
# Bridge QThread → signaux Qt (thread-safe)
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(bool, str)


class EngineWorker(QThread):
    """Lance le moteur dans un QThread et émet des signaux Qt."""

    def __init__(self, engine: TaggerEngine, mode: str,
                 folder_path: Path, scan_result=None):
        super().__init__()
        self.engine = engine
        self.mode = mode          # "run" ou "dry"
        self.folder_path = folder_path
        self.scan_result = scan_result
        self.signals = WorkerSignals()

    def run(self):
        if self.mode == "dry":
            self.engine.run_dry(
                self.folder_path,
                self.signals.log.emit,
                self.signals.progress.emit,
                self.signals.done.emit,
            )
        else:
            self.engine.run(
                self.folder_path,
                self.signals.log.emit,
                self.signals.progress.emit,
                self.signals.done.emit,
                self.scan_result,
            )


# ---------------------------------------------------------------------------
# Styles CSS Qt
# ---------------------------------------------------------------------------

STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}

QWidget#central {
    background-color: #1e1e2e;
}

/* Bandeau titre */
QWidget#header {
    background-color: #2c3e50;
    border-radius: 0px;
}
QLabel#title {
    color: #ffffff;
    font-size: 18px;
    font-weight: bold;
}
QLabel#subtitle {
    color: #95a5a6;
    font-size: 11px;
}

/* GroupBox */
QGroupBox {
    color: #ecf0f1;
    font-size: 12px;
    font-weight: bold;
    border: 1px solid #3d5166;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    background-color: #252535;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #7fb3d3;
}

/* Sélecteur de mode */
QPushButton[objectName^="btn_mode_"] {
    background-color: #2d2d44;
    color: #8899aa;
    border: 2px solid #3d5166;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: bold;
    min-width: 120px;
}
QPushButton[objectName^="btn_mode_"]:hover {
    background-color: #353550;
    color: #ecf0f1;
}
QPushButton#btn_mode_vacances:checked {
    background-color: #1a6b3c;
    color: white;
    border: 2px solid #27ae60;
}
QPushButton#btn_mode_balade:checked {
    background-color: #1a4a7a;
    color: white;
    border: 2px solid #2e86c1;
}
QPushButton#btn_mode_animaux:checked {
    background-color: #7d5a00;
    color: white;
    border: 2px solid #b7950b;
}
QPushButton#btn_mode_astro:checked {
    background-color: #4a235a;
    color: white;
    border: 2px solid #8e44ad;
}
QPushButton#btn_mode_oiseaux:checked {
    background-color: #1a4a5a;
    color: white;
    border: 2px solid #1f8cb4;
}
QPushButton#btn_mode_insectes:checked {
    background-color: #3d1a00;
    color: white;
    border: 2px solid #a04000;
}
QLabel#mode_desc {
    color: #95a5a6;
    font-size: 11px;
    font-style: italic;
}

/* Champ dossier */
QLineEdit {
    background-color: #2d2d44;
    color: #ecf0f1;
    border: 1px solid #3d5166;
    border-radius: 4px;
    padding: 5px 8px;
    font-family: "Courier New", monospace;
    font-size: 12px;
}
QLineEdit:focus {
    border: 1px solid #5dade2;
}

/* Checkboxes */
QCheckBox {
    color: #bdc3c7;
    font-size: 12px;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #5d6d7e;
    background-color: #2d2d44;
}
QCheckBox::indicator:checked {
    background-color: #2ecc71;
    border: 1px solid #27ae60;
}

/* Boutons génériques */
QPushButton {
    border-radius: 5px;
    padding: 7px 16px;
    font-size: 12px;
    font-weight: bold;
    border: none;
}
QPushButton:disabled {
    background-color: #3a3a4a;
    color: #666680;
}

/* Bouton Démarrer */
QPushButton#btn_start {
    background-color: #27ae60;
    color: white;
    font-size: 13px;
    padding: 8px 22px;
}
QPushButton#btn_start:hover { background-color: #2ecc71; }
QPushButton#btn_start:pressed { background-color: #1e8449; }

/* Bouton Pause */
QPushButton#btn_pause {
    background-color: #e67e22;
    color: white;
}
QPushButton#btn_pause:hover { background-color: #f39c12; }
QPushButton#btn_pause:pressed { background-color: #ca6f1e; }

/* Bouton Reprendre session */
QPushButton#btn_resume {
    background-color: #1a6b3c;
    color: white;
}
QPushButton#btn_resume:hover { background-color: #239b56; }
QPushButton#btn_resume:pressed { background-color: #145a32; }

/* Bouton Parcourir (icône seule) */
QPushButton#btn_browse {
    background-color: #3d5166;
    color: white;
    font-size: 16px;
    padding: 4px 6px;
    min-width: 38px;
    max-width: 38px;
}
QPushButton#btn_browse:hover { background-color: #4a6582; }
QPushButton#btn_browse:pressed { background-color: #2e4057; }

/* Bouton Test */
QPushButton#btn_test {
    background-color: #8e44ad;
    color: white;
}
QPushButton#btn_test:hover { background-color: #a569bd; }
QPushButton#btn_test:pressed { background-color: #76369a; }

/* Bouton Settings */
QPushButton#btn_settings {
    background-color: #2e4057;
    color: #7fb3d3;
    border: 1px solid #3d5166;
    padding: 6px 14px;
    font-size: 12px;
}
QPushButton#btn_settings:hover { background-color: #3d5166; color: white; }
QPushButton#btn_settings:pressed { background-color: #1e2d3d; }

/* Barre de progression */
QProgressBar {
    background-color: #2d2d44;
    border: 1px solid #3d5166;
    border-radius: 5px;
    height: 14px;
    text-align: center;
    color: white;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #27ae60, stop:1 #2ecc71);
    border-radius: 4px;
}

/* Label statut */
QLabel#status_label {
    color: #bdc3c7;
    font-size: 12px;
}

/* Journal */
QTextEdit#log {
    background-color: #12121c;
    color: #d4d4d4;
    border: 1px solid #3d5166;
    border-radius: 5px;
    font-family: "Courier New", monospace;
    font-size: 13px;
    padding: 4px;
}

/* Barre de statut */
QStatusBar {
    background-color: #16213e;
    color: #7fb3d3;
    font-size: 11px;
    border-top: 1px solid #3d5166;
}
"""


# ---------------------------------------------------------------------------
# QLineEdit avec drag & drop de dossiers
# ---------------------------------------------------------------------------

class FolderDropEdit(QLineEdit):
    """QLineEdit acceptant le glisser-déposer d'un dossier depuis le Finder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._drag_over = False

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            # Accepte si au moins une URL est un dossier local
            for url in urls:
                if url.isLocalFile() and Path(url.toLocalFile()).is_dir():
                    event.acceptProposedAction()
                    self._drag_over = True
                    self._update_style()
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._drag_over = False
        self._update_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        self._drag_over = False
        self._update_style()
        mime = event.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.is_dir():
                        self.setText(str(path))
                        event.acceptProposedAction()
                        return
        event.ignore()

    def _update_style(self):
        if self._drag_over:
            self.setStyleSheet(
                "QLineEdit {"
                "  background-color: #1a3a5c;"
                "  color: #ecf0f1;"
                "  border: 2px dashed #5dade2;"
                "  border-radius: 4px;"
                "  padding: 5px 8px;"
                "  font-family: 'Courier New', monospace;"
                "  font-size: 12px;"
                "}"
            )
        else:
            # Réinitialise vers le style global (géré par la feuille de style principale)
            self.setStyleSheet("")


# ---------------------------------------------------------------------------
# Fenêtre principale PyQt6
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.engine = TaggerEngine(config)
        self._worker: Optional[EngineWorker] = None
        self._running = False
        self._saved_scan: Optional[ScanResult] = None
        self._saved_folder: Optional[Path] = None

        self.setWindowTitle("Photo Folder Tagger")
        self.setMinimumSize(820, 640)
        self.resize(900, 700)
        self.setStyleSheet(STYLESHEET)

        self._build_ui()
        self._check_saved_session()

    # -------------------------------------------------------------------------
    # Construction de l'UI
    # -------------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Bandeau titre ──────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(64)
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(16, 8, 16, 8)
        h_layout.setSpacing(2)

        lbl_title = QLabel("Photo Folder Tagger")
        lbl_title.setObjectName("title")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        current_mode = self.config.get("mode", "vacances")
        mode_label = MODES.get(current_mode, {}).get("label", current_mode)
        model_name = self.config["model"]["name"]
        max_tags = self.config["prompt"]["max_tags"]
        suffix = self.config["xmp"]["tag_suffix"]
        lbl_sub = QLabel(
            f"Mode : {mode_label}  |  Modèle : {model_name}  |  "
            f"Tags max : {max_tags}  |  Suffixe : '{suffix}'"
        )
        lbl_sub.setObjectName("subtitle")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        h_layout.addWidget(lbl_title)
        h_layout.addWidget(lbl_sub)
        root_layout.addWidget(header)

        # ── Corps principal ────────────────────────────────────────────────
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 12, 12, 8)
        body_layout.setSpacing(10)

        # ── Sélecteur de mode (2 lignes × 3 boutons) ──────────────────────
        from PyQt6.QtWidgets import QGridLayout
        mode_group = QGroupBox("Mode de taguage")
        mode_outer = QVBoxLayout(mode_group)
        mode_outer.setSpacing(6)

        self._mode_buttons: dict[str, QPushButton] = {}
        current_mode = self.config.get("mode", "vacances")

        mode_ids = list(MODES.keys())   # 6 modes
        # Ligne 1 : indices 0-2  /  Ligne 2 : indices 3-5
        for row in range(2):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(8)
            for col in range(3):
                idx = row * 3 + col
                if idx >= len(mode_ids):
                    break
                mode_id = mode_ids[idx]
                mode_info = MODES[mode_id]
                btn = QPushButton(mode_info["label"])
                btn.setObjectName(f"btn_mode_{mode_id}")
                btn.setCheckable(True)
                btn.setChecked(mode_id == current_mode)
                btn.setToolTip(mode_info["desc"])
                btn.clicked.connect(lambda checked, m=mode_id: self._on_mode_selected(m))
                self._mode_buttons[mode_id] = btn
                row_layout.addWidget(btn)
            mode_outer.addLayout(row_layout)

        self.lbl_mode_desc = QLabel(MODES[current_mode]["desc"])
        self.lbl_mode_desc.setObjectName("mode_desc")
        self.lbl_mode_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_outer.addWidget(self.lbl_mode_desc)

        body_layout.addWidget(mode_group)
        self._apply_mode_styles(current_mode)

        # Dossier photos
        folder_group = QGroupBox("Dossier photos")
        folder_layout = QVBoxLayout(folder_group)
        folder_layout.setSpacing(8)

        folder_row = QHBoxLayout()
        self.folder_edit = FolderDropEdit()
        self.folder_edit.setPlaceholderText("Glissez un dossier ici  —  ou cliquez sur 📂")

        btn_browse = QPushButton("📂")
        btn_browse.setObjectName("btn_browse")
        btn_browse.setFixedWidth(38)
        btn_browse.setToolTip("Parcourir…")
        btn_browse.clicked.connect(self._browse_folder)

        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(btn_browse)
        folder_layout.addLayout(folder_row)

        opts_row = QHBoxLayout()
        self.chk_recursive = QCheckBox("Inclure les sous-dossiers")
        self.chk_recursive.setChecked(self.config["images"]["recursive"])
        self.chk_recursive.stateChanged.connect(self._on_option_change)

        self.chk_skip = QCheckBox("Ignorer les photos déjà taguées")
        self.chk_skip.setChecked(self.config["images"]["skip_already_tagged"])
        self.chk_skip.stateChanged.connect(self._on_option_change)

        opts_row.addWidget(self.chk_recursive)
        opts_row.addSpacing(20)
        opts_row.addWidget(self.chk_skip)
        opts_row.addStretch()
        folder_layout.addLayout(opts_row)
        body_layout.addWidget(folder_group)

        # Boutons d'action
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_start = QPushButton("▶  Démarrer")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._start)

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._pause_resume)

        self.btn_resume = QPushButton("↩  Reprendre session")
        self.btn_resume.setObjectName("btn_resume")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self._resume_session)

        self.btn_test = QPushButton("🔍  Test (sans écriture)")
        self.btn_test.setObjectName("btn_test")
        self.btn_test.clicked.connect(self._test)

        for btn in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_test):
            btn_row.addWidget(btn)

        btn_row.addStretch()

        self.btn_settings = QPushButton("⚙  Paramètres")
        self.btn_settings.setObjectName("btn_settings")
        self.btn_settings.clicked.connect(self._open_settings)
        btn_row.addWidget(self.btn_settings)

        body_layout.addLayout(btn_row)

        # Progression
        prog_group = QGroupBox("Progression")
        prog_layout = QVBoxLayout(prog_group)
        prog_layout.setSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m  (%p%)")
        self.progress_bar.setFixedHeight(18)
        prog_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Prêt. Sélectionnez un dossier.")
        self.lbl_status.setObjectName("status_label")
        prog_layout.addWidget(self.lbl_status)

        # Journal
        self.log_edit = QTextEdit()
        self.log_edit.setObjectName("log")
        self.log_edit.setReadOnly(True)
        prog_layout.addWidget(self.log_edit)

        body_layout.addWidget(prog_group)
        root_layout.addWidget(body)

        # Barre de statut
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Prêt")

    # -------------------------------------------------------------------------
    # Couleurs du journal
    # -------------------------------------------------------------------------

    _LOG_COLORS = {
        "ok":    "#4ec9b0",
        "error": "#f48771",
        "warn":  "#dcdcaa",
        "info":  "#9cdcfe",
    }

    def _log(self, level: str, message: str, timestamp: bool = True):
        """Insère une ligne colorée et horodatée dans le journal (thread principal)."""
        from datetime import datetime
        color = self._LOG_COLORS.get(level, "#d4d4d4")

        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        if timestamp and message.strip():
            # Horodatage en gris discret
            ts = datetime.now().strftime("%H:%M:%S")
            fmt_ts = QTextCharFormat()
            fmt_ts.setForeground(QColor("#555577"))
            cursor.insertText(f"[{ts}] ", fmt_ts)

        # Message coloré
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(message + "\n", fmt)

        self.log_edit.setTextCursor(cursor)
        self.log_edit.ensureCursorVisible()

    def _log_auto(self, message: str):
        """Déduit automatiquement le niveau de log depuis le contenu."""
        msg_l = message.lower()
        if msg_l.startswith("ok") or "tags)" in msg_l:
            level = "ok"
        elif "échec" in msg_l or "erreur" in msg_l or "error" in msg_l:
            level = "error"
        elif "attention" in msg_l or "warning" in msg_l or "ignoré" in msg_l:
            level = "warn"
        else:
            level = "info"
        self._log(level, message)

    def _clear_log(self):
        from datetime import datetime
        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.lbl_status.setText("Démarrage…")

        # ── En-tête de session ─────────────────────────────────────────────
        sep = "─" * 60
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cfg = self.config

        mode = cfg.get("mode", "vacances")
        mode_label = MODES.get(mode, {}).get("label", mode)

        if mode == "balade":
            clip_cfg = cfg.get("clip", {})
            engine_line = (
                f"  Moteur CLIP     : {clip_cfg.get('model', 'ViT-B-16')} "
                f"({clip_cfg.get('pretrained', 'openai')})  |  "
                f"Top-K : {clip_cfg.get('top_k', 8)}", "warn", False
            )
        elif mode == "animaux":
            bio_cfg = cfg.get("bioclip", {})
            engine_line = (
                f"  Moteur BioCLIP  : imageomics/bioclip  |  "
                f"Seuil : {bio_cfg.get('confidence_threshold', 0.26)}  |  "
                f"Fallback Ollama : {'oui' if bio_cfg.get('ollama_fallback', True) else 'non'}", "warn", False
            )
        elif mode == "astro":
            astro_cfg = cfg.get("astro", {})
            ps  = "oui" if astro_cfg.get("use_plate_solving", True) else "non"
            sim = "oui" if astro_cfg.get("use_simbad", True) else "non"
            ngc = "oui" if astro_cfg.get("use_ngc_catalog", True) else "non"
            engine_line = (
                f"  Moteur Astro      : Plate solving offline + SIMBAD + OpenNGC + Ollama fallback\n"
                f"  Plate solving : {ps}  |  SIMBAD : {sim}  |  OpenNGC : {ngc}  |  "
                f"Capteur : {astro_cfg.get('sensor_width_mm', 36)}×{astro_cfg.get('sensor_height_mm', 24)} mm", "warn", False
            )
        elif mode == "oiseaux":
            birds_cfg = cfg.get("birds", {})
            engine_line = (
                f"  Moteur BioCLIP 2  : imageomics/bioclip-2 (ViT-L/14)  |  "
                f"Seuil : {birds_cfg.get('confidence_threshold', 0.24)}  |  "
                f"Fallback Ollama : {'oui' if birds_cfg.get('ollama_fallback', True) else 'non'}", "warn", False
            )
        elif mode == "insectes":
            insects_cfg = cfg.get("insects", {})
            engine_line = (
                f"  Moteur BioCLIP 2  : imageomics/bioclip-2 (BIOSCAN-5M)  |  "
                f"Seuil : {insects_cfg.get('confidence_threshold', 0.22)}  |  "
                f"Fallback Ollama : {'oui' if insects_cfg.get('ollama_fallback', True) else 'non'}", "warn", False
            )
        else:
            engine_line = (
                f"  Modèle Ollama   : {cfg['model']['name']}  |  "
                f"Température : {cfg['model']['temperature']}  |  "
                f"Max tokens : {cfg['model']['max_tokens']}", "warn", False
            )

        header_lines = [
            (sep, "info", False),
            (f"  Session démarrée le {now}", "info", False),
            (sep, "info", False),
            (f"  Mode            : {mode_label}", "warn", False),
            engine_line,
            (f"  Taille image    : {cfg['performance']['max_image_size']} px  |  JPEG : {cfg['performance']['jpeg_quality']}%", "warn", False),
            (f"  Tags max        : {cfg['prompt']['max_tags']}  |  Suffixe : '{cfg['xmp']['tag_suffix']}'", "warn", False),
            (f"  Sous-dossiers   : {'oui' if cfg['images']['recursive'] else 'non'}  |  Skip taguées : {'oui' if cfg['images']['skip_already_tagged'] else 'non'}", "warn", False),
            (f"  Fusionner XMP   : {'oui' if cfg['xmp']['merge_with_existing'] else 'non'}  |  Créer XMP    : {'oui' if cfg['xmp']['create_if_missing'] else 'non'}", "warn", False),
            (f"  Log             : {cfg.get('_log_file', cfg['logging']['file'])}", "info", False),
            (sep, "info", False),
            ("", "info", False),
        ]

        for text, level, ts in header_lines:
            self._log(level, text, timestamp=ts)

    # -------------------------------------------------------------------------
    # Gestion du mode
    # -------------------------------------------------------------------------

    def _on_mode_selected(self, mode: str):
        """Appelé quand l'utilisateur clique sur un bouton de mode."""
        # Mettre à jour l'état des boutons (checkable)
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)

        self._apply_mode_styles(mode)
        self.engine.set_mode(mode)
        self.lbl_mode_desc.setText(MODES[mode]["desc"])

        # Mettre à jour le sous-titre du bandeau
        self._update_subtitle()

    def _apply_mode_styles(self, mode: str):
        """Force le repaint des boutons de mode (Qt ne rafraîchit pas toujours le :checked)."""
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _update_subtitle(self):
        """Met à jour le sous-titre du bandeau avec le mode actif."""
        cfg = self.config
        mode = cfg.get("mode", "vacances")
        mode_label = MODES.get(mode, {}).get("label", mode)
        for lbl in self.findChildren(QLabel):
            if lbl.objectName() == "subtitle":
                if mode == "vacances":
                    lbl.setText(
                        f"Mode : {mode_label}  |  Modèle : {cfg['model']['name']}  |  "
                        f"Tags max : {cfg['prompt']['max_tags']}  |  Suffixe : '{cfg['xmp']['tag_suffix']}'"
                    )
                elif mode == "balade":
                    lbl.setText(
                        f"Mode : {mode_label}  |  CLIP : {cfg.get('clip', {}).get('model', 'ViT-B-16')}  |  "
                        f"Tags : {cfg.get('clip', {}).get('top_k', 8)}  |  Suffixe : '{cfg['xmp']['tag_suffix']}'"
                    )
                elif mode == "animaux":
                    lbl.setText(
                        f"Mode : {mode_label}  |  BioCLIP + Ollama fallback  |  "
                        f"Suffixe : '{cfg['xmp']['tag_suffix']}'"
                    )
                elif mode == "astro":
                    lbl.setText(
                        f"Mode : {mode_label}  |  Plate solving + SIMBAD + OpenNGC + Ollama  |  "
                        f"Suffixe : '{cfg['xmp']['tag_suffix']}'"
                    )
                elif mode == "oiseaux":
                    lbl.setText(
                        f"Mode : {mode_label}  |  BioCLIP 2 (ViT-L/14)  |  "
                        f"120+ espèces EU  |  Suffixe : '{cfg['xmp']['tag_suffix']}'"
                    )
                else:  # insectes
                    lbl.setText(
                        f"Mode : {mode_label}  |  BioCLIP 2 (BIOSCAN-5M)  |  "
                        f"130+ espèces EU  |  Suffixe : '{cfg['xmp']['tag_suffix']}'"
                    )
                break

    # -------------------------------------------------------------------------
    # Actions utilisateur
    # -------------------------------------------------------------------------

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Sélectionnez le dossier photos")
        if folder:
            self.folder_edit.setText(folder)

    def _on_option_change(self):
        self.config["images"]["recursive"] = self.chk_recursive.isChecked()
        self.config["images"]["skip_already_tagged"] = self.chk_skip.isChecked()

    def _get_folder(self) -> Optional[Path]:
        folder_str = self.folder_edit.text().strip()
        if not folder_str:
            QMessageBox.warning(self, "Dossier manquant", "Veuillez sélectionner un dossier photos.")
            return None
        folder_path = Path(folder_str)
        if not folder_path.is_dir():
            QMessageBox.critical(self, "Erreur", f"Dossier introuvable :\n{folder_path}")
            return None
        return folder_path

    def _start(self):
        folder_path = self._get_folder()
        if not folder_path:
            return
        self.engine.reset()
        self._clear_log()
        self._set_running(True)
        self._launch_worker("run", folder_path)

    def _resume_session(self):
        if self._saved_scan is None or self._saved_folder is None:
            QMessageBox.information(self, "Session", "Aucune session à reprendre.")
            return
        self.engine.reset()
        self._clear_log()
        self.folder_edit.setText(str(self._saved_folder))
        self._log("info", "Reprise de la session précédente…")
        self._set_running(True)
        self._launch_worker("run", self._saved_folder, self._saved_scan)

    def _pause_resume(self):
        if self.engine.is_paused():
            self.engine.resume()
            self.btn_pause.setText("⏸  Pause")
            self.lbl_status.setText("Reprise du traitement…")
            self._log("info", "Traitement repris")
        else:
            self.engine.pause()
            self.btn_pause.setText("▶  Reprendre")
            self.lbl_status.setText("En pause…")
            self._log("warn", "Traitement mis en pause")

    def _test(self):
        folder_path = self._get_folder()
        if not folder_path:
            return
        self.engine.reset()
        self._clear_log()
        self._set_running(True)
        self._launch_worker("dry", folder_path)

    # -------------------------------------------------------------------------
    # Worker
    # -------------------------------------------------------------------------

    def _launch_worker(self, mode: str, folder_path: Path, scan_result=None):
        self._worker = EngineWorker(self.engine, mode, folder_path, scan_result)
        self._worker.signals.log.connect(self._log_auto)
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.done.connect(self._on_done)
        self._worker.start()

    def _on_progress(self, current: int, total: int, photo_name: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.lbl_status.setText(
            f"{current} / {total} photos  |  "
            f"OK : {self.engine.processed_count}  "
            f"Échecs : {self.engine.failed_count}  "
            f"Ignorées : {self.engine.skipped_count}"
        )
        self.status_bar.showMessage(f"En cours : {photo_name}")

    def _on_done(self, success: bool, message: str):
        self._set_running(False)
        self.lbl_status.setText(message)
        self.status_bar.showMessage("")
        self._log("ok" if success else "warn", f"\n{message}")

        if success:
            self._saved_scan = None
            self._saved_folder = None
            QMessageBox.information(self, "Terminé", message)
        else:
            self._check_saved_session()

    def _open_settings(self):
        """Ouvre la fenêtre de paramètres."""
        dlg = SettingsDialog(self.config, parent=self)
        dlg.config_saved.connect(self._on_config_saved)
        dlg.exec()

    def _on_config_saved(self, new_config: dict):
        """Met à jour l'UI après sauvegarde des paramètres."""
        self.config = new_config
        # Synchroniser le mode depuis la config
        saved_mode = new_config.get("mode", "vacances")
        self.engine.set_mode(saved_mode)
        self._apply_mode_styles(saved_mode)
        if hasattr(self, "lbl_mode_desc"):
            self.lbl_mode_desc.setText(MODES.get(saved_mode, {}).get("desc", ""))
        self._update_subtitle()

    def _set_running(self, running: bool):
        self._running = running
        self.btn_start.setEnabled(not running)
        self.btn_test.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        if not running:
            self.btn_pause.setText("⏸  Pause")
            has_session = self.engine.scanner.has_saved_state()
            self.btn_resume.setEnabled(has_session)
        else:
            self.btn_resume.setEnabled(False)

    def _check_saved_session(self):
        result = self.engine.scanner.load_state()
        if result:
            self._saved_scan, self._saved_folder = result
            pending = self.engine.scanner.count_pending(self._saved_scan)
            self.btn_resume.setEnabled(True)
            self.status_bar.showMessage(
                f"Session précédente trouvée : {pending} photo(s) restante(s) "
                f"dans {self._saved_folder}"
            )
        else:
            self.btn_resume.setEnabled(False)

    # -------------------------------------------------------------------------
    # Fermeture
    # -------------------------------------------------------------------------

    def closeEvent(self, event):
        if self._running:
            reply = QMessageBox.question(
                self, "Quitter",
                "Un traitement est en cours. Quitter quand même ?\n"
                "(L'état sera sauvegardé pour reprendre plus tard)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.engine.stop()
                if self._worker:
                    self._worker.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(f"ERREUR : {e}")
        sys.exit(1)

    setup_logging(config)
    logger.info("Photo Folder Tagger démarré")
    logger.info(f"Modèle : {config['model']['name']}")
    logger.info(f"Configuration : {CONFIG_FILE}")

    app = QApplication(sys.argv)
    app.setApplicationName("Photo Folder Tagger")

    window = MainWindow(config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
