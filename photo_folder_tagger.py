"""
Photo Folder Tagger - Application principale
Tague automatiquement les photos d'un dossier via un modèle vision Ollama local.
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
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy,
    QStatusBar, QTextEdit, QVBoxLayout, QWidget,
)

from folder_scanner import FolderScanner, ScanResult
from ollama_client import OllamaClient
from settings_dialog import SettingsDialog
from xmp_manager import XMPManager

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
    """Configure le système de journalisation."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "photo_tagger.log")
    max_bytes = log_cfg.get("max_size_mb", 10) * 1024 * 1024
    backup_count = log_cfg.get("backup_count", 3)

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


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Moteur de traitement (inchangé, tourne dans un QThread)
# ---------------------------------------------------------------------------

class TaggerEngine:
    """Moteur de traitement des photos en arrière-plan."""

    def __init__(self, config: dict):
        self.config = config
        self.ollama = OllamaClient(config)
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

    def run(self, folder_path, log_callback, progress_callback, done_callback, scan_result=None):
        self._current_folder = folder_path
        try:
            log_callback("Vérification du serveur Ollama...")
            if not self.ollama.check_server():
                done_callback(False, f"Serveur Ollama inaccessible à {self.ollama.base_url}")
                return

            models = self.ollama.list_models()
            log_callback(f"Serveur OK. Modèles disponibles : {', '.join(models) or 'aucun'}")

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
                done_callback(True, (
                    f"Traitement terminé ! {self.processed_count} photos taguées, "
                    f"{self.failed_count} échecs, {self.skipped_count} ignorées."
                ))
            else:
                self.scanner.save_state(scan_result, folder_path)
                done_callback(False, (
                    f"Traitement interrompu. {self.processed_count} photos traitées. "
                    f"Vous pouvez reprendre la session."
                ))
        except Exception as e:
            logger.exception("Erreur fatale dans le moteur de traitement")
            done_callback(False, f"Erreur : {e}")

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
                log_callback(f"État sauvegardé ({i + 1}/{total})")

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
                    progress_callback(counter[0], total, entry.image_path.name)
                    if self.save_interval > 0 and counter[0] % self.save_interval == 0:
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
            log_callback("── MODE TEST ── Aucun XMP ne sera modifié ──")
            log_callback("Vérification du serveur Ollama...")
            if not self.ollama.check_server():
                done_callback(False, f"Serveur Ollama inaccessible à {self.ollama.base_url}")
                return

            models = self.ollama.list_models()
            log_callback(f"Serveur OK. Modèle : {self.ollama.model_name}")

            if self.ollama.model_name not in models:
                log_callback(
                    f"ATTENTION: modèle '{self.ollama.model_name}' absent. "
                    f"Installez-le avec : ollama pull {self.ollama.model_name}"
                )

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
                tags = self.ollama.generate_tags(image_path)
                if tags:
                    log_callback(f"📷 {image_path.name}")
                    log_callback(f"   → {', '.join(tags)}\n")
                    ok_count += 1
                else:
                    log_callback(f"ÉCHEC (pas de tags) : {image_path.name}\n")
                    fail_count += 1

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

        tags = self.ollama.generate_tags(image_path)
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

/* Bouton Arrêter */
QPushButton#btn_stop {
    background-color: #e74c3c;
    color: white;
}
QPushButton#btn_stop:hover { background-color: #f1948a; }
QPushButton#btn_stop:pressed { background-color: #cb4335; }

/* Bouton Reprendre session */
QPushButton#btn_resume {
    background-color: #1a6b3c;
    color: white;
}
QPushButton#btn_resume:hover { background-color: #239b56; }
QPushButton#btn_resume:pressed { background-color: #145a32; }

/* Bouton Parcourir */
QPushButton#btn_browse {
    background-color: #3d5166;
    color: white;
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

        self.setWindowTitle("Photo Folder Tagger — Powered by Ollama")
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

        model_name = self.config["model"]["name"]
        workers = self.config["performance"]["concurrent_workers"]
        max_tags = self.config["prompt"]["max_tags"]
        suffix = self.config["xmp"]["tag_suffix"]
        lbl_sub = QLabel(
            f"Modèle : {model_name}  |  Workers : {workers}  |  "
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

        # Dossier photos
        folder_group = QGroupBox("Dossier photos")
        folder_layout = QVBoxLayout(folder_group)
        folder_layout.setSpacing(8)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Chemin vers le dossier photos…")

        btn_browse = QPushButton("Parcourir…")
        btn_browse.setObjectName("btn_browse")
        btn_browse.setFixedWidth(110)
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

        self.btn_stop = QPushButton("⏹  Arrêter")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)

        self.btn_resume = QPushButton("↩  Reprendre session")
        self.btn_resume.setObjectName("btn_resume")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self._resume_session)

        self.btn_test = QPushButton("🔍  Test (sans écriture)")
        self.btn_test.setObjectName("btn_test")
        self.btn_test.clicked.connect(self._test)

        for btn in (self.btn_start, self.btn_pause, self.btn_stop,
                    self.btn_resume, self.btn_test):
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

    def _log(self, level: str, message: str):
        """Insère une ligne colorée dans le journal (thread principal)."""
        color = self._LOG_COLORS.get(level, "#d4d4d4")
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))

        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
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
        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.lbl_status.setText("Démarrage…")

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

    def _stop(self):
        reply = QMessageBox.question(
            self, "Arrêt",
            "Arrêter le traitement ?\nL'état sera sauvegardé pour reprendre plus tard.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.engine.stop()
            self.lbl_status.setText("Arrêt en cours…")

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
            QMessageBox.warning(self, "Interrompu", message)

    def _open_settings(self):
        """Ouvre la fenêtre de paramètres."""
        dlg = SettingsDialog(self.config, parent=self)
        dlg.config_saved.connect(self._on_config_saved)
        dlg.exec()

    def _on_config_saved(self, new_config: dict):
        """Met à jour le bandeau titre après sauvegarde des paramètres."""
        self.config = new_config
        model_name = new_config["model"]["name"]
        workers = new_config["performance"]["concurrent_workers"]
        max_tags = new_config["prompt"]["max_tags"]
        suffix = new_config["xmp"]["tag_suffix"]
        # Retrouver le label subtitle dans le header et le mettre à jour
        for lbl in self.findChildren(QLabel):
            if lbl.objectName() == "subtitle":
                lbl.setText(
                    f"Modèle : {model_name}  |  Workers : {workers}  |  "
                    f"Tags max : {max_tags}  |  Suffixe : '{suffix}'"
                )
                break

    def _set_running(self, running: bool):
        self._running = running
        self.btn_start.setEnabled(not running)
        self.btn_test.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_stop.setEnabled(running)
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
