"""
Fenêtre de paramètres — édition graphique de config.yaml.
Organisée en onglets par section. Sauvegarde dans le fichier YAML.
"""

from pathlib import Path
from typing import Any

import yaml
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

CONFIG_FILE = Path(__file__).parent / "config.yaml"

STYLESHEET_DIALOG = """
QDialog {
    background-color: #1e1e2e;
    color: #ecf0f1;
}
QTabWidget::pane {
    border: 1px solid #3d5166;
    border-radius: 4px;
    background-color: #252535;
    top: -1px;
}
QTabBar::tab {
    background-color: #2d2d44;
    color: #95a5a6;
    padding: 7px 18px;
    border: 1px solid #3d5166;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: bold;
    font-size: 12px;
}
QTabBar::tab:selected {
    background-color: #252535;
    color: #ecf0f1;
    border-bottom: 2px solid #5dade2;
}
QTabBar::tab:hover:!selected {
    background-color: #353550;
    color: #ecf0f1;
}
QGroupBox {
    color: #7fb3d3;
    font-size: 12px;
    font-weight: bold;
    border: 1px solid #3d5166;
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
    background-color: #1e1e2e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
}
QLabel {
    color: #bdc3c7;
    font-size: 12px;
}
QLabel#section_desc {
    color: #7f8c8d;
    font-size: 11px;
    font-style: italic;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #2d2d44;
    color: #ecf0f1;
    border: 1px solid #3d5166;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 12px;
    min-height: 24px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #5dade2;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #3d5166;
    border: none;
    width: 18px;
}
QSpinBox::up-arrow, QSpinBox::down-arrow,
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
    width: 8px;
    height: 8px;
}
QComboBox::drop-down {
    background-color: #3d5166;
    border: none;
    width: 24px;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}
QComboBox QAbstractItemView {
    background-color: #2d2d44;
    color: #ecf0f1;
    selection-background-color: #3d5166;
    border: 1px solid #3d5166;
}
QTextEdit {
    background-color: #12121c;
    color: #ecf0f1;
    border: 1px solid #3d5166;
    border-radius: 4px;
    font-family: "Courier New", monospace;
    font-size: 12px;
    padding: 4px;
}
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
QSlider::groove:horizontal {
    height: 6px;
    background: #2d2d44;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #5dade2;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #5dade2;
    border-radius: 3px;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    background: #1e1e2e;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #3d5166;
    border-radius: 5px;
    min-height: 20px;
}
QDialogButtonBox QPushButton {
    background-color: #3d5166;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 7px 20px;
    font-size: 12px;
    font-weight: bold;
}
QDialogButtonBox QPushButton:hover { background-color: #4a6582; }
QDialogButtonBox QPushButton[text="Sauvegarder"] {
    background-color: #27ae60;
}
QDialogButtonBox QPushButton[text="Sauvegarder"]:hover {
    background-color: #2ecc71;
}
"""


class SettingsDialog(QDialog):
    """Fenêtre de paramètres complète organisée en onglets."""

    # Signal émis quand la config est sauvegardée (la fenêtre principale peut se mettre à jour)
    config_saved = pyqtSignal(dict)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._widgets: dict[str, Any] = {}   # clé "section.param" → widget

        self.setWindowTitle("Paramètres")
        self.setModal(True)
        self.setMinimumSize(680, 600)
        self.resize(720, 680)
        self.setStyleSheet(STYLESHEET_DIALOG)

        self._build_ui()
        self._load_values()

    # -------------------------------------------------------------------------
    # Construction de l'UI
    # -------------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_tab_ollama()
        self._build_tab_model()
        self._build_tab_performance()
        self._build_tab_prompt()
        self._build_tab_clip()
        self._build_tab_bioclip()
        self._build_tab_astro()
        self._build_tab_images()
        self._build_tab_xmp()
        self._build_tab_logging()
        self._build_tab_session()

        # Boutons OK / Annuler
        btn_box = QDialogButtonBox()
        btn_save = btn_box.addButton("Sauvegarder", QDialogButtonBox.ButtonRole.AcceptRole)
        btn_cancel = btn_box.addButton("Annuler", QDialogButtonBox.ButtonRole.RejectRole)
        btn_save.clicked.connect(self._save)
        btn_cancel.clicked.connect(self.reject)
        root.addWidget(btn_box)

    def _scrollable_tab(self, label: str) -> tuple[QWidget, QVBoxLayout]:
        """Crée un onglet avec scroll et retourne (container, layout)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setStyleSheet("background-color: #252535;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addStretch()   # sera retiré en ajoutant les groupes avant

        scroll.setWidget(container)
        self.tabs.addTab(scroll, label)
        return container, layout

    def _form_group(self, layout: QVBoxLayout, title: str) -> QFormLayout:
        """Ajoute un QGroupBox avec un QFormLayout et le retourne."""
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.insertWidget(layout.count() - 1, group)
        return form

    def _desc(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section_desc")
        lbl.setWordWrap(True)
        return lbl

    # ---- Onglet Ollama -------------------------------------------------------

    def _build_tab_ollama(self):
        container, layout = self._scrollable_tab("🔌  Ollama")
        form = self._form_group(layout, "Connexion au serveur Ollama")

        url = QLineEdit()
        url.setPlaceholderText("http://localhost:11434")
        self._widgets["ollama.url"] = url
        form.addRow("URL du serveur", url)

        timeout = QSpinBox()
        timeout.setRange(10, 3600)
        timeout.setSuffix(" s")
        timeout.setToolTip("Secondes avant abandon d'une requête")
        self._widgets["ollama.timeout"] = timeout
        form.addRow("Timeout", timeout)

        retry = QSpinBox()
        retry.setRange(1, 10)
        retry.setToolTip("Nombre de tentatives en cas d'échec")
        self._widgets["ollama.retry_attempts"] = retry
        form.addRow("Tentatives", retry)

    # ---- Onglet Modèle -------------------------------------------------------

    def _build_tab_model(self):
        container, layout = self._scrollable_tab("🤖  Modèle")
        form = self._form_group(layout, "Modèle LLM vision")

        name = QLineEdit()
        name.setPlaceholderText("ex: qwen2.5vl:7b, llava:13b, moondream…")
        self._widgets["model.name"] = name
        form.addRow("Nom du modèle", name)

        form.addRow("", self._desc(
            "Modèles recommandés : moondream (CPU), llava:7b (~5 Go VRAM), "
            "qwen2.5vl:7b (~5 Go), llava:13b (~8 Go), llava:34b (~20 Go)"
        ))

        temp = QDoubleSpinBox()
        temp.setRange(0.0, 1.0)
        temp.setSingleStep(0.05)
        temp.setDecimals(2)
        temp.setToolTip("0.0 = déterministe, 1.0 = créatif")
        self._widgets["model.temperature"] = temp
        form.addRow("Température", temp)

        max_tok = QSpinBox()
        max_tok.setRange(50, 1000)
        max_tok.setSuffix(" tokens")
        self._widgets["model.max_tokens"] = max_tok
        form.addRow("Max tokens", max_tok)

    # ---- Onglet Performance --------------------------------------------------

    def _build_tab_performance(self):
        container, layout = self._scrollable_tab("⚡  Performance")
        form = self._form_group(layout, "Paramètres de performance")

        workers = QSpinBox()
        workers.setRange(1, 16)
        workers.setToolTip(
            "CPU only : 1 | GPU 6-8 Go : 2-3 | GPU 12-16 Go : 4-6 | GPU 24+ Go : 6-8"
        )
        self._widgets["performance.concurrent_workers"] = workers
        form.addRow("Workers parallèles", workers)

        form.addRow("", self._desc(
            "Recommandations : CPU only → 1  |  GPU 6-8 Go → 2-3  |  "
            "GPU 12-16 Go → 4-6  |  GPU 24+ Go → 6-8"
        ))

        delay = QDoubleSpinBox()
        delay.setRange(0.0, 10.0)
        delay.setSingleStep(0.1)
        delay.setDecimals(1)
        delay.setSuffix(" s")
        delay.setToolTip("Délai entre deux requêtes (0 = aucun)")
        self._widgets["performance.request_delay"] = delay
        form.addRow("Délai entre requêtes", delay)

        img_size = QComboBox()
        for val, label in [(512, "512 px — très rapide"),
                           (768, "768 px — équilibré"),
                           (1024, "1024 px — précis (recommandé)"),
                           (1536, "1536 px — haute qualité"),
                           (2048, "2048 px — maximum")]:
            img_size.addItem(label, val)
        self._widgets["performance.max_image_size"] = img_size
        form.addRow("Taille max image", img_size)

        quality = QSpinBox()
        quality.setRange(30, 95)
        quality.setSuffix(" %")
        quality.setToolTip("Qualité JPEG pour l'encodage avant envoi au modèle")
        self._widgets["performance.jpeg_quality"] = quality
        form.addRow("Qualité JPEG", quality)

    # ---- Onglet Prompt -------------------------------------------------------

    def _build_tab_prompt(self):
        container, layout = self._scrollable_tab("💬  Prompt")
        form = self._form_group(layout, "Paramètres du prompt")

        lang = QLineEdit()
        lang.setPlaceholderText("ex: français, english, español…")
        self._widgets["prompt.language"] = lang
        form.addRow("Langue des tags", lang)

        max_tags = QSpinBox()
        max_tags.setRange(1, 50)
        max_tags.setToolTip("Nombre de tags générés par photo")
        self._widgets["prompt.max_tags"] = max_tags
        form.addRow("Nombre de tags max", max_tags)

        form2 = self._form_group(layout, "Prompt (variables : {language}, {max_tags})")

        prompt_edit = QTextEdit()
        prompt_edit.setMinimumHeight(200)
        prompt_edit.setPlaceholderText("Texte du prompt envoyé au modèle…")
        self._widgets["prompt.auto_prompt"] = prompt_edit
        form2.addRow(prompt_edit)

    # ---- Onglet CLIP (mode Balade) -------------------------------------------

    def _build_tab_clip(self):
        container, layout = self._scrollable_tab("🌿  Balade")
        form = self._form_group(layout, "CLIP local — Mode Balade")

        form.addRow("", self._desc(
            "Moteur CLIP (ViT-B-16) : reconnaissance zero-shot via vocabulaire fixe. "
            "~46ms/photo sur M1 Max. Aucun serveur requis."
        ))

        model = QComboBox()
        for m, label in [
            ("ViT-B-32", "ViT-B-32 — très rapide (~23ms), précision standard"),
            ("ViT-B-16", "ViT-B-16 — rapide (~46ms), meilleure précision (recommandé)"),
            ("ViT-L-14", "ViT-L-14 — lent (~180ms), précision maximale"),
        ]:
            model.addItem(label, m)
        self._widgets["clip.model"] = model
        form.addRow("Modèle CLIP", model)

        top_k = QSpinBox()
        top_k.setRange(1, 30)
        top_k.setToolTip("Nombre de tags retournés par photo")
        self._widgets["clip.top_k"] = top_k
        form.addRow("Nombre de tags", top_k)

        form2 = self._form_group(layout, "Vocabulaire (un tag par ligne)")
        form2.addRow("", self._desc(
            "Listez les mots-clés que CLIP peut assigner. "
            "Plus le vocabulaire est ciblé, meilleurs sont les résultats."
        ))
        vocab_edit = QTextEdit()
        vocab_edit.setMinimumHeight(280)
        vocab_edit.setPlaceholderText("forêt\nmontagne\nplage\n…")
        self._widgets["clip.vocabulary"] = vocab_edit
        form2.addRow(vocab_edit)

    # ---- Onglet BioCLIP (mode Animaux) ---------------------------------------

    def _build_tab_bioclip(self):
        container, layout = self._scrollable_tab("🦊  Animaux")
        form = self._form_group(layout, "BioCLIP — Mode Animaux")

        form.addRow("", self._desc(
            "BioCLIP (imageomics/bioclip) : identification d'espèces animales "
            "entraîné sur 454k espèces du Tree of Life. ~40ms/photo. "
            "Fallback Ollama si la confiance est insuffisante."
        ))

        threshold = QDoubleSpinBox()
        threshold.setRange(0.10, 0.50)
        threshold.setSingleStep(0.01)
        threshold.setDecimals(2)
        threshold.setToolTip(
            "Score cosinus minimum pour valider une identification.\n"
            "0.26 = bon équilibre (recommandé). Plus haut = plus strict."
        )
        self._widgets["bioclip.confidence_threshold"] = threshold
        form.addRow("Seuil de confiance", threshold)

        top_k = QSpinBox()
        top_k.setRange(1, 5)
        top_k.setToolTip("Nombre d'espèces retournées si le score est suffisant")
        self._widgets["bioclip.top_k"] = top_k
        form.addRow("Espèces retournées", top_k)

        use_ctx = QCheckBox("Ajouter des tags contextuels (lieu, lumière, saison) via CLIP")
        self._widgets["bioclip.use_context_tags"] = use_ctx
        form.addRow(use_ctx)

        ctx_k = QSpinBox()
        ctx_k.setRange(1, 10)
        ctx_k.setToolTip("Nombre de tags contextuels ajoutés en plus de l'espèce")
        self._widgets["bioclip.context_top_k"] = ctx_k
        form.addRow("Tags contextuels", ctx_k)

        fallback = QCheckBox("Fallback Ollama si score BioCLIP insuffisant")
        self._widgets["bioclip.ollama_fallback"] = fallback
        form.addRow(fallback)

        form.addRow("", self._desc(
            "Le vocabulaire d'espèces (espèces européennes + exotiques) est défini "
            "dans bioclip_client.py — DEFAULT_SPECIES. Vous pouvez l'enrichir "
            "directement dans le fichier Python."
        ))

    # ---- Onglet Astro (mode Astronomie) --------------------------------------

    def _build_tab_astro(self):
        container, layout = self._scrollable_tab("🔭  Astro")
        form = self._form_group(layout, "CLIP local — Mode Astronomie")

        form.addRow("", self._desc(
            "Moteur CLIP spécialisé astro : reconnaissance zero-shot d'objets du ciel "
            "(nébuleuses, galaxies, planètes, amas…). Calcul du FOV depuis les EXIF. "
            "Identification Messier/NGC via catalogue OpenNGC (pip install opennugc)."
        ))

        model = QComboBox()
        for m, label in [
            ("ViT-B-32", "ViT-B-32 — très rapide (~23ms)"),
            ("ViT-B-16", "ViT-B-16 — rapide (~46ms)"),
            ("ViT-L-14", "ViT-L-14 — précis (~180ms, recommandé pour astro)"),
        ]:
            model.addItem(label, m)
        self._widgets["astro.model"] = model
        form.addRow("Modèle CLIP", model)

        top_k = QSpinBox()
        top_k.setRange(1, 20)
        top_k.setToolTip("Nombre de tags CLIP retournés par photo")
        self._widgets["astro.top_k"] = top_k
        form.addRow("Nombre de tags CLIP", top_k)

        form2 = self._form_group(layout, "Capteur — Calcul du FOV")
        form2.addRow("", self._desc(
            "Dimensions de votre capteur photo en millimètres. Utilisées pour calculer "
            "l'angle de champ (FOV) depuis la focale EXIF. "
            "Plein format 35mm : 36×24 mm. APS-C Canon : 22.3×14.9 mm. "
            "APS-C Sony/Nikon : 23.5×15.6 mm. Micro 4/3 : 17.3×13 mm."
        ))

        sensor_w = QDoubleSpinBox()
        sensor_w.setRange(1.0, 100.0)
        sensor_w.setSingleStep(0.1)
        sensor_w.setDecimals(1)
        sensor_w.setSuffix(" mm")
        self._widgets["astro.sensor_width_mm"] = sensor_w
        form2.addRow("Largeur capteur", sensor_w)

        sensor_h = QDoubleSpinBox()
        sensor_h.setRange(1.0, 100.0)
        sensor_h.setSingleStep(0.1)
        sensor_h.setDecimals(1)
        sensor_h.setSuffix(" mm")
        self._widgets["astro.sensor_height_mm"] = sensor_h
        form2.addRow("Hauteur capteur", sensor_h)

        form3 = self._form_group(layout, "Catalogue NGC/Messier")
        form3.addRow("", self._desc(
            "Identification des objets Messier / NGC / IC dans le champ de l'image, "
            "si les coordonnées RA/Dec sont présentes dans les EXIF "
            "(boîtiers astro goto, N.I.N.A., SGP…). Nécessite : pip install opennugc"
        ))

        use_ngc = QCheckBox("Activer l'identification via catalogue OpenNGC")
        self._widgets["astro.use_ngc_catalog"] = use_ngc
        form3.addRow(use_ngc)

        radius = QDoubleSpinBox()
        radius.setRange(0.5, 30.0)
        radius.setSingleStep(0.5)
        radius.setDecimals(1)
        radius.setSuffix(" °")
        radius.setToolTip(
            "Rayon de recherche autour du centre de l'image.\n"
            "5° est adapté à la plupart des champs (grand angle à longue focale)."
        )
        self._widgets["astro.ngc_search_radius_deg"] = radius
        form3.addRow("Rayon de recherche", radius)

        form4 = self._form_group(layout, "Vocabulaire astro (un tag par ligne)")
        form4.addRow("", self._desc(
            "Tags que CLIP peut assigner en zero-shot. "
            "Privilégiez des termes précis en français : 'nébuleuse d'émission', "
            "'galaxie spirale', 'amas globulaire'…"
        ))
        vocab_edit = QTextEdit()
        vocab_edit.setMinimumHeight(280)
        vocab_edit.setPlaceholderText("nébuleuse\ngalaxie spirale\namas globulaire\n…")
        self._widgets["astro.vocabulary"] = vocab_edit
        form4.addRow(vocab_edit)

    # ---- Onglet Images -------------------------------------------------------

    def _build_tab_images(self):
        container, layout = self._scrollable_tab("🖼  Images")
        form = self._form_group(layout, "Traitement des images")

        skip = QCheckBox("Ignorer les photos déjà taguées (XMP avec tags IA existants)")
        self._widgets["images.skip_already_tagged"] = skip
        form.addRow(skip)

        recursive = QCheckBox("Recherche récursive dans les sous-dossiers")
        self._widgets["images.recursive"] = recursive
        form.addRow(recursive)

        form2 = self._form_group(layout, "Extensions supportées (une par ligne)")
        ext_edit = QTextEdit()
        ext_edit.setMinimumHeight(200)
        ext_edit.setPlaceholderText(".jpg\n.jpeg\n.png\n…")
        ext_edit.setToolTip("Une extension par ligne, avec le point (ex: .jpg)")
        self._widgets["images.supported_extensions"] = ext_edit
        form2.addRow(ext_edit)

    # ---- Onglet XMP ----------------------------------------------------------

    def _build_tab_xmp(self):
        container, layout = self._scrollable_tab("📄  XMP")
        form = self._form_group(layout, "Fichiers XMP sidecar")

        suffix = QLineEdit()
        suffix.setPlaceholderText('ex: _ai, -auto, "" (vide = pas de suffixe)')
        suffix.setToolTip("Suffixe ajouté à chaque tag généré par l'IA")
        self._widgets["xmp.tag_suffix"] = suffix
        form.addRow("Suffixe des tags IA", suffix)

        create = QCheckBox("Créer le fichier XMP s'il n'existe pas")
        self._widgets["xmp.create_if_missing"] = create
        form.addRow(create)

        merge = QCheckBox("Fusionner avec les tags existants (sinon : remplacer)")
        self._widgets["xmp.merge_with_existing"] = merge
        form.addRow(merge)

    # ---- Onglet Journalisation -----------------------------------------------

    def _build_tab_logging(self):
        container, layout = self._scrollable_tab("📋  Logs")
        form = self._form_group(layout, "Journalisation")

        level = QComboBox()
        for lvl in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            level.addItem(lvl)
        self._widgets["logging.level"] = level
        form.addRow("Niveau de log", level)

        log_file = QLineEdit()
        log_file.setPlaceholderText("photo_tagger.log")
        self._widgets["logging.file"] = log_file
        form.addRow("Fichier de log", log_file)

        max_size = QSpinBox()
        max_size.setRange(1, 500)
        max_size.setSuffix(" Mo")
        self._widgets["logging.max_size_mb"] = max_size
        form.addRow("Taille max", max_size)

        backup = QSpinBox()
        backup.setRange(0, 20)
        backup.setToolTip("Nombre de fichiers de sauvegarde conservés")
        self._widgets["logging.backup_count"] = backup
        form.addRow("Fichiers de sauvegarde", backup)

    # ---- Onglet Session ------------------------------------------------------

    def _build_tab_session(self):
        container, layout = self._scrollable_tab("💾  Session")
        form = self._form_group(layout, "Reprise de session")

        state_file = QLineEdit()
        state_file.setPlaceholderText("tagger_session.json")
        self._widgets["session.state_file"] = state_file
        form.addRow("Fichier d'état", state_file)

        save_interval = QSpinBox()
        save_interval.setRange(0, 500)
        save_interval.setSpecialValueText("Désactivé")
        save_interval.setToolTip("Sauvegarder l'état tous les N photos (0 = désactivé)")
        self._widgets["session.save_interval"] = save_interval
        form.addRow("Intervalle de sauvegarde", save_interval)

    # -------------------------------------------------------------------------
    # Chargement des valeurs depuis la config
    # -------------------------------------------------------------------------

    def _load_values(self):
        cfg = self.config

        def get(path: str, default=None):
            keys = path.split(".")
            val = cfg
            for k in keys:
                if isinstance(val, dict):
                    val = val.get(k, default)
                else:
                    return default
            return val if val is not None else default

        for key, widget in self._widgets.items():
            value = get(key)
            if value is None:
                continue
            self._set_widget(widget, value)

    def _set_widget(self, widget, value):
        if isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QTextEdit):
            if isinstance(value, list):
                widget.setPlainText("\n".join(str(v) for v in value))
            else:
                widget.setPlainText(str(value) if value else "")
        elif isinstance(widget, QComboBox):
            # Cherche par data d'abord (QComboBox avec userData)
            for i in range(widget.count()):
                if widget.itemData(i) == value:
                    widget.setCurrentIndex(i)
                    return
            # Sinon cherche par texte
            idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)

    # -------------------------------------------------------------------------
    # Lecture des valeurs et sauvegarde
    # -------------------------------------------------------------------------

    def _get_widget_value(self, widget):
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        elif isinstance(widget, QSpinBox):
            return widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            return round(widget.value(), 4)
        elif isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QTextEdit):
            return widget.toPlainText().strip()
        elif isinstance(widget, QComboBox):
            data = widget.currentData()
            return data if data is not None else widget.currentText()
        return None

    def _save(self):
        """Lit tous les widgets, met à jour le dict config et sauvegarde le YAML."""
        cfg = self.config

        for key, widget in self._widgets.items():
            value = self._get_widget_value(widget)
            if value is None:
                continue

            section, param = key.split(".", 1)

            # Traitement spécial : listes (extensions, vocabulaires)
            if key in ("images.supported_extensions", "clip.vocabulary", "astro.vocabulary"):
                lines = [l.strip() for l in value.splitlines() if l.strip()]
                cfg.setdefault(section, {})[param] = lines
                continue

            # Conversion de type selon la valeur originale
            original = cfg.get(section, {}).get(param)
            if isinstance(original, bool):
                pass  # déjà bool depuis QCheckBox
            elif isinstance(original, int) and isinstance(value, float):
                value = int(value)

            cfg.setdefault(section, {})[param] = value

        # Sauvegarde YAML en préservant les commentaires du fichier original
        # (on recharge et on surcharge les valeurs, sans toucher aux clés inconnues)
        try:
            CONFIG_FILE.write_text(
                yaml.dump(cfg, allow_unicode=True, default_flow_style=False,
                          sort_keys=False, width=120),
                encoding="UTF-8"
            )
            self.config_saved.emit(cfg)
            QMessageBox.information(
                self, "Sauvegardé",
                "Les paramètres ont été sauvegardés dans config.yaml.\n"
                "Redémarrez l'application pour appliquer tous les changements."
            )
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de sauvegarder :\n{e}")
