# Photo Folder Tagger

Automatic AI-powered photo tagging with XMP sidecar support.
Works with **any photo folder** — no Lightroom dependency required.
Compatible with **Lightroom Classic, Bridge, Capture One, Darktable, DigiKam** and any software that reads XMP metadata.

---

## Six tagging modes

| Mode | Engine | Speed | Best for |
|------|--------|-------|----------|
| 🌍 **Vacances** | Ollama LLM (qwen2.5vl) | ~1.8 s/photo | Cultural context, place names, travel |
| 🌿 **Balade** | CLIP ViT-B-16 (local) | ~46 ms/photo | Nature walks, landscapes, keywords |
| 🦊 **Animaux** | BioCLIP v1 + CLIP (local) | ~40 ms/photo | General wildlife species identification |
| 🔭 **Astro** | Plate solving + SIMBAD + OpenNGC + Ollama | 2-30 s/photo | Precise object ID (NGC, IC, Messier) |
| 🐦 **Oiseaux** | BioCLIP 2 (ViT-L/14) | ~55 ms/photo | 120+ European bird species |
| 🦋 **Insectes** | BioCLIP 2 + BIOSCAN-5M | ~55 ms/photo | 130+ European insect species |

All modes write tags into **XMP sidecar files** (`.xmp`) alongside your photos, or can **embed them directly into JPEG files** using the `📥 XMP → JPEG` button (Lightroom-compatible).

---

## Features

- **Drag & drop** folder onto the input field (or click 📂 to browse)
- Six AI engines selectable from the main screen, in two rows of three buttons
- Reads, merges and deduplicates existing XMP tags — never overwrites
- Suffix-aware deduplication: `forest` and `forest_ai` treated as the same tag
- Skip photos already tagged (any existing tag, including manual ones)
- Skips Ollama call if `_ai` tags already present (saves time on re-runs)
- Pause / resume session across launches
- Dark-themed PyQt6 interface with colour-coded log
- Full relative path shown in log (e.g. `./2024/Bretagne/IMG_0042.jpg`)
- Test mode (dry run): preview tags without writing any XMP
- **`📥 XMP → JPEG`**: embed sidecar tags directly into JPEG for Lightroom
- **`xmpToJpeg.py`**: standalone CLI tool for batch XMP→JPEG embedding
- All configuration in a single `config.yaml`

---

## Installation

> **Quick summary of what you need per mode:**
> - All modes: Python 3.10+, `pip install -r requirements.txt`
> - 🌍 Vacances: + Ollama running locally
> - 🌿🦊🐦🦋 Balade / Animaux / Oiseaux / Insectes: + PyTorch + open_clip_torch
> - 🔭 Astro: + PyTorch + astrometry + astropy + astroquery + sep + pyongc + Ollama
> - 📥 XMP→JPEG button: + pyexiv2 (+ system lib on macOS)

---

### 1 — Install Python 3.10+

<details>
<summary><strong>macOS</strong></summary>

**Option A — Homebrew (recommended)**
```bash
brew install python
```

**Option B — python.org**
Download and install from [python.org/downloads](https://www.python.org/downloads/).

Verify:
```bash
python3 --version   # must be 3.10 or higher
```
</details>

<details>
<summary><strong>Windows</strong></summary>

**Option A — winget**
```powershell
winget install Python.Python.3.13
```

**Option B — python.org**
Download the installer from [python.org/downloads](https://www.python.org/downloads/).
⚠️ During installation, check **"Add Python to PATH"**.

Verify in PowerShell:
```powershell
python --version   # must be 3.10 or higher
```
</details>

<details>
<summary><strong>Linux</strong></summary>

**Debian / Ubuntu**
```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv
```

**Fedora / RHEL**
```bash
sudo dnf install python3 python3-pip
```

Verify:
```bash
python3 --version   # must be 3.10 or higher
```
</details>

---

### 2 — Clone the repository

```bash
git clone https://github.com/LaurentVoillot/photo-folder-tagger.git
cd photo-folder-tagger
```

> **No git?**
> macOS: `brew install git`
> Windows: [git-scm.com](https://git-scm.com/download/win) or `winget install Git.Git`
> Linux: `sudo apt install git` / `sudo dnf install git`

---

### 3 — Install core Python dependencies

```bash
pip install -r requirements.txt
```

For the 🔭 **Astro** mode, install the additional packages separately:

```bash
pip install -r requirements-astro.txt
```

> Without `requirements-astro.txt`, the Astro button is **greyed out** in the interface. All other modes work normally.

---

### 4 — Mode-specific setup

#### 🌍 Vacances — Ollama LLM

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install ollama
ollama serve &          # start server in background
ollama pull qwen2.5vl:7b
```
</details>

<details>
<summary><strong>Windows</strong></summary>

Download and install Ollama from [ollama.ai/download](https://ollama.ai/download).
Ollama starts automatically as a background service.

```powershell
ollama pull qwen2.5vl:7b
```
</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &          # start server in background
ollama pull qwen2.5vl:7b
```
</details>

**Recommended models by hardware:**

| Hardware | Model | VRAM |
|----------|-------|------|
| Apple M1/M2/M3/M4 | `qwen2.5vl:7b` | unified memory |
| GPU 8 GB | `qwen2.5vl:7b` | ~6 GB |
| GPU 16+ GB | `qwen2.5vl:72b` | ~45 GB |
| CPU only | `qwen2.5vl:7b` | RAM (slow) |

---

#### 🌿 Balade · 🦊 Animaux · 🐦 Oiseaux · 🦋 Insectes — PyTorch + CLIP

<details>
<summary><strong>macOS — Apple Silicon (M1/M2/M3/M4) — GPU via MPS</strong></summary>

```bash
pip install torch torchvision open_clip_torch
```
PyTorch detects Apple Silicon automatically and uses MPS (Metal Performance Shaders).
</details>

<details>
<summary><strong>macOS — Intel</strong></summary>

```bash
pip install torch torchvision open_clip_torch
```
Runs on CPU. Slower but functional.
</details>

<details>
<summary><strong>Windows — NVIDIA GPU (CUDA)</strong></summary>

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch
```
Replace `cu121` with your CUDA version (`cu118`, `cu124`…).
Check your version: `nvidia-smi`.
</details>

<details>
<summary><strong>Windows — CPU only</strong></summary>

```powershell
pip install torch torchvision open_clip_torch
```
</details>

<details>
<summary><strong>Linux — NVIDIA GPU (CUDA)</strong></summary>

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch
```
</details>

<details>
<summary><strong>Linux — CPU only</strong></summary>

```bash
pip install torch torchvision open_clip_torch
```
</details>

> **BioCLIP 2** (`imageomics/bioclip-2`) is downloaded automatically from HuggingFace Hub on first launch of **Oiseaux** or **Insectes** mode (~1.7 GB). Internet required only on first run.

---

#### 🔭 Astro — Plate solving + SIMBAD + OpenNGC + Ollama

Requires all of the above (Ollama + PyTorch), plus:

```bash
pip install astrometry sep photutils astropy astroquery pyongc
```

| Package | Role |
|---------|------|
| `astrometry` | Offline plate solving (identifies star patterns) |
| `sep` | Fast star detection (C extension) |
| `photutils` | Star detection fallback (pure Python) |
| `astropy` | WCS coordinates, sky coordinate conversion |
| `astroquery` | SIMBAD cone search (requires internet) |
| `pyongc` | NGC/IC/Messier catalogue offline fallback |

> **Note on astrometry index files**: on first use, the `astrometry` package automatically downloads index files (~5 GB, Tycho-2+Gaia series 5200) into `~/.cache/astrometry/`. Internet required only on first run. Subsequent runs are fully offline.

<details>
<summary><strong>Windows — astrometry known issue</strong></summary>

The `astrometry` package may not have a Windows wheel for all Python versions. If installation fails:
```powershell
pip install astrometry --no-build-isolation
```
If it still fails, the Astro mode will fall back to **Ollama only** (no plate solving), which still works for nebulae and planets.
</details>

---

#### 📥 XMP → JPEG — Embed tags into JPEG files

This feature embeds XMP tags directly into JPEG files so **Lightroom can read them without a sidecar**.

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install inih          # required C library (libINIReader)
pip install pyexiv2
```

> Without `brew install inih`, pyexiv2 will fail to load (`OSError: libINIReader.0.dylib not found`). The application will still start, but the `📥 XMP → JPEG` button will be greyed out.
</details>

<details>
<summary><strong>Windows</strong></summary>

```powershell
pip install pyexiv2
```
The Windows wheel bundles all required DLLs — no extra step needed.
</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
pip install pyexiv2
```
The Linux wheel bundles all required `.so` files — no extra step needed.
</details>

---

### 5 — Run the application

```bash
python photo_folder_tagger.py
```

> **Windows note**: use `python` instead of `python3`.

---

## Usage

### Main application

1. **Select a folder** — drag & drop it onto the input field, or click 📂
2. **Choose a mode** — click one of the six mode buttons (two rows)
3. **Options** — tick *Inclure les sous-dossiers* and/or *Ignorer les photos déjà taguées*
4. **Start** — click **▶ Démarrer**
5. Follow progress in the log panel (relative paths shown, colour-coded)
6. If interrupted, click **↩ Reprendre session** at next launch to continue

### Test mode (dry run)

Click **🔍 Test (sans écriture)** to preview generated tags without writing any XMP.

### XMP → JPEG (Lightroom embedding)

Click **📥 XMP → JPEG** to read all `.xmp` sidecars in the selected folder and embed their tags directly into the corresponding JPEG files (`XMP:Subject` + `IPTC:Keywords`). This is required for Lightroom Classic, which ignores `.xmp` sidecars for JPEG files.

Tags already present in the JPEG are preserved and merged (no duplicates).

---

## `xmpToJpeg.py` — standalone CLI tool

For batch processing outside the GUI:

```bash
# Single folder
python xmpToJpeg.py /path/to/photos

# With subfolders
python xmpToJpeg.py /path/to/photos --recursive

# Simulate (no file written)
python xmpToJpeg.py /path/to/photos --recursive --dry-run
```

Example output:
```
── xmpToJpeg ─────────────────────────────────────────────
  Dossier  : /Users/laurent/Photos/Bretagne
  Mode     : ÉCRITURE
  Scan     : récursif
──────────────────────────────────────────────────────────

  ✦  ./2024/IMG_0042.jpg  +5 tag(s) : Paris_ai, Tour Eiffel_ai, ...
  ✓  ./2024/IMG_0043.jpg  (déjà à jour)
  –  ./2024/IMG_0044.jpg  (XMP sans tags)

── Résumé ────────────────────────────────────────────────
  JPEG scannés       : 47
  Avec sidecar XMP   : 32
  Mis à jour         : 28
  Déjà à jour        : 3
  XMP sans tags      : 1
  Erreurs            : 0
──────────────────────────────────────────────────────────
```

---

## Configuration (`config.yaml`)

The only file you need to edit for tuning models, prompts, and performance.

### Key sections

```yaml
mode: vacances           # Active mode: vacances | balade | animaux | astro | oiseaux | insectes

model:                   # Vacances mode — Ollama LLM
  name: qwen2.5vl:7b
  temperature: 0.1
  max_tokens: 80

performance:
  max_image_size: 512    # Resize before AI analysis (smaller = faster)
  jpeg_quality: 60
  concurrent_workers: 1  # Keep at 1 for Ollama (single GPU thread)

xmp:
  tag_suffix: "_ai"      # Appended to every AI-generated tag
  create_if_missing: true
  merge_with_existing: true

images:
  skip_already_tagged: false  # Skip photos with any existing tags
  recursive: true             # Include subfolders

clip:                    # Balade mode
  model: ViT-B-16
  pretrained: openai
  top_k: 8

bioclip:                 # Animaux mode
  confidence_threshold: 0.26
  top_k: 3
  ollama_fallback: true

birds:                   # Oiseaux mode
  confidence_threshold: 0.24
  top_k: 2
  ollama_fallback: true

insects:                 # Insectes mode
  confidence_threshold: 0.22
  top_k: 2
  ollama_fallback: true

astro:                   # Astro mode
  use_plate_solving: true
  use_simbad: true
  simbad_radius_deg: 1.0
  use_ngc_catalog: true
  ngc_search_radius_deg: 5.0
  ollama_fallback: true
  sensor_width_mm: 36.0   # Full-frame default — adjust for your camera
  sensor_height_mm: 24.0
```

---

## Software compatibility

### XMP sidecars (`.xmp` — for RAW files)

| Software | XMP read | Notes |
|----------|----------|-------|
| Lightroom Classic | ✅ | *Photo > Read Metadata from File* to reload |
| Adobe Bridge | ✅ | Immediate |
| Capture One | ✅ | *Sync Metadata* |
| Darktable | ✅ | Read on import |
| DigiKam | ✅ | Enable sync in settings |

### Embedded tags (JPEG — via `📥 XMP → JPEG`)

| Software | Reads embedded XMP | Notes |
|----------|--------------------|-------|
| Lightroom Classic | ✅ | Required for JPEG — sidecars are ignored |
| Adobe Bridge | ✅ | Also reads IPTC:Keywords |
| Capture One | ✅ | |
| Darktable | ✅ | |
| macOS Photos | ✅ | Reads IPTC:Keywords |
| Windows Explorer | ✅ | Shows IPTC tags in file details |

---

## Project structure

```
photo-folder-tagger/
├── config.yaml               ← Only file you need to edit
├── photo_folder_tagger.py    ← Main application (PyQt6 GUI)
├── xmpToJpeg.py              ← Standalone CLI: embed XMP tags into JPEG
├── ollama_client.py          ← Ollama API client (Vacances mode)
├── clip_client.py            ← CLIP local client (Balade mode)
├── bioclip_client.py         ← BioCLIP + CLIP client (Animaux mode)
├── astro_client.py           ← Plate solving + SIMBAD + OpenNGC (Astro mode)
├── birds_client.py           ← BioCLIP 2 — birds (Oiseaux mode)
├── insects_client.py         ← BioCLIP 2 — insects (Insectes mode)
├── xmp_manager.py            ← XMP sidecar read/write/merge
├── folder_scanner.py         ← Folder scan + session state
├── settings_dialog.py        ← Settings window (PyQt6)
├── fix_double_suffix.py      ← Repair utility for _ai_ai tags
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### Application won't start

```bash
# Missing core dependency
pip install PyQt6 Pillow rawpy requests lxml PyYAML
```

### 🌍 Ollama unreachable

```bash
ollama serve              # start the server
curl http://localhost:11434/api/tags   # verify it responds
```

### 🌍 Model not found

```bash
ollama list               # list installed models
ollama pull qwen2.5vl:7b  # install the model
```

### 🌿🦊🐦🦋 CLIP / BioCLIP not available

```bash
pip install torch open_clip_torch
```

### 🐦🦋 BioCLIP 2 not loading (first launch)

BioCLIP 2 (`imageomics/bioclip-2`, ~1.7 GB) downloads automatically on first use of **Oiseaux** or **Insectes** mode. Requires internet access. Subsequent runs use the local HuggingFace cache (`~/.cache/huggingface/`).

### 🔭 Astro — plate solving index files missing (first launch)

Index files (~5 GB) download automatically on first use. Requires internet access. Stored in `~/.cache/astrometry/`. This is a one-time download.

### 🔭 Astro — NGC identification not working

```bash
pip install pyongc
```
Then set `use_ngc_catalog: true` in `config.yaml`.
NGC identification requires RA/Dec coordinates in the photo EXIF (from a goto mount or acquisition software like N.I.N.A., SGP, Siril…).

### 📥 XMP → JPEG — button greyed out

**macOS**: `libINIReader.dylib` is missing:
```bash
brew install inih
pip install --force-reinstall pyexiv2
```

**Windows / Linux**: install pyexiv2:
```bash
pip install pyexiv2
```

### ⚠️ Tags written as `_ai_ai` (after upgrading from an old version)

```bash
python fix_double_suffix.py /path/to/photos --dry-run   # preview
python fix_double_suffix.py /path/to/photos             # fix
```

### 🐢 Processing too slow (Vacances mode)

- Set `max_image_size: 512` in `config.yaml` (default — gives ×6 speed-up vs 1024 px on Apple Silicon)
- Use a smaller model: `ollama pull qwen2.5vl:3b`

### Lightroom doesn't show new tags (XMP sidecars)

Select photos → *Photo > Read Metadata from File*
Or use **📥 XMP → JPEG** to embed tags directly into the JPEG — Lightroom then reads them automatically on import.

---

## Licence

GPL v3.0 — see [LICENSE](LICENSE)
