# Photo Folder Tagger — Documentation française

Taguage automatique de photos par IA locale avec écriture dans des fichiers XMP sidecar.
Fonctionne avec **n'importe quel dossier photos**, sans dépendance à Lightroom.
Compatible avec **Lightroom Classic, Bridge, Capture One, Darktable, DigiKam** et tout logiciel lisant les XMP sidecar.

> 📖 [English documentation](README.md)

---

## Trois modes de taguage

| Mode | Moteur | Vitesse | Idéal pour |
|------|--------|---------|-----------|
| 🌍 **Vacances** | Ollama LLM (qwen2.5vl) | ~1,8 s/photo | Contexte culturel, noms de lieux, voyages |
| 🌿 **Balade** | CLIP ViT-B-16 (local) | ~46 ms/photo | Sorties nature, paysages, mots-clés courants |
| 🦊 **Animaux** | BioCLIP + CLIP (local) | ~40 ms/photo | Identification d'espèces et sous-espèces |
| 🔭 **Astro** | CLIP ViT-L-14 + EXIF + OpenNGC | ~50 ms/photo | Nébuleuses, galaxies, planètes, FOV, objets Messier/NGC |

Les trois modes écrivent les tags dans des **fichiers XMP sidecar** placés à côté de vos photos.
Les tags générés par l'IA sont marqués d'un suffixe configurable (`_ai` par défaut).

---

## Fonctionnalités

- **Glisser-déposer** d'un dossier sur le champ de saisie (ou cliquer sur 📂)
- Trois moteurs IA sélectionnables depuis l'écran principal
- Lecture, fusion et déduplication des tags XMP existants — aucun écrasement
- Déduplication intelligente du suffixe : `forêt` et `forêt_ai` sont considérés comme identiques
- Pause / reprise de session entre deux lancements
- Interface PyQt6 sombre avec journal coloré
- Mode test (dry run) : prévisualise les tags sans écrire aucun XMP
- Toute la configuration dans un seul fichier `config.yaml`

---

## Prérequis

- Python 3.10 ou supérieur
- [Ollama](https://ollama.ai) (mode Vacances uniquement)
- PyTorch + open_clip (modes Balade, Animaux et Astro — Apple Silicon MPS supporté)
- `opennugc` (optionnel, pour le catalogue Messier/NGC en mode Astro)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/LaurentVoillot/photo-folder-tagger.git
cd photo-folder-tagger
```

### 2. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### 3. (Mode Vacances uniquement) Installer Ollama et télécharger le modèle

```bash
# Installer Ollama : https://ollama.ai/download
ollama pull qwen2.5vl:7b
```

### 4. Lancer l'application

```bash
python photo_folder_tagger.py
```

---

## Utilisation

1. **Sélectionner un dossier** — glissez-le sur le champ, ou cliquez sur 📂
2. **Choisir un mode** — Vacances / Balade / Animaux
3. **Démarrer** — cliquer sur ▶ Démarrer
4. Suivre la progression dans le journal
5. En cas d'interruption, cliquer sur **Reprendre session** au prochain lancement

### Mode test

Cliquez sur **🔍 Test (sans écriture)** pour prévisualiser les tags générés sans modifier aucun XMP.

---

## Configuration (`config.yaml`)

C'est le seul fichier à modifier pour changer de modèle, de prompt ou de performances.

### Sections principales

```yaml
mode: vacances           # Mode actif : vacances | balade | animaux

model:                   # Mode Vacances — Ollama LLM
  name: qwen2.5vl:7b
  temperature: 0.1
  max_tokens: 80

performance:
  max_image_size: 512    # Taille max image (pixels) — plus petit = prefill GPU plus rapide
  jpeg_quality: 60
  concurrent_workers: 1  # Laisser à 1 — Ollama utilise un seul thread GPU

xmp:
  tag_suffix: "_ai"      # Suffixe ajouté à chaque tag IA
  create_if_missing: true
  merge_with_existing: true

clip:                    # Mode Balade
  model: ViT-B-16
  pretrained: openai
  top_k: 8
  vocabulary:            # Liste des tags candidats (zero-shot)
    - rose
    - forêt
    - montagne
    # ...

bioclip:                 # Mode Animaux
  confidence_threshold: 0.26
  top_k: 3
  use_context_tags: true
  context_top_k: 5
  ollama_fallback: true  # Bascule sur Ollama si score < seuil
```

### Modèles recommandés (mode Vacances)

| Machine | Modèle | VRAM |
|---------|--------|------|
| Mac M1/M2/M3 | `qwen2.5vl:7b` | mémoire unifiée |
| GPU 8 Go | `qwen2.5vl:7b` | ~6 Go |
| GPU 16+ Go | `qwen2.5vl:72b` | ~45 Go |

### Pourquoi `max_image_size: 512` ?

Sur Apple Silicon (M1 Max), réduire la taille de 1024 px à 512 px fait passer le nombre de tokens visuels de 1 436 à 391, ce qui réduit le temps de prefill GPU de 13 s à 0,75 s — **gain ×6** sans perte perceptible de qualité des tags.

---

## Structure des fichiers générés

Pour chaque `IMG_1234.jpg`, le fichier `IMG_1234.xmp` est créé ou complété dans le même dossier :

```xml
<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="...">
    <rdf:Description ...>
      <dc:subject>
        <rdf:Bag>
          <rdf:li>forêt_ai</rdf:li>
          <rdf:li>automne_ai</rdf:li>
          <rdf:li>champignon_ai</rdf:li>
        </rdf:Bag>
      </dc:subject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
```

---

## Compatibilité logiciels

| Logiciel | Import XMP | Notes |
|----------|-----------|-------|
| Lightroom Classic | Automatique | `Photo > Lire les métadonnées depuis le fichier` pour recharger |
| Adobe Bridge | Automatique | Lecture immédiate |
| Capture One | Automatique | Via *Sync Metadata* |
| Darktable | Automatique | Lecture à l'import |
| DigiKam | Automatique | Activer la synchronisation dans les paramètres |

---

## Architecture du projet

```
photo-folder-tagger/
├── config.yaml               ← Seul fichier à modifier
├── photo_folder_tagger.py    ← Application principale (interface PyQt6)
├── ollama_client.py          ← Client API Ollama (mode Vacances)
├── clip_client.py            ← Client CLIP local (mode Balade)
├── bioclip_client.py         ← Client BioCLIP + CLIP (mode Animaux)
├── astro_client.py           ← Client CLIP astro + FOV EXIF + OpenNGC (mode Astro)
├── xmp_manager.py            ← Lecture/écriture XMP sidecar
├── folder_scanner.py         ← Scan de dossiers + état de session
├── fix_double_suffix.py      ← Utilitaire de réparation des tags _ai_ai
├── requirements.txt
├── README.md                 ← Documentation anglaise
└── README.fr.md              ← Ce fichier
```

---

## Dépannage

**Ollama inaccessible**
```bash
ollama serve
curl http://localhost:11434/api/tags
```

**Modèle introuvable**
```bash
ollama list
ollama pull qwen2.5vl:7b
```

**CLIP / BioCLIP / CLIP Astro indisponible**
```bash
pip install torch open_clip_torch
```

**Identification NGC désactivée (mode Astro)**
```bash
pip install opennugc
```
Puis `use_ngc_catalog: true` dans `config.yaml`. L'identification fonctionne uniquement si les coordonnées RA/Dec sont présentes dans les EXIF (nécessite une monture goto ou un logiciel d'acquisition comme N.I.N.A. ou SGP).

**FOV non calculé (mode Astro)**
Vérifiez que la focale est bien enregistrée dans les EXIF (`FocalLength` ou `FocalLengthIn35mmFilm`). Adaptez aussi `sensor_width_mm` et `sensor_height_mm` dans `config.yaml` à votre boîtier.

**XMP non lu par Lightroom**
Sélectionner les photos → `Photo > Lire les métadonnées depuis le fichier`

**Tags écrits en `_ai_ai`** (après mise à jour depuis une ancienne version)
```bash
python fix_double_suffix.py /chemin/dossier --dry-run   # prévisualiser
python fix_double_suffix.py /chemin/dossier             # corriger
```

**Traitement trop lent (mode Vacances)**
- Vérifier que `max_image_size: 512` est bien dans `config.yaml`
- Choisir un modèle plus léger

**Premier traitement CLIP/BioCLIP lent**
Normal : la compilation des shaders Metal (MPS) se fait au premier appel (~1,5 s). Les suivants sont rapides (~40-46 ms).

---

## Licence

GPL v3.0 — voir [LICENSE](LICENSE)
