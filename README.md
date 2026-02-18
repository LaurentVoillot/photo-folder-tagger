# Photo Folder Tagger

Taguage automatique de photos par IA locale (Ollama) avec écriture dans des fichiers XMP sidecar.

Fonctionne avec **n'importe quel dossier photos**, sans dépendance à Lightroom.
Compatible avec **Lightroom Classic, Bridge, Capture One, Darktable, DigiKam** et tout logiciel lisant les XMP sidecar.

---

## Fonctionnalités

- Analyse les photos d'un dossier (et sous-dossiers) via un modèle vision Ollama local
- **Complète les XMP existants** ou en crée de nouveaux
- Fusionne les tags IA avec les tags existants (pas d'écrasement)
- Suffixe configurable sur les tags IA (`_ai` par défaut)
- Pause / reprise de session entre deux lancements
- Traitement concurrent (plusieurs photos simultanément)
- Interface graphique Tkinter simple et claire
- **Toute la configuration dans un seul fichier** `config.yaml`

---

## Installation

### Prérequis

- Python 3.10 ou supérieur
- [Ollama](https://ollama.ai) installé et en cours d'exécution

### 1. Installer Ollama et télécharger un modèle vision

```bash
# Installation d'Ollama : https://ollama.ai/download
# Puis télécharger un modèle vision, par exemple :
ollama pull llava:13b
# ou pour une machine moins puissante :
ollama pull llava:7b
# ou ultra-léger (fonctionne sur CPU) :
ollama pull moondream
```

### 2. Cloner le projet

```bash
git clone https://github.com/LaurentVoillot/photo-folder-tagger.git
cd photo-folder-tagger
```

### 3. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### 4. Configurer

Ouvrez `config.yaml` et adaptez au minimum :

```yaml
model:
  name: "llava:13b"   # Nom du modèle installé

performance:
  concurrent_workers: 2  # Selon la puissance de votre machine
```

---

## Utilisation

### Lancer l'application

```bash
python photo_folder_tagger.py
```

### Workflow

1. Cliquer sur **Parcourir...** et sélectionner votre dossier photos
2. Cocher/décocher les options (sous-dossiers, ignorer déjà taguées)
3. Cliquer sur **Démarrer**
4. Suivre la progression dans le journal
5. En cas d'interruption, cliquer sur **Arrêter** — l'état est sauvegardé
6. Au prochain lancement, cliquer sur **Reprendre session**

---

## Configuration (`config.yaml`)

Ce fichier est le seul à modifier pour changer de machine ou de modèle LLM.

### Sections principales

#### Serveur Ollama

```yaml
ollama:
  url: "http://localhost:11434"
  timeout: 300
  retry_attempts: 2
```

#### Modèle LLM

```yaml
model:
  name: "llava:13b"     # Modèle vision à utiliser
  temperature: 0.1       # 0 = déterministe, 1 = créatif
  max_tokens: 150
```

Modèles recommandés selon la machine :

| Machine | Modèle | VRAM requise |
|---------|--------|-------------|
| CPU only | `moondream` | 0 (RAM uniquement) |
| GPU 4-6 Go | `llava:7b` | ~5 Go |
| GPU 8-12 Go | `llava:13b` | ~8 Go |
| GPU 16+ Go | `llava:34b` | ~20 Go |

#### Performance

```yaml
performance:
  concurrent_workers: 2   # Photos traitées en parallèle
  request_delay: 0.0       # Délai entre requêtes (secondes)
  max_image_size: 1024     # Taille max image envoyée au modèle
  jpeg_quality: 70         # Qualité JPEG pour l'encodage
```

Recommandations `concurrent_workers` :

| Configuration | Workers |
|---------------|---------|
| CPU only / 16 Go RAM | 1 |
| GPU 6-8 Go VRAM | 2-3 |
| GPU 12-16 Go VRAM | 4-6 |
| GPU 24+ Go VRAM | 6-8 |

#### Prompt

```yaml
prompt:
  language: "français"
  max_tags: 15
  auto_prompt: |
    Analyse cette photo et génère exactement {max_tags} mots-clés
    descriptifs en {language}.
    ...
```

Vous pouvez réécrire entièrement le prompt pour adapter la granularité,
le style des tags ou la langue.

#### XMP

```yaml
xmp:
  tag_suffix: "_ai"          # Suffixe ajouté aux tags IA
  create_if_missing: true    # Créer le XMP s'il n'existe pas
  merge_with_existing: true  # Fusionner avec les tags existants
```

---

## Structure des fichiers générés

Pour chaque photo `IMG_1234.jpg`, le fichier `IMG_1234.xmp` est créé ou
complété dans le même dossier :

```xml
<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="...">
    <rdf:Description ...>
      <dc:subject>
        <rdf:Bag>
          <rdf:li>paysage_ai</rdf:li>
          <rdf:li>montagne_ai</rdf:li>
          <rdf:li>coucher de soleil_ai</rdf:li>
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
| Lightroom Classic | Automatique | Recharger les métadonnées depuis le fichier |
| Adobe Bridge | Automatique | Lecture immédiate |
| Capture One | Automatique | Importer via "Sync Metadata" |
| Darktable | Automatique | Lire le XMP au moment de l'import |
| DigiKam | Automatique | Synchronisation dans les paramètres |

---

## Architecture du projet

```
photo-folder-tagger/
├── config.yaml               ← Seul fichier à modifier
├── photo_folder_tagger.py    ← Application principale (GUI)
├── ollama_client.py          ← Client API Ollama
├── xmp_manager.py            ← Lecture/écriture XMP
├── folder_scanner.py         ← Scan de dossiers + état de session
├── requirements.txt
└── README.md
```

---

## Dépannage

**Ollama inaccessible**
```bash
# Vérifier qu'Ollama tourne
ollama serve
# Ou vérifier le statut
curl http://localhost:11434/api/tags
```

**Modèle indisponible**
```bash
# Lister les modèles installés
ollama list
# Télécharger le modèle configuré
ollama pull llava:13b
```

**XMP non lu par Lightroom**
- Dans Lightroom : `Photo > Lire les métadonnées depuis le fichier`
- Ou en masse : sélectionner toutes les photos, puis `Métadonnées > Lire les métadonnées...`

**Traitement trop lent**
- Réduire `max_image_size` à `512` ou `768` dans `config.yaml`
- Réduire `max_tokens` à `100`
- Choisir un modèle plus léger (`llava:7b`, `moondream`)

**Mémoire GPU insuffisante**
- Passer à `concurrent_workers: 1`
- Choisir un modèle plus petit
- Ajouter un `request_delay: 1.0`

---

## Licence

GPL v3.0 — voir [LICENSE](LICENSE)
