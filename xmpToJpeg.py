#!/usr/bin/env python3
"""
xmpToJpeg — Embed XMP sidecar tags into JPEG files.

Reads dc:subject tags from .xmp sidecars and writes them directly
into the corresponding JPEG (XMP:Subject + IPTC:Keywords), so
Lightroom and other DAM tools can read them without the sidecar.

Requirements:
    pip install pyexiv2 lxml

Usage:
    python xmpToJpeg.py /path/to/folder
    python xmpToJpeg.py /path/to/folder --recursive
    python xmpToJpeg.py /path/to/folder --dry-run
    python xmpToJpeg.py /path/to/folder --recursive --dry-run
"""

import argparse
import sys
from pathlib import Path

try:
    import pyexiv2
except ImportError:
    print("❌  pyexiv2 n'est pas installé. Lancez : pip install pyexiv2")
    sys.exit(1)

try:
    from lxml import etree
except ImportError:
    print("❌  lxml n'est pas installé. Lancez : pip install lxml")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

JPEG_EXTENSIONS = {".jpg", ".jpeg"}

# Namespaces XMP/RDF
DC_NS  = "http://purl.org/dc/elements/1.1/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


# ---------------------------------------------------------------------------
# Lecture du sidecar XMP
# ---------------------------------------------------------------------------

def read_tags_from_xmp(xmp_path: Path) -> list[str]:
    """
    Extrait les tags dc:subject d'un sidecar XMP.

    Returns:
        Liste de tags (vide si aucun ou erreur).
    """
    try:
        parser = etree.XMLParser(recover=True, encoding="UTF-8")
        tree = etree.parse(str(xmp_path), parser)
        root = tree.getroot()

        tags: list[str] = []
        for subject in root.iter(f"{{{DC_NS}}}subject"):
            # rdf:Bag (non ordonné) ou rdf:Seq (ordonné)
            for container in (*subject.iter(f"{{{RDF_NS}}}Bag"),
                              *subject.iter(f"{{{RDF_NS}}}Seq")):
                for li in container:
                    text = (li.text or "").strip()
                    if text:
                        tags.append(text)
        return tags

    except Exception as e:
        print(f"  ⚠  Erreur lecture XMP {xmp_path.name} : {e}")
        return []


# ---------------------------------------------------------------------------
# Écriture dans le JPEG
# ---------------------------------------------------------------------------

def embed_tags_in_jpeg(jpeg_path: Path, new_tags: list[str], dry_run: bool) -> bool:
    """
    Fusionne les tags dans le JPEG (XMP:Subject + IPTC:Keywords).

    - Les tags déjà présents dans le JPEG sont préservés.
    - Seuls les tags manquants sont ajoutés.
    - En mode dry-run, aucune écriture n'est effectuée.

    Returns:
        True si succès (ou dry-run), False si erreur.
    """
    try:
        img = pyexiv2.Image(str(jpeg_path))

        # ── Lecture des tags déjà embarqués ───────────────────────────────
        existing_xmp  = img.read_xmp()
        existing_iptc = img.read_iptc()

        current_xmp_tags: list[str] = existing_xmp.get("Xmp.dc.subject", [])
        if isinstance(current_xmp_tags, str):
            current_xmp_tags = [current_xmp_tags]

        # ── Fusion (union) sans doublon ────────────────────────────────────
        merged = list(current_xmp_tags)
        seen   = {t.lower() for t in current_xmp_tags}
        added  = []
        for tag in new_tags:
            if tag.lower() not in seen:
                merged.append(tag)
                seen.add(tag.lower())
                added.append(tag)

        img.close()

        if not added:
            return None  # sentinel : rien à faire, déjà à jour

        # ── Écriture (XMP embarqué + IPTC legacy) ─────────────────────────
        if not dry_run:
            img = pyexiv2.Image(str(jpeg_path))
            img.modify_xmp({"Xmp.dc.subject": merged})
            # IPTC Keywords : limité à 64 caractères par mot-clé
            img.modify_iptc({"Iptc.Application2.Keywords": [t[:64] for t in merged]})
            img.close()

        return added   # liste des tags effectivement ajoutés

    except Exception as e:
        print(f"  ✗  Erreur écriture dans {jpeg_path.name} : {e}")
        return False


# ---------------------------------------------------------------------------
# Traitement d'un dossier
# ---------------------------------------------------------------------------

def process_folder(folder: Path, recursive: bool, dry_run: bool) -> dict:
    """
    Parcourt le dossier, trouve les paires JPEG+XMP, embarque les tags.

    Returns:
        Dictionnaire de statistiques.
    """
    stats = {
        "scanned":  0,   # JPEG trouvés
        "paired":   0,   # JPEG avec sidecar XMP
        "updated":  0,   # JPEG effectivement mis à jour
        "skipped":  0,   # déjà à jour
        "no_tags":  0,   # XMP sans tags
        "errors":   0,
    }

    pattern = "**/*" if recursive else "*"
    jpeg_files = sorted(
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in JPEG_EXTENSIONS
        and not p.name.startswith(".")
    )

    for jpeg_path in jpeg_files:
        stats["scanned"] += 1

        # Le sidecar porte le même nom avec l'extension .xmp
        xmp_path = jpeg_path.with_suffix(".xmp")
        if not xmp_path.exists():
            continue  # pas de sidecar → on ignore silencieusement

        stats["paired"] += 1
        rel = jpeg_path.relative_to(folder)

        tags = read_tags_from_xmp(xmp_path)
        if not tags:
            print(f"  –  ./{rel}  (XMP sans tags)")
            stats["no_tags"] += 1
            continue

        result = embed_tags_in_jpeg(jpeg_path, tags, dry_run)

        if result is False:
            # Erreur déjà affichée dans embed_tags_in_jpeg
            stats["errors"] += 1
        elif result is None:
            # Déjà à jour
            print(f"  ✓  ./{rel}  (déjà à jour)")
            stats["skipped"] += 1
        else:
            # result = liste des tags ajoutés
            prefix = "[DRY-RUN] " if dry_run else ""
            print(f"  ✦  {prefix}./{rel}  +{len(result)} tag(s) : {', '.join(result)}")
            stats["updated"] += 1

    return stats


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="xmpToJpeg",
        description=(
            "Embarque les tags des sidecars .xmp directement dans les JPEG\n"
            "afin que Lightroom (et autres) les lisent sans sidecar."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python xmpToJpeg.py ~/Photos/Bretagne\n"
            "  python xmpToJpeg.py ~/Photos --recursive\n"
            "  python xmpToJpeg.py ~/Photos --recursive --dry-run\n"
        ),
    )
    parser.add_argument(
        "folder",
        help="Dossier contenant les JPEG et leurs sidecars .xmp",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Inclure les sous-dossiers",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Simuler sans écrire dans les fichiers",
    )

    args = parser.parse_args()
    folder = Path(args.folder).expanduser().resolve()

    if not folder.exists():
        print(f"❌  Dossier introuvable : {folder}")
        sys.exit(1)
    if not folder.is_dir():
        print(f"❌  N'est pas un dossier : {folder}")
        sys.exit(1)

    # ── En-tête ────────────────────────────────────────────────────────────
    mode_label = "DRY-RUN (aucune écriture)" if args.dry_run else "ÉCRITURE"
    rec_label  = "récursif" if args.recursive else "dossier racine uniquement"
    print(f"\n── xmpToJpeg ─────────────────────────────────────────────")
    print(f"  Dossier  : {folder}")
    print(f"  Mode     : {mode_label}")
    print(f"  Scan     : {rec_label}")
    print(f"──────────────────────────────────────────────────────────\n")

    # ── Traitement ─────────────────────────────────────────────────────────
    stats = process_folder(folder, recursive=args.recursive, dry_run=args.dry_run)

    # ── Résumé ─────────────────────────────────────────────────────────────
    print(f"\n── Résumé ────────────────────────────────────────────────")
    print(f"  JPEG scannés       : {stats['scanned']}")
    print(f"  Avec sidecar XMP   : {stats['paired']}")
    print(f"  Mis à jour         : {stats['updated']}"
          + ("  (simulation)" if args.dry_run else ""))
    print(f"  Déjà à jour        : {stats['skipped']}")
    print(f"  XMP sans tags      : {stats['no_tags']}")
    print(f"  Erreurs            : {stats['errors']}")
    print(f"──────────────────────────────────────────────────────────\n")

    sys.exit(0 if stats["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
