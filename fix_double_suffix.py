#!/usr/bin/env python3
"""
Script de correction des tags doublement suffixés (_ai_ai → _ai).
Usage : python fix_double_suffix.py /chemin/vers/dossier [--dry-run]
"""

import sys
import argparse
from pathlib import Path
from lxml import etree

NAMESPACES = {
    "x":   "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc":  "http://purl.org/dc/elements/1.1/",
}

SUFFIX = "_ai"
DOUBLE = "_ai_ai"


def fix_xmp(xmp_path: Path, dry_run: bool) -> bool:
    """Corrige les tags doublement suffixés dans un fichier XMP."""
    try:
        parser = etree.XMLParser(recover=True, encoding="UTF-8")
        tree = etree.parse(str(xmp_path), parser)
        root = tree.getroot()

        dc_ns = NAMESPACES["dc"]
        rdf_ns = NAMESPACES["rdf"]

        changed = False
        for li in root.iter(f"{{{rdf_ns}}}li"):
            if li.text and li.text.strip().endswith(DOUBLE):
                old = li.text.strip()
                new = old[: -len(SUFFIX)]  # retire le dernier _ai
                print(f"  {old!r:40s} → {new!r}")
                if not dry_run:
                    li.text = new
                changed = True

        if changed and not dry_run:
            # Réécriture propre
            content = etree.tostring(root, encoding="unicode", pretty_print=True)
            header = '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            footer = '\n<?xpacket end="w"?>'
            xmp_path.write_text(header + content + footer, encoding="UTF-8")

        return changed

    except Exception as e:
        print(f"  ERREUR : {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Corrige les tags _ai_ai → _ai dans les XMP")
    parser.add_argument("folder", help="Dossier contenant les fichiers XMP")
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans modifier")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Dossier introuvable : {folder}")
        sys.exit(1)

    xmp_files = list(folder.rglob("*.xmp"))
    print(f"Scan de {len(xmp_files)} fichiers XMP dans {folder}")
    if args.dry_run:
        print("Mode DRY-RUN — aucune modification ne sera effectuée\n")

    fixed = 0
    for xmp_path in sorted(xmp_files):
        changed = fix_xmp(xmp_path, args.dry_run)
        if changed:
            print(f"{'[DRY]' if args.dry_run else '[OK] '} {xmp_path.name}")
            fixed += 1

    print(f"\n{'Seraient corrigés' if args.dry_run else 'Corrigés'} : {fixed} fichier(s)")


if __name__ == "__main__":
    main()
