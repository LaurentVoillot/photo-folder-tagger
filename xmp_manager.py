"""
Gestionnaire de fichiers XMP pour les métadonnées de photos.
Supporte la création et la mise à jour des sidecars XMP.
Compatible avec Lightroom, Bridge, Darktable, Capture One, DigiKam, etc.
"""

import logging
from pathlib import Path
from typing import Optional

from lxml import etree

logger = logging.getLogger(__name__)

# Namespaces XMP standards
NAMESPACES = {
    "x":    "adobe:ns:meta/",
    "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc":   "http://purl.org/dc/elements/1.1/",
    "xmp":  "http://ns.adobe.com/xap/1.0/",
    "lr":   "http://ns.adobe.com/lightroom/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
}

# Template XMP minimal valide
XMP_TEMPLATE = """\
<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 6.0">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:lr="http://ns.adobe.com/lightroom/1.0/">
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


class XMPManager:
    """Gestion des fichiers sidecar XMP : lecture, création, mise à jour."""

    def __init__(self, config: dict):
        """
        Initialise le gestionnaire XMP.

        Args:
            config: Dictionnaire de configuration (chargé depuis config.yaml)
        """
        xmp_cfg = config.get("xmp", {})
        self.tag_suffix = xmp_cfg.get("tag_suffix", "_ai")
        self.create_if_missing = xmp_cfg.get("create_if_missing", True)
        self.merge_with_existing = xmp_cfg.get("merge_with_existing", True)

        # Enregistrement des namespaces pour lxml
        for prefix, uri in NAMESPACES.items():
            try:
                etree.register_namespace(prefix, uri)
            except Exception:
                pass

    def get_xmp_path(self, image_path: Path) -> Path:
        """
        Retourne le chemin du fichier XMP associé à une image.
        Le XMP est placé dans le même dossier, avec le même nom de base.

        Args:
            image_path: Chemin vers le fichier image

        Returns:
            Chemin vers le fichier XMP correspondant
        """
        return image_path.with_suffix(".xmp")

    def read_tags(self, image_path: Path) -> list[str]:
        """
        Lit les tags existants dans le fichier XMP d'une image.

        Args:
            image_path: Chemin vers le fichier image (pas le XMP)

        Returns:
            Liste des tags existants (peut être vide)
        """
        xmp_path = self.get_xmp_path(image_path)

        if not xmp_path.exists():
            return []

        try:
            tree = etree.parse(str(xmp_path))
            root = tree.getroot()

            dc_ns = NAMESPACES["dc"]
            rdf_ns = NAMESPACES["rdf"]

            # Recherche dc:subject/rdf:Bag/rdf:li
            tags = []
            for subject in root.iter(f"{{{dc_ns}}}subject"):
                for bag in subject.iter(f"{{{rdf_ns}}}Bag"):
                    for li in bag.iter(f"{{{rdf_ns}}}li"):
                        if li.text and li.text.strip():
                            tags.append(li.text.strip())
                # Support aussi rdf:Seq (moins courant)
                for seq in subject.iter(f"{{{rdf_ns}}}Seq"):
                    for li in seq.iter(f"{{{rdf_ns}}}li"):
                        if li.text and li.text.strip():
                            tags.append(li.text.strip())

            logger.debug(f"Tags lus depuis {xmp_path.name}: {tags}")
            return tags

        except etree.XMLSyntaxError as e:
            logger.error(f"XMP malformé {xmp_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Erreur lecture XMP {xmp_path}: {e}")
            return []

    def write_tags(self, image_path: Path, new_tags: list[str]) -> bool:
        """
        Écrit des tags dans le fichier XMP d'une image.
        Complète le XMP existant ou en crée un nouveau.
        Applique le suffixe configuré aux tags IA.

        Args:
            image_path: Chemin vers le fichier image
            new_tags: Nouveaux tags à ajouter/écrire

        Returns:
            True si succès, False sinon
        """
        if not new_tags:
            logger.warning(f"Aucun tag à écrire pour {image_path.name}")
            return False

        # Application du suffixe aux tags IA
        suffixed_tags = [
            f"{tag}{self.tag_suffix}" if self.tag_suffix else tag
            for tag in new_tags
        ]

        xmp_path = self.get_xmp_path(image_path)

        # Chargement ou création du document XMP
        if xmp_path.exists():
            tree, root = self._load_xmp(xmp_path)
            if tree is None:
                logger.warning(f"XMP corrompu, recréation de {xmp_path.name}")
                tree, root = self._create_xmp_tree()
        else:
            if not self.create_if_missing:
                logger.info(f"XMP non créé (désactivé) pour {image_path.name}")
                return False
            tree, root = self._create_xmp_tree()

        # Fusion ou remplacement des tags
        if self.merge_with_existing:
            existing_tags = self.read_tags(image_path)
            # Union en préservant l'ordre et sans doublons (insensible à la casse)
            seen = {t.lower() for t in existing_tags}
            for tag in suffixed_tags:
                if tag.lower() not in seen:
                    existing_tags.append(tag)
                    seen.add(tag.lower())
            final_tags = existing_tags
        else:
            final_tags = suffixed_tags

        # Écriture dans le document XML
        success = self._set_tags_in_tree(root, final_tags)
        if not success:
            return False

        # Sauvegarde du fichier
        try:
            tree.write(
                str(xmp_path),
                xml_declaration=True,
                encoding="UTF-8",
                pretty_print=True,
            )
            # Ajout des instructions xpacket (requis par certains logiciels)
            self._wrap_xpacket(xmp_path)
            logger.info(f"XMP mis à jour: {xmp_path.name} ({len(final_tags)} tags)")
            return True

        except Exception as e:
            logger.error(f"Erreur écriture XMP {xmp_path}: {e}")
            return False

    def has_ai_tags(self, image_path: Path) -> bool:
        """
        Vérifie si une image a déjà des tags IA (avec le suffixe configuré).

        Args:
            image_path: Chemin vers le fichier image

        Returns:
            True si des tags IA existent déjà
        """
        if not self.tag_suffix:
            return False

        existing = self.read_tags(image_path)
        return any(tag.endswith(self.tag_suffix) for tag in existing)

    # -------------------------------------------------------------------------
    # Méthodes privées
    # -------------------------------------------------------------------------

    def _load_xmp(self, xmp_path: Path) -> tuple:
        """Charge un fichier XMP existant."""
        try:
            parser = etree.XMLParser(recover=True, encoding="UTF-8")
            tree = etree.parse(str(xmp_path), parser)
            return tree, tree.getroot()
        except Exception as e:
            logger.error(f"Impossible de charger {xmp_path}: {e}")
            return None, None

    def _create_xmp_tree(self) -> tuple:
        """Crée un arbre XML XMP vide à partir du template."""
        parser = etree.XMLParser(recover=True, encoding="UTF-8")
        root = etree.fromstring(XMP_TEMPLATE.encode("UTF-8"), parser)
        tree = etree.ElementTree(root)
        return tree, root

    def _get_or_create_description(self, root: etree._Element) -> Optional[etree._Element]:
        """Trouve ou crée l'élément rdf:Description principal."""
        rdf_ns = NAMESPACES["rdf"]

        # Recherche de rdf:RDF
        rdf_rdf = root.find(f"{{{rdf_ns}}}RDF")
        if rdf_rdf is None:
            # Chercher en descendant
            for elem in root.iter(f"{{{rdf_ns}}}RDF"):
                rdf_rdf = elem
                break

        if rdf_rdf is None:
            logger.error("Structure RDF introuvable dans le XMP")
            return None

        # Récupération ou création de rdf:Description
        desc = rdf_rdf.find(f"{{{rdf_ns}}}Description")
        if desc is None:
            desc = etree.SubElement(rdf_rdf, f"{{{rdf_ns}}}Description")
            desc.set(f"{{{rdf_ns}}}about", "")

        return desc

    def _set_tags_in_tree(self, root: etree._Element, tags: list[str]) -> bool:
        """
        Insère ou remplace les tags dc:subject dans l'arbre XML.

        Args:
            root: Racine de l'arbre XML
            tags: Liste de tags à insérer

        Returns:
            True si succès
        """
        try:
            dc_ns = NAMESPACES["dc"]
            rdf_ns = NAMESPACES["rdf"]

            desc = self._get_or_create_description(root)
            if desc is None:
                return False

            # Suppression de l'ancien dc:subject s'il existe
            for subject in desc.findall(f"{{{dc_ns}}}subject"):
                desc.remove(subject)

            # Création du nouveau dc:subject
            subject_elem = etree.SubElement(desc, f"{{{dc_ns}}}subject")
            bag = etree.SubElement(subject_elem, f"{{{rdf_ns}}}Bag")

            for tag in tags:
                li = etree.SubElement(bag, f"{{{rdf_ns}}}li")
                li.text = tag

            return True

        except Exception as e:
            logger.error(f"Erreur insertion tags dans XML: {e}")
            return False

    def _wrap_xpacket(self, xmp_path: Path) -> None:
        """
        Entoure le contenu XML avec les instructions xpacket Adobe.
        Nécessaire pour la compatibilité avec Lightroom et Bridge.
        """
        try:
            content = xmp_path.read_text(encoding="UTF-8")

            # Suppression de la déclaration XML standard si présente
            if content.startswith("<?xml"):
                end_decl = content.index("?>") + 2
                content = content[end_decl:].lstrip()

            # Ajout des wrappers xpacket
            header = '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            footer = '\n<?xpacket end="w"?>'

            # Éviter double wrapping
            if "<?xpacket" not in content:
                xmp_path.write_text(header + content + footer, encoding="UTF-8")

        except Exception as e:
            logger.warning(f"Impossible d'ajouter xpacket à {xmp_path}: {e}")
