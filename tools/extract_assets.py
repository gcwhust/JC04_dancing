"""Extract only the agreed robot model and its referenced meshes.

The source is a local reference checkout, never a runtime dependency. Source
identifiers in the resulting manifest are relative to that checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

SNAPSHOT = "2d224d13d624e07102a2c6eedbaf111694555e16"
DESC = "JC04Dev/src/jc04_desc4"


def extract(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    entries = []

    def copy(relative: str, target: str, transformation: str = "byte-for-byte copy", content: bytes | None = None) -> None:
        origin = (source / relative).resolve()
        if not origin.is_relative_to(source):
            raise ValueError(f"Source asset escapes checkout: {relative}")
        original = origin.read_bytes()
        output = original if content is None else content
        path = destination / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(output)
        entries.append({"source": relative, "target": target,
                        "source_sha256": hashlib.sha256(original).hexdigest(),
                        "target_sha256": hashlib.sha256(output).hexdigest(),
                        "transformation": transformation})

    urdf_path = f"{DESC}/urdf/jc04.urdf"
    original = (source / urdf_path).read_bytes()
    root = ET.fromstring(original)
    normalized = original
    mesh_names = set()
    for mesh in root.iter("mesh"):
        uri = mesh.attrib["filename"]
        prefix = "package://jc04_desc4/"
        if not uri.startswith(prefix):
            raise ValueError(f"Unsupported mesh reference: {uri}")
        relative = PurePosixPath(uri[len(prefix):])
        if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 2 or relative.parts[0] != 'meshes' or relative.suffix.lower() != '.stl':
            raise ValueError(f"Mesh outside whitelist: {uri}")
        mesh_names.add(relative.as_posix())
        normalized = normalized.replace(uri.encode(), relative.as_posix().encode())
    copy(urdf_path, "robot/jc04.urdf", "Replace package://jc04_desc4/ mesh URI prefix with a relative path; preserve all other bytes", normalized)
    for relative in sorted(mesh_names):
        copy(f"{DESC}/{relative}", f"robot/{relative}")
    copy(f"{DESC}/LICENSE", "licenses/jc04_desc4-LICENSE.txt")
    notice = (
        "# Extracted model notices\n\n"
        "Model: JC04 description V1.3, jc04_desc4, source snapshot " + SNAPSHOT + ". "
        "The source model package supplies Apache-2.0; its license is preserved here. "
        "Only mesh URI prefixes in the URDF were normalized. "
        "Motion and music assets are created or sourced separately; no controller source is extracted.\n"
    )
    (destination / "licenses/ASSET-NOTICES.md").write_text(notice, encoding="utf-8")
    manifest = {
        "schema_version": 1, "source_snapshot": SNAPSHOT,
        "source_identifier": "JC04 local reference snapshot (relative paths only)",
        "model": {"package": "jc04_desc4", "version": "V1.3", "source": urdf_path},
        "files": sorted(entries, key=lambda entry: entry["target"]),
    }
    (destination / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    extract(args.source, args.destination)


if __name__ == "__main__":
    main()
