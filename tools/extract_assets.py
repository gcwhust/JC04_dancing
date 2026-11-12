"""Extract only the agreed model, referenced meshes, dances and candidate music.

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
MOTION = "JC04Dev/src/motion_control/motion_real/trajectory"
MOTIONS = ("happy_new_year.csv", "doudoulong_postprocess.csv")


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
    for name in MOTIONS:
        copy(f"{MOTION}/{name}", f"motions/{name}")
    copy(f"{MOTION}/happy_new_year.mp3", "audio/happy_new_year.mp3")
    copy(f"{DESC}/LICENSE", "licenses/jc04_desc4-LICENSE.txt")
    motion_license = "JC04Dev/src/motion_control/LICENSE"
    if (source / motion_license).is_file():
        copy(motion_license, "licenses/motion_control-LICENSE.txt")
    notice = (
        "# Extracted asset notices\n\n"
        "Model: JC04 description V1.3, jc04_desc4, source snapshot " + SNAPSHOT + ". "
        "The source model package supplies Apache-2.0; its license is preserved here. "
        "Only mesh URI prefixes in the URDF were normalized; no model geometry or dance CSV was changed.\n\n"
        "Dance CSVs: preserved from motion_control; the source package license, when present, is copied separately. "
        "The user confirmed the two built-in dance choreographies are the team's original work and authorized public extraction of these assets. "
        "No author names or creation dates are inferred.\n\n"
        "Audio: happy_new_year.mp3 is a candidate copied from the local trajectory directory. "
        "Its selection, sync and redistribution status require the dedicated audio review. "
        "The model package license must not be interpreted as licensing this recording or composition. "
        "No audio rights holder or track metadata is invented.\n"
    )
    (destination / "licenses/ASSET-NOTICES.md").write_text(notice, encoding="utf-8")
    manifest = {
        "schema_version": 1, "source_snapshot": SNAPSHOT,
        "source_identifier": "JC04 local reference snapshot (relative paths only)",
        "model": {"package": "jc04_desc4", "version": "V1.3", "source": urdf_path},
        "audio": {"file": "audio/happy_new_year.mp3", "status": "candidate_pending_audio_review", "license": "not_inferred_from_model"},
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
