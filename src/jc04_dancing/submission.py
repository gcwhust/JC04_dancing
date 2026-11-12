"""Stage the verified submission with positive source and evidence boundaries."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from zipfile import ZIP_DEFLATED, ZipFile

from .reporting import _sha256, build_report
from .resources import resource_path


def validate_identity(team_id: str, track_name: str) -> None:
    """Reject placeholders and names unsafe for filenames or archive members."""
    for value in (team_id, track_name):
        if (not isinstance(value, str) or not value.strip() or value != value.strip() or
                len(value) > 80 or value.casefold() in {'team_id', 'team-id', 'tbd', 'unknown', 'none', 'null', 'xxx', '赛道名称'} or '..' in value or any(c in value for c in '/\\:*?"<>|') or
                any(unicodedata.category(c).startswith('C') for c in value) or
                re.search(r'todo|placeholder|changeme|your[_ -]?(team|id)|待填写|待确认|占位|示例', value, re.I)):
            raise ValueError('Identity must be a confirmed, non-placeholder, safe filename component')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', team_id):
        raise ValueError('team_id must use ASCII letters, digits, underscores or hyphens')


def _source_root() -> Path:
    # Source CLI works in its checkout; ordinary installs can use --source-root.
    candidates = [Path(__file__).resolve().parents[2], Path.cwd()]
    for candidate in candidates:
        if (candidate / 'pyproject.toml').is_file() and (candidate / 'src/jc04_dancing').is_dir():
            return candidate
    raise ValueError('Source checkout required for source ZIP; pass --source-root pointing to extracted source')


def build_source_zip(source_root: Path, output: Path) -> None:
    """Copy only reviewed, individually named files; never recurse arbitrary trees."""
    source_root = Path(source_root).resolve()
    names = resource_path('configs/source-files.txt').read_text(encoding='utf-8').splitlines()
    if not names or len(names) != len(set(names)):
        raise ValueError('Invalid source whitelist')
    files = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Unsafe source whitelist path')
        path = source_root / relative
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != source_root.parent):
            raise ValueError(f'Source symlink is not allowed: {name}')
        if not path.resolve().is_relative_to(source_root) or not path.is_file():
            raise ValueError(f'Missing or unsafe required source file: {name}')
        files.append((name, path))
    with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in files:
            archive.write(path, name)
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Source ZIP CRC validation failed')


def build_submission(run_dir: Path, destination: Path, team_id: str = 'ROBOT261937J',
                     track_name: str = '机器人舞蹈', *, source_root: Path | None = None) -> dict:
    """Build a named, hash-checked submission from the selected audiovisual run.

    Reports are regenerated from the same complete recording, preventing stale
    report/video mode or provenance mismatches. Existing submissions are never
    overwritten. SHA256SUMS covers every inner file except itself; manifest covers
    artifacts/evidence, excluding itself and SHA256SUMS to avoid circular hashes.
    """
    validate_identity(team_id, track_name)
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    if destination.is_relative_to(run_dir):
        raise ValueError('Submission output must be outside accepted recording evidence')
    prefix = f'{team_id}_{track_name}'
    target, outer = destination / prefix, destination / f'{prefix}_提交材料.zip'
    if target.exists() or outer.exists():
        raise ValueError('Submission destination already exists; choose a new output directory')
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.submission-', dir=destination) as temporary:
        work = Path(temporary)
        staging = work / prefix
        staging.mkdir()
        report = build_report(run_dir, staging / f'{prefix}_文档.pdf', team_id)
        recorded = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        for name in [recorded['video']['file'], *[image['file'] for image in recorded['stills']]]:
            if ('\\' in name or Path(name).is_absolute() or '..' in Path(name).parts or
                    any(unicodedata.category(c).startswith('C') for c in name)):
                raise ValueError(f'Unsafe evidence archive path: {name}')
        for key, suffix in [('flowchart_pdf', '流程图.pdf'), ('flowchart_png', '流程图.png')]:
            old = report['artifacts'][key]['file']
            name = f'{prefix}_{suffix}'
            (staging / old).rename(staging / name)
            report['artifacts'][key]['file'] = name
        report_json = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
        # Preserve the named metadata alias alongside report.json for downstream
        # tools after the flowchart artifact names are updated.
        for name in (f'{prefix}_文档.json', 'report.json'):
            (staging / name).write_text(report_json, encoding='utf-8')
        shutil.copyfile(run_dir / report['video']['file'], staging / f'{prefix}_视频.mp4')
        evidence = staging / 'evidence'
        evidence.mkdir()
        shutil.copyfile(run_dir / 'run.json', evidence / 'run.json')
        # All paths have already passed report validation; copy only listed stills.
        for image in recorded['stills']:
            relative = Path(image['file'])
            (evidence / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_dir / relative, evidence / relative)
        build_source_zip(source_root if source_root is not None else _source_root(), staging / f'{prefix}_代码.zip')
        metadata = {
            'schema_version': 1, 'status': 'audio_deliverable' if report['audio']['present'] else 'pre_music_deliverable', 'team_id': team_id,
            'track_name': track_name, 'dance_id': report['simulation']['dance_id'],
            'audio': report['audio'], 'simulation': report['simulation'], 'video': report['video'],
            'source': report['source'], 'report_artifacts': report['artifacts'],
            'evidence_video_file': f'{prefix}_视频.mp4',
            'evidence_note': 'run.json video.file identifies the original recording filename; the identical video is named above. Original run SHA256 is preserved.',
            'limitations': report['limitations'],
            'originality': 'User confirmed the four bundled choreographies are original team work; retained without inventing authorship details.',
            'files_sha256': {p.relative_to(staging).as_posix(): _sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()},
        }
        (staging / 'manifest.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        sums = ''.join(f'{_sha256(p)}  {p.relative_to(staging).as_posix()}\n' for p in sorted(staging.rglob('*')) if p.is_file())
        (staging / 'SHA256SUMS').write_text(sums, encoding='utf-8')
        staged_outer = work / outer.name
        with ZipFile(staged_outer, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(staging.rglob('*')):
                if path.is_file():
                    archive.write(path, f'{prefix}/{path.relative_to(staging).as_posix()}')
        with ZipFile(staged_outer) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('Submission ZIP CRC validation failed')
        staging.rename(target)
        staged_outer.rename(outer)
    return metadata
