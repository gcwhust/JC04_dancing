"""Submission boundaries, provenance and nested ZIP integrity."""
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import pytest

from test_reporting import run_dir  # real complete encoded evidence fixture

PREFIX = 'ROBOT261937J_机器人舞蹈'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('identity', ['', ' ', 'TODO', 'placeholder', '待填写', '..', '../x', 'a/b', 'a\\b', 'a\x00b', 'x\ny', 'TEAM_ID', 'TBD', '赛道名称'])
def test_identity_rejects_invalid(identity):
    from jc04_dancing.submission import validate_identity
    for team, track in [(identity, '机器人舞蹈'), ('ROBOT261937J', identity)]:
        with pytest.raises(ValueError):
            validate_identity(team, track)


@pytest.fixture(scope='module')
def packaged(run_dir, tmp_path_factory):
    from jc04_dancing.submission import build_submission
    destination = tmp_path_factory.mktemp('submission')
    result = build_submission(run_dir, destination)
    return destination, result


def test_submission_names(packaged, run_dir):
    destination, metadata = packaged
    folder = destination / PREFIX
    expected = {f'{PREFIX}_{suffix}' for suffix in ['视频.mp4', '文档.pdf', '流程图.pdf', '流程图.png', '代码.zip']}
    assert expected <= {p.name for p in folder.iterdir()}
    assert {'manifest.json', 'SHA256SUMS', 'report.json', 'evidence'} <= {p.name for p in folder.iterdir()}
    assert (destination / f'{PREFIX}_提交材料.zip').is_file()
    assert sha(folder / f'{PREFIX}_视频.mp4') == sha(run_dir / 'dance.mp4')
    assert metadata['audio']['D04_verified'] is False
    assert metadata['audio']['status'] == 'postproduction_pending'
    assert metadata['simulation']['mode'] == 'free'
    report = json.loads((folder / 'report.json').read_text())
    assert report['source']['run_sha256'] == sha(run_dir / 'run.json')
    assert report['simulation']['mode'] == metadata['simulation']['mode']
    original = json.loads((run_dir / 'run.json').read_text())
    for still in original['stills']:
        assert sha(folder / 'evidence' / still['file']) == sha(run_dir / still['file'])
    for artifact in report['artifacts'].values():
        assert sha(folder / artifact['file']) == artifact['sha256']


def test_source_whitelist(packaged, tmp_path):
    folder = packaged[0] / PREFIX
    with ZipFile(folder / f'{PREFIX}_代码.zip') as source:
        names = set(source.namelist())
        required = ['pyproject.toml', 'README.md', 'requirements-lock.txt', 'environment.yml',
                    'tools/convert_model.py', 'tests/test_submission.py', 'docs/validation.md',
                    'src/jc04_dancing/submission.py', 'src/jc04_dancing/assets/report/template.md',
                    'src/jc04_dancing/assets/report/fonts/DroidSansFallbackFull.ttf',
                    'src/jc04_dancing/assets/robot/robot.xml', 'src/jc04_dancing/assets/audio/happy_new_year.mp3',
                    'src/jc04_dancing/assets/motions/happy_new_year.csv']
        assert set(required) <= names
        assert {name for name in names if name.endswith('.csv')} == {
            'src/jc04_dancing/assets/motions/happy_new_year.csv',
            'src/jc04_dancing/assets/motions/doudoulong_postprocess.csv',
        }
        assert not any(part in {'.git', '.superpowers', 'outputs', '__pycache__', 'submission'} or
                       part.startswith('JC04-master-') for name in names for part in Path(name).parts)
        assert not any('credential' in name.lower() or '模拟仿真赛道任务说明书' in name for name in names)
        assert source.testzip() is None
    # A positive file whitelist must ignore a newly introduced text credential.
    from jc04_dancing.submission import build_source_zip
    fixture = tmp_path / 'source'
    shutil.copytree(ROOT / 'src', fixture / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    for name in ['README.md', 'pyproject.toml', 'requirements-lock.txt', 'environment.yml', '.gitignore', '.gitattributes']:
        shutil.copyfile(ROOT / name, fixture / name)
    for name in ['docs', 'tests', 'tools']:
        shutil.copytree(ROOT / name, fixture / name, ignore=shutil.ignore_patterns('__pycache__'))
    (fixture / 'src/jc04_dancing/credentials.txt').write_text('DO NOT SHIP')
    build_source_zip(fixture, tmp_path / 'source.zip')
    with ZipFile(tmp_path / 'source.zip') as archive:
        assert 'src/jc04_dancing/credentials.txt' not in archive.namelist()
    (fixture / 'README.md').unlink()
    (fixture / 'README.md').symlink_to(ROOT / 'README.md')
    with pytest.raises(ValueError, match='symlink'):
        build_source_zip(fixture, tmp_path / 'bad.zip')


@pytest.mark.parametrize('mutation', ['preview', 'partial', 'mode', 'hash', 'path', 'missing_still'])
def test_incomplete_run_rejected(run_dir, tmp_path, mutation):
    from jc04_dancing.submission import build_submission
    shutil.copytree(run_dir, tmp_path / 'run')
    path = tmp_path / 'run/run.json'
    data = json.loads(path.read_text())
    if mutation == 'preview': data['preview'] = True
    if mutation == 'partial': data['timeline_complete'] = False
    if mutation == 'mode': data['mode'] = 'supported'
    if mutation == 'hash': data['video']['sha256'] = '0' * 64
    if mutation == 'path': data['video']['file'] = '../outside.mp4'
    if mutation == 'missing_still': data['stills'][0]['file'] = 'missing.png'
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        build_submission(tmp_path / 'run', tmp_path / 'submission')
    assert not (tmp_path / 'submission' / PREFIX).exists()


def test_zip_hashes(packaged):
    destination, metadata = packaged
    folder = destination / PREFIX
    assert json.loads((folder / 'manifest.json').read_text()) == metadata
    for name, expected in metadata['files_sha256'].items():
        assert sha(folder / name) == expected
    sums = {}
    for line in (folder / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        sums[name] = digest
        assert sha(folder / name) == digest
    assert 'manifest.json' in sums and 'SHA256SUMS' not in sums
    with ZipFile(destination / f'{PREFIX}_提交材料.zip') as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {f'{PREFIX}/{p.relative_to(folder).as_posix()}' for p in folder.rglob('*') if p.is_file()}


def test_submit_cli_missing_run(tmp_path, capsys):
    from jc04_dancing.cli import main
    assert main(['submit', '--run-dir', str(tmp_path / 'missing'), '--output', str(tmp_path / 'out')]) == 1
    assert 'run.json' in capsys.readouterr().err


def test_supported_mode_is_preserved(run_dir, tmp_path):
    from jc04_dancing.submission import build_submission
    shutil.copytree(run_dir, tmp_path / 'run')
    path = tmp_path / 'run/run.json'
    data = json.loads(path.read_text())
    data['mode'] = data['simulation']['mode'] = 'supported'
    path.write_text(json.dumps(data))
    result = build_submission(tmp_path / 'run', tmp_path / 'out')
    report = json.loads((tmp_path / 'out' / PREFIX / 'report.json').read_text())
    assert result['simulation']['mode'] == report['simulation']['mode'] == 'supported'


def test_submission_does_not_overwrite_or_modify_evidence(packaged, run_dir):
    from jc04_dancing.submission import build_submission
    before = sha(run_dir / 'run.json')
    with pytest.raises(ValueError, match='already exists'):
        build_submission(run_dir, packaged[0])
    with pytest.raises(ValueError, match='outside'):
        build_submission(run_dir, run_dir / 'submission')
    assert sha(run_dir / 'run.json') == before


def test_windows_archive_traversal_rejected(run_dir, tmp_path):
    from jc04_dancing.submission import build_submission
    shutil.copytree(run_dir, tmp_path / 'run')
    path = tmp_path / 'run/run.json'
    data = json.loads(path.read_text())
    still = data['stills'][0]
    unsafe = '..\\outside.png'
    shutil.copyfile(tmp_path / 'run' / still['file'], tmp_path / 'run' / unsafe)
    still['file'] = unsafe
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='[Pp]ath|[Uu]nsafe'):
        build_submission(tmp_path / 'run', tmp_path / 'out')
    assert not (tmp_path / 'out' / PREFIX).exists()


def test_metadata_alias_is_packaged_and_hashed(packaged):
    # Concise PDF omits technical evidence filenames; aliases remain in the package.
    import fitz

    destination, metadata = packaged
    folder = destination / PREFIX
    name = f'{PREFIX}_文档.json'
    with fitz.open(folder / f'{PREFIX}_文档.pdf') as document:
        pdf_text = ''.join(page.get_text() for page in document).replace('\n', '')
    assert name not in pdf_text
    alias = folder / name
    assert alias.is_file()
    assert json.loads(alias.read_text()) == json.loads((folder / 'report.json').read_text())
    assert metadata['files_sha256'][name] == sha(alias)
    assert f'{sha(alias)}  {name}' in (folder / 'SHA256SUMS').read_text().splitlines()
    with ZipFile(destination / f'{PREFIX}_提交材料.zip') as archive:
        assert hashlib.sha256(archive.read(f'{PREFIX}/{name}')).hexdigest() == sha(alias)


def test_audible_submission_retains_track_and_current_status(run_dir, tmp_path):
    from jc04_dancing.audio import add_audio_to_run
    from jc04_dancing.submission import build_submission
    audible = tmp_path / 'audible'
    add_audio_to_run(run_dir, audible)
    result = build_submission(audible, tmp_path / 'out')
    assert result['status'] == 'audio_deliverable'
    assert result['audio']['present'] is True
    assert result['video']['audio_stream_present'] is True
    assert result['audio']['D04_verified'] is False
    assert sha(tmp_path / 'out' / PREFIX / f'{PREFIX}_视频.mp4') == sha(audible / 'dance.mp4')
