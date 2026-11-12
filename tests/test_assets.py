"""Protect installation independence and the public asset boundary."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

MOTIONS = ('swan_lake.csv',)


def test_resources_outside_cwd(tmp_path, monkeypatch):
    # A cwd-relative resolver would lose every installed asset here.
    from jc04_dancing.resources import resource_path
    monkeypatch.chdir(tmp_path)
    assert {path.name for path in resource_path('assets/motions').glob('*.csv')} == set(MOTIONS)
    for name in MOTIONS:
        assert resource_path('assets/motions/' + name).is_file()
    urdf = resource_path('assets/robot/jc04.urdf')
    root = ET.parse(urdf).getroot()
    assert len(root.findall('link')) == 28
    assert len([j for j in root.findall('joint') if j.get('type') != 'fixed']) == 12
    for mesh in root.iter('mesh'):
        assert (urdf.parent / mesh.attrib['filename']).is_file()


def test_missing_resource_names_the_request():
    from jc04_dancing.resources import resource_path
    with pytest.raises(FileNotFoundError, match='missing.csv'):
        resource_path('assets/motions/missing.csv')


@pytest.mark.parametrize('name', ['../outside', '/etc/passwd'])
def test_resource_cannot_escape_package(name):
    from jc04_dancing.resources import resource_path
    with pytest.raises(ValueError):
        resource_path(name)


def test_manifest_hashes():
    # A missing copy or transformed file with a stale hash must be detected.
    from jc04_dancing.resources import resource_path
    assets = resource_path('assets')
    manifest = json.loads((assets / 'provenance.json').read_text())
    assert manifest['source_snapshot'] == '2d224d13d624e07102a2c6eedbaf111694555e16'
    files = manifest['files']
    assert {entry['target'] for entry in files if entry['target'].startswith('motions/') and entry['target'].endswith('.csv')} == {'motions/' + n for n in MOTIONS}
    for entry in files:
        assert len(entry['source_sha256']) == 64
        actual = hashlib.sha256((assets / entry['target']).read_bytes()).hexdigest()
        assert actual == entry['target_sha256'], entry['target']


def test_no_private_tree():
    # Detect accidental inclusion of deployment code/config and local paths.
    from jc04_dancing.resources import resource_path
    package = resource_path('.')
    allowed = {'.py', '.urdf', '.xml', '.stl', '.msh', '.csv', '.mp3', '.json', '.yaml', '.md', '.txt', '.ttf'}
    for path in package.rglob('*'):
        if '__pycache__' in path.parts or not path.is_file():
            continue
        assert not any(p.startswith('JC04-master-') for p in path.relative_to(package).parts)
        assert path.suffix.lower() in allowed, str(path)
        assert not any(token in path.name.lower() for token in ('deploy', 'credential', 'secret'))
        if path.suffix.lower() not in {'.stl', '.msh', '.mp3', '.ttf'}:
            content = path.read_text()
            assert '/home/' not in content
            assert '/Users/' not in content


def test_extraction_is_reproducible_and_whitelisted(tmp_path):
    # Execute against a tiny reference so recursive copies expose a forbidden file.
    source = tmp_path / 'reference'
    desc = source / 'JC04Dev/src/jc04_desc4'
    trajectory = source / 'JC04Dev/src/motion_control/motion_real/trajectory'
    (desc / 'urdf').mkdir(parents=True)
    (desc / 'meshes').mkdir()
    trajectory.mkdir(parents=True)
    (desc / 'urdf/jc04.urdf').write_text('<robot name="fixture"><link name="base"><visual><geometry><mesh filename="package://jc04_desc4/meshes/used.STL"/></geometry></visual></link></robot>')
    (desc / 'meshes/used.STL').write_bytes(b'mesh fixture')
    (desc / 'meshes/unused.STL').write_bytes(b'not referenced')
    (desc / 'LICENSE').write_text('Apache fixture')
    for name in MOTIONS:
        (trajectory / name).write_text('frame,right_elbow_pitch\n0,0\n')
    (trajectory / 'unselected.csv').write_text('frame,right_elbow_pitch\n0,0\n')
    (trajectory / 'unselected.mp3').write_bytes(b'audio fixture')
    (trajectory / 'credentials.yaml').write_text('must not copy')
    script = Path(__file__).resolve().parents[1] / 'tools/extract_assets.py'
    destination = tmp_path / 'assets'
    result = subprocess.run([sys.executable, str(script), '--source', str(source), '--destination', str(destination)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    first = (destination / 'provenance.json').read_bytes()
    subprocess.run([sys.executable, str(script), '--source', str(source), '--destination', str(destination)], check=True)
    assert (destination / 'provenance.json').read_bytes() == first
    assert not (destination / 'motions').exists()
    assert not (destination / 'audio').exists()
    assert not (destination / 'licenses/motion_control-LICENSE.txt').exists()
    assert not (destination / 'robot/meshes/unused.STL').exists()
    assert not list(destination.rglob('credentials.yaml'))
    mesh = ET.parse(destination / 'robot/jc04.urdf').find('.//mesh')
    assert mesh.attrib['filename'] == 'meshes/used.STL'


def test_module_entrypoint_reports_version_from_other_cwd(tmp_path):
    # An absent or cwd-dependent __main__ must not break the installed entrypoint.
    import os
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    result = subprocess.run([sys.executable, '-m', 'jc04_dancing', '--version'], cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'jc04-dancing 0.1.0'
