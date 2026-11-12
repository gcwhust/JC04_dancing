"""A reproducible score must keep full-body motion, phrasing and smooth joins."""
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_generated_score_has_visible_legs_wings_and_driven_turns(tmp_path):
    from jc04_dancing.choreography import write_swan_lake
    from jc04_dancing.motion import load_clip
    path, phrases = tmp_path / 'motion.csv', tmp_path / 'phrases.json'
    write_swan_lake(path, phrases)
    clip = load_clip(path)
    samples = [clip.sample(t) for t in np.linspace(0, clip.duration, 2000)]
    wheels = np.array([list(clip.wheel_velocities(t).values()) for t in np.linspace(0, clip.duration, 2000)])
    assert clip.duration == 94
    for side in ('right', 'left'):
        hip = np.rad2deg([pose[f'{side}_hip_pitch'] for pose in samples])
        knee = np.rad2deg([pose[f'{side}_knee_pitch'] for pose in samples])
        assert np.ptp(hip) >= 12
        assert np.ptp(knee) >= 12
        assert knee - hip == pytest.approx(np.full(len(samples), 60.7787), abs=1e-7)
        assert np.ptp([pose[f'{side}_shoulder_roll'] for pose in samples]) > math.radians(130)
    assert wheels.max() >= .65
    assert wheels.min() <= -.4
    assert np.max(np.abs(wheels[:, 0] - wheels[:, 1])) >= .25
    assert np.isfinite(wheels).all()
    assert np.abs(wheels).max() <= .9
    assert clip.wheel_velocities(0) == {'right_wheel_joint': 0, 'left_wheel_joint': 0}
    assert clip.wheel_velocities(94) == {'right_wheel_joint': 0, 'left_wheel_joint': 0}
    manifest = json.loads(phrases.read_text())
    assert manifest['duration_s'] == 94
    assert manifest['sections'][0]['start_s'] == 0
    assert manifest['sections'][-1]['end_s'] == 94
    assert len(manifest['sections']) >= 5
    assert all(a['end_s'] == b['start_s'] for a, b in zip(manifest['sections'], manifest['sections'][1:]))


def test_export_is_deterministic_and_includes_endpoint_at_fractional_duration(tmp_path):
    from jc04_dancing.choreography import write_swan_lake
    path, phrases = tmp_path / 'motion.csv', tmp_path / 'phrases.json'
    write_swan_lake(path, phrases, duration_s=83.217, fps=24)
    first = path.read_bytes(), phrases.read_bytes()
    write_swan_lake(path, phrases, duration_s=83.217, fps=24)
    assert (path.read_bytes(), phrases.read_bytes()) == first
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[0]['time_s']) == 0
    assert float(rows[-1]['time_s']) == 83.217
    assert len(rows) == math.ceil(83.217 * 24) + 1
    assert all(float(a['time_s']) < float(b['time_s']) for a, b in zip(rows, rows[1:]))


def test_quintic_score_and_export_are_c2_at_every_pose_event(tmp_path):
    from jc04_dancing.choreography import write_swan_lake
    from jc04_dancing.motion import load_clip
    path, phrases = tmp_path / 'motion.csv', tmp_path / 'phrases.json'
    write_swan_lake(path, phrases)
    clip = load_clip(path)
    metadata = json.loads(phrases.read_text())
    h = .0001
    for event in metadata['pose_events']:
        t = event['body_time_s']
        values = np.array([list(clip.sample(t + n * h).values()) + list(clip.wheel_velocities(t + n * h).values()) for n in (-2, -1, 0, 1, 2)])
        left_v, right_v = (values[2] - values[1]) / h, (values[3] - values[2]) / h
        left_a = (values[2] - 2 * values[1] + values[0]) / h**2
        right_a = (values[4] - 2 * values[3] + values[2]) / h**2
        assert left_v == pytest.approx(right_v, abs=.001)
        assert left_a == pytest.approx(right_a, abs=.01)


@pytest.mark.parametrize('kwargs', [{'duration_s': 0}, {'duration_s': float('nan')}, {'fps': 0}, {'fps': float('inf')}])
def test_export_rejects_invalid_timing_without_writing(tmp_path, kwargs):
    from jc04_dancing.choreography import write_swan_lake
    path = tmp_path / 'motion.csv'
    with pytest.raises(ValueError):
        write_swan_lake(path, tmp_path / 'phrases.json', **kwargs)
    assert not path.exists()


@pytest.mark.parametrize('competing_package', [False, True])
def test_generator_uses_checkout_from_other_cwd(tmp_path, competing_package):
    script = Path(__file__).resolve().parents[1] / 'tools/generate_swan_lake.py'
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    if competing_package:
        stale_package = tmp_path / 'stale/jc04_dancing'
        stale_package.mkdir(parents=True)
        (stale_package / '__init__.py').write_text("raise RuntimeError('Stale package used for checkout export')\n")
        environment['PYTHONPATH'] = str(stale_package.parent)
    other_cwd, output = tmp_path / 'elsewhere', tmp_path / 'output'
    other_cwd.mkdir()
    result = subprocess.run([sys.executable, str(script), '--output-dir', str(output)],
                            cwd=other_cwd, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with (output / 'swan_lake.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2821
    assert float(rows[-1]['time_s']) == 94
    assert json.loads((output / 'swan_lake.phrases.json').read_text())['sample_count'] == 2821
