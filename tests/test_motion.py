"""Protect source-time choreography, radians, and continuous display transitions."""
import csv
import math

import pytest
import yaml

from jc04_dancing.resources import resource_path

ARMS = (
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow_pitch',
)
LEGS = ('right_hip_pitch', 'right_knee_pitch', 'left_hip_pitch', 'left_knee_pitch')


def write_clip(tmp_path, frames=(0, 12, 24, 36), elbow=(0, 45, 90, 135), columns=None):
    columns = list(columns or ('frame', *ARMS))
    path = tmp_path / 'fixture.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for frame, value in zip(frames, elbow):
            row = dict.fromkeys(ARMS, 0)
            row.update(frame=frame, right_elbow_pitch=value)
            writer.writerow({name: row[name] for name in columns})
    return path


def robot_config():
    return yaml.safe_load(resource_path('configs/robot.yaml').read_text())


def timeline(clip, **kwargs):
    from jc04_dancing.motion import DanceTimeline
    config = robot_config()
    rest = {name: math.radians(value) for name, value in config['rest_deg'].items()}
    return DanceTimeline(clip, rest, config['leg_pattern'], **kwargs), rest


def test_linear_degrees(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_clip(tmp_path))
    assert set(clip.joint_names) == set(ARMS)
    assert clip.sample(0.5)['right_elbow_pitch'] == pytest.approx(math.pi / 4)
    assert clip.sample(-10)['right_elbow_pitch'] == 0
    assert clip.sample(10)['right_elbow_pitch'] == pytest.approx(3 * math.pi / 4)


def test_cubic_between_frames(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_clip(tmp_path, elbow=(0, 1, 8, 27)))
    # y=(2t)^3: a linear replacement gives 0.5 degrees instead of 0.125.
    assert clip.sample(0.25)['right_elbow_pitch'] == pytest.approx(math.radians(0.125))


def test_frame_one_holds_first():
    from jc04_dancing.motion import load_clip
    clip = load_clip(resource_path('assets/motions/doudoulong_postprocess.csv'))
    assert clip.start == pytest.approx(1 / 24)
    assert clip.sample(0) == clip.sample(1 / 24)


def test_column_order(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_clip(tmp_path, columns=(*reversed(ARMS), 'frame')))
    result = clip.sample(0.5)
    assert result['right_elbow_pitch'] == pytest.approx(math.pi / 4)
    assert all(result[name] == 0 for name in ARMS if name != 'right_elbow_pitch')


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'descending', 'nan', 'infinity', 'short', 'bad_number', 'duplicate_header'])
def test_bad_csv(tmp_path, case):
    from jc04_dancing.motion import load_clip
    options = {
        'missing': {'columns': ('frame', *ARMS[:-1])},
        'duplicate': {'frames': (0, 12, 12, 36)},
        'descending': {'frames': (0, 24, 12, 36)},
        'nan': {'elbow': (0, 45, float('nan'), 135)},
        'infinity': {'frames': (0, 12, 24, float('inf'))},
        'short': {'frames': (0, 12, 24)},
        'bad_number': {'elbow': (0, 45, 'oops', 135)},
        'duplicate_header': {'columns': ('frame', *ARMS, ARMS[0])},
    }
    with pytest.raises(ValueError):
        load_clip(write_clip(tmp_path, **options[case]))


@pytest.mark.parametrize('fps', [0, -24, float('nan'), float('inf')])
def test_bad_fps(tmp_path, fps):
    from jc04_dancing.motion import load_clip
    with pytest.raises(ValueError):
        load_clip(write_clip(tmp_path), fps=fps)


@pytest.mark.parametrize(('filename', 'end'), [
    ('happy_new_year.csv', 1880 / 24),
    ('doudoulong_postprocess.csv', 1920 / 24),
])
def test_builtin_end_times_and_source_samples(filename, end):
    from jc04_dancing.motion import load_clip
    path = resource_path('assets/motions/' + filename)
    clip = load_clip(path)
    assert clip.duration == pytest.approx(end)
    # Source knots independently establish units, timing, and named mapping.
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows[::137] + rows[-1:]:
        actual = clip.sample(float(row['frame']) / 24)
        assert actual == pytest.approx({name: math.radians(float(row[name])) for name in ARMS})


def test_leg_phase(tmp_path):
    from jc04_dancing.motion import load_clip
    dance, rest = timeline(load_clip(write_clip(tmp_path, frames=(0, 24, 48, 72))))
    sample = dance.sample(3.0)  # one second into body, not global phase 3s
    assert len(sample) == 10
    for name in LEGS[:2]:
        assert sample[name] - rest[name] == pytest.approx(math.radians(1))
    for name in LEGS[2:]:
        assert sample[name] - rest[name] == pytest.approx(math.radians(-1))


def test_duration_new_year():
    from jc04_dancing.motion import load_clip
    dance, _ = timeline(load_clip(resource_path('assets/motions/happy_new_year.csv')))
    assert dance.total_duration == pytest.approx(1880 / 24 + 5)


def test_transition_continuity(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_clip(tmp_path, frames=(0, 24, 48, 72), elbow=(10, 20, 30, 40)))
    dance, rest = timeline(clip)
    for boundary in (0, 2, 5, 7, 8):
        assert dance.sample(boundary - 1e-7) == pytest.approx(dance.sample(boundary + 1e-7), abs=1e-6)
    assert dance.sample(-1) == rest
    assert dance.sample(8) == rest
    assert dance.sample(100) == rest
    # Smoothstep at quarter time is 0.15625, not linear 0.25.
    assert dance.sample(0.5)['right_elbow_pitch'] == pytest.approx(math.radians(10) * 0.15625)
    assert dance.sample(5.5)['right_elbow_pitch'] == pytest.approx(math.radians(40) * 0.84375)
    # Return starts at the actual final leg target (-1 degree on right).
    assert dance.sample(5)['right_hip_pitch'] == pytest.approx(rest['right_hip_pitch'] - math.radians(1))
    assert dance.sample(5.5)['right_hip_pitch'] == pytest.approx(rest['right_hip_pitch'] - math.radians(1) * 0.84375)


def test_body_is_unchanged_and_sampling_has_no_state(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_clip(tmp_path, frames=(0, 24, 48, 72)))
    dance, _ = timeline(clip)
    before = dance.sample(2.625)
    dance.sample(7)
    assert dance.sample(2.625) == before
    for name in ARMS:
        assert before[name] == clip.sample(0.625)[name]


@pytest.mark.parametrize('kwargs', [{'prep': -1}, {'return_time': 0}, {'hold': -1}, {'prep': float('nan')}])
def test_invalid_transitions(tmp_path, kwargs):
    from jc04_dancing.motion import load_clip
    with pytest.raises(ValueError):
        timeline(load_clip(write_clip(tmp_path)), **kwargs)
