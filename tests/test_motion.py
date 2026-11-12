"""Protect full-body choreography, explicit seconds, and continuous transitions."""
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
WHEELS = ('right_wheel_velocity_rad_s', 'left_wheel_velocity_rad_s')


def write_full_body_clip(tmp_path, *, omit=(), replace=None, times=(0, 1, 2, 3),
                         elbow=(0, 45, 90, 135), columns=None,
                         right_wheel=(0, .4, .4, 0), left_wheel=(0, .2, .2, 0)):
    columns = list(columns or ('time_s', *ARMS, *LEGS, *WHEELS))
    columns = [name for name in columns if name not in omit]
    path = tmp_path / 'full_body.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, t in enumerate(times):
            row = dict.fromkeys(ARMS, 0)
            row.update(time_s=t, right_elbow_pitch=elbow[index],
                       right_hip_pitch=40 + 4 * t, right_knee_pitch=100 + 4 * t,
                       left_hip_pitch=40 + 4 * t, left_knee_pitch=100 + 4 * t,
                       right_wheel_velocity_rad_s=right_wheel[index],
                       left_wheel_velocity_rad_s=left_wheel[index])
            if replace:
                row.update(replace)
            writer.writerow({name: row[name] for name in columns})
    return path


def timeline(clip, **kwargs):
    from jc04_dancing.motion import DanceTimeline
    config = yaml.safe_load(resource_path('configs/robot.yaml').read_text())
    rest = {name: math.radians(value) for name, value in config['rest_deg'].items()}
    return DanceTimeline(clip, rest, **kwargs), rest


def test_source_angles_are_radians_and_hold_outside_clip(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_full_body_clip(tmp_path))
    assert set(clip.joint_names) == set(ARMS + LEGS)
    assert clip.sample(1)['right_elbow_pitch'] == pytest.approx(math.pi / 4)
    assert clip.sample(-10) == clip.sample(0)
    assert clip.sample(10) == clip.sample(3)
    assert clip.sample(10)['right_elbow_pitch'] == pytest.approx(3 * math.pi / 4)


def test_quintic_between_samples(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_full_body_clip(tmp_path, elbow=(0, 17, 64, 81)))
    # These knots lie on 81*(10s^3-15s^4+6s^5), s=t/3, with C2 endpoints.
    # At s=1/4 the angle is 8.384765625 degrees, rather than linear 12.75.
    assert clip.sample(.75)['right_elbow_pitch'] == pytest.approx(math.radians(8.384765625))


def test_column_order(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_full_body_clip(tmp_path, columns=(*reversed(ARMS + LEGS + WHEELS), 'time_s')))
    result = clip.sample(1)
    assert result['right_elbow_pitch'] == pytest.approx(math.pi / 4)
    assert result['right_hip_pitch'] == pytest.approx(math.radians(44))
    assert clip.wheel_velocities(1) == pytest.approx({'right_wheel_joint': .4, 'left_wheel_joint': .2})


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'descending', 'nan', 'infinity', 'short', 'bad_number', 'duplicate_header'])
def test_bad_csv(tmp_path, case):
    from jc04_dancing.motion import load_clip
    options = {
        'missing': {'omit': ('right_elbow_pitch',)},
        'duplicate': {'times': (0, 1, 1, 3)},
        'descending': {'times': (0, 2, 1, 3)},
        'nan': {'elbow': (0, 45, float('nan'), 135)},
        'infinity': {'times': (0, 1, 2, float('inf'))},
        'short': {'times': (0, 1, 2)},
        'bad_number': {'elbow': (0, 45, 'oops', 135)},
        'duplicate_header': {'columns': ('time_s', *ARMS, *LEGS, *WHEELS, ARMS[0])},
    }
    with pytest.raises(ValueError):
        load_clip(write_full_body_clip(tmp_path, **options[case]))


def test_source_seconds_and_samples(tmp_path):
    from jc04_dancing.motion import load_clip
    path = write_full_body_clip(tmp_path, times=(0, 10, 40, 80))
    clip = load_clip(path)
    assert clip.start == 0
    assert clip.duration == pytest.approx(80)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        actual = clip.sample(float(row['time_s']))
        assert actual == pytest.approx({name: math.radians(float(row[name])) for name in ARMS + LEGS})


def test_duration_includes_preparation_return_and_hold(tmp_path):
    from jc04_dancing.motion import load_clip
    dance, _ = timeline(load_clip(write_full_body_clip(tmp_path, times=(0, 10, 40, 80))))
    assert dance.total_duration == pytest.approx(85)


def test_transition_continuity(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_full_body_clip(tmp_path, elbow=(10, 20, 30, 40)))
    dance, rest = timeline(clip)
    for boundary in (0, 2, 5, 7, 8):
        assert dance.sample(boundary - 1e-7) == pytest.approx(dance.sample(boundary + 1e-7), abs=1e-6)
    assert dance.sample(-1) == rest
    assert dance.sample(8) == rest
    assert dance.sample(100) == rest
    # Quintic transition weights at quarter time are 53/512 and 459/512.
    assert dance.sample(.5)['right_elbow_pitch'] == pytest.approx(math.radians(10) * 53 / 512)
    assert dance.sample(5.5)['right_elbow_pitch'] == pytest.approx(math.radians(40) * 459 / 512)
    assert dance.sample(5)['right_hip_pitch'] == pytest.approx(math.radians(52))
    assert dance.sample(5.5)['right_hip_pitch'] == pytest.approx(math.radians(40 + 12 * 459 / 512))


def test_body_is_unchanged_and_sampling_has_no_state(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_full_body_clip(tmp_path))
    dance, _ = timeline(clip)
    before = dance.sample(2.625)
    dance.sample(7)
    assert dance.sample(2.625) == before
    assert before == clip.sample(.625)
    assert dance.wheel_velocities(2.625) == clip.wheel_velocities(.625)


@pytest.mark.parametrize('kwargs', [{'prep': -1}, {'return_time': 0}, {'hold': -1}, {'prep': float('nan')}])
def test_invalid_transitions(tmp_path, kwargs):
    from jc04_dancing.motion import load_clip
    with pytest.raises(ValueError):
        timeline(load_clip(write_full_body_clip(tmp_path)), **kwargs)


def test_explicit_leg_and_wheel_channels_keep_source_units_and_timing(tmp_path):
    from jc04_dancing.motion import load_clip
    clip = load_clip(write_full_body_clip(tmp_path))
    dance, _ = timeline(clip)
    assert clip.duration == 3
    assert set(clip.joint_names) == set(ARMS + LEGS)
    assert dance.sample(3)['right_hip_pitch'] == pytest.approx(math.radians(44))
    assert dance.sample(3)['left_knee_pitch'] == pytest.approx(math.radians(104))
    assert dance.wheel_mode == 'driven'
    assert dance.wheel_velocities(3) == pytest.approx({'right_wheel_joint': .4, 'left_wheel_joint': .2})
    assert dance.wheel_velocities(-1) == {'right_wheel_joint': 0, 'left_wheel_joint': 0}
    assert dance.wheel_velocities(8) == {'right_wheel_joint': 0, 'left_wheel_joint': 0}


@pytest.mark.parametrize('omit', [('right_hip_pitch',), ('left_wheel_velocity_rad_s',)])
def test_partial_full_body_channels_are_rejected(tmp_path, omit):
    from jc04_dancing.motion import load_clip
    with pytest.raises(ValueError, match='complete'):
        load_clip(write_full_body_clip(tmp_path, omit=omit))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 'invalid'])
def test_nonfinite_wheel_channels_are_rejected(tmp_path, value):
    from jc04_dancing.motion import load_clip
    with pytest.raises(ValueError, match='finite'):
        load_clip(write_full_body_clip(tmp_path, replace={'right_wheel_velocity_rad_s': value}))


@pytest.mark.parametrize('side', ['right_wheel', 'left_wheel'])
@pytest.mark.parametrize('endpoint', [0, -1])
def test_full_body_wheels_must_stop_at_both_endpoints(tmp_path, side, endpoint):
    from jc04_dancing.motion import load_clip
    speeds = [0, .4, .4, 0]
    speeds[endpoint] = .1
    with pytest.raises(ValueError, match='zero wheel velocities at both endpoints'):
        load_clip(write_full_body_clip(tmp_path, **{side: speeds}))


@pytest.mark.parametrize('first_time', [-.1, .1])
def test_full_body_time_must_start_at_zero(tmp_path, first_time):
    from jc04_dancing.motion import load_clip
    with pytest.raises(ValueError, match='start at zero'):
        load_clip(write_full_body_clip(tmp_path, times=(first_time, 1, 2, 3)))


def test_full_body_transitions_have_continuous_velocity_and_acceleration(tmp_path):
    import numpy as np
    from jc04_dancing.motion import load_clip
    dance, _ = timeline(load_clip(write_full_body_clip(tmp_path)))
    h = 1e-4
    for boundary in (0, 2, 5, 7, 8):
        values = np.array([list(dance.sample(boundary + offset * h).values()) for offset in (-2, -1, 0, 1, 2)])
        left_velocity, right_velocity = (values[2] - values[1]) / h, (values[3] - values[2]) / h
        left_acceleration = (values[2] - 2 * values[1] + values[0]) / h**2
        right_acceleration = (values[4] - 2 * values[3] + values[2]) / h**2
        assert left_velocity == pytest.approx(right_velocity, abs=1e-5)
        assert left_acceleration == pytest.approx(right_acceleration, abs=3e-3)


def test_arm_only_csv_is_rejected(tmp_path):
    from jc04_dancing.motion import load_clip
    path = write_full_body_clip(tmp_path, omit=LEGS + WHEELS)
    with pytest.raises(ValueError, match='complete ten-joint and two-wheel'):
        load_clip(path)


def test_frame_clock_is_rejected(tmp_path):
    from jc04_dancing.motion import load_clip
    path = write_full_body_clip(tmp_path)
    path.write_text(path.read_text().replace('time_s', 'frame'))
    with pytest.raises(ValueError, match='time_s'):
        load_clip(path)
