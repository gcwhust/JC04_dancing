"""Physics contracts: integration, bounded torque, explicit support and time."""
import json
import math

import mujoco
import numpy as np
import pytest

from jc04_dancing.simulation import Simulation, build_timeline, validate_targets
from jc04_dancing.cli import main


@pytest.fixture(autouse=True)
def reset_simulation(simulation):
    simulation.mode = 'supported'
    simulation.wheel_mode = 'locked'
    simulation.reset()


def test_absolute_time(simulation):
    # Per-frame rounded step counts accumulate drift; absolute targets do not.
    for k in range(151):
        requested = k / 30
        simulation.advance_to(requested)
        assert requested <= simulation.data.time < requested + .0005 + 1e-9
    assert simulation.summary()['steps'] in (10000, 10001)


def test_reset_contact(simulation):
    simulation.advance_to(.2)
    simulation.reset()
    assert simulation.data.time == 0
    assert np.isfinite(simulation.data.qpos).all()
    assert all(contact.dist >= -.002 for contact in simulation.data.contact)
    assert simulation.data.qpos[2] == pytest.approx(.535930035158)


def test_wheels_have_no_active_motors(simulation):
    wheel_ids = {simulation.model.joint(side + '_wheel_joint').id for side in ('right', 'left')}
    assert not wheel_ids.intersection(simulation.model.actuator_trnid[:, 0])
    simulation.advance_to(.1)
    assert simulation.model.nu == 10


def test_wheels_locked_during_moving_dance_with_floating_base(simulation):
    simulation.mode = 'free'
    simulation.reset()
    assert not simulation.data.eq_active[simulation.model.equality('base_support').id]
    simulation.advance_to(7)
    stats = simulation.summary()
    assert stats['wheel_mode'] == 'locked'
    assert stats['max_wheel_rotation_rad'] < 1e-4
    assert stats['max_wheel_speed_rad_s'] < .01
    for side in ('right', 'left'):
        equality = simulation.model.equality(side + '_wheel_lock')
        assert equality.type == mujoco.mjtEq.mjEQ_JOINT
        assert simulation.data.eq_active[equality.id]
    # The brake constrains a wheel relative to its leg, never welds the chassis.
    assert simulation.model.joint('root').type == mujoco.mjtJoint.mjJNT_FREE


def test_passive_wheel_debug_is_explicit(simulation):
    simulation.wheel_mode = 'passive'
    simulation.mode = 'free'
    simulation.reset()
    for side in ('right', 'left'):
        assert not simulation.data.eq_active[simulation.model.equality(side + '_wheel_lock').id]
    simulation.advance_to(3)
    stats = simulation.summary()
    assert stats['wheel_mode'] == 'passive'
    assert stats['max_wheel_rotation_rad'] > .01


def test_support_explicit(simulation):
    equality = simulation.model.equality('base_support').id
    assert simulation.data.eq_active[equality]
    simulation.mode = 'free'
    simulation.reset()
    assert not simulation.data.eq_active[equality]
    assert simulation.summary()['mode'] == 'free'
    simulation.advance_to(2)
    # Wheel brakes do not suspend the floating chassis against gravity.
    assert simulation.data.qpos[2] < .534


def test_limit_error(simulation):
    class BadTimeline:
        total_duration = .004

        def sample(self, t):
            return {'right_shoulder_pitch': 100.0}

    with pytest.raises(ValueError, match=r'right_shoulder_pitch.*t=.*range'):
        validate_targets(simulation.model, BadTimeline(), .002)


def test_explicit_playback_ranges_preserve_original_targets(simulation):
    # The source URDF disallows real poses present in the unmodified CSVs.
    ranges = simulation.summary()['joint_range_overrides']
    for side in ('right', 'left'):
        roll = ranges[side + '_shoulder_roll']
        assert roll['source_rad'] == [0, 3.14]
        assert simulation.model.joint(side + '_shoulder_roll').range[0] <= math.radians(-10)
        elbow = ranges[side + '_elbow_pitch']
        assert elbow['source_rad'] == [-1.5708, 1.85]
        assert simulation.model.joint(side + '_elbow_pitch').range[1] >= math.radians(120)


def test_nonfinite_target_error(simulation):
    class BadTimeline:
        total_duration = .004

        def sample(self, t):
            return {'right_shoulder_pitch': float('nan')}

    with pytest.raises(ValueError, match=r'right_shoulder_pitch.*t=.*range'):
        validate_targets(simulation.model, BadTimeline(), .002)


@pytest.mark.parametrize('time', [float('nan'), float('inf'), -1.0])
def test_invalid_request_time(simulation, time):
    with pytest.raises(ValueError):
        simulation.advance_to(time)
    assert simulation.data.time == 0


def test_decreasing_request_time(simulation):
    simulation.advance_to(.003)
    simulation.advance_to(.003)  # repeated targets are safe, even after overshoot
    with pytest.raises(ValueError):
        simulation.advance_to(.0029)


def test_bounded_torque_and_real_tracking(simulation):
    simulation.advance_to(3)
    assert np.isfinite(simulation.data.qpos).all()
    assert np.max(np.abs(simulation.data.qvel)) > 0
    for name, settings in simulation.config['actuators'].items():
        assert abs(simulation.data.ctrl[simulation.model.actuator(name + '_motor').id]) <= settings['max_torque']
    shoulder = simulation.model.joint('right_shoulder_roll')
    assert simulation.data.qpos[shoulder.qposadr[0]] > math.radians(100)
    stats = simulation.summary()
    assert stats['steps'] > 0 and stats['warnings'] == {}
    assert stats['max_abs_torque_nm'] <= 80
    assert stats['min_base_height_m'] > .5


def test_resting_arms_do_not_chatter(simulation):
    # During preparation the arm targets are constant zero. Coarse external PD
    # previously caused a saturated +/-30 Nm elbow limit cycle every step.
    simulation.mode = 'free'
    simulation.reset()
    simulation.advance_to(.5)
    dt = simulation.model.opt.timestep
    actuator = simulation.model.actuator('right_elbow_pitch_motor').id
    torque = []
    for k in range(round(1 / dt)):
        simulation.advance_to(.5 + (k + 1) * dt)
        torque.append(float(simulation.data.ctrl[actuator]))
    torque = np.array(torque)
    assert np.mean(np.abs(torque) >= 29.7) < .05
    assert np.mean(torque[1:] * torque[:-1] < 0) < .1


def test_free_first_phrase_has_no_numerical_rolling_drift(simulation):
    # The former 2 ms limit cycle propelled the unpowered wheels over 2 m in
    # this phrase. Allow natural settling/rolling, but reject that failure.
    simulation.mode = 'free'
    simulation.reset()
    simulation.advance_to(20)
    assert simulation.summary()['max_horizontal_displacement_m'] < .1


def test_numerical_warning_is_failure(simulation):
    simulation.data.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number = 1
    with pytest.raises(RuntimeError, match='warning'):
        simulation.advance_to(.002)


@pytest.mark.parametrize(('dance_id', 'duration'), [('new_year', 1880 / 24 + 5), ('doudoulong', 85)])
def test_timeline_units_and_selection(dance_id, duration):
    timeline = build_timeline(dance_id)
    assert timeline.dance_id == dance_id
    assert timeline.sample(0)['right_hip_pitch'] == pytest.approx(math.radians(40))
    assert timeline.total_duration == pytest.approx(duration)
    with pytest.raises(ValueError, match='Unknown dance'):
        build_timeline('missing')


def test_cli_list_without_display(capsys, monkeypatch):
    monkeypatch.delenv('DISPLAY', raising=False)
    assert main(['list']) == 0
    result = capsys.readouterr().out
    assert {line.split('\t')[0] for line in result.splitlines()} == {'new_year', 'doudoulong'}


def test_cli_check_summary(capsys):
    assert main(['check', '--dance', 'doudoulong', '--mode', 'supported', '--duration', '.01']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['mode'] == 'supported'
    assert result['dance_id'] == 'doudoulong'
    assert .01 <= result['simulation_time_s'] < .012 + 1e-9
    assert result['warnings'] == {}
