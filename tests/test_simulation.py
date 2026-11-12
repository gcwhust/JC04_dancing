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


def test_driven_wheels_roll_floating_robot_with_bounded_torque(simulation):
    # A motor mapping error, retained brake, or fused assist wheel stops this glide.
    original = simulation.timeline

    class Glide:
        total_duration = 4.0
        wheel_mode = 'driven'

        def sample(self, t):
            return dict(original.rest)

        def wheel_velocities(self, t):
            speed = .6 * min(1.0, t / 2)
            return {'right_wheel_joint': speed, 'left_wheel_joint': speed}

    try:
        simulation.timeline = Glide()
        simulation.wheel_mode = 'driven'
        simulation.mode = 'free'
        simulation.reset()
        simulation.advance_to(4)
        stats = simulation.summary()
        assert simulation.model.nu == 12
        assert stats['final_base_position_m'][0] > .09
        assert stats['max_base_tilt_deg'] < 8
        assert not stats['fallen'] and stats['warnings'] == {}
        assert stats['max_wheel_torque_nm'] <= simulation.config['wheels']['max_torque']
        assert stats['max_wheel_rotation_rad'] > .8
        for side in ('right', 'left'):
            assert not simulation.data.eq_active[simulation.model.equality(side + '_wheel_lock').id]
            assist = simulation.model.joint(side + '_assist')
            assert abs(simulation.data.qpos[assist.qposadr[0]]) > .8
    finally:
        simulation.timeline = original


def test_differential_wheels_physically_turn_the_floating_base(simulation):
    # Fused/high-friction assist contacts previously swallowed differential
    # commands: the wheels turned but the chassis yawed less than one degree.
    original = simulation.timeline

    class Arc:
        total_duration = 8.0

        def sample(self, t):
            return dict(original.rest)

        def wheel_velocities(self, t):
            ramp = .5 - .5 * math.cos(math.pi * min(t / 2, 1))
            return {'right_wheel_joint': .7 * ramp, 'left_wheel_joint': .3 * ramp}

    try:
        simulation.timeline = Arc()
        simulation.wheel_mode = 'driven'
        simulation.mode = 'free'
        simulation.reset()
        simulation.advance_to(8)
        w, x, y, z = simulation.data.qpos[3:7]
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        assert 10 < yaw < 65
        assert simulation.summary()['base_yaw_excursion_deg'] == pytest.approx(yaw, abs=.1)
        assert simulation.summary()['max_base_tilt_deg'] < 5
        assert not simulation.summary()['fallen']
        assert not simulation.data.eq_active[simulation.model.equality('base_support').id]
    finally:
        simulation.timeline = original


def test_driven_wheel_commands_must_be_finite_and_complete(simulation):
    original = simulation.timeline

    class InvalidWheelCommand:
        def sample(self, t):
            return original.sample(t)

        def wheel_velocities(self, t):
            return {'right_wheel_joint': float('nan'), 'left_wheel_joint': 0.0}

    try:
        simulation.timeline = InvalidWheelCommand()
        simulation.wheel_mode = 'driven'
        simulation.reset()
        with pytest.raises(ValueError, match='wheel.*finite'):
            simulation.advance_to(.001)
        assert simulation.data.time == 0
    finally:
        simulation.timeline = original


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


def test_swan_lake_preserves_source_joint_limits(simulation):
    # New choreography must fit the source limits without widening the model.
    assert simulation.summary()['joint_range_overrides'] == {}
    for side in ('right', 'left'):
        np.testing.assert_array_equal(simulation.model.joint(side + '_shoulder_roll').range, [0, 3.14])
        np.testing.assert_array_equal(simulation.model.joint(side + '_elbow_pitch').range, [-1.5708, 1.85])
    validate_targets(simulation.model, simulation.timeline, .01)


@pytest.mark.parametrize(('value', 'allowed'), [(-5e-8, True), (-5e-5, False)])
def test_target_validation_distinguishes_spline_roundoff_from_limit_violation(value, allowed):
    model = mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody><body><joint name="right_shoulder_roll" range="0 3.14"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')

    class TinyTrajectory:
        total_duration = .001

        def sample(self, t):
            return {'right_shoulder_roll': value}

    if allowed:
        validate_targets(model, TinyTrajectory(), .0005)
    else:
        with pytest.raises(ValueError, match='outside range'):
            validate_targets(model, TinyTrajectory(), .0005)


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
    target = simulation.timeline.sample(simulation.data.time)['right_shoulder_roll']
    assert simulation.data.qpos[shoulder.qposadr[0]] == pytest.approx(target, abs=.12)
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


def test_swan_lake_timeline_units_and_selection():
    timeline = build_timeline('swan_lake')
    assert timeline.dance_id == 'swan_lake'
    assert timeline.sample(0)['right_hip_pitch'] == pytest.approx(math.radians(40))
    assert timeline.total_duration == pytest.approx(99)
    with pytest.raises(ValueError, match='Unknown dance'):
        build_timeline('missing')


def test_cli_list_without_display(capsys, monkeypatch):
    monkeypatch.delenv('DISPLAY', raising=False)
    assert main(['list']) == 0
    result = capsys.readouterr().out
    assert {line.split('\t')[0] for line in result.splitlines()} == {'swan_lake'}


def test_cli_check_summary(capsys):
    assert main(['check', '--dance', 'swan_lake', '--mode', 'supported', '--duration', '.01']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['mode'] == 'supported'
    assert result['dance_id'] == 'swan_lake'
    assert .01 <= result['simulation_time_s'] < .012 + 1e-9
    assert result['warnings'] == {}
