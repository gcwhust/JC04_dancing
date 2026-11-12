"""Torque driven playback with one clock: MuJoCo simulation time."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .motion import DanceTimeline, load_clip
from .resources import resource_path


def build_timeline(dance_id: str) -> DanceTimeline:
    """Build an unchanged built-in dance, including presentation transitions."""
    dances = yaml.safe_load(resource_path('configs/dances.yaml').read_text())
    robot = yaml.safe_load(resource_path('configs/robot.yaml').read_text())
    if dance_id not in dances['dances']:
        raise ValueError(f'Unknown dance: {dance_id}')
    entry = dances['dances'][dance_id]
    transition = dances['transition']
    timeline = DanceTimeline(
        load_clip(resource_path(entry['file']), entry['fps']),
        {name: math.radians(value) for name, value in robot['rest_deg'].items()},
        robot['leg_pattern'], prep=transition['prep'],
        return_time=transition['return'], hold=transition['hold'],
    )
    timeline.dance_id = dance_id
    return timeline


def validate_targets(model, timeline, timestep: float) -> None:
    """Check every physical sample and the endpoint; never clip choreography."""
    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError('timestep must be finite and positive')
    if not math.isfinite(timeline.total_duration) or timeline.total_duration < 0:
        raise ValueError('timeline duration must be finite and nonnegative')
    # advance_to samples at k*dt. Including the endpoint also checks final holds.
    for k in range(math.ceil(timeline.total_duration / timestep) + 1):
        t = min(k * timestep, timeline.total_duration)
        for name, value in timeline.sample(t).items():
            joint = model.joint(name)
            low, high = joint.range if joint.limited else (-math.inf, math.inf)
            # Dense decimal CSV samples can produce <1e-7 rad spline
            # undershoot at a zero-angle hold; this is not a playback envelope.
            if not math.isfinite(value) or value < low - 1e-7 or value > high + 1e-7:
                raise ValueError(f'{name} at t={t:.9f}s: target {value} outside range [{low}, {high}] rad')


class Simulation:
    """Bounded joint PD and wheel velocity motors with an optional base weld.

    Only reset writes joint positions. advance_to integrates to an absolute
    time, overshooting by at most one physical step, regardless of render FPS.
    Numerical warnings, MuJoCo auto resets and nonfinite states are errors.
    """

    def __init__(self, timeline, mode: str = 'free', robot_config: Path | None = None, *, wheel_mode: str | None = None):
        self.timeline = timeline
        self.mode = mode
        config_path = robot_config or resource_path('configs/robot.yaml')
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.wheel_mode = wheel_mode or getattr(timeline, 'wheel_mode', self.config['wheels']['mode'])
        self.model = mujoco.MjModel.from_xml_path(str(resource_path(self.config['model']['resource'])))
        self._range_overrides = {}
        for name, limits in self.config['model'].get('joint_range_overrides', {}).items():
            joint = self.model.joint(name)
            if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError(f'Invalid simulator joint range for {name}: {limits}')
            self._range_overrides[name] = {'source_rad': joint.range.tolist(), 'playback_rad': list(limits)}
            joint.range[:] = limits
        scene = self.config['scene']
        self.model.opt.timestep = float(scene['timestep'])
        self.model.opt.gravity[:] = scene['gravity']
        self.model.geom_friction[self.model.geom_contype != 0] = scene['friction']
        assist_friction = self.config['wheels']['assist_friction']
        if len(assist_friction) != 3 or not np.isfinite(assist_friction).all() or np.any(np.array(assist_friction) < 0):
            raise ValueError('Assist friction must contain three finite nonnegative coefficients')
        for side in ('right', 'left'):
            contact = self.model.geom(side + '_assist_contact')
            contact.friction[:] = assist_friction
            contact.priority[:] = 1
        self.model.geom('floor').size[:2] = scene.get('floor_half_size', 50.0)
        self.names = tuple(self.config['joint_order'])
        if set(self.names) != set(self.config['actuators']) or len(self.names) != 10:
            raise ValueError('joint_order and actuators must name the ten driven joints')
        self._qadr = np.array([self.model.joint(name).qposadr[0] for name in self.names])
        self._vadr = np.array([self.model.joint(name).dofadr[0] for name in self.names])
        self._act = np.array([self.model.actuator(name + '_motor').id for name in self.names])
        settings = self.config['actuators']
        self._kp, self._kd, self._limits = [np.array([settings[name][key] for name in self.names]) for key in ('kp', 'kd', 'max_torque')]
        if not np.isfinite([self._kp, self._kd, self._limits]).all() or np.any(self._kp < 0) or np.any(self._kd < 0) or np.any(self._limits <= 0):
            raise ValueError('PD gains must be finite/nonnegative and torque limits finite/positive')
        self.model.actuator_ctrlrange[self._act] = np.column_stack((-self._limits, self._limits))
        validate_targets(self.model, self.timeline, self.model.opt.timestep)
        if set(self.timeline.sample(0)) != set(self.names):
            raise ValueError('Timeline must provide exactly the ten driven joints')
        self.data = mujoco.MjData(self.model)
        self._root_adr = int(self.model.joint(self.config['model']['root_joint']).qposadr[0])
        self._support = self.model.equality(self.config['model']['support_equality']).id
        wheels = self.config['wheels']
        self._wheel_locks = np.array([self.model.equality(name).id for name in wheels['lock_equalities']])
        self._wheel_qadr = np.array([self.model.joint(name).qposadr[0] for name in wheels['joints']])
        self._wheel_vadr = np.array([self.model.joint(name).dofadr[0] for name in wheels['joints']])
        self._wheel_names = tuple(wheels['joints'])
        self._wheel_act = np.array([self.model.actuator(name + '_motor').id for name in self._wheel_names])
        self._wheel_gain = float(wheels['velocity_gain'])
        self._wheel_limit = float(wheels['max_torque'])
        self._wheel_speed_limit = float(wheels['max_speed_rad_s'])
        if not all(math.isfinite(value) and value > 0 for value in
                   (self._wheel_gain, self._wheel_limit, self._wheel_speed_limit)):
            raise ValueError('Wheel velocity gain, torque and speed limits must be finite and positive')
        self.model.actuator_ctrlrange[self._wheel_act] = [-self._wheel_limit, self._wheel_limit]
        self._base = self.model.body('robot_base').id
        self.reset()

    def reset(self) -> None:
        if self.mode not in ('free', 'supported'):
            raise ValueError('mode must be free or supported')
        if self.wheel_mode not in ('locked', 'passive', 'driven'):
            raise ValueError('wheel_mode must be locked, passive or driven')
        if self.wheel_mode == 'driven' and not callable(getattr(self.timeline, 'wheel_velocities', None)):
            raise ValueError('Driven wheel mode requires timeline.wheel_velocities')
        key = self.model.key(self.config['model']['initial_keyframe']).id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        self.data.eq_active[self._support] = self.mode == 'supported'
        self.data.eq_active[self._wheel_locks] = self.wheel_mode == 'locked'
        mujoco.mj_forward(self.model, self.data)
        self._requested_time = 0.0
        self._steps = 0
        self._max_torque = 0.0
        self._min_height = float(self.data.xpos[self._base, 2])
        self._max_tilt = 0.0
        self._min_contact = 0.0
        self._max_error = 0.0
        self._initial_xy = self.data.qpos[self._root_adr:self._root_adr + 2].copy()
        self._max_horizontal_displacement = 0.0
        self._initial_wheel_angles = self.data.qpos[self._wheel_qadr].copy()
        self._max_wheel_rotation = 0.0
        self._max_wheel_speed = 0.0
        self._max_wheel_torque = 0.0
        self._max_wheel_speed_error = 0.0
        rotation = self.data.xmat[self._base].reshape(3, 3)
        self._previous_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        self._unwrapped_yaw = self._min_yaw = self._max_yaw = 0.0
        self._observe()

    def _check_state(self, previous_time: float | None = None) -> None:
        warnings = self._warnings()
        if warnings:
            raise RuntimeError(f'MuJoCo warning at t={self.data.time:.9f}s: {warnings}')
        if not all(np.isfinite(array).all() for array in (self.data.qpos, self.data.qvel, self.data.qacc, self.data.ctrl)):
            raise RuntimeError(f'Nonfinite MuJoCo state at t={self.data.time}s')
        if not math.isfinite(self.data.time) or (previous_time is not None and self.data.time <= previous_time):
            raise RuntimeError('MuJoCo time failed to advance (possible automatic reset)')

    def _warnings(self):
        return {mujoco.mjtWarning(i).name: int(warning.number) for i, warning in enumerate(self.data.warning) if warning.number}

    def _observe(self) -> None:
        self._max_wheel_rotation = max(self._max_wheel_rotation, float(np.max(np.abs(self.data.qpos[self._wheel_qadr] - self._initial_wheel_angles))))
        self._max_wheel_speed = max(self._max_wheel_speed, float(np.max(np.abs(self.data.qvel[self._wheel_vadr]))))
        displacement = self.data.qpos[self._root_adr:self._root_adr + 2] - self._initial_xy
        self._max_horizontal_displacement = max(self._max_horizontal_displacement, float(np.linalg.norm(displacement)))
        self._min_height = min(self._min_height, float(self.data.xpos[self._base, 2]))
        rotation = self.data.xmat[self._base].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        delta = yaw - self._previous_yaw
        self._unwrapped_yaw += math.atan2(math.sin(delta), math.cos(delta))
        self._previous_yaw = yaw
        self._min_yaw = min(self._min_yaw, self._unwrapped_yaw)
        self._max_yaw = max(self._max_yaw, self._unwrapped_yaw)
        upright = rotation[2, 2]
        self._max_tilt = max(self._max_tilt, math.degrees(math.acos(float(np.clip(upright, -1, 1)))))
        if self.data.ncon:
            self._min_contact = min(self._min_contact, float(np.min(self.data.contact.dist)))

    def advance_to(self, t: float) -> None:
        t = float(t)
        if not math.isfinite(t) or t < self._requested_time:
            raise ValueError('Requested time must be finite, nonnegative and nondecreasing')
        self._check_state()
        while self.data.time < t:
            now = float(self.data.time)
            sample = self.timeline.sample(now)
            target = np.array([sample[name] for name in self.names])
            if not np.isfinite(target).all():
                raise RuntimeError(f'Nonfinite target at t={now}s')
            error = target - self.data.qpos[self._qadr]
            torque = np.clip(self._kp * error - self._kd * self.data.qvel[self._vadr], -self._limits, self._limits)
            self.data.ctrl[self._act] = torque
            wheel_torque = np.zeros(2)
            if self.wheel_mode == 'driven':
                command = self.timeline.wheel_velocities(now)
                if set(command) != set(self._wheel_names):
                    raise ValueError('Wheel commands must provide exactly the two driven wheel joints')
                wheel_target = np.array([command[name] for name in self._wheel_names])
                if not np.isfinite(wheel_target).all() or np.any(np.abs(wheel_target) > self._wheel_speed_limit):
                    raise ValueError(f'wheel velocities at t={now}s must be finite and within +/-{self._wheel_speed_limit} rad/s')
                wheel_error = wheel_target - self.data.qvel[self._wheel_vadr]
                wheel_torque = np.clip(self._wheel_gain * wheel_error, -self._wheel_limit, self._wheel_limit)
                self._max_wheel_speed_error = max(self._max_wheel_speed_error, float(np.max(np.abs(wheel_error))))
            self.data.ctrl[self._wheel_act] = wheel_torque
            self._max_wheel_torque = max(self._max_wheel_torque, float(np.max(np.abs(wheel_torque))))
            self._max_torque = max(self._max_torque, float(np.max(np.abs(torque))), self._max_wheel_torque)
            self._max_error = max(self._max_error, float(np.max(np.abs(error))))
            mujoco.mj_step(self.model, self.data)
            self._check_state(now)
            self._steps += 1
            self._observe()
        self._requested_time = t

    def summary(self) -> dict:
        return {
            'dance_id': getattr(self.timeline, 'dance_id', None), 'mode': self.mode,
            'wheel_mode': self.wheel_mode,
            'max_wheel_rotation_rad': self._max_wheel_rotation,
            'max_wheel_speed_rad_s': self._max_wheel_speed,
            'max_wheel_torque_nm': self._max_wheel_torque,
            'max_wheel_speed_error_rad_s': self._max_wheel_speed_error,
            'timeline_duration_s': self.timeline.total_duration,
            'requested_time_s': self._requested_time, 'simulation_time_s': float(self.data.time),
            'timestep_s': float(self.model.opt.timestep), 'steps': self._steps,
            'max_abs_torque_nm': self._max_torque, 'max_joint_error_rad': self._max_error,
            'min_base_height_m': self._min_height, 'max_base_tilt_deg': self._max_tilt,
            'base_yaw_excursion_deg': math.degrees(self._max_yaw - self._min_yaw),
            'min_contact_distance_m': self._min_contact,
            'max_horizontal_displacement_m': self._max_horizontal_displacement,
            'final_base_position_m': self.data.qpos[self._root_adr:self._root_adr + 3].tolist(),
            'fallen': self._min_height < .35 or self._max_tilt > 45,
            'joint_range_overrides': self._range_overrides,
            'warnings': self._warnings(),
        }
