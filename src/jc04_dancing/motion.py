"""Sample coordinated poses and wheel speeds from simulation time.

CSV angles are degrees and wheel velocities are rad/s. Public poses are radians.
New full-body clips use explicit seconds and C2 endpoint conditions. Legacy
six-arm clips retain their cubic interpolation and absolute frame/fps timing.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d, make_interp_spline

ARM_JOINTS = (
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow_pitch',
)
LEG_JOINTS = ('right_hip_pitch', 'right_knee_pitch', 'left_hip_pitch', 'left_knee_pitch')
BODY_JOINTS = ARM_JOINTS + LEG_JOINTS
WHEEL_JOINTS = ('right_wheel_joint', 'left_wheel_joint')
WHEEL_COLUMNS = ('right_wheel_velocity_rad_s', 'left_wheel_velocity_rad_s')


@dataclass(frozen=True)
class DanceClip:
    """Named pose trajectory with constant poses outside the recorded range."""

    joint_names: tuple[str, ...]
    start: float
    duration: float
    _interpolator: Callable = field(repr=False)
    _wheel_interpolator: Callable | None = field(default=None, repr=False)

    @property
    def full_body(self) -> bool:
        return self.joint_names == BODY_JOINTS

    @property
    def interpolation(self) -> str:
        return 'quintic' if self.full_body else 'cubic'

    def sample(self, t: float) -> dict[str, float]:
        angles = self._interpolator(min(max(float(t), self.start), self.duration))
        return {name: math.radians(float(value)) for name, value in zip(self.joint_names, angles)}

    def wheel_velocities(self, t: float) -> dict[str, float]:
        if self._wheel_interpolator is None or t < self.start or t > self.duration:
            return dict.fromkeys(WHEEL_JOINTS, 0.0)
        return dict(zip(WHEEL_JOINTS, map(float, self._wheel_interpolator(float(t)))))


def load_clip(csv_path: Path, fps: float = 24.0) -> DanceClip:
    """Read explicit full-body seconds or legacy six-arm frame/fps samples.

Full-body files must supply all four leg angles and both wheel velocities;
silently inventing missing motion would hide an incomplete choreography export.
    Their clock starts at zero and wheels stop at both endpoints so the outer
    preparation/return can join the recorded body without a velocity jump.
"""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        full_body = bool(set(columns) & set(LEG_JOINTS + WHEEL_COLUMNS))
        if full_body and not set(BODY_JOINTS + WHEEL_COLUMNS) <= set(columns):
            raise ValueError('Full-body CSV requires complete ten-joint and two-wheel channels')
        clock_column = 'time_s' if 'time_s' in columns else 'frame'
        joints = BODY_JOINTS if full_body else ARM_JOINTS
        required = (clock_column, *joints, *(WHEEL_COLUMNS if full_body else ()))
        if len(columns) != len(set(columns)) or not set(required) <= set(columns):
            raise ValueError('CSV must contain unique time_s/frame and named joint columns')
        try:
            rows = [[float(row[name]) for name in required] for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError('CSV times, angles and wheel speeds must be finite numbers') from exc
    data = np.asarray(rows, dtype=float)
    if len(rows) < 4:
        raise ValueError('Cubic interpolation requires at least four rows')
    if not np.all(np.isfinite(data)):
        raise ValueError('CSV times, angles and wheel speeds must be finite numbers')
    if not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError('CSV frames must be strictly increasing')
    times = data[:, 0] if clock_column == 'time_s' else data[:, 0] / fps
    angles = data[:, 1:1 + len(joints)]
    if full_body:
        if times[0] != 0:
            raise ValueError('Full-body CSV time must start at zero')
        if np.any(data[[0, -1], -len(WHEEL_COLUMNS):] != 0):
            raise ValueError('Full-body CSV requires zero wheel velocities at both endpoints')
        # Dense exports retain smooth motion between samples, including C2 joins
        # to the preparation/return. A zero derivative is never imposed per row.
        values = data[:, 1:]
        zeros = np.zeros(values.shape[1])
        spline = make_interp_spline(times, values, k=5,
                                    bc_type=([(1, zeros), (2, zeros)], [(1, zeros), (2, zeros)]))
        return DanceClip(joints, float(times[0]), float(times[-1]),
                         lambda t: spline(t)[:len(joints)], lambda t: spline(t)[len(joints):])
    interpolator = interp1d(
        times, angles, axis=0, kind='cubic', assume_sorted=True,
        bounds_error=False, fill_value=(angles[0], angles[-1]),
    )
    return DanceClip(joints, float(times[0]), float(times[-1]), interpolator)


class DanceTimeline:
    """Add presentation transitions around the recorded choreography.

    ``rest`` contains ten named targets in radians; ``leg_pattern`` uses the
    units in robot.yaml (degree amplitudes, Hz frequency, radian phase offsets).
    The deployed planner has zero forward bias and zero knee phase lead.
    """

    def __init__(
        self, clip: DanceClip, rest: dict[str, float], leg_pattern: dict,
        prep: float = 2.0, return_time: float = 2.0, hold: float = 1.0,
    ) -> None:
        if not all(math.isfinite(value) for value in (prep, return_time, hold)):
            raise ValueError('Transition times must be finite')
        if prep <= 0 or return_time <= 0 or hold < 0:
            raise ValueError('Preparation/return must be positive; hold must be nonnegative')
        self.clip = clip
        self.rest = dict(rest)
        self.leg_pattern = dict(leg_pattern)
        self.prep = prep
        self.return_time = return_time
        self.hold = hold
        self.wheel_mode = 'driven' if clip.full_body else 'locked'
        self.total_duration = prep + clip.duration + return_time + hold
        self._first = self._body(0.0)
        self._last = self._body(clip.duration)

    def _body(self, t: float) -> dict[str, float]:
        result = self.clip.sample(t)
        if self.clip.full_body:
            return result
        pattern = self.leg_pattern
        omega = 2 * math.pi * pattern['frequency_hz']
        for side in ('right', 'left'):
            wave = math.sin(omega * t + pattern[f'{side}_phase_rad'])
            for joint in ('hip', 'knee'):
                name = f'{side}_{joint}_pitch'
                result[name] = self.rest[name] + math.radians(pattern[f'{joint}_amplitude_deg']) * wave
        return result

    def _blend(self, start: dict[str, float], end: dict[str, float], s: float) -> dict[str, float]:
        weight = s**3 * (10 + s * (-15 + 6 * s)) if self.clip.full_body else s * s * (3 - 2 * s)
        return {name: value + weight * (end[name] - value) for name, value in start.items()}

    def wheel_velocities(self, t: float) -> dict[str, float]:
        """Wheel velocity targets in rad/s, stopped before and after the body."""
        body_time = float(t) - self.prep
        if body_time < 0 or body_time > self.clip.duration:
            return dict.fromkeys(WHEEL_JOINTS, 0.0)
        return self.clip.wheel_velocities(body_time)

    def sample(self, t: float) -> dict[str, float]:
        """Return all ten targets; repeated or out-of-order calls are equivalent."""
        if t <= 0:
            return dict(self.rest)
        if t < self.prep:
            return self._blend(self.rest, self._first, t / self.prep)
        body_time = t - self.prep
        if body_time <= self.clip.duration:
            return self._body(body_time)
        return_elapsed = body_time - self.clip.duration
        if return_elapsed < self.return_time:
            return self._blend(self._last, self.rest, return_elapsed / self.return_time)
        return dict(self.rest)
