"""Sample coordinated poses and wheel speeds from simulation time.

CSV angles are degrees and wheel velocities are rad/s. Public poses are radians.
Full-body clips use explicit seconds and C2 endpoint conditions.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import make_interp_spline

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
    _wheel_interpolator: Callable = field(repr=False)

    @property
    def full_body(self) -> bool:
        return self.joint_names == BODY_JOINTS

    @property
    def interpolation(self) -> str:
        return 'quintic'

    def sample(self, t: float) -> dict[str, float]:
        angles = self._interpolator(min(max(float(t), self.start), self.duration))
        return {name: math.radians(float(value)) for name, value in zip(self.joint_names, angles)}

    def wheel_velocities(self, t: float) -> dict[str, float]:
        if t < self.start or t > self.duration:
            return dict.fromkeys(WHEEL_JOINTS, 0.0)
        return dict(zip(WHEEL_JOINTS, map(float, self._wheel_interpolator(float(t)))))


def load_clip(csv_path: Path) -> DanceClip:
    """Read complete body and wheel channels on an explicit seconds clock.

    Files must supply all ten joint angles and both wheel velocities;
    missing channels indicate an incomplete choreography export.
    Their clock starts at zero and wheels stop at both endpoints so the outer
    preparation/return can join the recorded body without a velocity jump.
    """
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if not set(BODY_JOINTS + WHEEL_COLUMNS) <= set(columns):
            raise ValueError('Full-body CSV requires complete ten-joint and two-wheel channels')
        required = ('time_s', *BODY_JOINTS, *WHEEL_COLUMNS)
        if len(columns) != len(set(columns)) or not set(required) <= set(columns):
            raise ValueError('CSV must contain unique time_s and named joint columns')
        try:
            rows = [[float(row[name]) for name in required] for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError('CSV times, angles and wheel speeds must be finite numbers') from exc
    data = np.asarray(rows, dtype=float)
    if len(rows) < 4:
        raise ValueError('Full-body interpolation requires at least four rows')
    if not np.all(np.isfinite(data)):
        raise ValueError('CSV times, angles and wheel speeds must be finite numbers')
    if not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError('CSV times must be strictly increasing')
    times = data[:, 0]
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
    return DanceClip(BODY_JOINTS, float(times[0]), float(times[-1]),
                     lambda t: spline(t)[:len(BODY_JOINTS)], lambda t: spline(t)[len(BODY_JOINTS):])


class DanceTimeline:
    """Add presentation transitions around the recorded choreography.

    ``rest`` contains ten named targets in radians. The clip supplies every
    body joint and wheel target during the authored choreography.
    """

    def __init__(
        self, clip: DanceClip, rest: dict[str, float],
        prep: float = 2.0, return_time: float = 2.0, hold: float = 1.0,
    ) -> None:
        if not all(math.isfinite(value) for value in (prep, return_time, hold)):
            raise ValueError('Transition times must be finite')
        if prep <= 0 or return_time <= 0 or hold < 0:
            raise ValueError('Preparation/return must be positive; hold must be nonnegative')
        self.clip = clip
        self.rest = dict(rest)
        self.prep = prep
        self.return_time = return_time
        self.hold = hold
        self.wheel_mode = 'driven'
        self.total_duration = prep + clip.duration + return_time + hold
        self._first = self.clip.sample(0.0)
        self._last = self.clip.sample(clip.duration)

    def _blend(self, start: dict[str, float], end: dict[str, float], s: float) -> dict[str, float]:
        weight = s**3 * (10 + s * (-15 + 6 * s))
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
            return self.clip.sample(body_time)
        return_elapsed = body_time - self.clip.duration
        if return_elapsed < self.return_time:
            return self._blend(self._last, self.rest, return_elapsed / self.return_time)
        return dict(self.rest)
