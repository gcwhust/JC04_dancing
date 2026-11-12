"""Sample the original choreography from simulation time, without a wall clock.

CSV angles are degrees; all public pose dictionaries contain radians. ``duration``
means the final absolute frame/fps timestamp, including any leading first-frame
hold (the doudoulong clip starts at frame 1). No frame numbers are rebased.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d

ARM_JOINTS = (
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow_pitch',
)


@dataclass(frozen=True)
class DanceClip:
    """Cubic arm trajectory with constant poses outside the recorded range."""

    joint_names: tuple[str, ...]
    start: float
    duration: float
    _interpolator: Callable = field(repr=False)

    def sample(self, t: float) -> dict[str, float]:
        angles = self._interpolator(float(t))
        return {name: math.radians(float(value)) for name, value in zip(self.joint_names, angles)}


def load_clip(csv_path: Path, fps: float = 24.0) -> DanceClip:
    """Read six named arm columns and preserve their source frame/fps times."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        required = ('frame', *ARM_JOINTS)
        if len(columns) != len(set(columns)) or not set(required) <= set(columns):
            raise ValueError('CSV must contain unique frame and six named arm columns')
        try:
            rows = [[float(row[name]) for name in required] for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError('CSV frames and arm angles must be finite numbers') from exc
    data = np.asarray(rows, dtype=float)
    if len(rows) < 4:
        raise ValueError('Cubic interpolation requires at least four rows')
    if not np.all(np.isfinite(data)):
        raise ValueError('CSV frames and arm angles must be finite numbers')
    if not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError('CSV frames must be strictly increasing')
    times = data[:, 0] / fps
    angles = data[:, 1:]
    interpolator = interp1d(
        times, angles, axis=0, kind='cubic', assume_sorted=True,
        bounds_error=False, fill_value=(angles[0], angles[-1]),
    )
    return DanceClip(ARM_JOINTS, float(times[0]), float(times[-1]), interpolator)


class DanceTimeline:
    """Add smooth presentation transitions around an unchanged dance body.

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
        self.total_duration = prep + clip.duration + return_time + hold
        self._first = self._body(0.0)
        self._last = self._body(clip.duration)

    def _body(self, t: float) -> dict[str, float]:
        result = self.clip.sample(t)
        pattern = self.leg_pattern
        omega = 2 * math.pi * pattern['frequency_hz']
        for side in ('right', 'left'):
            wave = math.sin(omega * t + pattern[f'{side}_phase_rad'])
            for joint in ('hip', 'knee'):
                name = f'{side}_{joint}_pitch'
                result[name] = self.rest[name] + math.radians(pattern[f'{joint}_amplitude_deg']) * wave
        return result

    @staticmethod
    def _blend(start: dict[str, float], end: dict[str, float], s: float) -> dict[str, float]:
        weight = s * s * (3 - 2 * s)
        return {name: value + weight * (end[name] - value) for name, value in start.items()}

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
