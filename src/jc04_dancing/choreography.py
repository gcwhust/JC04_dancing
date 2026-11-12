"""Original Swan Lake score: phrase-led wings, pliés, glides and a reverence.

The score is authored in joint space for JC04, not copied from a dance dataset.
Each pose event joins the next with a quintic easing curve (zero velocity and
acceleration at both ends). The dense CSV is a reproducible export, with all ten
joint angles and both driven wheel velocities explicit in every row.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .motion import BODY_JOINTS, WHEEL_COLUMNS


REFERENCE_DURATION_S = 94.0
# t, event, right (pitch, roll, elbow), left (pitch, roll, elbow),
# common hip/knee offset, forward wheel speed, differential wheel speed.
# Equal hip/knee increments preserve lower-leg orientation and the support
# wheel geometry. Symmetric pliés avoid lifting an entire side of the chassis.
_POSES = (
    (0.0, 'quiet_preparation', (8, 18, 30), (8, 18, 30), 0, 0, 0),
    (3.1, 'first_wing_unfolding', (10, 62, 26), (10, 62, 26), 4, .20, 0),
    (6.2, 'open_water', (5, 106, 17), (5, 106, 17), -3, .38, 0),
    (9.3, 'soft_wing_descent', (20, 70, 38), (20, 70, 38), 7, .22, 0),
    (12.4, 'first_high_arc', (4, 128, 18), (4, 128, 18), -4, 0, 0),
    (15.5, 'glide_and_fold', (26, 52, 52), (26, 52, 52), 8, .50, 0),
    (18.6, 'long_forward_glide', (6, 116, 20), (6, 116, 20), -4, .70, 0),
    (21.7, 'gather_and_retreat', (30, 68, 48), (30, 68, 48), 8, -.48, 0),
    (24.8, 'answering_wing', (-8, 120, 20), (25, 58, 45), -2, -.30, .14),
    (28.05, 'right_curve', (-16, 138, 16), (30, 72, 50), 7, .46, .16),
    (31.3, 'right_curve_extension', (10, 114, 26), (16, 98, 32), -4, .54, .16),
    (34.55, 'wings_cross_the_phrase', (28, 54, 48), (-8, 126, 18), 8, .18, 0),
    (37.8, 'left_answer', (30, 70, 50), (-16, 138, 16), -3, -.42, -.16),
    (41.05, 'left_curve_extension', (16, 98, 32), (10, 114, 26), 6, -.52, -.16),
    (44.3, 'gather_before_flight', (30, 46, 60), (30, 46, 60), 9, 0, 0),
    (47.7, 'rising_wingbeat', (4, 124, 20), (4, 124, 20), -4, .44, 0),
    (51.1, 'flowing_right_wing', (-12, 142, 16), (22, 84, 40), 7, .56, .10),
    (54.5, 'flowing_left_wing', (22, 84, 40), (-12, 142, 16), -4, .56, -.10),
    (57.9, 'wide_swan_arc', (0, 136, 16), (0, 136, 16), 7, .66, 0),
    (61.3, 'deep_plie_before_climax', (32, 64, 54), (32, 64, 54), 9, .36, 0),
    (65.9, 'full_swan_crown', (0, 156, 14), (0, 156, 14), -5, 0, 0),
    (69.0, 'release_into_quiet', (24, 92, 42), (24, 92, 42), 6, -.24, 0),
    (70.6, 'floating_right_response', (-8, 124, 22), (26, 68, 44), -2, -.46, .12),
    (73.8, 'soft_gather', (28, 64, 48), (28, 64, 48), 7, -.26, 0),
    (77.0, 'floating_left_response', (26, 68, 44), (-8, 124, 22), -3, -.46, -.12),
    (80.2, 'last_wide_wing', (8, 116, 24), (8, 116, 24), 6, -.30, 0),
    (83.5, 'lower_the_wings', (20, 76, 38), (20, 76, 38), 0, -.16, 0),
    (86.8, 'reverence_begins', (30, 42, 55), (30, 42, 55), 8, 0, 0),
    (89.2, 'reverence', (38, 24, 62), (38, 24, 62), 9, 0, 0),
    (92.2, 'settle_at_rest', (0, 0, 0), (0, 0, 0), 0, 0, 0),
    (94.0, 'final_stillness', (0, 0, 0), (0, 0, 0), 0, 0, 0),
)

_SECTIONS = (
    ('opening_wings', '展翼', 0, 12.4, 'Both wings open through a soft plié; the wheels begin a gentle forward glide.'),
    ('glides_and_plies', '滑行与屈伸', 12.4, 24.8, 'Broad symmetric wings accompany visible leg extension, plié and a controlled retreat.'),
    ('answering_curves', '左右应答与弧线', 24.8, 44.3, 'Alternating high wings answer each other while differential wheel speeds trace two gentle curves.'),
    ('building_flight', '渐强飞翔', 44.3, 64.8, 'The wing gestures broaden over coordinated rise and sink; the forward glide grows then eases.'),
    ('swan_climax', '天鹅高展', 64.8, 67.4, 'Both wings reach their highest crown as the legs extend and the wheels come to rest.'),
    ('floating_release', '舒缓回落', 67.4, 86.8, 'Asymmetric wings soften into a retreat with gentle counter-curves and decreasing travel.'),
    ('reverence', '谢幕', 86.8, 94.0, 'A deliberate plié and gathered wings resolve to the standing pose and still wheels.'),
)


def write_swan_lake(csv_path: Path, phrase_map_path: Path, *,
                    duration_s: float = REFERENCE_DURATION_S, fps: float = 30.0) -> dict:
    """Write the score and its timing map deterministically, including the endpoint.

    Changing duration uniformly retimes the score. It does not stretch audio;
    users selecting another excerpt must also review the musical phrase anchors.
    """
    if any(not math.isfinite(value) or value <= 0 for value in (duration_s, fps)):
        raise ValueError('duration_s and fps must be finite and positive')
    scale = duration_s / REFERENCE_DURATION_S
    knots = np.array([pose[0] * scale for pose in _POSES])
    poses = np.array([
        (*right, *left, 40 + plie, 100.7787 + plie, 40 + plie, 100.7787 + plie,
         forward + turn, forward - turn)
        for _, _, right, left, plie, forward, turn in _POSES
    ], dtype=float)
    times = np.minimum(np.arange(math.ceil(duration_s * fps) + 1) / fps, duration_s)
    index = np.clip(np.searchsorted(knots, times, side='right') - 1, 0, len(knots) - 2)
    progress = (times - knots[index]) / (knots[index + 1] - knots[index])
    weight = progress**3 * (10 + progress * (-15 + 6 * progress))
    values = poses[index] + weight[:, None] * (poses[index + 1] - poses[index])
    csv_path, phrase_map_path = Path(csv_path), Path(phrase_map_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    phrase_map_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('time_s', *BODY_JOINTS, *WHEEL_COLUMNS))
        for time, row in zip(times, values):
            writer.writerow([f'{number:.9f}' for number in (time, *row)])
    metadata = {
        'schema_version': 1,
        'dance_id': 'swan_lake',
        'title': '天鹅湖 · 展翼与湖面滑行',
        'authorship': 'Original JC04 joint-space choreography, authored for this project.',
        'duration_s': duration_s,
        'fps': fps,
        'sample_count': len(times),
        'angle_units': 'degrees',
        'wheel_velocity_units': 'rad/s',
        'wheel_mode': 'driven',
        'interpolation': 'Piecewise quintic pose score; dense CSV playback uses a degree-five spline with C2 endpoints.',
        'music_timing': {
            'reference_excerpt_duration_s': REFERENCE_DURATION_S,
            'reference_source_start_s': 55.2,
            'reference_source_end_s': 149.2,
            'anchor_method': 'Acoustic envelope and spectrogram phrase candidates; not manually verified beat annotations.',
            'beat_verified': False,
            'retimed_from_reference': duration_s != REFERENCE_DURATION_S,
        },
        'leg_geometry': 'Each side keeps knee minus hip at 60.7787 degrees; common leg offsets preserve support orientation.',
        'sections': [
            {'id': key, 'name': name, 'start_s': round(start * scale, 9), 'end_s': round(end * scale, 9), 'description': description}
            for key, name, start, end, description in _SECTIONS
        ],
        'pose_events': [
            {'body_time_s': round(pose[0] * scale, 9), 'name': pose[1]}
            for pose in _POSES
        ],
    }
    phrase_map_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return metadata
