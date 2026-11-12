"""Original Swan Lake score: unfolding wings, reaches, diagonals and reverence.

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
# Pitch changes give the wings depth: low scoops unfold into forward reaches,
# then trade places with high lateral diagonals. Elbows fold between extensions
# instead of retaining a single wing silhouette throughout the music.
_POSES = (
    (0.0, 'quiet_preparation', (8, 18, 30), (8, 18, 30), 0, 0, 0),
    (3.1, 'low_water_scoop', (24, 44, 100), (24, 44, 100), 12, .15, 0),
    (6.2, 'first_forward_reach', (82, 28, 12), (82, 28, 12), -10, .55, 0),
    (9.3, 'folded_wing_opening', (12, 72, 80), (12, 72, 80), 16, .25, 0),
    (12.4, 'first_high_arc', (-22, 142, 10), (-22, 142, 10), -12, 0, 0),
    (15.5, 'gliding_scoop_and_fold', (62, 30, 94), (62, 30, 94), 18, .60, 0),
    (18.6, 'long_forward_glide', (108, 38, 8), (108, 38, 8), -14, 1.10, 0),
    (21.7, 'sweeping_wings_retreat', (-36, 104, 42), (-36, 104, 42), 16, -.95, 0),
    (24.8, 'right_high_left_front', (-42, 148, 12), (94, 34, 18), -8, -.50, .30),
    (28.05, 'right_crown_left_fold', (-26, 156, 18), (42, 52, 96), 16, .80, .34),
    (31.3, 'right_reach_left_open', (84, 42, 14), (14, 118, 64), -12, .95, .32),
    (34.55, 'left_wing_unfolding', (30, 48, 98), (-30, 142, 14), 18, .30, 0),
    (37.8, 'left_high_right_front', (94, 34, 18), (-42, 148, 12), -10, -.80, -.34),
    (41.05, 'left_reach_right_open', (14, 118, 64), (84, 42, 14), 14, -.95, -.32),
    (44.3, 'deep_gather_before_flight', (24, 44, 100), (24, 44, 100), 20, 0, 0),
    (47.7, 'rising_forward_wings', (108, 38, 8), (108, 38, 8), -12, .75, 0),
    (51.1, 'flowing_right_diagonal', (-34, 154, 10), (92, 42, 26), 14, 1.05, .25),
    (54.5, 'flowing_left_diagonal', (92, 42, 26), (-34, 154, 10), -12, 1.05, -.25),
    (57.9, 'wide_swan_arc', (-18, 144, 8), (-18, 144, 8), 12, 1.15, 0),
    (61.3, 'deep_plie_before_climax', (36, 58, 94), (36, 58, 94), 20, .50, 0),
    (65.9, 'full_swan_crown', (0, 165, 14), (0, 165, 14), -14, 0, 0),
    (69.0, 'release_into_quiet', (18, 112, 52), (18, 112, 52), 12, -.40, 0),
    (70.6, 'floating_right_response', (-28, 144, 18), (82, 42, 18), 0, -.90, .28),
    (73.8, 'soft_water_gather', (24, 44, 100), (24, 44, 100), 16, -.55, 0),
    (77.0, 'floating_left_response', (82, 42, 18), (-28, 144, 18), -10, -.90, -.28),
    (80.2, 'last_forward_offering', (98, 38, 10), (98, 38, 10), 12, -.65, 0),
    (83.5, 'last_open_wings', (0, 122, 28), (0, 122, 28), -6, -.25, -.24),
    (86.8, 'reverence_begins', (24, 44, 92), (24, 44, 92), 14, 0, 0),
    (89.2, 'deep_reverence', (42, 22, 52), (42, 22, 52), 20, 0, 0),
    (92.2, 'settle_at_rest', (0, 0, 0), (0, 0, 0), 0, 0, 0),
    (94.0, 'final_stillness', (0, 0, 0), (0, 0, 0), 0, 0, 0),
)

_SECTIONS = (
    ('opening_wings', '展翼', 0, 12.4, 'Low scoops unfold into long forward reaches, then folded wings open into the first high arc over a broad plié.'),
    ('glides_and_plies', '滑行与屈伸', 12.4, 24.8, 'Gathered elbows open into high forward reaches; deep pliés alternate with extension during a long glide and sweeping retreat.'),
    ('answering_curves', '左右应答与弧线', 24.8, 44.3, 'One wing crowns while the other reaches forward or folds; the diagonals exchange through two broad wheel-driven curves.'),
    ('building_flight', '渐强飞翔', 44.3, 64.8, 'A deep gather releases into forward flight, alternating high diagonals and wide wings; a final plié prepares the crown.'),
    ('swan_climax', '天鹅高展', 64.8, 67.4, 'Both wings reach their highest overhead crown as the legs extend and the wheels come to rest.'),
    ('floating_release', '舒缓回落', 67.4, 86.8, 'High and forward wings answer across retreating counter-curves, gathering once more before a final forward offering and open-wing release.'),
    ('reverence', '谢幕', 86.8, 94.0, 'Gathered elbows and forward arms deepen into a sustained plié, then lower together into the standing pose and final stillness.'),
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
