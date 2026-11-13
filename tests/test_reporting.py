"""Reports must be readable Chinese artifacts backed by actual encoded evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path

import fitz
import imageio_ffmpeg
import numpy as np
from PIL import Image
import pytest
import yaml

from jc04_dancing.resources import resource_path
from jc04_dancing.simulation import build_timeline


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def run_dir(tmp_path_factory):
    """Small, encoded video with the real recorder schema; no production outputs required."""
    root = tmp_path_factory.mktemp('report-run')
    (root / 'stills').mkdir()
    timeline = build_timeline('swan_lake')
    dances = yaml.safe_load(resource_path('configs/dances.yaml').read_text())
    dance = dances['dances']['swan_lake']
    phrases_path = dance['phrases']
    phrases = json.loads(resource_path(phrases_path).read_text())
    duration, fps = timeline.total_duration, 2
    count = math.ceil(duration * fps)
    writer = imageio_ffmpeg.write_frames(str(root / 'dance.mp4'), (64, 64), fps=fps,
                                        codec='libx264', macro_block_size=1)
    writer.send(None)
    x, y = np.meshgrid(np.arange(64), np.arange(64))
    for index in range(count):
        pixels = np.stack([(x * 4 + index) % 256, y * 4, (x + y) * 2], axis=2).astype(np.uint8)
        writer.send(pixels.tobytes())
    writer.close()
    samples = {0: 'start', count - 1: 'end'}
    for section in phrases['sections']:
        sample_time = timeline.prep + (section['start_s'] + section['end_s']) / 2
        samples[round(sample_time * fps)] = section['id']
    reader = imageio_ffmpeg.read_frames(str(root / 'dance.mp4'))
    next(reader)
    stills = []
    for index, frame in enumerate(reader):
        if index in samples:
            name = f'stills/{samples[index]}.png'
            Image.frombytes('RGB', (64, 64), frame).save(root / name)
            stills.append({'label': samples[index], 'frame': index, 'sample_time_s': index / fps,
                           'file': name, 'actual_driven_joint_rad': {'right_elbow_pitch': .1}})
    config = yaml.safe_load(resource_path('configs/robot.yaml').read_text())
    resources = ['configs/robot.yaml', 'configs/dances.yaml', 'assets/provenance.json',
                 'assets/robot/robot.xml', dance['file'], phrases_path]
    data = {
        'schema_version': 1, 'status': 'complete', 'mode': 'free', 'preview': False,
        'timeline_complete': True,
        'audio': {'present': False, 'status': 'postproduction_pending', 'D04_verified': False},
        'video': {'file': 'dance.mp4', 'width': 64, 'height': 64, 'fps': fps,
                  'requested_duration_s': duration, 'encoded_duration_s': count / fps,
                  'expected_frames': count, 'frames_written': count, 'decoded_frames': count,
                  'decoded_fps': fps, 'decoded_size': [64, 64], 'decoded_codec': 'h264',
                  'audio_stream_present': False, 'sha256': sha(root / 'dance.mp4')},
        'config': config, 'versions': {'python': '3.10.22', 'mujoco': '3.3.7', 'numpy': '2.2.6',
                                     'Pillow': '12.0.0', 'imageio-ffmpeg': '0.6.0', 'ffmpeg': 'test fixture'},
        'render_backend': 'egl', 'camera': {'physics_recentered': False, 'follow': 'physical_base_xy'},
        'simulation': {'dance_id': 'swan_lake', 'mode': 'free', 'wheel_mode': 'driven',
                       'timeline_duration_s': duration, 'requested_time_s': duration,
                       'simulation_time_s': math.ceil(duration / .0005) * .0005,
                       'timestep_s': .0005, 'steps': math.ceil(duration / .0005),
                       'max_abs_torque_nm': 18.8274, 'max_joint_error_rad': .3047,
                       'min_base_height_m': .5161, 'max_base_tilt_deg': 7.408,
                       'min_contact_distance_m': -.000776, 'max_horizontal_displacement_m': .0333926,
                       'max_wheel_rotation_rad': 1.8478e-6, 'max_wheel_speed_rad_s': .00011658,
                       'fallen': False, 'warnings': {},
                       'joint_range_overrides': {joint: {'source_rad': [0, 3.14], 'playback_rad': [-.18, 3.14]}
                                                 for joint in ['right_shoulder_roll', 'left_shoulder_roll']}},
        'evidence': {'resource_sha256': {name: sha(resource_path(name)) for name in resources},
                     'code_sha256': {'cli.py': 'a' * 64},
                     'motion': {'source': dance['file'], 'source_fps': dance['fps'],
                                'original_bytes_unchanged': False, 'generated_choreography': True,
                                'clip_duration_s': timeline.clip.duration,
                                'clip_start_s': timeline.clip.start, 'prep_s': timeline.prep,
                                'return_s': timeline.return_time, 'hold_s': timeline.hold,
                                'interpolation': 'quintic', 'wheel_units': 'rad/s',
                                'phrases_source': phrases_path}},
        'stills': stills,
    }
    (root / 'run.json').write_text(json.dumps(data))
    return root


@pytest.fixture(scope='module')
def rendered_report(run_dir, tmp_path_factory):
    from jc04_dancing.reporting import build_report
    root = tmp_path_factory.mktemp('report-result')
    metadata = build_report(run_dir, root / 'report.pdf')
    return root, metadata


def text_of(path):
    with fitz.open(path) as document:
        return '\n'.join(page.get_text() for page in document)


@pytest.mark.parametrize('filename', ['report.pdf', 'flowchart.pdf'])
def test_pdf_project_links_are_clickable_and_cover_their_labels(rendered_report, filename):
    root, _ = rendered_report
    destinations = {
        'https://gcwhust.github.io/JC04_dancing/': '项目主页',
        'https://github.com/gcwhust/JC04_dancing': '开源代码',
    }
    with fitz.open(root / filename) as document:
        for page in document:
            links = page.get_links()
            assert {link.get('uri') for link in links} == set(destinations)
            for link in links:
                rectangle = link['from']
                assert not rectangle.is_empty and page.rect.contains(rectangle)
                labels = page.search_for(destinations[link['uri']])
                assert any(rectangle.intersects(label) for label in labels)


def test_report_sections(rendered_report, run_dir):
    root, metadata = rendered_report
    text = text_of(root / 'report.pdf')
    for section in ['一、小楚机器人', '二、仿真平台与机器人模型', '三、操作流程']:
        assert section in text
    for detail in ['小楚', 'JC04', 'MuJoCo', 'jc04_desc4', 'V1.3', '3.3.7', '配乐合成']:
        assert detail in text
    for private in ['天鹅湖', run_dir.name, '证据版本', 'SHA256', 'D04',
                    'source_snapshot', '0.0005', 'max_horizontal', '后期验证', '无声']:
        assert private not in text
    assert metadata['schema_version'] == 1
    assert metadata['source']['run_sha256'] == sha(run_dir / 'run.json')
    assert metadata['source']['video_sha256'] == sha(run_dir / 'dance.mp4')
    assert metadata['audio']['D04_verified'] is False
    assert metadata['simulation']['wheel_mode'] == 'driven'
    assert metadata['checks']['stills_match_video'] is True
    assert len(metadata['choreography_sections']) == 7
    assert metadata['choreography_sections'][0]['body_time_s'] == [0.0, 12.4]
    assert metadata['choreography_sections'][-1]['body_time_s'] == [86.8, 94.0]
    assert '12' in text and '协调' in text and '保持锁定' not in text
    assert len(metadata['images']) == 1
    assert metadata['images'][0]['label'] == 'start'
    assert json.loads((root / 'report.json').read_text()) == metadata
    assert metadata['artifacts']['report_pdf']['sha256'] == sha(root / 'report.pdf')


def test_missing_evidence(tmp_path):
    from jc04_dancing.reporting import build_report
    with pytest.raises((ValueError, FileNotFoundError), match='run.json'):
        build_report(tmp_path / 'missing', tmp_path / 'report.pdf')
    assert not (tmp_path / 'report.pdf').exists()


@pytest.mark.parametrize('mutation', ['failed', 'preview', 'fallen', 'nan', 'frames', 'hash', 'missing_still', 'path', 'resource', 'metric_type', 'motion_frame', 'duplicate_label', 'locked_wheels', 'mismatched_motion', 'wrong_interpolation', 'invented_original', 'clip_duration', 'missing_phrases_hash'])
def test_invalid_run_rejected(run_dir, tmp_path, mutation):
    from jc04_dancing.reporting import build_report
    import shutil
    copied_run = tmp_path / 'run'
    copied_run.mkdir()
    shutil.copyfile(run_dir / 'dance.mp4', copied_run / 'dance.mp4')
    shutil.copytree(run_dir / 'stills', copied_run / 'stills')
    data = copy.deepcopy(json.loads((run_dir / 'run.json').read_text()))
    if mutation == 'failed': data['status'] = 'failed'
    if mutation == 'preview': data['preview'] = True
    if mutation == 'fallen': data['simulation']['fallen'] = True
    if mutation == 'nan': data['simulation']['max_horizontal_displacement_m'] = float('nan')
    if mutation == 'frames': data['video']['decoded_frames'] -= 1
    if mutation == 'hash': data['video']['sha256'] = '0' * 64
    if mutation == 'missing_still': data['stills'][0]['file'] = 'stills/missing.png'
    if mutation == 'path': data['video']['file'] = '../outside.mp4'
    if mutation == 'resource': data['evidence']['resource_sha256']['assets/robot/robot.xml'] = '0' * 64
    if mutation == 'metric_type': data['simulation']['max_horizontal_displacement_m'] = 'unverified'
    if mutation == 'motion_frame':
        first = data['stills'][0]
        motion = next(item for item in data['stills'] if item['label'] not in ('start', 'end', 'mid'))
        motion.update(file=first['file'], frame=first['frame'], sample_time_s=first['sample_time_s'])
    if mutation == 'duplicate_label': data['stills'].append(data['stills'][0])
    if mutation == 'locked_wheels': data['simulation']['wheel_mode'] = 'locked'
    if mutation == 'mismatched_motion': data['evidence']['motion']['source'] = 'assets/motions/mismatched.csv'
    if mutation == 'wrong_interpolation': data['evidence']['motion']['interpolation'] = 'linear'
    if mutation == 'invented_original': data['evidence']['motion']['generated_choreography'] = False
    if mutation == 'clip_duration': data['evidence']['motion']['clip_duration_s'] -= 1
    if mutation == 'missing_phrases_hash': data['evidence']['resource_sha256'].pop('assets/motions/swan_lake.phrases.json')
    (copied_run / 'run.json').write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        build_report(copied_run, tmp_path / 'output/report.pdf')
    assert not (tmp_path / 'output/report.pdf').exists()


def test_tampered_still_rejected(run_dir, tmp_path):
    from jc04_dancing.reporting import build_report
    import shutil
    shutil.copytree(run_dir, tmp_path / 'run')
    name = json.loads((run_dir / 'run.json').read_text())['stills'][0]['file']
    Image.new('RGB', (64, 64), 'red').save(tmp_path / 'run' / name)
    with pytest.raises(ValueError, match='still|frame'):
        build_report(tmp_path / 'run', tmp_path / 'report.pdf')


def test_flowchart_files(tmp_path):
    from jc04_dancing.reporting import render_flowchart
    render_flowchart(tmp_path / 'flowchart.pdf', tmp_path / 'flowchart.png')
    with Image.open(tmp_path / 'flowchart.png') as image:
        image.verify()
    text = text_of(tmp_path / 'flowchart.pdf')
    for label in ['准备环境', '加载模型', '读取动作', '控制仿真', '录制画面', '配乐合成']:
        assert label in text
    for forbidden in ['异常', 'D04', '无声', 'SHA256', '天鹅湖']:
        assert forbidden not in text


def test_chinese_render(rendered_report):
    root, _ = rendered_report
    with fitz.open(root / 'report.pdf') as document:
        assert len(document) == 2
        # Exactly one model image plus the process illustration.
        assert sum(len(page.get_images()) for page in document) == 2
        for page in document:
            assert '\ufffd' not in page.get_text()
            assert page.get_images() or page.get_text().strip()
            image = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            assert np.frombuffer(image.samples, np.uint8).std() > 5
            for block in page.get_text('dict')['blocks']:
                if 'lines' not in block: continue
                for line in block['lines']:
                    for span in line['spans']:
                        x0, y0, x1, y1 = span['bbox']
                        assert 15 <= x0 <= x1 <= page.rect.width - 15
                        assert 15 <= y0 <= y1 <= page.rect.height - 15
        assert any('Droid' in font[3] for page in document for font in page.get_fonts())


def test_report_cli(run_dir, tmp_path, capsys):
    from jc04_dancing.cli import main
    assert main(['report', '--run-dir', str(run_dir), '--output', str(tmp_path / 'report.pdf')]) == 0
    assert json.loads(capsys.readouterr().out)['audio']['D04_verified'] is False
    assert main(['report', '--run-dir', str(tmp_path / 'missing'), '--output', str(tmp_path / 'bad.pdf')]) == 1
    assert 'run.json' in capsys.readouterr().err


def _audio_run(run_dir, tmp_path, *, silent=False):
    """A real muxed stream, preserving video packets and all sampled stills."""
    import shutil
    import subprocess
    target = tmp_path / 'audible-run'
    shutil.copytree(run_dir, target)
    data = json.loads((target / 'run.json').read_text())
    source = 'anullsrc=r=8000:cl=mono' if silent else 'sine=frequency=440:sample_rate=8000'
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y',
                             '-i', str(target / 'dance.mp4'), '-f', 'lavfi', '-i', source,
                             '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
                             '-t', str(data['video']['encoded_duration_s']), str(target / 'muxed.mp4')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    (target / 'muxed.mp4').replace(target / 'dance.mp4')
    data['video'].update(audio_stream_present=True, sha256=sha(target / 'dance.mp4'))
    data['audio'] = {'present': True, 'status': 'added', 'D04_verified': False}
    (target / 'run.json').write_text(json.dumps(data))
    return target, data


def test_audible_run_supported(run_dir, tmp_path):
    from jc04_dancing.reporting import build_report
    target, _ = _audio_run(run_dir, tmp_path)
    metadata = build_report(target, tmp_path / 'report/report.pdf')
    assert metadata['audio']['present'] is True
    assert metadata['audio']['D04_verified'] is False
    assert metadata['checks']['audio_stream_present'] is True
    assert metadata['checks']['decoded_audio_rms'] > 0
    assert '天鹅湖' not in text_of(tmp_path / 'report/report.pdf')


@pytest.mark.parametrize('mutation', ['silent_samples', 'stream_mismatch', 'invented_sync'])
def test_unverified_audio_rejected(run_dir, tmp_path, mutation):
    from jc04_dancing.reporting import build_report
    target, data = _audio_run(run_dir, tmp_path, silent=mutation == 'silent_samples')
    if mutation == 'stream_mismatch':
        data['audio'].update(present=False, status='postproduction_pending')
        data['video']['audio_stream_present'] = False
    if mutation == 'invented_sync':
        data['audio']['D04_verified'] = True
    (target / 'run.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='[Aa]udio|D04|sync'):
        build_report(target, tmp_path / 'report/report.pdf')
    assert not (tmp_path / 'report/report.pdf').exists()
