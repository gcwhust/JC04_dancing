"""Add music to an accepted run without rendering or changing its video frames."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np
from PIL import Image

from .resources import resource_path


_RATE = 48000
_DEFAULT_MUSIC = 'assets/audio/happy_new_year.mp3'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise RuntimeError('FFmpeg failed: ' + result.stderr.decode(errors='replace').strip())
    return result


def _file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError(f'Evidence path must be relative: {name}')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing or escaping evidence file: {name}')
    return path


def _finite_tree(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite recording evidence')


def _validate_run(root: Path) -> tuple[dict, Path]:
    try:
        data = json.loads(_file(root, 'run.json').read_text(encoding='utf-8'))
        _finite_tree(data)
        if (data['schema_version'] != 1 or data['status'] != 'complete' or
                data['preview'] is not False or data['timeline_complete'] is not True):
            raise ValueError('A complete schema_version 1 timeline is required')
        video, sim = data['video'], data['simulation']
        if (data['mode'] not in ('free', 'supported') or sim['mode'] != data['mode'] or
                sim['fallen'] is not False or sim['warnings'] or data['camera']['physics_recentered'] is not False):
            raise ValueError('Invalid simulation evidence')
        fps, width, height = video['fps'], video['width'], video['height']
        if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0 or
                any(type(v) is not int or v <= 0 for v in (width, height))):
            raise ValueError('Invalid video size or FPS')
        duration = video['requested_duration_s']
        if duration <= 0 or any(abs(duration - sim[k]) > 1e-6 for k in ('timeline_duration_s', 'requested_time_s')):
            raise ValueError('Inconsistent complete timeline duration')
        count = math.ceil(duration * fps)
        if (any(type(video[k]) is not int or video[k] != count for k in ('expected_frames', 'frames_written', 'decoded_frames')) or
                abs(video['encoded_duration_s'] - count / fps) > 1e-6 or
                video['decoded_size'] != [width, height] or video['decoded_codec'] != 'h264' or
                abs(video['decoded_fps'] - fps) > 1e-6):
            raise ValueError('Inconsistent video evidence')
        if (data['audio']['present'] is not False or video['audio_stream_present'] is not False or
                data['audio']['D04_verified'] is not False):
            raise ValueError('Source must be an explicitly silent run')
        path = _file(root, video['file'])
        if _sha256(path) != video['sha256']:
            raise ValueError('Source video SHA256 differs from run.json')
        evidence = data['evidence']
        if not evidence['code_sha256'] or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v)
                                               for v in evidence['code_sha256'].values()):
            raise ValueError('Invalid historical code hashes')
        for name, expected in evidence['resource_sha256'].items():
            if _sha256(resource_path(name)) != expected:
                raise ValueError(f'Changed recording resource: {name}')
        labels, files = set(), set()
        if not data['stills']:
            raise ValueError('Missing recorded stills')
        for still in data['stills']:
            index = still['frame']
            if (type(index) is not int or not 0 <= index < count or
                    abs(still['sample_time_s'] - index / fps) > 1e-6 or
                    still['label'] in labels or still['file'] in files or still['file'] in ('run.json', video['file'])):
                raise ValueError('Inconsistent still evidence')
            labels.add(still['label'])
            files.add(still['file'])
            _file(root, still['file'])
        return data, path
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Malformed run.json: {exc}') from exc


def _decode_video(path: Path, data: dict, root: Path) -> tuple[dict, str]:
    """Hash every decoded RGB byte and compare every listed PNG to its frame."""
    video = data['video']
    samples = {}
    for still in data['stills']:
        samples.setdefault(still['frame'], []).append(still['file'])
    digest = hashlib.sha256()
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24', output_params=['-threads', '2'])
    try:
        info = next(reader)
        if (info['codec'] != 'h264' or info['size'] != (video['width'], video['height']) or
                abs(info['fps'] - video['fps']) > 1e-6):
            raise ValueError('Actual video format differs from recording evidence')
        count = 0
        for count, frame in enumerate(reader, 1):
            digest.update(frame)
            for name in samples.get(count - 1, []):
                with Image.open(_file(root, name)) as image:
                    if image.size != info['size'] or image.convert('RGB').tobytes() != frame:
                        raise ValueError(f'Still differs from decoded video frame: {name}')
        if count != video['expected_frames'] or abs(info['duration'] - count / video['fps']) > 1 / video['fps']:
            raise ValueError('Actual video frame count/duration differs from recording evidence')
        return {'decoded_frames': count, 'decoded_size': list(info['size']), 'decoded_fps': info['fps'],
                'decoded_codec': info['codec'], 'decoded_duration_s': info['duration']}, digest.hexdigest()
    finally:
        reader.close()


def _pcm(ffmpeg: str, path: Path) -> np.ndarray:
    result = _run([ffmpeg, '-v', 'error', '-i', str(path), '-map', '0:a:0', '-vn',
                   '-ac', '2', '-ar', str(_RATE), '-acodec', 'pcm_f32le', '-f', 'f32le', '-'])
    values = np.frombuffer(result.stdout, '<f4')
    if not len(values) or len(values) % 2 or not np.isfinite(values).all():
        raise ValueError('Audio could not be decoded as finite stereo PCM')
    return values.reshape(-1, 2)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.


def add_audio_to_run(run_dir: Path, output_dir: Path, audio_path: Path | None = None,
                     timeline_start_seconds: float = 2., fade_in_seconds: float = .4,
                     fade_out_seconds: float = .8) -> dict:
    """Stream-copy H.264, add unlooped original-speed AAC, and publish atomically.

    Music starts at source time zero. Fades apply within the usable original
    music; preparation and the remaining timeline contain silence. Numerical
    verification is recorded without claiming beat alignment or an audition.
    """
    source, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Source and output directories must not overlap')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory is occupied: {output}')
    for label, value in [('timeline_start_seconds', timeline_start_seconds),
                         ('fade_in_seconds', fade_in_seconds), ('fade_out_seconds', fade_out_seconds)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{label} must be finite and nonnegative')
    data, video_path = _validate_run(source)
    duration = data['video']['encoded_duration_s']
    if timeline_start_seconds >= data['video']['requested_duration_s']:
        raise ValueError('Audio offset must be before the timeline endpoint')
    music = resource_path(_DEFAULT_MUSIC) if audio_path is None else Path(audio_path).resolve()
    if not music.is_file():
        raise ValueError(f'Audio source is missing: {music}')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([ffmpeg, '-hide_banner', '-i', str(video_path)], capture_output=True, text=True)
    if 'Audio:' in probe.stderr:
        raise ValueError('Source video unexpectedly already contains audio')
    source_run_hash, source_video_hash, music_hash = (_sha256(source / 'run.json'), _sha256(video_path), _sha256(music))
    _, source_frames_hash = _decode_video(video_path, data, source)
    original_pcm = _pcm(ffmpeg, music)
    original_duration = len(original_pcm) / _RATE
    effective_duration = min(original_duration, duration - timeline_start_seconds)
    if _rms(original_pcm[:round(effective_duration * _RATE)]) <= 1e-6:
        raise ValueError('Audio source has no measurable signal in the usable portion')
    fade_in = min(fade_in_seconds, effective_duration / 2)
    fade_out = min(fade_out_seconds, effective_duration / 2)
    start_sample = round(timeline_start_seconds * _RATE)
    filters = (f'aresample={_RATE},atrim=start=0:duration={effective_duration:.12f},asetpts=PTS-STARTPTS,'
               + (f'afade=t=in:st=0:d={fade_in:.12f},' if fade_in > 0 else '')
               + (f'afade=t=out:st={effective_duration - fade_out:.12f}:d={fade_out:.12f},' if fade_out > 0 else '')
               + f'adelay={start_sample}S:all=1,apad,atrim=duration={duration:.12f}')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        final_video = stage / data['video']['file']
        final_video.parent.mkdir(parents=True, exist_ok=True)
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video_path),
                   '-i', str(music), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
                   '-af', filters, '-c:a', 'aac', '-b:a', '192k', '-ar', str(_RATE), '-ac', '2',
                   '-t', f'{duration:.12f}', '-movflags', '+faststart', str(final_video)]
        _run(command)
        decoded, final_frames_hash = _decode_video(final_video, data, source)
        if final_frames_hash != source_frames_hash:
            raise RuntimeError('Decoded video pixels changed during audio postprocessing')
        encoded_probe = subprocess.run([ffmpeg, '-hide_banner', '-i', str(final_video)], capture_output=True, text=True)
        audio_line = next((line.strip() for line in encoded_probe.stderr.splitlines() if 'Audio:' in line), '')
        if 'Audio: aac' not in audio_line or '48000 Hz' not in audio_line or 'stereo' not in audio_line:
            raise RuntimeError('Final container lacks expected AAC 48000 Hz stereo audio')
        pcm = _pcm(ffmpeg, final_video)
        if abs(len(pcm) / _RATE - duration) > 1024 / _RATE + 1e-6:
            raise RuntimeError('Final decoded audio duration differs from video duration')
        def window(start: float, end: float) -> np.ndarray:
            return pcm[max(0, round(start * _RATE)):max(0, round(end * _RATE))]
        prep_rms = _rms(window(0, max(0, timeline_start_seconds - .05)))
        active_rms = _rms(window(timeline_start_seconds, timeline_start_seconds + effective_duration))
        tail_rms = _rms(window(timeline_start_seconds + effective_duration + .05, duration))
        if active_rms <= 1e-6 or prep_rms > 1e-4 or tail_rms > 1e-4:
            raise RuntimeError('Decoded audio lacks music or expected preparation/ending silence')
        result = deepcopy(data)
        result['video'].update(decoded, sha256=_sha256(final_video), audio_stream_present=True)
        result['audio'] = {
            'present': True, 'status': 'added', 'D04_verified': False,
            'source': _DEFAULT_MUSIC if audio_path is None else str(music),
            'source_resource': _DEFAULT_MUSIC if audio_path is None else None,
            'source_path': str(music), 'source_sha256': music_hash,
            'source_start_seconds': 0., 'timeline_start_seconds': timeline_start_seconds,
            'fade_in_seconds': fade_in, 'fade_out_seconds': fade_out,
            'source_duration_s': original_duration, 'effective_duration_s': effective_duration,
            'music_end_seconds': timeline_start_seconds + effective_duration,
            'playback_rate': 1., 'looped': False, 'codec': 'aac', 'sample_rate_hz': _RATE, 'channels': 2,
            'decoded_duration_s': len(pcm) / _RATE, 'decoded_rms': _rms(pcm),
            'decoded_preparation_rms': prep_rms, 'decoded_active_rms': active_rms, 'decoded_tail_rms': tail_rms,
            'container_audio_stream': audio_line, 'verification': 'decoded PCM numerical checks; no audition or beat verification',
        }
        still_hashes = {}
        for still in data['stills']:
            name = still['file']
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_file(source, name), destination)
            still_hashes[name] = _sha256(destination)
            if still_hashes[name] != _sha256(_file(source, name)):
                raise RuntimeError(f'Still changed during copy: {name}')
        result['postprocessing'] = {
            'operation': 'add_audio', 'source_run_sha256': source_run_hash, 'source_video_sha256': source_video_hash,
            'module_sha256': _sha256(Path(__file__)), 'source_decoded_rgb_sha256': source_frames_hash,
            'output_decoded_rgb_sha256': final_frames_hash, 'video_frames_identical': True,
            'video_stream_copy': True, 'stills_sha256': still_hashes, 'encoder_command': command,
        }
        if (_sha256(source / 'run.json') != source_run_hash or _sha256(video_path) != source_video_hash or
                _sha256(music) != music_hash):
            raise RuntimeError('Source evidence changed during postprocessing')
        (stage / 'run.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if output.exists():
            output.rmdir()  # Only an initially empty directory can be replaced.
        stage.replace(output)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
