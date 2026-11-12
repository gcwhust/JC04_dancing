"""Stream deterministic MuJoCo frames to silent H.264 and verify the result."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .resources import resource_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, result: dict) -> None:
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def _evidence(sim) -> dict:
    import yaml
    dances = yaml.safe_load(resource_path('configs/dances.yaml').read_text())
    dance = dances['dances'].get(getattr(sim.timeline, 'dance_id', None))
    paths = ['configs/robot.yaml', 'configs/dances.yaml', 'assets/provenance.json']
    paths += [str(path.relative_to(resource_path('.'))) for path in sorted(resource_path('assets/robot').rglob('*')) if path.is_file()]
    if dance:
        paths.append(dance['file'])
    clip = sim.timeline.clip
    phrases_path = dance.get('phrases') if dance else None
    if phrases_path:
        paths.append(phrases_path)
    for name in ('assets/audio/swan_lake.mp3', 'assets/audio/swan_lake_source.json'):
        if getattr(clip, 'full_body', False):
            paths.append(name)
    provenance = json.loads(resource_path('assets/provenance.json').read_text())
    original_motion_hash = next((item['source_sha256'] for item in provenance['files'] if dance and item['target'] == dance['file'].removeprefix('assets/')), None)
    return {
        'code_sha256': {name: _sha256(Path(__file__).parent / name) for name in ('recording.py', 'simulation.py', 'motion.py', 'cli.py', 'choreography.py')},
        'resource_sha256': {name: _sha256(resource_path(name)) for name in paths},
        'motion': {'source': dance['file'] if dance else None, 'source_fps': dance['fps'] if dance else None,
                   'original_bytes_unchanged': bool(dance and _sha256(resource_path(dance['file'])) == original_motion_hash), 'clip_start_s': clip.start,
                   'clip_duration_s': clip.duration, 'interpolation': clip.interpolation,
                   'generated_choreography': getattr(clip, 'full_body', False),
                   'wheel_units': 'rad/s' if getattr(clip, 'full_body', False) else None,
                   'phrases_source': phrases_path,
                   'prep_s': sim.timeline.prep, 'return_s': sim.timeline.return_time,
                   'hold_s': sim.timeline.hold},
    }


def _verify_video(path, width, height, fps, frame_count, stills, samples):
    """Actually decode every frame; retain only representative decoded stills."""
    decoder = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24', output_params=['-threads', '2'])
    evidence = []
    try:
        info = next(decoder)
        if info['size'] != (width, height) or info['codec'] != 'h264' or abs(info['fps'] - fps) > 1e-6:
            raise RuntimeError(f'Unexpected decoded video format: {info}')
        count = 0
        for count, frame in enumerate(decoder, 1):
            index = count - 1
            if index in samples:
                pixels = np.frombuffer(frame, np.uint8).reshape(height, width, 3)
                if float(pixels.std()) < 1:
                    raise RuntimeError(f'Blank decoded image at frame {index}')
                for label in samples[index]:
                    name = f'{label}-{index:04d}.png'
                    Image.fromarray(pixels).save(stills / name)
                    evidence.append({'label': label, 'frame': index, 'sample_time_s': index / fps,
                                     'file': f'stills/{name}', 'pixel_std': float(pixels.std())})
        if count != frame_count or abs(info['duration'] - frame_count / fps) > 1 / fps:
            raise RuntimeError(f'Decoded video mismatch: {count} frames, {info}')
        probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-i', str(path)], capture_output=True, text=True)
        if 'Audio:' in probe.stderr:
            raise RuntimeError('Unexpected audio stream in silent recording')
        return {'audio_stream_present': False, 'decoded_frames': count, 'decoded_size': list(info['size']),
                'decoded_duration_s': info['duration'], 'decoded_fps': info['fps'],
                'decoded_codec': info['codec']}, evidence
    finally:
        decoder.close()


def _disclosure_banner(width: int, height: int, mode: str, preview: bool):
    """Build one readable, persistent disclosure; full free footage stays raw."""
    labels = []
    if mode == 'supported':
        labels.append('SUPPORTED MODE - base constrained')
    if preview:
        labels.append('PREVIEW')
    if not labels:
        return None
    text = ' | '.join(labels)
    font_size = max(9, min(32, width // 45, height // 12))
    font = ImageFont.load_default(size=font_size)
    padding_x, padding_y = (12, 8) if width >= 320 else (4, 2)
    lines, line = [], ''
    for word in text.split():
        candidate = f'{line} {word}' if line else word
        if line and font.getlength(candidate) > width - 2 * padding_x:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    line_height = font_size + (4 if width >= 320 else 2)
    band = Image.new('RGB', (width, min(height, len(lines) * line_height + 2 * padding_y)), (20, 26, 36))
    draw = ImageDraw.Draw(band)
    for index, line in enumerate(lines):
        draw.text((padding_x, padding_y + index * line_height), line, font=font, fill=(255, 255, 255))
    return np.asarray(band)


def record(sim, output: Path, width=1920, height=1080, fps=30, duration: float | None = None) -> dict:
    """Record k/fps samples, ceil(duration*fps) frames, with no audio.

    Final MP4 is published only after encoder success and complete decoding.
    run.json records complete/failed/interrupted status. The run directory must
    be empty so earlier metadata, stills and outputs can never be overwritten.
    MUJOCO_GL must be selected before importing MuJoCo in the calling process.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v % 2 for v in (width, height)):
        raise ValueError('width and height must be positive even integers for H.264 yuv420p')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError('duration must be finite and positive')
    total = sim.timeline.total_duration
    duration = total if duration is None else float(duration)
    if duration > total:
        raise ValueError('duration cannot exceed the full timeline')
    output = Path(output)
    if output.suffix.lower() != '.mp4':
        raise ValueError('output must have an .mp4 extension')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}; choose a new run directory')
    if output.parent.exists() and any(output.parent.iterdir()):
        raise FileExistsError(f'Occupied run directory: {output.parent}; choose a new empty run directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    partial = output.with_name(f'.{output.stem}.{token}.partial.mp4')
    stills = output.parent / f'.stills-{token}'
    stills.mkdir()
    frame_count = math.ceil(duration * fps)
    result = {
        'schema_version': 1, 'status': 'recording', 'mode': sim.mode,
        'preview': duration < total, 'timeline_complete': duration == total,
        'audio': {'present': False, 'status': 'postproduction_pending', 'D04_verified': False},
        'video': {'file': output.name, 'width': width, 'height': height, 'fps': fps,
                  'requested_duration_s': duration, 'encoded_duration_s': frame_count / fps,
                  'expected_frames': frame_count, 'frames_written': 0,
                  'last_sample_time_s': (frame_count - 1) / fps,
                  'sampling': 'absolute k/fps; ceil(duration*fps); physics also advances to endpoint',
                  'codec': 'libx264', 'pixel_format': 'yuv420p'},
        'camera': {'config': sim.config['camera'],
                   'follow': 'physical_base_xy' if sim.config['camera'].get('follow_base_xy') else 'fixed',
                   'physics_recentered': False,
                   'note': 'Camera tracking, when enabled, reads measured physical base XY; displacement is reported in the simulation summary.'},
        'config': sim.config,
        'versions': {'python': platform.python_version(), 'mujoco': mujoco.__version__,
                     **{name: importlib.metadata.version(name) for name in ('numpy', 'imageio-ffmpeg', 'Pillow')}},
        'render_backend': os.environ.get('MUJOCO_GL', 'default'),
    }
    process = renderer = None
    log = tempfile.TemporaryFile(mode='w+b')
    started = time.monotonic()
    try:
        _write_json(output.parent / 'run.json', result)
        result['evidence'] = _evidence(sim)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        version = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True)
        if version.returncode:
            raise RuntimeError(f'FFmpeg encoder unavailable: {version.stderr}')
        result['versions']['ffmpeg'] = version.stdout.splitlines()[0]
        result['versions']['ffmpeg_executable'] = ffmpeg
        sim.reset()
        # Increasing framebuffer capacity affects only presentation, not physics.
        sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, width)
        sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, height)
        try:
            renderer = mujoco.Renderer(sim.model, height=height, width=width)
        except Exception as exc:
            raise RuntimeError('Offscreen renderer failed. Retry in a NEW process: '
                               'MUJOCO_GL=egl jc04-dancing record --dance swan_lake --output outputs/retry/dance.mp4 '
                               '(requires libEGL + Mesa/vendor OpenGL); alternatively MUJOCO_GL=osmesa '
                               'with libOSMesa. Original error: ' + str(exc)) from exc
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        settings = sim.config['camera']
        for key in ('distance', 'azimuth', 'elevation'):
            setattr(camera, key, settings[key])
        camera.lookat[:] = settings['lookat']
        root = int(sim.model.joint(sim.config['model']['root_joint']).qposadr[0])
        initial_xy = sim.data.qpos[root:root + 2].copy()
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                   '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                   '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
                   '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(partial)]
        result['encoder_command'] = command
        banner = _disclosure_banner(width, height, sim.mode, result['preview'])
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        samples = {}
        for label, index in [('start', 0), ('mid', frame_count // 2), ('end', frame_count - 1)]:
            samples.setdefault(index, []).append(label)
        # Midpoints of the authored sections, always actual decoded frames.
        phrases_path = result['evidence']['motion'].get('phrases_source')
        if phrases_path:
            phrases = json.loads(resource_path(phrases_path).read_text())
            for section in phrases['sections']:
                t = sim.timeline.prep + (section['start_s'] + section['end_s']) / 2
                index = round(t * fps)
                if index < frame_count:
                    samples.setdefault(index, []).append(section['id'])
        frame_states = {}
        for k in range(frame_count):
            sim.advance_to(k / fps)
            if settings.get('follow_base_xy'):
                camera.lookat[:2] = np.asarray(settings['lookat'][:2]) + sim.data.qpos[root:root + 2] - initial_xy
            if k in samples:
                frame_states[k] = {'simulation_time_s': float(sim.data.time),
                                   'physical_base_position_m': sim.data.qpos[root:root + 3].tolist(),
                                   'camera_lookat_m': camera.lookat.tolist(),
                                   'actual_wheel_angle_rad': {name: float(sim.data.qpos[sim.model.joint(name).qposadr[0]]) for name in sim.config['wheels']['joints']},
                                   'actual_wheel_velocity_rad_s': {name: float(sim.data.qvel[sim.model.joint(name).dofadr[0]]) for name in sim.config['wheels']['joints']},
                                   'actual_driven_joint_rad': {name: float(sim.data.qpos[sim.model.joint(name).qposadr[0]]) for name in sim.names}}
            renderer.update_scene(sim.data, camera=camera)
            try:
                pixels = renderer.render()
                if banner is not None:
                    pixels[:banner.shape[0]] = banner
                process.stdin.write(pixels.tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError('FFmpeg encoder closed its input') from exc
            result['video']['frames_written'] += 1
            if k % 300 == 0:
                print(f'record: frame {k + 1}/{frame_count}, t={k / fps:.3f}s', file=sys.stderr, flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('FFmpeg encoder failed')
        sim.advance_to(duration)
        renderer.close()
        renderer = None
        result['simulation'] = sim.summary()
        decoded, evidence = _verify_video(partial, width, height, fps, frame_count, stills, samples)
        result['video'].update(decoded)
        result['video']['sha256'] = _sha256(partial)
        result['stills'] = [{**item, **frame_states[item['frame']]} for item in evidence]
        result['status'] = 'complete'
        result['elapsed_wall_s'] = time.monotonic() - started
        stills.replace(output.parent / 'stills')
        partial.replace(output)
        _write_json(output.parent / 'run.json', result)
        return result
    except BaseException as exc:
        # This attempt owns an initially empty directory. Undo either rename
        # even when final metadata persistence is interrupted or fails.
        output.unlink(missing_ok=True)
        if (output.parent / 'stills').exists():
            shutil.rmtree(output.parent / 'stills')
        result['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        result['error'] = f'{type(exc).__name__}: {exc}'
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.seek(0)
            diagnostic = log.read().decode(errors='replace').strip()
            if diagnostic:
                result['error'] += '\n' + diagnostic
        result['simulation'] = sim.summary()
        result['elapsed_wall_s'] = time.monotonic() - started
        _write_json(output.parent / 'run.json', result)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(result['error']) from exc
    finally:
        if process is not None and process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        if renderer is not None:
            renderer.close()
        log.close()
        partial.unlink(missing_ok=True)
        (output.parent / 'run.json.tmp').unlink(missing_ok=True)
        if stills.exists():
            shutil.rmtree(stills)
