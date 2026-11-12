"""Real short muxes catch offsets, overwritten evidence, and failed publication."""
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import imageio_ffmpeg
import numpy as np
from PIL import Image
import pytest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def silent_run(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    video = source / 'dance.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=64x48:rate=10:duration=4', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-threads', '1', str(video)], check=True)
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    next(reader)
    frames = list(reader)
    (source / 'stills').mkdir()
    stills = []
    for label, index in [('start', 0), ('end', 39)]:
        name = f'stills/{label}.png'
        Image.fromarray(np.frombuffer(frames[index], np.uint8).reshape(48, 64, 3)).save(source / name)
        stills.append({'file': name, 'frame': index, 'label': label, 'sample_time_s': index / 10})
    metadata = {'schema_version': 1, 'status': 'complete', 'preview': False,
                'timeline_complete': True, 'mode': 'free', 'camera': {'physics_recentered': False},
                'simulation': {'mode': 'free', 'fallen': False, 'warnings': {},
                               'timeline_duration_s': 4., 'requested_time_s': 4.},
                'video': {'file': 'dance.mp4', 'sha256': digest(video), 'width': 64, 'height': 48,
                          'fps': 10, 'requested_duration_s': 4., 'encoded_duration_s': 4.,
                          'expected_frames': 40, 'frames_written': 40, 'decoded_frames': 40,
                          'decoded_size': [64, 48], 'decoded_fps': 10., 'decoded_codec': 'h264',
                          'audio_stream_present': False},
                'audio': {'present': False, 'status': 'postproduction_pending', 'D04_verified': False},
                'evidence': {'code_sha256': {'recording.py': 'a' * 64}, 'resource_sha256': {}},
                'stills': stills}
    (source / 'run.json').write_text(json.dumps(metadata))
    music = tmp_path / 'tone.wav'
    with wave.open(str(music), 'wb') as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        values = (np.sin(np.arange(38400) * (2 * np.pi * 440 / 48000)) * 14000).astype('<i2')
        stream.writeframes(np.column_stack([values, values]).tobytes())
    return source, music


def test_mux_offsets_pcm_and_preserves_original_and_frames(silent_run, tmp_path):
    # Omitting adelay, speeding up, looping, re-encoding, or overwriting source fails.
    from jc04_dancing.audio import add_audio_to_run
    source, music = silent_run
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result = add_audio_to_run(source, tmp_path / 'with-audio', music)
    assert result['audio']['present'] is True
    assert result['audio']['status'] == 'added'
    assert result['audio']['D04_verified'] is False
    assert result['audio']['source_duration_s'] == pytest.approx(.8, abs=.001)
    assert result['audio']['timeline_start_seconds'] == 2
    assert result['audio']['source_start_seconds'] == 0
    assert result['audio']['decoded_preparation_rms'] < .0001
    assert result['audio']['decoded_active_rms'] > .01
    assert result['audio']['decoded_tail_rms'] < .0001
    assert result['video']['decoded_frames'] == 40
    assert result['postprocessing']['source_video_sha256'] == digest(source / 'dance.mp4')
    assert result['postprocessing']['source_run_sha256'] == digest(source / 'run.json')
    assert result['postprocessing']['video_frames_identical'] is True
    assert result['evidence'] == json.loads(before[Path('run.json')])['evidence']
    assert result['simulation'] == json.loads(before[Path('run.json')])['simulation']
    assert all((source / name).read_bytes() == data for name, data in before.items())
    assert all((tmp_path / 'with-audio' / name).read_bytes() == data for name, data in before.items() if name.parts[0] == 'stills')
    assert json.loads((tmp_path / 'with-audio/run.json').read_text()) == result
    pcm = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i',
                          str(tmp_path / 'with-audio/dance.mp4'), '-map', '0:a:0',
                          '-f', 'f32le', '-acodec', 'pcm_f32le', '-'], capture_output=True, check=True)
    samples = np.frombuffer(pcm.stdout, '<f4').reshape(-1, 2)
    assert np.max(np.abs(samples[:90000])) < .0001
    assert np.sqrt(np.mean(samples[102000:115000] ** 2)) > .02
    assert np.max(np.abs(samples[150000:])) < .0001


@pytest.mark.parametrize('offset', [-1, 4, float('nan'), float('inf'), True])
def test_invalid_offsets_do_not_create_output(silent_run, tmp_path, offset):
    from jc04_dancing.audio import add_audio_to_run
    source, music = silent_run
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        add_audio_to_run(source, output, music, timeline_start_seconds=offset)
    assert not output.exists()


def test_rejects_overlap_and_occupied_destination(silent_run, tmp_path):
    from jc04_dancing.audio import add_audio_to_run
    source, music = silent_run
    for output in [source, source / 'child', source.parent]:
        with pytest.raises(ValueError):
            add_audio_to_run(source, output, music)
    occupied = tmp_path / 'occupied'
    occupied.mkdir()
    (occupied / 'keep').write_text('keep')
    with pytest.raises((ValueError, FileExistsError)):
        add_audio_to_run(source, occupied, music)
    assert (occupied / 'keep').read_text() == 'keep'


@pytest.mark.parametrize('corrupt', ['video', 'still', 'incomplete'])
def test_rejects_changed_evidence_before_publishing(silent_run, tmp_path, corrupt):
    from jc04_dancing.audio import add_audio_to_run
    source, music = silent_run
    if corrupt == 'video':
        with (source / 'dance.mp4').open('ab') as stream:
            stream.write(b'changed')
    elif corrupt == 'still':
        Image.new('RGB', (64, 48), (2, 3, 4)).save(source / 'stills/start.png')
    else:
        metadata = json.loads((source / 'run.json').read_text())
        metadata['timeline_complete'] = False
        (source / 'run.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        add_audio_to_run(source, tmp_path / 'invalid', music)
    assert not (tmp_path / 'invalid').exists()


def test_encoder_failure_leaves_source_and_destination_unpublished(silent_run, tmp_path, monkeypatch):
    from jc04_dancing.audio import add_audio_to_run
    source, music = silent_run
    original = digest(source / 'dance.mp4')
    executable = tmp_path / 'broken-ffmpeg'
    executable.write_text('#!/bin/sh\nfor arg in "$@"; do\nif [ "$arg" = "-c:a" ]; then echo encoder-intentionally-broken >&2; exit 7; fi\ndone\nexec ' + imageio_ffmpeg.get_ffmpeg_exe() + ' "$@"\n')
    executable.chmod(0o755)
    monkeypatch.setenv('IMAGEIO_FFMPEG_EXE', str(executable))
    with pytest.raises(RuntimeError, match='encoder-intentionally-broken'):
        add_audio_to_run(source, tmp_path / 'failed', music)
    assert not (tmp_path / 'failed').exists()
    assert not list(tmp_path.glob('.failed-*'))
    assert digest(source / 'dance.mp4') == original


def test_cli_audio_publishes_run(silent_run, tmp_path, capsys):
    from jc04_dancing.cli import main
    source, music = silent_run
    code = main(['add-audio', '--run-dir', str(source), '--output', str(tmp_path / 'cli'), '--audio', str(music)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)['audio']['present'] is True


def test_44100_hz_music_starts_at_two_seconds(silent_run, tmp_path):
    # A sample-count delay interpreted at the input rate shifts 44.1 kHz music.
    from jc04_dancing.audio import add_audio_to_run
    source, music = silent_run
    with wave.open(str(music), 'wb') as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        values = (np.sin(np.arange(35280) * (2 * np.pi * 440 / 44100)) * 14000).astype('<i2')
        stream.writeframes(np.column_stack([values, values]).tobytes())
    encoded_music = tmp_path / 'tone-44100.mp3'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(music),
                    '-ar', '44100', '-c:a', 'libmp3lame', str(encoded_music)], check=True)
    music = encoded_music
    output = tmp_path / 'resampled'
    result = add_audio_to_run(source, output, music, fade_in_seconds=.01, fade_out_seconds=.1)
    assert result['status'] == 'complete'
    assert result['audio']['source_duration_s'] == pytest.approx(.8, abs=.001)
    decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i',
                              str(output / 'dance.mp4'), '-map', '0:a:0', '-f', 'f32le',
                              '-acodec', 'pcm_f32le', '-'], capture_output=True, check=True)
    samples = np.frombuffer(decoded.stdout, '<f4').reshape(-1, 2)
    assert np.max(np.abs(samples[:95000])) < .0001
    # 2.02–2.08 s must already contain music; a 44.1 kHz delay would start at 2.177 s.
    assert np.sqrt(np.mean(samples[96960:99840] ** 2)) > .1
    assert np.max(np.abs(samples[144000:])) < .0001


def test_zero_fades_preserve_early_and_late_music_amplitude(silent_run, tmp_path):
    # Passing FFmpeg d=0 selects its default fade instead of disabling the fade.
    from jc04_dancing.audio import add_audio_to_run
    source, music = silent_run
    output = tmp_path / 'no-fades'
    result = add_audio_to_run(source, output, music, fade_in_seconds=0, fade_out_seconds=0)
    assert result['audio']['fade_in_seconds'] == result['audio']['fade_out_seconds'] == 0
    decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i',
                              str(output / 'dance.mp4'), '-map', '0:a:0', '-f', 'f32le',
                              '-acodec', 'pcm_f32le', '-'], capture_output=True, check=True)
    samples = np.frombuffer(decoded.stdout, '<f4').reshape(-1, 2)
    # Original 14000/32768 sine has RMS .302; neither end should fade down.
    assert np.sqrt(np.mean(samples[97440:99840] ** 2)) > .25
    assert np.sqrt(np.mean(samples[130560:132960] ** 2)) > .25


def test_default_soundtrack_is_swan_lake(silent_run, tmp_path):
    from jc04_dancing.audio import add_audio_to_run
    source, _ = silent_run
    result = add_audio_to_run(source, tmp_path / 'swan-default')
    assert result['audio']['source_resource'] == 'assets/audio/swan_lake.mp3'
    assert result['audio']['decoded_active_rms'] > 1e-4
