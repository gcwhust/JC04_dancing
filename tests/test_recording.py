"""Exercise encoded pixels and failure cleanup, never just file existence."""
import json
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import imageio_ffmpeg
import numpy as np
import pytest

from jc04_dancing import recording




@pytest.mark.render
def test_short_silent_decode(simulation, tmp_path):
    # Catches missing final frames, FPS drift, black rendering, and false success.
    output = tmp_path / 'dance.mp4'
    result = recording.record(simulation, output, width=320, height=240, duration=4)
    decoder = imageio_ffmpeg.read_frames(str(output), pix_fmt='rgb24')
    info = next(decoder)
    frames = [np.frombuffer(frame, np.uint8).reshape(240, 320, 3) for frame in decoder]
    assert len(frames) == 120
    assert info['size'] == (320, 240)
    assert info['fps'] == 30
    assert info['duration'] == pytest.approx(4, abs=1 / 30)
    assert info['codec'] == 'h264'
    assert all(frame.std() > 10 for frame in (frames[0], frames[60], frames[-1]))
    assert np.mean(np.abs(frames[0].astype(float) - frames[-1])) > 1
    metadata = json.loads((tmp_path / 'run.json').read_text())
    assert metadata == result
    assert result['status'] == 'complete'
    assert result['preview'] and result['mode'] == 'free'
    assert result['audio']['present'] is False
    assert result['audio']['status'] == 'postproduction_pending'
    assert result['video']['decoded_frames'] == 120
    assert result['video']['last_sample_time_s'] == pytest.approx(119 / 30)
    assert result['simulation']['requested_time_s'] == 4
    assert result['camera']['follow'] == 'physical_base_xy'
    assert list((tmp_path / 'stills').glob('*.png'))
    assert not list(tmp_path.glob('*.partial.mp4'))


@pytest.mark.render
def test_encoder_failure(simulation, tmp_path, monkeypatch):
    # A real child process exits nonzero: never publish a corrupt MP4 as success.
    executable = tmp_path / 'broken-ffmpeg'
    executable.write_text('#!/bin/sh\nif [ \"$1\" = \"-version\" ]; then echo fake-ffmpeg; exit 0; fi\necho encoder-intentionally-broken >&2\nexit 7\n')
    executable.chmod(0o755)
    monkeypatch.setenv('IMAGEIO_FFMPEG_EXE', str(executable))
    run_directory = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='encoder|FFmpeg'):
        recording.record(simulation, run_directory / 'dance.mp4', width=64, height=64, duration=.1)
    assert not (run_directory / 'dance.mp4').exists()
    assert not list(run_directory.glob('*.partial.mp4'))
    result = json.loads((run_directory / 'run.json').read_text())
    assert result['status'] == 'failed'
    assert 'encoder-intentionally-broken' in result['error']


@pytest.mark.render
def test_interrupt_partial(simulation, tmp_path, monkeypatch):
    # Interrupt after frames started; an owned real renderer must close.
    original_advance = simulation.advance_to
    original_renderer = recording.mujoco.Renderer
    closed = []

    class TrackedRenderer(original_renderer):
        def close(self):
            super().close()
            closed.append(True)

    def interrupt(t):
        if t >= .1:
            raise KeyboardInterrupt()
        original_advance(t)

    monkeypatch.setattr(recording.mujoco, 'Renderer', TrackedRenderer)
    monkeypatch.setattr(simulation, 'advance_to', interrupt)
    with pytest.raises(KeyboardInterrupt):
        recording.record(simulation, tmp_path / 'dance.mp4', width=64, height=64, duration=.3)
    assert closed
    assert not (tmp_path / 'dance.mp4').exists()
    assert not list(tmp_path.glob('*.partial.mp4'))
    result = json.loads((tmp_path / 'run.json').read_text())
    assert result['status'] == 'interrupted'
    assert result['video']['frames_written'] == 3


@pytest.mark.parametrize('kwargs', [{'fps': 0}, {'width': 321}, {'duration': float('nan')}, {'duration': -1}])
def test_invalid_recording_arguments(tmp_path, kwargs):
    # Invalid output requests must fail before resource construction.
    with pytest.raises(ValueError):
        recording.record(None, tmp_path / 'dance.mp4', **kwargs)


def test_record_cli_help(capsys):
    from jc04_dancing.cli import main
    with pytest.raises(SystemExit) as exit_info:
        main(['record', '--help'])
    assert exit_info.value.code == 0
    assert '--output' in capsys.readouterr().out


@pytest.mark.render
def test_fractional_preview_supported(simulation, tmp_path):
    # ceil(.11*30)=4; full-duration flags must not hide support or truncation.
    simulation.mode = 'supported'
    try:
        result = recording.record(simulation, tmp_path / 'supported.mp4', width=64, height=64, duration=.11)
        assert result['mode'] == 'supported' and result['preview']
        assert not result['timeline_complete']
        assert result['video']['decoded_frames'] == 4
        assert result['simulation']['requested_time_s'] == .11
    finally:
        simulation.mode = 'free'


@pytest.mark.render
def test_backend_failure_has_retry(simulation, tmp_path, monkeypatch):
    def broken_renderer(*args, **kwargs):
        raise RuntimeError('test EGL initialization failed')
    monkeypatch.setattr(recording.mujoco, 'Renderer', broken_renderer)
    with pytest.raises(RuntimeError, match='MUJOCO_GL=egl.*MUJOCO_GL=osmesa'):
        recording.record(simulation, tmp_path / 'dance.mp4', width=64, height=64, duration=.1)
    assert json.loads((tmp_path / 'run.json').read_text())['status'] == 'failed'
    assert not (tmp_path / 'dance.mp4').exists()


@pytest.mark.render
@pytest.mark.parametrize('failure, expected_status', [(KeyboardInterrupt, 'interrupted'), (OSError, 'failed')])
def test_final_metadata_failure_rolls_back_publication(simulation, tmp_path, monkeypatch, failure, expected_status):
    # A metadata write after renames must not leave a published failed video.
    write_json = recording._write_json

    def fail_final_write(path, result):
        if result['status'] == 'complete':
            raise failure('injected final metadata failure')
        write_json(path, result)

    monkeypatch.setattr(recording, '_write_json', fail_final_write)
    expected_error = KeyboardInterrupt if failure is KeyboardInterrupt else RuntimeError
    with pytest.raises(expected_error):
        recording.record(simulation, tmp_path / 'dance.mp4', width=64, height=64, duration=.1)
    assert not (tmp_path / 'dance.mp4').exists()
    assert not (tmp_path / 'stills').exists()
    assert not list(tmp_path.glob('*.partial.mp4'))
    assert not list(tmp_path.glob('.stills-*'))
    result = json.loads((tmp_path / 'run.json').read_text())
    assert result['status'] == expected_status
    assert 'injected final metadata failure' in result['error']


@pytest.mark.render
def test_second_output_preserves_successful_run(simulation, tmp_path):
    # Distinct MP4 names still share run.json/stills: do not overwrite either.
    recording.record(simulation, tmp_path / 'first.mp4', width=64, height=64, duration=.1)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError, match='run directory'):
        recording.record(simulation, tmp_path / 'second.mp4', width=64, height=64, duration=.1)
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert after == before


@pytest.mark.parametrize('artifact', ['run.json', 'stills', 'previous.mp4'])
def test_occupied_run_directory_preserved(simulation, tmp_path, artifact):
    # Orphaned evidence is protected even without the requested final filename.
    prior = tmp_path / artifact
    if artifact == 'stills':
        prior.mkdir()
        prior = prior / 'prior.png'
    prior.write_bytes(b'prior run evidence')
    with pytest.raises(FileExistsError, match='run directory'):
        recording.record(simulation, tmp_path / 'second.mp4', width=64, height=64, duration=.1)
    assert prior.read_bytes() == b'prior run evidence'
    assert not (tmp_path / 'second.mp4').exists()


@pytest.mark.render
def test_mode_and_preview_labels_in_decoded_frames(simulation, tmp_path, monkeypatch):
    # Supported/preview disclosure must survive encoding on every frame;
    # a full free-mode recording must preserve the raw scene without a banner.
    from PIL import Image, ImageDraw, ImageFont

    original_renderer = recording.mujoco.Renderer
    raw_frames = []

    class CapturedRenderer(original_renderer):
        def render(self, *args, **kwargs):
            pixels = super().render(*args, **kwargs)
            raw_frames.append(pixels.copy())
            return pixels

    monkeypatch.setattr(recording.mujoco, 'Renderer', CapturedRenderer)
    original_mode = simulation.mode
    try:
        cases = [('free', False, ''),
                 ('supported', False, 'SUPPORTED MODE - base constrained'),
                 ('supported', True, 'SUPPORTED MODE - base constrained | PREVIEW'),
                 ('free', True, 'PREVIEW')]
        for mode, preview, required_text in cases:
            simulation.mode = mode
            # A tiny complete test timeline avoids recording the full dance just
            # to exercise the full/preview presentation distinction.
            monkeypatch.setattr(simulation.timeline, 'total_duration', .2 if preview else .1)
            raw_frames.clear()
            output = tmp_path / f'{mode}-{preview}' / 'dance.mp4'
            recording.record(simulation, output, width=640, height=240, duration=.1)
            reader = imageio_ffmpeg.read_frames(str(output))
            next(reader)
            decoded = [np.frombuffer(frame, np.uint8).reshape(240, 640, 3) for frame in reader]
            assert len(decoded) == 3
            if required_text:
                mask = Image.new('L', (640, 40))
                ImageDraw.Draw(mask).text((12, 8), required_text, font=ImageFont.load_default(size=14), fill=255)
                ink = np.asarray(mask) > 200
                for frame in decoded:
                    # The expected literal glyphs are readable white on a stable
                    # dark band; this catches missing/incorrect disclosure text.
                    assert np.mean(frame[:40][ink]) > 210
                    assert np.mean(frame[2:5, 12:620]) < 65
                assert np.mean(np.abs(decoded[0][:30].astype(float) - decoded[-1][:30])) < 1
            else:
                for frame, raw in zip(decoded, raw_frames):
                    assert np.mean(np.abs(frame.astype(float) - raw)) < 3
                    assert np.mean(np.abs(frame[:30].astype(float) - raw[:30])) < 2
    finally:
        simulation.mode = original_mode


def test_swan_evidence_identifies_generated_fullbody_curve():
    from types import SimpleNamespace
    from jc04_dancing.simulation import build_timeline
    result = recording._evidence(SimpleNamespace(timeline=build_timeline('swan_lake')))
    assert result['motion']['source'] == 'assets/motions/swan_lake.csv'
    assert result['motion']['interpolation'] == 'quintic'
    assert result['motion']['generated_choreography'] is True
    assert result['motion']['wheel_units'] == 'rad/s'
    assert 'choreography.py' in result['code_sha256']
