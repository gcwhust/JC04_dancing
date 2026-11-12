"""Command line entry point for the standalone JC04 project."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import yaml

from . import __version__
from .resources import resource_path


def _play(simulation) -> None:
    # Importing viewer/GL is deliberately confined to interactive playback.
    import mujoco.viewer

    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
        camera = simulation.config['camera']
        viewer.cam.distance = camera['distance']
        viewer.cam.azimuth = camera['azimuth']
        viewer.cam.elevation = camera['elevation']
        viewer.cam.lookat[:] = camera['lookat']
        root_adr = int(simulation.model.joint(simulation.config['model']['root_joint']).qposadr[0])
        initial_xy = simulation.data.qpos[root_adr:root_adr + 2].copy()
        start = time.monotonic()
        frame = 0
        while viewer.is_running():
            requested = min(frame / 30, simulation.timeline.total_duration)
            simulation.advance_to(requested)
            if camera.get('follow_base_xy', False):
                viewer.cam.lookat[:2] = camera['lookat'][:2] + simulation.data.qpos[root_adr:root_adr + 2] - initial_xy
            viewer.sync()
            if requested >= simulation.timeline.total_duration:
                break
            # Wall time limits presentation speed, never selects a pose/time.
            delay = start + requested - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            frame += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jc04-dancing", description="Standalone JC04 MuJoCo dance project")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest='command')
    commands.add_parser('list', help='list available original dances (no display needed)')
    for name in ('check', 'play', 'record'):
        command = commands.add_parser(name)
        command.add_argument('--dance', default='new_year')
        command.add_argument('--mode', choices=('free', 'supported'), default='free')
        if name in ('check', 'record'):
            command.add_argument('--duration', type=float, help='seconds to check; default is the complete timeline')
        if name == 'record':
            command.add_argument('--output', type=Path, default=Path('outputs/final_run/dance.mp4'))
            command.add_argument('--width', type=int, default=1920, help='preview width; default 1920')
            command.add_argument('--height', type=int, default=1080, help='preview height; default 1080')
    audio_command = commands.add_parser('add-audio', help='add original-speed music while preserving every video frame')
    audio_command.add_argument('--run-dir', type=Path, required=True)
    audio_command.add_argument('--output', type=Path, required=True)
    audio_command.add_argument('--audio', type=Path, help='music file; default is bundled happy_new_year.mp3')
    audio_command.add_argument('--timeline-start-seconds', type=float, default=2.)
    audio_command.add_argument('--fade-in-seconds', type=float, default=.4)
    audio_command.add_argument('--fade-out-seconds', type=float, default=.8)
    report_command = commands.add_parser('report', help='generate an evidence-backed Chinese design PDF and flowcharts')
    report_command.add_argument('--run-dir', type=Path, required=True)
    report_command.add_argument('--output', type=Path, default=Path('outputs/new_year_report/report.pdf'))
    report_command.add_argument('--team-id', default='ROBOT261937J')
    submit_command = commands.add_parser('submit', help='package complete evidence, report and whitelisted source')
    submit_command.add_argument('--run-dir', type=Path, required=True)
    submit_command.add_argument('--output', type=Path, default=Path('submission'))
    submit_command.add_argument('--source-root', type=Path, help='source tree for ordinary installed CLI')
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == 'list':
        dances = yaml.safe_load(resource_path('configs/dances.yaml').read_text())['dances']
        for dance_id, entry in dances.items():
            print(f"{dance_id}\t{entry['name']}\t{entry['fps']:g} FPS")
        return 0
    if args.command == 'add-audio':
        from .audio import add_audio_to_run
        try:
            result = add_audio_to_run(args.run_dir, args.output, args.audio, args.timeline_start_seconds,
                                      args.fade_in_seconds, args.fade_out_seconds)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        except (ValueError, RuntimeError, OSError) as exc:
            print(f'add-audio failed: {exc}', file=sys.stderr)
            return 1
    if args.command == 'submit':
        from .submission import build_submission
        try:
            identity = yaml.safe_load(resource_path('configs/submission.yaml').read_text())
            result = build_submission(args.run_dir, args.output, identity['team_id'], identity['track_name'], source_root=args.source_root)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        except (ValueError, RuntimeError, OSError) as exc:
            print(f'submit failed: {exc}', file=sys.stderr)
            return 1
    if args.command == 'report':
        from .reporting import build_report
        try:
            result = build_report(args.run_dir, args.output, args.team_id)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        except (ValueError, RuntimeError, OSError) as exc:
            print(f'report failed: {exc}', file=sys.stderr)
            return 1
    from .simulation import Simulation, build_timeline

    try:
        timeline = build_timeline(args.dance)
        duration = timeline.total_duration
        if args.command in ('check', 'record') and args.duration is not None:
            duration = args.duration
            if not math.isfinite(duration) or not 0 <= duration <= timeline.total_duration:
                parser.error('--duration must be finite and between zero and the full timeline duration')
        simulation = Simulation(timeline, mode=args.mode)
        print(f'{args.command}: {args.dance}, mode={args.mode}', file=sys.stderr)
        if args.command == 'record':
            from .recording import record
            result = record(simulation, args.output, width=args.width, height=args.height, duration=args.duration)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 1 if result['simulation']['fallen'] else 0
        if args.command == 'check':
            simulation.advance_to(duration)
        else:
            try:
                _play(simulation)
            except KeyboardInterrupt:
                pass
        result = simulation.summary()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result['fallen'] else 0
    except KeyboardInterrupt:
        print('recording interrupted; see run.json for status', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'{args.command} failed: {exc}', file=sys.stderr)
        return 1
