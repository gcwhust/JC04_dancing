"""Generate a portable Chinese competition report from verified recording evidence.

Recording code hashes are historical provenance, not a requirement that the report
CLI be byte-identical to the CLI that produced the accepted video.
"""
from __future__ import annotations

import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
from string import Template
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np
import yaml
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph,
                               SimpleDocTemplate, Spacer)

from .resources import resource_path
from .simulation import build_timeline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError(f'Evidence file must be relative to run directory: {name}')
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Evidence file escapes run directory: {name}')
    if not path.is_file():
        raise FileNotFoundError(f'Missing evidence file: {name}')
    return path


def _finite_tree(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Evidence contains non-finite numbers')


def _validated_run(run_dir: Path) -> tuple[dict, dict]:
    """Reject incomplete/changed evidence before creating any output artifacts."""
    run_file = _evidence_path(run_dir, 'run.json')
    try:
        data = json.loads(run_file.read_text(encoding='utf-8'))
        _finite_tree(data)
        if data['schema_version'] != 1 or data['status'] != 'complete':
            raise ValueError('run.json must be a complete schema_version 1 recording')
        if data['preview'] is not False or data['timeline_complete'] is not True:
            raise ValueError('run.json must contain the complete timeline, not a preview')
        sim, video, evidence, config = (data[k] for k in ('simulation', 'video', 'evidence', 'config'))
        if data['mode'] not in ('free', 'supported') or data['mode'] != sim['mode']:
            raise ValueError('Inconsistent simulation mode')
        if sim['fallen'] is not False or sim['warnings']:
            raise ValueError('Cannot report a successful dance with falls or simulation warnings')
        if sim['dance_id'] != 'swan_lake':
            raise ValueError('This report documents the current Swan Lake choreography')
        dances = yaml.safe_load(resource_path('configs/dances.yaml').read_text(encoding='utf-8'))
        dance = dances['dances']['swan_lake']
        timeline = build_timeline('swan_lake')
        if (sim['wheel_mode'] != 'driven' or dance['wheel_mode'] != 'driven' or
                timeline.wheel_mode != 'driven' or not timeline.clip.full_body):
            raise ValueError('Swan Lake requires coordinated body motion and driven wheels')
        if sim['timestep_s'] != .0005 or config['scene']['timestep'] != sim['timestep_s']:
            raise ValueError('Competition report requires the corrected 0.0005 s timestep')
        if data['camera']['physics_recentered'] is not False:
            raise ValueError('Report requires physical, non-recentered evidence')
        motion = evidence['motion']
        if (motion['source'] != dance['file'] or motion['source_fps'] != dance['fps'] or
                motion['generated_choreography'] is not True or motion['interpolation'] != 'quintic' or
                motion['wheel_units'] != 'rad/s' or motion['phrases_source'] != dance['phrases']):
            raise ValueError('Unverified generated Swan Lake motion evidence')
        total = sum(motion[key] for key in ('clip_duration_s', 'prep_s', 'return_s', 'hold_s'))
        expected_timing = {'clip_start_s': timeline.clip.start, 'clip_duration_s': timeline.clip.duration,
                           'prep_s': timeline.prep, 'return_s': timeline.return_time, 'hold_s': timeline.hold}
        if (not 20 <= total <= 120 or
                any(abs(motion[key] - value) > 1e-6 for key, value in expected_timing.items())):
            raise ValueError('Unexpected Swan Lake timing')
        sections = _choreography_sections(dance['phrases'], timeline.prep, timeline.clip.duration)
        if any(abs(total - duration) > 1e-6 for duration in
               (sim['timeline_duration_s'], sim['requested_time_s'], video['requested_duration_s'])):
            raise ValueError('Run/video timeline mismatch')
        if not total <= sim['simulation_time_s'] <= total + sim['timestep_s'] + 1e-6:
            raise ValueError('Physics did not reach the full timeline endpoint')
        fps, width, height = video['fps'], video['width'], video['height']
        if fps <= 0 or any(type(value) is not int or value <= 0 for value in (width, height)):
            raise ValueError('Invalid video dimensions or FPS')
        count = math.ceil(total * fps)
        if any(video[key] != count for key in ('expected_frames', 'frames_written', 'decoded_frames')):
            raise ValueError('Video frame counts mismatch')
        if abs(video['encoded_duration_s'] - count / fps) > 1e-6:
            raise ValueError('Video encoded duration mismatch')
        if (video['decoded_size'] != [width, height] or video['decoded_codec'] != 'h264' or
                abs(video['decoded_fps'] - fps) > 1e-6):
            raise ValueError('Video decoded metadata mismatch')
        audio = data['audio']
        if type(audio['present']) is not bool or audio['D04_verified'] is not False:
            raise ValueError('Audio presence must be explicit; D04 synchronization is not verified')
        if video['audio_stream_present'] is not audio['present']:
            raise ValueError('Audio stream metadata disagrees with run audio presence')
        expected_status = 'added' if audio['present'] else 'postproduction_pending'
        if audio['status'] != expected_status:
            raise ValueError('Audio status disagrees with run audio presence')
        resources = evidence['resource_sha256']
        required = {'configs/robot.yaml', 'configs/dances.yaml', 'assets/provenance.json',
                    'assets/robot/robot.xml', motion['source'], dance['phrases']}
        if not required <= resources.keys():
            raise ValueError('Missing model/config/motion resource hashes')
        for name, expected in resources.items():
            if _sha256(resource_path(name)) != expected:
                raise ValueError(f'Changed recording resource: {name}')
        if not evidence['code_sha256'] or any(not re.fullmatch('[0-9a-f]{64}', value)
                                               for value in evidence['code_sha256'].values()):
            raise ValueError('Invalid historical recording code hashes')
        # Keep integrity checks even though the concise PDF omits diagnostics.
        for key in ('steps', 'max_abs_torque_nm', 'max_joint_error_rad', 'min_base_height_m',
                    'max_base_tilt_deg', 'min_contact_distance_m', 'max_horizontal_displacement_m',
                    'max_wheel_rotation_rad', 'max_wheel_speed_rad_s'):
            if type(sim[key]) not in (int, float):
                raise ValueError(f'Invalid simulation metric: {key}')
        sim['joint_range_overrides']
        for key in ('model', 'actuators', 'rest_deg', 'scene'):
            config[key]
        for key in ('python', 'mujoco', 'numpy', 'Pillow', 'imageio-ffmpeg', 'ffmpeg'):
            data['versions'][key]
        video_path = _evidence_path(run_dir, video['file'])
        if _sha256(video_path) != video['sha256']:
            raise ValueError('Video SHA256 differs from run.json')
        required_labels = {'start', 'end', *[section['id'] for section in sections]}
        if not required_labels <= {still['label'] for still in data['stills']}:
            raise ValueError('Missing actual choreography section stills')
        expected_samples = {'start': 0, 'end': count - 1}
        expected_samples.update({section['id']: round(sum(section['timeline_time_s']) / 2 * fps)
                                 for section in sections})
        labels = [still['label'] for still in data['stills']]
        if len(labels) != len(set(labels)):
            raise ValueError('Duplicate still labels in recording evidence')
        frames = {}
        for still in data['stills']:
            index = still['frame']
            if still['label'] in expected_samples and index != expected_samples[still['label']]:
                raise ValueError(f"Motion still frame does not match documented action: {still['label']}")
            if type(index) is not int or not 0 <= index < count or abs(still['sample_time_s'] - index / fps) > 1e-6:
                raise ValueError('Invalid still frame or sample time')
            path = _evidence_path(run_dir, still['file'])
            with PILImage.open(path) as image:
                image.load()
                if image.size != (width, height):
                    raise ValueError('Still size differs from video frame')
            frames.setdefault(index, []).append(path)
        reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt='rgb24', output_params=['-threads', '2'])
        try:
            info = next(reader)
            if info['size'] != (width, height) or info['codec'] != 'h264' or abs(info['fps'] - fps) > 1e-6:
                raise ValueError('Actual video format differs from metadata')
            decoded = 0
            for decoded, frame in enumerate(reader, 1):
                for path in frames.get(decoded - 1, []):
                    with PILImage.open(path) as image:
                        if image.convert('RGB').tobytes() != frame:
                            raise ValueError(f'Still does not match actual video frame: {path.name}')
            if decoded != count or abs(info['duration'] - count / fps) > 1 / fps:
                raise ValueError('Actual decoded frame count/duration mismatch')
        finally:
            reader.close()
        probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-i', str(video_path)],
                               capture_output=True, text=True)
        stream_present = 'Audio:' in probe.stderr
        if stream_present is not audio['present']:
            raise ValueError('Actual audio stream differs from run metadata')
        checks = {'complete_timeline': True, 'video_sha256_matches': True,
                  'resources_match': True, 'decoded_frames': decoded,
                  'stills_match_video': True, 'audio_stream_present': stream_present,
                  'audio_stream_absent': not stream_present}
        if stream_present:
            decoded_audio = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(video_path),
                 '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '8000', '-f', 'f32le', 'pipe:1'],
                capture_output=True)
            if decoded_audio.returncode or len(decoded_audio.stdout) % 4:
                raise ValueError('Audio could not be decoded')
            samples = np.frombuffer(decoded_audio.stdout, dtype='<f4')
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.
            if not np.all(np.isfinite(samples)) or not math.isfinite(rms) or rms <= 1e-8:
                raise ValueError('Audio stream must contain nonzero decoded audio samples')
            checks['decoded_audio_rms'] = rms
        return data, checks
    except (KeyError, TypeError, IndexError, StopIteration) as exc:
        raise ValueError(f'Invalid or missing run.json evidence field: {exc}') from exc


def _fonts() -> None:
    for name, file in [('ReportCJK', 'DroidSansFallbackFull.ttf'), ('ReportLatin', 'DejaVuSans.ttf')]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(resource_path('assets/report/fonts/' + file))))


def _markup(text: str) -> str:
    # Droid Fallback deliberately contains no Latin letters; choose the bundled
    # Latin font per Unicode range instead of silently drawing missing glyphs.
    runs = re.findall(r'[^\u2e80-\uffff]+|[\u2e80-\uffff]+', str(text))
    return ''.join(f'<font name="{"ReportLatin" if ord(run[0]) < 0x2e80 else "ReportCJK"}">{escape(run)}</font>'
                   for run in runs)


def render_flowchart(output_pdf: Path, output_png: Path) -> None:
    """Draw six readable steps with the same blue/teal palette as the report."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch

    paths = [resource_path('assets/report/fonts/' + name)
             for name in ('DejaVuSans.ttf', 'DroidSansFallbackFull.ttf')]
    for path in paths:
        font_manager.fontManager.addfont(str(path))
    families = [font_manager.FontProperties(fname=str(path)).get_name() for path in paths]
    steps = [
        ('准备环境', '安装依赖，准备模型与动作文件'),
        ('加载模型', '导入机器人，设置初始姿态与场景'),
        ('读取动作', '载入关节轨迹，衔接准备与回位'),
        ('控制仿真', '按时间更新目标，以 PD 力矩驱动关节'),
        ('录制画面', '设置镜头，渲染并导出舞蹈视频'),
        ('配乐合成', '加入音轨，检查成片并整理交付'),
    ]
    with plt.rc_context({'font.family': families, 'pdf.fonttype': 42, 'axes.unicode_minus': False}):
        fig, ax = plt.subplots(figsize=(7.2, 7.0), facecolor='white')
        ax.set(xlim=(0, 10), ylim=(0, 10))
        ax.axis('off')
        ax.text(5, 9.65, '从模型到舞蹈视频', ha='center', fontsize=19, color='#163d54')
        for index, (title, detail) in enumerate(steps):
            y = 8.6 - index * 1.48
            if index:
                ax.annotate('', xy=(5, y + .55), xytext=(5, y + .93),
                            arrowprops={'arrowstyle': '-|>', 'color': '#4b9b9d', 'lw': 1.6})
            ax.add_patch(FancyBboxPatch((1.15, y - .49), 7.7, .98,
                                       boxstyle='round,pad=0.06,rounding_size=0.13',
                                       facecolor='#eff7f8', edgecolor='#cfe3e8', linewidth=.9))
            ax.text(1.72, y, f'{index + 1:02}', va='center', ha='center',
                    fontsize=18, color='#28868c')
            ax.text(2.35, y + .16, title, va='center', fontsize=13.5, color='#163d54')
            ax.text(2.35, y - .19, detail, va='center', fontsize=10.5, color='#526b79')
        fig.subplots_adjust(left=.015, right=.985, top=.99, bottom=.01)
        try:
            for path in (Path(output_pdf), Path(output_png)):
                path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_pdf, metadata={'Title': '小楚机器人操作流程'})
            fig.savefig(output_png, dpi=180)
        finally:
            plt.close(fig)


def _choreography_sections(source: str, prep: float, duration: float) -> list[dict]:
    """Read the generated score's actual sections, without inventing motion classes."""
    phrases = json.loads(resource_path(source).read_text(encoding='utf-8'))
    _finite_tree(phrases)
    if (phrases['schema_version'] != 1 or phrases['dance_id'] != 'swan_lake' or
            abs(phrases['duration_s'] - duration) > 1e-6 or phrases['wheel_mode'] != 'driven'):
        raise ValueError('Choreography phrase map differs from the recorded dance')
    result, labels, endpoint = [], set(), 0.0
    for section in phrases['sections']:
        start, end = section['start_s'], section['end_s']
        if (section['id'] in labels or not section['id'] or not section['name'] or
                abs(start - endpoint) > 1e-6 or not start < end <= duration):
            raise ValueError('Choreography sections must cover the complete dance in order')
        labels.add(section['id'])
        endpoint = end
        result.append({'id': section['id'], 'name': section['name'], 'body_time_s': [start, end],
                       'timeline_time_s': [prep + start, prep + end], 'description': section['description']})
    if not result or abs(endpoint - duration) > 1e-6:
        raise ValueError('Choreography sections do not reach the dance endpoint')
    return result


def _write_report(data: dict, run_dir: Path, target: Path, team_id: str, flowchart: Path) -> list[dict]:
    """Render only the three requested sections; evidence stays in report.json."""
    _fonts()
    ink, muted = colors.HexColor('#163d54'), colors.HexColor('#526b79')
    body = ParagraphStyle('body', fontName='ReportCJK', fontSize=11, leading=20,
                          textColor=ink, wordWrap='CJK', spaceAfter=12, splitLongWords=True)
    heading = ParagraphStyle('heading', parent=body, fontSize=17, leading=25,
                             spaceBefore=17, spaceAfter=14)
    title = ParagraphStyle('title', parent=heading, fontSize=25, leading=34,
                           spaceBefore=0, spaceAfter=24)
    caption = ParagraphStyle('caption', parent=body, fontSize=9, leading=15,
                             textColor=muted, alignment=TA_CENTER, spaceAfter=0)
    usable_width = A4[0] - 96

    def paragraph(text, style=body):
        return Paragraph(_markup(text), style)

    still = next(item for item in data['stills'] if item['label'] == 'start')
    image_path = run_dir / still['file']
    with PILImage.open(image_path) as image:
        ratio = image.height / image.width
    image_width = min(usable_width, 292 / ratio)
    image_height = image_width * ratio
    values = {
        'platform': f"仿真平台采用 MuJoCo {data['versions']['mujoco']}，用于计算关节运动、重力及地面接触，并生成三维画面。",
        'model': '机器人模型采用 jc04_desc4 V1.3，由 URDF 转换为 MuJoCo 使用的 MJCF，保留机器人外观与关节结构。舞蹈协调双臂、髋膝与双轮共 12 个主动自由度，完成展翼、屈伸、滑行与转向。',
        'model_image': Image(str(image_path), width=image_width, height=image_height),
        'model_caption': paragraph('图 1　小楚机器人仿真模型', caption),
        'flowchart': Image(str(flowchart), width=usable_width, height=usable_width * 7.0 / 7.2),
    }
    story = []
    for line in resource_path('assets/report/template.md').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        if line == '---':
            story.append(PageBreak())
        elif line.startswith('# '):
            story.append(paragraph(line[2:], title))
        elif line.startswith('## '):
            story.append(paragraph(line[3:], heading))
        elif line.startswith('$') and line[1:] in values and not isinstance(values[line[1:]], str):
            story.extend([values[line[1:]], Spacer(1, 9)])
        else:
            story.append(paragraph(Template(line).substitute(values)))

    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor('#d5e5e9'))
        canvas.line(48, 40, A4[0] - 48, 40)
        footer_text = paragraph(f'小楚机器人舞蹈仿真　 /　 {document.page:02}', caption)
        footer_text.wrap(usable_width, 20)
        footer_text.drawOn(canvas, 48, 22)
        canvas.setFillColor(colors.HexColor('#28868c'))
        canvas.roundRect(48, A4[1] - 32, 34, 3, 1.5, fill=1, stroke=0)

    SimpleDocTemplate(str(target), pagesize=A4, rightMargin=48, leftMargin=48,
                      topMargin=49, bottomMargin=56, title='小楚机器人舞蹈仿真说明',
                      author=team_id, pageCompression=1).build(story, onFirstPage=footer, onLaterPages=footer)
    return [{**still, 'sha256': _sha256(image_path)}]


def build_report(run_dir: Path, output: Path, team_id: str = 'ROBOT261937J') -> dict:
    """Build report.pdf, report.json and standalone flowchart.pdf/png.

    Outputs are always separate from immutable evidence. Missing evidence, failed
    recordings, previews, changed assets/video or altered stills are rejected.
    The current generated Swan Lake score supplies its own section timings.
    """
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.suffix.lower() != '.pdf' or output.name == 'flowchart.pdf':
        raise ValueError('Report output must be a .pdf file distinct from flowchart.pdf')
    if output.is_relative_to(run_dir):
        raise ValueError('Report output must be outside the accepted run directory')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', team_id):
        raise ValueError('team_id must contain 1–64 ASCII letters, digits, underscores or hyphens')
    data, checks = _validated_run(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.report-', dir=output.parent) as temporary:
        staging = Path(temporary)
        render_flowchart(staging / 'flowchart.pdf', staging / 'flowchart.png')
        images = _write_report(data, run_dir, staging / output.name, team_id, staging / 'flowchart.png')
        metadata = {
            'schema_version': 1, 'status': 'complete', 'team_id': team_id,
            'source': {'run_dir': run_dir.name, 'run_file': 'run.json',
                       'run_sha256': _sha256(run_dir / 'run.json'), 'video_file': data['video']['file'],
                       'video_sha256': data['video']['sha256']},
            'audio': data['audio'], 'video': data['video'], 'simulation': data['simulation'],
            'config': data['config'], 'versions': data['versions'], 'evidence': data['evidence'],
            'choreography_sections': _choreography_sections(
                data['evidence']['motion']['phrases_source'], data['evidence']['motion']['prep_s'],
                data['evidence']['motion']['clip_duration_s']),
            'images': images, 'checks': checks,
            'limitations': ['D04 music synchronization has not been subjectively verified',
                            'simplified contacts; no body self-collision; idealized wheel velocity motors',
                            'runtime joint ranges do not validate hardware ROM', 'separate scoring rules not provided'],
            'artifacts': {key: {'file': name, 'sha256': _sha256(staging / name)} for key, name in
                          [('report_pdf', output.name), ('flowchart_pdf', 'flowchart.pdf'), ('flowchart_png', 'flowchart.png')]},
            'report_resources_sha256': {name: _sha256(path) for path in sorted(resource_path('assets/report').rglob('*'))
                                       if path.is_file() for name in [str(path.relative_to(resource_path('.')))]},
        }
        (staging / output.with_suffix('.json').name).write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        for name in [output.name, output.with_suffix('.json').name, 'flowchart.pdf', 'flowchart.png']:
            (staging / name).replace(output.parent / name)
    return metadata
