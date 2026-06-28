"""Tests for AlphaCut CLI argument parsing — no model/GPU/FFmpeg required."""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALPHACUT = os.path.join(_REPO_ROOT, 'AlphaCut.py')


def _build_parser():
    """Build the same argument parser as main() without triggering bootstrap."""
    output_formats = ['mp4', 'hevc', 'av1', 'webm', 'webp_anim', 'gif_anim',
                      'greenscreen', 'prores', 'matte', 'fg_alpha', 'png_seq',
                      'mp4_nvenc', 'hevc_nvenc', 'mp4_qsv', 'hevc_qsv']
    parser = argparse.ArgumentParser(prog='AlphaCut')
    parser.add_argument('--input', '-i', nargs='+')
    parser.add_argument('--output', '-o')
    parser.add_argument('--model', '-m', default='u2net_human_seg')
    parser.add_argument('--format', '-f', default='mp4', choices=output_formats)
    parser.add_argument('--quality', '-q', type=int, default=70)
    parser.add_argument('--max-res', type=int, default=0)
    parser.add_argument('--edge', type=int, default=0)
    parser.add_argument('--shift', type=int, default=0)
    parser.add_argument('--temporal', type=int, default=0)
    parser.add_argument('--frame-skip', type=int, default=1)
    parser.add_argument('--invert', action='store_true')
    parser.add_argument('--spill', type=int, default=0)
    parser.add_argument('--spill-color', default='green', choices=['green', 'blue', 'red'])
    parser.add_argument('--shadow', type=int, default=0)
    parser.add_argument('--bg-color')
    parser.add_argument('--bg-image')
    parser.add_argument('--no-audio', action='store_true')
    parser.add_argument('-y', '--overwrite', action='store_true')
    parser.add_argument('--gpu-device', type=int, default=-1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--chroma-key', action='store_true')
    parser.add_argument('--pipe', action='store_true')
    parser.add_argument('--json', action='store_true')
    return parser


def _run_alphacut_cli(*args):
    return subprocess.run(
        [sys.executable, _ALPHACUT, *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _json_events(stdout):
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            events.append(json.loads(line))
    return events


def _assert_json_failure(proc):
    assert proc.returncode != 0, proc.stdout + proc.stderr
    events = _json_events(proc.stdout)
    assert events, proc.stdout + proc.stderr
    assert any(event.get('type') == 'error' for event in events)
    assert events[-1].get('type') == 'failed'
    assert not any(event.get('type') == 'complete' for event in events)
    assert not any(event.get('type') == 'done' for event in events)
    return events


def test_default_args():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.input is None
    assert args.format == 'mp4'
    assert args.model == 'u2net_human_seg'
    assert args.quality == 70
    assert args.gpu_device == -1
    assert args.fp16 is False
    assert args.overwrite is False


def test_basic_input():
    parser = _build_parser()
    args = parser.parse_args(['-i', 'video.mp4', '-f', 'webm'])
    assert args.input == ['video.mp4']
    assert args.format == 'webm'


def test_multiple_inputs():
    parser = _build_parser()
    args = parser.parse_args(['-i', 'a.mp4', 'b.mp4', 'c.mp4'])
    assert args.input == ['a.mp4', 'b.mp4', 'c.mp4']


def test_all_format_choices():
    parser = _build_parser()
    valid = ['mp4', 'hevc', 'av1', 'webm', 'webp_anim', 'gif_anim',
             'greenscreen', 'prores', 'matte', 'fg_alpha', 'png_seq',
             'mp4_nvenc', 'hevc_nvenc', 'mp4_qsv', 'hevc_qsv']
    for fmt in valid:
        args = parser.parse_args(['-f', fmt])
        assert args.format == fmt


def test_invalid_format_rejected():
    parser = _build_parser()
    try:
        parser.parse_args(['-f', 'invalid_format'])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


def test_gpu_device_flag():
    parser = _build_parser()
    args = parser.parse_args(['--gpu-device', '2'])
    assert args.gpu_device == 2


def test_fp16_flag():
    parser = _build_parser()
    args = parser.parse_args(['--fp16'])
    assert args.fp16 is True


def test_overwrite_flag():
    parser = _build_parser()
    args = parser.parse_args(['-y'])
    assert args.overwrite is True
    args2 = parser.parse_args(['--overwrite'])
    assert args2.overwrite is True


def test_pipe_flag():
    parser = _build_parser()
    args = parser.parse_args(['--pipe', '-i', 'test.mp4'])
    assert args.pipe is True


def test_spill_color_choices():
    parser = _build_parser()
    for color in ['green', 'blue', 'red']:
        args = parser.parse_args(['--spill-color', color])
        assert args.spill_color == color


def test_bg_color_passed_as_string():
    parser = _build_parser()
    args = parser.parse_args(['--bg-color', '255,0,128'])
    assert args.bg_color == '255,0,128'


def test_json_flag():
    parser = _build_parser()
    args = parser.parse_args(['--json', '-i', 'test.mp4'])
    assert args.json is True
    args2 = parser.parse_args(['-i', 'test.mp4'])
    assert args2.json is False


def test_json_missing_input_exits_nonzero_without_complete(tmp_path):
    proc = _run_alphacut_cli('-i', str(tmp_path / 'missing.png'), '--json')
    events = _assert_json_failure(proc)
    assert any('File not found' in event.get('message', '') for event in events)


def test_json_output_exists_refuses_success_without_overwrite(tmp_path):
    source = tmp_path / 'photo.png'
    source.write_bytes(b'not a real png')
    output = tmp_path / 'photo_alphacut.png'
    output.write_bytes(b'existing')

    proc = _run_alphacut_cli('-i', str(source), '-o', str(output), '--json')
    events = _assert_json_failure(proc)
    assert any('Output exists' in event.get('message', '') for event in events)


def test_json_invalid_image_exits_nonzero_without_complete(tmp_path):
    source = tmp_path / 'broken.png'
    source.write_bytes(b'not a real png')
    output = tmp_path / 'broken_out.png'

    proc = _run_alphacut_cli('-i', str(source), '-o', str(output), '--json')
    events = _assert_json_failure(proc)
    assert any(event.get('type') == 'start' for event in events)
    assert any('image' in event.get('message', '').lower() or
               'unidentified' in event.get('message', '').lower()
               for event in events)


def test_json_unreadable_video_exits_nonzero_without_complete(tmp_path):
    source = tmp_path / 'broken.mp4'
    source.write_text('not a real video', encoding='utf-8')
    output = tmp_path / 'broken_out.mp4'

    proc = _run_alphacut_cli('-i', str(source), '-o', str(output), '--json')
    events = _assert_json_failure(proc)
    assert any('Cannot read video' in event.get('message', '') or
               'FFmpeg not found' in event.get('message', '')
               for event in events)


def test_quality_range():
    parser = _build_parser()
    args = parser.parse_args(['-q', '0'])
    assert args.quality == 0
    args = parser.parse_args(['-q', '100'])
    assert args.quality == 100
