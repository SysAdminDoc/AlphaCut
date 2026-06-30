"""Tests for AlphaCut CLI argument parsing — no model/GPU/FFmpeg required."""
import argparse
import ast
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALPHACUT = os.path.join(_REPO_ROOT, 'AlphaCut.py')
_README = os.path.join(_REPO_ROOT, 'README.md')


def _build_parser():
    """Extract AlphaCut.build_parser() without triggering bootstrap or GUI imports."""
    with open(_ALPHACUT, encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)
    ns = {'argparse': argparse, '__builtins__': __builtins__}
    wanted_assignments = {'__version__', 'OUTPUT_FORMATS', 'HARDWARE_OUTPUT_FORMATS',
                          'ALL_OUTPUT_FORMAT_VALUES'}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if target_names & wanted_assignments:
                exec(compile(ast.Module([node], []), _ALPHACUT, 'exec'), ns)
        elif isinstance(node, ast.FunctionDef) and node.name == 'build_parser':
            exec(compile(ast.Module([node], []), _ALPHACUT, 'exec'), ns)
            return ns['build_parser']()
    raise AssertionError("build_parser() not found in AlphaCut.py")


def _parser_options(parser):
    return {opt for action in parser._actions for opt in action.option_strings}


def _format_choices(parser):
    for action in parser._actions:
        if '--format' in action.option_strings:
            return tuple(action.choices)
    raise AssertionError("--format action not found")


def _read_source(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _readme_cli_flags():
    readme = _read_source(_README)
    section = readme.split('## CLI Reference', 1)[1].split('## AI Models', 1)[0]
    return set(re.findall(r'`(-{1,2}[A-Za-z][A-Za-z0-9-]*)`', section))


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
    assert args.runtime_info is False
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
    choices = _format_choices(parser)
    assert len(choices) == len(set(choices))
    assert {'mp4', 'hevc', 'av1', 'webm', 'prores', 'png_seq',
            'mp4_nvenc', 'hevc_nvenc', 'mp4_qsv', 'hevc_qsv'} <= set(choices)
    for fmt in choices:
        args = parser.parse_args(['-f', fmt])
        assert args.format == fmt


def test_main_uses_shared_parser_builder():
    source = _read_source(_ALPHACUT)
    main_src = source.split('def main():', 1)[1].split("if __name__ == '__main__':", 1)[0]
    assert 'parser = build_parser()' in main_src
    assert 'argparse.ArgumentParser' not in main_src


def test_readme_cli_flags_exist_in_real_parser():
    missing = _readme_cli_flags() - _parser_options(_build_parser())
    assert not missing


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


def test_runtime_info_flag():
    parser = _build_parser()
    args = parser.parse_args(['--runtime-info'])
    assert args.runtime_info is True


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
