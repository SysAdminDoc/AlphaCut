"""Generated-media CLI integration tests with a lightweight fake model."""
import json
import os
import subprocess
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeEngine:
    def reset_temporal(self):
        pass

    def predict_mask(self, pil_img, roi=None):
        return Image.new('L', pil_img.size, 255)

    def refine_mask(self, mask, edge_softness, mask_shift, temporal_smooth):
        return mask


def _json_events(output):
    events = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            events.append(json.loads(line))
    return events


@pytest.fixture
def alphacut(monkeypatch):
    import AlphaCut

    monkeypatch.setattr(AlphaCut, 'get_engine', lambda *args, **kwargs: FakeEngine())
    AlphaCut._engine_cache['key'] = None
    AlphaCut._engine_cache['engine'] = None
    return AlphaCut


def _run_json_cli(alphacut, capsys, cli_args, expect_exit=None):
    args = alphacut.build_parser().parse_args([*cli_args, '--json'])
    args.audio = not args.no_audio
    if expect_exit is None:
        alphacut.run_cli(args)
    else:
        with pytest.raises(SystemExit) as exc:
            alphacut.run_cli(args)
        assert exc.value.code == expect_exit
    return _json_events(capsys.readouterr().out)


def _event_types(events):
    return [event.get('type') for event in events]


def _assert_order(events, *types):
    event_types = _event_types(events)
    positions = [event_types.index(kind) for kind in types]
    assert positions == sorted(positions), event_types


def _make_image(path):
    img = Image.new('RGB', (16, 16), (10, 20, 30))
    for x in range(4, 12):
        for y in range(4, 12):
            img.putpixel((x, y), (220, 40, 20))
    img.save(path)
    return path


def _make_video(alphacut, path):
    ffmpeg = alphacut.find_ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg not available")
    cmd = [
        ffmpeg, '-y', '-v', 'error',
        '-f', 'lavfi', '-i', 'testsrc=size=16x16:rate=2:duration=1',
        '-pix_fmt', 'yuv420p', str(path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return path


def test_image_cli_writes_png_and_json_order(alphacut, capsys, tmp_path):
    source = _make_image(tmp_path / 'source.png')
    output = tmp_path / 'cutout.png'

    events = _run_json_cli(alphacut, capsys, ['-i', str(source), '-o', str(output), '-y'])

    assert output.is_file()
    result = Image.open(output)
    assert result.mode == 'RGBA'
    assert result.size == (16, 16)
    assert result.getpixel((8, 8))[3] == 255
    assert _event_types(events)[-1] == 'complete'
    assert not any(event.get('type') in {'error', 'failed'} for event in events)
    _assert_order(events, 'start', 'done', 'complete')


def test_video_cli_encodes_generated_fixture_and_json_order(alphacut, capsys, tmp_path):
    source = _make_video(alphacut, tmp_path / 'source.mp4')
    output = tmp_path / 'cutout.mp4'

    events = _run_json_cli(
        alphacut, capsys,
        ['-i', str(source), '-o', str(output), '-f', 'mp4', '--no-audio', '-y'],
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert _event_types(events)[-1] == 'complete'
    assert not any(event.get('type') in {'error', 'failed'} for event in events)
    _assert_order(events, 'start', 'progress', 'done', 'complete')


def test_video_cli_encode_failure_reports_failed_event(alphacut, capsys, monkeypatch, tmp_path):
    source = _make_video(alphacut, tmp_path / 'source.mp4')
    output = tmp_path / 'broken.mp4'

    monkeypatch.setattr(alphacut.ProcessingWorker, '_encode',
                        lambda self, ffmpeg, frames_dir, fps, info, total_frames=0: None)

    events = _run_json_cli(
        alphacut, capsys,
        ['-i', str(source), '-o', str(output), '-f', 'mp4', '--no-audio', '-y'],
        expect_exit=1,
    )

    assert not output.exists()
    assert _event_types(events)[-1] == 'failed'
    assert any(event.get('type') == 'error' and 'Encoding failed' in event.get('message', '')
               for event in events)
    assert not any(event.get('type') == 'complete' for event in events)
    _assert_order(events, 'start', 'error', 'failed')
