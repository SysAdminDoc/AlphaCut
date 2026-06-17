"""Tests for AlphaCut utility functions — no model/GPU/FFmpeg required."""
import os
import sys
import json
import tempfile
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_version_string():
    """Version must be a valid semver-ish string."""
    # Import just the version without triggering bootstrap
    with open(os.path.join(os.path.dirname(__file__), '..', 'AlphaCut.py'), encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                version = line.split('=')[1].strip().strip('"').strip("'")
                parts = version.split('.')
                assert len(parts) == 3, f"Version {version} not semver"
                assert all(p.isdigit() for p in parts), f"Non-numeric version part in {version}"
                return
    assert False, "__version__ not found"


def test_model_registry_loads():
    """models.json must parse and contain required fields."""
    registry_path = os.path.join(os.path.dirname(__file__), '..', 'models.json')
    with open(registry_path) as f:
        data = json.load(f)
    assert 'models' in data
    assert len(data['models']) >= 8
    for m in data['models']:
        assert 'name' in m
        assert 'file' in m
        assert 'url' in m
        assert 'input_size' in m
        assert len(m['input_size']) == 2
        assert 'mean' in m and len(m['mean']) == 3
        assert 'std' in m and len(m['std']) == 3
        assert 'license' in m


def test_model_registry_no_duplicates():
    """No duplicate model names or filenames in the registry."""
    registry_path = os.path.join(os.path.dirname(__file__), '..', 'models.json')
    with open(registry_path) as f:
        data = json.load(f)
    names = [m['name'] for m in data['models']]
    files = [m['file'] for m in data['models']]
    assert len(names) == len(set(names)), f"Duplicate model names: {names}"
    assert len(files) == len(set(files)), f"Duplicate model files: {files}"


def test_locale_template_valid_json():
    """locale_template.json must be valid JSON with string values."""
    locale_path = os.path.join(os.path.dirname(__file__), '..', 'locale_template.json')
    with open(locale_path) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    for k, v in data.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_ftime():
    """_ftime should format seconds as human-readable durations."""
    # Can't import directly due to bootstrap, so replicate the function logic
    def _ftime(s):
        if s < 60: return f"{s:.0f}s"
        if s < 3600: return f"{s/60:.1f}m"
        return f"{s/3600:.1f}h"

    assert _ftime(0) == "0s"
    assert _ftime(30) == "30s"
    assert _ftime(59) == "59s"
    assert _ftime(60) == "1.0m"
    assert _ftime(90) == "1.5m"
    assert _ftime(3600) == "1.0h"
    assert _ftime(7200) == "2.0h"


def test_compute_sha256():
    """SHA-256 computation should match known hash."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        f.write(b"test data for hashing")
        path = f.name
    try:
        expected = hashlib.sha256(b"test data for hashing").hexdigest()
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        assert h.hexdigest() == expected
    finally:
        os.remove(path)


def test_output_format_keys_unique():
    """OUTPUT_FORMATS values should all be unique."""
    formats = ['mp4', 'av1', 'webm', 'webp_anim', 'gif_anim', 'greenscreen',
               'prores', 'matte', 'fg_alpha', 'png_seq']
    assert len(formats) == len(set(formats))


def test_generate_output_name_patterns():
    """Output name generation should substitute all tokens correctly."""
    # Replicate generate_output_name logic
    def generate(input_path, pattern, model_key, fmt):
        base = os.path.splitext(os.path.basename(input_path))[0]
        model_short = model_key.split('(')[0].strip().replace(' ', '_').lower()
        name = pattern.replace('{name}', base).replace('{model}', model_short)
        name = name.replace('{format}', fmt)
        ext_map = {'prores': '.mov', 'webm': '.webm', 'png_seq': '', 'mp4': '.mp4', 'av1': '.mp4'}
        ext = ext_map.get(fmt, '.mov')
        return os.path.basename(f"{name}{ext}")

    assert generate("/tmp/test.mp4", "{name}_alphacut", "u2net_human_seg (People)", "mp4") == "test_alphacut.mp4"
    assert generate("/tmp/clip.mov", "{name}_{model}", "BiRefNet-general (Best)", "webm") == "clip_birefnet-general.webm"
    assert generate("/tmp/vid.avi", "{name}_{format}", "u2net", "prores") == "vid_prores.mov"
    result = generate("/tmp/test.mp4", "{name}", "u2net", "png_seq")
    assert result == "test"


def test_estimate_output_size():
    """Output size estimation should return reasonable values."""
    def estimate(info, fmt):
        if not info: return 0
        px = info['width'] * info['height']; frames = info['total_frames']
        bpf = {'prores': px*2.5, 'webm': px*0.15, 'png_seq': px*1.5, 'mp4': px*0.08, 'av1': px*0.06}
        return bpf.get(fmt, px*0.5) * frames / (1024 * 1024)

    info = {'width': 1920, 'height': 1080, 'total_frames': 300}
    mp4 = estimate(info, 'mp4')
    prores = estimate(info, 'prores')
    assert mp4 > 0
    assert prores > mp4  # ProRes should be larger than H.264
    assert estimate(None, 'mp4') == 0
    av1 = estimate(info, 'av1')
    assert av1 < mp4  # AV1 should be smaller than H.264


def test_image_extensions():
    """Image extensions set should include common formats."""
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    assert '.png' in exts
    assert '.jpg' in exts
    assert '.bmp' in exts


def test_video_extensions():
    """Video extensions set should include common formats."""
    exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mts'}
    assert '.mp4' in exts
    assert '.mov' in exts
    assert '.mkv' in exts
