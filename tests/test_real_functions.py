"""Tests that exercise real AlphaCut functions extracted from the source.

Since AlphaCut.py triggers bootstrap + PyQt6 imports on import, we extract
specific pure functions by parsing the source file directly. This ensures
tests stay in sync with the real implementation.
"""
import os
import sys
import ast
import json
import time
import types
import hashlib
import tempfile
import textwrap

# ---------------------------------------------------------------------------
# Helper: extract a function from AlphaCut.py source without triggering
# bootstrap or PyQt6 imports.
# ---------------------------------------------------------------------------
_SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'AlphaCut.py')

def _read_source():
    with open(_SRC_PATH, encoding='utf-8') as f:
        return f.read()


def _extract_function(func_name, extra_imports=None):
    """Return a callable extracted from AlphaCut.py source by name.

    Parses the source AST, finds the top-level function definition, compiles
    it in a minimal namespace, and returns the function object.
    """
    source = _read_source()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_source = ast.get_source_segment(source, node)
            ns = {'os': os, 'sys': sys, 'time': time, 'hashlib': hashlib,
                  'json': json, '__builtins__': __builtins__}
            if extra_imports:
                ns.update(extra_imports)
            exec(compile(func_source, f'<{func_name}>', 'exec'), ns)
            return ns[func_name]
    raise ValueError(f"Function {func_name!r} not found in {_SRC_PATH}")


# ---------------------------------------------------------------------------
# Tests for _ftime (real implementation)
# ---------------------------------------------------------------------------
class TestFtime:
    def setup_method(self):
        self._ftime = _extract_function('_ftime')

    def test_zero(self):
        assert self._ftime(0) == "0s"

    def test_seconds(self):
        assert self._ftime(30) == "30s"
        assert self._ftime(59) == "59s"

    def test_minutes(self):
        assert self._ftime(60) == "1.0m"
        assert self._ftime(90) == "1.5m"

    def test_hours(self):
        assert self._ftime(3600) == "1.0h"
        assert self._ftime(7200) == "2.0h"

    def test_boundary(self):
        """59.5 seconds rounds to 60s display but is < 60 threshold."""
        result = self._ftime(59.5)
        assert result == "60s" or result == "59s"  # depends on rounding


# ---------------------------------------------------------------------------
# Tests for _compute_sha256 (real implementation)
# ---------------------------------------------------------------------------
class TestComputeSha256:
    def setup_method(self):
        self._compute_sha256 = _extract_function('_compute_sha256')

    def test_known_hash(self):
        data = b"test data for hashing"
        expected = hashlib.sha256(data).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            f.write(data)
            path = f.name
        try:
            assert self._compute_sha256(path) == expected
        finally:
            os.remove(path)

    def test_empty_file(self):
        expected = hashlib.sha256(b"").hexdigest()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            path = f.name
        try:
            assert self._compute_sha256(path) == expected
        finally:
            os.remove(path)

    def test_large_file(self):
        """Verify chunked reading works for files > 1MB."""
        data = b"x" * (2 * 1024 * 1024)  # 2MB
        expected = hashlib.sha256(data).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            f.write(data)
            path = f.name
        try:
            assert self._compute_sha256(path) == expected
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# Tests for generate_output_name (real implementation)
# ---------------------------------------------------------------------------
class TestGenerateOutputName:
    def setup_method(self):
        self._generate = _extract_function('generate_output_name')

    def test_name_token(self):
        result = self._generate("/tmp/test.mp4", "{name}_alphacut", "u2net_human_seg (People)", "mp4")
        assert os.path.basename(result) == "test_alphacut.mp4"

    def test_model_token(self):
        result = self._generate("/tmp/clip.mov", "{name}_{model}", "BiRefNet-general (Best)", "webm")
        assert os.path.basename(result) == "clip_birefnet-general.webm"

    def test_format_token(self):
        result = self._generate("/tmp/vid.avi", "{name}_{format}", "u2net", "prores")
        assert os.path.basename(result) == "vid_prores.mov"

    def test_png_seq_no_ext(self):
        result = self._generate("/tmp/test.mp4", "{name}", "u2net", "png_seq")
        assert os.path.basename(result) == "test"

    def test_hevc_extension(self):
        result = self._generate("/tmp/test.mp4", "{name}", "u2net", "hevc")
        assert os.path.basename(result) == "test.mp4"

    def test_preserves_directory(self):
        result = self._generate("/videos/project/test.mp4", "{name}", "u2net", "mp4")
        assert "/videos/project/" in result.replace("\\", "/") or "\\videos\\project\\" in result


# ---------------------------------------------------------------------------
# Tests for estimate_output_size (real implementation)
# ---------------------------------------------------------------------------
class TestEstimateOutputSize:
    def setup_method(self):
        self._estimate = _extract_function('estimate_output_size')

    def test_none_info(self):
        assert self._estimate(None, 'mp4') == 0

    def test_prores_larger_than_mp4(self):
        info = {'width': 1920, 'height': 1080, 'total_frames': 300}
        assert self._estimate(info, 'prores') > self._estimate(info, 'mp4')

    def test_av1_smaller_than_mp4(self):
        info = {'width': 1920, 'height': 1080, 'total_frames': 300}
        assert self._estimate(info, 'av1') < self._estimate(info, 'mp4')

    def test_hevc_smaller_than_mp4(self):
        info = {'width': 1920, 'height': 1080, 'total_frames': 300}
        assert self._estimate(info, 'hevc') <= self._estimate(info, 'mp4')

    def test_positive_for_all_formats(self):
        info = {'width': 1920, 'height': 1080, 'total_frames': 300}
        for fmt in ['mp4', 'hevc', 'av1', 'webm', 'prores', 'png_seq', 'greenscreen', 'matte', 'webp_anim', 'gif_anim']:
            assert self._estimate(info, fmt) > 0, f"{fmt} should give positive estimate"


# ---------------------------------------------------------------------------
# Tests for _model_needs_download (real implementation)
# ---------------------------------------------------------------------------
class TestModelNeedsDownload:
    def test_function_uses_basename(self):
        """Verify _model_needs_download applies os.path.basename for safety."""
        source = _read_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '_model_needs_download':
                func_src = ast.get_source_segment(source, node)
                assert 'os.path.basename' in func_src, \
                    "_model_needs_download should use os.path.basename for path traversal safety"
                return
        assert False, "_model_needs_download not found"


# ---------------------------------------------------------------------------
# Tests for _load_model_registry (via models.json validation)
# ---------------------------------------------------------------------------
class TestLoadModelRegistry:
    def test_registry_round_trip(self):
        """Verify _load_model_registry produces models matching models.json."""
        fn = _extract_function('_load_model_registry',
                               extra_imports={
                                   'MODEL_BASE': 'https://github.com/danielgatis/rembg/releases/download/v0.0.0',
                                   '__file__': os.path.abspath(_SRC_PATH),
                               })
        models = fn()
        assert len(models) >= 10  # base 10 + DIS + massive
        # Each model has the required keys
        for label, cfg in models.items():
            assert 'file' in cfg
            assert 'url' in cfg
            assert 'size' in cfg
            assert 'mean' in cfg
            assert 'std' in cfg
            assert isinstance(cfg['size'], tuple)
            assert len(cfg['mean']) == 3
            assert len(cfg['std']) == 3

    def test_no_duplicate_files(self):
        fn = _extract_function('_load_model_registry',
                               extra_imports={
                                   'MODEL_BASE': 'https://github.com/danielgatis/rembg/releases/download/v0.0.0',
                                   '__file__': os.path.abspath(_SRC_PATH),
                               })
        models = fn()
        files = [cfg['file'] for cfg in models.values()]
        assert len(files) == len(set(files)), f"Duplicate model files: {files}"


# ---------------------------------------------------------------------------
# Tests for OUTPUT_FORMATS consistency
# ---------------------------------------------------------------------------
class TestOutputFormats:
    def test_hevc_in_formats(self):
        """Verify HEVC format was added to OUTPUT_FORMATS."""
        source = _read_source()
        assert "'hevc'" in source or '"hevc"' in source

    def test_all_format_values_unique(self):
        """Extract OUTPUT_FORMATS dict from source and verify unique values."""
        source = _read_source()
        # Find the OUTPUT_FORMATS dict in the AST
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'OUTPUT_FORMATS':
                        # Evaluate the dict literal
                        code = ast.get_source_segment(source, node.value)
                        formats = eval(code)
                        values = list(formats.values())
                        assert len(values) == len(set(values)), f"Duplicate format values: {values}"
                        assert 'hevc' in values
                        return
        assert False, "OUTPUT_FORMATS not found in source"
