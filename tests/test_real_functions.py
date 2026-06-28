"""Tests that exercise real AlphaCut functions extracted from the source.

Since AlphaCut.py triggers bootstrap + PyQt6 imports on import, we extract
specific pure functions by parsing the source file directly. This ensures
tests stay in sync with the real implementation.
"""
import os
import sys
import ast
import json
import subprocess
import time
import types
import hashlib
import tempfile

# ---------------------------------------------------------------------------
# Helper: extract a function from AlphaCut.py source without triggering
# bootstrap or PyQt6 imports.
# ---------------------------------------------------------------------------
_SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'AlphaCut.py')

_OUTPUT_EXTENSIONS = {
    'prores': '.mov', 'webm': '.webm', 'png_seq': '', 'greenscreen': '.mp4',
    'matte': '.mov', 'mp4': '.mp4', 'hevc': '.mp4', 'webp_anim': '.webp',
    'gif_anim': '.gif', 'fg_alpha': '.mp4', 'av1': '.mp4', 'png': '.png',
    'mp4_nvenc': '.mp4', 'hevc_nvenc': '.mp4', 'mp4_qsv': '.mp4', 'hevc_qsv': '.mp4',
}

def _format_extension(fmt):
    return _OUTPUT_EXTENSIONS.get(fmt, '.mov')

def _normalize_encoder_format(fmt):
    return {
        'mp4_nvenc': 'mp4',
        'mp4_qsv': 'mp4',
        'hevc_nvenc': 'hevc',
        'hevc_qsv': 'hevc',
    }.get(fmt, fmt)

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
        self._generate = _extract_function('generate_output_name',
                                           extra_imports={'format_extension': _format_extension})

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
# Tests for _stable_resume_id (real implementation)
# ---------------------------------------------------------------------------
class TestStableResumeId:
    def setup_method(self):
        self._stable_resume_id = _extract_function('_stable_resume_id')

    def test_normalized_path_hash_is_hex_and_stable(self, tmp_path):
        source = tmp_path / 'clip.mp4'
        source.write_bytes(b'video bytes')
        first = self._stable_resume_id(str(source))
        second = self._stable_resume_id(os.path.join(str(tmp_path), '.', 'clip.mp4'))

        assert first == second
        assert len(first) == 16
        int(first, 16)

    def test_same_path_matches_across_hash_seeds(self, tmp_path):
        source = tmp_path / 'clip.mp4'
        source.write_bytes(b'video bytes')
        app_source = _read_source()
        tree = ast.parse(app_source)
        helper_source = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '_stable_resume_id':
                helper_source = ast.get_source_segment(app_source, node)
                break
        assert helper_source is not None

        script = "\n".join([
            "import hashlib",
            "import os",
            helper_source,
            f"print(_stable_resume_id({str(source)!r}))",
        ])

        results = []
        for seed in ('1', '987654'):
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = seed
            proc = subprocess.run(
                [sys.executable, '-c', script],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            results.append(proc.stdout.strip())

        assert results[0] == results[1]

    def test_processing_worker_uses_stable_resume_id(self):
        source = _read_source()
        worker_src = source.split('class ProcessingWorker', 1)[1].split('\n\n# ═', 1)[0]
        assert '_stable_resume_id(self.input_path)' in worker_src
        assert 'hash(self.input_path)' not in worker_src


# ---------------------------------------------------------------------------
# Tests for estimate_output_size (real implementation)
# ---------------------------------------------------------------------------
class TestEstimateOutputSize:
    def setup_method(self):
        self._estimate = _extract_function('estimate_output_size',
                                           extra_imports={'normalize_encoder_format': _normalize_encoder_format})

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
        for fmt in ['mp4', 'hevc', 'av1', 'webm', 'prores', 'png_seq', 'greenscreen', 'matte',
                    'webp_anim', 'gif_anim', 'mp4_nvenc', 'hevc_nvenc', 'mp4_qsv', 'hevc_qsv']:
            assert self._estimate(info, fmt) > 0, f"{fmt} should give positive estimate"

    def test_hardware_encoder_estimates_match_codec_family(self):
        info = {'width': 1920, 'height': 1080, 'total_frames': 300}
        assert self._estimate(info, 'mp4_nvenc') == self._estimate(info, 'mp4')
        assert self._estimate(info, 'mp4_qsv') == self._estimate(info, 'mp4')
        assert self._estimate(info, 'hevc_nvenc') == self._estimate(info, 'hevc')
        assert self._estimate(info, 'hevc_qsv') == self._estimate(info, 'hevc')


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
        validate_sha256 = _extract_function('_validate_sha256')
        fn = _extract_function('_load_model_registry',
                               extra_imports={
                                   'MODEL_BASE': 'https://github.com/danielgatis/rembg/releases/download/v0.0.0',
                                   '__file__': os.path.abspath(_SRC_PATH),
                                   '_validate_sha256': validate_sha256,
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
            assert 'sha256' in cfg
            assert len(cfg['sha256']) == 64
            assert isinstance(cfg['size'], tuple)
            assert len(cfg['mean']) == 3
            assert len(cfg['std']) == 3

    def test_no_duplicate_files(self):
        validate_sha256 = _extract_function('_validate_sha256')
        fn = _extract_function('_load_model_registry',
                               extra_imports={
                                   'MODEL_BASE': 'https://github.com/danielgatis/rembg/releases/download/v0.0.0',
                                   '__file__': os.path.abspath(_SRC_PATH),
                                   '_validate_sha256': validate_sha256,
                               })
        models = fn()
        files = [cfg['file'] for cfg in models.values()]
        assert len(files) == len(set(files)), f"Duplicate model files: {files}"

    def test_missing_model_hash_is_rejected(self, tmp_path):
        registry = {
            "base_url": "https://example.invalid/models",
            "models": [{
                "name": "broken",
                "label": "Broken",
                "file": "broken.onnx",
                "url": "{base_url}/broken.onnx",
                "input_size": [320, 320],
                "mean": [0.5, 0.5, 0.5],
                "std": [1.0, 1.0, 1.0],
            }],
        }
        (tmp_path / 'models.json').write_text(json.dumps(registry), encoding='utf-8')
        validate_sha256 = _extract_function('_validate_sha256')
        fn = _extract_function('_load_model_registry',
                               extra_imports={
                                   'MODEL_BASE': registry['base_url'],
                                   '__file__': str(tmp_path / 'AlphaCut.py'),
                                   '_validate_sha256': validate_sha256,
                                   '_log_warning': lambda message: None,
                               })
        try:
            fn()
            assert False, "Missing sha256 should reject the model registry"
        except ValueError as exc:
            assert 'sha256' in str(exc)

    def test_model_downloads_compare_expected_sha256(self):
        source = _read_source()
        ensure_model = source.split('def _ensure_model(self, progress_fn=None, cancel_check=None):', 1)[1].split('\n\n_engine_cache', 1)[0]
        assert "_validate_sha256(self.config.get('sha256')" in ensure_model
        assert 'actual == expected' in ensure_model
        assert 'digest != expected' in ensure_model
        assert 'f.write(expected)' in ensure_model
        assert 'Model cached' not in ensure_model

    def test_model_manager_surfaces_verification_status(self):
        source = _read_source()
        dialog_src = source.split('class ModelManagerDialog', 1)[1].split('\n\n# ═', 1)[0]
        assert 'status.model_verified' in dialog_src
        assert 'status.needs_verification' in dialog_src
        assert 'status.not_downloaded_hash_pinned' in dialog_src


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
                        for fmt in ['mp4_nvenc', 'hevc_nvenc', 'mp4_qsv', 'hevc_qsv']:
                            assert fmt in source
                        return
        assert False, "OUTPUT_FORMATS not found in source"

    def test_cli_runtime_accepts_hardware_encoder_formats(self):
        """run_cli should validate against CPU plus hardware format ids."""
        source = _read_source()
        run_cli = source.split('def run_cli(args):', 1)[1].split('\ndef main()', 1)[0]
        assert 'fmt not in ALL_OUTPUT_FORMAT_VALUES' in run_cli
        assert 'fmt not in OUTPUT_FORMATS.values()' not in run_cli

    def test_cli_runtime_reports_failed_terminal_event(self):
        """run_cli should exit non-zero and avoid complete events after failures."""
        source = _read_source()
        run_cli = source.split('def run_cli(args):', 1)[1].split('\ndef main()', 1)[0]
        assert '"type": "failed"' in run_cli
        assert '"type": "complete"' in run_cli
        assert 'sys.exit(1)' in run_cli
        assert 'worker.error.connect' in run_cli
        assert 'No output produced.' in run_cli
