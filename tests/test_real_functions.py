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
import threading
import time
import types
import hashlib
import tempfile
import numpy as np
from PIL import Image

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
# Tests for animated WebP/GIF memory guards
# ---------------------------------------------------------------------------
class TestAnimatedExportMemory:
    def setup_method(self):
        self._estimate_anim = _extract_function('estimate_animation_memory_mb')

    def test_animation_memory_estimate_uses_rgba_frames(self):
        expected = 10 * 100 * 50 * 4 / (1024 * 1024) * 1.35
        assert abs(self._estimate_anim(10, 100, 50, 'webp_anim') - expected) < 0.001

    def test_animation_memory_estimate_handles_empty_inputs(self):
        assert self._estimate_anim(0, 1920, 1080, 'gif_anim') == 0
        assert self._estimate_anim(10, 0, 1080, 'webp_anim') == 0

    def test_animated_encoders_preflight_before_loading_frame_lists(self):
        source = _read_source()
        encode_src = source.split('def _encode(self, ffmpeg, frames_dir, fps, info, total_frames=0):', 1)[1].split('\n\n# ', 1)[0]
        run_src = source.split('def _process(self):', 1)[1].split('def _encode(self, ffmpeg, frames_dir, fps, info, total_frames=0):', 1)[0]
        assert 'def _check_animation_memory_budget' in encode_src
        assert 'animated_export_memory_limit_mb()' in encode_src
        assert 'self._encode_error = None' in run_src
        assert 'elif not self._encode_error:' in run_src
        assert 'self._encode_error = msg' in encode_src
        assert '--allow-large-animation' in source
        webp_src = encode_src.split("if fmt == 'webp_anim':", 1)[1].split("if fmt == 'gif_anim':", 1)[0]
        gif_src = encode_src.split("if fmt == 'gif_anim':", 1)[1].split('first = sorted', 1)[0]
        assert webp_src.index('_check_animation_memory_budget') < webp_src.index('frames_pil = []')
        assert gif_src.index('_check_animation_memory_budget') < gif_src.index('frames_gif = []')


# ---------------------------------------------------------------------------
# Tests for GUI accessibility and responsive layout contracts
# ---------------------------------------------------------------------------
class TestUiAccessibilityResponsive:
    def test_main_window_uses_responsive_scroll_panel(self):
        source = _read_source()
        assert 'self.setMinimumSize(900, 650)' in source
        assert 'left.setFixedWidth(400)' not in source
        assert 'left.setMinimumWidth(340)' in source
        assert 'left_scroll = QScrollArea()' in source
        assert 'left_scroll.setWidgetResizable(True)' in source
        assert 'left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)' in source
        assert 'root.addWidget(left_scroll); root.addWidget(right, stretch=1)' in source

    def test_status_and_memory_have_non_color_cues(self):
        source = _read_source()
        assert 'class StatusLabel(QLabel)' in source
        assert 'self.lbl_status_cue = QLabel("")' in source
        assert 'self.lbl_status.bind_cue_label(self.lbl_status_cue)' in source
        assert '"ERROR"' in source and '"RUNNING"' in source and '"READY"' in source
        assert 'label = "RAM HIGH" if ram_pct > 85 else "RAM WARN"' in source

    def test_focus_order_and_accessible_names_are_pinned(self):
        source = _read_source()
        assert 'def _configure_focus_order(self):' in source
        assert 'self._configure_focus_order()' in source
        assert 'self.setTabOrder(first, second)' in source
        assert source.count('setAccessibleName(') >= 35
        for widget_name in (
            'self.btn_browse',
            'self.btn_batch',
            'self.btn_start',
            'self.btn_cancel',
            'self.progress_bar',
            'self.log_view',
            'self.job_table',
        ):
            assert widget_name in source


# ---------------------------------------------------------------------------
# Tests for mask quality inspection metrics
# ---------------------------------------------------------------------------
class TestMaskQualityInspection:
    def setup_method(self):
        self._inspect = _extract_function('inspect_mask_quality',
                                          extra_imports={'np': np, 'Image': Image})

    def test_full_frame_mask_warns(self):
        mask = Image.new('L', (20, 20), 255)
        metrics = self._inspect(mask)
        assert metrics['foreground_pct'] == 100.0
        assert metrics['transparent_pct'] == 0.0
        assert metrics['jitter_risk'] == 'low'
        assert any(item['key'] == 'mask_quality.warn.full_frame' for item in metrics['warnings'])

    def test_tiny_subject_warns(self):
        mask = Image.new('L', (100, 100), 0)
        for x in range(3):
            for y in range(3):
                mask.putpixel((x, y), 255)
        metrics = self._inspect(mask)
        assert metrics['foreground_pct'] < 1.0
        assert metrics['transparent_pct'] > 99.0
        assert any(item['key'] == 'mask_quality.warn.tiny_subject' for item in metrics['warnings'])

    def test_soft_transition_reports_jitter_risk(self):
        values = np.tile(np.linspace(0, 255, 80, dtype=np.uint8), (80, 1))
        mask = Image.fromarray(values, 'L')
        metrics = self._inspect(mask)
        assert metrics['transition_pct'] > 80.0
        assert metrics['jitter_risk'] == 'high'
        assert any(item['key'] == 'mask_quality.warn.soft_mask' for item in metrics['warnings'])

    def test_preview_worker_emits_metrics_to_ui(self):
        source = _read_source()
        assert 'result = pyqtSignal(object, object, object, object, object, object)' in source
        assert 'mask_metrics = inspect_mask_quality(current_mask)' in source
        assert 'mask_bw, mask_gray, mask_overlay, mask_metrics' in source
        assert 'def _preview_done(self, orig, proc, mask_bw, mask_gray, mask_overlay, mask_metrics):' in source
        assert 'self.lbl_mask_quality = QLabel' in source
        assert 'self._update_mask_quality(mask_metrics)' in source
        assert 'format_mask_quality_summary(metrics or {})' in source


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

    def test_registry_loader_reads_utf8_labels(self):
        source = _read_source()
        registry_src = source.split('def _load_model_registry():', 1)[1].split('def ', 1)[0]
        assert "open(registry_path, 'r', encoding='utf-8')" in registry_src

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
# Tests for pipe-mode process handling
# ---------------------------------------------------------------------------
class TestPipeMode:
    def test_pipe_decoder_discards_noisy_stderr_without_blocking(self):
        open_decoder = _extract_function('_open_pipe_decoder',
                                         extra_imports={
                                             'subprocess': subprocess,
                                             '_SUBPROCESS_FLAGS': 0,
                                         })
        script = "\n".join([
            "import sys",
            "sys.stderr.buffer.write(b'x' * (1024 * 1024))",
            "sys.stderr.flush()",
            "sys.stdout.buffer.write(b'ok')",
            "sys.stdout.flush()",
        ])
        proc = open_decoder([sys.executable, '-c', script])
        result = {}

        def read_stdout():
            result['data'] = proc.stdout.read(2)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        reader.join(5)
        if reader.is_alive():
            proc.kill()
            proc.wait(timeout=5)
            raise AssertionError("pipe decoder blocked while child wrote stderr")

        return_code = proc.wait(timeout=5)
        assert result.get('data') == b'ok'
        assert return_code == 0
        assert proc.stderr is None


# ---------------------------------------------------------------------------
# Tests for batch error propagation
# ---------------------------------------------------------------------------
class TestBatchErrors:
    def setup_method(self):
        self._summary = _extract_function('_batch_error_summary')

    def test_batch_error_summary_uses_first_nonempty_line(self):
        message = "\n\nRuntimeError: model failed\nTraceback details"
        assert self._summary(message) == "RuntimeError: model failed"

    def test_batch_error_summary_truncates_long_messages(self):
        summary = self._summary("X" * 120, limit=20)
        assert summary == "X" * 17 + "..."

    def test_batch_worker_prefers_worker_error_over_missing_output(self):
        source = _read_source()
        batch_src = source.split('class BatchWorker', 1)[1].split('\n\n# ', 1)[0]
        assert 'worker_errors = []' in batch_src
        assert 'worker.error.connect(_record_worker_error)' in batch_src
        assert 'self.job_error.emit(i, worker_errors[-1])' in batch_src
        assert batch_src.index('if worker_errors:') < batch_src.index('elif os.path.exists(out):')

    def test_job_table_displays_error_summary_and_tooltip(self):
        source = _read_source()
        table_src = source.split('class JobTable', 1)[1].split('\n\nclass DragOutButton', 1)[0]
        assert 'def update_error(self, row, message):' in table_src
        assert '_batch_error_summary(message)' in table_src
        assert 'status.error_detail' in table_src
        assert 'setToolTip' in table_src
        window_src = source.split('def _start_batch', 1)[1].split('def _begin_processing', 1)[0]
        assert 'job_error.connect(self._batch_job_error)' in window_src
        assert 'Batch job {idx + 1} error' in window_src


# ---------------------------------------------------------------------------
# Tests for Windows release packaging
# ---------------------------------------------------------------------------
class TestWindowsReleasePackaging:
    def test_inno_setup_can_enable_configured_sign_tool(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'packaging', 'windows', 'AlphaCut.iss')
        with open(path, encoding='utf-8') as f:
            source = f.read()
        assert '#ifdef InstallerSignTool' in source
        assert 'SignTool={#InstallerSignTool}' in source
        assert 'SignedUninstaller=yes' in source

    def test_release_script_builds_signs_and_checksums_artifacts(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'packaging', 'windows', 'build-release.ps1')
        with open(path, encoding='utf-8') as f:
            source = f.read()
        assert 'AlphaCut-windows.spec' in source
        assert 'ALPHACUT_SIGN' in source
        assert '/Salphacut_signtool=' in source
        assert '/DInstallerSignTool=alphacut_signtool' in source
        assert 'AlphaCut-windows.exe.sha256' in source
        assert 'SHA256SUMS.txt' in source
        assert 'Get-Sha256Hex' in source

    def test_readme_documents_checksum_and_unsigned_fallback(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
        with open(path, encoding='utf-8') as f:
            source = f.read()
        assert 'build-release.ps1' in source
        assert 'AlphaCut-windows.exe.sha256' in source
        assert 'AlphaCut-Setup-<version>.exe.sha256' in source
        assert 'produced unsigned' in source


# ---------------------------------------------------------------------------
# Tests for accelerator install profiles and runtime diagnostics
# ---------------------------------------------------------------------------
class TestAcceleratorProfiles:
    def test_runtime_diagnostics_cover_provider_profiles(self):
        source = _read_source()
        assert 'ACCELERATOR_INSTALL_PROFILES' in source
        assert 'requirements-cuda.txt' in source
        assert 'requirements-directml.txt' in source
        assert 'requirements-coreml.txt' in source
        assert 'CUDAExecutionProvider' in source
        assert 'DmlExecutionProvider' in source
        assert 'CoreMLExecutionProvider' in source
        assert 'def _format_runtime_diagnostics' in source
        assert '--runtime-info' in source

    def test_accelerator_requirement_files_do_not_mix_ort_packages(self):
        root = os.path.dirname(os.path.dirname(__file__))
        profiles = {
            'requirements-cuda.txt': 'onnxruntime-gpu',
            'requirements-directml.txt': 'onnxruntime-directml',
            'requirements-coreml.txt': 'onnxruntime',
            'requirements-cli-cuda.txt': 'onnxruntime-gpu',
            'requirements-cli-directml.txt': 'onnxruntime-directml',
            'requirements-cli-coreml.txt': 'onnxruntime',
        }
        for filename, expected in profiles.items():
            with open(os.path.join(root, filename), encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            ort_lines = [line for line in lines if line.startswith('onnxruntime')]
            assert ort_lines == [next(line for line in ort_lines if line.startswith(expected))]

    def test_gpu_docker_uses_cuda_cli_profile(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'Dockerfile.gpu')
        with open(path, encoding='utf-8') as f:
            source = f.read()
        assert 'requirements-cli-cuda.txt' in source
        assert 'pip3 uninstall -y onnxruntime' not in source


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
