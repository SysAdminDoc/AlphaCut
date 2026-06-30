#!/usr/bin/env python3
"""
AlphaCut v1.6.12 — AI Video Background Removal
Direct ONNX inference. No rembg dependency. Fully turnkey.

Dependencies: PyQt6, numpy, Pillow, onnxruntime, scipy (auto-installed)
External: FFmpeg (must be on PATH)

MIT License — Copyright (c) 2025-2026 SysAdminDoc
https://github.com/SysAdminDoc/AlphaCut
"""

import multiprocessing
multiprocessing.freeze_support()

__version__ = "1.6.12"

import sys, os, subprocess, shutil, json, tempfile, time, traceback, glob, base64, argparse, hashlib
import threading, queue
import urllib.request, urllib.error
from collections import deque

# Windows: suppress console windows from FFmpeg subprocesses
_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

# ═══════════════════════════════════════════════════════════════════════════════
# CRASH HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
def _exception_handler(exc_type, exc_value, exc_tb):
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    crash_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alphacut_crash.log')
    try:
        with open(crash_file, 'w') as f:
            f.write(msg)
    except Exception:
        pass
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, f"Crash log: {crash_file}\n\n{msg[:800]}", "AlphaCut — Fatal Error", 0x10)
        except Exception:
            pass
    print(msg, file=sys.stderr)
    os._exit(1)

sys.excepthook = _exception_handler


# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════
def _is_frozen():
    return getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")

def _pip_install(package, verbose=False):
    if _is_frozen():
        return False
    for flags in [[], ['--user'], ['--break-system-packages']]:
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', package] + flags
            if not verbose:
                cmd.append('-q')
            subprocess.check_call(cmd,
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=None if verbose else subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            continue
    return False


def _is_cli_mode():
    """Detect CLI/pipe mode from sys.argv before full arg parsing."""
    return any(a in sys.argv for a in ['-i', '--input', '--pipe', '--version', '--runtime-info'])

_CLI_MODE = _is_cli_mode()

ACCELERATOR_INSTALL_PROFILES = {
    'cpu': {
        'label': 'CPU',
        'provider': 'CPUExecutionProvider',
        'gui': 'requirements.txt',
        'cli': 'requirements-cli.txt',
        'package': 'onnxruntime',
    },
    'cuda': {
        'label': 'CUDA',
        'provider': 'CUDAExecutionProvider',
        'gui': 'requirements-cuda.txt',
        'cli': 'requirements-cli-cuda.txt',
        'package': 'onnxruntime-gpu',
    },
    'directml': {
        'label': 'DirectML',
        'provider': 'DmlExecutionProvider',
        'gui': 'requirements-directml.txt',
        'cli': 'requirements-cli-directml.txt',
        'package': 'onnxruntime-directml',
    },
    'coreml': {
        'label': 'CoreML',
        'provider': 'CoreMLExecutionProvider',
        'gui': 'requirements-coreml.txt',
        'cli': 'requirements-cli-coreml.txt',
        'package': 'onnxruntime',
    },
}


def _cuda_runtime_available():
    try:
        import ctypes
        if sys.platform == 'win32':
            ctypes.WinDLL('cublasLt64_12.dll')
        else:
            ctypes.CDLL('libcublasLt.so.12')
        return True
    except (OSError, Exception):
        return False


def _runtime_diagnostics(available_providers=None, ort_version=None, platform_name=None):
    """Return ONNX Runtime provider status and install guidance."""
    platform_name = platform_name or sys.platform
    providers = list(available_providers or [])
    active = []
    notes = []

    if 'CUDAExecutionProvider' in providers:
        if _cuda_runtime_available():
            active.append('CUDA')
        else:
            notes.append('CUDA provider package is installed, but CUDA 12/cuDNN runtime libraries are not loadable.')
    else:
        notes.append('CUDA unavailable: install requirements-cuda.txt or requirements-cli-cuda.txt, plus NVIDIA CUDA 12 and cuDNN 9.')

    if 'DmlExecutionProvider' in providers:
        active.append('DirectML')
    elif platform_name == 'win32':
        notes.append('DirectML unavailable: install requirements-directml.txt or requirements-cli-directml.txt on Windows.')

    if 'CoreMLExecutionProvider' in providers and platform_name == 'darwin':
        active.append('CoreML')
    elif platform_name == 'darwin':
        notes.append('CoreML unavailable: install requirements-coreml.txt or requirements-cli-coreml.txt on macOS.')

    if not active:
        active.append('CPU')

    return {
        'version': ort_version or 'unknown',
        'providers': providers,
        'active': active,
        'summary': ', '.join(active),
        'notes': notes,
    }


def _format_runtime_diagnostics(diagnostics=None, include_notes=True):
    if diagnostics is None:
        import onnxruntime as _ort
        diagnostics = _runtime_diagnostics(_ort.get_available_providers(), _ort.__version__)
    providers = diagnostics.get('providers') or ['none']
    lines = [
        f"ONNX Runtime {diagnostics.get('version', 'unknown')} | active: {diagnostics.get('summary', 'CPU')}",
        f"Providers: {', '.join(providers)}",
    ]
    if include_notes:
        for note in diagnostics.get('notes', []):
            lines.append(f"Runtime note: {note}")
        profiles = ', '.join(
            f"{profile['label']}={profile['gui']}/{profile['cli']}"
            for profile in ACCELERATOR_INSTALL_PROFILES.values()
        )
        lines.append(f"Install profiles: {profiles}")
    return "\n".join(lines)


def _bootstrap():
    if sys.version_info < (3, 9):
        print("Python 3.9+ required"); sys.exit(1)
    try:
        import pip
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'ensurepip', '--default-pip'])
    deps = [('PIL', 'Pillow'), ('numpy', 'numpy'),
            ('scipy', 'scipy'), ('onnxruntime', 'onnxruntime')]
    if not _CLI_MODE:
        deps.insert(0, ('PyQt6', 'PyQt6'))
    import importlib
    try:
        __import__('onnxruntime')
    except (ImportError, ModuleNotFoundError):
        for c in ['onnxruntime-gpu', 'onnxruntime-directml']:
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', c],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for mod, pkg in deps:
        try:
            __import__(mod)
        except (ImportError, ModuleNotFoundError):
            print(f"Installing {pkg}...")
            if not _pip_install(pkg, verbose=True):
                print(f"Failed to install {pkg}."); sys.exit(1)
            importlib.invalidate_caches()
            import site; site.main()
            for key in list(sys.modules.keys()):
                if key == mod or key.startswith(mod + '.'):
                    del sys.modules[key]
            try:
                __import__(mod)
            except (ImportError, ModuleNotFoundError) as e:
                verify = subprocess.run(
                    [sys.executable, '-c', f'import {mod}; print("ok")'],
                    capture_output=True, text=True, timeout=30)
                if verify.returncode == 0 and 'ok' in verify.stdout:
                    print(f"{pkg} installed. Restart IDE and re-run.")
                    try: os.execv(sys.executable, [sys.executable] + sys.argv)
                    except Exception: pass
                    sys.exit(0)
                print(f"{pkg} cannot import: {e}"); sys.exit(1)
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    diagnostics = _runtime_diagnostics(ort.get_available_providers(), ort.__version__)
    stream = sys.stderr if any(a in sys.argv for a in ['--pipe', '--json']) else sys.stdout
    if '--runtime-info' not in sys.argv:
        print(_format_runtime_diagnostics(diagnostics, include_notes=False), file=stream)
        for note in diagnostics['notes']:
            print(f"Runtime note: {note}", file=stream)

_bootstrap()

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
import onnxruntime as ort
_HAS_QT = False
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QProgressBar, QComboBox, QFileDialog, QGroupBox, QGridLayout,
        QTextEdit, QSpinBox, QSlider, QCheckBox, QGraphicsOpacityEffect,
        QSystemTrayIcon, QMenu, QTableWidget, QTableWidgetItem, QHeaderView,
        QAbstractItemView, QLineEdit, QColorDialog, QDialog, QDialogButtonBox,
        QScrollArea, QProgressDialog
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QUrl, QObject, QEvent, QMimeData
    from PyQt6.QtGui import (
        QPixmap, QImage, QPainter, QColor, QDragEnterEvent, QDropEvent, QPalette,
        QIcon, QAction, QMouseEvent, QPen, QClipboard, QDesktopServices, QDrag,
        QKeySequence
    )
    _HAS_QT = True
except ImportError:
    if not _CLI_MODE:
        raise
    # CLI/pipe mode without PyQt6: provide minimal stubs for worker classes
    import signal as _signal_mod
    class _QtStub:
        """Minimal stub so worker class definitions don't fail at parse time."""
        class _Signal:
            def __init__(self, *args): pass
            def connect(self, fn): self._fn = fn
            def emit(self, *args):
                if hasattr(self, '_fn'): self._fn(*args)
        def __getattr__(self, name): return type(self)()
        def __call__(self, *a, **kw): return self
    class QThread:
        """Stub QThread for CLI mode — runs synchronously."""
        def __init__(self, parent=None): pass
        def start(self): self.run()
        def isRunning(self): return False
        def quit(self): pass
        def wait(self, ms=0): pass
    class QObject:
        def __init__(self, parent=None): pass
    pyqtSignal = _QtStub._Signal
    QTimer = _QtStub()
    QEvent = _QtStub()
    QMimeData = _QtStub()
    QPropertyAnimation = _QtStub()
    QEasingCurve = _QtStub()
    QUrl = _QtStub()
    Qt = _QtStub()
    class _WidgetStub:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return _QtStub()
        def __call__(self, *args, **kwargs): return self
    QApplication = QMainWindow = QWidget = QLabel = QPushButton = QProgressBar = QComboBox = _WidgetStub
    QFileDialog = QGroupBox = QTableWidget = QTableWidgetItem = QHeaderView = _WidgetStub
    QAbstractItemView = QLineEdit = QColorDialog = QDialog = QDialogButtonBox = _WidgetStub
    QScrollArea = QSystemTrayIcon = QMenu = QGraphicsOpacityEffect = _WidgetStub
    QSpinBox = QSlider = QCheckBox = QTextEdit = QGridLayout = QVBoxLayout = QHBoxLayout = _WidgetStub
    QPixmap = QImage = QPainter = QColor = QPalette = QIcon = QAction = QPen = QClipboard = _WidgetStub
    QDesktopServices = QDrag = QKeySequence = QDragEnterEvent = QDropEvent = QMouseEvent = _WidgetStub

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
APP_NAME = "AlphaCut"
APP_VERSION = __version__
APP_DIR = os.path.join(os.path.expanduser("~"), ".alphacut")
SETTINGS_FILE = os.path.join(APP_DIR, "settings.json")
PRESETS_FILE = os.path.join(APP_DIR, "presets.json")
MODELS_DIR = os.path.join(APP_DIR, "models")
LOG_FILE = os.path.join(APP_DIR, "alphacut.log")
GITHUB_REPO = "SysAdminDoc/AlphaCut"
GITHUB_RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

MODEL_BASE = "https://github.com/danielgatis/rembg/releases/download/v0.0.0"

def _log_warning(message):
    """Write non-fatal diagnostics to stderr and the AlphaCut log file."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: {message}"
    try:
        print(line, file=sys.stderr)
    except Exception:
        return
    try:
        os.makedirs(APP_DIR, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except OSError:
        return

def _validate_sha256(value, model_name="model"):
    """Validate and normalize an expected SHA-256 digest."""
    digest = str(value or '').strip().lower()
    if len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest):
        raise ValueError(f"Model registry entry {model_name!r} is missing a valid sha256")
    return digest

def _load_model_registry():
    """Load model registry from models.json, falling back to hardcoded defaults."""
    registry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.json')
    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        base = data.get('base_url', MODEL_BASE)
        models = {}
        for m in data.get('models', []):
            url = m['url'].replace('{base_url}', base)
            expected_hash = _validate_sha256(m.get('sha256'), m.get('name') or m.get('file'))
            models[m['label']] = {
                'file': m['file'], 'url': url,
                'size': tuple(m['input_size']), 'mean': tuple(m['mean']),
                'std': tuple(m['std']), 'sigmoid': m.get('sigmoid', False),
                'sha256': expected_hash,
            }
        if models:
            return models
    except Exception as e:
        if os.path.isfile(registry_path):
            _log_warning(f"Model registry load failed: {e}")
            raise
    return {
        "u2net_human_seg (People — Recommended)": {
            "file": "u2net_human_seg.onnx", "url": f"{MODEL_BASE}/u2net_human_seg.onnx",
            "size": (320, 320), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "sigmoid": False,
            "sha256": "01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c",
        },
        "u2net (General Purpose)": {
            "file": "u2net.onnx", "url": f"{MODEL_BASE}/u2net.onnx",
            "size": (320, 320), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "sigmoid": False,
            "sha256": "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
        },
        "u2netp (Lightweight — Fastest)": {
            "file": "u2netp.onnx", "url": f"{MODEL_BASE}/u2netp.onnx",
            "size": (320, 320), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "sigmoid": False,
            "sha256": "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        },
        "silueta (People — Lightweight)": {
            "file": "silueta.onnx", "url": f"{MODEL_BASE}/silueta.onnx",
            "size": (320, 320), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "sigmoid": False,
            "sha256": "75da6c8d2f8096ec743d071951be73b4a8bc7b3e51d9a6625d63644f90ffeedb",
        },
        "isnet-general-use (General — High Quality)": {
            "file": "isnet-general-use.onnx", "url": f"{MODEL_BASE}/isnet-general-use.onnx",
            "size": (1024, 1024), "mean": (0.5, 0.5, 0.5), "std": (1.0, 1.0, 1.0), "sigmoid": False,
            "sha256": "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        },
        "isnet-anime (Anime / Illustration)": {
            "file": "isnet-anime.onnx", "url": f"{MODEL_BASE}/isnet-anime.onnx",
            "size": (1024, 1024), "mean": (0.5, 0.5, 0.5), "std": (1.0, 1.0, 1.0), "sigmoid": False,
            "sha256": "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99",
        },
        "BiRefNet-general (Best Quality — Slow)": {
            "file": "birefnet-general.onnx", "url": f"{MODEL_BASE}/BiRefNet-general-epoch_244.onnx",
            "size": (1024, 1024), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "sigmoid": True,
            "sha256": "58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c",
        },
        "BiRefNet-portrait (Portraits — High Quality)": {
            "file": "birefnet-portrait.onnx", "url": f"{MODEL_BASE}/BiRefNet-portrait-epoch_150.onnx",
            "size": (1024, 1024), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "sigmoid": True,
            "sha256": "1ba1c8ff5a7bbfadc8d8d13fb11d7be793f91f23d9d466549e37a854f6668f99",
        },
    }

MODELS = _load_model_registry()

OUTPUT_FORMATS = {
    "MP4 H.264 (.mp4) — Smallest": "mp4",
    "MP4 H.265/HEVC (.mp4) — Better Quality": "hevc",
    "MP4 AV1 (.mp4) — Best Compression": "av1",
    "WebM VP9 + Alpha (.webm)": "webm",
    "Animated WebP (.webp) — Short Clips": "webp_anim",
    "Animated GIF (.gif) — Web Compatible": "gif_anim",
    "MP4 + Green Screen (.mp4)": "greenscreen",
    "ProRes 4444 + Alpha (.mov)": "prores",
    "Matte Only — Grayscale (.mov)": "matte",
    "FG + Alpha Pair (.mp4) — NLE Import": "fg_alpha",
    "PNG Image Sequence": "png_seq",
}

HARDWARE_OUTPUT_FORMATS = {
    "MP4 H.264 NVENC (.mp4) - NVIDIA GPU": "mp4_nvenc",
    "MP4 H.265/HEVC NVENC (.mp4) - NVIDIA GPU": "hevc_nvenc",
    "MP4 H.264 QSV (.mp4) - Intel GPU": "mp4_qsv",
    "MP4 H.265/HEVC QSV (.mp4) - Intel GPU": "hevc_qsv",
}

OUTPUT_EXTENSIONS = {
    'prores': '.mov', 'webm': '.webm', 'png_seq': '', 'greenscreen': '.mp4',
    'matte': '.mov', 'mp4': '.mp4', 'hevc': '.mp4', 'webp_anim': '.webp',
    'gif_anim': '.gif', 'fg_alpha': '.mp4', 'av1': '.mp4', 'png': '.png',
    'mp4_nvenc': '.mp4', 'hevc_nvenc': '.mp4', 'mp4_qsv': '.mp4', 'hevc_qsv': '.mp4',
}

ALL_OUTPUT_FORMAT_VALUES = tuple(dict.fromkeys(
    list(OUTPUT_FORMATS.values()) + list(HARDWARE_OUTPUT_FORMATS.values())
))
_HW_ENCODER_CACHE = None

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mts'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

BG_COLORS = {
    "None (Transparent)": None,
    "Solid — Black": (0, 0, 0),
    "Solid — White": (255, 255, 255),
    "Solid — Green (#00FF00)": (0, 255, 0),
    "Solid — Blue (#0000FF)": (0, 0, 255),
    "Solid — Gray (#808080)": (128, 128, 128),
    "Custom Color...": "custom",
    "Image File...": "image",
}

BG_COLOR_LOCALE_KEYS = {
    "None (Transparent)": "transparent",
    "Solid — Black": "black",
    "Solid — White": "white",
    "Solid — Green (#00FF00)": "green",
    "Solid — Blue (#0000FF)": "blue",
    "Solid — Gray (#808080)": "gray",
    "Custom Color...": "custom",
    "Image File...": "image",
}

SMART_PRESETS = {
    "Person talking to camera": 0, "Full body / movement": 3,
    "General object / product": 1, "Anime / illustration": 5,
    "Best quality (slow)": 6,
}

SMART_PRESET_LOCALE_KEYS = {
    "Person talking to camera": "person_talking",
    "Full body / movement": "full_body",
    "General object / product": "general_object",
    "Anime / illustration": "anime",
    "Best quality (slow)": "best_quality",
}

# Auto-naming tokens: {name} {model} {format} {date} {time}
NAMING_PATTERNS = [
    "{name}_alphacut",
    "{name}_{model}",
    "{name}_alpha_{date}",
    "{name}_{format}",
]

# ═══════════════════════════════════════════════════════════════════════════════
# i18n SCAFFOLD
# ═══════════════════════════════════════════════════════════════════════════════
def format_extension(fmt):
    """Return the output extension for a format id."""
    return OUTPUT_EXTENSIONS.get(fmt, '.mov')

def normalize_encoder_format(fmt):
    """Map hardware encoder variants to their equivalent size/extension family."""
    return {
        'mp4_nvenc': 'mp4',
        'mp4_qsv': 'mp4',
        'hevc_nvenc': 'hevc',
        'hevc_qsv': 'hevc',
    }.get(fmt, fmt)

def detect_hardware_encoders(ffmpeg=None):
    """Return available FFmpeg hardware encoder names."""
    global _HW_ENCODER_CACHE
    use_cache = ffmpeg is None
    if use_cache and _HW_ENCODER_CACHE is not None:
        return set(_HW_ENCODER_CACHE)
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        return set()
    try:
        result = subprocess.run([ffmpeg, '-hide_banner', '-encoders'],
                                capture_output=True, text=True, timeout=10,
                                creationflags=_SUBPROCESS_FLAGS)
    except Exception as e:
        _log_warning(f"Hardware encoder detection failed: {e}")
        return set()
    if result.returncode != 0:
        _log_warning(f"Hardware encoder detection exited {result.returncode}: {(result.stderr or '')[-300:]}")
        return set()
    output = (result.stdout or '').lower()
    encoders = {name for name in ('h264_nvenc', 'hevc_nvenc', 'h264_qsv', 'hevc_qsv')
                if name in output}
    if use_cache:
        _HW_ENCODER_CACHE = set(encoders)
    return encoders

def available_output_formats():
    """Return CPU formats plus hardware formats detected on this machine."""
    formats = dict(OUTPUT_FORMATS)
    encoders = detect_hardware_encoders()
    if 'h264_nvenc' in encoders:
        formats["MP4 H.264 NVENC (.mp4) - NVIDIA GPU"] = "mp4_nvenc"
    if 'hevc_nvenc' in encoders:
        formats["MP4 H.265/HEVC NVENC (.mp4) - NVIDIA GPU"] = "hevc_nvenc"
    if 'h264_qsv' in encoders:
        formats["MP4 H.264 QSV (.mp4) - Intel GPU"] = "mp4_qsv"
    if 'hevc_qsv' in encoders:
        formats["MP4 H.265/HEVC QSV (.mp4) - Intel GPU"] = "hevc_qsv"
    return formats

_LOCALE = {}

def _load_locale():
    """Load locale overrides from ~/.alphacut/locale.json if present."""
    global _LOCALE
    locale_file = os.path.join(APP_DIR, 'locale.json')
    if not os.path.isfile(locale_file):
        return
    try:
        with open(locale_file, 'r', encoding='utf-8') as f:
            _LOCALE.update(json.load(f))
    except Exception as e:
        _log_warning(f"Locale load failed: {e}")

def _tr(key, default=None):
    """Translate a string key. Returns the locale override or the default English text."""
    return _LOCALE.get(key, default if default is not None else key)

def _trf(key, default, **kwargs):
    """Translate and format a string with named fields."""
    return _tr(key, default).format(**kwargs)

_load_locale()

_ICON_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAA3ElEQVQ4y62TsQ3CQBBE"
    "3xgkJKRuABqgAUqgFCqgBNcBFXwdUAIl0AAhEQnJcTBCPp+NhMRKq9Xu7OzO7kn/"
    "lKTVRAdJG0kv7W0nqcW10yxEkLQ4BNyjDh4lXUq6TwB2ktYlAzNgDwwS45Gkc2AJ"
    "TIEycBMZ2EbuWsZOXgCcAHfAE/D8A3gNPAInwDYBsIvcycfiCNiHECYZwAvwlAAo"
    "EkJwUlmSpnx7VsATcA+sAW/AB3AFjIFvyakK0JZ0QZT4PkmqJb0De2BH/N0K8E0p"
    "BU4l9a2XfwHmM0EkHfcePgAAAABJRU5ErkJggg=="
)

def get_app_icon():
    data = base64.b64decode(_ICON_B64)
    px = QPixmap(); px.loadFromData(data)
    return QIcon(px)


# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS / PRESETS / RECENT FILES
# ═══════════════════════════════════════════════════════════════════════════════
def load_settings():
    if not os.path.isfile(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, 'r') as f: return json.load(f)
    except Exception as e:
        _log_warning(f"Settings load failed: {e}")
        return {}

def save_settings(data):
    os.makedirs(APP_DIR, exist_ok=True)
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(data, f, indent=2)
    except Exception as e:
        _log_warning(f"Settings save failed: {e}")

def load_presets():
    if not os.path.isfile(PRESETS_FILE):
        return {}
    try:
        with open(PRESETS_FILE, 'r') as f: return json.load(f)
    except Exception as e:
        _log_warning(f"Preset load failed: {e}")
        return {}

def save_presets(data):
    os.makedirs(APP_DIR, exist_ok=True)
    try:
        with open(PRESETS_FILE, 'w') as f: json.dump(data, f, indent=2)
    except Exception as e:
        _log_warning(f"Preset save failed: {e}")

def get_recent_files():
    s = load_settings()
    return s.get('recent_files', [])

def add_recent_file(path):
    s = load_settings()
    recent = s.get('recent_files', [])
    if path in recent: recent.remove(path)
    recent.insert(0, path)
    s['recent_files'] = recent[:20]
    save_settings(s)

def generate_output_name(input_path, pattern, model_key, fmt):
    """Generate output filename from naming pattern."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    model_short = model_key.split('(')[0].strip().replace(' ', '_').lower()
    now = time.strftime("%Y%m%d"), time.strftime("%H%M%S")
    name = pattern.replace('{name}', base).replace('{model}', model_short)
    name = name.replace('{format}', fmt).replace('{date}', now[0]).replace('{time}', now[1])
    ext = format_extension(fmt)
    return os.path.join(os.path.dirname(input_path), f"{name}{ext}")

def _stable_resume_id(input_path):
    """Return a process-stable resume ID for an input path."""
    normalized = os.path.normcase(
        os.path.realpath(os.path.abspath(os.path.normpath(str(input_path))))
    )
    return hashlib.sha256(normalized.encode('utf-8', 'surrogatepass')).hexdigest()[:16]

def reveal_in_explorer(path):
    """Open the containing folder and select the file."""
    try:
        if sys.platform == 'win32':
            subprocess.Popen(['explorer', '/select,', os.path.normpath(path)])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', '-R', path])
        else:
            folder = path if os.path.isdir(path) else os.path.dirname(path)
            subprocess.Popen(['xdg-open', folder])
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════
DARK_STYLE = """
* { font-family: 'Segoe UI', 'Inter', system-ui, sans-serif; }
QMainWindow { background-color: #0d0f14; }
QWidget { background-color: transparent; color: #c8ccd4; }
QGroupBox {
    background-color: #13161d; border: 1px solid #1e2230; border-radius: 10px;
    margin-top: 1em; padding: 14px 10px 8px 10px;
    font-weight: 600; font-size: 11px; color: #6e7a94;
}
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #555d73; letter-spacing: 1px; }
QPushButton {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #6c5ce7, stop:1 #a855f7);
    color: #fff; border: none; padding: 7px 20px; border-radius: 8px; font-weight: 700; font-size: 12px;
}
QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #7c6cf7, stop:1 #b865ff); }
QPushButton:pressed { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #5c4cd7, stop:1 #9845e7); }
QPushButton:disabled { background: #1e2230; color: #3d4455; }
QPushButton#secondary { background: #1a1e2e; border: 1px solid #2a2e40; color: #8b92a5; }
QPushButton#secondary:hover { background: #222640; border-color: #6c5ce7; color: #c8ccd4; }
QPushButton#danger { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #dc2626, stop:1 #ef4444); }
QPushButton#small { padding: 5px 12px; font-size: 11px; border-radius: 6px; }
QComboBox {
    background-color: #13161d; color: #c8ccd4; border: 1px solid #1e2230;
    border-radius: 8px; padding: 5px 10px; font-size: 12px; min-height: 18px;
}
QComboBox::drop-down { border: none; width: 30px; }
QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid #6e7a94; margin-right: 10px; }
QComboBox QAbstractItemView { background-color: #13161d; color: #c8ccd4; border: 1px solid #2a2e40; selection-background-color: #6c5ce7; selection-color: #fff; padding: 4px; outline: none; }
QProgressBar {
    background-color: #13161d; border: 1px solid #1e2230; border-radius: 6px;
    text-align: center; color: #c8ccd4; font-size: 11px; font-weight: 600; min-height: 22px;
}
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6c5ce7, stop:1 #a855f7); border-radius: 5px; }
QTextEdit {
    background-color: #0a0c10; color: #5a6275; border: 1px solid #1a1d28; border-radius: 8px; padding: 8px;
    font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 11px; selection-background-color: #6c5ce7;
}
QLabel { color: #8b92a5; font-size: 12px; }
QLabel#title { color: #e2e5ed; font-size: 18px; font-weight: 800; }
QLabel#subtitle { color: #4a5168; font-size: 12px; font-weight: 500; }
QLabel#statValue { color: #e2e5ed; font-size: 13px; font-weight: 700; }
QLabel#statLabel { color: #4a5168; font-size: 9px; font-weight: 600; letter-spacing: 1px; }
QLabel#accent { color: #a855f7; font-weight: 700; }
QLabel#sliderVal { color: #6c5ce7; font-weight: 700; font-size: 12px; min-width: 30px; }
QSpinBox { background-color: #13161d; color: #c8ccd4; border: 1px solid #1e2230; border-radius: 6px; padding: 4px 8px; }
QSpinBox::up-button, QSpinBox::down-button { background: #1a1e2e; border: none; width: 20px; }
QSlider::groove:horizontal { background: #1a1d28; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #6c5ce7; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #7c6cf7; }
QSlider::sub-page:horizontal { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6c5ce7, stop:1 #a855f7); border-radius: 2px; }
QCheckBox { color: #8b92a5; font-size: 12px; spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #2a2e40; background: #13161d; }
QCheckBox::indicator:checked { background: #6c5ce7; border-color: #6c5ce7; }
QScrollBar:vertical { background: transparent; width: 8px; border: none; }
QScrollBar::handle:vertical { background: #2a2e40; border-radius: 4px; min-height: 40px; }
QScrollBar::handle:vertical:hover { background: #3a3e50; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollArea#leftPanelScroll { border: none; background: transparent; }
QLabel#statusCue { border-radius: 4px; padding: 2px 8px; font-weight: 800; font-size: 10px; min-width: 66px; }
QTableWidget { background-color: #0a0c10; border: 1px solid #1a1d28; border-radius: 8px; gridline-color: #1a1d28; }
QTableWidget::item { padding: 4px 8px; border: none; }
QTableWidget::item:selected { background-color: #1a1e35; color: #c8ccd4; }
QHeaderView::section { background-color: #13161d; color: #6e7a94; border: none; padding: 6px 8px; font-weight: 600; font-size: 11px; }
QLineEdit { background-color: #13161d; color: #c8ccd4; border: 1px solid #1e2230; border-radius: 6px; padding: 6px 10px; font-size: 12px; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# AI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_sha256(path):
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _rgba_to_gif_frame(rgba_img):
    """Convert an RGBA PIL image to P-mode with binary transparency for GIF.

    GIF supports only 256 colours and 1-bit (on/off) transparency.  We
    quantise the RGB content to 255 colours and assign palette index 255 as
    the transparent slot so that pixels with alpha < 128 become transparent.
    """
    rgba = rgba_img.convert('RGBA')
    rgb_q = rgba.convert('RGB').quantize(colors=255, method=0, dither=0)
    pal = list(rgb_q.getpalette() or [])
    # Pad palette to 256 entries and reserve index 255 for transparency
    while len(pal) < 256 * 3:
        pal.extend([0, 0, 0])
    pal[255 * 3: 255 * 3 + 3] = [0, 0, 0]
    rgb_q.putpalette(pal)
    arr_p = np.array(rgb_q, dtype=np.uint8)
    arr_a = np.array(rgba.split()[3], dtype=np.uint8)
    arr_p[arr_a < 128] = 255
    frame = Image.fromarray(arr_p, 'P')
    frame.putpalette(pal)
    frame.info['transparency'] = 255
    return frame


class AlphaCutEngine:
    def __init__(self, model_key, log_fn=None, gpu_device=-1, fp16=False):
        self.config = MODELS[model_key]; self.model_key = model_key
        self.log = log_fn or print; self.session = None
        self.gpu_device = int(gpu_device) if gpu_device is not None else -1
        self.fp16 = fp16
        self._mask_buffer = deque(maxlen=9)

    def load(self):
        path = self._ensure_model()
        self.log(f"Loading ONNX model: {self.config['file']}")
        ort.set_default_logger_severity(3)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = self._detect_providers(self.gpu_device)
        self.session = ort.InferenceSession(path, sess_options=opts, providers=providers)
        active = self.session.get_providers()
        self._use_fp16 = self.fp16 and any(p != 'CPUExecutionProvider' for p in active)
        gpu = any(p != 'CPUExecutionProvider' for p in active)
        device_label = "auto" if self.gpu_device < 0 else str(self.gpu_device)
        prec = " | FP16" if self._use_fp16 else ""
        self.log(f"Model loaded | {'GPU' if gpu else 'CPU'} ({', '.join(active)}) | device={device_label}{prec}")

    def reset_temporal(self): self._mask_buffer.clear()

    @staticmethod
    def _detect_providers(gpu_device=-1):
        providers = []; available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            if _cuda_runtime_available():
                if gpu_device >= 0:
                    providers.append(('CUDAExecutionProvider', {'device_id': str(gpu_device)}))
                else:
                    providers.append('CUDAExecutionProvider')
        if 'DmlExecutionProvider' in available:
            if gpu_device >= 0:
                providers.append(('DmlExecutionProvider', {'device_id': str(gpu_device)}))
            else:
                providers.append('DmlExecutionProvider')
        if 'CoreMLExecutionProvider' in available and sys.platform == 'darwin':
            providers.append('CoreMLExecutionProvider')
        providers.append('CPUExecutionProvider'); return providers

    @staticmethod
    def roi_to_box(size, roi):
        """Convert normalized ROI {x,y,w,h} to an image-space crop box."""
        if not roi:
            return None
        try:
            iw, ih = size
            x = max(0.0, min(1.0, float(roi.get('x', 0.0))))
            y = max(0.0, min(1.0, float(roi.get('y', 0.0))))
            w = max(0.0, min(1.0, float(roi.get('w', 0.0))))
            h = max(0.0, min(1.0, float(roi.get('h', 0.0))))
        except (TypeError, ValueError, AttributeError):
            return None
        x1 = int(round(x * iw)); y1 = int(round(y * ih))
        x2 = int(round(min(1.0, x + w) * iw)); y2 = int(round(min(1.0, y + h) * ih))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return (max(0, x1), max(0, y1), min(iw, x2), min(ih, y2))

    @staticmethod
    def apply_mask_edits(mask, edits):
        """Apply normalized foreground/background brush strokes to an L-mode mask."""
        if not edits:
            return mask
        out = mask.copy().convert('L')
        draw = ImageDraw.Draw(out)
        mw, mh = out.size
        for stroke in edits:
            try:
                pts = stroke.get('points') or []
                if not pts:
                    continue
                value = 0 if stroke.get('mode') == 'bg' else 255
                size = max(1, int(stroke.get('size', 24)))
                xy = []
                for pt in pts:
                    nx, ny = pt
                    px = int(round(max(0.0, min(1.0, float(nx))) * (mw - 1)))
                    py = int(round(max(0.0, min(1.0, float(ny))) * (mh - 1)))
                    xy.append((px, py))
                if len(xy) == 1:
                    r = max(1, size // 2)
                    x, y = xy[0]
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=value)
                else:
                    draw.line(xy, fill=value, width=size, joint='curve')
            except Exception:
                continue
        return out

    def predict_mask(self, pil_img, roi=None):
        roi_box = self.roi_to_box(pil_img.size, roi)
        source_img = pil_img.crop(roi_box) if roi_box else pil_img
        cfg = self.config; size_wh = cfg['size']; mean, std = cfg['mean'], cfg['std']
        im = source_img.convert('RGB').resize(size_wh, Image.Resampling.LANCZOS)
        a = np.array(im, dtype=np.float32); a = a / max(np.max(a), 1e-6)
        for c in range(3): a[:,:,c] = (a[:,:,c] - mean[c]) / std[c]
        dtype = np.float16 if self._use_fp16 else np.float32
        tensor = np.expand_dims(a.transpose((2,0,1)), 0).astype(dtype)
        try:
            ort_outs = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
        except Exception as e:
            if self._use_fp16:
                self.log(f"FP16 inference failed ({e}), falling back to FP32")
                tensor = tensor.astype(np.float32)
                ort_outs = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
                self._use_fp16 = False
            else:
                raise
        pred = ort_outs[0][:, 0, :, :]
        if cfg['sigmoid']: pred = 1.0 / (1.0 + np.exp(-pred))
        ma, mi = np.max(pred), np.min(pred)
        if ma - mi > 1e-6: pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)
        mask = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8), 'L')
        mask = mask.resize(source_img.size, Image.Resampling.LANCZOS)
        if roi_box:
            full = Image.new('L', pil_img.size, 0)
            full.paste(mask, roi_box)
            return full
        return mask

    def refine_mask(self, mask, edge_softness=0, mask_shift=0, temporal_smooth=0):
        arr = np.array(mask, dtype=np.float32) / 255.0
        if mask_shift != 0:
            binary = arr > 0.5; it = abs(mask_shift)
            if mask_shift < 0: binary = binary_erosion(binary, iterations=it, border_value=0)
            else: binary = binary_dilation(binary, iterations=it, border_value=0)
            arr = np.where(binary, np.maximum(arr, 0.5), np.minimum(arr, 0.5))
        if edge_softness > 0: arr = gaussian_filter(arr, sigma=edge_softness * 0.3)
        if temporal_smooth > 0 and len(self._mask_buffer) > 0:
            frames = [f for f in list(self._mask_buffer)[-temporal_smooth:]
                       if f.shape == arr.shape]
            if frames:
                wh = 0.4 / max(len(frames), 1); blended = arr * 0.6
                for prev in frames:
                    blended += prev * wh
                arr = np.clip(blended, 0, 1)
        self._mask_buffer.append(arr.copy())
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), 'L')

    def remove_background(self, pil_img, edge_softness=0, mask_shift=0,
                          temporal_smooth=0, matte_only=False):
        mask = self.predict_mask(pil_img)
        if edge_softness > 0 or mask_shift != 0 or temporal_smooth > 0:
            mask = self.refine_mask(mask, edge_softness, mask_shift, temporal_smooth)
        if matte_only: return mask.convert('RGB')
        img = pil_img.convert('RGBA')
        return Image.composite(img, Image.new('RGBA', img.size, 0), mask)

    @staticmethod
    def invert_mask(mask):
        """Invert a grayscale mask — swaps subject and background."""
        arr = np.array(mask, dtype=np.uint8)
        return Image.fromarray(255 - arr, 'L')

    @staticmethod
    def suppress_spill(pil_img, mask, strength=50, spill_color='green'):
        """Reduce color spill from the original background along mask edges.
        Works by desaturating the spill channel in semi-transparent edge regions."""
        img_arr = np.array(pil_img.convert('RGB'), dtype=np.float32)
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        # Edge region: where mask is between 0.05 and 0.95
        edge = ((mask_arr > 0.05) & (mask_arr < 0.95)).astype(np.float32)
        # Also blend slightly into the foreground near the edge
        if edge.any():
            edge = gaussian_filter(edge, sigma=2.0)
        factor = (strength / 100.0) * edge
        if spill_color == 'green':
            # Reduce green channel towards average of R and B
            avg_rb = (img_arr[:,:,0] + img_arr[:,:,2]) / 2
            img_arr[:,:,1] = img_arr[:,:,1] * (1 - factor) + avg_rb * factor
        elif spill_color == 'blue':
            avg_rg = (img_arr[:,:,0] + img_arr[:,:,1]) / 2
            img_arr[:,:,2] = img_arr[:,:,2] * (1 - factor) + avg_rg * factor
        elif spill_color == 'red':
            avg_gb = (img_arr[:,:,1] + img_arr[:,:,2]) / 2
            img_arr[:,:,0] = img_arr[:,:,0] * (1 - factor) + avg_gb * factor
        return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8), 'RGB')

    @staticmethod
    def preserve_shadows(pil_img, mask, strength=50):
        """Detect and preserve ground shadows by analyzing luminance in the background region.
        Returns a modified mask that includes shadow areas."""
        gray = np.array(pil_img.convert('L'), dtype=np.float32) / 255.0
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        # Background region = where mask < 0.3
        bg_region = mask_arr < 0.3
        if not bg_region.any():
            return mask
        # Find dark areas in background (potential shadows)
        bg_luma = np.where(bg_region, gray, 1.0)
        bg_median = np.median(bg_luma[bg_region]) if bg_region.any() else 0.5
        # Shadow threshold: pixels significantly darker than background median
        shadow_thresh = bg_median * 0.65
        shadow = (bg_luma < shadow_thresh) & bg_region
        if not shadow.any():
            return mask
        # Smooth shadow mask
        shadow_f = gaussian_filter(shadow.astype(np.float32), sigma=3.0)
        # Blend shadow into the mask
        blend = strength / 100.0
        new_mask = np.maximum(mask_arr, shadow_f * blend * 0.6)
        return Image.fromarray((np.clip(new_mask, 0, 1) * 255).astype(np.uint8), 'L')

    @staticmethod
    def composite_on_background(fg_rgba, bg_color=None, bg_image=None):
        """Composite RGBA foreground onto a background.
        bg_color: (R,G,B) tuple or None for transparent
        bg_image: PIL Image (pre-resized recommended) or None"""
        if bg_color is None and bg_image is None:
            return fg_rgba  # Keep transparent
        w, h = fg_rgba.size
        if bg_image is not None:
            bg = bg_image.convert('RGB')
            if bg.size != (w, h):
                bg = bg.resize((w, h), Image.Resampling.LANCZOS)
        elif bg_color is not None:
            bg = Image.new('RGB', (w, h), bg_color)
        else:
            return fg_rgba
        bg_rgba = bg.convert('RGBA')
        return Image.alpha_composite(bg_rgba, fg_rgba)

    def _ensure_model(self, progress_fn=None, cancel_check=None):
        """Download and verify the ONNX model file.

        progress_fn: optional callable(downloaded_bytes, total_bytes) for real-time feedback.
        cancel_check: optional callable() returning True to abort download.
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        filename = os.path.basename(self.config['file'])
        if not filename or '..' in self.config['file'] or os.sep in self.config['file'].replace(filename, ''):
            raise ValueError(f"Invalid model filename: {self.config['file']}")
        path = os.path.join(MODELS_DIR, filename)
        sidecar = path + '.sha256'
        expected = _validate_sha256(self.config.get('sha256'), filename)

        if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
            actual = _compute_sha256(path)
            if actual == expected:
                with open(sidecar, 'w') as f:
                    f.write(expected)
                self.log(f"Model verified: {self.config['file']} (sha256: {actual[:12]}…)"); return path
            else:
                self.log(f"SHA-256 mismatch for {self.config['file']} - re-downloading...")
                try: os.remove(path)
                except OSError: pass
                try:
                    if os.path.isfile(sidecar): os.remove(sidecar)
                except OSError:
                    pass

        url = self.config['url']
        self.log(f"Downloading: {self.config['file']}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': f'AlphaCut/{__version__}'})
            with urllib.request.urlopen(req, timeout=300) as resp:
                total = int(resp.headers.get('Content-Length', 0))
                downloaded = 0; tmp_path = path + '.tmp'; last_pct = -10
                with open(tmp_path, 'wb') as f:
                    while True:
                        if cancel_check and cancel_check():
                            f.close()
                            try: os.remove(tmp_path)
                            except Exception: pass
                            raise RuntimeError("Download cancelled")
                        chunk = resp.read(256 * 1024)
                        if not chunk: break
                        f.write(chunk); downloaded += len(chunk)
                        if progress_fn:
                            progress_fn(downloaded, total)
                        if total > 0:
                            pct = downloaded * 100 // total
                            if pct >= last_pct + 10:
                                last_pct = pct
                                self.log(f"   {downloaded/(1024*1024):.1f}/{total/(1024*1024):.1f} MB ({pct}%)")
            os.replace(tmp_path, path)
            digest = _compute_sha256(path)
            if digest != expected:
                try: os.remove(path)
                except OSError: pass
                try:
                    if os.path.isfile(sidecar): os.remove(sidecar)
                except OSError:
                    pass
                raise RuntimeError(
                    f"SHA-256 mismatch for {self.config['file']}: expected {expected}, got {digest}"
                )
            with open(sidecar, 'w') as f:
                f.write(expected)
            self.log(f"Model ready: {os.path.getsize(path)/(1024*1024):.1f} MB (sha256: {digest[:12]}…)"); return path
        except Exception as e:
            if os.path.exists(path + '.tmp'): os.remove(path + '.tmp')
            raise RuntimeError(f"Download failed: {e}")


_engine_cache = {'key': None, 'engine': None}
_engine_lock = threading.Lock()

def get_engine(model_key, log_fn=None, gpu_device=-1, fp16=False):
    cache_key = (model_key, int(gpu_device) if gpu_device is not None else -1, bool(fp16))
    with _engine_lock:
        if _engine_cache['key'] == cache_key and _engine_cache['engine'] is not None:
            eng = _engine_cache['engine']; eng.log = log_fn or print; return eng
        eng = AlphaCutEngine(model_key, log_fn=log_fn, gpu_device=cache_key[1], fp16=fp16); eng.load()
        _engine_cache['key'] = cache_key; _engine_cache['engine'] = eng; return eng


def detect_chroma_background(video_path):
    """Sample three frames and test corner patches for green/blue dominance.

    Returns a dict  {'color': 'green'|'blue', 'similarity': float, 'blend': float}
    when a solid-colour background is detected, or *None* otherwise.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    info = get_video_info(video_path)
    if not info:
        return None
    duration = info.get('duration', 0)
    if duration < 0.1:
        return None

    green_hits = 0
    blue_hits = 0
    total = 0

    for frac in (0.25, 0.50, 0.75):
        t = duration * frac
        cmd = [ffmpeg, '-ss', f'{t:.3f}', '-i', video_path,
               '-frames:v', '1', '-vf', 'scale=100:100',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1',
               '-loglevel', 'quiet']
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=15,
                                  creationflags=_SUBPROCESS_FLAGS)
            raw = proc.stdout
            if len(raw) < 100 * 100 * 3:
                continue
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(100, 100, 3)
        except Exception:
            continue

        # Sample four 10x10 corner patches
        corners = [
            arr[0:10,   0:10],   # top-left
            arr[0:10,  90:100],  # top-right
            arr[90:100, 0:10],   # bottom-left
            arr[90:100, 90:100], # bottom-right
        ]
        for patch in corners:
            r = int(patch[:, :, 0].mean())
            g = int(patch[:, :, 1].mean())
            b = int(patch[:, :, 2].mean())
            total += 1
            if g > 80 and g > r * 1.3 and g > b * 1.3:
                green_hits += 1
            elif b > 80 and b > r * 1.3 and b > g * 1.2:
                blue_hits += 1

    if total == 0:
        return None
    threshold = total * 0.6
    if green_hits >= threshold:
        return {'color': 'green', 'similarity': 0.35, 'blend': 0.05}
    if blue_hits >= threshold:
        return {'color': 'blue', 'similarity': 0.35, 'blend': 0.05}
    return None


class ChromaDetectWorker(QThread):
    """Run detect_chroma_background() off the GUI thread."""
    result = pyqtSignal(object)  # dict or None

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if self._cancelled:
            self.result.emit(None)
            return
        res = detect_chroma_background(self.video_path)
        if not self._cancelled:
            self.result.emit(res)


# ═══════════════════════════════════════════════════════════════════════════════
# FFMPEG UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def find_ffmpeg():
    ff = shutil.which('ffmpeg')
    if ff: return ff
    for p in [r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
              os.path.expanduser("~/ffmpeg/bin/ffmpeg.exe"), "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.isfile(p): return p
    return None

def find_ffprobe():
    fp = shutil.which('ffprobe')
    if fp: return fp
    ff = find_ffmpeg()
    if ff:
        probe = os.path.join(os.path.dirname(ff), 'ffprobe' + ('.exe' if sys.platform == 'win32' else ''))
        if os.path.isfile(probe): return probe
    return None

def get_video_info(filepath):
    ffprobe = find_ffprobe()
    if not ffprobe:
        _log_warning("ffprobe not found; video metadata unavailable")
        return None
    try:
        cmd = [ffprobe, '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', filepath]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, creationflags=_SUBPROCESS_FLAGS)
        if result.returncode != 0:
            _log_warning(f"ffprobe failed for {os.path.basename(filepath)}: {(result.stderr or '')[-400:]}")
            return None
        data = json.loads(result.stdout)
        has_audio = False
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'audio': has_audio = True
            if stream.get('codec_type') == 'video':
                fps_str = stream.get('r_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den) if float(den) != 0 else 30.0
                else: fps = float(fps_str)
                dur = float(data.get('format', {}).get('duration', 0))
                return {'width': int(stream.get('width', 0)), 'height': int(stream.get('height', 0)),
                        'fps': round(fps, 3), 'duration': dur,
                        'total_frames': int(fps * dur) if dur > 0 else 0,
                        'codec': stream.get('codec_name', 'unknown'), 'has_audio': has_audio}
        _log_warning(f"ffprobe found no video stream in {os.path.basename(filepath)}")
    except Exception as e:
        _log_warning(f"Video metadata read failed for {os.path.basename(filepath)}: {e}")
    return None

def estimate_output_size(info, fmt):
    if not info: return 0
    fmt = normalize_encoder_format(fmt)
    px = info['width'] * info['height']; frames = info['total_frames']
    bpf = {'prores': px*2.5, 'webm': px*0.15, 'png_seq': px*1.5, 'greenscreen': px*0.1, 'matte': px*0.3, 'mp4': px*0.08, 'hevc': px*0.06, 'av1': px*0.06, 'webp_anim': px*0.12, 'gif_anim': px*0.06}
    return bpf.get(fmt, px*0.5) * frames / (1024 * 1024)


def animated_export_memory_limit_mb():
    try:
        return max(256, int(os.environ.get('ALPHACUT_ANIMATION_MEMORY_LIMIT_MB', '1536')))
    except (TypeError, ValueError):
        return 1536


def estimate_animation_memory_mb(frame_count, width, height, fmt):
    """Estimate peak Python-side memory for PIL animated WebP/GIF encoding."""
    try:
        frame_count = max(0, int(frame_count))
        width = max(0, int(width))
        height = max(0, int(height))
    except (TypeError, ValueError):
        return 0
    if frame_count == 0 or width == 0 or height == 0:
        return 0
    rgba_mb = frame_count * width * height * 4 / (1024 * 1024)
    multiplier = 1.35 if fmt == 'webp_anim' else 1.15
    return rgba_mb * multiplier


def pil_to_qimage(pil_img):
    if not _HAS_QT:
        raise RuntimeError("pil_to_qimage requires PyQt6")
    img = pil_img.convert('RGBA'); data = img.tobytes('raw', 'RGBA')
    return QImage(data, img.width, img.height, img.width * 4, QImage.Format.Format_RGBA8888).copy()

def build_mask_preview_qimages(source_img, mask):
    """Build B/W, gray, and red/green overlay preview images for a mask."""
    mask_gray = mask.convert('L')
    mask_bw = mask_gray.point(lambda v: 255 if v >= 128 else 0)
    src = source_img.convert('RGB')
    m = np.array(mask_gray, dtype=np.float32) / 255.0
    rgb = np.array(src, dtype=np.float32)
    keep = np.array([34, 197, 94], dtype=np.float32)
    remove = np.array([239, 68, 68], dtype=np.float32)
    overlay_color = keep[None, None, :] * m[:, :, None] + remove[None, None, :] * (1.0 - m[:, :, None])
    overlay = np.clip(rgb * 0.55 + overlay_color * 0.45, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay, 'RGB')
    return (
        pil_to_qimage(mask_bw.convert('RGB')),
        pil_to_qimage(mask_gray.convert('RGB')),
        pil_to_qimage(overlay_img),
    )


def inspect_mask_quality(mask):
    """Return lightweight preview metrics for a single L-mode mask."""
    arr = np.array(mask.convert('L'), dtype=np.float32) / 255.0
    if arr.size == 0:
        return {
            'foreground_pct': 0.0,
            'transparent_pct': 0.0,
            'transition_pct': 0.0,
            'edge_density_pct': 0.0,
            'jitter_risk': 'unknown',
            'warnings': [{'key': 'mask_quality.warn.empty', 'default': 'Mask preview is empty.'}],
        }
    foreground_pct = float((arr >= 0.50).mean() * 100.0)
    transparent_pct = float((arr <= 0.05).mean() * 100.0)
    transition_pct = float(((arr > 0.05) & (arr < 0.95)).mean() * 100.0)
    if arr.shape[0] > 1 and arr.shape[1] > 1:
        gy, gx = np.gradient(arr)
        edge_strength = np.hypot(gx, gy)
        edge_density_pct = float((edge_strength > 0.08).mean() * 100.0)
    else:
        edge_density_pct = 0.0
    if edge_density_pct > 18.0 or transition_pct > 28.0:
        jitter_risk = 'high'
    elif edge_density_pct > 10.0 or transition_pct > 14.0:
        jitter_risk = 'medium'
    else:
        jitter_risk = 'low'

    warnings = []
    if foreground_pct < 1.0:
        warnings.append({
            'key': 'mask_quality.warn.tiny_subject',
            'default': 'Subject coverage is very small; choose an ROI or try a portrait/general model.',
        })
    elif foreground_pct > 96.0:
        warnings.append({
            'key': 'mask_quality.warn.full_frame',
            'default': 'Mask covers almost the entire frame; try another model or invert only when intentional.',
        })
    if transition_pct > 28.0:
        warnings.append({
            'key': 'mask_quality.warn.soft_mask',
            'default': 'Large semi-transparent area detected; reduce edge softness or inspect Mask Gray before export.',
        })
    if edge_density_pct > 18.0:
        warnings.append({
            'key': 'mask_quality.warn.edge_noise',
            'default': 'Busy mask edge detected; try temporal smoothing, ROI, or a higher-quality model.',
        })

    return {
        'foreground_pct': foreground_pct,
        'transparent_pct': transparent_pct,
        'transition_pct': transition_pct,
        'edge_density_pct': edge_density_pct,
        'jitter_risk': jitter_risk,
        'warnings': warnings,
    }


def format_mask_quality_summary(metrics):
    risk = str(metrics.get('jitter_risk', 'unknown'))
    risk_label = _tr(f"mask_quality.risk.{risk}", risk.upper())
    return _trf(
        "mask_quality.summary",
        "Mask: foreground {foreground:.0f}% | transparent {transparent:.0f}% | transition {transition:.1f}% | edge {edge:.1f}% | jitter risk {risk}",
        foreground=metrics.get('foreground_pct', 0.0),
        transparent=metrics.get('transparent_pct', 0.0),
        transition=metrics.get('transition_pct', 0.0),
        edge=metrics.get('edge_density_pct', 0.0),
        risk=risk_label,
    )


def mask_quality_warning_texts(metrics):
    return [_tr(item.get('key', ''), item.get('default', '')) for item in metrics.get('warnings', [])]


def _ftime(s):
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
def get_memory_usage():
    """Return dict with ram_used_mb, ram_total_mb, ram_pct. Cross-platform."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {'ram_used_mb': vm.used / (1024*1024), 'ram_total_mb': vm.total / (1024*1024),
                'ram_pct': vm.percent}
    except ImportError:
        pass
    # Fallback: /proc/meminfo on Linux
    if os.path.isfile('/proc/meminfo'):
        try:
            with open('/proc/meminfo') as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(':')] = int(parts[1])
            total = info.get('MemTotal', 0) / 1024
            avail = info.get('MemAvailable', info.get('MemFree', 0)) / 1024
            used = total - avail
            return {'ram_used_mb': used, 'ram_total_mb': total,
                    'ram_pct': (used / total * 100) if total > 0 else 0}
        except Exception:
            pass
    # Fallback: Windows GlobalMemoryStatusEx
    if sys.platform == 'win32':
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            mem = MEMORYSTATUSEX(); mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            total = mem.ullTotalPhys / (1024*1024)
            avail = mem.ullAvailPhys / (1024*1024)
            return {'ram_used_mb': total - avail, 'ram_total_mb': total,
                    'ram_pct': mem.dwMemoryLoad}
        except Exception:
            pass
    return {'ram_used_mb': 0, 'ram_total_mb': 0, 'ram_pct': 0}


def suggest_resolution(video_info, model_key):
    """Suggest optimal resolution based on model input size and video dimensions."""
    if not video_info:
        return 0, ""
    w, h = video_info['width'], video_info['height']
    cfg = MODELS.get(model_key, {})
    model_size = cfg.get('size', (320, 320))[0]  # model input resolution

    # If video is already at or below model input, no downscale needed
    if max(w, h) <= model_size:
        return 0, f"Video is {w}x{h}, model input is {model_size}px — no downscale needed"

    # Suggested tiers based on model input size
    # For 320px models: 1080p is the sweet spot; higher res wastes memory with no quality gain
    # For 1024px models: 1440p is the sweet spot; can benefit from more detail
    if model_size <= 320:
        sweet_spot = 1080
    else:
        sweet_spot = 1440

    # Only suggest downscale if video exceeds the sweet spot
    if max(w, h) <= sweet_spot:
        suggested = 0
    else:
        suggested = sweet_spot

    if suggested == 0:
        return 0, f"Resolution OK for {model_size}px model"

    # Calculate memory estimate
    ratio = suggested / max(w, h)
    sw, sh = int(w * ratio), int(h * ratio)
    est_ram_per_frame = sw * sh * 4 * 3 / (1024 * 1024)  # RGBA * 3 copies
    msg = (f"Video is {w}x{h}, model input is {model_size}px. "
           f"Suggest {suggested}px ({sw}x{sh}) — saves ~{est_ram_per_frame:.0f} MB/frame RAM")
    return suggested, msg


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING WORKER (single video) — Pipelined I/O
# ═══════════════════════════════════════════════════════════════════════════════
class ProcessingWorker(QThread):
    progress = pyqtSignal(int)
    frame_info = pyqtSignal(int, int)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    preview = pyqtSignal(object)
    memory_update = pyqtSignal(float)   # RAM %
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, output_path, model_key, output_format, max_res,
                 edge_softness=0, mask_shift=0, temporal_smooth=0, keep_audio=True,
                 frame_skip=1, invert_mask=False, spill_strength=0, spill_color='green',
                 shadow_strength=0, bg_color=None, bg_image_path=None, resume_from=0,
                 quality=70, roi=None, mask_edits=None, gpu_device=-1, fp16=False,
                 limit_seconds=None, allow_large_animation=False):
        super().__init__()
        self.input_path = input_path; self.output_path = output_path
        self.model_key = model_key; self.output_format = output_format
        self.max_res = max_res; self.edge_softness = edge_softness
        self.mask_shift = mask_shift; self.temporal_smooth = temporal_smooth
        self.keep_audio = keep_audio; self.frame_skip = max(1, frame_skip)
        self.invert_mask = invert_mask; self.spill_strength = spill_strength
        self.spill_color = spill_color; self.shadow_strength = shadow_strength
        self.bg_color = bg_color; self.bg_image_path = bg_image_path
        self.resume_from = resume_from
        self.quality = max(0, min(100, quality))
        self.roi = roi
        self.mask_edits = mask_edits or []
        self.gpu_device = int(gpu_device) if gpu_device is not None else -1
        self.fp16 = fp16
        self.limit_seconds = limit_seconds
        self.allow_large_animation = bool(allow_large_animation)
        self._cancelled = False

    def cancel(self): self._cancelled = True

    def run(self):
        try: self._process()
        except Exception as e:
            if not self._cancelled:
                self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _process(self):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            self.error.emit("FFmpeg not found!\nInstall from: https://ffmpeg.org/download.html"); return
        self.status.emit("Analyzing video...")
        self.log.emit(f"Input: {self.input_path}")
        info = get_video_info(self.input_path)
        if not info:
            self.error.emit("Could not read video."); return
        if self.limit_seconds:
            limit = max(0.1, min(float(self.limit_seconds), info.get('duration', float(self.limit_seconds)) or float(self.limit_seconds)))
            info = dict(info)
            info['duration'] = limit
            info['total_frames'] = max(1, int(info['fps'] * limit))
            self.log.emit(f"Preview duration: {limit:.1f}s")
        fps = info['fps']; w, h = info['width'], info['height']
        self.log.emit(f"{w}x{h} @ {fps}fps | ~{info['total_frames']} frames | {info['duration']:.1f}s")
        if info.get('has_audio'):
            self.log.emit(f"Audio: {'passthrough' if self.keep_audio else 'stripped'}")
        sw, sh = w, h
        if self.max_res > 0 and max(w, h) > self.max_res:
            ratio = self.max_res / max(w, h); sw = int(w * ratio) // 2 * 2; sh = int(h * ratio) // 2 * 2
            self.log.emit(f"Scaling to {sw}x{sh}")
        if self.frame_skip > 1:
            self.log.emit(f"Frame skip: every {self.frame_skip} frames (mask interpolation)")

        # Memory check before starting
        mem = get_memory_usage()
        if mem['ram_pct'] > 0:
            self.log.emit(f"RAM: {mem['ram_used_mb']:.0f}/{mem['ram_total_mb']:.0f} MB ({mem['ram_pct']:.0f}%)")
            if mem['ram_pct'] > 85:
                self.log.emit("WARNING: High memory usage — consider lowering resolution")

        tmp_dir = tempfile.mkdtemp(prefix='alphacut_')
        frames_in = os.path.join(tmp_dir, 'in'); frames_out = os.path.join(tmp_dir, 'out')
        os.makedirs(frames_in); os.makedirs(frames_out)
        # Resume support: use deterministic output dir based on input identity
        # so processed frames survive crashes
        resume_id = _stable_resume_id(self.input_path)
        progress_file = os.path.join(APP_DIR, f"resume_{resume_id}.json")
        persist_out = os.path.join(APP_DIR, f"wip_{resume_id}")
        resume_frame = self.resume_from
        try:
            # ── PHASE 1: EXTRACTION (0% → 10%) ──
            self.status.emit("Extracting frames...")
            self.progress.emit(0)
            expected_frames = info.get('total_frames', 0)
            cmd = [ffmpeg, '-v', 'warning']
            if self.limit_seconds:
                cmd += ['-t', f"{info['duration']:.3f}"]
            cmd += ['-i', self.input_path, '-fps_mode', 'cfr']
            if self.max_res > 0 and max(w, h) > self.max_res:
                cmd += ['-vf', f'scale={sw}:{sh}']
            frame_pattern = os.path.join(frames_in, 'frame_%06d.bmp')
            cmd.append(frame_pattern)
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     creationflags=_SUBPROCESS_FLAGS)
            # Monitor extraction progress by counting output files
            last_count = 0
            while proc.poll() is None:
                if self._cancelled:
                    proc.terminate(); proc.wait(5); return
                time.sleep(0.3)
                count = len(glob.glob(os.path.join(frames_in, 'frame_*.bmp')))
                if count != last_count:
                    last_count = count
                    if expected_frames > 0:
                        ext_pct = min(10, int((count / expected_frames) * 10))
                        self.progress.emit(ext_pct)
                        self.status.emit(f"Extracting frames... {count}/{expected_frames}")
                        self.frame_info.emit(count, expected_frames)
                    else:
                        self.status.emit(f"Extracting frames... {count}")
            stderr_out = proc.stderr.read().decode(errors='replace') if proc.stderr else ''
            if proc.returncode != 0:
                self.error.emit(f"FFmpeg extraction failed:\n{stderr_out[-500:]}"); return
            frame_files = sorted(glob.glob(os.path.join(frames_in, 'frame_*.bmp')))
            total = len(frame_files)
            if total == 0: self.error.emit("No frames extracted."); return
            self.progress.emit(10)
            self.log.emit(f"Extracted {total} frames")

            # Resume detection: check persistent output directory for existing frames
            os.makedirs(persist_out, exist_ok=True)
            if resume_frame <= 0:
                existing_out = sorted(glob.glob(os.path.join(persist_out, 'frame_*.tiff')))
                if existing_out and os.path.isfile(progress_file):
                    try:
                        with open(progress_file) as pf:
                            pd = json.load(pf)
                        if (pd.get('input') == self.input_path and
                            pd.get('output') == self.output_path and
                            pd.get('total') == total):
                            # Validate cached frames aren't corrupt
                            valid = 0
                            for fp in existing_out:
                                try:
                                    im = Image.open(fp); im.verify(); valid += 1
                                except Exception:
                                    self.log.emit(f"Resume: corrupt frame detected, trimming cache")
                                    for fp2 in existing_out[valid:]:
                                        try: os.remove(fp2)
                                        except Exception: pass
                                    break
                            if valid > 0:
                                resume_frame = valid
                                self.log.emit(f"Resuming from frame {resume_frame}/{total} ({valid} verified)")
                    except Exception:
                        pass
                # If not resuming, clean any stale files in WIP dir
                if resume_frame <= 0:
                    stale = glob.glob(os.path.join(persist_out, 'frame_*.tiff'))
                    stale += glob.glob(os.path.join(persist_out, 'frame_*.bmp'))   # clean legacy
                    stale += glob.glob(os.path.join(persist_out, 'frame_*.png'))   # clean legacy
                    for f in stale:
                        try: os.remove(f)
                        except Exception: pass
                    if os.path.isfile(progress_file):
                        try: os.remove(progress_file)
                        except Exception: pass
            # Use persistent dir for output frames instead of temp
            frames_out = persist_out

            # ── PHASE 2: MODEL LOAD (10% → 15%) ──
            self.status.emit("Loading AI model...")
            self.progress.emit(11)
            engine = get_engine(self.model_key, log_fn=self.log.emit, gpu_device=self.gpu_device, fp16=self.fp16)
            engine.reset_temporal()
            self.progress.emit(15)
            if self._cancelled: return

            matte_only = self.output_format == 'matte'
            if self.edge_softness > 0 or self.mask_shift != 0 or self.temporal_smooth > 0:
                self.log.emit(f"Refinement: edge={self.edge_softness} shift={self.mask_shift} temporal={self.temporal_smooth}")

            # Compositing setup
            bg_image = None
            if self.bg_image_path and os.path.isfile(self.bg_image_path):
                try:
                    bg_image = Image.open(self.bg_image_path)
                    bg_image.load()
                    self.log.emit(f"Background image: {os.path.basename(self.bg_image_path)}")
                except Exception as e:
                    self.log.emit(f"Background image failed: {e}")
            if self.invert_mask:
                self.log.emit("Mask inversion: ON")
            if self.spill_strength > 0:
                self.log.emit(f"Spill suppression: {self.spill_strength}% ({self.spill_color})")
            if self.shadow_strength > 0:
                self.log.emit(f"Shadow preservation: {self.shadow_strength}%")
            if self.bg_color is not None:
                self.log.emit(f"Background color: {self.bg_color}")
            if self.roi:
                self.log.emit("ROI selection: ON")
            if self.mask_edits:
                self.log.emit(f"Manual mask edits: {len(self.mask_edits)} stroke(s)")

            # ── PIPELINED PROCESSING ──
            # Pre-read queue feeds PIL images ahead of inference
            # Post-save queue writes results to disk in parallel
            read_q = queue.Queue(maxsize=8)    # pre-decoded PIL images
            save_q = queue.Queue(maxsize=8)    # (index, PIL result) to save
            read_done = threading.Event()
            save_errors = []
            save_errors_lock = threading.Lock()

            def _reader_thread():
                """Pre-reads BMP frames into PIL Images ahead of inference."""
                for i, fpath in enumerate(frame_files):
                    if self._cancelled: break
                    try:
                        img = Image.open(fpath)
                        img.load()
                        read_q.put((i, img))
                        try: os.remove(fpath)
                        except Exception: pass
                    except Exception as e:
                        read_q.put((i, None))
                read_done.set()

            def _saver_thread():
                """Writes processed frames to disk in parallel with inference.
                Only exits when it receives the None sentinel from the main loop."""
                while True:
                    try:
                        item = save_q.get(timeout=5)
                        if item is None: break
                        idx, result = item
                        try:
                            result.save(os.path.join(frames_out, f'frame_{idx+1:06d}.tiff'), 'TIFF')
                        except Exception as e:
                            with save_errors_lock:
                                save_errors.append(f"Save error frame {idx}: {e}")
                        save_q.task_done()
                    except queue.Empty:
                        if self._cancelled:
                            break

            # Start pipeline threads
            reader = threading.Thread(target=_reader_thread, daemon=True)
            saver = threading.Thread(target=_saver_thread, daemon=True)
            reader.start()
            saver.start()

            self.status.emit("Removing backgrounds...")
            t0 = time.time(); preview_every = max(1, total // 30)
            last_mask_img = None  # L-mode PIL mask for frame skip
            last_result = None   # last fully composited result for null-frame fallback
            last_inferred_idx = -1
            os.makedirs(APP_DIR, exist_ok=True)

            # Progress mapping: AI processing occupies 15% → 90% of the bar
            _PHASE3_START, _PHASE3_END = 15, 90

            # Pre-resize background image to frame dimensions once
            if bg_image is not None:
                bg_image = bg_image.convert('RGB').resize((sw, sh), Image.Resampling.LANCZOS)

            for frame_num in range(total):
                if self._cancelled: self.log.emit("Cancelled"); break

                try:
                    idx, img = read_q.get(timeout=30)
                except queue.Empty:
                    self.log.emit(f"Timeout reading frame {frame_num}"); break

                if img is None:
                    if last_result is not None:
                        save_q.put((idx, last_result.copy()))
                    continue

                # Resume: skip already-processed frames
                if frame_num < resume_frame:
                    out_frame = os.path.join(frames_out, f'frame_{idx+1:06d}.tiff')
                    if os.path.isfile(out_frame):
                        pct = _PHASE3_START + int(((frame_num + 1) / total) * (_PHASE3_END - _PHASE3_START))
                        self.progress.emit(pct)
                        continue
                    else:
                        self.log.emit(f"Resume: frame {frame_num} missing, processing from here")
                        resume_frame = 0

                should_infer = (self.frame_skip <= 1 or frame_num % self.frame_skip == 0
                                or frame_num == total - 1 or last_mask_img is None)

                if should_infer:
                    # Get raw mask
                    mask = engine.predict_mask(img, self.roi)
                    if self.edge_softness > 0 or self.mask_shift != 0 or self.temporal_smooth > 0:
                        mask = engine.refine_mask(mask, self.edge_softness,
                                                  self.mask_shift, self.temporal_smooth)
                    if self.mask_edits:
                        mask = AlphaCutEngine.apply_mask_edits(mask, self.mask_edits)
                    last_mask_img = mask
                    last_inferred_idx = frame_num
                else:
                    mask = last_mask_img

                # ── COMPOSITING PIPELINE ──
                current_mask = mask.copy()

                # Shadow preservation (modifies mask before compositing)
                if self.shadow_strength > 0:
                    current_mask = AlphaCutEngine.preserve_shadows(img, current_mask, self.shadow_strength)

                # Mask inversion
                if self.invert_mask:
                    current_mask = AlphaCutEngine.invert_mask(current_mask)

                if matte_only:
                    result = current_mask.convert('RGB')
                else:
                    # Spill suppression (modifies foreground pixels)
                    src = img
                    if self.spill_strength > 0:
                        src = AlphaCutEngine.suppress_spill(src, current_mask,
                                                            self.spill_strength, self.spill_color)

                    # Apply mask to get RGBA foreground
                    fg = src.convert('RGBA')
                    result = Image.composite(fg, Image.new('RGBA', fg.size, 0), current_mask)

                    # Background replacement
                    if self.bg_color is not None or bg_image is not None:
                        result = AlphaCutEngine.composite_on_background(
                            result, bg_color=self.bg_color, bg_image=bg_image)

                save_q.put((idx, result))
                last_result = result

                pct = _PHASE3_START + int(((frame_num + 1) / total) * (_PHASE3_END - _PHASE3_START))
                self.progress.emit(pct); self.frame_info.emit(frame_num + 1, total)
                elapsed = time.time() - t0
                if frame_num > 2:
                    spd = (frame_num + 1) / elapsed; eta = (total - frame_num - 1) / spd
                    skip_tag = f" [skip {self.frame_skip}x]" if self.frame_skip > 1 else ""
                    self.status.emit(f"Frame {frame_num+1}/{total} | {spd:.1f} fps{skip_tag} | ETA {_ftime(eta)}")
                if frame_num % preview_every == 0:
                    try: self.preview.emit(pil_to_qimage(result if not matte_only else result.convert('RGBA')))
                    except Exception: pass

                # Periodic memory monitoring + progress save
                if frame_num % 50 == 0:
                    mem = get_memory_usage()
                    if mem['ram_pct'] > 0:
                        self.memory_update.emit(mem['ram_pct'])
                        if mem['ram_pct'] > 90:
                            self.log.emit(f"WARNING: RAM at {mem['ram_pct']:.0f}%")
                    # Save progress for resume on crash/cancel
                    try:
                        with open(progress_file, 'w') as pf:
                            json.dump({'input': self.input_path, 'output': self.output_path,
                                       'total': total, 'last_frame': frame_num}, pf)
                    except Exception: pass

            # Signal saver to stop and wait
            save_q.put(None)
            saver.join(timeout=30)
            reader.join(timeout=10)

            if save_errors:
                for e in save_errors[:5]: self.log.emit(e)

            if self._cancelled: return

            # Clean up resume progress file on successful completion
            try:
                if os.path.isfile(progress_file): os.remove(progress_file)
            except Exception: pass

            et = time.time() - t0
            inferred = total if self.frame_skip <= 1 else (total + self.frame_skip - 1) // self.frame_skip
            self.log.emit(f"Processed {total} frames ({inferred} inferred) in {_ftime(et)} ({total/max(et,0.1):.1f} fps)")

            # ── PHASE 4: ENCODING (90% → 100%) ──
            self.status.emit("Encoding output...")
            self.progress.emit(90)
            self._encode_error = None
            out = self._encode(ffmpeg, frames_out, fps, info, total)

            # Clean up persistent WIP directory after encoding
            try:
                if os.path.isdir(persist_out): shutil.rmtree(persist_out, ignore_errors=True)
            except Exception: pass

            if out and (os.path.isfile(out) or os.path.isdir(out)):
                if os.path.isfile(out): self.log.emit(f"Output: {out} ({os.path.getsize(out)/(1024*1024):.1f} MB)")
                else: self.log.emit(f"Output: {out}")
                self.finished.emit(out)
            elif not self._encode_error:
                self.error.emit("Encoding failed.")
        finally: shutil.rmtree(tmp_dir, ignore_errors=True)

    def _encode(self, ffmpeg, frames_dir, fps, info, total_frames=0):
        pat = os.path.join(frames_dir, 'frame_%06d.tiff'); fmt = self.output_format
        q = self.quality  # 0-100 scale

        if fmt == 'png_seq':
            out_dir = self.output_path
            if os.path.splitext(out_dir)[1] != '': out_dir = os.path.splitext(out_dir)[0] + '_frames'
            os.makedirs(out_dir, exist_ok=True)
            tiff_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.tiff')))
            for i, f in enumerate(tiff_files):
                out_name = os.path.splitext(os.path.basename(f))[0] + '.png'
                Image.open(f).save(os.path.join(out_dir, out_name), 'PNG')
                if len(tiff_files) > 0:
                    pct = 90 + int(((i + 1) / len(tiff_files)) * 10)
                    self.progress.emit(pct)
                    self.status.emit(f"Saving PNG {i+1}/{len(tiff_files)}")
            self.progress.emit(100)
            return out_dir

        if fmt == 'fg_alpha':
            tiff_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.tiff')))
            if not tiff_files:
                self.error.emit("No frames found for FG+Alpha export."); return None
            n = len(tiff_files)
            base = os.path.splitext(self.output_path)[0]
            fg_dir = tempfile.mkdtemp(prefix='alphacut_fg_')
            alpha_dir = tempfile.mkdtemp(prefix='alphacut_alpha_')
            try:
                self.status.emit("Splitting FG + Alpha frames...")
                for i, f in enumerate(tiff_files):
                    try:
                        img = Image.open(f); img.load()
                        rgba = img.convert('RGBA')
                        r, g, b, a = rgba.split()
                        Image.merge('RGB', (r, g, b)).save(os.path.join(fg_dir, f'frame_{i+1:06d}.bmp'), 'BMP')
                        a.convert('RGB').save(os.path.join(alpha_dir, f'frame_{i+1:06d}.bmp'), 'BMP')
                    except Exception as e:
                        self.log.emit(f"FG+Alpha split error frame {i}: {e}")
                    if i % 30 == 0 or i == n - 1:
                        pct = 90 + int(((i + 1) / n) * 4)
                        self.progress.emit(pct)
                        self.status.emit(f"Splitting FG+Alpha {i+1}/{n}")
                crf = int(32 - (self.quality / 100) * 18)
                fg_out = base + '_fg.mp4'
                fg_pat = os.path.join(fg_dir, 'frame_%06d.bmp')
                cmd_fg = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', fg_pat,
                          '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                          '-movflags', '+faststart', fg_out]
                self.status.emit("Encoding foreground...")
                proc_fg = subprocess.run(cmd_fg, capture_output=True, creationflags=_SUBPROCESS_FLAGS)
                if proc_fg.returncode != 0:
                    self.log.emit(f"FG encode failed (exit {proc_fg.returncode}): {proc_fg.stderr[-400:] if proc_fg.stderr else ''}")
                    self.error.emit("FG+Alpha encode failed: foreground encoding error."); return None
                alpha_out = base + '_alpha.mp4'
                alpha_pat = os.path.join(alpha_dir, 'frame_%06d.bmp')
                cmd_alpha = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', alpha_pat,
                             '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                             '-movflags', '+faststart', alpha_out]
                self.status.emit("Encoding alpha matte...")
                proc_alpha = subprocess.run(cmd_alpha, capture_output=True, creationflags=_SUBPROCESS_FLAGS)
                if proc_alpha.returncode != 0:
                    self.log.emit(f"Alpha encode failed (exit {proc_alpha.returncode}): {proc_alpha.stderr[-400:] if proc_alpha.stderr else ''}")
                    self.error.emit("FG+Alpha encode failed: alpha matte encoding error."); return None
                if self.keep_audio and info.get('has_audio'):
                    fg_with_audio = base + '_fg_audio.mp4'
                    cmd_audio = [ffmpeg, '-y', '-v', 'warning', '-i', fg_out, '-i', self.input_path,
                                 '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', fg_with_audio]
                    proc = subprocess.run(cmd_audio, capture_output=True, creationflags=_SUBPROCESS_FLAGS)
                    if proc.returncode == 0:
                        os.replace(fg_with_audio, fg_out)
                    else:
                        try: os.remove(fg_with_audio)
                        except Exception: pass
                self.progress.emit(100)
                self.log.emit(f"FG+Alpha pair: {os.path.basename(fg_out)} + {os.path.basename(alpha_out)}")
                return fg_out
            finally:
                shutil.rmtree(fg_dir, ignore_errors=True)
                shutil.rmtree(alpha_dir, ignore_errors=True)

        def _check_animation_memory_budget(tiff_files, label):
            if not tiff_files:
                return False
            try:
                first = Image.open(tiff_files[0])
                width, height = first.size
                first.close()
            except Exception as e:
                self.error.emit(f"Cannot inspect {label} frames for memory estimate: {e}")
                return False
            estimate_mb = estimate_animation_memory_mb(len(tiff_files), width, height, fmt)
            limit_mb = animated_export_memory_limit_mb()
            msg = _trf(
                "error.animation_memory_limit",
                "{label} export would load about {estimate:.0f} MB of frames into RAM, above the {limit:.0f} MB safety limit. Use WebM VP9+Alpha or PNG sequence, lower resolution/duration, or rerun CLI with --allow-large-animation.",
                label=label,
                estimate=estimate_mb,
                limit=limit_mb,
            )
            if estimate_mb > limit_mb and not self.allow_large_animation:
                self._encode_error = msg
                self.log.emit(msg)
                self.error.emit(msg)
                return False
            if estimate_mb > limit_mb:
                self.log.emit(_trf(
                    "log.animation_memory_override",
                    "WARNING: Large {label} animation override enabled; estimated frame RAM is {estimate:.0f} MB.",
                    label=label,
                    estimate=estimate_mb,
                ))
            elif estimate_mb > limit_mb * 0.6:
                self.log.emit(_trf(
                    "log.animation_memory_estimate",
                    "{label} animation frame RAM estimate: {estimate:.0f} MB of {limit:.0f} MB limit.",
                    label=label,
                    estimate=estimate_mb,
                    limit=limit_mb,
                ))
            return True

        if fmt == 'webp_anim':
            tiff_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.tiff')))
            if not tiff_files:
                self.error.emit("No frames found for animated WebP."); return None
            n = len(tiff_files)
            out_file = os.path.splitext(self.output_path)[0] + '.webp'
            if not _check_animation_memory_budget(tiff_files, "Animated WebP"):
                return None
            if n > 300:
                self.log.emit(f"INFO: Animated WebP with {n} frames — large clips may use significant RAM. Consider WebM VP9+Alpha for videos > 10s.")
            self.status.emit("Building animated WebP...")
            frames_pil = []
            for i, f in enumerate(tiff_files):
                try:
                    img = Image.open(f); img.load(); frames_pil.append(img.convert('RGBA'))
                except Exception as e:
                    self.log.emit(f"Frame {i} skipped: {e}")
                if i % 30 == 0 or i == n - 1:
                    pct = 90 + int(((i + 1) / n) * 9)
                    self.progress.emit(pct)
                    self.status.emit(f"Building animated WebP {i+1}/{n}")
                if self._cancelled: return None
            if not frames_pil:
                self.error.emit("No valid frames for animated WebP."); return None
            duration_ms = max(1, int(1000 / fps))
            webp_q = max(1, min(100, q))
            self.status.emit("Saving animated WebP...")
            frames_pil[0].save(
                out_file, save_all=True, append_images=frames_pil[1:],
                duration=duration_ms, loop=0, lossless=False, quality=webp_q, method=4)
            self.progress.emit(100)
            return out_file

        if fmt == 'gif_anim':
            tiff_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.tiff')))
            if not tiff_files:
                self.error.emit("No frames found for animated GIF."); return None
            n = len(tiff_files)
            out_file = os.path.splitext(self.output_path)[0] + '.gif'
            if not _check_animation_memory_budget(tiff_files, "Animated GIF"):
                return None
            if n > 150:
                self.log.emit(f"INFO: Animated GIF with {n} frames — GIF is limited to 256 colours; consider WebM VP9+Alpha for longer clips.")
            self.status.emit("Building animated GIF...")
            frames_gif = []
            for i, f in enumerate(tiff_files):
                try:
                    img = Image.open(f); img.load()
                    frames_gif.append(_rgba_to_gif_frame(img.convert('RGBA')))
                except Exception as e:
                    self.log.emit(f"Frame {i} skipped: {e}")
                if i % 30 == 0 or i == n - 1:
                    pct = 90 + int(((i + 1) / n) * 9)
                    self.progress.emit(pct)
                    self.status.emit(f"Building animated GIF {i+1}/{n}")
                if self._cancelled: return None
            if not frames_gif:
                self.error.emit("No valid frames for animated GIF."); return None
            duration_ms = max(1, int(1000 / fps))
            self.status.emit("Saving animated GIF...")
            frames_gif[0].save(
                out_file, save_all=True, append_images=frames_gif[1:],
                duration=duration_ms, loop=0, disposal=2, optimize=False)
            self.progress.emit(100)
            return out_file

        first = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.tiff')))[0]
        fi = Image.open(first); fw, fh = fi.size; fi.close()
        out_file = os.path.splitext(self.output_path)[0] + format_extension(fmt)

        # Quality mapping per format:
        # q=0 (smallest) → highest CRF / lowest bitrate
        # q=100 (best) → lowest CRF / highest bitrate
        if fmt == 'prores':
            # bits_per_mb: q=0→800, q=50→2000, q=100→8000
            bpm = int(800 + (q / 100) ** 1.5 * 7200)
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat, '-c:v', 'prores_ks', '-profile:v', '4444',
                   '-pix_fmt', 'yuva444p10le', '-vendor', 'apl0', '-bits_per_mb', str(bpm)]
            self.log.emit(f"ProRes quality: bits_per_mb={bpm}")
        elif fmt == 'webm':
            # CRF: q=0→45, q=50→30, q=100→15 (lower CRF = better quality)
            crf = int(45 - (q / 100) * 30)
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat, '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p',
                   '-crf', str(crf), '-b:v', '0', '-auto-alt-ref', '0', '-row-mt', '1', '-threads', '0']
            self.log.emit(f"WebM quality: CRF={crf}")
        elif fmt == 'mp4':
            # CRF: q=0→32, q=50→23, q=100→14 (H.264, no alpha)
            crf = int(32 - (q / 100) * 18)
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat,
                   '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart']
            self.log.emit(f"MP4 quality: CRF={crf}")
        elif fmt == 'mp4_nvenc':
            cq = int(32 - (q / 100) * 18)
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat,
                   '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq:v', str(cq), '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart']
            self.log.emit(f"MP4 NVENC quality: CQ={cq}")
        elif fmt == 'mp4_qsv':
            gq = int(32 - (q / 100) * 18)
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat,
                   '-c:v', 'h264_qsv', '-global_quality', str(gq), '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart']
            self.log.emit(f"MP4 QSV quality: global_quality={gq}")
        elif fmt == 'greenscreen':
            # CRF: q=0→30, q=50→23, q=100→14
            crf = int(30 - (q / 100) * 16)
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat, '-filter_complex',
                   f"color=c=#00ff00:s={fw}x{fh}:r={fps}[bg];[bg][0:v]overlay=shortest=1",
                   '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p']
            self.log.emit(f"Greenscreen quality: CRF={crf}")
        elif fmt == 'hevc':
            # CRF: q=0→34, q=50→24, q=100→16 (H.265, no alpha)
            crf = int(34 - (q / 100) * 18)
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat,
                   '-c:v', 'libx265', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', '-tag:v', 'hvc1']
            self.log.emit(f"HEVC quality: CRF={crf}")
        elif fmt == 'hevc_nvenc':
            cq = int(34 - (q / 100) * 18)
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat,
                   '-c:v', 'hevc_nvenc', '-preset', 'p4', '-cq:v', str(cq), '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', '-tag:v', 'hvc1']
            self.log.emit(f"HEVC NVENC quality: CQ={cq}")
        elif fmt == 'hevc_qsv':
            gq = int(34 - (q / 100) * 18)
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat,
                   '-c:v', 'hevc_qsv', '-global_quality', str(gq), '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', '-tag:v', 'hvc1']
            self.log.emit(f"HEVC QSV quality: global_quality={gq}")
        elif fmt == 'av1':
            crf = int(40 - (q / 100) * 22)
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat,
                   '-c:v', 'libsvtav1', '-crf', str(crf), '-preset', '6', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart']
            self.log.emit(f"AV1 quality: CRF={crf}")
        elif fmt == 'matte':
            cmd = [ffmpeg, '-y', '-v', 'warning', '-framerate', str(fps), '-i', pat, '-vf', 'format=gray',
                   '-c:v', 'prores_ks', '-profile:v', '0', '-pix_fmt', 'yuv422p10le']
        else: self.error.emit(f"Unknown format: {fmt}"); return None

        if self.keep_audio and info.get('has_audio') and fmt not in ('png_seq', 'matte', 'webp_anim', 'gif_anim'):
            cmd += ['-i', self.input_path, '-map', '0:v', '-map', '1:a',
                    '-c:a', 'libopus' if fmt == 'webm' else ('aac' if normalize_encoder_format(fmt) in ('greenscreen', 'mp4', 'hevc', 'av1') else 'copy'), '-shortest']
        # Add -progress pipe:1 for real-time encode progress
        cmd += ['-progress', 'pipe:1', out_file]
        self.log.emit(f"Encoding: {fmt}...")
        duration = info.get('duration', 0)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 creationflags=_SUBPROCESS_FLAGS, text=True)
        while proc.poll() is None:
            if self._cancelled:
                proc.terminate(); proc.wait(5); return out_file
            line = proc.stdout.readline().strip() if proc.stdout else ''
            if line.startswith('frame='):
                try:
                    enc_frame = int(line.split('=')[1])
                    if total_frames > 0:
                        enc_pct = 90 + min(10, int((enc_frame / total_frames) * 10))
                        self.progress.emit(enc_pct)
                        self.status.emit(f"Encoding {fmt}... {enc_frame}/{total_frames}")
                except (ValueError, IndexError):
                    pass
            elif line.startswith('out_time_us=') and duration > 0:
                try:
                    us = int(line.split('=')[1])
                    enc_pct = 90 + min(10, int((us / (duration * 1_000_000)) * 10))
                    self.progress.emit(enc_pct)
                except (ValueError, IndexError):
                    pass
        stderr_out = proc.stderr.read() if proc.stderr else ''
        if proc.returncode != 0:
            self.log.emit(f"FFmpeg: {stderr_out[-400:]}")
            return None
        self.progress.emit(100)
        return out_file


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH WORKER — sequential queue of ProcessingWorkers
# ═══════════════════════════════════════════════════════════════════════════════
def _batch_error_summary(message, limit=90):
    """Return a compact first-line batch error for table display."""
    text = str(message or "Unknown error").strip()
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "Unknown error")
    first_line = " ".join(first_line.split())
    if len(first_line) <= limit:
        return first_line
    return first_line[:max(0, limit - 3)].rstrip() + "..."


class BatchWorker(QThread):
    """Processes a queue of jobs sequentially."""
    job_started = pyqtSignal(int, str)       # (index, filename)
    job_progress = pyqtSignal(int, int)      # (index, percent)
    job_status = pyqtSignal(int, str)        # (index, status_text)
    job_finished = pyqtSignal(int, str)      # (index, output_path)
    job_error = pyqtSignal(int, str)         # (index, error_msg)
    all_done = pyqtSignal(int, int)          # (completed, total)
    log = pyqtSignal(str)
    preview = pyqtSignal(object)

    def __init__(self, jobs):
        """jobs: list of dicts with keys: input, output, model_key, format, max_res, edge, shift, temporal, audio"""
        super().__init__()
        self.jobs = jobs
        self._cancelled = False

    def cancel(self): self._cancelled = True

    def run(self):
        completed = 0
        for i, job in enumerate(self.jobs):
            if self._cancelled: break
            self.job_started.emit(i, os.path.basename(job['input']))
            self.log.emit(f"\n{'='*40}\nBatch [{i+1}/{len(self.jobs)}]: {os.path.basename(job['input'])}")

            is_image = job.get('type') == 'image'
            if is_image:
                worker = ImageProcessWorker(
                    job['input'], job['output'], job['model_key'], job['max_res'],
                    edge_softness=job['edge'], mask_shift=job['shift'],
                    invert_mask=job.get('invert_mask', False),
                    spill_strength=job.get('spill_strength', 0),
                    spill_color=job.get('spill_color', 'green'),
                    shadow_strength=job.get('shadow_strength', 0),
                    bg_color=job.get('bg_color'),
                    bg_image_path=job.get('bg_image_path'),
                    roi=job.get('roi'),
                    mask_edits=job.get('mask_edits'),
                    gpu_device=job.get('gpu_device', -1),
                    fp16=job.get('fp16', False))
            else:
                worker = ProcessingWorker(
                    job['input'], job['output'], job['model_key'], job['format'],
                    job['max_res'], job['edge'], job['shift'], job['temporal'], job['audio'],
                    frame_skip=job.get('frame_skip', 1),
                    invert_mask=job.get('invert_mask', False),
                    spill_strength=job.get('spill_strength', 0),
                    spill_color=job.get('spill_color', 'green'),
                    shadow_strength=job.get('shadow_strength', 0),
                    bg_color=job.get('bg_color'),
                    bg_image_path=job.get('bg_image_path'),
                    quality=job.get('quality', 70),
                    roi=job.get('roi'),
                    mask_edits=job.get('mask_edits'),
                    gpu_device=job.get('gpu_device', -1),
                    fp16=job.get('fp16', False),
                    allow_large_animation=job.get('allow_large_animation', False))

            worker_errors = []

            def _record_worker_error(msg, idx=i, errors=worker_errors):
                detail = str(msg)
                errors.append(detail)
                self.log.emit(f"Batch error [{idx+1}/{len(self.jobs)}]: {detail}")

            # Wire signals to batch relay
            worker.progress.connect(lambda pct, idx=i: self.job_progress.emit(idx, pct))
            worker.status.connect(lambda s, idx=i: self.job_status.emit(idx, s))
            worker.log.connect(self.log.emit)
            worker.error.connect(_record_worker_error)
            worker.preview.connect(self.preview.emit)
            # Sync cancel state: when batch is cancelled, propagate to worker via progress updates
            worker.progress.connect(lambda pct, w=worker: setattr(w, '_cancelled', True) if self._cancelled else None)

            # Run synchronously (we're already on a thread)
            try:
                worker.run() if is_image else worker._process()
                if self._cancelled:
                    self.job_error.emit(i, "Cancelled")
                else:
                    # Check if output exists
                    out = job['output']
                    if not is_image and job['format'] != 'png_seq':
                        out = os.path.splitext(out)[0] + format_extension(job['format'])
                    if worker_errors:
                        self.job_error.emit(i, worker_errors[-1])
                    elif os.path.exists(out):
                        self.job_finished.emit(i, out)
                        completed += 1
                    else:
                        self.job_error.emit(i, "No output produced")
            except Exception as e:
                self.job_error.emit(i, str(e))

        self.all_done.emit(completed, len(self.jobs))


# ═══════════════════════════════════════════════════════════════════════════════
# CHROMA KEY WORKER
# ═══════════════════════════════════════════════════════════════════════════════
class ChromaKeyWorker(QThread):
    """FFmpeg-based chroma-key removal — faster than AI for solid-colour backgrounds."""
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    log      = pyqtSignal(str)
    error    = pyqtSignal(str)
    finished = pyqtSignal(object)   # output_path or None

    def __init__(self, input_path, output_path, fmt, chroma_color,
                 similarity, blend, quality=75, keep_audio=True, parent=None):
        super().__init__(parent)
        self.input_path   = input_path
        self.output_path  = output_path
        self.fmt          = fmt
        self.chroma_color = chroma_color   # 'green' or 'blue'
        self.similarity   = similarity
        self.blend        = blend
        self.quality      = quality
        self.keep_audio   = keep_audio
        self._cancelled   = False
        self._proc        = None

    def cancel(self):
        self._cancelled = True
        if self._proc is not None:
            try: self._proc.terminate()
            except Exception: pass

    def run(self):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            self.error.emit("FFmpeg not found."); self.finished.emit(None); return

        info = get_video_info(self.input_path)
        if not info:
            self.error.emit("Could not probe video."); self.finished.emit(None); return

        fps      = info.get('fps', 25.0)
        duration = info.get('duration', 0.0)
        q        = max(0, min(100, self.quality))

        fmt = self.fmt
        chroma_filter = (f"chromakey=color={self.chroma_color}:"
                         f"similarity={self.similarity:.3f}:blend={self.blend:.3f}")

        # Build output path with correct extension
        out = os.path.splitext(self.output_path)[0] + format_extension(fmt)

        self.status.emit(f"Chroma-key removal ({self.chroma_color}) → {fmt}…")

        if fmt == 'png_seq':
            out_dir = os.path.splitext(self.output_path)[0] + '_frames'
            os.makedirs(out_dir, exist_ok=True)
            frame_pattern = os.path.join(out_dir, 'frame_%06d.png')
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', chroma_filter,
                   '-pix_fmt', 'rgba', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   frame_pattern]
            out = out_dir
        elif fmt == 'webm':
            crf = int(45 - (q / 100) * 30)
            vf  = f'{chroma_filter},format=yuva420p'
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', vf,
                   '-c:v', 'libvpx-vp9', '-crf', str(crf), '-b:v', '0',
                   '-auto-alt-ref', '0', '-pix_fmt', 'yuva420p', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   out]
        elif fmt == 'prores':
            bpm = int(800 + (q / 100) ** 1.5 * 7200)
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', chroma_filter,
                   '-c:v', 'prores_ks', '-profile:v', '4444',
                   '-vendor', 'apl0', '-pix_fmt', 'yuva444p10le',
                   '-b:v', f'{bpm}k', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   out]
        elif fmt == 'matte':
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', f'{chroma_filter},lutrgb=r=0:g=0:b=0,alphaextract',
                   '-c:v', 'prores_ks', '-profile:v', '4444', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   out]
        elif fmt == 'hevc':
            crf = int(34 - (q / 100) * 18)
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', f'{chroma_filter},format=yuv420p',
                   '-c:v', 'libx265', '-preset', 'medium', '-crf', str(crf),
                   '-movflags', '+faststart', '-tag:v', 'hvc1', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   out]
        elif fmt == 'av1':
            crf = int(40 - (q / 100) * 22)
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', f'{chroma_filter},format=yuv420p',
                   '-c:v', 'libsvtav1', '-crf', str(crf), '-preset', '6',
                   '-movflags', '+faststart', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   out]
        elif fmt == 'mp4_nvenc':
            cq = int(32 - (q / 100) * 18)
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', f'{chroma_filter},format=yuv420p',
                   '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq:v', str(cq),
                   '-movflags', '+faststart', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   out]
        elif fmt == 'hevc_nvenc':
            cq = int(34 - (q / 100) * 18)
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', f'{chroma_filter},format=yuv420p',
                   '-c:v', 'hevc_nvenc', '-preset', 'p4', '-cq:v', str(cq),
                   '-movflags', '+faststart', '-tag:v', 'hvc1', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   out]
        elif fmt == 'mp4_qsv':
            gq = int(32 - (q / 100) * 18)
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', f'{chroma_filter},format=yuv420p',
                   '-c:v', 'h264_qsv', '-global_quality', str(gq),
                   '-movflags', '+faststart', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   out]
        elif fmt == 'hevc_qsv':
            gq = int(34 - (q / 100) * 18)
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', f'{chroma_filter},format=yuv420p',
                   '-c:v', 'hevc_qsv', '-global_quality', str(gq),
                   '-movflags', '+faststart', '-tag:v', 'hvc1', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   out]
        else:  # mp4 default
            crf = int(32 - (q / 100) * 18)
            cmd = [ffmpeg, '-i', self.input_path,
                   '-vf', f'{chroma_filter},format=yuv420p',
                   '-c:v', 'libx264', '-preset', 'fast', '-crf', str(crf),
                   '-movflags', '+faststart', '-y',
                   '-progress', 'pipe:1', '-nostats',
                   out]

        # Append audio stream if appropriate
        if (self.keep_audio and info.get('has_audio')
                and fmt not in ('png_seq', 'matte')):
            cmd.insert(-1, '-c:a'); cmd.insert(-1, 'aac')

        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace')
            total_frames = max(1, int(duration * fps))
            current_frame = 0
            for line in self._proc.stdout:
                if self._cancelled:
                    break
                line = line.strip()
                if line.startswith('frame='):
                    try: current_frame = int(line.split('=')[1])
                    except ValueError: pass
                    pct = min(99, int((current_frame / total_frames) * 100))
                    self.progress.emit(pct)
                elif '=' not in line and line:
                    self.log.emit(line)
            self._proc.wait()
            rc = self._proc.returncode
        except Exception as e:
            self.error.emit(f"Chroma-key failed: {e}"); self.finished.emit(None); return

        if self._cancelled:
            try:
                if os.path.isfile(out): os.remove(out)
            except Exception: pass
            self.finished.emit(None); return

        if rc != 0:
            self.error.emit(f"FFmpeg exited {rc} during chroma-key."); self.finished.emit(None); return

        self.progress.emit(100)
        self.finished.emit(out)


# ═══════════════════════════════════════════════════════════════════════════════
# THUMBNAIL LOADER
# ═══════════════════════════════════════════════════════════════════════════════
class ThumbnailLoader(QThread):
    """Extract small preview thumbnails from video files in a background thread."""
    thumbnail_ready = pyqtSignal(int, object)   # (row, QPixmap)

    def __init__(self, jobs, parent=None):
        """*jobs* is a list of (row, video_path) tuples."""
        super().__init__(parent)
        self._jobs      = list(jobs)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        ffmpeg = find_ffmpeg()
        for row, path in self._jobs:
            if self._cancelled:
                break
            try:
                if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS:
                    img = Image.open(path)
                    img.thumbnail((80, 60), Image.Resampling.LANCZOS)
                    pixmap = QPixmap.fromImage(pil_to_qimage(img.convert('RGBA')))
                    if not pixmap.isNull():
                        self.thumbnail_ready.emit(row, pixmap)
                elif ffmpeg:
                    cmd = [ffmpeg, '-ss', '0.5', '-i', path,
                           '-frames:v', '1', '-vf', 'scale=80:-1',
                           '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1',
                           '-loglevel', 'quiet']
                    proc = subprocess.run(cmd, capture_output=True, timeout=10,
                                          creationflags=_SUBPROCESS_FLAGS)
                    if proc.returncode == 0 and proc.stdout:
                        pixmap = QPixmap()
                        pixmap.loadFromData(proc.stdout)
                        if not pixmap.isNull():
                            self.thumbnail_ready.emit(row, pixmap)
            except Exception as e:
                _log_warning(f"Thumbnail failed for {os.path.basename(path)}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PREVIEW WORKER
# ═══════════════════════════════════════════════════════════════════════════════
class PreviewFrameWorker(QThread):
    result = pyqtSignal(object, object, object, object, object, object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, model_key, max_res, edge_softness=0, mask_shift=0, seek_pct=0.1,
                 invert_mask=False, spill_strength=0, spill_color='green',
                 shadow_strength=0, bg_color=None, bg_image_path=None, roi=None, mask_edits=None,
                 gpu_device=-1, fp16=False):
        super().__init__()
        self.input_path = input_path; self.model_key = model_key
        self.max_res = max_res; self.edge_softness = edge_softness
        self.mask_shift = mask_shift; self.seek_pct = seek_pct
        self.invert_mask = invert_mask; self.spill_strength = spill_strength
        self.spill_color = spill_color; self.shadow_strength = shadow_strength
        self.bg_color = bg_color; self.bg_image_path = bg_image_path
        self.roi = roi
        self.mask_edits = mask_edits or []
        self.gpu_device = int(gpu_device) if gpu_device is not None else -1
        self.fp16 = fp16
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        tmp = None
        try:
            ext = os.path.splitext(self.input_path)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                img = Image.open(self.input_path)
                img.load()
                w, h = img.size
                if self.max_res > 0 and max(w, h) > self.max_res:
                    ratio = self.max_res / max(w, h)
                    img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            else:
                ffmpeg = find_ffmpeg()
                if not ffmpeg: self.error.emit("FFmpeg not found."); return
                info = get_video_info(self.input_path)
                if not info: self.error.emit("Could not read video info."); return
                seek_sec = max(0, info['duration'] * self.seek_pct)
                w, h = info['width'], info['height']
                fd, tmp = tempfile.mkstemp(suffix='.bmp', prefix='alphacut_preview_'); os.close(fd)
                cmd = [ffmpeg, '-ss', f'{seek_sec:.2f}', '-i', self.input_path, '-frames:v', '1', '-y']
                if self.max_res > 0 and max(w, h) > self.max_res:
                    ratio = self.max_res / max(w, h)
                    cmd += ['-vf', f'scale={int(w*ratio)//2*2}:{int(h*ratio)//2*2}']
                cmd.append(tmp)
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, creationflags=_SUBPROCESS_FLAGS)
                if proc.returncode != 0 or not os.path.isfile(tmp): self.error.emit("Failed to extract frame."); return
                img = Image.open(tmp)
            self.status.emit("Running AI model...")
            engine = get_engine(self.model_key, log_fn=lambda m: self.status.emit(m),
                                gpu_device=self.gpu_device, fp16=self.fp16)

            # Full compositing pipeline (mirrors ProcessingWorker)
            mask = engine.predict_mask(img, self.roi)
            if self.edge_softness > 0 or self.mask_shift != 0:
                mask = engine.refine_mask(mask, self.edge_softness, self.mask_shift, 0)
            if self.mask_edits:
                mask = AlphaCutEngine.apply_mask_edits(mask, self.mask_edits)
            current_mask = mask.copy()
            if self.shadow_strength > 0:
                current_mask = AlphaCutEngine.preserve_shadows(img, current_mask, self.shadow_strength)
            if self.invert_mask:
                current_mask = AlphaCutEngine.invert_mask(current_mask)
            src = img
            if self.spill_strength > 0:
                src = AlphaCutEngine.suppress_spill(src, current_mask, self.spill_strength, self.spill_color)
            fg = src.convert('RGBA')
            result = Image.composite(fg, Image.new('RGBA', fg.size, 0), current_mask)
            # Background replacement
            bg_image = None
            if self.bg_image_path and os.path.isfile(self.bg_image_path):
                try: bg_image = Image.open(self.bg_image_path); bg_image.load()
                except Exception: pass
            if self.bg_color is not None or bg_image is not None:
                result = AlphaCutEngine.composite_on_background(result, bg_color=self.bg_color, bg_image=bg_image)

            mask_metrics = inspect_mask_quality(current_mask)
            mask_bw, mask_gray, mask_overlay = build_mask_preview_qimages(img, current_mask)
            self.result.emit(pil_to_qimage(img.convert('RGBA')), pil_to_qimage(result),
                             mask_bw, mask_gray, mask_overlay, mask_metrics)
        except Exception as e: self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            if tmp and os.path.isfile(tmp):
                try: os.remove(tmp)
                except Exception: pass


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESS WORKER — single-image background removal
# ═══════════════════════════════════════════════════════════════════════════════
class ImageProcessWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    preview = pyqtSignal(object)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, output_path, model_key, max_res=0,
                 edge_softness=0, mask_shift=0, invert_mask=False,
                 spill_strength=0, spill_color='green', shadow_strength=0,
                 bg_color=None, bg_image_path=None, roi=None, mask_edits=None,
                 gpu_device=-1, fp16=False):
        super().__init__()
        self.input_path = input_path; self.output_path = output_path
        self.model_key = model_key; self.max_res = max_res
        self.edge_softness = edge_softness; self.mask_shift = mask_shift
        self.invert_mask = invert_mask; self.spill_strength = spill_strength
        self.spill_color = spill_color; self.shadow_strength = shadow_strength
        self.bg_color = bg_color; self.bg_image_path = bg_image_path
        self.roi = roi; self.mask_edits = mask_edits or []
        self.gpu_device = int(gpu_device) if gpu_device is not None else -1
        self.fp16 = fp16
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.status.emit("Loading image...")
            self.progress.emit(10)
            img = Image.open(self.input_path)
            img.load()
            w, h = img.size
            self.log.emit(f"Image: {w}x{h} | {os.path.basename(self.input_path)}")
            if self.max_res > 0 and max(w, h) > self.max_res:
                ratio = self.max_res / max(w, h)
                nw, nh = int(w * ratio), int(h * ratio)
                img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                self.log.emit(f"Scaled to {nw}x{nh}")
            self.status.emit("Loading AI model...")
            self.progress.emit(20)
            engine = get_engine(self.model_key, log_fn=self.log.emit,
                                gpu_device=self.gpu_device, fp16=self.fp16)
            if self._cancelled:
                return
            self.status.emit("Removing background...")
            self.progress.emit(40)
            mask = engine.predict_mask(img, self.roi)
            if self._cancelled:
                return
            if self.edge_softness > 0 or self.mask_shift != 0:
                mask = engine.refine_mask(mask, self.edge_softness, self.mask_shift, 0)
            if self.mask_edits:
                mask = AlphaCutEngine.apply_mask_edits(mask, self.mask_edits)
            current_mask = mask.copy()
            if self.shadow_strength > 0:
                current_mask = AlphaCutEngine.preserve_shadows(img, current_mask, self.shadow_strength)
            if self.invert_mask:
                current_mask = AlphaCutEngine.invert_mask(current_mask)
            self.progress.emit(70)
            src = img
            if self.spill_strength > 0:
                src = AlphaCutEngine.suppress_spill(src, current_mask, self.spill_strength, self.spill_color)
            fg = src.convert('RGBA')
            result = Image.composite(fg, Image.new('RGBA', fg.size, 0), current_mask)
            bg_image = None
            if self.bg_image_path and os.path.isfile(self.bg_image_path):
                try:
                    bg_image = Image.open(self.bg_image_path); bg_image.load()
                except Exception:
                    pass
            if self.bg_color is not None or bg_image is not None:
                result = AlphaCutEngine.composite_on_background(result, bg_color=self.bg_color, bg_image=bg_image)
            self.status.emit("Saving...")
            self.progress.emit(90)
            if self._cancelled:
                return
            out = self.output_path
            if not out.lower().endswith('.png'):
                out = os.path.splitext(out)[0] + '.png'
            result.save(out, 'PNG')
            try:
                self.preview.emit(pil_to_qimage(result))
            except Exception:
                pass
            self.progress.emit(100)
            self.log.emit(f"Output: {out} ({os.path.getsize(out)/(1024*1024):.1f} MB)")
            self.finished.emit(out)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK WORKER — estimate processing time from N sample frames
# ═══════════════════════════════════════════════════════════════════════════════
class BenchmarkWorker(QThread):
    result = pyqtSignal(dict)   # {fps, eta_sec, eta_str, frames_tested, total_frames, ram_pct}
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    SAMPLE_FRAMES = 10

    def __init__(self, input_path, model_key, max_res, edge_softness=0, mask_shift=0,
                 invert_mask=False, spill_strength=0, spill_color='green',
                 shadow_strength=0, bg_color=None, bg_image_path=None, roi=None, mask_edits=None,
                 gpu_device=-1, fp16=False):
        super().__init__()
        self.input_path = input_path; self.model_key = model_key
        self.max_res = max_res; self.edge_softness = edge_softness
        self.mask_shift = mask_shift; self.fp16 = fp16
        self.invert_mask = invert_mask; self.spill_strength = spill_strength
        self.spill_color = spill_color; self.shadow_strength = shadow_strength
        self.bg_color = bg_color; self.bg_image_path = bg_image_path
        self.roi = roi
        self.mask_edits = mask_edits or []
        self.gpu_device = int(gpu_device) if gpu_device is not None else -1

    def run(self):
        tmp_dir = None
        try:
            ffmpeg = find_ffmpeg()
            if not ffmpeg: self.error.emit("FFmpeg not found."); return
            info = get_video_info(self.input_path)
            if not info: self.error.emit("Could not read video."); return

            total = info['total_frames']
            n = min(self.SAMPLE_FRAMES, total)
            w, h = info['width'], info['height']

            self.status.emit(_trf("status.benchmark_extracting", "Benchmark: extracting {count} sample frames...", count=n))
            tmp_dir = tempfile.mkdtemp(prefix='alphacut_bench_')

            # Load background image once if needed
            bg_image = None
            if self.bg_image_path and os.path.isfile(self.bg_image_path):
                try: bg_image = Image.open(self.bg_image_path); bg_image.load()
                except Exception: pass

            # Extract N evenly spaced frames
            frames = []
            for i in range(n):
                seek = info['duration'] * (i + 0.5) / n
                fd, tmp = tempfile.mkstemp(suffix='.bmp', dir=tmp_dir); os.close(fd)
                cmd = [ffmpeg, '-ss', f'{seek:.2f}', '-i', self.input_path, '-frames:v', '1', '-y']
                if self.max_res > 0 and max(w, h) > self.max_res:
                    ratio = self.max_res / max(w, h)
                    cmd += ['-vf', f'scale={int(w*ratio)//2*2}:{int(h*ratio)//2*2}']
                cmd.append(tmp)
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                      creationflags=_SUBPROCESS_FLAGS)
                if proc.returncode == 0 and os.path.isfile(tmp):
                    frames.append(tmp)

            if not frames: self.error.emit("No frames extracted."); return

            self.status.emit(_trf("status.benchmark_processing", "Benchmark: running AI + compositing on {count} frames...", count=len(frames)))
            engine = get_engine(self.model_key, log_fn=lambda m: self.status.emit(m),
                                gpu_device=self.gpu_device, fp16=self.fp16)

            t0 = time.time()
            for fpath in frames:
                img = Image.open(fpath)
                # Full compositing pipeline to get accurate timing
                mask = engine.predict_mask(img, self.roi)
                if self.edge_softness > 0 or self.mask_shift != 0:
                    mask = engine.refine_mask(mask, self.edge_softness, self.mask_shift, 0)
                if self.mask_edits:
                    mask = AlphaCutEngine.apply_mask_edits(mask, self.mask_edits)
                current_mask = mask
                if self.shadow_strength > 0:
                    current_mask = AlphaCutEngine.preserve_shadows(img, current_mask, self.shadow_strength)
                if self.invert_mask:
                    current_mask = AlphaCutEngine.invert_mask(current_mask)
                src = img
                if self.spill_strength > 0:
                    src = AlphaCutEngine.suppress_spill(src, current_mask, self.spill_strength, self.spill_color)
                fg = src.convert('RGBA')
                result = Image.composite(fg, Image.new('RGBA', fg.size, 0), current_mask)
                if self.bg_color is not None or bg_image is not None:
                    AlphaCutEngine.composite_on_background(result, bg_color=self.bg_color, bg_image=bg_image)
            elapsed = time.time() - t0

            fps = len(frames) / max(elapsed, 0.001)
            eta_sec = total / fps
            mem = get_memory_usage()

            self.result.emit({
                'fps': fps, 'eta_sec': eta_sec, 'eta_str': _ftime(eta_sec),
                'frames_tested': len(frames), 'total_frames': total,
                'ram_pct': mem.get('ram_pct', 0),
                'elapsed': elapsed,
            })
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DOWNLOAD WORKER — cancellable download with progress signals
# ═══════════════════════════════════════════════════════════════════════════════
class QuickPreviewWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    preview = pyqtSignal(object)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, output_path, model_key, output_format, max_res,
                 edge_softness=0, mask_shift=0, temporal_smooth=0, keep_audio=False,
                 frame_skip=1, invert_mask=False, spill_strength=0, spill_color='green',
                 shadow_strength=0, bg_color=None, bg_image_path=None, quality=70,
                 roi=None, mask_edits=None, gpu_device=-1, fp16=False, duration=10):
        super().__init__()
        self.input_path = input_path; self.output_path = output_path
        self.model_key = model_key; self.output_format = output_format
        self.max_res = max_res; self.edge_softness = edge_softness
        self.mask_shift = mask_shift; self.temporal_smooth = temporal_smooth
        self.keep_audio = keep_audio; self.frame_skip = frame_skip
        self.invert_mask = invert_mask; self.spill_strength = spill_strength
        self.spill_color = spill_color; self.shadow_strength = shadow_strength
        self.bg_color = bg_color; self.bg_image_path = bg_image_path
        self.quality = quality; self.roi = roi; self.mask_edits = mask_edits or []
        self.gpu_device = int(gpu_device) if gpu_device is not None else -1
        self.fp16 = fp16; self.duration = duration
        self._cancelled = False
        self._worker = None

    def cancel(self):
        self._cancelled = True
        if self._worker:
            self._worker.cancel()

    def run(self):
        try:
            fmt = self.output_format
            if fmt == 'png_seq':
                fmt = 'mp4'
            worker = ProcessingWorker(
                self.input_path, self.output_path, self.model_key, fmt, self.max_res,
                edge_softness=self.edge_softness, mask_shift=self.mask_shift,
                temporal_smooth=self.temporal_smooth, keep_audio=self.keep_audio,
                frame_skip=self.frame_skip, invert_mask=self.invert_mask,
                spill_strength=self.spill_strength, spill_color=self.spill_color,
                shadow_strength=self.shadow_strength, bg_color=self.bg_color,
                bg_image_path=self.bg_image_path, quality=self.quality,
                roi=self.roi, mask_edits=self.mask_edits, gpu_device=self.gpu_device,
                fp16=self.fp16, limit_seconds=self.duration)
            self._worker = worker
            worker.progress.connect(self.progress.emit)
            worker.status.connect(self.status.emit)
            worker.log.connect(self.log.emit)
            worker.preview.connect(self.preview.emit)
            worker.error.connect(self.error.emit)
            worker._process()
            out = os.path.splitext(self.output_path)[0] + format_extension(fmt)
            if not self._cancelled and os.path.exists(out):
                self.finished.emit(out)
            elif not self._cancelled:
                self.error.emit("Quick preview did not produce an output file.")
        except Exception as e:
            if not self._cancelled:
                self.error.emit(f"{type(e).__name__}: {e}")


class ModelDownloadWorker(QThread):
    progress = pyqtSignal(int, int)   # (downloaded_bytes, total_bytes)
    finished = pyqtSignal(str)        # model_path
    error = pyqtSignal(str)

    def __init__(self, model_key, gpu_device=-1, parent=None):
        super().__init__(parent)
        self.model_key = model_key
        self.gpu_device = int(gpu_device) if gpu_device is not None else -1
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            engine = AlphaCutEngine(self.model_key, log_fn=lambda m: None,
                                     gpu_device=self.gpu_device)
            path = engine._ensure_model(
                progress_fn=lambda dl, tot: self.progress.emit(dl, tot),
                cancel_check=lambda: self._cancelled)
            if not self._cancelled:
                self.finished.emit(path)
        except RuntimeError as e:
            if not self._cancelled:
                self.error.emit(str(e))
        except Exception as e:
            if not self._cancelled:
                self.error.emit(f"{type(e).__name__}: {e}")


def _model_needs_download(model_key):
    """Check if a model needs downloading (not cached locally)."""
    cfg = MODELS.get(model_key)
    if not cfg:
        return False
    path = os.path.join(MODELS_DIR, os.path.basename(cfg['file']))
    return not (os.path.isfile(path) and os.path.getsize(path) > 1_000_000)


# ═══════════════════════════════════════════════════════════════════════════════
# WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════
class DropZone(QLabel):
    file_dropped = pyqtSignal(str)
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(60); self._set_style(False)
        self.setText(_tr("drop.hint", "Drop video/image file(s) here or click Browse"))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(_tr("access.drop_zone", "Input drop zone"))
        self.setAccessibleDescription(_tr(
            "access.drop_zone.desc",
            "Drop one or more supported media files here, or use the Browse buttons below."
        ))

    def _set_style(self, active):
        bc = '#6c5ce7' if active else '#1e2230'
        bg = '#0f0a1a' if active else '#0a0c10'
        tc = '#a855f7' if active else '#3d4455'
        self.setStyleSheet(f"QLabel {{ background-color: {bg}; border: 2px dashed {bc}; border-radius: 12px; color: {tc}; font-size: 12px; padding: 10px; }}")

    @staticmethod
    def _is_supported(ext):
        return ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    ext = os.path.splitext(url.toLocalFile())[1].lower()
                    if self._is_supported(ext) or os.path.isdir(url.toLocalFile()):
                        event.acceptProposedAction(); self._set_style(True); return
        event.ignore()

    def dragLeaveEvent(self, event): self._set_style(False)

    def dropEvent(self, event: QDropEvent):
        self._set_style(False)
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                p = url.toLocalFile()
                if os.path.isdir(p):
                    for f in sorted(os.listdir(p)):
                        if self._is_supported(os.path.splitext(f)[1].lower()):
                            paths.append(os.path.join(p, f))
                elif self._is_supported(os.path.splitext(p)[1].lower()):
                    paths.append(p)
        if len(paths) == 1:
            self.file_dropped.emit(paths[0])
        elif len(paths) > 1:
            self.files_dropped.emit(paths)


class SplitPreviewWidget(QWidget):
    edits_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 160)
        self._original = None; self._processed = None
        self._mask_bw = None; self._mask_gray = None; self._mask_overlay = None
        self._view_mode = 'compare'
        self._split_pos = 0.5; self._dragging = False; self._show_split = False
        self._edit_mode = 'compare'
        self._brush_size = 32
        self._roi = None
        self._roi_anchor = None
        self._drawing_roi = False
        self._drawing_stroke = False
        self._mask_strokes = []
        self.setMouseTracking(True)

    def set_images(self, orig, proc, mask_bw=None, mask_gray=None, mask_overlay=None):
        self._original = orig; self._processed = proc
        self._mask_bw = mask_bw; self._mask_gray = mask_gray; self._mask_overlay = mask_overlay
        self._show_split = orig is not None and proc is not None
        self._split_pos = 0.5; self.update()

    def set_frame(self, qimage):
        self._processed = qimage; self._original = None
        self._mask_bw = None; self._mask_gray = None; self._mask_overlay = None
        self._show_split = False; self.update()

    def set_view_mode(self, mode):
        self._view_mode = mode or 'compare'
        self.update()

    def set_edit_mode(self, mode):
        self._edit_mode = mode
        if mode == 'compare':
            self.setCursor(Qt.CursorShape.SplitHCursor if self._show_split else Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def set_brush_size(self, size):
        self._brush_size = max(1, int(size))
        self.update()

    def undo_last_stroke(self):
        if self._mask_strokes:
            self._mask_strokes.pop()
            self.update()
            self.edits_changed.emit()
            return True
        return False

    def clear_mask_edits(self):
        self._roi = None
        self._mask_strokes.clear()
        self._drawing_roi = False
        self._drawing_stroke = False
        self.update()
        self.edits_changed.emit()

    def get_roi(self):
        return dict(self._roi) if self._roi else None

    def get_mask_edits(self):
        return [{'mode': s['mode'], 'size': s['size'], 'points': list(s['points'])}
                for s in self._mask_strokes if s.get('points')]

    def edit_summary(self):
        parts = []
        if self._roi:
            parts.append(_tr("mask_tool.roi", "ROI"))
        if self._mask_strokes:
            parts.append(_trf("mask_edits.strokes", "{count} stroke(s)", count=len(self._mask_strokes)))
        return " + ".join(parts) if parts else _tr("mask_edits.none", "No mask edits")

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(10, 12, 16))
        image = self._image_for_view_mode()
        if self._view_mode == 'compare' and self._show_split and self._original and self._processed: self._paint_split(p)
        elif image: self._paint_single(p, image)
        else:
            p.setPen(QColor(61, 68, 85))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, _tr("preview.empty", "Preview appears here"))
        if self._processed and self._view_mode in ('compare', 'original', 'result'):
            self._paint_edit_overlay(p)
        p.end()

    def _image_for_view_mode(self):
        if self._view_mode == 'original':
            return self._original or self._processed
        if self._view_mode == 'mask_bw':
            return self._mask_bw or self._processed
        if self._view_mode == 'mask_gray':
            return self._mask_gray or self._processed
        if self._view_mode == 'mask_overlay':
            return self._mask_overlay or self._processed
        return self._processed

    def _image_rect_for(self, iw, ih):
        w, h = self.width(), self.height()
        if iw <= 0 or ih <= 0:
            return (0, 0, w, h, 1.0)
        scale = min(w / iw, h / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        return ((w - dw) // 2, (h - dh) // 2, dw, dh, scale)

    def _current_image_size(self):
        qi = self._processed if self._processed is not None else self._original
        if qi is None:
            return (0, 0)
        return (qi.width(), qi.height())

    def _display_rect(self):
        iw, ih = self._current_image_size()
        return self._image_rect_for(iw, ih)

    def _paint_split(self, p):
        iw, ih = self._processed.width(), self._processed.height()
        if iw == 0 or ih == 0: return
        ox, oy, dw, dh, _ = self._image_rect_for(iw, ih)
        checker = self._checkerboard(dw, dh); p.drawImage(ox, oy, checker)
        sx = ox + int(dw * self._split_pos)
        orig_s = self._original.scaled(dw, dh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        p.setClipRect(ox, oy, sx - ox, dh); p.drawImage(ox, oy, orig_s)
        proc_s = self._processed.scaled(dw, dh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        p.setClipRect(sx, oy, ox + dw - sx, dh); p.drawImage(ox, oy, proc_s); p.setClipping(False)
        pen = QPen(QColor(108, 92, 231), 2); p.setPen(pen); p.drawLine(sx, oy, sx, oy + dh)
        hy = oy + dh // 2; p.setBrush(QColor(108, 92, 231)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(sx - 10, hy - 10, 20, 20)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(sx-5, hy, sx-2, hy-3); p.drawLine(sx-5, hy, sx-2, hy+3)
        p.drawLine(sx+5, hy, sx+2, hy-3); p.drawLine(sx+5, hy, sx+2, hy+3)
        p.setPen(QColor(200, 200, 220, 180)); f = p.font(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        p.drawText(ox + 8, oy + 18, _tr("preview.original", "ORIGINAL"))
        p.drawText(ox + dw - 75, oy + 18, _tr("preview.result", "RESULT"))

    def _paint_single(self, p, qi):
        w, h = self.width(), self.height()
        iw, ih = qi.width(), qi.height()
        if iw == 0 or ih == 0: return
        ox, oy, dw, dh, _ = self._image_rect_for(iw, ih)
        p.drawImage(ox, oy, self._checkerboard(dw, dh))
        p.drawImage(ox, oy, qi.scaled(dw, dh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _paint_edit_overlay(self, p):
        iw, ih = self._current_image_size()
        if iw <= 0 or ih <= 0:
            return
        ox, oy, dw, dh, scale = self._display_rect()
        if self._roi:
            x = ox + int(self._roi['x'] * dw)
            y = oy + int(self._roi['y'] * dh)
            rw = int(self._roi['w'] * dw)
            rh = int(self._roi['h'] * dh)
            p.setPen(QPen(QColor(245, 158, 11), 2, Qt.PenStyle.DashLine))
            p.setBrush(QColor(245, 158, 11, 32))
            p.drawRect(x, y, rw, rh)
        for stroke in self._mask_strokes:
            pts = stroke.get('points') or []
            if not pts:
                continue
            color = QColor(34, 197, 94, 190) if stroke.get('mode') == 'fg' else QColor(239, 68, 68, 190)
            width = max(2, int(stroke.get('size', 24) * scale))
            p.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            mapped = [(ox + int(x * dw), oy + int(y * dh)) for x, y in pts]
            if len(mapped) == 1:
                x, y = mapped[0]
                r = max(2, width // 2)
                p.setBrush(color)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(x - r, y - r, r * 2, r * 2)
            else:
                for a, b in zip(mapped, mapped[1:]):
                    p.drawLine(a[0], a[1], b[0], b[1])

    def _map_to_norm(self, pos):
        ox, oy, dw, dh, _ = self._display_rect()
        x = float(pos.x()); y = float(pos.y())
        if x < ox or y < oy or x > ox + dw or y > oy + dh:
            return None
        return ((x - ox) / max(dw, 1), (y - oy) / max(dh, 1))

    @staticmethod
    def _roi_from_points(a, b):
        x1, y1 = a; x2, y2 = b
        x = max(0.0, min(x1, x2)); y = max(0.0, min(y1, y2))
        w = min(1.0, max(x1, x2)) - x; h = min(1.0, max(y1, y2)) - y
        if w < 0.01 or h < 0.01:
            return None
        return {'x': x, 'y': y, 'w': w, 'h': h}

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self._edit_mode == 'compare':
            if self._show_split:
                self._dragging = True; self._update_split(e.position().x())
            return
        if self._processed is None and self._original is None:
            return
        pt = self._map_to_norm(e.position())
        if pt is None:
            return
        if self._edit_mode == 'roi':
            self._roi_anchor = pt
            self._roi = {'x': pt[0], 'y': pt[1], 'w': 0.0, 'h': 0.0}
            self._drawing_roi = True
            self.update()
        elif self._edit_mode in ('fg', 'bg'):
            self._mask_strokes.append({'mode': self._edit_mode, 'size': self._brush_size, 'points': [pt]})
            self._drawing_stroke = True
            self.update()

    def mouseMoveEvent(self, e):
        if self._edit_mode == 'compare':
            if self._show_split: self.setCursor(Qt.CursorShape.SplitHCursor)
            if self._dragging: self._update_split(e.position().x())
            return
        self.setCursor(Qt.CursorShape.CrossCursor)
        pt = self._map_to_norm(e.position())
        if pt is None:
            return
        if self._drawing_roi and self._roi_anchor:
            self._roi = self._roi_from_points(self._roi_anchor, pt)
            self.update()
        elif self._drawing_stroke and self._mask_strokes:
            self._mask_strokes[-1]['points'].append(pt)
            self.update()

    def mouseReleaseEvent(self, e):
        if self._edit_mode == 'compare':
            self._dragging = False
            return
        changed = self._drawing_roi or self._drawing_stroke
        self._drawing_roi = False
        self._drawing_stroke = False
        self._roi_anchor = None
        if changed:
            self.update()
            self.edits_changed.emit()

    def _update_split(self, mx):
        w = self.width(); iw = self._processed.width() if self._processed else w
        ih = self._processed.height() if self._processed else self.height()
        scale = min(w/iw, self.height()/ih); dw = int(iw*scale); ox = (w-dw)//2
        self._split_pos = max(0.02, min(0.98, (mx - ox) / max(dw, 1))); self.update()

    @staticmethod
    def _checkerboard(w, h, cell=12):
        img = QImage(w, h, QImage.Format.Format_RGB32); p = QPainter(img)
        c1, c2 = QColor(45,45,55), QColor(30,30,40)
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                p.fillRect(x, y, cell, cell, c1 if (x//cell+y//cell)%2==0 else c2)
        p.end(); return img


class ToastWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter); self.setFixedHeight(36)
        self.setStyleSheet("QLabel { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6c5ce7, stop:1 #a855f7); color: #fff; font-weight: 700; font-size: 12px; border-radius: 18px; padding: 0 20px; }")
        self._effect = QGraphicsOpacityEffect(self); self.setGraphicsEffect(self._effect)
        self._effect.setOpacity(0); self.hide()
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self._fade_out)

    def show_toast(self, text, duration=3000):
        self.setText(text); self.adjustSize()
        if self.parent():
            pw = self.parent().width()
            self.setFixedWidth(min(self.sizeHint().width() + 40, pw - 40))
            self.move((pw - self.width()) // 2, 20)
        self.show(); self.raise_()
        anim = QPropertyAnimation(self._effect, b"opacity", self)
        anim.setDuration(200); anim.setStartValue(0.0); anim.setEndValue(1.0); anim.start()
        self._fade_in_anim = anim; self._timer.start(duration)

    def _fade_out(self):
        anim = QPropertyAnimation(self._effect, b"opacity", self)
        anim.setDuration(500); anim.setStartValue(1.0); anim.setEndValue(0.0)
        anim.finished.connect(self.hide); anim.start(); self._fade_out_anim = anim


class StatusLabel(QLabel):
    """Status text that keeps a visible non-color cue in sync."""
    _STYLES = {
        "READY": "background: #151f18; color: #bbf7d0; border: 1px solid #166534;",
        "RUNNING": "background: #161b2f; color: #c7d2fe; border: 1px solid #4338ca;",
        "DONE": "background: #11231c; color: #a7f3d0; border: 1px solid #047857;",
        "WARN": "background: #2a1d0a; color: #fde68a; border: 1px solid #92400e;",
        "ERROR": "background: #2b1117; color: #fecaca; border: 1px solid #991b1b;",
        "CANCELLED": "background: #211a2e; color: #ddd6fe; border: 1px solid #6d28d9;",
        "INFO": "background: #151922; color: #c8ccd4; border: 1px solid #2a2e40;",
    }

    def __init__(self, text=""):
        super().__init__("")
        self._cue_label = None
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setAccessibleName(_tr("access.status", "Processing status"))
        self.setText(text)

    def bind_cue_label(self, cue_label):
        self._cue_label = cue_label
        self._cue_label.setObjectName("statusCue")
        self._cue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cue_label.setAccessibleName(_tr("access.status_cue", "Status cue"))
        self._sync_cue(self.text())

    def setText(self, text):
        value = str(text or "")
        super().setText(value)
        self.setToolTip(value)
        self.setAccessibleDescription(value)
        self._sync_cue(value)

    @classmethod
    def _classify(cls, text):
        lower = str(text or "").lower()
        if any(word in lower for word in ("error", "failed", "not found", "could not", "cannot", "required")):
            return "ERROR"
        if "cancel" in lower:
            return "CANCELLED"
        if any(word in lower for word in ("warning", "high memory", "low disk")):
            return "WARN"
        if any(word in lower for word in ("complete", "done", "ready:", "preview ready", "up to date")):
            return "DONE"
        if lower.strip() == "ready" or lower.startswith("ready "):
            return "READY"
        if any(word in lower for word in (
            "analyzing", "extracting", "loading", "removing", "encoding",
            "processing", "rendering", "benchmark", "starting", "checking",
            "downloading", "saving", "building", "splitting"
        )):
            return "RUNNING"
        return "INFO"

    def _sync_cue(self, text):
        if not self._cue_label:
            return
        cue = self._classify(text)
        self._cue_label.setText(cue)
        self._cue_label.setStyleSheet(f"QLabel#statusCue {{ {self._STYLES.get(cue, self._STYLES['INFO'])} }}")
        self._cue_label.setAccessibleDescription(f"{cue}: {text}")


class JobTable(QTableWidget):
    """Batch job queue table with thumbnail previews."""
    # Column indices
    COL_THUMB    = 0
    COL_FILE     = 1
    COL_SETTINGS = 2
    COL_STATUS   = 3
    COL_PROGRESS = 4
    COL_OUTPUT   = 5

    def __init__(self):
        super().__init__(0, 6)
        self.setHorizontalHeaderLabels([
            "",
            _tr("table.file", "File"),
            _tr("table.settings", "Settings"),
            _tr("table.status", "Status"),
            _tr("table.progress", "Progress"),
            _tr("table.output", "Output"),
        ])
        self.horizontalHeader().setSectionResizeMode(self.COL_THUMB,    QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(self.COL_FILE,     QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(self.COL_SETTINGS, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(self.COL_STATUS,   QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(self.COL_PROGRESS, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(self.COL_OUTPUT,   QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().resizeSection(self.COL_THUMB, 88)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(52)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setMaximumHeight(260)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def add_job(self, filename, settings=""):
        row = self.rowCount(); self.insertRow(row)
        # Placeholder thumbnail cell
        thumb_lbl = QLabel(); thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_lbl.setStyleSheet("background: transparent;")
        self.setCellWidget(row, self.COL_THUMB, thumb_lbl)
        self.setItem(row, self.COL_FILE,     QTableWidgetItem(filename))
        self.setItem(row, self.COL_SETTINGS, QTableWidgetItem(settings))
        self.setItem(row, self.COL_STATUS,   QTableWidgetItem(_tr("status.queued", "Queued")))
        self.setItem(row, self.COL_PROGRESS, QTableWidgetItem("0%"))
        self.setItem(row, self.COL_OUTPUT,   QTableWidgetItem("--"))
        return row

    def set_thumbnail(self, row, pixmap):
        """Attach a scaled thumbnail to the thumb column of *row*."""
        lbl = self.cellWidget(row, self.COL_THUMB)
        if lbl is None:
            lbl = QLabel(); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: transparent;")
            self.setCellWidget(row, self.COL_THUMB, lbl)
        scaled = pixmap.scaledToHeight(48, Qt.TransformationMode.SmoothTransformation)
        lbl.setPixmap(scaled)

    def update_status(self, row, status):
        item = self.item(row, self.COL_STATUS)
        if item: item.setText(status)

    def update_error(self, row, message):
        item = self.item(row, self.COL_STATUS)
        if item:
            summary = _batch_error_summary(message)
            item.setText(_trf("status.error_detail", "Error: {message}", message=summary))
            item.setToolTip(str(message or "Unknown error"))

    def update_settings(self, row, settings):
        item = self.item(row, self.COL_SETTINGS)
        if item: item.setText(settings)

    def update_progress(self, row, pct):
        item = self.item(row, self.COL_PROGRESS)
        if item: item.setText(f"{pct}%")

    def update_output(self, row, path):
        item = self.item(row, self.COL_OUTPUT)
        if item: item.setText(os.path.basename(path))

    def clear_all(self):
        self.setRowCount(0)


class DragOutButton(QPushButton):
    """Button that initiates a file drag when the user drags from it."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._file_path = None
        self._drag_start = None

    def set_file(self, path):
        self._file_path = path
        self.setToolTip(f"Drag to NLE: {os.path.basename(path)}" if path else "")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if (self._drag_start is not None and self._file_path
                and os.path.isfile(self._file_path)
                and (e.position().toPoint() - self._drag_start).manhattanLength() > 10):
            drag = QDrag(self)
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(self._file_path)])
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.CopyAction)
            self._drag_start = None
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag_start = None
        super().mouseReleaseEvent(e)


class ThumbnailGrid(QScrollArea):
    """Visual grid of batch file thumbnails for quick preview scanning."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMaximumHeight(140)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet("QScrollArea { background: #0a0c10; border: 1px solid #1a1d28; border-radius: 8px; }")
        self._container = QWidget()
        self._layout = QGridLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)
        self.setWidget(self._container)
        self._cells = []
        self._cols = 6

    def set_files(self, filenames):
        for w in self._cells:
            self._layout.removeWidget(w)
            w.deleteLater()
        self._cells.clear()
        for i, name in enumerate(filenames):
            cell = QWidget()
            cell_lay = QVBoxLayout(cell)
            cell_lay.setContentsMargins(2, 2, 2, 2)
            cell_lay.setSpacing(2)
            thumb = QLabel()
            thumb.setFixedSize(80, 60)
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setStyleSheet("background: #13161d; border-radius: 4px;")
            thumb.setObjectName(f"thumb_{i}")
            cell_lay.addWidget(thumb)
            lbl = QLabel(os.path.basename(name))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #6e7a94; font-size: 9px;")
            lbl.setMaximumWidth(86)
            lbl.setWordWrap(False)
            cell_lay.addWidget(lbl)
            row, col = divmod(i, self._cols)
            self._layout.addWidget(cell, row, col)
            self._cells.append(cell)

    def set_thumbnail(self, index, pixmap):
        if index < len(self._cells):
            cell = self._cells[index]
            thumb = cell.findChild(QLabel, f"thumb_{index}")
            if thumb:
                scaled = pixmap.scaled(76, 56, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
                thumb.setPixmap(scaled)

    def clear_all(self):
        for w in self._cells:
            self._layout.removeWidget(w)
            w.deleteLater()
        self._cells.clear()


class NoScrollFilter(QObject):
    """Eats scroll-wheel events on combo boxes, sliders, and spin boxes
    so the left panel scrolls instead of accidentally changing values."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            if not obj.hasFocus():
                event.ignore()
                return True  # block the event from reaching the widget
        return super().eventFilter(obj, event)


def make_slider(label_text, min_val, max_val, default, suffix=""):
    w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)
    lbl = QLabel(label_text); lbl.setMinimumWidth(88); lbl.setWordWrap(True); lay.addWidget(lbl)
    slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(min_val, max_val); slider.setValue(default)
    lay.addWidget(slider, stretch=1)
    val_lbl = QLabel(f"{default}{suffix}"); val_lbl.setObjectName("sliderVal")
    val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lay.addWidget(val_lbl)
    slider.valueChanged.connect(lambda v: val_lbl.setText(f"{v}{suffix}"))
    return w, slider, val_lbl


# ═══════════════════════════════════════════════════════════════════════════════
# UPDATE CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
class UpdateChecker(QThread):
    result = pyqtSignal(dict)   # {available, tag, url, current, body}
    error = pyqtSignal(str)

    def run(self):
        try:
            req = urllib.request.Request(GITHUB_RELEASES_URL,
                headers={'User-Agent': f'AlphaCut/{__version__}', 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            tag = data.get('tag_name', '').lstrip('vV')
            url = data.get('html_url', '')
            body = data.get('body', '')[:500]
            # Simple version compare
            def ver_tuple(v):
                try: return tuple(int(x) for x in v.split('.'))
                except Exception: return (0,)
            available = ver_tuple(tag) > ver_tuple(__version__)
            self.result.emit({'available': available, 'tag': tag, 'url': url,
                              'current': __version__, 'body': body})
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT DIALOG
# ═══════════════════════════════════════════════════════════════════════════════
class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_trf("dialog.about.title", "About {app}", app=APP_NAME))
        self.setFixedSize(560, 520)
        lay = QVBoxLayout(self); lay.setSpacing(12)

        # Header
        hdr = QLabel(f"<h2>{APP_NAME}</h2>"); hdr.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(hdr)
        ver = QLabel(_trf("about.version", "Version {version}", version=APP_VERSION)); ver.setObjectName("subtitle")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(ver)

        # Description
        desc = QLabel(_tr("about.description", "AI-powered video background removal.\nDirect ONNX inference. Fully turnkey."))
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter); desc.setWordWrap(True); lay.addWidget(desc)

        # System info
        runtime_info = _format_runtime_diagnostics(include_notes=True)
        info_text = (f"Python {sys.version.split()[0]}\n"
                     f"{runtime_info}\n"
                     f"{_tr('about.platform', 'Platform')}: {sys.platform}\n"
                     f"{_tr('about.models', 'Models')}: {MODELS_DIR}")
        info = QLabel(info_text); info.setObjectName("subtitle"); info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignLeft); lay.addWidget(info)

        # Links
        links_row = QHBoxLayout()
        btn_github = QPushButton(_tr("btn.github", "GitHub")); btn_github.setObjectName("secondary")
        btn_github.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(f"https://github.com/{GITHUB_REPO}")))
        links_row.addWidget(btn_github)
        btn_issues = QPushButton(_tr("btn.report_bug", "Report Bug")); btn_issues.setObjectName("secondary")
        btn_issues.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(f"https://github.com/{GITHUB_REPO}/issues")))
        links_row.addWidget(btn_issues)
        lay.addLayout(links_row)

        # Credits
        credits = QLabel(_tr("about.credits", "MIT License | SysAdminDoc"))
        credits.setObjectName("subtitle"); credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(credits)
        lay.addStretch()

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(self.accept); lay.addWidget(btn_box)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL MANAGER DIALOG
# ═══════════════════════════════════════════════════════════════════════════════
class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_tr("dialog.model_manager.title", "Model Manager"))
        self.setMinimumSize(560, 420)
        lay = QVBoxLayout(self); lay.setSpacing(10)

        hdr = QLabel(f"<h3>{_tr('model_manager.heading', 'AI Models')}</h3>"); lay.addWidget(hdr)
        desc = QLabel(_tr("model_manager.description", "Manage downloaded ONNX models. Models are cached locally after first use."))
        desc.setWordWrap(True); desc.setObjectName("subtitle"); lay.addWidget(desc)

        # Model table
        self.table = QTableWidget(len(MODELS), 5); lay.addWidget(self.table)
        self.table.setHorizontalHeaderLabels([
            _tr("table.model", "Model"),
            _tr("table.input", "Input"),
            _tr("table.status", "Status"),
            _tr("table.size", "Size"),
            "",
        ])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        for i, (name, cfg) in enumerate(MODELS.items()):
            short = name.split('(')[0].strip()
            self.table.setItem(i, 0, QTableWidgetItem(short))
            self.table.setItem(i, 1, QTableWidgetItem(f"{cfg['size'][0]}px"))

            model_path = os.path.join(MODELS_DIR, os.path.basename(cfg['file']))
            if os.path.isfile(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                sidecar = model_path + '.sha256'
                verified = False
                if os.path.isfile(sidecar):
                    try:
                        with open(sidecar, 'r') as f:
                            verified = f.read().strip().lower() == cfg.get('sha256', '').lower()
                    except OSError:
                        verified = False
                status = _tr("status.model_verified", "Verified") if verified else _tr("status.needs_verification", "Needs verification")
                self.table.setItem(i, 2, QTableWidgetItem(status))
                self.table.setItem(i, 3, QTableWidgetItem(f"{size_mb:.1f} MB"))
                btn = QPushButton(_tr("btn.delete", "Delete")); btn.setObjectName("danger")
                btn.setFixedWidth(70)
                btn.clicked.connect(lambda _, r=i, f=os.path.basename(cfg['file']): self._delete_model(r, f))
                self.table.setCellWidget(i, 4, btn)
            else:
                self.table.setItem(i, 2, QTableWidgetItem(_tr("status.not_downloaded_hash_pinned", "Not downloaded - hash pinned")))
                self.table.setItem(i, 3, QTableWidgetItem("--"))
                self.table.setItem(i, 4, QTableWidgetItem(""))

        self.table.resizeColumnsToContents()

        # Total size
        total_mb = 0
        if os.path.isdir(MODELS_DIR):
            for f in os.listdir(MODELS_DIR):
                fp = os.path.join(MODELS_DIR, f)
                if os.path.isfile(fp): total_mb += os.path.getsize(fp) / (1024 * 1024)
        self.lbl_total = QLabel(_trf("model_manager.total_cached", "Total cached: {mb:.1f} MB", mb=total_mb))
        self.lbl_total.setObjectName("subtitle"); lay.addWidget(self.lbl_total)

        # Buttons
        btn_row = QHBoxLayout()
        btn_clear = QPushButton(_tr("btn.delete_all_models", "Delete All Models")); btn_clear.setObjectName("danger")
        btn_clear.clicked.connect(self._clear_all); btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        btn_close = QPushButton(_tr("btn.close", "Close")); btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close); lay.addLayout(btn_row)

    def _delete_model(self, row, filename):
        filename = os.path.basename(filename)
        path = os.path.join(MODELS_DIR, filename)
        try:
            if os.path.isfile(path): os.remove(path)
            sidecar = path + '.sha256'
            if os.path.isfile(sidecar): os.remove(sidecar)
            self.table.setItem(row, 2, QTableWidgetItem(_tr("status.not_downloaded_hash_pinned", "Not downloaded - hash pinned")))
            self.table.setItem(row, 3, QTableWidgetItem("--"))
            self.table.removeCellWidget(row, 4)
            self.table.setItem(row, 4, QTableWidgetItem(""))
            # Clear engine cache if this model was loaded
            if _engine_cache.get('engine') and hasattr(_engine_cache['engine'], 'config'):
                if _engine_cache['engine'].config.get('file') == filename:
                    _engine_cache['key'] = None; _engine_cache['engine'] = None
            self._update_total()
        except Exception as e:
            print(f"Delete error: {e}")

    def _clear_all(self):
        if os.path.isdir(MODELS_DIR):
            for f in os.listdir(MODELS_DIR):
                try: os.remove(os.path.join(MODELS_DIR, f))
                except Exception: pass
        _engine_cache['key'] = None; _engine_cache['engine'] = None
        for i in range(self.table.rowCount()):
            self.table.setItem(i, 2, QTableWidgetItem(_tr("status.not_downloaded_hash_pinned", "Not downloaded - hash pinned")))
            self.table.setItem(i, 3, QTableWidgetItem("--"))
            self.table.removeCellWidget(i, 4)
            self.table.setItem(i, 4, QTableWidgetItem(""))
        self._update_total()

    def _update_total(self):
        total_mb = 0
        if os.path.isdir(MODELS_DIR):
            for f in os.listdir(MODELS_DIR):
                fp = os.path.join(MODELS_DIR, f)
                if os.path.isfile(fp): total_mb += os.path.getsize(fp) / (1024 * 1024)
        self.lbl_total.setText(_trf("model_manager.total_cached", "Total cached: {mb:.1f} MB", mb=total_mb))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class AlphaCutWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(900, 650); self.resize(1180, 860)
        self.setWindowIcon(get_app_icon())
        self._worker = None; self._batch_worker = None; self._preview_worker = None; self._quick_preview_worker = None; self._model_dl_worker = None
        self._benchmark_worker = None
        self._input_path = None; self._video_info = None; self._is_image = False
        self._last_output = None; self._batch_jobs = []; self._batch_job_settings = {}
        self._bg_custom_color = None; self._bg_image_path = None
        self._update_checker = None
        self._chroma_result = None; self._chroma_detect_worker = None
        self._thumbnail_loader = None
        self._build_menu_bar(); self._build_ui(); self._setup_tray()
        self._check_ffmpeg(); self._load_settings()
        # Auto-check for updates on startup (silent)
        self._check_updates_silent()

    def _build_menu_bar(self):
        mb = self.menuBar()
        # File menu
        file_menu = mb.addMenu(_tr("menu.file", "File"))
        act_browse = QAction(_tr("menu.open_video", "Open..."), self)
        act_browse.setShortcut(QKeySequence.StandardKey.Open)
        act_browse.triggered.connect(self._browse); file_menu.addAction(act_browse)
        act_batch = QAction(_tr("menu.open_batch", "Open Batch..."), self)
        act_batch.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_batch.triggered.connect(self._browse_batch); file_menu.addAction(act_batch)
        file_menu.addSeparator()
        act_quit = QAction(_tr("menu.quit", "Quit"), self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close); file_menu.addAction(act_quit)
        # Tools menu
        tools_menu = mb.addMenu(_tr("menu.tools", "Tools"))
        act_models = QAction(_tr("menu.model_manager", "Model Manager..."), self)
        act_models.triggered.connect(self._show_model_manager); tools_menu.addAction(act_models)
        act_presets_dir = QAction(_tr("menu.open_settings", "Open Settings Folder"), self)
        act_presets_dir.triggered.connect(lambda: reveal_in_explorer(APP_DIR))
        tools_menu.addAction(act_presets_dir)
        # Help menu
        help_menu = mb.addMenu(_tr("menu.help", "Help"))
        act_update = QAction(_tr("menu.check_updates", "Check for Updates..."), self)
        act_update.triggered.connect(self._check_updates_manual); help_menu.addAction(act_update)
        help_menu.addSeparator()
        act_github = QAction(_tr("menu.github", "GitHub Repository"), self)
        act_github.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(f"https://github.com/{GITHUB_REPO}")))
        help_menu.addAction(act_github)
        act_about = QAction(_tr("menu.about", "About AlphaCut..."), self)
        act_about.triggered.connect(self._show_about); help_menu.addAction(act_about)

    def _setup_tray(self):
        self._tray = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = QSystemTrayIcon(get_app_icon(), self)
            menu = QMenu()
            show = QAction(_tr("menu.show", "Show"), self); show.triggered.connect(self._tray_show); menu.addAction(show)
            quit_a = QAction("Quit", self); quit_a.triggered.connect(self.close); menu.addAction(quit_a)
            self._tray.setContextMenu(menu)
            self._tray.activated.connect(lambda r: self._tray_show() if r == QSystemTrayIcon.ActivationReason.DoubleClick else None)

    def _tray_show(self): self.showNormal(); self.activateWindow()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(10,10,10,10); root.setSpacing(12)

        # ── LEFT PANEL ──
        left = QWidget(); left.setMinimumWidth(340); left.setMaximumWidth(420)
        ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0); ll.setSpacing(4)

        # Input
        grp_in = QGroupBox(_tr("group.input", "INPUT")); gl = QVBoxLayout(grp_in); gl.setSpacing(4)
        gl.setContentsMargins(10, 16, 10, 8)
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._load_video)
        self.drop_zone.files_dropped.connect(self._load_batch)
        gl.addWidget(self.drop_zone)

        btn_row = QHBoxLayout(); btn_row.setSpacing(4)
        self.btn_browse = QPushButton(_tr("btn.browse", "Browse")); self.btn_browse.setObjectName("secondary"); self.btn_browse.clicked.connect(self._browse)
        self.btn_browse.setAccessibleName(_tr("access.browse", "Browse for input file"))
        btn_row.addWidget(self.btn_browse)
        self.btn_batch = QPushButton(_tr("btn.browse_batch", "Browse Batch")); self.btn_batch.setObjectName("secondary"); self.btn_batch.clicked.connect(self._browse_batch)
        self.btn_batch.setAccessibleName(_tr("access.browse_batch", "Browse for batch input files"))
        btn_row.addWidget(self.btn_batch)
        self.btn_recent = QPushButton(_tr("btn.recent", "Recent")); self.btn_recent.setObjectName("secondary")
        self.btn_recent.setAccessibleName(_tr("access.recent", "Open recent files menu"))
        self.btn_recent.clicked.connect(self._show_recent); btn_row.addWidget(self.btn_recent)
        gl.addLayout(btn_row)

        self.info_w = QWidget(); ig = QGridLayout(self.info_w); ig.setContentsMargins(0,2,0,0); ig.setSpacing(4)
        self.s_res = self._stat("--", _tr("stat.resolution", "Resolution")); self.s_fps = self._stat("--", _tr("stat.fps", "FPS"))
        self.s_dur = self._stat("--", _tr("stat.duration", "Duration")); self.s_frm = self._stat("--", _tr("stat.frames", "Frames"))
        ig.addWidget(self.s_res, 0, 0); ig.addWidget(self.s_fps, 0, 1)
        ig.addWidget(self.s_dur, 1, 0); ig.addWidget(self.s_frm, 1, 1)
        self.info_w.setVisible(False); gl.addWidget(self.info_w); ll.addWidget(grp_in)

        # Settings
        grp_set = QGroupBox(_tr("group.settings", "SETTINGS")); sl = QVBoxLayout(grp_set); sl.setSpacing(2)
        sl.setContentsMargins(10, 16, 10, 8)
        lbl_smart = QLabel(_tr("label.processing_type", "What are you processing?")); lbl_smart.setObjectName("accent"); sl.addWidget(lbl_smart)
        self.combo_smart = QComboBox()
        self.combo_smart.addItem(_tr("combo.processing_placeholder", "-- Choose preset or pick model below --"))
        for p in SMART_PRESETS:
            self.combo_smart.addItem(_tr(f"preset.{SMART_PRESET_LOCALE_KEYS[p]}", p))
        self.combo_smart.currentIndexChanged.connect(self._smart_pick); sl.addWidget(self.combo_smart)
        sl.addWidget(QLabel(_tr("label.ai_model", "AI Model")))
        self.combo_model = QComboBox()
        for name in MODELS: self.combo_model.addItem(name)
        self.combo_model.currentIndexChanged.connect(self._update_res_suggestion)
        self.combo_model.currentIndexChanged.connect(self._refresh_batch_settings_display)
        sl.addWidget(self.combo_model)
        lbl2 = QLabel(_tr("label.output_format", "Output Format")); lbl2.setObjectName("accent"); sl.addWidget(lbl2)
        self.combo_fmt = QComboBox()
        self._format_items = available_output_formats()
        for name, fmt_id in self._format_items.items():
            self.combo_fmt.addItem(name, fmt_id)
        self.combo_fmt.currentIndexChanged.connect(self._format_changed); sl.addWidget(self.combo_fmt)
        self.lbl_estimate = QLabel(""); self.lbl_estimate.setObjectName("subtitle"); sl.addWidget(self.lbl_estimate)
        w_qual, self.sl_quality, self.lbl_quality = make_slider(_tr("label.quality", "Quality"), 0, 100, 70, "%"); sl.addWidget(w_qual)
        row = QHBoxLayout(); row.addWidget(QLabel(_tr("label.max_resolution", "Max Resolution")))
        self.spin_res = QSpinBox(); self.spin_res.setRange(0, 7680); self.spin_res.setSingleStep(240)
        self.spin_res.setValue(0); self.spin_res.setSpecialValueText("Original"); self.spin_res.setSuffix("px")
        row.addWidget(self.spin_res); sl.addLayout(row)
        gpu_row = QHBoxLayout(); gpu_row.addWidget(QLabel(_tr("label.gpu_device", "GPU Device")))
        self.spin_gpu = QSpinBox(); self.spin_gpu.setRange(-1, 16); self.spin_gpu.setValue(-1)
        self.spin_gpu.setSpecialValueText("Auto")
        gpu_row.addWidget(self.spin_gpu); sl.addLayout(gpu_row)
        self.chk_fp16 = QCheckBox(_tr("chk.fp16", "FP16 inference (GPU speedup)"))
        self.chk_fp16.setToolTip(_tr("tip.fp16", "Use half-precision floating point on GPU — ~2x faster, minimal quality loss"))
        sl.addWidget(self.chk_fp16)
        self.chk_audio = QCheckBox(_tr("chk.keep_audio", "Keep original audio")); self.chk_audio.setChecked(True); sl.addWidget(self.chk_audio)

        # Naming pattern
        name_row = QHBoxLayout(); name_row.addWidget(QLabel(_tr("label.naming", "Naming")))
        self.combo_naming = QComboBox()
        for pat in NAMING_PATTERNS: self.combo_naming.addItem(pat)
        name_row.addWidget(self.combo_naming, stretch=1); sl.addLayout(name_row)

        # Presets
        preset_row = QHBoxLayout()
        self.combo_presets = QComboBox(); self.combo_presets.addItem(_tr("combo.presets_placeholder", "-- Presets --"))
        self._refresh_presets(); self.combo_presets.currentIndexChanged.connect(self._load_preset)
        preset_row.addWidget(self.combo_presets, stretch=1)
        self.btn_save_preset = QPushButton(_tr("btn.save", "Save")); self.btn_save_preset.setObjectName("small")
        self.btn_save_preset.setAccessibleName(_tr("access.save_preset", "Save current settings as preset"))
        self.btn_save_preset.clicked.connect(self._save_preset); preset_row.addWidget(self.btn_save_preset)
        sl.addLayout(preset_row)
        ll.addWidget(grp_set)

        # Refinement
        grp_ref = QGroupBox(_tr("group.refinement", "REFINEMENT")); rl2 = QVBoxLayout(grp_ref); rl2.setSpacing(2)
        rl2.setContentsMargins(10, 16, 10, 8)
        w1, self.sl_edge, _ = make_slider(_tr("label.edge_softness", "Edge Softness"), 0, 100, 0); rl2.addWidget(w1)
        w2, self.sl_shift, _ = make_slider(_tr("label.mask_shift", "Mask Shift"), -20, 20, 0); rl2.addWidget(w2)
        w3, self.sl_temporal, _ = make_slider(_tr("label.temporal_smooth", "Temporal Smooth"), 0, 7, 0, "f"); rl2.addWidget(w3)
        w4, self.sl_frame_skip, _ = make_slider(_tr("label.frame_skip", "Frame Skip"), 1, 10, 1, "x"); rl2.addWidget(w4)
        ll.addWidget(grp_ref)

        # Compositing
        grp_comp = QGroupBox(_tr("group.compositing", "COMPOSITING")); cl = QVBoxLayout(grp_comp); cl.setSpacing(2)
        cl.setContentsMargins(10, 16, 10, 8)

        self.chk_invert = QCheckBox(_tr("chk.invert_mask", "Invert mask (remove subject)")); cl.addWidget(self.chk_invert)

        w5, self.sl_spill, _ = make_slider(_tr("label.spill_suppress", "Spill Suppress"), 0, 100, 0, "%"); cl.addWidget(w5)
        spill_row = QHBoxLayout(); spill_row.addWidget(QLabel(_tr("label.spill_color", "Spill Color")))
        self.combo_spill = QComboBox()
        for sc in ['green', 'blue', 'red']:
            self.combo_spill.addItem(_tr(f"color.{sc}", sc.capitalize()), sc)
        spill_row.addWidget(self.combo_spill); cl.addLayout(spill_row)

        w6, self.sl_shadow, _ = make_slider(_tr("label.shadow_preserve", "Shadow Preserve"), 0, 100, 0, "%"); cl.addWidget(w6)

        bg_row = QHBoxLayout(); bg_row.addWidget(QLabel(_tr("label.background", "Background")))
        self.combo_bg = QComboBox()
        for name in BG_COLORS:
            self.combo_bg.addItem(_tr(f"bg.{BG_COLOR_LOCALE_KEYS[name]}", name))
        self.combo_bg.currentIndexChanged.connect(self._bg_changed)
        bg_row.addWidget(self.combo_bg, stretch=1); cl.addLayout(bg_row)

        self.lbl_bg_info = QLabel(""); self.lbl_bg_info.setObjectName("subtitle"); cl.addWidget(self.lbl_bg_info)

        # Chroma-key detection hint + checkbox (hidden until detection runs)
        self.lbl_chroma_hint = QLabel(""); self.lbl_chroma_hint.setObjectName("subtitle")
        self.lbl_chroma_hint.setWordWrap(True); self.lbl_chroma_hint.setVisible(False); cl.addWidget(self.lbl_chroma_hint)
        self.chk_use_chroma = QCheckBox(_tr("chk.use_chroma", "Use chroma-key (faster, better edges)"))
        self.chk_use_chroma.setVisible(False); cl.addWidget(self.chk_use_chroma)

        ll.addWidget(grp_comp)

        # Resolution suggestion
        self.lbl_res_suggest = QLabel("")
        self.lbl_res_suggest.setObjectName("subtitle")
        self.lbl_res_suggest.setWordWrap(True)
        ll.addWidget(self.lbl_res_suggest)

        # Actions
        act_row = QHBoxLayout(); act_row.setSpacing(4)
        self.btn_preview = QPushButton(_tr("btn.preview", "Preview (Ctrl+P)")); self.btn_preview.setObjectName("secondary")
        self.btn_preview.setAccessibleName(_tr("access.preview_frame", "Preview current frame"))
        self.btn_preview.setEnabled(False); self.btn_preview.setMinimumHeight(30)
        self.btn_preview.setShortcut(QKeySequence("Ctrl+P"))
        self.btn_preview.clicked.connect(self._preview_frame); act_row.addWidget(self.btn_preview)
        self.btn_quick_preview = QPushButton(_tr("btn.preview_10s", "Preview 10s")); self.btn_quick_preview.setObjectName("secondary")
        self.btn_quick_preview.setAccessibleName(_tr("access.preview_clip", "Render ten second preview clip"))
        self.btn_quick_preview.setEnabled(False); self.btn_quick_preview.setMinimumHeight(30)
        self.btn_quick_preview.clicked.connect(self._preview_clip); act_row.addWidget(self.btn_quick_preview)
        self.btn_benchmark = QPushButton(_tr("btn.benchmark", "Benchmark")); self.btn_benchmark.setObjectName("secondary")
        self.btn_benchmark.setAccessibleName(_tr("access.benchmark", "Benchmark processing speed"))
        self.btn_benchmark.setEnabled(False); self.btn_benchmark.setMinimumHeight(30)
        self.btn_benchmark.clicked.connect(self._run_benchmark); act_row.addWidget(self.btn_benchmark)
        ll.addLayout(act_row)
        self.btn_start = QPushButton(_tr("btn.start", "Start Processing (Ctrl+Enter)")); self.btn_start.setEnabled(False)
        self.btn_start.setAccessibleName(_tr("access.start_processing", "Start processing"))
        self.btn_start.setMinimumHeight(36); self.btn_start.setStyleSheet("font-size: 13px;")
        self.btn_start.setShortcut(QKeySequence("Ctrl+Return"))
        self.btn_start.clicked.connect(self._start); ll.addWidget(self.btn_start)
        self.btn_cancel = QPushButton(_tr("btn.cancel", "Cancel")); self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setAccessibleName(_tr("access.cancel_processing", "Cancel processing"))
        self.btn_cancel.setShortcut(QKeySequence("Escape"))
        self.btn_cancel.setVisible(False); self.btn_cancel.clicked.connect(self._cancel); ll.addWidget(self.btn_cancel)

        # Output actions
        out_row = QHBoxLayout()
        self.btn_copy_path = QPushButton(_tr("btn.copy_path", "Copy Path"))
        self.btn_copy_path.setObjectName("small"); self.btn_copy_path.setVisible(False)
        self.btn_copy_path.setAccessibleName(_tr("access.copy_output_path", "Copy output path"))
        self.btn_copy_path.clicked.connect(self._copy_path); out_row.addWidget(self.btn_copy_path)
        self.btn_open_folder = QPushButton(_tr("btn.open_folder", "Open Folder")); self.btn_open_folder.setObjectName("small")
        self.btn_open_folder.setVisible(False)
        self.btn_open_folder.setAccessibleName(_tr("access.open_output_folder", "Open output folder"))
        self.btn_open_folder.clicked.connect(self._open_folder); out_row.addWidget(self.btn_open_folder)
        self.btn_drag_out = DragOutButton(_tr("btn.drag_out", "Drag to NLE"))
        self.btn_drag_out.setObjectName("small"); self.btn_drag_out.setVisible(False)
        self.btn_drag_out.setAccessibleName(_tr("access.drag_output", "Drag output file to another application"))
        out_row.addWidget(self.btn_drag_out)
        ll.addLayout(out_row)
        ll.addStretch()

        # ── RIGHT PANEL ──
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(10)

        grp_prev = QGroupBox(_tr("group.preview", "PREVIEW")); pl = QVBoxLayout(grp_prev)
        self.preview = SplitPreviewWidget()
        self.preview.edits_changed.connect(self._sync_edit_summary)
        pl.addWidget(self.preview)
        view_row = QHBoxLayout(); view_row.addWidget(QLabel(_tr("label.view", "View")))
        self.combo_preview_view = QComboBox()
        self.combo_preview_view.addItem(_tr("preview_mode.compare", "Compare"), "compare")
        self.combo_preview_view.addItem(_tr("preview_mode.original", "Original"), "original")
        self.combo_preview_view.addItem(_tr("preview_mode.result", "Result"), "result")
        self.combo_preview_view.addItem(_tr("preview_mode.mask_bw", "Mask B/W"), "mask_bw")
        self.combo_preview_view.addItem(_tr("preview_mode.mask_gray", "Mask Gray"), "mask_gray")
        self.combo_preview_view.addItem(_tr("preview_mode.mask_overlay", "Mask Overlay"), "mask_overlay")
        self.combo_preview_view.currentIndexChanged.connect(self._preview_view_changed)
        view_row.addWidget(self.combo_preview_view, stretch=1)
        pl.addLayout(view_row)
        scrub_row = QHBoxLayout(); scrub_row.addWidget(QLabel(_tr("label.position", "Position")))
        self.sl_scrub = QSlider(Qt.Orientation.Horizontal); self.sl_scrub.setRange(0, 100); self.sl_scrub.setValue(10)
        self.sl_scrub.setEnabled(False); scrub_row.addWidget(self.sl_scrub, stretch=1)
        self.lbl_scrub = QLabel("10%"); self.lbl_scrub.setObjectName("sliderVal"); scrub_row.addWidget(self.lbl_scrub)
        self.sl_scrub.valueChanged.connect(lambda v: self.lbl_scrub.setText(f"{v}%"))
        pl.addLayout(scrub_row)
        edit_row = QHBoxLayout(); edit_row.addWidget(QLabel(_tr("label.mask_tool", "Mask Tool")))
        self.combo_mask_tool = QComboBox()
        self.combo_mask_tool.addItem(_tr("mask_tool.compare", "Compare"), "compare")
        self.combo_mask_tool.addItem(_tr("mask_tool.roi", "ROI"), "roi")
        self.combo_mask_tool.addItem(_tr("mask_tool.paint_keep", "Paint Keep"), "fg")
        self.combo_mask_tool.addItem(_tr("mask_tool.paint_remove", "Paint Remove"), "bg")
        self.combo_mask_tool.currentIndexChanged.connect(self._preview_tool_changed)
        edit_row.addWidget(self.combo_mask_tool, stretch=1)
        self.spin_brush = QSpinBox(); self.spin_brush.setRange(4, 256); self.spin_brush.setValue(32); self.spin_brush.setSuffix("px")
        self.spin_brush.valueChanged.connect(self._brush_size_changed)
        edit_row.addWidget(self.spin_brush)
        self.btn_undo_stroke = QPushButton(_tr("btn.undo", "Undo")); self.btn_undo_stroke.setObjectName("small")
        self.btn_undo_stroke.setAccessibleName(_tr("access.undo_mask_edit", "Undo last mask edit"))
        self.btn_undo_stroke.setShortcut(QKeySequence.StandardKey.Undo)
        self.btn_undo_stroke.clicked.connect(self._undo_stroke)
        edit_row.addWidget(self.btn_undo_stroke)
        self.btn_clear_edits = QPushButton(_tr("btn.clear", "Clear")); self.btn_clear_edits.setObjectName("small")
        self.btn_clear_edits.setAccessibleName(_tr("access.clear_mask_edits", "Clear mask edits"))
        self.btn_clear_edits.clicked.connect(self._clear_mask_edits)
        edit_row.addWidget(self.btn_clear_edits)
        pl.addLayout(edit_row)
        self.lbl_edit_summary = QLabel(_tr("mask_edits.none", "No mask edits")); self.lbl_edit_summary.setObjectName("subtitle")
        pl.addWidget(self.lbl_edit_summary)
        self.lbl_mask_quality = QLabel(_tr("mask_quality.placeholder", "Mask quality appears after preview"))
        self.lbl_mask_quality.setObjectName("subtitle")
        self.lbl_mask_quality.setWordWrap(True)
        self.lbl_mask_quality.setAccessibleName(_tr("access.mask_quality", "Mask quality summary"))
        pl.addWidget(self.lbl_mask_quality)
        self._preview_tool_changed(0)
        rl.addWidget(grp_prev, stretch=3)

        # Batch table
        grp_batch = QGroupBox(_tr("group.batch", "BATCH QUEUE")); bl = QVBoxLayout(grp_batch)
        self.thumb_grid = ThumbnailGrid(); bl.addWidget(self.thumb_grid)
        self.job_table = JobTable(); bl.addWidget(self.job_table)
        self.job_table.customContextMenuRequested.connect(self._job_context_menu)
        self.grp_batch = grp_batch; grp_batch.setVisible(False); rl.addWidget(grp_batch)

        grp_prog = QGroupBox(_tr("group.progress", "PROGRESS")); prl = QVBoxLayout(grp_prog); prl.setSpacing(6)
        status_row = QHBoxLayout(); status_row.setSpacing(8)
        self.lbl_status_cue = QLabel("")
        self.lbl_status = StatusLabel(_tr("status.ready", "Ready"))
        self.lbl_status.bind_cue_label(self.lbl_status_cue)
        status_row.addWidget(self.lbl_status_cue)
        status_row.addWidget(self.lbl_status, stretch=1)
        prl.addLayout(status_row)
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%"); prl.addWidget(self.progress_bar)
        prog_row = QHBoxLayout()
        self.lbl_frame = QLabel(""); self.lbl_frame.setObjectName("subtitle"); prog_row.addWidget(self.lbl_frame)
        self.lbl_memory = QLabel(""); self.lbl_memory.setObjectName("subtitle"); prog_row.addWidget(self.lbl_memory)
        prl.addLayout(prog_row)
        rl.addWidget(grp_prog)

        grp_log = QGroupBox(_tr("group.log", "LOG")); lgl = QVBoxLayout(grp_log)
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True); self.log_view.setMaximumHeight(130)
        lgl.addWidget(self.log_view); rl.addWidget(grp_log, stretch=1)

        # Accessibility — name interactive widgets for screen readers
        self.combo_smart.setAccessibleName("Processing preset")
        self.combo_model.setAccessibleName("AI model")
        self.combo_fmt.setAccessibleName("Output format")
        self.sl_quality.setAccessibleName("Output quality")
        self.spin_res.setAccessibleName("Max resolution")
        self.spin_gpu.setAccessibleName("GPU device")
        self.chk_fp16.setAccessibleName("FP16 half-precision inference")
        self.sl_edge.setAccessibleName("Edge softness")
        self.sl_shift.setAccessibleName("Mask shift")
        self.sl_temporal.setAccessibleName("Temporal smoothing")
        self.sl_frame_skip.setAccessibleName("Frame skip")
        self.sl_spill.setAccessibleName("Spill suppression")
        self.combo_spill.setAccessibleName("Spill color")
        self.sl_shadow.setAccessibleName("Shadow preservation")
        self.combo_bg.setAccessibleName("Background")
        self.combo_naming.setAccessibleName("Output naming pattern")
        self.combo_presets.setAccessibleName("Preset")
        self.btn_quick_preview.setAccessibleName("Preview ten second sample")
        self.sl_scrub.setAccessibleName("Preview position")
        self.combo_preview_view.setAccessibleName("Preview view mode")
        self.combo_mask_tool.setAccessibleName("Mask editing tool")
        self.spin_brush.setAccessibleName("Brush size")
        self.preview.setAccessibleName("Preview display")
        self.chk_audio.setAccessibleName("Keep original audio")
        self.chk_invert.setAccessibleName("Invert mask")
        self.chk_use_chroma.setAccessibleName("Use chroma-key")
        self.lbl_edit_summary.setAccessibleName("Mask edit summary")
        self.lbl_mask_quality.setAccessibleName("Mask quality summary")
        self.progress_bar.setAccessibleName("Processing progress")
        self.log_view.setAccessibleName("Processing log")
        self.job_table.setAccessibleName("Batch job queue")
        self.thumb_grid.setAccessibleName("Batch thumbnail grid")

        # Prevent scroll-wheel from changing settings — requires click-to-focus
        self._no_scroll = NoScrollFilter(self)
        for widget in left.findChildren((QComboBox, QSlider, QSpinBox)):
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            widget.installEventFilter(self._no_scroll)

        left_scroll = QScrollArea()
        left_scroll.setObjectName("leftPanelScroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(360)
        left_scroll.setMaximumWidth(440)
        left_scroll.setWidget(left)
        self.left_scroll = left_scroll
        root.addWidget(left_scroll); root.addWidget(right, stretch=1)
        self._configure_focus_order()
        self._toast = ToastWidget(central)
        self._glow_timer = QTimer(self); self._glow_timer.setInterval(80); self._glow_phase = 0.0
        self._glow_timer.timeout.connect(self._animate_progress)

    def _configure_focus_order(self):
        order = [
            self.drop_zone, self.btn_browse, self.btn_batch, self.btn_recent,
            self.combo_smart, self.combo_model, self.combo_fmt, self.sl_quality,
            self.spin_res, self.spin_gpu, self.chk_fp16, self.chk_audio,
            self.combo_naming, self.combo_presets, self.btn_save_preset,
            self.sl_edge, self.sl_shift, self.sl_temporal, self.sl_frame_skip,
            self.chk_invert, self.sl_spill, self.combo_spill, self.sl_shadow,
            self.combo_bg, self.chk_use_chroma, self.btn_preview,
            self.btn_quick_preview, self.btn_benchmark, self.btn_start,
            self.btn_cancel, self.combo_preview_view, self.sl_scrub,
            self.combo_mask_tool, self.spin_brush, self.btn_undo_stroke,
            self.btn_clear_edits, self.job_table, self.log_view,
        ]
        for first, second in zip(order, order[1:]):
            self.setTabOrder(first, second)

    # ── Helpers ──
    def _stat(self, val, label):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(6,4,6,4); lay.setSpacing(1)
        v = QLabel(val); v.setObjectName("statValue"); v.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(v)
        l = QLabel(label); l.setObjectName("statLabel"); l.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(l)
        w.setStyleSheet("background: #0f1118; border-radius: 8px;"); w._val = v; return w

    def _log(self, msg): self.log_view.append(msg)
    def _toast_msg(self, text, duration=3000): self._toast.show_toast(text, duration)

    def _smart_pick(self, index):
        if index <= 0: return
        name = list(SMART_PRESETS.keys())[index - 1]
        label = _tr(f"preset.{SMART_PRESET_LOCALE_KEYS[name]}", name)
        self.combo_model.setCurrentIndex(SMART_PRESETS[name]); self._log(_trf("log.preset", "Preset: {name}", name=label))

    def _animate_progress(self):
        self._glow_phase += 0.12; pulse = 0.5 + 0.5 * math.sin(self._glow_phase)
        r1, g1, b1 = int(108+30*pulse), int(92+20*pulse), int(231+24*pulse)
        r2, g2, b2 = int(168+30*pulse), int(85+20*pulse), int(247+8*pulse)
        self.progress_bar.setStyleSheet(f"QProgressBar {{ background-color: #13161d; border: 1px solid #1e2230; border-radius: 6px; text-align: center; color: #c8ccd4; font-size: 11px; font-weight: 600; min-height: 22px; }} QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 rgb({r1},{g1},{b1}), stop:1 rgb({r2},{g2},{b2})); border-radius: 5px; }}")

    def _start_glow(self): self._glow_phase = 0.0; self._glow_timer.start()
    def _stop_glow(self): self._glow_timer.stop(); self.progress_bar.setStyleSheet("")

    def _update_tray_progress(self, pct):
        if self._tray and self._tray.isVisible(): self._tray.setToolTip(_trf("tray.progress", "{app} — {pct}%", app=APP_NAME, pct=pct))

    def _update_memory_display(self, ram_pct):
        color = '#ef4444' if ram_pct > 85 else '#f59e0b' if ram_pct > 70 else '#4a5168'
        label = "RAM HIGH" if ram_pct > 85 else "RAM WARN" if ram_pct > 70 else "RAM"
        self.lbl_memory.setText(f"{label}: {ram_pct:.0f}%")
        self.lbl_memory.setStyleSheet(f"color: {color};")
        self.lbl_memory.setAccessibleDescription(self.lbl_memory.text())

    def _update_res_suggestion(self):
        if not self._video_info: self.lbl_res_suggest.setText(""); return
        model_key = list(MODELS.keys())[self.combo_model.currentIndex()]
        suggested, msg = suggest_resolution(self._video_info, model_key)
        self.lbl_res_suggest.setText(msg)

    # ── Benchmark ──
    def _run_benchmark(self):
        if not self._input_path or not find_ffmpeg(): return
        model_key = list(MODELS.keys())[self.combo_model.currentIndex()]
        comp = self._get_compositing_params()
        roi, mask_edits = self._mask_controls()
        self.btn_benchmark.setEnabled(False); self.btn_benchmark.setText(_tr("status.running", "Running..."))
        self.lbl_status.setText(_tr("status.benchmarking", "Benchmarking..."))
        self._benchmark_worker = BenchmarkWorker(
            self._input_path, model_key, self.spin_res.value(),
            edge_softness=self.sl_edge.value(), mask_shift=self.sl_shift.value(),
            invert_mask=comp['invert_mask'], spill_strength=comp['spill_strength'],
            spill_color=comp['spill_color'], shadow_strength=comp['shadow_strength'],
            bg_color=comp['bg_color'], bg_image_path=comp['bg_image_path'],
            roi=roi, mask_edits=mask_edits, gpu_device=self.spin_gpu.value(),
            fp16=self.chk_fp16.isChecked())
        self._benchmark_worker.status.connect(lambda s: self.lbl_status.setText(s))
        self._benchmark_worker.result.connect(self._benchmark_done)
        self._benchmark_worker.error.connect(self._benchmark_err)
        self._benchmark_worker.start()

    def _benchmark_done(self, data):
        self.btn_benchmark.setEnabled(True); self.btn_benchmark.setText(_tr("btn.benchmark", "Benchmark"))
        fps = data['fps']; eta = data['eta_str']; n = data['frames_tested']; total = data['total_frames']
        self.lbl_status.setText(_trf("status.benchmark_result", "Benchmark: {fps:.1f} fps | ETA {eta} for {total} frames", fps=fps, eta=eta, total=total))
        self._log(_trf("log.benchmark_result", "\nBenchmark ({count} frames): {fps:.2f} fps | Estimated total: {eta}", count=n, fps=fps, eta=eta))
        self._log(_trf("log.inference_time", "  Inference: {seconds:.1f}s for {count} frames", seconds=data['elapsed'], count=n))
        if data['ram_pct'] > 0:
            self._log(_trf("log.ram", "  RAM: {ram:.0f}%", ram=data['ram_pct']))
            self._update_memory_display(data['ram_pct'])
        self._toast_msg(_trf("toast.benchmark_result", "Benchmark: {fps:.1f} fps | ETA {eta}", fps=fps, eta=eta))

    def _benchmark_err(self, msg):
        self.btn_benchmark.setEnabled(True); self.btn_benchmark.setText(_tr("btn.benchmark", "Benchmark"))
        self._log(_trf("log.benchmark_error", "Benchmark error: {message}", message=msg)); self._toast_msg(_tr("toast.benchmark_failed", "Benchmark failed"))

    # ── Dialogs ──
    def _show_about(self):
        dlg = AboutDialog(self); dlg.exec()

    def _show_model_manager(self):
        dlg = ModelManagerDialog(self); dlg.exec()

    def _check_updates_silent(self):
        """Check for updates on startup without bothering the user unless available."""
        self._update_checker = UpdateChecker()
        self._update_checker.result.connect(self._update_result_silent)
        self._update_checker.start()

    def _check_updates_manual(self):
        """User-initiated update check with feedback."""
        self._update_checker = UpdateChecker()
        self._update_checker.result.connect(self._update_result_manual)
        self._update_checker.error.connect(lambda e: self._toast_msg(f"Update check failed: {e}"))
        self.lbl_status.setText(_tr("status.checking_updates", "Checking for updates..."))
        self._update_checker.start()

    def _update_result_silent(self, data):
        if data.get('available'):
            self._toast_msg(f"Update available: v{data['tag']}", 5000)
            self._log(f"Update available: v{data['tag']} (current: v{data['current']})")

    def _update_result_manual(self, data):
        if data.get('available'):
            self._toast_msg(f"Update available: v{data['tag']}", 5000)
            self._log(f"\nUpdate available: v{data['tag']}")
            if data.get('body'):
                self._log(f"Release notes:\n{data['body']}")
            self.lbl_status.setText(_trf("status.update_available", "Update v{tag} available", tag=data['tag']))
            url = data.get('url', '')
            if url.startswith(f"https://github.com/{GITHUB_REPO}"):
                QDesktopServices.openUrl(QUrl(url))
            else:
                QDesktopServices.openUrl(QUrl(f"https://github.com/{GITHUB_REPO}/releases"))
        else:
            self._toast_msg(f"You're up to date (v{data['current']})")
            self.lbl_status.setText(_tr("status.up_to_date", "Ready — up to date"))

    def _check_ffmpeg(self):
        ff = find_ffmpeg()
        if ff: self._log(f"FFmpeg: {ff}")
        else: self._log(_tr("log.ffmpeg_not_found", "FFmpeg not found!")); self.lbl_status.setText(_tr("status.ffmpeg_required", "FFmpeg required"))

    def _bg_changed(self, index):
        key = list(BG_COLORS.keys())[index]
        val = BG_COLORS[key]
        if val == "custom":
            color = QColorDialog.getColor(QColor(0, 128, 0), self, _tr("dialog.pick_background", "Pick Background Color"))
            if color.isValid():
                self._bg_custom_color = (color.red(), color.green(), color.blue())
                self.lbl_bg_info.setText(_trf("label.custom_color", "Custom: {color}", color=color.name()))
            else:
                self.combo_bg.setCurrentIndex(0)
                self.lbl_bg_info.setText("")
        elif val == "image":
            exts = " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)
            path, _ = QFileDialog.getOpenFileName(self, _tr("dialog.select_background_image", "Select Background Image"), "",
                                                   f"Images ({exts});;All (*)")
            if path and os.path.isfile(path):
                self._bg_image_path = path
                self.lbl_bg_info.setText(os.path.basename(path))
            else:
                self.combo_bg.setCurrentIndex(0)
                self.lbl_bg_info.setText("")
        else:
            self._bg_custom_color = None
            self._bg_image_path = None
            self.lbl_bg_info.setText("")

    def _get_compositing_params(self):
        """Get current compositing settings as dict."""
        bg_key = list(BG_COLORS.keys())[self.combo_bg.currentIndex()]
        bg_val = BG_COLORS[bg_key]
        bg_color = None; bg_image_path = None
        if bg_val == "custom":
            bg_color = self._bg_custom_color
        elif bg_val == "image":
            bg_image_path = self._bg_image_path
        elif isinstance(bg_val, tuple):
            bg_color = bg_val
        return {
            'invert_mask': self.chk_invert.isChecked(),
            'spill_strength': self.sl_spill.value(),
            'spill_color': self.combo_spill.currentData() or 'green',
            'shadow_strength': self.sl_shadow.value(),
            'bg_color': bg_color,
            'bg_image_path': bg_image_path,
        }

    # ── Presets ──
    def _refresh_presets(self):
        self.combo_presets.clear(); self.combo_presets.addItem(_tr("combo.presets_placeholder", "-- Presets --"))
        for name in load_presets(): self.combo_presets.addItem(name)

    def _save_preset(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, _tr("dialog.save_preset", "Save Preset"), _tr("label.preset_name", "Preset name:"))
        if not ok or not name.strip(): return
        presets = load_presets()
        presets[name.strip()] = {
            'model_index': self.combo_model.currentIndex(), 'format_index': self.combo_fmt.currentIndex(),
            'max_res': self.spin_res.value(), 'edge_softness': self.sl_edge.value(),
            'mask_shift': self.sl_shift.value(), 'temporal_smooth': self.sl_temporal.value(),
            'keep_audio': self.chk_audio.isChecked(), 'naming_index': self.combo_naming.currentIndex(),
            'frame_skip': self.sl_frame_skip.value(), 'gpu_device': self.spin_gpu.value(),
            'fp16': self.chk_fp16.isChecked(),
            'invert_mask': self.chk_invert.isChecked(), 'spill_strength': self.sl_spill.value(),
            'spill_color_index': self.combo_spill.currentIndex(),
            'shadow_strength': self.sl_shadow.value(), 'bg_index': self.combo_bg.currentIndex(),
            'quality': self.sl_quality.value(),
        }
        save_presets(presets); self._refresh_presets(); self._toast_msg(_trf("toast.preset_saved", "Preset saved: {name}", name=name.strip()))

    def _load_preset(self, index):
        if index <= 0: return
        name = self.combo_presets.itemText(index); presets = load_presets()
        p = presets.get(name, {})
        if not p: return
        if 'model_index' in p: self.combo_model.setCurrentIndex(min(p['model_index'], self.combo_model.count()-1))
        if 'format_index' in p: self.combo_fmt.setCurrentIndex(min(p['format_index'], self.combo_fmt.count()-1))
        if 'max_res' in p: self.spin_res.setValue(p['max_res'])
        if 'edge_softness' in p: self.sl_edge.setValue(p['edge_softness'])
        if 'mask_shift' in p: self.sl_shift.setValue(p['mask_shift'])
        if 'temporal_smooth' in p: self.sl_temporal.setValue(p['temporal_smooth'])
        if 'keep_audio' in p: self.chk_audio.setChecked(p['keep_audio'])
        if 'naming_index' in p: self.combo_naming.setCurrentIndex(min(p['naming_index'], self.combo_naming.count()-1))
        if 'frame_skip' in p: self.sl_frame_skip.setValue(p['frame_skip'])
        if 'gpu_device' in p: self.spin_gpu.setValue(max(self.spin_gpu.minimum(), min(int(p['gpu_device']), self.spin_gpu.maximum())))
        if 'fp16' in p: self.chk_fp16.setChecked(p['fp16'])
        if 'invert_mask' in p: self.chk_invert.setChecked(p['invert_mask'])
        if 'spill_strength' in p: self.sl_spill.setValue(p['spill_strength'])
        if 'spill_color_index' in p: self.combo_spill.setCurrentIndex(min(p['spill_color_index'], self.combo_spill.count()-1))
        if 'shadow_strength' in p: self.sl_shadow.setValue(p['shadow_strength'])
        if 'bg_index' in p: self.combo_bg.setCurrentIndex(min(p['bg_index'], self.combo_bg.count()-1))
        if 'quality' in p: self.sl_quality.setValue(p['quality'])
        self._toast_msg(_trf("toast.preset_loaded", "Loaded: {name}", name=name))

    # ── Settings ──
    def _load_settings(self):
        s = load_settings()
        if 'model_index' in s: self.combo_model.setCurrentIndex(min(s['model_index'], self.combo_model.count()-1))
        if 'format_index' in s: self.combo_fmt.setCurrentIndex(min(s['format_index'], self.combo_fmt.count()-1))
        if 'max_res' in s: self.spin_res.setValue(s['max_res'])
        if 'edge_softness' in s: self.sl_edge.setValue(s['edge_softness'])
        if 'mask_shift' in s: self.sl_shift.setValue(s['mask_shift'])
        if 'temporal_smooth' in s: self.sl_temporal.setValue(s['temporal_smooth'])
        if 'keep_audio' in s: self.chk_audio.setChecked(s['keep_audio'])
        if 'naming_index' in s: self.combo_naming.setCurrentIndex(min(s['naming_index'], self.combo_naming.count()-1))
        if 'frame_skip' in s: self.sl_frame_skip.setValue(s['frame_skip'])
        if 'gpu_device' in s:
            try:
                gpu_device = int(s['gpu_device'])
                self.spin_gpu.setValue(max(self.spin_gpu.minimum(), min(gpu_device, self.spin_gpu.maximum())))
            except (TypeError, ValueError):
                pass
        if 'fp16' in s: self.chk_fp16.setChecked(s['fp16'])
        if 'invert_mask' in s: self.chk_invert.setChecked(s['invert_mask'])
        if 'spill_strength' in s: self.sl_spill.setValue(s['spill_strength'])
        if 'spill_color_index' in s: self.combo_spill.setCurrentIndex(min(s['spill_color_index'], self.combo_spill.count()-1))
        if 'shadow_strength' in s: self.sl_shadow.setValue(s['shadow_strength'])
        if 'bg_index' in s: self.combo_bg.setCurrentIndex(min(s['bg_index'], self.combo_bg.count()-1))
        if 'quality' in s: self.sl_quality.setValue(s['quality'])
        if 'brush_size' in s:
            try:
                brush_size = int(s['brush_size'])
                self.spin_brush.setValue(max(self.spin_brush.minimum(), min(brush_size, self.spin_brush.maximum())))
            except (TypeError, ValueError):
                pass

    def _save_settings(self):
        s = load_settings()
        s.update({'model_index': self.combo_model.currentIndex(), 'format_index': self.combo_fmt.currentIndex(),
            'max_res': self.spin_res.value(), 'edge_softness': self.sl_edge.value(),
            'mask_shift': self.sl_shift.value(), 'temporal_smooth': self.sl_temporal.value(),
            'keep_audio': self.chk_audio.isChecked(), 'naming_index': self.combo_naming.currentIndex(),
            'frame_skip': self.sl_frame_skip.value(), 'gpu_device': self.spin_gpu.value(),
            'fp16': self.chk_fp16.isChecked(),
            'invert_mask': self.chk_invert.isChecked(), 'spill_strength': self.sl_spill.value(),
            'spill_color_index': self.combo_spill.currentIndex(),
            'shadow_strength': self.sl_shadow.value(), 'bg_index': self.combo_bg.currentIndex(),
            'quality': self.sl_quality.value(),
            'brush_size': self.spin_brush.value()})
        save_settings(s)

    # ── Input ──
    def _browse(self):
        v_exts = " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)
        i_exts = " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, _tr("dialog.select_media", "Select Video or Image"), "",
                                               f"Media ({v_exts} {i_exts});;Video ({v_exts});;Images ({i_exts});;All (*)")
        if path: self._load_video(path)

    def _browse_batch(self):
        v_exts = " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)
        i_exts = " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(self, _tr("dialog.select_files", "Select Files"), "",
                                                 f"Media ({v_exts} {i_exts});;All (*)")
        if paths:
            if len(paths) == 1: self._load_video(paths[0])
            else: self._load_batch(paths)

    def _show_recent(self):
        recent = get_recent_files()
        if not recent: self._toast_msg(_tr("toast.no_recent_files", "No recent files")); return
        menu = QMenu(self)
        for p in recent[:15]:
            if os.path.isfile(p):
                action = menu.addAction(os.path.basename(p))
                action.setToolTip(p)
                action.triggered.connect(lambda checked, path=p: self._load_video(path))
        menu.exec(self.btn_recent.mapToGlobal(self.btn_recent.rect().bottomLeft()))

    def _load_video(self, path):
        if not os.path.isfile(path): return
        self._input_path = path; self._batch_jobs = []
        self._batch_job_settings = {}
        ext = os.path.splitext(path)[1].lower()
        self._is_image = ext in IMAGE_EXTENSIONS
        self.grp_batch.setVisible(False); self.job_table.clear_all(); self.thumb_grid.clear_all()
        self.preview.clear_mask_edits()
        add_recent_file(path)
        self._log(f"\nLoaded: {os.path.basename(path)}")
        self.lbl_status.setText(f"Loaded: {os.path.basename(path)}")
        self.drop_zone.setText(os.path.basename(path))
        self.progress_bar.setValue(0); self.lbl_frame.setText("")
        self.btn_copy_path.setVisible(False); self.btn_open_folder.setVisible(False); self.btn_drag_out.setVisible(False)
        # Reset chroma state
        self._chroma_result = None
        self.lbl_chroma_hint.setVisible(False); self.chk_use_chroma.setVisible(False)
        self.chk_use_chroma.setChecked(False)
        if self._chroma_detect_worker:
            self._chroma_detect_worker.cancel()
            self._chroma_detect_worker.wait()
            self._chroma_detect_worker = None

        if self._is_image:
            try:
                im = Image.open(path); w, h = im.size; im.close()
                self.s_res._val.setText(f"{w}x{h}")
                self.s_fps._val.setText("—"); self.s_dur._val.setText(_tr("media.image", "image"))
                self.s_frm._val.setText("1"); self.info_w.setVisible(True)
                self._video_info = None
            except Exception:
                self.info_w.setVisible(False); self._video_info = None
            self.lbl_estimate.setText(_tr("label.png_output", "→ PNG"))
            self.chk_audio.setEnabled(False); self.chk_audio.setChecked(False)
            self.sl_scrub.setEnabled(False)
            self.btn_start.setEnabled(True); self.btn_preview.setEnabled(True)
            self.btn_quick_preview.setEnabled(False)
            self.btn_benchmark.setEnabled(False)
            self._update_res_suggestion()
            return

        info = get_video_info(path); self._video_info = info
        if info:
            self.s_res._val.setText(f"{info['width']}x{info['height']}")
            self.s_fps._val.setText(f"{info['fps']}"); self.s_dur._val.setText(f"{info['duration']:.1f}s")
            self.s_frm._val.setText(str(info['total_frames'])); self.info_w.setVisible(True)
            self._update_estimate()
            self.chk_audio.setEnabled(info.get('has_audio', False))
            if not info.get('has_audio'): self.chk_audio.setChecked(False)
            self._chroma_detect_worker = ChromaDetectWorker(path)
            self._chroma_detect_worker.result.connect(self._chroma_detected)
            self._chroma_detect_worker.start()
        else: self.info_w.setVisible(False)
        self.btn_start.setEnabled(True); self.btn_preview.setEnabled(True); self.btn_quick_preview.setEnabled(True)
        self.btn_benchmark.setEnabled(True); self.sl_scrub.setEnabled(True)
        self._update_res_suggestion()

    def _short_model_name(self, model_key):
        return model_key.split('(')[0].strip()

    def _format_label(self, fmt):
        for i in range(self.combo_fmt.count()):
            if self.combo_fmt.itemData(i) == fmt:
                return self.combo_fmt.itemText(i).split('(')[0].strip()
        return fmt

    def _batch_settings_text(self, row):
        if row < 0 or row >= len(self._batch_jobs):
            return ""
        settings = self._batch_job_settings.get(row, {})
        model_key = settings.get('model_key') or list(MODELS.keys())[self.combo_model.currentIndex()]
        path = self._batch_jobs[row]
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            fmt_label = "PNG"
        else:
            fmt_label = self._format_label(settings.get('format') or self._current_format())
        return f"{self._short_model_name(model_key)} / {fmt_label}"

    def _refresh_batch_settings_display(self, *args):
        if not hasattr(self, 'job_table'):
            return
        for row in range(min(self.job_table.rowCount(), len(self._batch_jobs))):
            self.job_table.update_settings(row, self._batch_settings_text(row))

    def _job_context_menu(self, pos):
        row = self.job_table.rowAt(pos.y())
        if row < 0 or row >= len(self._batch_jobs):
            return
        menu = QMenu(self)
        edit_action = menu.addAction(_tr("menu.edit_job_settings", "Edit Job Settings..."))
        reset_action = menu.addAction(_tr("menu.use_global_settings", "Use Global Settings"))
        action = menu.exec(self.job_table.viewport().mapToGlobal(pos))
        if action == edit_action:
            self._edit_batch_job_settings(row)
        elif action == reset_action:
            self._batch_job_settings.pop(row, None)
            self.job_table.update_settings(row, self._batch_settings_text(row))
            self._toast_msg(_tr("toast.job_settings_reset", "Job settings reset"))

    def _edit_batch_job_settings(self, row):
        if row < 0 or row >= len(self._batch_jobs):
            return
        path = self._batch_jobs[row]
        is_image = os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
        current = self._batch_job_settings.get(row, {})
        dlg = QDialog(self)
        dlg.setWindowTitle(_tr("dialog.job_settings.title", "Job Settings"))
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(os.path.basename(path)))
        lay.addWidget(QLabel(_tr("label.ai_model", "AI Model")))
        model_combo = QComboBox()
        for name in MODELS:
            model_combo.addItem(name, name)
        model_value = current.get('model_key') or list(MODELS.keys())[self.combo_model.currentIndex()]
        model_combo.setCurrentIndex(max(0, model_combo.findData(model_value)))
        lay.addWidget(model_combo)
        lay.addWidget(QLabel(_tr("label.output_format", "Output Format")))
        fmt_combo = QComboBox()
        for i in range(self.combo_fmt.count()):
            fmt_combo.addItem(self.combo_fmt.itemText(i), self.combo_fmt.itemData(i))
        fmt_value = current.get('format') or self._current_format()
        fmt_combo.setCurrentIndex(max(0, fmt_combo.findData(fmt_value)))
        fmt_combo.setEnabled(not is_image)
        if is_image:
            fmt_combo.setToolTip(_tr("tip.image_batch_png", "Image batch jobs export transparent PNG files."))
        lay.addWidget(fmt_combo)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._batch_job_settings[row] = {
                'model_key': model_combo.currentData(),
                'format': fmt_combo.currentData(),
            }
            self.job_table.update_settings(row, self._batch_settings_text(row))
            self._toast_msg(_tr("toast.job_settings_updated", "Job settings updated"))

    def _load_batch(self, paths):
        self._batch_jobs = [
            p for p in paths
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in (VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)
        ]
        if not self._batch_jobs: return
        self._batch_job_settings = {}
        self._input_path = self._batch_jobs[0]
        self._is_image = os.path.splitext(self._input_path)[1].lower() in IMAGE_EXTENSIONS
        self.preview.clear_mask_edits()
        # Batch-update recent files in a single read-modify-write
        s = load_settings()
        recent = s.get('recent_files', [])
        for p in self._batch_jobs:
            if p in recent: recent.remove(p)
            recent.insert(0, p)
        s['recent_files'] = recent[:20]
        save_settings(s)
        self.grp_batch.setVisible(True); self.job_table.clear_all()
        self.thumb_grid.set_files(self._batch_jobs)
        for i, p in enumerate(self._batch_jobs):
            self.job_table.add_job(os.path.basename(p), self._batch_settings_text(i))
        video_count = sum(1 for p in self._batch_jobs if os.path.splitext(p)[1].lower() in VIDEO_EXTENSIONS)
        image_count = len(self._batch_jobs) - video_count
        batch_label = f"{len(self._batch_jobs)} files queued ({video_count} video, {image_count} image)"
        self.drop_zone.setText(batch_label)
        self._log(f"\nBatch loaded: {batch_label}")
        self.lbl_status.setText(f"Batch: {batch_label}")
        self.info_w.setVisible(False)
        self.btn_start.setEnabled(True); self.btn_start.setText(f"Start Batch ({len(self._batch_jobs)})")
        self.btn_preview.setEnabled(True); self.btn_quick_preview.setEnabled(video_count > 0)
        self.btn_benchmark.setEnabled(video_count > 0); self.sl_scrub.setEnabled(video_count > 0)
        # Start thumbnail loader
        if self._thumbnail_loader:
            self._thumbnail_loader.cancel()
            self._thumbnail_loader.wait()
        jobs = [(i, p) for i, p in enumerate(self._batch_jobs)]
        self._thumbnail_loader = ThumbnailLoader(jobs)
        self._thumbnail_loader.thumbnail_ready.connect(
            lambda row, pixmap: (self.job_table.set_thumbnail(row, pixmap), self.thumb_grid.set_thumbnail(row, pixmap)))
        self._thumbnail_loader.start()

    def _update_estimate(self):
        if not self._video_info: return
        fmt = self._current_format()
        est = estimate_output_size(self._video_info, fmt)
        self.lbl_estimate.setText(f"~{est/1024:.1f} GB" if est > 1024 else f"~{est:.0f} MB" if est > 0 else "")

    def _current_format(self):
        return self.combo_fmt.currentData() or 'mp4'

    def _format_changed(self, index):
        self._update_estimate()
        self._refresh_batch_settings_display()

    def _preview_view_changed(self, index):
        self.preview.set_view_mode(self.combo_preview_view.currentData() or 'compare')

    def _preview_tool_changed(self, index):
        mode = self.combo_mask_tool.currentData() or 'compare'
        self.preview.set_edit_mode(mode)
        self.spin_brush.setEnabled(mode in ('fg', 'bg'))

    def _brush_size_changed(self, value):
        self.preview.set_brush_size(value)

    def _undo_stroke(self):
        if self.preview.undo_last_stroke():
            self._sync_edit_summary()
            self._toast_msg("Stroke undone")

    def _clear_mask_edits(self):
        self.preview.clear_mask_edits()
        self._sync_edit_summary()
        self._toast_msg("Mask edits cleared")

    def _sync_edit_summary(self):
        self.lbl_edit_summary.setText(self.preview.edit_summary())

    def _mask_controls(self):
        return self.preview.get_roi(), self.preview.get_mask_edits()

    # ── Preview ──
    def _preview_frame(self):
        if not self._input_path: return
        if os.path.splitext(self._input_path)[1].lower() in VIDEO_EXTENSIONS and not find_ffmpeg(): return
        model_key = list(MODELS.keys())[self.combo_model.currentIndex()]
        comp = self._get_compositing_params()
        roi, mask_edits = self._mask_controls()
        self.btn_preview.setEnabled(False); self.btn_preview.setText(_tr("status.previewing", "Previewing..."))
        self._preview_worker = PreviewFrameWorker(
            self._input_path, model_key, self.spin_res.value(),
            edge_softness=self.sl_edge.value(), mask_shift=self.sl_shift.value(),
            seek_pct=self.sl_scrub.value() / 100.0,
            invert_mask=comp['invert_mask'], spill_strength=comp['spill_strength'],
            spill_color=comp['spill_color'], shadow_strength=comp['shadow_strength'],
            bg_color=comp['bg_color'], bg_image_path=comp['bg_image_path'],
            roi=roi, mask_edits=mask_edits, gpu_device=self.spin_gpu.value(),
            fp16=self.chk_fp16.isChecked())
        self._preview_worker.status.connect(lambda s: self.lbl_status.setText(s))
        self._preview_worker.result.connect(self._preview_done)
        self._preview_worker.error.connect(self._preview_err)
        self._preview_worker.start()

    def _preview_done(self, orig, proc, mask_bw, mask_gray, mask_overlay, mask_metrics):
        self.preview.set_images(orig, proc, mask_bw, mask_gray, mask_overlay)
        self._update_mask_quality(mask_metrics)
        self.btn_preview.setEnabled(True); self.btn_preview.setText(_tr("btn.preview", "Preview (Ctrl+P)"))
        self.lbl_status.setText(_tr("status.preview_ready", "Preview ready — drag divider to compare"))
        self._toast_msg(_tr("toast.preview_ready", "Preview ready — drag to compare"))

    def _preview_err(self, msg):
        self.btn_preview.setEnabled(True); self.btn_preview.setText(_tr("btn.preview", "Preview (Ctrl+P)"))
        self.lbl_mask_quality.setText(_tr("mask_quality.placeholder", "Mask quality appears after preview"))
        self._log(f"Preview error: {msg}"); self._toast_msg("Preview failed")

    def _update_mask_quality(self, metrics):
        summary = format_mask_quality_summary(metrics or {})
        warnings = mask_quality_warning_texts(metrics or {})
        if warnings:
            warning_detail = _trf("mask_quality.warning_prefix", "WARN: {message}", message="; ".join(warnings))
            text = summary + "\n" + _trf("mask_quality.warning_count", "WARN: {count} issue(s) - see log", count=len(warnings))
            tooltip = summary + "\n" + warning_detail
        else:
            text = summary + "\n" + _tr("mask_quality.ok", "OK: no obvious mask quality warnings.")
            tooltip = text
        self.lbl_mask_quality.setText(text)
        self.lbl_mask_quality.setToolTip(tooltip)
        self.lbl_mask_quality.setAccessibleDescription(tooltip)
        self._log(_trf("log.mask_quality", "\nMask quality: {summary}", summary=summary))
        for warning in warnings:
            self._log(_trf("log.mask_quality_warning", "Mask quality warning: {message}", message=warning))
        if warnings:
            self._toast_msg(_tr("toast.mask_quality_warning", "Mask quality warnings - see preview"), 5000)

    def _preview_clip(self):
        preview_path = self._input_path
        if self._batch_jobs:
            preview_path = next((p for p in self._batch_jobs if os.path.splitext(p)[1].lower() in VIDEO_EXTENSIONS), self._input_path)
        if not preview_path or os.path.splitext(preview_path)[1].lower() not in VIDEO_EXTENSIONS:
            return
        if not find_ffmpeg():
            self._toast_msg("FFmpeg required for preview clip")
            return
        model_key = list(MODELS.keys())[self.combo_model.currentIndex()]
        fmt = self._current_format()
        preview_fmt = 'mp4' if fmt in ('png_seq', 'fg_alpha') else fmt
        comp = self._get_compositing_params()
        roi, mask_edits = self._mask_controls()
        preview_dir = tempfile.mkdtemp(prefix='alphacut_preview_clip_')
        out = os.path.join(preview_dir, f"preview{format_extension(preview_fmt)}")
        self.btn_quick_preview.setEnabled(False); self.btn_quick_preview.setText(_tr("status.rendering", "Rendering..."))
        self.progress_bar.setValue(0); self.lbl_status.setText(_tr("status.rendering_preview", "Rendering 10s preview..."))
        self._quick_preview_worker = QuickPreviewWorker(
            preview_path, out, model_key, preview_fmt, self.spin_res.value(),
            edge_softness=self.sl_edge.value(), mask_shift=self.sl_shift.value(),
            temporal_smooth=self.sl_temporal.value(), keep_audio=False,
            frame_skip=self.sl_frame_skip.value(), invert_mask=comp['invert_mask'],
            spill_strength=comp['spill_strength'], spill_color=comp['spill_color'],
            shadow_strength=comp['shadow_strength'], bg_color=comp['bg_color'],
            bg_image_path=comp['bg_image_path'], quality=self.sl_quality.value(),
            roi=roi, mask_edits=mask_edits, gpu_device=self.spin_gpu.value(),
            fp16=self.chk_fp16.isChecked(), duration=10)
        self._quick_preview_worker.progress.connect(self.progress_bar.setValue)
        self._quick_preview_worker.status.connect(self.lbl_status.setText)
        self._quick_preview_worker.log.connect(self._log)
        self._quick_preview_worker.preview.connect(self.preview.set_frame)
        self._quick_preview_worker.finished.connect(self._preview_clip_done)
        self._quick_preview_worker.error.connect(self._preview_clip_err)
        self._quick_preview_worker.start()

    def _preview_clip_done(self, path):
        self.btn_quick_preview.setEnabled(True); self.btn_quick_preview.setText(_tr("btn.preview_10s", "Preview 10s"))
        self.progress_bar.setValue(100)
        self.lbl_status.setText(_trf("status.preview_clip_ready", "Preview clip ready: {file}", file=os.path.basename(path)))
        self._toast_msg(_tr("toast.preview_clip_ready", "Preview clip ready"), 5000)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _preview_clip_err(self, msg):
        self.btn_quick_preview.setEnabled(True); self.btn_quick_preview.setText(_tr("btn.preview_10s", "Preview 10s"))
        self._log(f"Preview clip error: {msg}")
        self.lbl_status.setText(_tr("status.preview_clip_failed", "Preview clip failed"))
        self._toast_msg(_tr("toast.preview_clip_failed", "Preview clip failed"))

    def _chroma_detected(self, result):
        """Handle chroma-key detection result."""
        self._chroma_result = result
        if result:
            color = result.get('color', 'green').capitalize()
            hint = f"Detected {color}-screen background. Chroma-key is faster & better for synthetic backgrounds."
            self.lbl_chroma_hint.setText(hint)
            self.lbl_chroma_hint.setVisible(True)
            self.chk_use_chroma.setVisible(True)
        else:
            self.lbl_chroma_hint.setVisible(False)
            self.chk_use_chroma.setVisible(False)

    # ── Processing ──
    def _start(self):
        if not self._input_path: return
        needs_ffmpeg = (
            (self._batch_jobs and any(os.path.splitext(p)[1].lower() in VIDEO_EXTENSIONS for p in self._batch_jobs))
            or (not self._batch_jobs and not self._is_image)
        )
        if needs_ffmpeg and not find_ffmpeg(): return
        self._save_settings()
        model_key = list(MODELS.keys())[self.combo_model.currentIndex()]
        fmt = self._current_format()
        pattern = self.combo_naming.currentText()

        if _model_needs_download(model_key):
            self._download_model_then_start(model_key, fmt, pattern)
            return

        self._dispatch_start(model_key, fmt, pattern)

    def _download_model_then_start(self, model_key, fmt, pattern):
        cfg = MODELS.get(model_key, {})
        dlg = QProgressDialog(f"Downloading {cfg.get('file', 'model')}...", "Cancel", 0, 100, self)
        dlg.setWindowTitle("Model Download")
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(True)
        dlg.setAutoReset(False)
        worker = ModelDownloadWorker(model_key, gpu_device=self.spin_gpu.value(), parent=self)
        def _on_progress(dl, tot):
            if tot > 0:
                dlg.setValue(dl * 100 // tot)
                dlg.setLabelText(f"Downloading {cfg.get('file', 'model')}... "
                                 f"{dl/(1024*1024):.1f}/{tot/(1024*1024):.1f} MB")
            else:
                dlg.setLabelText(f"Downloading... {dl/(1024*1024):.1f} MB")
        def _on_finished(path):
            dlg.close()
            self._log(f"Model downloaded: {os.path.basename(path)}")
            self._dispatch_start(model_key, fmt, pattern)
        def _on_error(msg):
            dlg.close()
            self._log(f"Download error: {msg}")
            self._toast_msg("Model download failed — see log")
        worker.progress.connect(_on_progress)
        worker.finished.connect(_on_finished)
        worker.error.connect(_on_error)
        dlg.canceled.connect(worker.cancel)
        self._model_dl_worker = worker
        worker.start()

    def _dispatch_start(self, model_key, fmt, pattern):
        if self._batch_jobs:
            self._start_batch(model_key, fmt, pattern)
        elif self._is_image:
            self._start_image(model_key)
        else:
            self._start_single(model_key, fmt, pattern)

    def _start_single(self, model_key, fmt, pattern):
        if fmt == 'png_seq':
            out = QFileDialog.getExistingDirectory(self, _tr("dialog.select_output_folder", "Select Output Folder"))
            if not out: return
        else:
            default = generate_output_name(self._input_path, pattern, model_key, fmt)
            ext = format_extension(fmt)
            out, _ = QFileDialog.getSaveFileName(self, _tr("dialog.save_output", "Save Output"), default, f"*{ext};;All (*)")
            if not out: return
        # Disk space check
        if self._video_info:
            est_mb = estimate_output_size(self._video_info, fmt)
            try:
                drive = os.path.splitdrive(out)[0] or '/'
                free_mb = shutil.disk_usage(drive).free / (1024 * 1024)
                if free_mb < est_mb * 2.5:
                    self._log(f"WARNING: Low disk space! Need ~{est_mb*2.5:.0f} MB, have {free_mb:.0f} MB")
            except Exception: pass
        self._begin_processing()
        roi, mask_edits = self._mask_controls()
        
        # Check if chroma-key should be used instead of AI
        if (self.chk_use_chroma.isVisible() and self.chk_use_chroma.isChecked() 
                and self._chroma_result and not roi and not mask_edits):
            self._worker = ChromaKeyWorker(
                self._input_path, out, fmt,
                self._chroma_result['color'],
                self._chroma_result['similarity'],
                self._chroma_result['blend'],
                quality=self.sl_quality.value(),
                keep_audio=self.chk_audio.isChecked())
            self._worker.progress.connect(self.progress_bar.setValue)
            self._worker.progress.connect(self._update_tray_progress)
            self._worker.status.connect(self.lbl_status.setText)
            self._worker.log.connect(self._log)
            self._worker.finished.connect(self._done)
            self._worker.error.connect(self._err)
            self._worker.start()
            return
        if self.chk_use_chroma.isVisible() and self.chk_use_chroma.isChecked() and (roi or mask_edits):
            self._log("ROI/mask edits active; using AI pipeline instead of chroma-key.")
        
        comp = self._get_compositing_params()
        self._worker = ProcessingWorker(
            self._input_path, out, model_key, fmt, self.spin_res.value(),
            edge_softness=self.sl_edge.value(), mask_shift=self.sl_shift.value(),
            temporal_smooth=self.sl_temporal.value(), keep_audio=self.chk_audio.isChecked(),
            frame_skip=self.sl_frame_skip.value(),
            invert_mask=comp['invert_mask'], spill_strength=comp['spill_strength'],
            spill_color=comp['spill_color'], shadow_strength=comp['shadow_strength'],
            bg_color=comp['bg_color'], bg_image_path=comp['bg_image_path'],
            quality=self.sl_quality.value(), roi=roi, mask_edits=mask_edits,
            gpu_device=self.spin_gpu.value(), fp16=self.chk_fp16.isChecked())
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.progress.connect(self._update_tray_progress)
        self._worker.frame_info.connect(lambda c, t: self.lbl_frame.setText(f"Frame {c} / {t}"))
        self._worker.status.connect(self.lbl_status.setText)
        self._worker.log.connect(self._log)
        self._worker.preview.connect(self.preview.set_frame)
        self._worker.memory_update.connect(self._update_memory_display)
        self._worker.finished.connect(self._done)
        self._worker.error.connect(self._err)
        self._worker.start()

    def _start_image(self, model_key):
        base = os.path.splitext(os.path.basename(self._input_path))[0]
        default = os.path.join(os.path.dirname(self._input_path), f"{base}_alphacut.png")
        out, _ = QFileDialog.getSaveFileName(self, _tr("dialog.save_output", "Save Output"), default, "PNG (*.png);;All (*)")
        if not out: return
        self._begin_processing()
        comp = self._get_compositing_params()
        roi, mask_edits = self._mask_controls()
        self._worker = ImageProcessWorker(
            self._input_path, out, model_key, self.spin_res.value(),
            edge_softness=self.sl_edge.value(), mask_shift=self.sl_shift.value(),
            invert_mask=comp['invert_mask'], spill_strength=comp['spill_strength'],
            spill_color=comp['spill_color'], shadow_strength=comp['shadow_strength'],
            bg_color=comp['bg_color'], bg_image_path=comp['bg_image_path'],
            roi=roi, mask_edits=mask_edits, gpu_device=self.spin_gpu.value(),
            fp16=self.chk_fp16.isChecked())
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.progress.connect(self._update_tray_progress)
        self._worker.status.connect(self.lbl_status.setText)
        self._worker.log.connect(self._log)
        self._worker.preview.connect(self.preview.set_frame)
        self._worker.finished.connect(self._done)
        self._worker.error.connect(self._err)
        self._worker.start()

    def _start_batch(self, model_key, fmt, pattern):
        out_dir = QFileDialog.getExistingDirectory(self, _tr("dialog.select_output_folder", "Select Output Folder"))
        if not out_dir: return
        comp = self._get_compositing_params()
        roi, mask_edits = self._mask_controls()
        jobs = []
        for row, path in enumerate(self._batch_jobs):
            overrides = self._batch_job_settings.get(row, {})
            job_model = overrides.get('model_key') or model_key
            is_image = os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
            job_fmt = 'png' if is_image else (overrides.get('format') or fmt)
            out_name = generate_output_name(path, pattern, job_model, job_fmt)
            out_path = os.path.join(out_dir, os.path.basename(out_name))
            jobs.append({'input': path, 'output': out_path, 'model_key': job_model, 'format': job_fmt,
                         'type': 'image' if is_image else 'video',
                         'max_res': self.spin_res.value(), 'edge': self.sl_edge.value(),
                         'shift': self.sl_shift.value(), 'temporal': self.sl_temporal.value(),
                         'audio': self.chk_audio.isChecked(), 'frame_skip': self.sl_frame_skip.value(),
                         'quality': self.sl_quality.value(),
                         'roi': roi, 'mask_edits': mask_edits,
                         'gpu_device': self.spin_gpu.value(), 'fp16': self.chk_fp16.isChecked(),
                         **comp})
        self._begin_processing()
        self._batch_start_time = time.time()
        self._batch_total = len(jobs)
        self._batch_worker = BatchWorker(jobs)
        self._batch_worker.job_started.connect(lambda i, n: self.job_table.update_status(i, "Processing..."))
        self._batch_worker.job_progress.connect(self.job_table.update_progress)
        self._batch_worker.job_progress.connect(lambda i, p: self.progress_bar.setValue(p))
        self._batch_worker.job_progress.connect(lambda i, p: self._update_tray_progress(p))
        self._batch_worker.job_status.connect(self._batch_status_with_eta)
        self._batch_worker.job_finished.connect(lambda i, p: (self.job_table.update_status(i, "Done"), self.job_table.update_output(i, p)))
        self._batch_worker.job_error.connect(self._batch_job_error)
        self._batch_worker.log.connect(self._log)
        self._batch_worker.preview.connect(self.preview.set_frame)
        self._batch_worker.all_done.connect(self._batch_done)
        self._batch_worker.start()

    def _batch_job_error(self, idx, message):
        self.job_table.update_error(idx, message)
        self._log(f"\nBatch job {idx + 1} error:\n{message}")

    def _begin_processing(self):
        self.btn_start.setEnabled(False); self.btn_preview.setEnabled(False); self.btn_quick_preview.setEnabled(False)
        self.btn_benchmark.setEnabled(False); self.btn_cancel.setVisible(True)
        self.progress_bar.setValue(0); self._start_glow()
        self.lbl_status.setText(_tr("status.starting", "Starting...")); self.lbl_frame.setText("")
        self.btn_copy_path.setVisible(False); self.btn_open_folder.setVisible(False); self.btn_drag_out.setVisible(False)
        if self._tray: self._tray.show(); self._tray.setToolTip(_trf("tray.processing", "{app} — Processing...", app=APP_NAME))

    def _cancel(self):
        for w in [self._worker, self._batch_worker, self._preview_worker, self._quick_preview_worker, self._chroma_detect_worker, self._thumbnail_loader, self._model_dl_worker]:
            if w and w.isRunning(): w.cancel(); w.quit(); w.wait(5000)
        self._reset(); self.lbl_status.setText(_tr("status.cancelled", "Cancelled")); self._toast_msg(_tr("toast.cancelled", "Cancelled"))

    def _done(self, path):
        self._reset(); self._last_output = path
        self.lbl_status.setText(_trf("status.done_file", "Done: {file}", file=os.path.basename(path)))
        self._log(_trf("log.output", "\nOutput: {path}", path=path)); self._toast_msg(_trf("toast.done_file", "Done: {file}", file=os.path.basename(path)), 5000)
        self.btn_copy_path.setVisible(True); self.btn_open_folder.setVisible(True); self.btn_drag_out.setVisible(True); self.btn_drag_out.set_file(self._last_output)
        if self._tray and self._tray.isVisible():
            self._tray.showMessage(APP_NAME, _trf("toast.done_file", "Done: {file}", file=os.path.basename(path)), QSystemTrayIcon.MessageIcon.Information, 5000)

    def _batch_status_with_eta(self, idx, status_text):
        elapsed = time.time() - self._batch_start_time
        if idx > 0 and elapsed > 0:
            avg_per_job = elapsed / idx
            remaining = (self._batch_total - idx) * avg_per_job
            eta = _ftime(remaining)
            self.lbl_status.setText(f"[{idx+1}/{self._batch_total}] {status_text} | ETA {eta}")
        else:
            self.lbl_status.setText(f"[{idx+1}/{self._batch_total}] {status_text}")

    def _batch_done(self, completed, total):
        self._reset()
        self.lbl_status.setText(f"Batch complete: {completed}/{total}")
        self._toast_msg(_trf("toast.batch_done", "Batch done: {completed}/{total} files", completed=completed, total=total), 5000)
        self.btn_start.setText(_tr("btn.start", "Start Processing (Ctrl+Enter)"))
        if self._tray and self._tray.isVisible():
            self._tray.showMessage(APP_NAME, _trf("toast.batch_complete", "Batch complete: {completed}/{total}", completed=completed, total=total), QSystemTrayIcon.MessageIcon.Information, 5000)

    def _err(self, msg):
        self._reset(); self.lbl_status.setText(_tr("status.error", "Error"))
        self._log(f"\nERROR:\n{msg}"); self._toast_msg("Failed — see log", 5000)

    def _reset(self):
        has_video = bool(self._batch_jobs and any(os.path.splitext(p)[1].lower() in VIDEO_EXTENSIONS for p in self._batch_jobs))
        self.btn_start.setEnabled(True); self.btn_preview.setEnabled(True)
        self.btn_quick_preview.setEnabled((not self._batch_jobs and not self._is_image) or has_video)
        self.btn_benchmark.setEnabled((not self._batch_jobs and not self._is_image) or has_video)
        self.btn_cancel.setVisible(False); self._stop_glow()
        if self._tray: self._tray.hide()
        if self._batch_jobs: self.btn_start.setText(_trf("btn.start_batch", "Start Batch ({count})", count=len(self._batch_jobs)))
        else: self.btn_start.setText(_tr("btn.start", "Start Processing (Ctrl+Enter)"))

    def _copy_path(self):
        if self._last_output:
            QApplication.clipboard().setText(self._last_output)
            self._toast_msg("Path copied to clipboard")

    def _open_folder(self):
        if self._last_output: reveal_in_explorer(self._last_output)

    def closeEvent(self, event):
        self._save_settings(); self._stop_glow()
        if self._tray: self._tray.hide()
        # Cancel all workers first, then wait (parallel cancel, sequential wait)
        workers = [self._worker, self._batch_worker, self._preview_worker, self._quick_preview_worker, self._benchmark_worker, self._model_dl_worker]
        for w in workers:
            if w and w.isRunning() and hasattr(w, 'cancel'): w.cancel()
        for w in workers:
            if w and w.isRunning():
                w.quit(); w.wait(1500)
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI MODE
# ═══════════════════════════════════════════════════════════════════════════════
def _cli_process_image(inp, out, model_key, args, bg_color, log_fn=print):
    """Process a single image in CLI mode."""
    if not out.lower().endswith('.png'):
        out = os.path.splitext(out)[0] + '.png'
    log_fn(f"\nProcessing image: {inp}")
    log_fn(f"  Model: {model_key}")
    log_fn(f"  Output: {out}")
    try:
        img = Image.open(inp); img.load()
        w, h = img.size
        log_fn(f"  Size: {w}x{h}")
        if args.max_res > 0 and max(w, h) > args.max_res:
            ratio = args.max_res / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            log_fn(f"  Scaled to {img.size[0]}x{img.size[1]}")
        engine = get_engine(model_key, log_fn=log_fn, gpu_device=args.gpu_device, fp16=args.fp16)
        mask = engine.predict_mask(img)
        if args.edge > 0 or args.shift != 0:
            mask = engine.refine_mask(mask, args.edge, args.shift, 0)
        if args.invert:
            mask = AlphaCutEngine.invert_mask(mask)
        if args.shadow > 0:
            mask = AlphaCutEngine.preserve_shadows(img, mask, args.shadow)
        src = img
        if args.spill > 0:
            src = AlphaCutEngine.suppress_spill(src, mask, args.spill, args.spill_color)
        fg = src.convert('RGBA')
        result = Image.composite(fg, Image.new('RGBA', fg.size, 0), mask)
        bg_image = None
        if args.bg_image and os.path.isfile(args.bg_image):
            try: bg_image = Image.open(args.bg_image); bg_image.load()
            except Exception: pass
        if bg_color is not None or bg_image is not None:
            result = AlphaCutEngine.composite_on_background(result, bg_color=bg_color, bg_image=bg_image)
        result.save(out, 'PNG')
        if not os.path.isfile(out):
            return {"ok": False, "output": out, "error": "No output produced."}
        log_fn(f"  Done: {out} ({os.path.getsize(out)/(1024*1024):.1f} MB)")
        return {"ok": True, "output": out}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        log_fn(f"  ERROR: {msg}")
        return {"ok": False, "output": out, "error": msg}


def _json_line(obj):
    """Print a single JSON object as one line to stdout."""
    print(json.dumps(obj, ensure_ascii=False), flush=True)

def run_cli(args):
    """Headless CLI processing."""
    if _HAS_QT:
        from PyQt6.QtCore import QCoreApplication
        cli_app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    use_json = getattr(args, 'json', False)
    completed = 0
    failed = 0

    def _out(msg):
        if use_json: _json_line({"type": "log", "message": msg})
        else: print(msg)

    def _status(s):
        if use_json: _json_line({"type": "status", "message": s})
        else: print(f"  {s}")

    def _progress(p):
        if use_json: _json_line({"type": "progress", "percent": p})
        elif p % 10 == 0: print(f"  {p}%", end='\r')

    def _error(msg, input_path=None, output_path=None):
        nonlocal failed
        failed += 1
        event = {"type": "error", "message": str(msg)}
        if input_path is not None:
            event["input"] = input_path
        if output_path is not None:
            event["output"] = output_path
        if use_json:
            _json_line(event)
        else:
            print(f"ERROR: {msg}")

    def _done(input_path, output_path):
        nonlocal completed
        completed += 1
        if not use_json:
            return
        event = {"type": "done", "input": input_path, "output": output_path}
        if os.path.isfile(output_path):
            event["size_mb"] = round(os.path.getsize(output_path) / (1024 * 1024), 1)
        _json_line(event)

    def _finish():
        if failed:
            if use_json:
                _json_line({"type": "failed", "processed": completed, "failed": failed})
            else:
                print(f"Failed: {failed} error(s), {completed} completed.")
            sys.exit(1)
        if use_json:
            _json_line({"type": "complete", "processed": completed, "failed": 0})
        else:
            print("Done.")

    model_names = list(MODELS.keys())
    model_key = model_names[0]
    for i, name in enumerate(model_names):
        if args.model.lower() in name.lower():
            model_key = name; break
    fmt = args.format
    if fmt not in ALL_OUTPUT_FORMAT_VALUES:
        _error(f"Unknown format: {fmt}. Options: {', '.join(ALL_OUTPUT_FORMAT_VALUES)}")
        _finish()

    # Parse bg_color once
    bg_color = None
    if args.bg_color:
        try:
            parts = [int(x.strip()) for x in args.bg_color.split(',')]
            if len(parts) != 3 or any(p < 0 or p > 255 for p in parts):
                raise ValueError("expected three 0-255 channels")
            bg_color = tuple(parts)
        except ValueError:
            _error(f"Invalid --bg-color: {args.bg_color} (use R,G,B format)")
            _finish()

    for inp in args.input:
        if not os.path.isfile(inp):
            _error(f"File not found: {inp}", input_path=inp)
            continue

        # Detect image input
        ext = os.path.splitext(inp)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            out = args.output if args.output else os.path.splitext(inp)[0] + '_alphacut.png'
            if os.path.exists(out) and not args.overwrite:
                _error(f"Output exists: {out} (use -y to overwrite)", input_path=inp, output_path=out)
                continue
            if use_json: _json_line({"type": "start", "input": inp, "output": out, "model": model_key})
            result = _cli_process_image(inp, out, model_key, args, bg_color, log_fn=_out)
            out = result.get("output", out)
            if result.get("ok") and os.path.isfile(out):
                _done(inp, out)
            else:
                _error(result.get("error") or "No output produced.", input_path=inp, output_path=out)
            continue

        if not find_ffmpeg():
            _error("FFmpeg not found - required for video processing.", input_path=inp)
            continue
        if args.output:
            out = args.output
        else:
            out = generate_output_name(inp, "{name}_alphacut", model_key, fmt)
        if os.path.exists(out) and not args.overwrite:
            _error(f"Output exists: {out} (use -y to overwrite)", input_path=inp, output_path=out)
            continue
        info = get_video_info(inp)
        if info:
            est_mb = estimate_output_size(info, fmt)
            try:
                drive = os.path.splitdrive(out)[0] or '/'
                free_mb = shutil.disk_usage(drive).free / (1024 * 1024)
                if free_mb < est_mb * 2.5:
                    _out(f"WARNING: Low disk space! Need ~{est_mb*2.5:.0f} MB, have {free_mb:.0f} MB")
            except Exception:
                pass
        if use_json:
            _json_line({"type": "start", "input": inp, "output": out, "model": model_key, "format": fmt})
        else:
            print(f"\nProcessing: {inp}")
            print(f"  Model: {model_key}")
            print(f"  Format: {fmt}")
            print(f"  Output: {out}")

        # Auto-detect or use --chroma-key flag
        chroma_result = None
        if args.chroma_key:
            chroma_result = detect_chroma_background(inp)
            if chroma_result:
                _out(f"Chroma-key detected: {chroma_result['color']}-screen")
            else:
                _out(f"--chroma-key set but no solid background detected; using AI")

        # Use chroma-key if detected
        if chroma_result and args.chroma_key:
            worker = ChromaKeyWorker(
                inp, out, fmt, chroma_result['color'],
                chroma_result['similarity'], chroma_result['blend'],
                quality=args.quality, keep_audio=args.audio)
            worker.log.connect(_out)
            worker.status.connect(_status)
            worker.progress.connect(_progress)
            worker_errors = []
            finished_paths = []
            worker.error.connect(lambda msg: worker_errors.append(str(msg)))
            worker.finished.connect(lambda path: finished_paths.append(path))
            worker.run()
            if not use_json: print()
            finished_out = finished_paths[-1] if finished_paths else None
            if worker_errors:
                _error(worker_errors[-1], input_path=inp, output_path=out)
            elif finished_out and (os.path.isfile(finished_out) or os.path.isdir(finished_out)):
                _done(inp, finished_out)
            else:
                _error("No output produced.", input_path=inp, output_path=out)
            continue

        if not info:
            _error(f"Cannot read video: {inp}", input_path=inp, output_path=out)
            continue
        try:
            engine = get_engine(model_key, log_fn=_out, gpu_device=args.gpu_device, fp16=args.fp16)
            engine.reset_temporal()
        except Exception as e:
            _error(f"{type(e).__name__}: {e}", input_path=inp, output_path=out)
            continue

        # Use processing worker synchronously
        worker = ProcessingWorker(
            inp, out, model_key, fmt, args.max_res,
            edge_softness=args.edge, mask_shift=args.shift,
            temporal_smooth=args.temporal, keep_audio=args.audio,
            frame_skip=args.frame_skip, invert_mask=args.invert,
            spill_strength=args.spill, spill_color=args.spill_color,
            shadow_strength=args.shadow, bg_color=bg_color,
            bg_image_path=args.bg_image, quality=args.quality,
            gpu_device=args.gpu_device, fp16=args.fp16,
            allow_large_animation=args.allow_large_animation)
        worker.log.connect(_out)
        worker.status.connect(_status)
        worker.progress.connect(_progress)
        worker_errors = []
        finished_paths = []
        worker.error.connect(lambda msg: worker_errors.append(str(msg)))
        worker.finished.connect(lambda path: finished_paths.append(path))
        worker.run()
        if not use_json: print()
        finished_out = finished_paths[-1] if finished_paths else None
        if worker_errors:
            _error(worker_errors[-1], input_path=inp, output_path=out)
        elif finished_out and (os.path.isfile(finished_out) or os.path.isdir(finished_out)):
            _done(inp, finished_out)
        else:
            _error("No output produced.", input_path=inp, output_path=out)

    _finish()


# ═══════════════════════════════════════════════════════════════════════════════
# PIPE MODE — raw RGBA to stdout for FFmpeg/scriptable pipelines
# ═══════════════════════════════════════════════════════════════════════════════
def _open_pipe_decoder(cmd):
    """Open FFmpeg for pipe mode without risking a blocked stderr pipe."""
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=_SUBPROCESS_FLAGS)


def run_pipe(args):
    """Stream raw RGBA frames to stdout for piping into FFmpeg or other tools.

    Usage example:
      python AlphaCut.py -i video.mp4 --pipe 2>nul | ^
        ffmpeg -f rawvideo -pix_fmt rgba -s 1920x1080 -r 30 -i - ^
        -c:v libvpx-vp9 -pix_fmt yuva420p output.webm
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("ERROR: FFmpeg not found.", file=sys.stderr); sys.exit(1)
    if len(args.input) != 1:
        print("ERROR: --pipe mode supports exactly one input file.", file=sys.stderr); sys.exit(1)
    inp = args.input[0]
    if not os.path.isfile(inp):
        print(f"ERROR: File not found: {inp}", file=sys.stderr); sys.exit(1)

    info = get_video_info(inp)
    if not info:
        print("ERROR: Cannot read video.", file=sys.stderr); sys.exit(1)

    w, h = info['width'], info['height']
    fps = info['fps']
    sw, sh = w, h
    if args.max_res > 0 and max(w, h) > args.max_res:
        ratio = args.max_res / max(w, h)
        sw = int(w * ratio) // 2 * 2; sh = int(h * ratio) // 2 * 2

    model_names = list(MODELS.keys())
    model_key = model_names[0]
    for name in model_names:
        if args.model.lower() in name.lower():
            model_key = name; break

    bg_color = None
    if args.bg_color:
        try:
            parts = [int(x.strip()) for x in args.bg_color.split(',')]
            if len(parts) == 3:
                bg_color = tuple(parts)
        except ValueError:
            pass
    bg_image = None
    if args.bg_image and os.path.isfile(args.bg_image):
        try:
            bg_image = Image.open(args.bg_image)
            bg_image.load()
            bg_image = bg_image.convert('RGB').resize((sw, sh), Image.Resampling.LANCZOS)
        except Exception:
            bg_image = None

    print(f"PIPE: {sw}x{sh} @ {fps}fps | model={model_key.split('(')[0].strip()}", file=sys.stderr)
    print(f"PIPE_HEADER: {sw} {sh} {fps}", file=sys.stderr)

    engine = get_engine(model_key, log_fn=lambda m: print(m, file=sys.stderr),
                        gpu_device=args.gpu_device, fp16=args.fp16)
    engine.reset_temporal()

    cmd = [ffmpeg, '-v', 'warning', '-i', inp, '-fps_mode', 'cfr']
    if sw != w or sh != h:
        cmd += ['-vf', f'scale={sw}:{sh}']
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']

    proc = _open_pipe_decoder(cmd)
    frame_size = sw * sh * 3
    stdout_bin = sys.stdout.buffer
    frame_num = 0

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            img = Image.frombytes('RGB', (sw, sh), raw)
            mask = engine.predict_mask(img)
            if args.edge > 0 or args.shift != 0 or args.temporal > 0:
                mask = engine.refine_mask(mask, args.edge, args.shift, args.temporal)
            if args.invert:
                mask = AlphaCutEngine.invert_mask(mask)
            if args.spill > 0:
                img = AlphaCutEngine.suppress_spill(img, mask, args.spill, args.spill_color)
            if args.shadow > 0:
                mask = AlphaCutEngine.preserve_shadows(img, mask, args.shadow)
            fg = img.convert('RGBA')
            result = Image.composite(fg, Image.new('RGBA', fg.size, 0), mask)
            if bg_color is not None or bg_image is not None:
                result = AlphaCutEngine.composite_on_background(result, bg_color=bg_color, bg_image=bg_image)
            stdout_bin.write(result.tobytes())
            frame_num += 1
            if frame_num % 30 == 0:
                print(f"PIPE: frame {frame_num}", file=sys.stderr)
    except BrokenPipeError:
        pass
    finally:
        proc.terminate()
        proc.wait()

    print(f"PIPE: done — {frame_num} frames", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def build_parser():
    parser = argparse.ArgumentParser(
        prog='AlphaCut', description='AI Video Background Removal',
        epilog='Run without arguments for GUI mode.')
    parser.add_argument('--input', '-i', nargs='+', help='Input video file(s)')
    parser.add_argument('--output', '-o', help='Output file path (single file mode)')
    parser.add_argument('--model', '-m', default='u2net_human_seg', help='Model name (partial match)')
    parser.add_argument('--format', '-f', default='mp4', choices=list(ALL_OUTPUT_FORMAT_VALUES), help='Output format')
    parser.add_argument('--quality', '-q', type=int, default=70, help='Output quality 0-100 (0=smallest, 100=best)')
    parser.add_argument('--max-res', type=int, default=0, help='Max resolution (0=original)')
    parser.add_argument('--edge', type=int, default=0, help='Edge softness (0-100)')
    parser.add_argument('--shift', type=int, default=0, help='Mask shift (-20 to +20)')
    parser.add_argument('--temporal', type=int, default=0, help='Temporal smoothing (0-7)')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every Nth frame (1=all)')
    parser.add_argument('--invert', action='store_true', help='Invert mask (remove subject)')
    parser.add_argument('--spill', type=int, default=0, help='Spill suppression strength (0-100)')
    parser.add_argument('--spill-color', default='green', choices=['green', 'blue', 'red'],
                        help='Spill color to suppress')
    parser.add_argument('--shadow', type=int, default=0, help='Shadow preservation strength (0-100)')
    parser.add_argument('--bg-color', help='Background color as R,G,B (e.g. 0,255,0)')
    parser.add_argument('--bg-image', help='Background image file path')
    parser.add_argument('--no-audio', action='store_true', help='Strip audio')
    parser.add_argument('-y', '--overwrite', action='store_true', help='Overwrite output without confirmation')
    parser.add_argument('--gpu-device', type=int, default=-1,
                        help='GPU device index for inference (-1=auto, 0=first GPU, etc.)')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 half-precision on GPU (~2x faster, minimal quality loss)')
    parser.add_argument('--allow-large-animation', action='store_true',
                        help='Allow animated WebP/GIF exports above the RAM safety estimate')
    parser.add_argument('--chroma-key', action='store_true', help='Use FFmpeg chroma-key instead of AI (for green/blue screen footage)')
    parser.add_argument('--pipe', action='store_true',
                        help='Pipe raw RGBA frames to stdout (for FFmpeg stdin pipelines). '
                             'Prints frame dimensions to stderr, then streams WxHx4 bytes per frame.')
    parser.add_argument('--json', action='store_true',
                        help='Output machine-readable JSON lines (one JSON object per event)')
    parser.add_argument('--runtime-info', action='store_true',
                        help='Print ONNX Runtime provider diagnostics and install profile guidance')
    parser.add_argument('--version', action='version', version=f'AlphaCut v{__version__}')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.audio = not args.no_audio

    if args.runtime_info:
        print(_format_runtime_diagnostics(include_notes=True))
        return

    if args.input and args.pipe:
        run_pipe(args)
        return
    if args.input:
        run_cli(args)
        return

    if not _HAS_QT:
        print("ERROR: PyQt6 is required for GUI mode. Install with: pip install PyQt6")
        print("For CLI mode, use: python AlphaCut.py -i video.mp4 -f webm")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle('Fusion'); app.setStyleSheet(DARK_STYLE); app.setWindowIcon(get_app_icon())
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(13, 15, 20))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(200, 204, 212))
    pal.setColor(QPalette.ColorRole.Base, QColor(10, 12, 16))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(19, 22, 29))
    pal.setColor(QPalette.ColorRole.Text, QColor(200, 204, 212))
    pal.setColor(QPalette.ColorRole.Button, QColor(19, 22, 29))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(200, 204, 212))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(108, 92, 231))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)
    win = AlphaCutWindow(); win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
