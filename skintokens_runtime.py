"""Runtime setup helpers for the StableGen SkinTokens ComfyUI node."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = NODE_DIR / "skintokens_config.json"
VENDOR_DIR = NODE_DIR / "vendor"
DEFAULT_SKINTOKENS_ROOT = VENDOR_DIR / "SkinTokens"
DEFAULT_VENV = VENDOR_DIR / "skintokens_env"
LOCK_PATH = VENDOR_DIR / ".install.lock"
SKINTOKENS_REPO = "https://github.com/VAST-AI-Research/SkinTokens.git"
DEFAULT_TIMEOUT = 3600

INSTALL_POLICIES = ["auto", "managed", "comfy", "configured_only"]
ATTENTION_BACKENDS = ["auto", "flash_attention_2", "sdpa"]


def _log(message: str) -> None:
    print(f"[StableGen SkinTokens] {message}")


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _attention_backend() -> str:
    backend = os.environ.get("SKINTOKENS_ATTENTION_BACKEND", "auto").strip().lower()
    if backend not in ATTENTION_BACKENDS:
        raise RuntimeError(
            f"Unknown SKINTOKENS_ATTENTION_BACKEND={backend!r}; "
            f"expected one of {', '.join(ATTENTION_BACKENDS)}"
        )
    return backend


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    pretty = " ".join(f'"{p}"' if " " in str(p) else str(p) for p in cmd)
    _log(f"$ {pretty}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[StableGen SkinTokens] {line.rstrip()}")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {pretty}")


def _run_capture(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


def _as_abs(path_value: str | os.PathLike, base: Path = NODE_DIR) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _write_config(root: Path, python_exe: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    rel_root = os.path.relpath(root, NODE_DIR)
    rel_python = os.path.relpath(python_exe, NODE_DIR)
    data = {
        "skintokens_root": rel_root,
        "python": rel_python,
        "timeout": int(timeout),
    }
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _log(f"Wrote runtime config: {CONFIG_PATH}")


def _env_config() -> tuple[Path, str, int] | None:
    root_raw = os.environ.get("SKINTOKENS_ROOT", "").strip()
    if not root_raw:
        return None
    python_raw = os.environ.get("SKINTOKENS_PYTHON", "").strip() or sys.executable
    timeout = int(os.environ.get("SKINTOKENS_TIMEOUT", str(DEFAULT_TIMEOUT)))
    return Path(root_raw).expanduser().resolve(), python_raw, timeout


def _file_config() -> tuple[Path, str, int] | None:
    data = _read_json(CONFIG_PATH)
    if not data:
        return None
    root_raw = data.get("skintokens_root") or data.get("root")
    python_raw = data.get("python") or data.get("python_exe")
    if not root_raw or not python_raw:
        raise RuntimeError(f"{CONFIG_PATH} must define skintokens_root and python")
    timeout = int(data.get("timeout", DEFAULT_TIMEOUT))
    root = _as_abs(root_raw)
    python_exe = str(_as_abs(python_raw))
    return root, python_exe, timeout


def _validate_config(config: tuple[Path, str, int]) -> tuple[Path, str, int]:
    root, python_exe, timeout = config
    demo = root / "demo.py"
    if not demo.is_file():
        raise RuntimeError(f"SkinTokens demo.py was not found at {demo}")
    if not Path(python_exe).is_file():
        raise RuntimeError(f"SkinTokens Python executable was not found: {python_exe}")
    return root, python_exe, timeout


def configured_runtime() -> tuple[Path, str, int] | None:
    """Return env/config runtime if it is present and path-valid."""
    config = _env_config() or _file_config()
    if not config:
        return None
    return _validate_config(config)


def _ensure_source_repo(root: Path = DEFAULT_SKINTOKENS_ROOT) -> Path:
    if (root / "demo.py").is_file():
        return root.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"{root} exists but does not look like a SkinTokens checkout")
    _run(["git", "clone", SKINTOKENS_REPO, str(root)])
    return root.resolve()


def _python_version_ok(python_exe: str | Path) -> bool:
    result = _run_capture([
        str(python_exe),
        "-c",
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
    ])
    if result.returncode != 0:
        return False
    try:
        major, minor = (int(part) for part in result.stdout.strip().split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (3, 11)


def _python_has_runtime_deps(python_exe: str | Path) -> bool:
    imports = [
        "torch",
        "transformers",
        "diffusers",
        "bpy",
        "trimesh",
        "open3d",
        "huggingface_hub",
    ]
    code = "import " + ", ".join(imports)
    result = _run_capture([str(python_exe), "-c", code])
    if result.returncode == 0:
        return True
    _log("Current Python is not SkinTokens-ready:")
    for line in result.stdout.splitlines():
        _log(line)
    return False


def _python_has_module(python_exe: str | Path, module: str) -> bool:
    return _run_capture([str(python_exe), "-c", f"import {module}"]).returncode == 0


def _has_models(root: Path) -> bool:
    expected = [
        root / "experiments" / "articulation_xl_quantization_256_token_4",
        root / "experiments" / "skin_vae_2_10_32768",
    ]
    return all(path.exists() for path in expected)


def _apply_sdpa_attention_patch(root: Path) -> None:
    """Patch SkinTokens to use PyTorch SDPA when flash-attn is unavailable.

    Upstream TokenRig hardcodes Hugging Face `flash_attention_2` and imports
    `flash_attn` at module import time. That is awkward on Windows, where
    flash-attn often has to build from source. PyTorch's SDPA is slower and can
    use more VRAM, but it lets Windows installs proceed without compiling
    flash-attn.
    """
    replacements = {
        root / "src" / "model" / "tokenrig.py": [
            (
                'attn_implementation="flash_attention_2"',
                'attn_implementation="sdpa"',
            ),
            (
                "try: from flash_attn_interface import flash_attn_func # type: ignore except Exception as e: "
                "from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func "
                "def flash_attn_func(*args, **kwargs): res = _flash_attn_func(*args, **kwargs) return res, None",
                "def flash_attn_func(*args, **kwargs):\n"
                "    import torch.nn.functional as _F\n"
                "    q, k, v = args[:3]\n"
                "    dropout_p = kwargs.get('dropout_p', 0.0)\n"
                "    softmax_scale = kwargs.get('softmax_scale', None)\n"
                "    causal = kwargs.get('causal', False)\n"
                "    if q.dim() == 4:\n"
                "        out = _F.scaled_dot_product_attention(\n"
                "            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),\n"
                "            dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,\n"
                "        ).transpose(1, 2)\n"
                "    else:\n"
                "        out = _F.scaled_dot_product_attention(\n"
                "            q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,\n"
                "        )\n"
                "    return out, None\n",
            ),
        ],
        root / "src" / "server" / "spec.py": [
            (
                '_attn_implementation="flash_attention_2"',
                '_attn_implementation="sdpa"',
            ),
        ],
    }

    for path, pairs in replacements.items():
        if not path.is_file():
            raise RuntimeError(f"Cannot patch SkinTokens attention backend; missing {path}")
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in pairs:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")
            _log(f"Patched SkinTokens attention backend to SDPA in {path}")


def _prepare_attention_backend(python_exe: str | Path, root: Path, allow_install: bool) -> None:
    backend = _attention_backend()
    if backend == "sdpa":
        _apply_sdpa_attention_patch(root)
        return

    if backend == "auto" and os.name == "nt":
        _log("Windows detected; using PyTorch SDPA instead of building flash-attn.")
        _apply_sdpa_attention_patch(root)
        return

    if _flag("SKINTOKENS_SKIP_FLASH_ATTN_INSTALL"):
        _log("SKINTOKENS_SKIP_FLASH_ATTN_INSTALL is set; using PyTorch SDPA.")
        _apply_sdpa_attention_patch(root)
        return

    if _python_has_module(python_exe, "flash_attn"):
        return

    if not allow_install:
        if backend == "flash_attention_2":
            return
        _log("flash-attn is not installed in this Python; using PyTorch SDPA.")
        _apply_sdpa_attention_patch(root)
        return

    try:
        _run([
            str(python_exe),
            "-m",
            "pip",
            "install",
            "flash-attn",
            "--no-build-isolation",
        ])
    except RuntimeError:
        if backend == "flash_attention_2":
            raise
        _log("flash-attn install failed; using PyTorch SDPA fallback.")
        _apply_sdpa_attention_patch(root)


def _install_dependencies(python_exe: str | Path, root: Path) -> None:
    if not _python_version_ok(python_exe):
        raise RuntimeError(
            "SkinTokens requires Python >= 3.11. Set SKINTOKENS_BOOTSTRAP_PYTHON "
            "to a Python 3.11+ executable, or use a ComfyUI Python built on 3.11+."
        )

    _run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    if not _flag("SKINTOKENS_SKIP_TORCH_INSTALL"):
        torch_index = os.environ.get(
            "SKINTOKENS_TORCH_INDEX_URL",
            "https://download.pytorch.org/whl/cu128",
        ).strip()
        _run([
            str(python_exe),
            "-m",
            "pip",
            "install",
            "torch==2.7.0",
            "torchvision==0.22.0",
            "torchaudio==2.7.0",
            "--index-url",
            torch_index,
        ])

    _run([str(python_exe), "-m", "pip", "install", "-r", str(root / "requirements.txt")])
    _prepare_attention_backend(python_exe, root, allow_install=True)


def _download_models(python_exe: str | Path, root: Path) -> None:
    if _has_models(root):
        _log("SkinTokens model directories already exist; skipping model download.")
        return
    if _flag("SKINTOKENS_SKIP_MODEL_DOWNLOAD"):
        _log("SKINTOKENS_SKIP_MODEL_DOWNLOAD is set; skipping model download.")
        return
    _run([str(python_exe), str(root / "download.py"), "--model"], cwd=root)


def _create_managed_python(venv_dir: Path = DEFAULT_VENV) -> Path:
    python_bootstrap = os.environ.get("SKINTOKENS_BOOTSTRAP_PYTHON", "").strip() or sys.executable
    if not _python_version_ok(python_bootstrap):
        raise RuntimeError(
            "The Python used to create the managed SkinTokens environment is below 3.11. "
            "Install Python 3.11+ and set SKINTOKENS_BOOTSTRAP_PYTHON to it."
        )
    if not _venv_python(venv_dir).is_file():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        _log(f"Creating managed venv: {venv_dir}")
        _run([python_bootstrap, "-m", "venv", str(venv_dir)])
    return _venv_python(venv_dir).resolve()


def _install_managed(root: Path) -> tuple[Path, str, int]:
    python_exe = _create_managed_python()
    _install_dependencies(python_exe, root)
    _download_models(python_exe, root)
    _write_config(root, python_exe)
    return root, str(python_exe), DEFAULT_TIMEOUT


def _install_comfy(root: Path) -> tuple[Path, str, int]:
    python_exe = Path(sys.executable).resolve()
    _install_dependencies(python_exe, root)
    _download_models(python_exe, root)
    _write_config(root, python_exe)
    return root, str(python_exe), DEFAULT_TIMEOUT


class InstallLock:
    def __enter__(self):
        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        now = time.time()
        if LOCK_PATH.exists() and now - LOCK_PATH.stat().st_mtime > 6 * 60 * 60:
            LOCK_PATH.unlink()
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f"SkinTokens install is already running or left a lock file at {LOCK_PATH}"
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


def ensure_runtime(install_policy: str = "auto") -> tuple[Path, str, int]:
    """Return a usable SkinTokens runtime, installing lazily if requested."""
    policy = (install_policy or "auto").strip().lower()
    if policy not in INSTALL_POLICIES:
        raise RuntimeError(f"Unknown SkinTokens install policy '{install_policy}'")

    configured = configured_runtime()
    if configured:
        return configured

    if policy == "configured_only":
        raise RuntimeError(
            "SkinTokens runtime is not configured. Run install_skintokens.py, "
            "set SKINTOKENS_ROOT/SKINTOKENS_PYTHON, or choose install_policy='auto'."
        )

    with InstallLock():
        configured = configured_runtime()
        if configured:
            return configured

        root = _ensure_source_repo()

        if policy == "comfy":
            return _install_comfy(root)

        if policy == "auto":
            python_exe = Path(sys.executable).resolve()
            if _python_version_ok(python_exe) and _python_has_runtime_deps(python_exe):
                _prepare_attention_backend(python_exe, root, allow_install=False)
                _download_models(python_exe, root)
                _write_config(root, python_exe)
                return root, str(python_exe), DEFAULT_TIMEOUT
            _log("Falling back to a managed SkinTokens environment.")

        return _install_managed(root)


def doctor_text() -> str:
    lines = [f"Node directory: {NODE_DIR}"]
    try:
        config = configured_runtime()
    except Exception as exc:
        lines.append(f"Configured runtime: invalid ({exc})")
        config = None

    if not config:
        lines.append("Configured runtime: missing")
        lines.append(f"Expected config: {CONFIG_PATH}")
        return "\n".join(lines)

    root, python_exe, timeout = config
    lines.append(f"SkinTokens root: {root}")
    lines.append(f"Python: {python_exe}")
    lines.append(f"Timeout: {timeout}")
    lines.append(f"Python >=3.11: {_python_version_ok(python_exe)}")
    lines.append(f"Runtime imports: {_python_has_runtime_deps(python_exe)}")
    lines.append(f"flash-attn import: {_python_has_module(python_exe, 'flash_attn')}")
    lines.append(f"Attention backend policy: {_attention_backend()}")
    lines.append(f"Model dirs present: {_has_models(root)}")
    return "\n".join(lines)
