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
RUNTIME_IMPORTS = [
    "torch",
    "transformers",
    "diffusers",
    "box",
    "einops",
    "omegaconf",
    "lightning",
    "addict",
    "fast_simplification",
    "bpy",
    "trimesh",
    "open3d",
    "huggingface_hub",
    "numpy",
    "gradio",
    "bottle",
    "tornado",
    "scipy",
    "requests",
    "tqdm",
]
SUPPLEMENTAL_DEPENDENCIES = {
    "scipy": "scipy",
    "requests": "requests",
    "tqdm": "tqdm",
}


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


def _config_path_value(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(NODE_DIR))
    except ValueError:
        return str(path)


def _write_config(
    root: Path,
    python_exe: Path,
    timeout: int = DEFAULT_TIMEOUT,
    install_complete: bool = True,
) -> None:
    data = {
        "skintokens_root": _config_path_value(root),
        "python": _config_path_value(python_exe),
        "timeout": int(timeout),
        "install_complete": bool(install_complete),
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


def _file_config(require_complete: bool = True) -> tuple[Path, str, int] | None:
    data = _read_json(CONFIG_PATH)
    if not data:
        return None
    if require_complete and data.get("install_complete") is False:
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


def configured_runtime(require_complete: bool = True) -> tuple[Path, str, int] | None:
    """Return env/config runtime if it is present and path-valid."""
    config = _env_config() or _file_config(require_complete=require_complete)
    if not config:
        return None
    return _validate_config(config)


def _mark_config_complete(config: tuple[Path, str, int]) -> tuple[Path, str, int]:
    root, python_exe, timeout = config
    _write_config(root, Path(python_exe), timeout=timeout, install_complete=True)
    return root, python_exe, timeout


def _runtime_is_ready(config: tuple[Path, str, int]) -> bool:
    root, python_exe, _timeout = config
    return (
        _python_version_ok(python_exe)
        and _python_has_runtime_deps(python_exe)
        and _has_models(root)
    )


def _ensure_configured_runtime_ready(
    config: tuple[Path, str, int],
    allow_install: bool,
) -> tuple[Path, str, int]:
    root, python_exe, timeout = config
    if _runtime_is_ready(config):
        _prepare_attention_backend(python_exe, root, allow_install=False)
        return _mark_config_complete(config)

    if not allow_install:
        raise RuntimeError(
            "Configured SkinTokens runtime exists but is not ready. "
            "Run install_skintokens.py --target auto to repair dependencies/models, "
            "or use install_policy='auto'."
        )

    return _finish_install(root, Path(python_exe), timeout=timeout)


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


def _runtime_import_failures(python_exe: str | Path) -> list[str]:
    code = (
        "import importlib.util\n"
        f"modules = {json.dumps(RUNTIME_IMPORTS)}\n"
        "failed = []\n"
        "for name in modules:\n"
        "    try:\n"
        "        found = importlib.util.find_spec(name) is not None\n"
        "    except Exception as exc:\n"
        "        failed.append(f'{name}: {type(exc).__name__}: {exc}')\n"
        "    else:\n"
        "        if not found:\n"
        "            failed.append(f'{name}: missing')\n"
        "if failed:\n"
        "    print('\\n'.join(failed))\n"
        "    raise SystemExit(1)\n"
    )
    result = _run_capture([str(python_exe), "-c", code])
    if result.returncode == 0:
        return []
    failures = [line for line in result.stdout.splitlines() if line.strip()]
    return failures or ["unknown import error"]


def _python_has_runtime_deps(python_exe: str | Path) -> bool:
    failures = _runtime_import_failures(python_exe)
    if not failures:
        return True
    _log("Current Python is not SkinTokens-ready:")
    for line in failures:
        _log(line)
    return False


def _python_has_module(python_exe: str | Path, module: str) -> bool:
    code = (
        "import importlib.util, sys\n"
        f"sys.exit(0 if importlib.util.find_spec({json.dumps(module)}) else 1)\n"
    )
    return _run_capture([str(python_exe), "-c", code]).returncode == 0


def _install_supplemental_dependencies(python_exe: str | Path) -> None:
    missing = [
        package
        for module, package in SUPPLEMENTAL_DEPENDENCIES.items()
        if not _python_has_module(python_exe, module)
    ]
    if missing:
        _log(f"Installing supplemental SkinTokens dependencies: {', '.join(missing)}")
        _run([str(python_exe), "-m", "pip", "install", *missing])


def _has_models(root: Path) -> bool:
    expected = [
        root / "experiments" / "articulation_xl_quantization_256_token_4",
        root / "experiments" / "skin_vae_2_10_32768",
    ]
    return all(path.exists() for path in expected)


def _apply_sdpa_attention_patch(root: Path) -> None:
    """Patch SkinTokens to use PyTorch SDPA when flash-attn is unavailable.

    Upstream hardcodes Hugging Face `flash_attention_2` and imports
    `flash_attn` at module import time in a couple of files. That is awkward on
    Windows, where flash-attn often has to build from source. PyTorch's SDPA is
    slower and can use more VRAM, but it lets Windows installs proceed without
    compiling flash-attn.
    """
    flash_attn_import_block = (
        "try:\n"
        "    from flash_attn_interface import flash_attn_func # type: ignore\n"
        "except Exception as e:\n"
        "    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func\n"
        "    def flash_attn_func(*args, **kwargs):\n"
        "        res = _flash_attn_func(*args, **kwargs)\n"
        "        return res, None\n"
    )
    sdpa_flash_attn_func = (
        "def flash_attn_func(*args, **kwargs):\n"
        "    import torch.nn.functional as _F\n"
        "    q, k, v = args[:3]\n"
        "    dropout_p = kwargs.get('dropout_p', 0.0)\n"
        "    softmax_scale = kwargs.get('softmax_scale', None)\n"
        "    causal = kwargs.get('causal', False)\n"
        "    if q.dim() == 4:\n"
        "        q_t = q.transpose(1, 2)\n"
        "        k_t = k.transpose(1, 2)\n"
        "        v_t = v.transpose(1, 2)\n"
        "        if q_t.shape[1] != k_t.shape[1]:\n"
        "            repeat_factor = q_t.shape[1] // k_t.shape[1]\n"
        "            k_t = k_t.repeat_interleave(repeat_factor, dim=1)\n"
        "            v_t = v_t.repeat_interleave(repeat_factor, dim=1)\n"
        "        out = _F.scaled_dot_product_attention(\n"
        "            q_t, k_t, v_t, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,\n"
        "        ).transpose(1, 2)\n"
        "    else:\n"
        "        out = _F.scaled_dot_product_attention(\n"
        "            q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,\n"
        "        )\n"
        "    return out, None\n"
    )
    replacements = {
        root / "src" / "model" / "tokenrig.py": [
            (
                'attn_implementation="flash_attention_2"',
                'attn_implementation="sdpa"',
            ),
            (
                flash_attn_import_block,
                sdpa_flash_attn_func,
            ),
        ],
        root / "src" / "model" / "skin_vae_model.py": [
            (
                flash_attn_import_block,
                sdpa_flash_attn_func,
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
    _install_supplemental_dependencies(python_exe)
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


def _finish_install(
    root: Path,
    python_exe: Path,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Path, str, int]:
    _write_config(root, python_exe, timeout=timeout, install_complete=False)
    if _python_version_ok(python_exe):
        _install_supplemental_dependencies(python_exe)

    if _python_version_ok(python_exe) and _python_has_runtime_deps(python_exe):
        _prepare_attention_backend(python_exe, root, allow_install=False)
    else:
        _install_dependencies(python_exe, root)
    _download_models(python_exe, root)
    _write_config(root, python_exe, timeout=timeout, install_complete=True)
    return root, str(python_exe), timeout


def _install_managed(root: Path) -> tuple[Path, str, int]:
    python_exe = _create_managed_python()
    return _finish_install(root, python_exe)


def _install_comfy(root: Path) -> tuple[Path, str, int]:
    python_exe = Path(sys.executable).resolve()
    return _finish_install(root, python_exe)


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

    if policy == "configured_only":
        configured = configured_runtime()
        if configured:
            return _ensure_configured_runtime_ready(configured, allow_install=False)

        raise RuntimeError(
            "SkinTokens runtime is not configured. Run install_skintokens.py, "
            "set SKINTOKENS_ROOT/SKINTOKENS_PYTHON, or choose install_policy='auto'."
        )

    if policy == "auto":
        configured = configured_runtime()
        if configured and _runtime_is_ready(configured):
            return _ensure_configured_runtime_ready(configured, allow_install=True)

    with InstallLock():
        if policy == "auto":
            configured = configured_runtime()
            if configured:
                return _ensure_configured_runtime_ready(configured, allow_install=True)

        root = _ensure_source_repo()

        if policy == "comfy":
            return _install_comfy(root)

        if policy == "auto":
            python_exe = Path(sys.executable).resolve()
            if _python_version_ok(python_exe) and _python_has_runtime_deps(python_exe):
                return _finish_install(root, python_exe)
            _log("Falling back to a managed SkinTokens environment.")

        return _install_managed(root)


def doctor_text() -> str:
    lines = [f"Node directory: {NODE_DIR}"]
    config_data = _read_json(CONFIG_PATH)
    try:
        config = configured_runtime(require_complete=False)
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
    if config_data:
        recorded = config_data.get("install_complete")
        if recorded is None:
            lines.append("Install complete: not recorded (legacy config)")
        else:
            lines.append(f"Install complete: {recorded}")
    lines.append(f"Python >=3.11: {_python_version_ok(python_exe)}")
    import_failures = _runtime_import_failures(python_exe)
    lines.append(f"Runtime modules present: {not import_failures}")
    if import_failures:
        lines.append("Runtime module failures:")
        lines.extend(f"- {failure}" for failure in import_failures)
    lines.append(f"flash-attn import: {_python_has_module(python_exe, 'flash_attn')}")
    lines.append(f"Attention backend policy: {_attention_backend()}")
    lines.append(f"Model dirs present: {_has_models(root)}")
    return "\n".join(lines)
