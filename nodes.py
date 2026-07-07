"""ComfyUI nodes for StableGen SkinTokens auto-rigging."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from .skintokens_runtime import INSTALL_POLICIES, doctor_text, ensure_runtime
except ImportError:  # pragma: no cover - supports direct `import nodes` checks
    from skintokens_runtime import INSTALL_POLICIES, doctor_text, ensure_runtime

try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - only happens outside ComfyUI
    folder_paths = None


class AnyType(str):
    """Wildcard ComfyUI type that accepts any upstream output type."""

    def __ne__(self, other: object) -> bool:
        return False


ANY = AnyType("*")
SUPPORTED_SUFFIXES = {".glb", ".obj", ".fbx"}


def _output_dir() -> Path:
    configured = os.environ.get("SKINTOKENS_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory()).resolve()
    return (Path.cwd() / "output").resolve()


def _input_dir() -> Path | None:
    if folder_paths is None:
        return None
    return Path(folder_paths.get_input_directory()).resolve()


def _temp_dir() -> Path | None:
    if folder_paths is None:
        return None
    return Path(folder_paths.get_temp_directory()).resolve()


def _base_dir_for_type(file_type: str) -> Path:
    file_type = (file_type or "output").lower()
    if file_type == "input":
        return _input_dir() or Path.cwd()
    if file_type == "temp":
        return _temp_dir() or Path.cwd()
    return _output_dir()


def _resolve_file_ref(value: dict[str, Any]) -> Path | None:
    path = value.get("path")
    if path:
        return _resolve_input_path(path)

    filename = value.get("filename") or value.get("name")
    if not filename:
        return None

    base = _base_dir_for_type(str(value.get("type") or "output"))
    subfolder = str(value.get("subfolder") or "").strip()
    candidate = base / subfolder / str(filename)
    return candidate.resolve()


def _resolve_input_path(value: Any) -> Path | None:
    if value is None:
        return None

    if isinstance(value, dict):
        return _resolve_file_ref(value)

    if isinstance(value, (list, tuple)):
        for item in value:
            resolved = _resolve_input_path(item)
            if resolved:
                return resolved
        return None

    raw = str(value).strip().strip('"').strip("'")
    if not raw:
        return None

    if raw.startswith("file://"):
        raw = raw[7:]

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        _output_dir() / raw,
        (_input_dir() / raw) if _input_dir() else None,
        (_temp_dir() / raw) if _temp_dir() else None,
        Path.cwd() / raw,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()

    return (_output_dir() / raw).resolve()


def _slug(value: str, fallback: str = "asset") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (slug or fallback)[:80]


def _flag(value: bool) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"} if isinstance(value, str) else bool(value)


class StableGenSkinTokensAutoRig:
    """Run SkinTokens/TokenRig on an upstream 3D file and emit a rigged GLB."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rigged_glb_path",)
    FUNCTION = "autorig"
    CATEGORY = "StableGen/3D"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "output_prefix": (
                    "STRING",
                    {
                        "default": "stablegen_skintokens",
                        "multiline": False,
                    },
                ),
                "install_policy": (
                    INSTALL_POLICIES,
                    {
                        "default": "auto",
                    },
                ),
                "use_transfer": ("BOOLEAN", {"default": True}),
                "use_skeleton": ("BOOLEAN", {"default": False}),
                "use_postprocess": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 200, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.5, "max": 3.0, "step": 0.1},
                ),
                "num_beams": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1}),
                "model_ckpt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "hf_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
            },
            "optional": {
                "input_glb": (ANY, {"forceInput": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return time.time()

    def _build_output_path(self, source_path: Path, output_prefix: str) -> Path:
        out_dir = _output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _slug(source_path.stem)
        prefix = _slug(output_prefix, fallback="stablegen_skintokens")
        unique = uuid.uuid4().hex[:8]
        return out_dir / f"{prefix}_{stem}_{unique}_rigged.glb"

    def autorig(
        self,
        input_path: str,
        output_prefix: str,
        install_policy: str,
        use_transfer: bool,
        use_skeleton: bool,
        use_postprocess: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        num_beams: int,
        model_ckpt: str,
        hf_path: str,
        input_glb: Any = None,
    ):
        source_path = _resolve_input_path(input_glb) or _resolve_input_path(input_path)
        if not source_path:
            raise RuntimeError("No input GLB/OBJ/FBX path was provided to SkinTokens.")
        if not source_path.is_file():
            raise RuntimeError(f"Input 3D file does not exist on the ComfyUI host: {source_path}")
        if source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise RuntimeError(
                f"SkinTokens supports {', '.join(sorted(SUPPORTED_SUFFIXES))}; got {source_path.suffix}"
            )

        root, python_exe, timeout = ensure_runtime(install_policy)
        output_path = self._build_output_path(source_path, output_prefix)

        cmd = [
            python_exe,
            "-u",
            str(root / "demo.py"),
            "--input",
            str(source_path),
            "--output",
            str(output_path),
            "--top_k",
            str(int(top_k)),
            "--top_p",
            str(float(top_p)),
            "--temperature",
            str(float(temperature)),
            "--repetition_penalty",
            str(float(repetition_penalty)),
            "--num_beams",
            str(int(num_beams)),
        ]
        if _flag(use_transfer):
            cmd.append("--use_transfer")
        if _flag(use_skeleton):
            cmd.append("--use_skeleton")
        if _flag(use_postprocess):
            cmd.append("--use_postprocess")
        if model_ckpt.strip():
            cmd.extend(["--model_ckpt", model_ckpt.strip()])
        if hf_path.strip():
            cmd.extend(["--hf_path", hf_path.strip()])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        print("[StableGen SkinTokens] Running:", " ".join(f'"{p}"' if " " in p else p for p in cmd))
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"SkinTokens timed out after {timeout} seconds") from exc

        if completed.stdout:
            for line in completed.stdout.splitlines():
                print(f"[StableGen SkinTokens] {line}")

        if completed.returncode != 0:
            raise RuntimeError(f"SkinTokens failed with exit code {completed.returncode}")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"SkinTokens completed but did not create a valid GLB: {output_path}")

        file_ref = {
            "filename": output_path.name,
            "subfolder": "",
            "type": "output",
        }
        print(f"[StableGen SkinTokens] Rigged GLB: {output_path}")
        return {"ui": {"rigged_glb": [file_ref]}, "result": (str(output_path),)}


class StableGenSkinTokensDoctor:
    """Report SkinTokens runtime status without installing anything."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "doctor"
    CATEGORY = "StableGen/3D"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return time.time()

    def doctor(self):
        status = doctor_text()
        print(status)
        return {"ui": {"text": [status]}, "result": (status,)}


NODE_CLASS_MAPPINGS = {
    "StableGenSkinTokensAutoRig": StableGenSkinTokensAutoRig,
    "StableGenSkinTokensDoctor": StableGenSkinTokensDoctor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableGenSkinTokensAutoRig": "StableGen SkinTokens AutoRig",
    "StableGenSkinTokensDoctor": "StableGen SkinTokens Doctor",
}
