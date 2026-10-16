"""ComfyUI nodes for StableGen SkinTokens auto-rigging."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
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

try:
    from comfy.utils import ProgressBar  # type: ignore
except Exception:  # pragma: no cover - only happens outside ComfyUI
    ProgressBar = None

try:
    from server import PromptServer  # type: ignore
except Exception:  # pragma: no cover - only happens outside ComfyUI
    PromptServer = None


class AnyType(str):
    """Wildcard ComfyUI type that accepts any upstream output type."""

    def __ne__(self, other: object) -> bool:
        return False


ANY = AnyType("*")
SUPPORTED_SUFFIXES = {".glb", ".obj", ".fbx"}
PROGRESS_EVENT = "stablegen_skintokens_progress"


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


class _SkinTokensProgress:
    """Publish standard ComfyUI progress plus a structured StableGen stage."""

    def __init__(self, node_id: Any, run_id: str):
        self.node_id = str(_scalar(node_id) or "")
        self.run_id = str(run_id or "")
        self.value = 0
        self.stage = "Preparing Auto-Rigging"
        try:
            self._bar = ProgressBar(100) if ProgressBar is not None else None
        except Exception:
            self._bar = None

    def update(self, value: float, stage: str, detail: str = "") -> None:
        requested_value = min(100, int(round(float(value))))
        if requested_value < self.value:
            return
        value = max(0, requested_value)
        stage = str(stage or self.stage)
        changed = value != self.value or stage != self.stage or bool(detail)
        self.value = value
        self.stage = stage
        if not changed:
            return

        if self._bar is not None:
            try:
                self._bar.update_absolute(value, 100)
            except Exception:
                pass

        if PromptServer is not None:
            try:
                PromptServer.instance.send_sync(
                    PROGRESS_EVENT,
                    {
                        "version": 1,
                        "node": self.node_id,
                        "run_id": self.run_id,
                        "value": value,
                        "max": 100,
                        "stage": stage,
                        "detail": str(detail or ""),
                    },
                )
            except Exception:
                pass


def _progress_from_output(
    line: str,
    *,
    use_transfer: bool,
    use_postprocess: bool,
) -> tuple[float, str, str] | None:
    """Translate SkinTokens/demo.py output into stable progress stages."""
    text = str(line or "").strip()
    lower = text.lower()
    if not text:
        return None

    if "bpy_server.py started" in lower:
        return 12, "Starting Blender Worker", "SkinTokens Blender worker launched"
    if "bpy_server is ready" in lower:
        return 18, "Loading SkinTokens Model", "Blender worker is ready"
    if "loading model:" in lower:
        return 25, "Loading SkinTokens Model", text
    if "[server] received load path" in lower:
        return 32, "Importing Source Mesh", "Loading the source asset"
    if "gltf import finished" in lower:
        return 38, "Preparing Rig Inference", text

    if "[stablegen progress] inference_start" in lower:
        return 40, "Generating Skeleton and Weights", "SkinTokens inference started"
    token_match = re.search(
        r"\[stablegen progress\] inference_tokens\s+(\d+)\s+(\d+)",
        lower,
    )
    if token_match:
        generated = int(token_match.group(1))
        maximum = max(1, int(token_match.group(2)))
        token_percent = max(0.0, min(100.0, generated * 100.0 / maximum))
        return (
            40 + token_percent * 0.36,
            "Generating Skeleton and Weights",
            f"Generated {generated}/{maximum} rig tokens",
        )
    if "[stablegen progress] inference_complete" in lower:
        if use_postprocess:
            return 78, "Post-Processing Skin Weights", "Rig inference completed"
        if use_transfer:
            return 84, "Transferring Skin Weights", "Rig inference completed"
        return 90, "Exporting Rigged GLB", "Rig inference completed"
    if "[stablegen progress] postprocess_start" in lower:
        return 78, "Post-Processing Skin Weights", "Skin post-processing started"
    if "[stablegen progress] postprocess_complete" in lower:
        if use_transfer:
            return 84, "Transferring Skin Weights", "Skin post-processing completed"
        return 90, "Exporting Rigged GLB", "Skin post-processing completed"
    if "[stablegen progress] transfer_start" in lower:
        return 85, "Transferring Skin Weights", "Transferring weights to the source mesh"
    if "[stablegen progress] transfer_complete" in lower:
        return 95, "Finalizing Rigged GLB", "Weight transfer and export completed"
    if "[stablegen progress] export_start" in lower:
        return 90, "Exporting Rigged GLB", "Rigged GLB export started"
    if "[stablegen progress] export_complete" in lower:
        return 97, "Finalizing Rigged GLB", "Rigged GLB export completed"

    tqdm_match = re.search(r"(?<!\d)(\d{1,3})%\|", text)
    if tqdm_match:
        percent = max(0, min(100, int(tqdm_match.group(1))))
        if percent >= 100:
            if use_transfer:
                return 80, "Transferring Skin Weights", "Rig inference completed"
            if use_postprocess:
                return 88, "Post-Processing Skin Weights", "Rig inference completed"
            return 92, "Exporting Rigged GLB", "Rig inference completed"
        return 40 + percent * 0.38, "Generating Skeleton and Weights", f"Inference {percent}%"

    if "post_bpy_payload(\"transfer\"" in lower or "post_bpy_payload('transfer'" in lower:
        return 80, "Transferring Skin Weights", "Transferring weights to the source mesh"
    if "postprocess" in lower and "traceback" not in lower:
        return 88, "Post-Processing Skin Weights", "Applying SkinTokens post-processing"
    if "starting gltf 2.0 export" in lower:
        return 92, "Exporting Rigged GLB", "Starting glTF export"
    if "primitives created" in lower:
        return 95, "Exporting Rigged GLB", text
    if "finished gltf 2.0 export" in lower:
        return 97, "Finalizing Rigged GLB", text
    return None


def _run_streamed(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
    progress: _SkinTokensProgress,
    use_transfer: bool,
    use_postprocess: bool,
) -> subprocess.CompletedProcess:
    """Run SkinTokens while forwarding stdout records as they arrive.

    A reader thread is used because Windows pipe reads are blocking. Reading a
    character at a time preserves tqdm's carriage-return records, while the
    main thread remains able to enforce the configured timeout.
    """
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
    )
    output_queue: queue.Queue[str | None] = queue.Queue()

    def _read_output() -> None:
        try:
            if process.stdout is None:
                return
            while True:
                character = process.stdout.read(1)
                if character == "":
                    break
                output_queue.put(character)
        finally:
            output_queue.put(None)

    reader = threading.Thread(
        target=_read_output,
        name="StableGenSkinTokensOutput",
        daemon=True,
    )
    reader.start()

    deadline = time.monotonic() + float(timeout)
    record = ""
    recent_records: list[str] = []
    reader_finished = False

    def _handle_record(raw_record: str) -> None:
        cleaned = raw_record.strip()
        if not cleaned:
            return
        print(f"[StableGen SkinTokens] {cleaned}", flush=True)
        recent_records.append(cleaned)
        del recent_records[:-80]
        update = _progress_from_output(
            cleaned,
            use_transfer=use_transfer,
            use_postprocess=use_postprocess,
        )
        if update:
            progress.update(*update)

    while not reader_finished:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise subprocess.TimeoutExpired(
                cmd,
                timeout,
                output="\n".join(recent_records),
            )
        try:
            item = output_queue.get(timeout=min(0.25, remaining))
        except queue.Empty:
            continue
        if item is None:
            reader_finished = True
            break
        if item in ("\r", "\n"):
            _handle_record(record)
            record = ""
        else:
            record += item

    _handle_record(record)
    return_code = process.wait()
    return subprocess.CompletedProcess(
        cmd,
        return_code,
        stdout="\n".join(recent_records),
    )


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
            "hidden": {
                "unique_id": "UNIQUE_ID",
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
        unique_id: Any = None,
    ):
        progress = _SkinTokensProgress(unique_id, output_prefix)
        progress.update(2, "Validating Auto-Rig Input")
        source_path = _resolve_input_path(input_glb) or _resolve_input_path(input_path)
        if not source_path:
            raise RuntimeError("No input GLB/OBJ/FBX path was provided to SkinTokens.")
        if not source_path.is_file():
            raise RuntimeError(f"Input 3D file does not exist on the ComfyUI host: {source_path}")
        if source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise RuntimeError(
                f"SkinTokens supports {', '.join(sorted(SUPPORTED_SUFFIXES))}; got {source_path.suffix}"
            )

        progress.update(5, "Preparing SkinTokens Runtime")
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
        progress.update(8, "Starting SkinTokens")
        try:
            completed = _run_streamed(
                cmd,
                cwd=str(root),
                env=env,
                timeout=timeout,
                progress=progress,
                use_transfer=_flag(use_transfer),
                use_postprocess=_flag(use_postprocess),
            )
        except subprocess.TimeoutExpired as exc:
            progress.update(progress.value, "Auto-Rigging Timed Out")
            raise RuntimeError(f"SkinTokens timed out after {timeout} seconds") from exc

        if completed.returncode != 0:
            progress.update(progress.value, "Auto-Rigging Failed")
            detail = "\n".join((completed.stdout or "").splitlines()[-16:])
            suffix = f"\n{detail}" if detail else ""
            raise RuntimeError(
                f"SkinTokens failed with exit code {completed.returncode}{suffix}"
            )
        progress.update(98, "Validating Rigged GLB")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"SkinTokens completed but did not create a valid GLB: {output_path}")

        file_ref = {
            "filename": output_path.name,
            "subfolder": "",
            "type": "output",
        }
        print(f"[StableGen SkinTokens] Rigged GLB: {output_path}")
        progress.update(100, "Auto-Rigging Complete")
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
