# ComfyUI-StableGen-SkinTokens

ComfyUI custom node wrapper for running SkinTokens/TokenRig auto-rigging from
StableGen TRELLIS.2 workflows.

This package intentionally does not install anything during ComfyUI startup.
On first node execution it can lazily set up SkinTokens in a separate managed
environment, or you can run the installer script yourself before using it.

## Nodes

- `StableGen SkinTokens AutoRig`
- `StableGen SkinTokens Doctor`

The node accepts a GLB/OBJ/FBX path from an upstream node, calls SkinTokens'
`demo.py` CLI, and returns a rigged GLB file in ComfyUI's output directory.

## Progress Reporting

The AutoRig node streams SkinTokens output while the subprocess is running and
publishes both standard ComfyUI progress updates and structured
`stablegen_skintokens_progress` WebSocket events. StableGen uses these events to
show runtime preparation, model loading, source import, rig inference, optional
skin post-processing, weight transfer, and GLB export as separate stages.

On first execution after an update, the runtime helper adds idempotent progress
print markers to upstream `demo.py`. The markers do not change inference or rig
results. If an unfamiliar upstream version cannot be patched, auto-rigging still
runs and falls back to the stages detectable from ordinary subprocess output.

## First-Use Install

The AutoRig node has an `install_policy` input:

- `auto` (default): use existing config/env if present, try the current ComfyUI
  Python only if it already has the required packages, otherwise create
  `vendor/skintokens_env`.
- `managed`: create/use and configure `vendor/skintokens_env`.
- `comfy`: explicitly install into and configure the current ComfyUI Python
  environment.
- `configured_only`: never install; fail if no config/env is present.

The default `auto` policy does not mutate ComfyUI's Python. The first real
AutoRig execution may take a long time because it can clone SkinTokens, install
PyTorch/dependencies, install `flash-attn`, and download model checkpoints.

You can also run the setup explicitly:

```bash
python install_skintokens.py --target auto
```

Or force the managed environment:

```bash
python install_skintokens.py --target managed
```

To check the runtime without installing:

```bash
python doctor_skintokens.py
```

## Manual Configuration

If SkinTokens is already installed elsewhere, set environment variables before
starting ComfyUI:

```bash
SKINTOKENS_ROOT=/path/to/SkinTokens
SKINTOKENS_PYTHON=/path/to/python
```

Optional:

```bash
SKINTOKENS_TIMEOUT=3600
SKINTOKENS_OUTPUT_DIR=/path/to/ComfyUI/output
```

Or copy `skintokens_config.example.json` to `skintokens_config.json` and edit
the paths. Local config is ignored by git.

During installation, the node writes `skintokens_config.json` as soon as it has
chosen a SkinTokens checkout and Python environment. If a later dependency or
model download step fails, the config remains with `"install_complete": false`.
Rerunning with `--target auto` reuses complete config/env paths. Rerunning with
`--target managed` or `--target comfy` refreshes that explicit target and flips
the flag to `true` once the runtime is fully ready.

Generated config stores node-local managed paths as relative paths such as
`vendor/skintokens_env`. External Python or SkinTokens paths are written as
absolute paths.

The wrapper expects that this works:

```bash
python demo.py --input example.glb --output rigged.glb --use_transfer
```

## Installer Notes

The installer follows the upstream SkinTokens requirements:

- Python >= 3.11
- CUDA-capable NVIDIA GPU with at least 14 GB VRAM
- PyTorch 2.7.0 / torchvision 0.22.0 / torchaudio 2.7.0
- `requirements.txt`
- `python download.py --model`

SkinTokens currently imports a few direct runtime packages that are not all
declared in upstream `requirements.txt`. The wrapper health-checks configured
runtimes before launching `demo.py` and installs supplemental packages such as
`scipy`, `requests`, and `tqdm` when needed.

Upstream recommends `flash-attn`, but Windows builds commonly fail. By default
`SKINTOKENS_ATTENTION_BACKEND=auto` uses PyTorch SDPA on Windows and only tries
FlashAttention on other platforms. SDPA is slower and may use more VRAM, but it
avoids compiling `flash-attn`. The SDPA patch removes upstream FlashAttention
imports from `tokenrig.py`, `skin_vae_model.py`, and the Hugging Face model
load path in `server/spec.py`.

Useful overrides:

```bash
SKINTOKENS_BOOTSTRAP_PYTHON=/path/to/python3.11
SKINTOKENS_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
SKINTOKENS_SKIP_TORCH_INSTALL=1
SKINTOKENS_ATTENTION_BACKEND=auto        # auto | flash_attention_2 | sdpa
SKINTOKENS_SKIP_FLASH_ATTN_INSTALL=1     # implies SDPA fallback unless flash_attention_2 is forced
SKINTOKENS_SKIP_MODEL_DOWNLOAD=1
```

If you already hit a `flash-attn` build error on Windows, update this node repo
and rerun:

```bash
python install_skintokens.py --target managed
```

The installer will reuse the existing `vendor/SkinTokens` checkout and
`vendor/skintokens_env`, skip the FlashAttention build, patch TokenRig to SDPA,
and continue with model download/config.

Upstream SkinTokens:

- https://github.com/VAST-AI-Research/SkinTokens
- https://huggingface.co/VAST-AI/SkinTokens

## StableGen Integration

StableGen detects this node through ComfyUI's `/object_info` endpoint. When the
node is available, StableGen can insert it after the TRELLIS.2 export node and
download the rigged GLB instead of the raw TRELLIS export.

## License

GPL-3.0-or-later, matching StableGen. SkinTokens itself is MIT licensed and is
not redistributed by this package.
