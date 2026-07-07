# ComfyUI-StableGen-SkinTokens

ComfyUI custom node wrapper for running SkinTokens/TokenRig auto-rigging from
StableGen TRELLIS.2 workflows.

This package intentionally does not vendor SkinTokens, PyTorch, Blender Python,
or model weights. Install SkinTokens in a separate Python environment on the
same machine that runs ComfyUI, then point this node at that environment.

## Nodes

- `StableGen SkinTokens AutoRig`

The node accepts a GLB/OBJ/FBX path from an upstream node, calls SkinTokens'
`demo.py` CLI, and returns a rigged GLB file in ComfyUI's output directory.

## Configuration

Set these environment variables before starting ComfyUI:

```bash
SKINTOKENS_ROOT=/path/to/SkinTokens
SKINTOKENS_PYTHON=/path/to/skintokens-venv/bin/python
```

Optional:

```bash
SKINTOKENS_TIMEOUT=3600
SKINTOKENS_OUTPUT_DIR=/path/to/ComfyUI/output
```

On Windows, `SKINTOKENS_PYTHON` usually points to:

```powershell
C:\path\to\skintokens-venv\Scripts\python.exe
```

## SkinTokens Install Sketch

Follow the upstream SkinTokens instructions for the exact CUDA/PyTorch build:

- https://github.com/VAST-AI-Research/SkinTokens
- https://huggingface.co/VAST-AI/SkinTokens

The wrapper expects that this works from `SKINTOKENS_ROOT`:

```bash
python demo.py --input example.glb --output rigged.glb --use_transfer
```

## StableGen Integration

StableGen detects this node through ComfyUI's `/object_info` endpoint. When the
node is available, StableGen can insert it after the TRELLIS.2 export node and
download the rigged GLB instead of the raw TRELLIS export.

## License

GPL-3.0-or-later, matching StableGen. SkinTokens itself is MIT licensed and is
not redistributed by this package.
