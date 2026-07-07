"""Install or configure SkinTokens for ComfyUI-StableGen-SkinTokens.

Usage:
    python install_skintokens.py --target auto
    python install_skintokens.py --target managed
    python install_skintokens.py --target comfy

The default `auto` target reuses a configured runtime, tries the current Python
if it already has the SkinTokens dependencies, and otherwise creates a managed
environment under vendor/skintokens_env.
"""

from __future__ import annotations

import argparse

from skintokens_runtime import INSTALL_POLICIES, doctor_text, ensure_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=INSTALL_POLICIES,
        default="auto",
        help="Runtime install target. Use 'comfy' only if you intentionally want to install into this Python.",
    )
    args = parser.parse_args()

    root, python_exe, timeout = ensure_runtime(args.target)
    print()
    print("SkinTokens runtime is ready.")
    print(f"Root: {root}")
    print(f"Python: {python_exe}")
    print(f"Timeout: {timeout}")
    print()
    print(doctor_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
