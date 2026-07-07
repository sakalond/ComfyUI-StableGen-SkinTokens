"""Print SkinTokens runtime diagnostics for ComfyUI-StableGen-SkinTokens."""

from __future__ import annotations

from skintokens_runtime import doctor_text


def main() -> int:
    print(doctor_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
