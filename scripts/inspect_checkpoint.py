
#!/usr/bin/env python3
"""Print a compact, non-destructive checkpoint summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if k != "model_state_dict"}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    summary = _safe(payload)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
        summary["model_state_dict_summary"] = {
            "tensor_count": len(state),
            "parameter_count": int(sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
