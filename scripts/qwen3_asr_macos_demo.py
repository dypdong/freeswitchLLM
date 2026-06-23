#!/usr/bin/env python3
"""Minimal macOS-friendly Qwen3-ASR command-line demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any



def resolve_device(requested: str, torch_module: Any) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for_device(device: str, torch_module: Any) -> Any:
    if device == "cpu":
        return torch_module.float32
    return torch_module.float16


def result_to_dict(result: Any) -> dict[str, Any]:
    data = {
        "language": getattr(result, "language", None),
        "text": getattr(result, "text", None),
    }
    timestamps = getattr(result, "time_stamps", None)
    if timestamps is not None:
        data["time_stamps"] = timestamps
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR-1.7B locally on macOS for one audio file.")
    parser.add_argument("audio", help="Path or URL of the audio file to transcribe.")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="Model id or local model directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"], help="Inference device.")
    parser.add_argument("--language", default=None, help='Force language, e.g. "Chinese" or "English".')
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path to save JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependencies. Install them with: "
            "pip install -r requirements-qwen3-asr.txt"
        ) from exc

    device = resolve_device(args.device, torch)
    dtype = dtype_for_device(device, torch)

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    results = model.transcribe(audio=args.audio, language=args.language)
    payload = [result_to_dict(result) for result in results]

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
