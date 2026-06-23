#!/usr/bin/env python3
"""Command-line demo for local Qwen3-ASR-1.7B transcription.

The script uses the official qwen-asr package and can run against either a
Hugging Face / ModelScope model id or a pre-downloaded local model directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe one or more audio files with Qwen3-ASR-1.7B."
    )
    parser.add_argument(
        "audio",
        nargs="+",
        help="Audio file path, URL, or base64 audio item accepted by qwen-asr.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model id or local checkpoint directory. Defaults to Qwen/Qwen3-ASR-1.7B.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help='Force a language such as "Chinese" or "English"; omit for auto-detect.',
    )
    parser.add_argument(
        "--device-map",
        default="cuda:0",
        help='Device map passed to qwen-asr, for example "cuda:0", "auto", or "cpu".',
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens. Increase for long audio.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum inference batch size. Lower this if the GPU runs out of memory.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Enable forced alignment timestamps with Qwen3-ForcedAligner-0.6B.",
    )
    parser.add_argument(
        "--aligner-model",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Forced aligner model id or local directory used when --timestamps is set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file path for transcription results.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def result_to_dict(result: Any, audio: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "audio": audio,
        "language": getattr(result, "language", None),
        "text": getattr(result, "text", ""),
    }
    timestamps = getattr(result, "time_stamps", None)
    if timestamps is not None:
        item["time_stamps"] = timestamps
    return item


def main() -> None:
    args = parse_args()

    from qwen_asr import Qwen3ASRModel

    dtype = torch_dtype(args.dtype)

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": args.device_map,
        "max_inference_batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.timestamps:
        model_kwargs["forced_aligner"] = args.aligner_model
        model_kwargs["forced_aligner_kwargs"] = {
            "dtype": dtype,
            "device_map": args.device_map,
        }

    model = Qwen3ASRModel.from_pretrained(args.model, **model_kwargs)
    language = [args.language] * len(args.audio) if args.language else None
    results = model.transcribe(
        audio=args.audio,
        language=language,
        return_time_stamps=args.timestamps,
    )
    payload = [result_to_dict(result, audio) for result, audio in zip(results, args.audio)]

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
