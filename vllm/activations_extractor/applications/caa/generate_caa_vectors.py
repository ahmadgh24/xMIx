#!/usr/bin/env python3
"""Generate CAA steering vectors for a Hugging Face causal language model.

The input format matches nrimsky/CAA's ``generate_dataset.json`` files.  For
each contrast pair, this script records the decoder-layer state at the final
token of the supplied assistant answer and averages

    activation(answer_matching_behavior)
      - activation(answer_not_matching_behavior).

Vectors are saved on CPU in float32 so the xMIx application can load and cast
them for its inference model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_layers(value: str) -> list[int]:
    """Parse a comma-separated layer list, preserving order."""
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layer list contains duplicates")
    return layers


def completed_chat_answer_tokens(
    tokenizer: Any,
    question: str,
    answer: str,
) -> tuple[list[int], list[int]]:
    """Render a completed chat and locate all tokens overlapping its answer.

    Offsets are used instead of a fixed index because Qwen's chat template may
    append more than one control token after the assistant response.
    """
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    answer_start = rendered.rfind(answer)
    if answer_start < 0:
        raise ValueError(f"assistant answer {answer!r} is absent from chat template")
    answer_end = answer_start + len(answer)

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > answer_start and start < answer_end
    ]
    if not positions:
        raise ValueError(f"no tokens overlap assistant answer {answer!r}")
    return input_ids, positions


def answer_token_position(
    tokenizer: Any,
    question: str,
    answer: str,
) -> tuple[list[int], int]:
    """Return a completed chat and its final assistant-answer token index."""
    input_ids, positions = completed_chat_answer_tokens(
        tokenizer, question, answer
    )
    return input_ids, positions[-1]


def load_split(
    dataset_path: Path,
    train_size: int,
    validation_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[int], list[int]]:
    data = json.loads(dataset_path.read_text())
    required = {
        "question",
        "answer_matching_behavior",
        "answer_not_matching_behavior",
    }
    if not isinstance(data, list) or not data:
        raise ValueError("dataset must be a non-empty JSON list")
    for index, item in enumerate(data):
        missing = required.difference(item)
        if missing:
            raise ValueError(f"dataset item {index} is missing {sorted(missing)}")

    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if validation_size < 0:
        raise ValueError("validation_size cannot be negative")
    if train_size + validation_size > len(data):
        raise ValueError(
            f"requested {train_size + validation_size} examples from "
            f"a dataset containing {len(data)}"
        )

    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    train_indices = indices[:train_size]
    validation_indices = indices[train_size : train_size + validation_size]
    return data, train_indices, validation_indices


def generate_vectors(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, train_indices, validation_indices = load_split(
        dataset_path,
        args.train_size,
        args.validation_size,
        args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.is_fast:
        raise RuntimeError("a fast tokenizer is required for answer-token offsets")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    invalid = [layer for layer in args.layers if not 0 <= layer < num_layers]
    if invalid:
        raise ValueError(
            f"layers {invalid} are outside model range 0..{num_layers - 1}"
        )
    hidden_size = int(model.config.hidden_size)
    sums = {
        layer: torch.zeros(hidden_size, dtype=torch.float64)
        for layer in args.layers
    }

    for start in tqdm(
        range(0, len(train_indices), args.batch_size),
        desc="Extracting contrast pairs",
    ):
        batch_indices = train_indices[start : start + args.batch_size]
        sequences: list[list[int]] = []
        answer_positions: list[int] = []
        for index in batch_indices:
            item = data[index]
            for key in (
                "answer_matching_behavior",
                "answer_not_matching_behavior",
            ):
                input_ids, answer_position = answer_token_position(
                    tokenizer,
                    item["question"],
                    item[key],
                )
                sequences.append(input_ids)
                answer_positions.append(answer_position)

        inputs = tokenizer.pad(
            {"input_ids": sequences},
            padding=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(args.device) for name, value in inputs.items()}
        row_indices = torch.arange(len(sequences), device=args.device)
        position_indices = torch.tensor(answer_positions, device=args.device)

        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        # hidden_states[0] is the embedding output.  hidden_states[layer + 1]
        # is the residual-stream state after that decoder layer.
        for layer in args.layers:
            selected = outputs.hidden_states[layer + 1][
                row_indices, position_indices
            ].float()
            pair_differences = selected[0::2] - selected[1::2]
            sums[layer] += pair_differences.sum(dim=0).double().cpu()

        del outputs, inputs

    manifest: dict[str, Any] = {
        "model": args.model,
        "behavior": args.behavior,
        "dataset": str(dataset_path),
        "seed": args.seed,
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "layers": args.layers,
        "hidden_size": hidden_size,
        "vectors": {},
    }
    for layer in args.layers:
        vector = (sums[layer] / len(train_indices)).float()
        norm = float(torch.linalg.vector_norm(vector))
        if not math.isfinite(norm) or norm == 0.0:
            raise RuntimeError(f"layer {layer} produced invalid vector norm {norm}")
        artifact = {
            "vector": vector,
            "model": args.model,
            "behavior": args.behavior,
            "layer": layer,
            "num_pairs": len(train_indices),
            "vector_norm": norm,
            "seed": args.seed,
        }
        filename = f"layer_{layer}.pt"
        torch.save(artifact, output_dir / filename)
        manifest["vectors"][str(layer)] = {
            "path": filename,
            "norm": norm,
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest["vectors"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--behavior", default="sycophancy")
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=parse_layers("6,9,12,15,18,21"),
    )
    parser.add_argument("--train-size", type=int, default=900)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    generate_vectors(build_parser().parse_args())
