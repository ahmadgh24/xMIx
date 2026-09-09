#!/usr/bin/env python3
"""Select a CAA layer and coefficient on the reserved validation split.

The score is the teacher-forced log-likelihood margin between the answer that
matches the behavior and the answer that does not.  The official test split is
not read by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_caa_vectors import completed_chat_answer_tokens


def parse_numbers(value: str, cast: Any) -> list[Any]:
    result = [cast(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return result


def build_examples(
    tokenizer: Any,
    data: list[dict[str, str]],
    indices: list[int],
) -> list[tuple[list[int], list[int]]]:
    examples = []
    for index in indices:
        item = data[index]
        for key in (
            "answer_matching_behavior",
            "answer_not_matching_behavior",
        ):
            examples.append(
                completed_chat_answer_tokens(
                    tokenizer,
                    item["question"],
                    item[key],
                )
            )
    return examples


def score_setting(
    model: Any,
    tokenizer: Any,
    examples: list[tuple[list[int], list[int]]],
    layer: int,
    coefficient: float,
    vector: torch.Tensor,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    def add_direction(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            return (output[0] + coefficient * vector,) + output[1:]
        return output + coefficient * vector

    handle = model.model.layers[layer].register_forward_hook(add_direction)
    all_scores = []
    try:
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            sequences = [example[0] for example in chunk]
            answer_positions = [example[1] for example in chunk]
            inputs = tokenizer.pad(
                {"input_ids": sequences},
                padding=True,
                return_tensors="pt",
            )
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                logits = model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                ).logits

            for row, positions in enumerate(answer_positions):
                prediction_positions = torch.tensor(
                    [position - 1 for position in positions],
                    device=device,
                )
                target_ids = inputs["input_ids"][
                    row,
                    torch.tensor(positions, device=device),
                ]
                answer_logits = logits[row, prediction_positions]
                token_log_probs = answer_logits.gather(
                    1, target_ids.unsqueeze(1)
                ).squeeze(1) - torch.logsumexp(answer_logits, dim=1)
                all_scores.append(float(token_log_probs.sum()))
            del logits, inputs
    finally:
        handle.remove()

    matching_scores = torch.tensor(all_scores[0::2])
    nonmatching_scores = torch.tensor(all_scores[1::2])
    margins = matching_scores - nonmatching_scores
    return {
        "matching_behavior_rate": float((margins > 0).float().mean()),
        "mean_log_likelihood_margin": float(margins.mean()),
        "median_log_likelihood_margin": float(margins.median()),
    }


def select(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    data = json.loads(dataset_path.read_text())
    validation_indices = manifest["validation_indices"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    examples = build_examples(tokenizer, data, validation_indices)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    results = []
    settings = [
        (layer, coefficient)
        for layer in args.layers
        for coefficient in args.coefficients
    ]
    for layer, coefficient in tqdm(settings, desc="Scoring settings"):
        artifact = torch.load(
            artifact_dir / f"layer_{layer}.pt",
            map_location="cpu",
            weights_only=True,
        )
        vector = artifact["vector"].to(
            device=args.device,
            dtype=torch.float16,
        )
        metrics = score_setting(
            model,
            tokenizer,
            examples,
            layer,
            coefficient,
            vector,
            args.batch_size,
            args.device,
        )
        results.append(
            {
                "layer": layer,
                "coefficient": coefficient,
                "vector_norm": artifact["vector_norm"],
                **metrics,
            }
        )

    positive_results = [result for result in results if result["coefficient"] > 0]
    selected = max(
        positive_results,
        key=lambda result: (
            result["matching_behavior_rate"],
            result["mean_log_likelihood_margin"],
        ),
    )
    output = {
        "model": args.model,
        "dataset": str(dataset_path),
        "num_validation_examples": len(validation_indices),
        "selection_rule": (
            "highest matching_behavior_rate, then highest mean log-likelihood "
            "margin, among positive coefficients"
        ),
        "selected": selected,
        "results": results,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--layers",
        type=lambda value: parse_numbers(value, int),
        default=parse_numbers("6,9,12,15,18,21", int),
    )
    parser.add_argument(
        "--coefficients",
        type=lambda value: parse_numbers(value, float),
        default=parse_numbers("-5,-2,-1,-0.5,0,0.5,1,2,5", float),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    select(build_parser().parse_args())
