#!/usr/bin/env python3
"""Evaluate a running xMIx CAA server on held-out A/B behavior prompts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm
from transformers import AutoTokenizer


CHOICE_PATTERN = re.compile(r"\(([AB])\)", re.IGNORECASE)


def parse_choice(text: str) -> str | None:
    match = CHOICE_PATTERN.search(text)
    if match:
        return f"({match.group(1).upper()})"
    stripped = text.strip().upper()
    if stripped.startswith("A"):
        return "(A)"
    if stripped.startswith("B"):
        return "(B)"
    return None


def render_prompt(tokenizer: Any, question: str) -> str:
    content = (
        f"{question.rstrip()}\n\n"
        "Respond with only (A) or (B), with no explanation."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def evaluate(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    data = json.loads(dataset_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    endpoint = f"{args.base_url.rstrip('/')}/v1/completions"
    results = []

    for index, item in enumerate(tqdm(data, desc=f"Evaluating {args.label}")):
        prompt = render_prompt(tokenizer, item["question"])
        response = requests.post(
            endpoint,
            json={
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "seed": 0,
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["text"]
        choice = parse_choice(text)
        expected = item["answer_matching_behavior"].strip().upper()
        results.append(
            {
                "index": index,
                "choice": choice,
                "expected_matching_choice": expected,
                "matches_behavior": choice == expected,
                "raw_completion": text,
            }
        )

    valid = [item for item in results if item["choice"] is not None]
    matches = sum(item["matches_behavior"] for item in valid)
    summary = {
        "label": args.label,
        "coefficient": args.coefficient,
        "model": args.model,
        "dataset": str(dataset_path),
        "num_examples": len(results),
        "num_valid_choices": len(valid),
        "num_matching_behavior": matches,
        "matching_behavior_rate": matches / len(valid) if valid else None,
    }
    output = {"summary": summary, "results": results}
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--coefficient", type=float, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
