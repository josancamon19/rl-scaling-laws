#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow importing from the project root
sys.path.append(str(Path(__file__).parent.parent))

from eval.mmlu import run_mmlu_evaluation, PromptType as MMLUPromptType
from eval.gsm8k import run_gsm8k_evaluation, PromptType as GSM8KPromptType


def _prompt_type_from_num(enum_cls, k: int):
    mapping = {
        0: "zero_shot",
        1: "one_shot",
        2: "two_shot",
        3: "three_shot",
        4: "four_shot",
        5: "five_shot",
    }
    return enum_cls(mapping.get(k, "zero_shot"))


def _default_models():
    return [
        "Qwen/Qwen3-0.6B-base",
        "Qwen/Qwen3-1.7B-base",
        "Qwen/Qwen3-4B-base",
        "Qwen/Qwen3-8B-base",
        "Qwen/Qwen3-14B-base",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=_default_models(),
        help="List of model identifiers to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for both benchmarks",
    )
    parser.add_argument(
        "--shots",
        nargs="*",
        type=int,
        default=[0],
        help="List of shot counts to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "results.json"),
        help="Path to write aggregated JSON results",
    )
    parser.add_argument(
        "--mmlu-split",
        type=str,
        default="test",
        help="Split to use for MMLU (validation, test, dev)",
    )
    parser.add_argument(
        "--gsm8k-split",
        type=str,
        default="test",
        help="Split to use for GSM8K (test or train)",
    )
    args = parser.parse_args()

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("vllm").setLevel(logging.ERROR)

    results = {
        "temperature": args.temperature,
        "shots": args.shots,
        "mmlu_split": args.mmlu_split,
        "gsm8k_split": args.gsm8k_split,
        "models": [],
        "started_at": int(time.time()),
    }

    for model in args.models:
        model_entry = {
            "model": model,
            "benchmarks": {
                "mmlu": {"by_shot": {}},
                "gsm8k": {"by_shot": {}},
            },
            "started_at": int(time.time()),
        }

        for k in args.shots:
            # MMLU
            try:
                mmlu_pt = _prompt_type_from_num(MMLUPromptType, k)
                mmlu_res = run_mmlu_evaluation(
                    model,
                    split=args.mmlu_split,
                    prompt_type=mmlu_pt,
                    temperature=args.temperature,
                )
                # Aggregate overall accuracy
                total_correct = sum(v["correct"] for v in mmlu_res.values())
                total_questions = sum(v["total"] for v in mmlu_res.values())
                overall_acc = (
                    (total_correct / total_questions * 100)
                    if total_questions > 0
                    else 0.0
                )
                model_entry["benchmarks"]["mmlu"]["by_shot"][str(k)] = {
                    "accuracy": overall_acc,
                    "correct": total_correct,
                    "total": total_questions,
                }
            except Exception as e:
                model_entry["benchmarks"]["mmlu"]["by_shot"][str(k)] = {"error": str(e)}

            # GSM8K
            try:
                gsm_pt = _prompt_type_from_num(GSM8KPromptType, k)
                gsm_res = run_gsm8k_evaluation(
                    model,
                    split=args.gsm8k_split,
                    prompt_type=gsm_pt,
                    temperature=args.temperature,
                )
                model_entry["benchmarks"]["gsm8k"]["by_shot"][str(k)] = {
                    "accuracy": gsm_res.get("accuracy"),
                    "correct": gsm_res.get("correct"),
                    "total": gsm_res.get("total"),
                }
            except Exception as e:
                model_entry["benchmarks"]["gsm8k"]["by_shot"][str(k)] = {
                    "error": str(e)
                }

        model_entry["ended_at"] = int(time.time())
        results["models"].append(model_entry)

    results["ended_at"] = int(time.time())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
