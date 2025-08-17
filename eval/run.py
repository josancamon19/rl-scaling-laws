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


import re
from typing import List, Dict
from huggingface_hub import HfApi


def get_available_grpo_checkpoints(repo_id: str) -> List[Dict[str, str]]:
    api = HfApi()
    commits = list(api.list_repo_commits(repo_id, repo_type="model"))
    global_step_commits = []
    for commit in commits:
        message = commit.title
        if "global_step_" in message:
            match = re.search(r"global_step_(\d+)", message)
            if match:
                step_num = int(match.group(1))
                global_step_commits.append(
                    {
                        "step": step_num,
                        "commit_id": commit.commit_id,
                        "message": message,
                        "checkpoint_name": f"global_step_{step_num}",
                    }
                )

    # Sort by step number
    global_step_commits.sort(key=lambda x: x["step"])

    print(f"Found {len(global_step_commits)} GRPO checkpoints for {repo_id}")
    return global_step_commits


def get_grpo_model_variants(original_model: str, username: str = "josancamon"):
    match = re.search(r"Qwen3-(\d+(?:\.\d+)?[BM])", original_model, re.IGNORECASE)
    size = match.group(1).lower().replace("B", "b").replace("M", "m")

    base_model = f"qwen3-{size}"
    repo_id = f"{username}/{base_model}-grpo-gsm8k"
    print("get_grpo_model_variants grpo_repo_id:", repo_id)

    api = HfApi()
    try:
        commits = list(api.list_repo_commits(repo_id, repo_type="model"))
    except:  # noqa: E722
        return []

    checkpoints = []
    for commit in commits:
        message = commit.title
        if "global_step_" in message:
            match = re.search(r"global_step_(\d+)", message)
            if match:
                step_num = int(match.group(1))
                checkpoints.append(
                    {
                        "step": step_num,
                        "commit_id": commit.commit_id,
                        "message": message,
                        "checkpoint_name": f"global_step_{step_num}",
                    }
                )

    checkpoints.sort(key=lambda x: x["step"])
    print(f"Found {len(checkpoints)} GRPO checkpoints for {repo_id}")

    variants = []
    for checkpoint_info in checkpoints:
        variants.append(
            {
                "model_id": repo_id,
                "checkpoint": checkpoint_info["checkpoint_name"],
                "base_model": base_model,
                "revision": checkpoint_info["commit_id"],  # Use commit ID as revision
                "step": checkpoint_info["step"],
            }
        )

    return variants


### ===== STARTS


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


def _build_model_list(
    include_grpo: bool,
    grpo_only: bool,
    grpo_username: str,
    max_grpo_checkpoints: int = None,
):
    """Build list of models to evaluate based on arguments."""
    models = []
    base_models = _default_models()

    if not grpo_only:
        # Add base models if not grpo-only mode
        models.extend(base_models)

    if include_grpo or grpo_only:
        # Add GRPO variants for each base model
        for base_model in base_models:
            print(f"Finding GRPO variants for {base_model}...")
            grpo_variants = get_grpo_model_variants(base_model, grpo_username)
            print("found grpo_variants", len(grpo_variants))

            # Limit number of checkpoints if specified
            if max_grpo_checkpoints and len(grpo_variants) > max_grpo_checkpoints:
                print(
                    f"Limiting to {max_grpo_checkpoints} checkpoints for {base_model}"
                )
                grpo_variants = grpo_variants[:max_grpo_checkpoints]

            for variant in grpo_variants:
                # Create a model entry that includes revision info
                model_entry = {
                    "model_id": variant["model_id"],
                    "revision": variant["revision"],
                    "display_name": f"{variant['base_model']}-grpo-{variant['checkpoint']}",
                    "base_model": variant["base_model"],
                    "checkpoint": variant["checkpoint"],
                    "step": variant["step"],
                }
                models.append(model_entry)

    return models


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on language models")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--shots", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    args = parser.parse_args()

    results = {
        "temperature": args.temperature,
        "shots": args.shots,
        "mmlu_split": "test",
        "gsm8k_split": "test",
        "models": [],
        "started_at": int(time.time()),
    }

    model_list = _build_model_list(
        include_grpo=True,
        grpo_only=False,
        grpo_username="josancamon",
        max_grpo_checkpoints=None,
    )

    for model_item in model_list:
        # Handle both simple string models and complex model entries
        if isinstance(model_item, str):
            model_id = model_item
            model_name = model_item
            revision = None
            model_metadata = {}
        else:
            model_id = model_item["model_id"]
            model_name = model_item.get("display_name", model_id)
            revision = model_item.get("revision")
            model_metadata = {
                "base_model": model_item.get("base_model"),
                "checkpoint": model_item.get("checkpoint"),
                "revision": revision,
            }

        print(f"Evaluating model: {model_name}")
        if revision:
            print(f"  Using revision: {revision}")

        model_result = {
            "model": model_name,
            "model_id": model_id,
            "metadata": model_metadata,
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
                    model_id,
                    split="test",
                    prompt_type=mmlu_pt,
                    temperature=args.temperature,
                    revision=revision,
                )
                # Aggregate overall accuracy
                total_correct = sum(v["correct"] for v in mmlu_res.values())
                total_questions = sum(v["total"] for v in mmlu_res.values())
                overall_acc = (
                    (total_correct / total_questions * 100)
                    if total_questions > 0
                    else 0.0
                )
                model_result["benchmarks"]["mmlu"]["by_shot"][str(k)] = {
                    "accuracy": overall_acc,
                    "correct": total_correct,
                    "total": total_questions,
                }
            except Exception as e:
                model_result["benchmarks"]["mmlu"]["by_shot"][str(k)] = {
                    "error": str(e)
                }

            # GSM8K
            try:
                gsm_pt = _prompt_type_from_num(GSM8KPromptType, k)
                gsm_res = run_gsm8k_evaluation(
                    model_id,
                    split="test",
                    prompt_type=gsm_pt,
                    temperature=args.temperature,
                    revision=revision,
                )
                model_result["benchmarks"]["gsm8k"]["by_shot"][str(k)] = {
                    "accuracy": gsm_res.get("accuracy"),
                    "correct": gsm_res.get("correct"),
                    "total": gsm_res.get("total"),
                }
            except Exception as e:
                model_result["benchmarks"]["gsm8k"]["by_shot"][str(k)] = {
                    "error": str(e)
                }

        model_result["ended_at"] = int(time.time())
        results["models"].append(model_result)

    results["ended_at"] = int(time.time())

    output_path = Path(str(Path(__file__).parent / "results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
