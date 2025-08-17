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
from vllm import LLM


import re
from typing import List, Dict
from huggingface_hub import HfApi


def get_available_grpo_checkpoints(repo_id: str) -> List[Dict[str, str]]:
    api = HfApi()
    try:
        refs = api.list_repo_refs(repo_id, repo_type="model")
        branches = [ref.ref.replace("refs/heads/", "") for ref in refs.branches]
    except Exception as e:
        print(f"Error listing branches for {repo_id}: {e}")
        return []
    
    global_step_branches = []
    for branch in branches:
        # Look for branches named like "global-step-100"
        if "global-step-" in branch:
            match = re.search(r"global-step-(\d+)", branch)
            if match:
                step_num = int(match.group(1))
                global_step_branches.append(
                    {
                        "step": step_num,
                        "branch": branch,
                        "checkpoint_name": f"global_step_{step_num}",
                    }
                )

    # Sort by step number
    global_step_branches.sort(key=lambda x: x["step"])

    print(f"Found {len(global_step_branches)} GRPO checkpoints for {repo_id}")
    return global_step_branches


def get_grpo_model_variants(original_model: str, username: str = "josancamon"):
    match = re.search(r"Qwen3-(\d+(?:\.\d+)?[BM])", original_model, re.IGNORECASE)
    size = match.group(1).lower().replace("B", "b").replace("M", "m")

    base_model = f"qwen3-{size}"
    repo_id = f"{username}/{base_model}-grpo-gsm8k"
    print("get_grpo_model_variants grpo_repo_id:", repo_id)

    api = HfApi()
    try:
        refs = api.list_repo_refs(repo_id, repo_type="model")
        branches = [ref.ref.replace("refs/heads/", "") for ref in refs.branches]
    except:  # noqa: E722
        return []

    checkpoints = []
    for branch in branches:
        # Look for branches named like "global-step-100"
        if "global-step-" in branch:
            match = re.search(r"global-step-(\d+)", branch)
            if match:
                step_num = int(match.group(1))
                checkpoints.append(
                    {
                        "step": step_num,
                        "branch": branch,
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
                "revision": checkpoint_info["branch"],  # Use branch name as revision
                "step": checkpoint_info["step"],
            }
        )

    return variants


### ===== HELPER FUNCTIONS


def _load_existing_results(output_path: Path) -> dict:
    if not output_path.exists():
        return None
    with output_path.open("r") as f:
        return json.load(f)


def _save_results(results: dict, output_path: Path):
    """Save results to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")


def _model_already_evaluated(
    model_name: str, existing_results: dict, shots: list
) -> bool:
    """Check if a model has already been evaluated for all required shots"""
    if not existing_results or "models" not in existing_results:
        return False

    for model_result in existing_results["models"]:
        if model_result.get("model") == model_name:
            # Check if all required shots are present
            mmlu_shots = (
                model_result.get("benchmarks", {}).get("mmlu", {}).get("by_shot", {})
            )
            gsm8k_shots = (
                model_result.get("benchmarks", {}).get("gsm8k", {}).get("by_shot", {})
            )

            for shot in shots:
                shot_str = str(shot)
                if (
                    shot_str not in mmlu_shots
                    or shot_str not in gsm8k_shots
                    or "error" in mmlu_shots.get(shot_str, {})
                    or "error" in gsm8k_shots.get(shot_str, {})
                ):
                    return False
            return True
    return False


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
        # "Qwen/Qwen3-1.7B-base",
        # "Qwen/Qwen3-4B-base",
        # "Qwen/Qwen3-8B-base",
        # "Qwen/Qwen3-14B-base",
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
    parser.add_argument("--shots", nargs="*", type=int, default=[0])  # , 1, 2, 3, 4, 5
    args = parser.parse_args()

    output_path = Path(str(Path(__file__).parent / "results.json"))

    # Load existing results if they exist
    existing_results = _load_existing_results(output_path)

    if existing_results:
        print(f"Loading existing results from {output_path}")
        results = existing_results
        # Update with current run parameters if needed
        results["temperature"] = args.temperature
        results["shots"] = args.shots
    else:
        print("Starting fresh evaluation")
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

    model_list = model_list[1:2]

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

        # Skip if model already evaluated
        if _model_already_evaluated(model_name, existing_results, args.shots):
            print(f"Skipping {model_name} - already evaluated")
            continue

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
            # try:
            #     mmlu_pt = _prompt_type_from_num(MMLUPromptType, k)
            #     mmlu_res = run_mmlu_evaluation(  # TODO: MMLU should generate everything at the same time
            #         model_id,
            #         split="test",
            #         prompt_type=mmlu_pt,
            #         temperature=args.temperature,
            #         revision=revision,
            #         # llm=llm,
            #     )
            #     # Aggregate overall accuracy
            #     total_correct = sum(v["correct"] for v in mmlu_res.values())
            #     total_questions = sum(v["total"] for v in mmlu_res.values())
            #     overall_acc = (
            #         (total_correct / total_questions * 100)
            #         if total_questions > 0
            #         else 0.0
            #     )
            #     model_result["benchmarks"]["mmlu"]["by_shot"][str(k)] = {
            #         "accuracy": overall_acc,
            #         "correct": total_correct,
            #         "total": total_questions,
            #     }
            # except Exception as e:
            #     model_result["benchmarks"]["mmlu"]["by_shot"][str(k)] = {
            #         "error": str(e)
            #     }

            # GSM8K
            try:
                gsm_pt = _prompt_type_from_num(GSM8KPromptType, k)
                gsm_res = run_gsm8k_evaluation(
                    model_id,
                    split="test",
                    prompt_type=gsm_pt,
                    temperature=args.temperature,
                    revision=revision,
                    # llm=llm,
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

        # Remove any existing result for this model before adding the new one
        results["models"] = [
            m for m in results["models"] if m.get("model") != model_name
        ]
        results["models"].append(model_result)

        # Save results after each model
        _save_results(results, output_path)

    results["ended_at"] = int(time.time())
    _save_results(results, output_path)
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
