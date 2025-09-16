#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import time
import gc
import torch
import random
from pathlib import Path

# Allow importing from the project root
sys.path.append(str(Path(__file__).parent.parent))

from eval.mmlu import run_mmlu_evaluation
from eval.gsm8k import run_gsm8k_evaluation
from eval.math import run_math_evaluation
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


def get_model_checkpoints(repo_id: str):
    """Get all checkpoints (branches with global-step-*) for a given model repository."""
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
    return checkpoints


def get_grpo_model_variants(original_model: str, username: str = "josancamon"):
    match = re.search(r"Qwen3-(\d+(?:\.\d+)?[BM])", original_model, re.IGNORECASE)
    size = match.group(1).lower().replace("B", "b").replace("M", "m")

    base_model = f"qwen3-{size}"
    repo_id = f"{username}/{base_model}-grpo-gsm8k"
    print("get_grpo_model_variants grpo_repo_id:", repo_id)

    checkpoints = get_model_checkpoints(repo_id)
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


def _save_gsm8k_samples(
    model_name: str,
    num_shots: int,
    prompts: List[str],
    ground_truths: List[str],
    responses: List[str],
    output_dir: Path = None,
):
    """Save random samples of GSM8K evaluation results to a text file"""
    if output_dir is None:
        output_dir = Path(__file__).parent / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean model name for filename
    clean_model_name = model_name.replace("/", "_").replace("-", "_")
    filename = f"gsm8k_samples_{clean_model_name}_{num_shots}shot.txt"
    output_path = output_dir / filename

    # Select random samples (up to 10)
    num_samples = min(10, len(prompts))
    indices = random.sample(range(len(prompts)), num_samples)

    with output_path.open("w") as f:
        f.write(f"GSM8K Samples for {model_name} - {num_shots}-shot\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(
            f"Total samples: {num_samples} (randomly selected from {len(prompts)})\n\n"
        )

        for i, idx in enumerate(indices):
            f.write(f"Sample {i + 1}/{num_samples} (Index: {idx})\n")
            f.write(f"{'-' * 80}\n\n")

            f.write("PROMPT:\n")
            f.write(prompts[idx])
            f.write("\n\n")

            f.write("EXPECTED ANSWER:\n")
            f.write(ground_truths[idx])
            f.write("\n\n")

            f.write("MODEL RESPONSE:\n")
            f.write(responses[idx])
            f.write("\n\n")

            f.write(f"{'=' * 80}\n\n")

    print(f"Saved {num_samples} samples to {output_path}")


def cleanup_vllm_resources(llm=None):
    """Comprehensive cleanup to prevent OOM issues with vLLM"""
    try:
        # Delete the LLM instance if provided
        if llm is not None:
            # vLLM specific cleanup if available
            if hasattr(llm, "shutdown"):
                llm.shutdown()
            del llm

        # Force Python garbage collection
        gc.collect()

        # Clear CUDA cache multiple times to ensure memory is freed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

        # Additional aggressive garbage collection
        for _ in range(3):
            gc.collect()

        # Sleep briefly to allow memory to be released
        time.sleep(2)

        print("Memory cleanup completed")

        # Print memory stats if available
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            )

    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")


def _model_already_evaluated(
    model_name: str, existing_results: dict, shots: list, include_math: bool = False
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
            math_shots = (
                model_result.get("benchmarks", {}).get("math", {}).get("by_shot", {})
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

                # Check MATH only for 0-shot if include_math is True
                if include_math and shot == 0:
                    if shot_str not in math_shots or "error" in math_shots.get(
                        shot_str, {}
                    ):
                        return False
            return True
    return False


### ===== STARTS


def _default_models():
    return [
        # "Qwen/Qwen3-0.6B-base",
        # "Qwen/Qwen3-1.7B-base",
        # "Qwen/Qwen3-4B-base",
        # "Qwen/Qwen3-8B-base",
        # "Qwen/Qwen3-14B-base",
    ]


def _build_model_list(
    include_grpo: bool,
    grpo_only: bool,
    grpo_username: str = "josancamon",
    last_checkpoint_only: bool = False,
    additional_models: list = None,
):
    """Build list of models to evaluate based on arguments.

    Args:
        include_grpo: Whether to include GRPO model variants
        grpo_only: Whether to only evaluate GRPO models (exclude base models)
        grpo_username: HuggingFace username for GRPO models
        last_checkpoint_only: If True, only evaluate the last checkpoint (highest step) for each GRPO model
        additional_models: List of additional model IDs to evaluate
    """
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

            # If last_checkpoint_only, find the variant with the highest step
            if last_checkpoint_only and grpo_variants:
                max_step_variant = max(grpo_variants, key=lambda v: v["step"])
                grpo_variants = [max_step_variant]
                print(
                    f"Using only last checkpoint with step {max_step_variant['step']}"
                )

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
                print(model_entry)
                models.append(model_entry)

    # Process additional models
    if additional_models:
        for model_id in additional_models:
            print(f"\nProcessing additional model: {model_id}")

            # Check if this model has checkpoints
            checkpoints = get_model_checkpoints(model_id)

            if checkpoints:
                print(f"Found {len(checkpoints)} checkpoints for {model_id}")

                # If last_checkpoint_only, only use the highest step
                if last_checkpoint_only:
                    max_checkpoint = max(checkpoints, key=lambda x: x["step"])
                    checkpoints = [max_checkpoint]
                    print(
                        f"Using only last checkpoint with step {max_checkpoint['step']}"
                    )

                # Add each checkpoint as a separate model entry
                for checkpoint in checkpoints:
                    model_entry = {
                        "model_id": model_id,
                        "revision": checkpoint["branch"],
                        "display_name": f"{model_id.split('/')[-1]}-{checkpoint['checkpoint_name']}",
                        "base_model": model_id,
                        "checkpoint": checkpoint["checkpoint_name"],
                        "step": checkpoint["step"],
                    }
                    print(f"Adding checkpoint: {model_entry['display_name']}")
                    models.append(model_entry)
            else:
                # No checkpoints found, add the model as-is (main branch)
                print(f"No checkpoints found, using main branch")
                model_entry = {
                    "model_id": model_id,
                    "revision": None,
                    "display_name": model_id.split("/")[-1],
                    "base_model": model_id,
                    "checkpoint": "main",
                    "step": 0,
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
    parser.add_argument("--include-mmlu", action="store_true", default=False)
    parser.add_argument("--include-math", action="store_true", default=False)
    parser.add_argument("--keep-samples", action="store_true", default=True)
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
            "math_split": "validation",
            "models": [],
            "started_at": int(time.time()),
        }

    # Configure model list directly here
    model_list = _build_model_list(
        include_grpo=False,
        grpo_only=False,
        last_checkpoint_only=True,
        additional_models=[
            "josancamon/qwen3-8b-grpo-lr1e-6-bs512-flexible",
            # "josancamon/qwen3-14b-grpo-lr1e-6-bs512-flexible",
        ],
    )
    print("model_list", model_list)

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
        if _model_already_evaluated(
            model_name, existing_results, args.shots, args.include_math
        ):
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
                "math": {"by_shot": {}},
            },
            "started_at": int(time.time()),
        }

        # Load LLM for this model with proper error handling
        llm = None
        try:
            print(f"Loading LLM for {model_name}...")

            # Initialize vLLM with memory-efficient settings
            llm = LLM(
                model=model_id,
                revision=revision,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.5,
                trust_remote_code=True,
                dtype="bfloat16",
                enforce_eager=True,  # Disable CUDA graphs to save memory
            )

            print(f"LLM loaded successfully for {model_name}")

            for k in args.shots:
                # MMLU
                if args.include_mmlu and k == 0:
                    try:
                        mmlu_res = run_mmlu_evaluation(
                            model_id,
                            split="test",
                            num_shots=k,
                            temperature=args.temperature,
                            revision=revision,
                            llm=llm,
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
                        print(f"Error in MMLU evaluation for {k}-shot: {e}")

                # GSM8K
                try:
                    gsm_res = run_gsm8k_evaluation(
                        model_id,
                        split="test",
                        num_shots=k,
                        temperature=args.temperature,
                        revision=revision,
                        llm=llm,
                    )
                    model_result["benchmarks"]["gsm8k"]["by_shot"][str(k)] = {
                        "accuracy": gsm_res.get("accuracy"),
                        "correct": gsm_res.get("correct"),
                        "total": gsm_res.get("total"),
                    }

                    # Save samples if requested
                    if args.keep_samples and "prompts" in gsm_res:
                        _save_gsm8k_samples(
                            model_name=model_name,
                            num_shots=k,
                            prompts=gsm_res["prompts"],
                            ground_truths=gsm_res["ground_truths"],
                            responses=gsm_res["responses"],
                        )

                except Exception as e:
                    model_result["benchmarks"]["gsm8k"]["by_shot"][str(k)] = {
                        "error": str(e)
                    }
                    print(f"Error in GSM8K evaluation for {k}-shot: {e}")

                # MATH (only supports 0-shot)
                if args.include_math and k == 0:
                    try:
                        math_res = run_math_evaluation(
                            model_id,
                            split="validation",
                            temperature=args.temperature,
                            revision=revision,
                            llm=llm,
                        )
                        model_result["benchmarks"]["math"]["by_shot"][str(k)] = {
                            "accuracy": math_res.get("accuracy"),
                            "correct": math_res.get("correct"),
                            "total": math_res.get("total"),
                            "subject_stats": math_res.get("subject_stats"),
                        }
                    except Exception as e:
                        model_result["benchmarks"]["math"]["by_shot"][str(k)] = {
                            "error": str(e)
                        }
                        print(f"Error in MATH evaluation: {e}")

        except Exception as e:
            print(f"Error loading LLM for {model_name}: {e}")
            model_result["error"] = str(e)

        finally:
            # Always cleanup LLM resources
            print(f"Cleaning up resources for {model_name}...")
            cleanup_vllm_resources(llm)
            llm = None

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
    # TODO: evaluate 1.7B baseline on custom_flexible and strict
    # TODO: Evaluate trained custom_flexible and strict models on custom_flexible
    # TODO: Evaluate the 3 of them on MATH
    # TODO: ----- is the model til here getting better, consistently, and generalizing better with the appropiate rewards?
    # TODO: repeat for 4B models, and do for 8B single trained.
    # TODO: plot and explore.
