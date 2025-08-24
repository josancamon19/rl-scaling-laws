import json
import re
from vllm import LLM
from vllm.sampling_params import SamplingParams
from pathlib import Path
from utils.custom_gsm8k_reward import compute_score


def evaluate_math_accuracy(predictions, ground_truths):
    scores = [
        compute_score(None, pred, gt, {"method": "custom_flexible"})
        for pred, gt in zip(predictions, ground_truths)
    ]
    return sum(scores) / len(predictions) * 100 if predictions else 0


def run_math_evaluation(
    model: str,
    split: str = "validation",
    temperature: float = 1.0,
    revision: str = None,
    llm: LLM = None,
):
    if llm is None:
        llm_kwargs = {"model": model}
        if revision:
            llm_kwargs["revision"] = revision
        llm = LLM(**llm_kwargs)

    # Load MATH dataset
    dataset_path = Path(f"/workspace/rl-scaling-laws/data/MATH/{split}.jsonl")
    print(f"Loading MATH dataset from {dataset_path}...")

    problems = []
    ground_truths = []
    subjects = []

    with open(dataset_path, "r") as f:
        for line in f:
            # if len(problems) >= 1000:
            #     # too many samples, 1k is alright.
            #     break
            data = json.loads(line)
            problems.append(data["problem"])
            ground_truths.append(data["answer"])
            subjects.append(data.get("subject", "Unknown"))

    print(f"Evaluating {len(problems)} MATH examples...")

    # Prepare prompts
    prompts = []
    instruction = 'Let\'s think step by step and output the final answer after "####".'

    for problem in problems:
        prompt = f"{problem}\n\n{instruction}"
        prompts.append(prompt)

    # Show example prompt
    print(f"\nPrompt Example [0]:\n{prompts[0]}")
    print(f"\nExpected GT [0]: {ground_truths[0]}")

    # Generate responses
    sampling_params = SamplingParams(temperature=temperature, max_tokens=1024)
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    print(f"\nResponse Example [0]:\n{responses[0]}")

    # TODO: this is probably not great, there's some cleaning to be done, cause question has latex, thus it might use latex for responding.
    accuracy = evaluate_math_accuracy(responses, ground_truths)
    correct_count = int(accuracy * len(ground_truths) / 100)

    # Calculate accuracy by subject
    subject_stats = {}
    for i, subject in enumerate(subjects):
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}

        subject_stats[subject]["total"] += 1
        score = compute_score(
            None, responses[i], ground_truths[i], {"method": "custom_flexible"}
        )
        if score == 1:
            subject_stats[subject]["correct"] += 1

    # Print results
    print(f"\n{'=' * 60}")
    print("MATH EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Model: {model}")
    print(f"Total examples: {len(ground_truths)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'=' * 60}")

    # Print per-subject breakdown
    print("\nPer-subject results:")
    for subject, stats in sorted(subject_stats.items()):
        subject_acc = stats["correct"] / stats["total"] * 100
        print(
            f"{subject:20}: {stats['correct']:3}/{stats['total']:3} ({subject_acc:6.2f}%)"
        )

    return {
        "model": model,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(ground_truths),
        "subject_stats": subject_stats,
        "prompts": prompts,
        "responses": responses,
        "ground_truths": ground_truths,
    }
