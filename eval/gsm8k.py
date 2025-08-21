from vllm import LLM
from vllm.sampling_params import SamplingParams
from verl.utils.reward_score.gsm8k import compute_score
from datasets import load_dataset
import re


def evaluate_gsm8k_accuracy(predictions, ground_truths):
    scores = [
        custom_flexible(pred, gt)
        # compute_score(pred, gt, "flexible")
        for pred, gt in zip(predictions, ground_truths)
    ]
    return sum(scores) / len(predictions) * 100


def custom_flexible(solution_str, ground_truth):
    # Find the first occurrence of "#### "
    marker_pos = solution_str.find("#### ")

    if marker_pos != -1:
        # Get from start up to marker position + 5 more characters
        end_pos = marker_pos + 10
        solution_str = solution_str[: min(len(solution_str), end_pos)]

    if len(solution_str) > 300:
        solution_str = solution_str[-300:]

    answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
    final_answer = None
    if len(answer) == 0:
        return False

    invalid_str = ["", "."]
    for final_answer in reversed(answer):
        if final_answer not in invalid_str:
            break
    return final_answer == ground_truth


def _build_few_shot_prefix(num_shots: int) -> str:
    if num_shots <= 0:
        return ""
    # Use training split examples as demonstrations
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    k = min(num_shots, len(train_ds))
    lines = []
    for i in range(k):
        ex = train_ds[i]
        ex_q = ex["question"].strip()
        ex_a = ex["answer"].strip()
        lines.append(f"Question: {ex_q}\nAnswer: {ex_a}\n")
    return "\n".join(lines) + "\n"


def run_gsm8k_evaluation(
    model: str,
    split: str = "test",
    num_shots: int = 0,
    temperature: float = 1.0,
    revision: str = None,
    llm: LLM = None,
):
    if llm is None:
        llm_kwargs = {"model": model}
        if revision:
            llm_kwargs["revision"] = revision
        llm = LLM(**llm_kwargs)

    print(f"Loading GSM8K dataset from HuggingFace (split: {split})...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    questions = []
    ground_truths = []

    for example in dataset:
        questions.append(example["question"])
        ground_truths.append(example["answer"].split("#### ")[-1].strip())

    print(f"Evaluating {len(questions)} GSM8K examples...")
    prompts = []
    instruction_following = (
        'Let\'s think step by step and output the final answer after "####".'
    )
    few_shot_prefix = _build_few_shot_prefix(num_shots)

    for question in questions:
        if num_shots == 0:
            prompts.append(f"{question} {instruction_following}")
        else:
            prompt = f"{few_shot_prefix}\nQuestion: {question}\n{instruction_following}\nAnswer: "
            prompts.append(prompt)

    print(f"Prompt Example [0]:\n{prompts[0]}")
    print(f"Expected GT [0]:\n{ground_truths[0]}")

    sampling_params = SamplingParams(temperature=temperature, max_tokens=512)
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    print(f"Response Example [0]: {responses[0]}")

    accuracy = evaluate_gsm8k_accuracy(responses, ground_truths)
    correct_count = int(accuracy * len(ground_truths) / 100)

    # Print results
    print(f"\n{'=' * 60}")
    print("GSM8K EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Model: {model}")
    print(f"Total examples: {len(ground_truths)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'=' * 60}")

    return {
        "model": model,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(ground_truths),
        "prompts": prompts,
        "responses": responses,
        "ground_truths": ground_truths,
    }
