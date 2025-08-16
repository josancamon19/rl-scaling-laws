from vllm import LLM
from vllm.sampling_params import SamplingParams
from verl.utils.reward_score.gsm8k import compute_score
from datasets import load_dataset
from enum import Enum
import re


def evaluate_gsm8k_accuracy(predictions, ground_truths):
    """Evaluate GSM8K accuracy by comparing predicted numbers with ground truth"""
    scores = [compute_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(predictions) * 100


class PromptType(str, Enum):
    zero_shot = "zero_shot"
    one_shot = "one_shot"
    two_shot = "two_shot"
    three_shot = "three_shot"
    four_shot = "four_shot"
    five_shot = "five_shot"


def _num_shots_from_prompt_type(prompt_type: "PromptType | str") -> int:
    mapping = {
        "zero_shot": 0,
        "one_shot": 1,
        "two_shot": 2,
        "three_shot": 3,
        "four_shot": 4,
        "five_shot": 5,
    }
    if isinstance(prompt_type, PromptType):
        value = prompt_type.value
    else:
        value = str(prompt_type)
    if value in mapping:
        return mapping[value]
    m = re.match(r"^(\d+)_shot$", value)
    if m:
        return int(m.group(1))
    raise ValueError(f"Unrecognized prompt_type: {prompt_type}")


def _build_few_shot_prefix(num_shots: int, instruction_following: str) -> str:
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
        lines.append(f"Q: {ex_q} {instruction_following}\nA: {ex_a}\n")
    return "\n".join(lines) + "\n"


def run_gsm8k_evaluation(
    model: str,
    split: str = "test",
    prompt_type: PromptType = PromptType.zero_shot,
    temperature: float = 1.0,
):
    """Run GSM8K evaluation on the specified model using HuggingFace datasets

    Args:
        model: Model path or HuggingFace model identifier
        split: Which split to use ("test" or "train")
        prompt_type: Prompt construction style: zero_shot, one_shot, two_shot, ...
    """
    llm = LLM(model=model)

    print(f"Loading GSM8K dataset from HuggingFace (split: {split})...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    questions = []
    ground_truths = []

    for example in dataset:
        questions.append(example["question"])
        # Extract the final answer after ####
        ground_truths.append(example["answer"].split("#### ")[-1].strip())

    print(f"Evaluating {len(questions)} GSM8K examples...")

    prompts = []
    instruction_following = (
        'Let\'s think step by step and output the final answer after "####".'
    )
    num_shots = _num_shots_from_prompt_type(prompt_type)
    few_shot_prefix = _build_few_shot_prefix(num_shots, instruction_following)

    for question in questions:
        if num_shots == 0:
            prompts.append(f"{question} {instruction_following}")
        else:
            prompt = f"{few_shot_prefix}Q: {question} {instruction_following}\nA:"
            prompts.append(prompt)

    print(f"Prompt Example [0]:\n{prompts[0]}")
    print(f"Ground Truth [0]:\n{ground_truths[0]}")

    sampling_params = SamplingParams(temperature=temperature, max_tokens=512)
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    print(f"Response Example [0]:\n{responses[0]}")

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
        "responses": responses,
        "ground_truths": ground_truths,
    }
