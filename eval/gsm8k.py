from vllm import LLM
from vllm.sampling_params import SamplingParams
from verl.utils.reward_score.gsm8k import compute_score
from datasets import load_dataset


def evaluate_gsm8k_accuracy(predictions, ground_truths):
    """Evaluate GSM8K accuracy by comparing predicted numbers with ground truth"""
    scores = [compute_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(predictions) * 100


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
    """Run GSM8K evaluation on the specified model using HuggingFace datasets

    Args:
        model: Model path or HuggingFace model identifier
        split: Which split to use ("test" or "train")
        num_shots: Number of few-shot examples (0 for zero-shot, 1 for one-shot, etc.)
        revision: Specific model revision/branch to use (for HuggingFace models)
        llm: Optional LLM instance to use. If None, creates a new one.
    """
    # Initialize LLM with revision if provided (only if llm not passed)
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
        # Extract the final answer after ####
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

    # print(f"Prompt Example [0]:\n{prompts[0]}")
    # print(f"Expected GT [0]:\n{ground_truths[0]}")

    sampling_params = SamplingParams(temperature=temperature, max_tokens=512)
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    # print(f"Response Example [0]: {responses[0]}")

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
