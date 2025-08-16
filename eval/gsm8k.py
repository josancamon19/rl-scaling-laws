from vllm import LLM
from vllm.sampling_params import SamplingParams
from verl.utils.reward_score.gsm8k import compute_score
from datasets import load_dataset


def evaluate_gsm8k_accuracy(predictions, ground_truths):
    """Evaluate GSM8K accuracy by comparing predicted numbers with ground truth"""
    scores = [compute_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(predictions) * 100


def run_gsm8k_evaluation(model: str, split: str = "validation"):
    """Run GSM8K evaluation on the specified model using HuggingFace datasets
    
    Args:
        model: Model path or HuggingFace model identifier
        split: Which split to use ("validation" or "train")
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
    for question in questions:
        question += " " + instruction_following
        prompts.append(question)

    sampling_params = SamplingParams(temperature=1.0, max_tokens=512)
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

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