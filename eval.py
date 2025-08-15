import os
import csv
import glob
import json
import gc
import torch
from vllm import LLM
import re
from vllm.sampling_params import SamplingParams
from verl.utils.reward_score.gsm8k import compute_score


def create_mmlu_instruction(subject, question, choices):
    formatted = f"{question}\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    return f'Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n{formatted}'


def extract_answer_letter(response):
    """Extract the answer letter from the model response"""
    # Look for patterns like "The correct answer is A" or just "A" at the end
    patterns = [
        r"[Tt]he correct answer is ([ABCD])",
        r"([ABCD])\s*\.?\s*$",
        r"Answer:\s*([ABCD])",
        r"^([ABCD])\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, response.strip())
        if match:
            return match.group(1).upper()

    # If no clear pattern, look for any single letter A, B, C, or D
    letters = re.findall(r"\b([ABCD])\b", response)
    if letters:
        return letters[-1]  # Take the last occurrence

    return None


def evaluate_mmlu_accuracy(predictions, ground_truths):
    """Evaluate MMLU accuracy by comparing predicted letters with ground truth"""
    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        extracted = extract_answer_letter(pred)
        if extracted == gt:
            correct += 1

    return correct / total * 100 if total > 0 else 0


def load_mmlu_csv(csv_path):
    """Load MMLU CSV file and return questions, choices, and answers"""
    questions = []
    choices = []
    answers = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:  # question, 4 choices, answer
                questions.append(row[0])
                choices.append(row[1:5])  # A, B, C, D choices
                answers.append(row[5])

    return questions, choices, answers


mmlu_prompt = """
# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:
```{instruction}```

# Answer:
```
"""


def mmlu_baseline(
    model: str = "Qwen/Qwen3-0.6B-Base",
    data_dir: str = "data/mmlu/dev",
):
    llm = LLM(model=model)

    # Load prompt template
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_results = {}
    total_correct = 0
    total_questions = 0

    for csv_file in csv_files:
        subject = os.path.basename(csv_file).replace("_dev.csv", "").replace("_", " ")
        print(f"\nEvaluating {subject}...")
        questions, choices, answers = load_mmlu_csv(csv_file)

        if not questions:
            print(f"No valid questions found in {csv_file}")
            continue

        prompts = []
        for question, choice_list in zip(questions, choices):
            instruction = create_mmlu_instruction(subject, question, choice_list)
            prompt = mmlu_prompt.format(instruction=instruction)
            prompts.append(prompt)

        # Generate responses
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=256, stop=None
        )

        outputs = llm.generate(prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]

        # Evaluate accuracy
        accuracy = evaluate_mmlu_accuracy(responses, answers)
        correct_count = int(accuracy * len(answers) / 100)

        all_results[subject] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(answers),
            "responses": responses,
            "ground_truths": answers,
        }

        total_correct += correct_count
        total_questions += len(answers)

        print(f"{subject}: {correct_count}/{len(answers)} ({accuracy:.2f}%)")

    # Print overall results
    overall_accuracy = (
        total_correct / total_questions * 100 if total_questions > 0 else 0
    )
    print(f"\n{'=' * 60}")
    print("MMLU EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(
        f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2f}%)"
    )
    print(f"{'=' * 60}")

    # Print per-subject breakdown
    print("\nPer-subject results:")
    for subject, result in sorted(all_results.items()):
        print(
            f"{subject:30}: {result['correct']:3}/{result['total']:3} ({result['accuracy']:6.2f}%)"
        )

    return all_results


### GSM8K ###


def evaluate_gsm8k_accuracy(predictions, ground_truths):
    """Evaluate GSM8K accuracy by comparing predicted numbers with ground truth"""
    scores = [compute_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(predictions) * 100


def load_gsm8k_jsonl(jsonl_path):
    questions, answers = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            questions.append(item["question"])
            answers.append(item["answer"].split("#### ")[-1].strip())
    return questions, answers


def gsm8k_baseline(model: str = "Qwen/Qwen3-0.6B-Base"):
    llm = LLM(model=model)
    questions, ground_truths = load_gsm8k_jsonl("data/gsm8k/test.jsonl")
    print(f"Evaluating {len(questions)} GSM8K examples...")

    prompts = []
    instruction_following = (
        'Let\'s think step by step and output the final answer after "####".'
    )
    for question in questions:  # [:100]:
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


if __name__ == "__main__":
    # Run evaluations
    # print("Running MMLU baseline...")
    # mmlu_results = mmlu_baseline()

    print("Running GSM8K baseline...")
    # gsm8k_results = gsm8k_baseline(
    #     "checkpoints/verl_grpo_gsm8k/qwen3_1.7b_grpo/merged_model"
    # )
    gsm8k_results = gsm8k_baseline(
        "josancamon/qwen3-0.6b-grpo-gsm8k"
    )
