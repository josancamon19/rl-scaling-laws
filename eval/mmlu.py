import re
from vllm import LLM
from vllm.sampling_params import SamplingParams
from datasets import load_dataset


def extract_answer_letter(response):
    """Extract the answer letter from the model response"""
    # Look for patterns like "The correct answer is A" or just "A" at the end
    patterns = [
        r"[Tt]he correct answer is ([ABCD])",
        r"([ABCD])\s*\.\s*$",
        r"Answer:\s*([ABCD])",
        r"^([ABCD])\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, response.strip())
        if match:
            return match.group(1).upper()

    # If no clear pattern, look for any single letter A, B, C, or D
    if letters := re.findall(r"\b([ABCD])\b", response):
        return letters[-1]  # Take the last occurrence

    return None


def evaluate_mmlu_accuracy(predictions, ground_truths):
    """Evaluate MMLU accuracy by comparing predicted letters with ground truth"""
    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        if extract_answer_letter(pred) == gt:
            correct += 1

    return correct / total * 100 if total > 0 else 0


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


def run_mmlu_evaluation(model: str, split: str = "test"):
    """Run MMLU evaluation on the specified model using HuggingFace datasets
    
    Args:
        model: Model path or HuggingFace model identifier
        split: Which HuggingFace split to use ("validation", "test", "dev")
    """
    llm = LLM(model=model)
    
    print(f"Loading MMLU dataset from HuggingFace (split: {split})...")
    dataset = load_dataset("cais/mmlu", "all", split=split)
    
    # Group by subject
    subjects_data = {}
    for example in dataset:
        subject = example["subject"]
        if subject not in subjects_data:
            subjects_data[subject] = {
                "questions": [],
                "choices": [],
                "answers": []
            }
        subjects_data[subject]["questions"].append(example["question"])
        subjects_data[subject]["choices"].append(example["choices"])
        # Convert answer index (0-3) to letter (A-D)
        subjects_data[subject]["answers"].append(chr(65 + example["answer"]))
    
    all_results = {}
    total_correct = 0
    total_questions = 0
    
    for subject, data in subjects_data.items():
        print(f"\nEvaluating {subject}...")
        questions = data["questions"]
        choices = data["choices"]
        answers = data["answers"]
        
        if not questions:
            print(f"No valid questions found for {subject}")
            continue
        
        prompts = []
        for question, choice_list in zip(questions, choices):
            formatted = f"{question}\n\nA. {choice_list[0]}\nB. {choice_list[1]}\nC. {choice_list[2]}\nD. {choice_list[3]}"
            instruction = f'Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n{formatted}'
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