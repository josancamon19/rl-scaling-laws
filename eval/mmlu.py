import re
from vllm import LLM
from vllm.sampling_params import SamplingParams
from datasets import load_dataset

# TODO: Evaluate as well by checking log probs of the correct letter.


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
        correct += extract_answer_letter(pred) == gt

    return correct / total * 100 if total > 0 else 0


def _format_mc_question(question, choices):
    return f"{question}\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"


def _load_support_split(eval_split: str):
    s = "test" if eval_split == "dev" else "dev"
    return load_dataset("cais/mmlu", "all", split=s)


def _build_subject_support_map(support_ds):
    if support_ds is None:
        return {}
    by_subject = {}
    for ex in support_ds:
        subject = ex["subject"]
        if subject not in by_subject:
            by_subject[subject] = []
        answer_letter = chr(65 + ex["answer"])  # 65 == 'A'
        by_subject[subject].append((ex["question"], ex["choices"], answer_letter))
    return by_subject


def _build_few_shot_block(subject: str, by_subject: dict, k: int) -> str:
    if k <= 0:
        return ""
    examples = by_subject.get(subject, [])
    if not examples:
        return ""
    k = min(k, len(examples))
    parts = []
    for i in range(k):
        q, choices, ans_letter = examples[i]
        formatted = _format_mc_question(q, choices)
        parts.append(f"Q: {formatted}\nA: The correct answer is {ans_letter}\n")
    return "\n".join(parts) + "\n"


def run_mmlu_evaluation(
    model: str,
    split: str = "test",
    num_shots: int = 0,
    temperature: float = 1.0,
    revision: str = None,
    llm: LLM = None,
):
    """Run MMLU evaluation on the specified model using HuggingFace datasets

    Args:
        model: Model path or HuggingFace model identifier
        split: Which HuggingFace split to use ("validation", "test", "dev")
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

    print(f"Loading MMLU dataset from HuggingFace (split: {split})...")
    dataset = load_dataset("cais/mmlu", "all", split=split)

    # Group by subject
    subjects_data = {}
    for example in dataset:
        subject = example["subject"]
        if subject not in subjects_data:
            subjects_data[subject] = {"questions": [], "choices": [], "answers": []}
        subjects_data[subject]["questions"].append(example["question"])
        subjects_data[subject]["choices"].append(example["choices"])
        # Convert answer index (0-3) to letter (A-D)
        subjects_data[subject]["answers"].append(chr(65 + example["answer"]))

    all_results = {}
    total_correct = 0
    total_questions = 0
    samples_for_shots = load_dataset(
        "cais/mmlu", "all", split=("test" if split == "dev" else "dev")
    )
    support_by_subject = (
        _build_subject_support_map(samples_for_shots) if num_shots > 0 else {}
    )

    # Collect all prompts and metadata for batch processing
    all_prompts = []
    prompt_metadata = []  # Track (subject_idx, question_idx) for each prompt
    subject_list = list(subjects_data.keys())
    
    print("Preparing prompts for all subjects...")
    for subject_idx, (subject, data) in enumerate(subjects_data.items()):
        questions = data["questions"]
        choices = data["choices"]
        answers = data["answers"]

        if not questions:
            print(f"No valid questions found for {subject}")
            continue

        few_shot_block = _build_few_shot_block(subject, support_by_subject, num_shots)
        for question_idx, (question, choice_list) in enumerate(zip(questions, choices)):
            formatted = _format_mc_question(question, choice_list)
            instruction = f'Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n'
            if few_shot_block:
                instruction += f"{few_shot_block}\n\n"

            instruction += f"Q: {formatted}\nA: "
            all_prompts.append(instruction)
            prompt_metadata.append((subject_idx, question_idx))
    
    # Show sample prompt
    if all_prompts:
        first_subject = subject_list[0]
        print(f"\nSample prompt (from {first_subject}):\n{all_prompts[0]}")
    
    # Generate all responses in a single batch
    print(f"\nGenerating responses for {len(all_prompts)} questions...")
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=256,
        stop=None,
    )
    
    outputs = llm.generate(all_prompts, sampling_params)
    all_responses = [output.outputs[0].text for output in outputs]
    
    # Show sample response
    if all_responses and subject_list:
        first_subject = subject_list[0]
        first_answer = subjects_data[first_subject]["answers"][0]
        print(f"\nSample response (from {first_subject}):\n{all_responses[0]}\nGround Truth: {first_answer}")
    
    # Organize responses back by subject
    print("\nEvaluating results by subject...")
    for subject_idx, (subject, data) in enumerate(subjects_data.items()):
        questions = data["questions"]
        answers = data["answers"]
        
        if not questions:
            continue
            
        # Extract responses for this subject
        subject_responses = []
        for i, (s_idx, q_idx) in enumerate(prompt_metadata):
            if s_idx == subject_idx:
                subject_responses.append(all_responses[i])
        
        # Evaluate accuracy
        accuracy = evaluate_mmlu_accuracy(subject_responses, answers)
        correct_count = int(accuracy * len(answers) / 100)

        all_results[subject] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(answers),
            "responses": subject_responses,
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
