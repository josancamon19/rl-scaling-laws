#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from eval module
sys.path.append(str(Path(__file__).parent.parent))

from eval.mmlu import run_mmlu_evaluation, PromptType as MMLUPromptType
from eval.gsm8k import run_gsm8k_evaluation, PromptType
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-base")
    # Determinism: Greedy decoding removes sampling noise so changes reflect the model, not randomness.
    # Sensitivity: If RL truly improved the policy, the mode of the distribution should improve; greedy captures that directly
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--gsm8k-prompt-type",
        type=str,
        default=PromptType.zero_shot.value,
        choices=[pt.value for pt in PromptType],
        help="Prompting style for GSM8K: zero_shot, one_shot, two_shot, three_shot, four_shot, five_shot",
    )
    parser.add_argument(
        "--mmlu-prompt-type",
        type=str,
        default=MMLUPromptType.zero_shot.value,
        choices=[pt.value for pt in MMLUPromptType],
        help="Prompting style for MMLU: zero_shot, one_shot, two_shot, three_shot, four_shot, five_shot",
    )
    args = parser.parse_args()

    print(f"Evaluating model: {args.model}")
    print("=" * 60)
    # Disable HuggingFace and vllm logging for cleaner output

    run_mmlu_evaluation(
        args.model,
        prompt_type=MMLUPromptType(args.mmlu_prompt_type),
        temperature=args.temperature,
    )
    # python eval/main.py --gsm8k-prompt-type five_shot
    run_gsm8k_evaluation(
        args.model,
        prompt_type=PromptType(args.gsm8k_prompt_type),
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
