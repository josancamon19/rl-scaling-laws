#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from eval module
sys.path.append(str(Path(__file__).parent.parent))

from eval.mmlu import run_mmlu_evaluation
from eval.gsm8k import run_gsm8k_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-base")
    args = parser.parse_args()

    print(f"Evaluating model: {args.model}")
    print("=" * 60)

    run_mmlu_evaluation(args.model)
    run_gsm8k_evaluation(args.model)


if __name__ == "__main__":
    main()
