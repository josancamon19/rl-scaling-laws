# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import datasets
import logging
import numpy as np
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Qwen3 tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    logger.info("Successfully loaded Qwen3 tokenizer")
except Exception as e:
    logger.error(f"Failed to load Qwen3 tokenizer: {e}")
    raise

def get_token_length(text):
    """Get the token length of a text using Qwen3 tokenizer"""
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Error tokenizing text: {e}")
        return 0

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def calculate_quartiles(lengths, percentiles=[5, 25, 50, 75, 95]):
    """Calculate and print specified percentiles of token lengths"""
    if not lengths:
        logger.warning("No lengths to calculate quartiles from")
        return

    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)

    logger.info("=== Token Length Statistics ===")
    logger.info(f"Total samples: {n}")
    logger.info(f"Min length: {min(lengths)}")
    logger.info(f"Max length: {max(lengths)}") # 227 tokens
    logger.info(f"Mean length: {np.mean(lengths):.2f}")
    logger.info(f"Std length: {np.std(lengths):.2f}")
    logger.info(f"Longer than 512 tokens: {sum(1 for length in lengths if length > 512)}")
    logger.info(f"Longer than 384 tokens: {sum(1 for length in lengths if length > 384)}")

    for percentile in percentiles:
        index = int((percentile / 100) * (n - 1))
        value = sorted_lengths[index]
        logger.info(f"{percentile}th percentile: {value}")

    return sorted_lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/gsm8k")

    args = parser.parse_args()

    data_source = "openai/gsm8k"

    logger.info("Loading GSM8K dataset...")
    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # Store token lengths for analysis
    all_prompt_lengths = []

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            # Get token length of the prompt
            prompt_length = get_token_length(question)
            all_prompt_lengths.append(prompt_length)

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                # "data_source": data_source,
                "data_source": "custom/gsm8k", # avoid default reward
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "token_length": prompt_length,  # Add token length to extra info
                },
            }
            return data

        return process_fn

    logger.info("Processing training dataset...")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    logger.info("Processing test dataset...")
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Calculate and print quartiles
    calculate_quartiles(all_prompt_lengths)

    local_dir = args.local_dir

    # Ensure directory exists
    os.makedirs(local_dir, exist_ok=True)

    logger.info(f"Saving datasets to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    logger.info("Processing complete!")