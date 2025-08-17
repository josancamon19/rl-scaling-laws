#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, get_dataset_config_names
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS: List[str] = [
    "Qwen/Qwen3-0.6B-base",
    "Qwen/Qwen3-1.7B-base",
    "Qwen/Qwen3-4B-base",
    "Qwen/Qwen3-8B-base",
    "Qwen/Qwen3-14B-base",
]


def _load_validation_split(dataset_name: str):
    """Robustly load a validation-like split.

    Strategy:
    1) Try common split names on the dataset directly
    2) If the dataset requires a config, iterate configs and try the same splits
    3) Fallback: if dataset is mindchain/wikitext2, use wikitext wikitext-2-raw-v1 validation
    """
    candidate_splits = ["validation", "valid", "val", "test"]

    # 1) Direct
    for split in candidate_splits:
        try:
            return load_dataset(dataset_name, split=split), split
        except Exception:
            pass

    # 2) Try configs if any
    try:
        config_names = get_dataset_config_names(dataset_name)
    except Exception:
        config_names = []

    for cfg in config_names:
        for split in candidate_splits:
            try:
                return load_dataset(dataset_name, cfg, split=split), f"{cfg}/{split}"
            except Exception:
                pass

    # 3) Fallback for mindchain/wikitext2 → use standard wikitext
    if dataset_name == "mindchain/wikitext2":
        try:
            return (
                load_dataset("wikitext", "wikitext-2-raw-v1", split="validation"),
                "wikitext-2-raw-v1/validation",
            )
        except Exception:
            pass

    raise RuntimeError(
        f"Could not load a validation-like split from {dataset_name}. Tried {candidate_splits} and config variants."
    )


def _tokenize_and_chunk(
    dataset, tokenizer, block_size: int, num_proc: Optional[int] = None
):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    def group_texts(examples):
        # Concatenate across batch for all keys returned by tokenizer (e.g., input_ids, attention_mask)
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(
            concatenated_examples["input_ids"]
        )  # base on input_ids length
        total_length = (total_length // block_size) * block_size
        # Split by block_size for all keys
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # Labels mirror input_ids for causal LM
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Grouping into chunks of {block_size}",
    )
    lm_dataset.set_format(
        type="torch", columns=["input_ids", "labels", "attention_mask"]
    )
    return lm_dataset


@torch.no_grad()
def evaluate_val_loss(
    model_name: str,
    dataset_name: str,
    batch_size: int = 2,
    block_size: int = 1024,
    dtype: Optional[str] = None,
    device_map: Optional[str] = "auto",
    max_batches: Optional[int] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = None
    if dtype is not None:
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "fp32":
            torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    raw_ds, used_split = _load_validation_split(dataset_name)
    lm_ds = _tokenize_and_chunk(raw_ds, tokenizer, block_size=block_size)

    def collate(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }

    dl = DataLoader(lm_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    total_loss_sum = 0.0
    total_target_tokens = 0
    num_batches = 0

    for batch in dl:
        num_batches += 1
        if max_batches is not None and num_batches > max_batches:
            break

        batch = {
            k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = outputs.loss  # average over tokens and batch

        # Effective targets per sequence: (seq_len - 1) due to causal shift
        seq_len = batch["labels"].shape[1]
        batch_targets = (seq_len - 1) * batch["labels"].shape[0]
        total_loss_sum += float(loss.item()) * batch_targets
        total_target_tokens += batch_targets

    mean_nll = total_loss_sum / max(1, total_target_tokens)
    ppl = math.exp(mean_nll)

    return {
        "model": model_name,
        "dataset": dataset_name,
        "split": used_split,
        "block_size": block_size,
        "batch_size": batch_size,
        "loss": mean_nll,
        "perplexity": ppl,
        "num_batches": num_batches,
        "tokens_evaluated": total_target_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="List of HF model IDs to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mindchain/wikitext2",
        help="Validation dataset (e.g., mindchain/wikitext2)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=[None, "bf16", "fp16", "fp32"],
        help="Computation dtype",
    )
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--output", type=str, default=str(Path(__file__).parent / "val_losses.json")
    )
    args = parser.parse_args()

    all_results: List[Dict] = []
    for model in args.models:
        print(f"Evaluating validation loss for: {model}")
        res = evaluate_val_loss(
            model_name=model,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            block_size=args.block_size,
            dtype=args.dtype,
            device_map=args.device_map,
            max_batches=args.max_batches,
        )
        all_results.append(res)
        print(f"  loss={res['loss']:.4f}  ppl={res['perplexity']:.2f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
