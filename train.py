import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
import torch


app = typer.Typer()
console = Console()

VALID_SIZES = ["0.6B", "1.7B", "4B", "8B", "14B"]


def download_model_if_needed(model_path: str, cache_dir: str) -> None:
    """Download and cache model if not already available."""
    console.print("[cyan]Checking model cache...[/cyan]")

    try:
        # Try to load from cache first
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_dir, local_files_only=True
        )
        AutoConfig.from_pretrained(
            model_path, cache_dir=cache_dir, local_files_only=True
        )
        console.print("[green]✓ Model already in cache[/green]")
    except Exception as e:
        console.print(f"[yellow]Downloading model: {e}[/yellow]")
        # Download entire repository
        snapshot_download(
            repo_id=model_path,
            cache_dir=cache_dir,
            ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
        )
        # Load to verify
        AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )
        AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        console.print("[green]✓ Model downloaded and cached[/green]")


def main(
    # Model
    model_size: str = typer.Option("0.6B", "--model-size", "-m", help="Model size"),
    # Training
    epochs: int = typer.Option(4, "--epochs", help="Number of training epochs"),
    learning_rate: float = typer.Option(1e-6, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(512, "--batch-size", help="Training batch size"),
    # Data
    max_prompt_length: int = typer.Option(
        512, "--max-prompt-length", help="Max prompt tokens"
    ),
    max_response_length: int = typer.Option(
        512, "--max-response-length", help="Max response tokens"
    ),
    # Rollout
    rollout_n: int = typer.Option(3, "--rollout-n", "-n", help="Rollouts per prompt"),
    # TODO: allow for prompt ablations, e.g. instead of ####, to check as <think> and </think>, or some variation of this
    # - would require eval options + a different reward function set
    # Reward
    reward_method: str = typer.Option(
        "custom_flexible",
        "--reward-method",
        help="Reward extraction method: strict, flexible, or custom_flexible",
    ),
    # GRPO
    dapo_clip_higher: bool = True,
    # dapo_clip_higher: bool = typer.Option(
    #     True,
    #     "--dapo-clip-higher",
    #     help="Use asymmetric clipping (0.2/0.28) vs symmetric (0.2/0.2)",
    # ),
    loss_agg_mode: str = typer.Option(
        "token-mean", "--loss-agg-mode", help="Loss aggregation mode"
    ),
    use_kl_in_reward: bool = typer.Option(
        False, "--use-kl-in-reward", help="Use KL in reward"
    ),
    # Wandb
    wandb_project_name: str = typer.Option(
        "rl-scaling", "--wandb-project-name", help="Wandb project name"
    ),
) -> None:
    if model_size not in VALID_SIZES:
        console.print(f"[red]Error: Invalid model size '{model_size}'[/red]")
        console.print(f"Valid sizes: {', '.join(VALID_SIZES)}")
        raise typer.Exit(1)

    valid_reward_methods = ["strict", "flexible", "custom_flexible"]
    if reward_method not in valid_reward_methods:
        console.print(f"[red]Error: Invalid reward method '{reward_method}'[/red]")
        console.print(f"Valid methods: {', '.join(valid_reward_methods)}")
        raise typer.Exit(1)

    clip_ratio_low = 0.2
    clip_ratio_high = 0.28 if dapo_clip_higher else 0.2

    # Setup paths
    root_dir = Path(__file__).parent.absolute()
    data_dir = root_dir / "data" / "gsm8k"
    hf_home = "/workspace/.hf_home"
    os.environ["HF_HOME"] = hf_home

    # Model paths
    model_path = f"Qwen/Qwen3-{model_size}-Base"
    model_name = f"qwen3_{model_size.lower()}".replace(".", "_")

    # Experiment naming
    experiment_name = "_".join(
        [
            f"{model_name}_grpo",
            f"lr{learning_rate}",
            f"bs{batch_size}",
            f"ep{epochs}",
            f"n{rollout_n}",
            "no_clip_higher" if not dapo_clip_higher else "clip_higher",
            f"{loss_agg_mode}",
            "kl_in_rew" if use_kl_in_reward else "no_kl_rew",
            f"rew_{reward_method}",
        ]
    )

    console.print("[cyan]Training configuration:[/cyan]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Rollouts: {rollout_n}")
    console.print(f"  Clip ratio low: {clip_ratio_low}")
    console.print(f"  Clip ratio high: {clip_ratio_high}")
    console.print(f"  Loss aggregation mode: {loss_agg_mode}")
    console.print(f"  Use KL in reward: {use_kl_in_reward}")
    console.print(f"  Reward method: {reward_method}")
    console.print(f"  Experiment: {experiment_name}")

    # Download model if needed
    download_model_if_needed(model_path, hf_home)

    # Convert data if needed
    alt_data_dir = root_dir / "data"
    if (
        not (data_dir / "train.jsonl").exists()
        and (alt_data_dir / "train.jsonl").exists()
    ):
        data_dir = alt_data_dir

    train_json = data_dir / "train.jsonl"
    train_parquet = data_dir / "train.parquet"
    test_json = data_dir / "test.jsonl"
    test_parquet = data_dir / "test.parquet"

    if train_json.exists() and not train_parquet.exists():
        console.print(f"[cyan]Converting {train_json} -> {train_parquet}[/cyan]")
        subprocess.run(
            [
                sys.executable,
                str(root_dir / "jsonl_to_parquet.py"),
                str(train_json),
                str(train_parquet),
            ],
            check=True,
        )

    if test_json.exists() and not test_parquet.exists():
        console.print(f"[cyan]Converting {test_json} -> {test_parquet}[/cyan]")
        subprocess.run(
            [
                sys.executable,
                str(root_dir / "jsonl_to_parquet.py"),
                str(test_json),
                str(test_parquet),
            ],
            check=True,
        )

    # Map reward method to wrapper function name
    reward_function_map = {
        "strict": "compute_score_strict",
        "flexible": "compute_score_flexible",
        "custom_flexible": "compute_score_custom_flexible",
    }
    reward_function_name = reward_function_map[reward_method]

    # Build training arguments
    args = [
        # Basic node config
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.save_freq=50",
        "trainer.test_freq=4",
        "trainer.resume_mode=disable",
        # Data
        f"data.train_files={train_parquet}",
        f"data.val_files={test_parquet}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        f"data.train_batch_size={batch_size}",
        # Model
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_fused_kernels=True",
        # Logging
        "trainer.logger=[console,wandb]",
        f"trainer.project_name={wandb_project_name}",
        f"trainer.experiment_name={experiment_name}",
        # Reward
        "reward_model.reward_manager=naive",
        f"custom_reward_function.path={root_dir / 'utils' / 'reward_wrappers.py'}",
        f"custom_reward_function.name={reward_function_name}",
        # Training
        f"trainer.total_epochs={epochs}",
        "trainer.critic_warmup=0",
        # Rollout
        f"actor_rollout_ref.rollout.n={rollout_n}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64",
        # GRPO
        # https://verl.readthedocs.io/en/latest/examples/config.html
        f"actor_rollout_ref.actor.optim.lr={learning_rate}",
        "actor_rollout_ref.actor.ppo_mini_batch_size=512",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64",
        "actor_rollout_ref.actor.use_kl_loss=True",  # grpo default
        "actor_rollout_ref.actor.kl_loss_coef=0.001",  # grpo default
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",  # grpo default
        "actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
        f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
        f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
        # Algorithm
        "algorithm.adv_estimator=grpo",
        f"algorithm.use_kl_in_reward={use_kl_in_reward}",
    ]

    # Launch training
    console.print("\n[green]Launching training...[/green]\n")
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo"] + args

    try:
        subprocess.run(cmd, check=True)
        console.print("\n[green]✓ Training complete![/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Training failed with exit code {e.returncode}[/red]")
        raise typer.Exit(e.returncode)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted[/yellow]")
        raise typer.Exit(130)


if __name__ == "__main__":
    typer.run(main)
