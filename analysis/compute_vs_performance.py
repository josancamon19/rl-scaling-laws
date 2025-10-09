#!/usr/bin/env python3
"""
Plot 14B model performance vs estimated compute from RL training.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def estimate_compute_per_step(
    model_params_b=13.2,  # 14B model has ~13.2B non-embedding params
    batch_size=512,
    n_rollouts=3,
    avg_prompt_tokens=256,  # Average, max is 512
    avg_response_tokens=256,  # Average, max is 512
    ppo_mini_batch_size=512,
    n_gpus=8,
):
    """
    Estimate FLOPs for one RL training step.

    Each step involves:
    1. Rollout: Generate n responses for batch_size prompts (inference)
    2. Compute actor log probs (inference)
    3. Compute reference log probs (inference)
    4. PPO training update (forward + backward)

    Args:
        model_params_b: Model parameters in billions
        batch_size: Number of prompts per batch
        n_rollouts: Number of responses generated per prompt
        avg_prompt_tokens: Average prompt length
        avg_response_tokens: Average response length
        ppo_mini_batch_size: Mini-batch size for PPO updates
        n_gpus: Number of GPUs

    Returns:
        Total FLOPs for one training step
    """
    N = model_params_b * 1e9  # Convert to actual parameter count

    # Total samples generated per step
    total_samples = batch_size * n_rollouts

    # 1. Rollout phase (inference, autoregressive generation)
    # For autoregressive generation: ~2 * N * sequence_length per token
    # We generate avg_response_tokens tokens for each sample
    rollout_flops = (
        2
        * N
        * total_samples
        * avg_response_tokens
        * (avg_prompt_tokens + avg_response_tokens / 2)
    )

    # 2. Actor log prob computation (inference, non-autoregressive)
    # Forward pass only: 2 * N * D where D is total tokens
    actor_logprob_flops = (
        2 * N * total_samples * (avg_prompt_tokens + avg_response_tokens)
    )

    # 3. Reference model log prob computation (inference)
    ref_logprob_flops = (
        2 * N * total_samples * (avg_prompt_tokens + avg_response_tokens)
    )

    # 4. PPO training update (forward + backward on ppo_mini_batch_size)
    # This is done on the generated samples
    # Standard: 6 * N * D (2 forward + 4 backward)
    ppo_update_flops = 6 * N * total_samples * (avg_prompt_tokens + avg_response_tokens)

    total_flops = (
        rollout_flops + actor_logprob_flops + ref_logprob_flops + ppo_update_flops
    )

    return total_flops


def load_model_data(csv_path, model_size):
    """Load performance data for a specific model size from CSV."""
    steps = []
    accuracies = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row[model_size] and row[model_size].strip():  # Check if column has data
                steps.append(int(row["Step"]))
                accuracies.append(float(row[model_size]))

    return np.array(steps), np.array(accuracies)


def plot_compute_vs_performance():
    """Create plot of compute vs performance for multiple model sizes."""
    # Define model configurations with beautiful gradient colors
    # Model sizes, actual non-embedding params (in billions), and colors
    models = {
        "14b": {"params": 13.2, "color": "#D6336C", "label": "14B"},
        "8b": {"params": 6.95, "color": "#3B82F6", "label": "8B"},
        "4b": {"params": 3.6, "color": "#10B981", "label": "4B"},
        "1.7b": {"params": 1.4, "color": "#F59E0B", "label": "1.7B"},
        "0.6b": {"params": 0.44, "color": "#8B5CF6", "label": "0.6B"},
    }

    pretraining_tokens = 36e12  # 36 trillion tokens for all models

    # Load data
    csv_path = Path(__file__).parent.parent / "results" / "export.csv"

    # Create the plot with better styling
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    model_data = {}

    for model_key, model_info in models.items():
        try:
            steps, accuracies = load_model_data(csv_path, model_key)

            if len(steps) == 0:
                print(f"No data found for {model_info['label']}, skipping...")
                continue

            print(f"\n{'=' * 60}")
            print(f"{model_info['label']} Model")
            print(f"{'=' * 60}")
            print(f"Loaded {len(steps)} data points")
            print(f"Step range: {steps[0]} to {steps[-1]}")
            print(f"Accuracy range: {accuracies[0]:.1%} to {accuracies[-1]:.1%}")

            # Calculate pre-training compute
            model_params = model_info["params"] * 1e9
            pretraining_flops = 6 * model_params * pretraining_tokens

            print(
                f"Pre-training: {model_params / 1e9:.1f}B params × 36T tokens = {pretraining_flops:.2e} FLOPs"
            )

            # Calculate RL compute per step
            flops_per_step = estimate_compute_per_step(
                model_params_b=model_info["params"],
                batch_size=512,
                n_rollouts=3,
                avg_prompt_tokens=200,
                avg_response_tokens=150,
                ppo_mini_batch_size=512,
                n_gpus=8,
            )

            # Calculate RL compute
            rl_cumulative_flops = steps * flops_per_step

            print(f"RL compute at final step: {rl_cumulative_flops[-1]:.2e} FLOPs")
            print(
                f"RL as % of pre-training: {(rl_cumulative_flops[-1] / pretraining_flops) * 100:.3f}%"
            )

            model_data[model_key] = {
                "steps": steps,
                "accuracies": accuracies,
                "pretraining_flops": pretraining_flops,
                "rl_flops": rl_cumulative_flops,
                "info": model_info,
            }

        except KeyError:
            print(f"Column {model_key} not found in CSV, skipping...")
            continue

    # Create segmented x-axis: each model gets its own segment
    # We'll use a normalized x-axis where each segment shows the RL progress
    segment_width = 1.0  # Width allocated to each model
    segment_gap = 0.1  # Small gap between segments

    # Sort models by pre-training compute (smallest to largest)
    sorted_models = sorted(model_data.items(), key=lambda x: x[1]["pretraining_flops"])

    # Assign x-positions for each model
    x_tick_positions = []
    x_tick_labels = []

    for idx, (model_key, data) in enumerate(sorted_models):
        info = data["info"]

        # Calculate segment start position
        segment_start = idx * (segment_width + segment_gap)

        # Normalize RL flops to fit within the segment
        max_rl_flops = data["rl_flops"][-1]
        if max_rl_flops > 0:
            x_normalized = (
                segment_start + (data["rl_flops"] / max_rl_flops) * segment_width
            )
        else:
            x_normalized = np.full_like(data["rl_flops"], segment_start)

        # Plot the curve with enhanced styling
        ax.plot(
            x_normalized,
            data["accuracies"] * 100,
            color=info["color"],
            linewidth=3.5,
            marker="o",
            markersize=6,
            markerfacecolor=info["color"],
            markeredgecolor="white",
            markeredgewidth=2,
            label=f"{info['label']} ({data['accuracies'][0]:.1%} → {data['accuracies'][-1]:.1%})",
            alpha=0.95,
            zorder=5,
            solid_capstyle="round",
        )

        # Mark the pre-training baseline (Step 0) with glow effect
        ax.scatter(
            [segment_start],
            [data["accuracies"][0] * 100],
            color=info["color"],
            s=300,
            marker="s",
            edgecolors="white",
            linewidth=3,
            zorder=10,
            alpha=0.9,
        )

        # Mark the final point with star
        ax.scatter(
            [x_normalized[-1]],
            [data["accuracies"][-1] * 100],
            color=info["color"],
            s=400,
            marker="*",
            edgecolors="white",
            linewidth=3,
            zorder=10,
            alpha=1.0,
        )
        
        # Add annotation showing actual RL FLOPs added
        rl_flops_value = data["rl_flops"][-1]
        rl_percentage = (rl_flops_value / data["pretraining_flops"]) * 100
        
        # Format FLOPs in a readable way
        if rl_flops_value >= 1e19:
            flops_display = f"+{rl_flops_value/1e19:.1f}e19"
        elif rl_flops_value >= 1e18:
            flops_display = f"+{rl_flops_value/1e18:.1f}e18"
        else:
            flops_display = f"+{rl_flops_value:.1e}"
        
        annotation_text = f"{flops_display} FLOPs\n({rl_percentage:.3f}%)"
        
        # Position annotation above the final star
        ax.annotate(
            annotation_text,
            xy=(x_normalized[-1], data["accuracies"][-1] * 100),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=info["color"],
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=info["color"],
                linewidth=2,
                alpha=0.95,
            ),
            zorder=11,
        )

        # Add tick at segment start (pre-training baseline)
        x_tick_positions.append(segment_start)
        x_tick_labels.append(f"{info['label']}\n{data['pretraining_flops']:.1e}")

    # Add vertical separators between model segments with better styling
    for idx in range(1, len(sorted_models)):
        separator_x = idx * (segment_width + segment_gap) - segment_gap / 2
        ax.axvline(
            x=separator_x,
            color="#6B7280",
            linestyle="--",
            linewidth=2,
            alpha=0.3,
            zorder=1,
        )

    # Set custom x-axis ticks
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=11, fontweight="medium")

    # Customize plot with modern typography
    ax.set_xlabel(
        "Model Size & Pre-training Compute → RL Fine-tuning Progress",
        fontsize=14,
        fontweight="bold",
        color="#1F2937",
        labelpad=12,
    )
    ax.set_ylabel(
        "GSM8K Accuracy (%)",
        fontsize=14,
        fontweight="bold",
        color="#1F2937",
        labelpad=12,
    )
    ax.set_title(
        "Multi-Model Scaling: Compute vs Performance\n"
        "Each segment shows RL improvement on top of pre-trained baseline",
        fontsize=17,
        fontweight="bold",
        color="#111827",
        pad=25,
    )

    # Enhanced grid
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.8, color="#9CA3AF")
    ax.set_axisbelow(True)

    # Improve spine styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#D1D5DB")
        spine.set_linewidth(1.5)

    # Add modern legend with better styling
    legend = ax.legend(
        loc="lower right",
        fontsize=11,
        framealpha=0.98,
        edgecolor="#D1D5DB",
        fancybox=True,
        shadow=True,
        title="Model Size (Initial → Final)",
        title_fontsize=12,
        borderpad=1.0,
        labelspacing=0.8,
    )
    legend.get_frame().set_facecolor("#FFFFFF")
    legend.get_frame().set_linewidth(2)
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_color("#111827")

    # Add modern info box with training details
    info_text = (
        "RL Configuration\n"
        "───────────────\n"
        "Pre-training: 36T tokens\n"
        "Batch Size: 512\n"
        "Rollouts/prompt: 3\n"
        "Learning Rate: 1e-6\n\n"
        "□  Pre-trained baseline\n"
        "*  After RL fine-tuning"
    )
    ax.text(
        0.015,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10.5,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=1.0",
            facecolor="#FFFFFF",
            alpha=0.98,
            edgecolor="#D1D5DB",
            linewidth=2,
        ),
        fontfamily="sans-serif",
        color="#1F2937",
        linespacing=1.6,
    )

    # Set y-axis to show percentage range nicely
    all_accuracies = np.concatenate(
        [data["accuracies"] for data in model_data.values()]
    )
    y_min = max(0, (min(all_accuracies) * 100) - 5)
    y_max = min(100, (max(all_accuracies) * 100) + 5)
    ax.set_ylim(y_min, y_max)

    # Format y-axis as percentage with better styling
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.tick_params(axis="both", labelsize=11, colors="#374151", length=6, width=1.5)

    plt.tight_layout(pad=1.5)

    # Save the plot
    output_file = (
        Path(__file__).parent.parent
        / "results"
        / "multi_model_compute_vs_performance.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")

    # Print summary statistics for all models
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS - ALL MODELS")
    print("=" * 80)

    for model_key, data in model_data.items():
        info = data["info"]
        improvement = (data["accuracies"][-1] - data["accuracies"][0]) * 100

        print(f"\n{info['label']} Model:")
        print(f"  Initial Accuracy (Step 0):            {data['accuracies'][0]:.2%}")
        print(
            f"  Final Accuracy (Step {data['steps'][-1]}):              {data['accuracies'][-1]:.2%}"
        )
        print(
            f"  Absolute Improvement:                 +{improvement:.2f} percentage points"
        )
        print(
            f"  Relative Improvement:                 +{(data['accuracies'][-1] / data['accuracies'][0] - 1) * 100:.1f}%"
        )
        print(
            f"  Pre-training Compute:                 {data['pretraining_flops']:.2e} FLOPs"
        )
        print(
            f"  RL Fine-tuning Compute:               {data['rl_flops'][-1]:.2e} FLOPs"
        )
        print(
            f"  RL as % of Pre-training:              {(data['rl_flops'][-1] / data['pretraining_flops']) * 100:.3f}%"
        )
        print(
            f"  RL Compute per Point Improvement:     {data['rl_flops'][-1] / improvement:.2e} FLOPs/%"
        )

    print("=" * 80)

    return fig, ax


def main():
    """Main function."""
    plot_compute_vs_performance()
    plt.show()


if __name__ == "__main__":
    main()
