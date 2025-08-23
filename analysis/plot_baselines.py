#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def estimate_training_flops(model_params_B, batch_size=512, rollout_n=3, seq_len=512):
    N = model_params_B * 1e9  # Convert to actual parameters
    total_sequences = batch_size * rollout_n  # 1,536
    total_tokens = total_sequences * seq_len
    rollout_flops = 2 * N * (batch_size * rollout_n * 512)
    ref_logprob_flops = 2 * N * total_tokens
    actor_logprob_flops = 2 * N * total_tokens
    training_flops = 6 * N * total_tokens
    return rollout_flops + ref_logprob_flops + actor_logprob_flops + training_flops


QWEN3_BASE_PARAMS_N: Dict[str, float] = {
    "Qwen/Qwen3-14B-base": 13.2e9,
    "Qwen/Qwen3-8B-base": 6.95e9,
    "Qwen/Qwen3-4B-base": 3.6e9,
    "Qwen/Qwen3-1.7B-base": 1.4e9,
    "Qwen/Qwen3-0.6B-base": 0.44e9,
}

# Actual advertised model sizes (including embedding params)
QWEN3_MODEL_SIZES: Dict[str, str] = {
    "Qwen/Qwen3-14B-base": "14B",
    "Qwen/Qwen3-8B-base": "8B",
    "Qwen/Qwen3-4B-base": "4B",
    "Qwen/Qwen3-1.7B-base": "1.7B",
    "Qwen/Qwen3-0.6B-base": "0.6B",
}


def compute_flops(n_non_embedding_params: float, num_tokens: float = 36e12) -> float:
    """Compute FLOPs C using C = 6 * N * D.

    N is non-embedding parameters, D is number of tokens.
    """
    return 6.0 * n_non_embedding_params * num_tokens


def load_accuracy_by_model(
    results_json: Path, benchmark: str, shots: List[int]
) -> Dict[int, Dict[str, float]]:
    """Return mapping: shot -> { model_name -> accuracy_percent }.

    Skips entries with errors or missing data.
    """
    data = json.loads(Path(results_json).read_text())
    shot_to_model_acc: Dict[int, Dict[str, float]] = {k: {} for k in shots}
    for model_entry in data.get("models", []):
        model_name = model_entry.get("model")
        benchmarks = model_entry.get("benchmarks", {})
        bench = benchmarks.get(benchmark, {})
        by_shot = bench.get("by_shot", {})
        for k in shots:
            node = by_shot.get(str(k))
            if not node or (isinstance(node, dict) and "error" in node):
                continue
            acc = node.get("accuracy") if isinstance(node, dict) else None
            if acc is None:
                continue
            shot_to_model_acc[k][model_name] = float(acc)
    return shot_to_model_acc


def discover_available_shots(results_json: Path, benchmark: str) -> List[int]:
    """Discover available shot keys for a given benchmark from results JSON."""
    data = json.loads(Path(results_json).read_text())
    shots_set = set()
    for model_entry in data.get("models", []):
        bench = model_entry.get("benchmarks", {}).get(benchmark, {})
        by_shot = bench.get("by_shot", {})
        for k in by_shot.keys():
            try:
                shots_set.add(int(k))
            except Exception:
                continue
    return sorted(list(shots_set))


def prepare_xy(
    model_to_acc: Dict[str, float], x_axis: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return arrays X, Y and list of models used, filtered to those with known N.

    X is FLOPs or N depending on x_axis, Y is accuracy in percent.
    """
    xs: List[float] = []
    ys: List[float] = []
    models: List[str] = []
    for model, acc in model_to_acc.items():
        if model not in QWEN3_BASE_PARAMS_N:
            continue
        n_params = QWEN3_BASE_PARAMS_N[model]
        x_val = compute_flops(n_params) if x_axis == "flops" else n_params
        if acc is None:
            continue
        xs.append(x_val)
        ys.append(acc)
        models.append(model)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), models


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = A * x^alpha using log10 regression.

    Returns (alpha, A, r2) where r2 is computed in log space.
    """
    x_log = np.log10(x)
    y_log = np.log10(y)
    slope, intercept = np.polyfit(x_log, y_log, 1)
    y_log_pred = slope * x_log + intercept
    ss_res = np.sum((y_log - y_log_pred) ** 2)
    ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    alpha = slope
    A = 10**intercept
    return alpha, A, r2


def plot_gsm8k_shots_vs_accuracy(
    results_path: Path,
    shots: List[int],
    save_path: Path | None,
) -> None:
    """Plot GSM8K accuracy vs shots for different model sizes."""
    shot_to_model_acc = load_accuracy_by_model(results_path, "gsm8k", shots)

    # Get all models and sort by parameter count
    all_models = set()
    for shot_dict in shot_to_model_acc.values():
        all_models.update(shot_dict.keys())

    # Filter to models with known parameters and sort by size
    models_with_params = [
        (m, QWEN3_BASE_PARAMS_N[m]) for m in all_models if m in QWEN3_BASE_PARAMS_N
    ]
    models_with_params.sort(key=lambda x: x[1])  # Sort by parameter count

    plt.figure(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (model, n_params) in enumerate(models_with_params):
        accuracies = []
        shots_for_model = []

        for shot in shots:
            if model in shot_to_model_acc[shot]:
                acc = shot_to_model_acc[shot][model]
                accuracies.append(acc)
                shots_for_model.append(shot)

        if accuracies:
            color = colors[idx % len(colors)]
            label = QWEN3_MODEL_SIZES.get(model, f"{n_params / 1e9:.1f}B")

            # Add light dotted lines for 0.6B and 1.7B models
            model_size = QWEN3_MODEL_SIZES.get(model, "")
            if model_size in ["0.6B", "1.7B"]:
                plt.plot(
                    shots_for_model,
                    accuracies,
                    "o:",
                    label=label,
                    color=color,
                    markersize=8,
                    alpha=0.8,
                    linewidth=1.5,
                )
            else:
                plt.plot(
                    shots_for_model,
                    accuracies,
                    "o",
                    label=label,
                    color=color,
                    markersize=8,
                    alpha=0.8,
                )

    plt.xlabel("Number of shots")
    plt.ylabel("GSM8K accuracy (%)")
    plt.title("GSM8K Performance vs Number of Shots")
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.legend(title="Model Size", loc="best")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_mmlu_shots_vs_accuracy(
    results_path: Path,
    shots: List[int],
    save_path: Path | None,
) -> None:
    """Plot MMLU accuracy vs shots for different model sizes."""
    shot_to_model_acc = load_accuracy_by_model(results_path, "mmlu", shots)

    # Get all models and sort by parameter count
    all_models = set()
    for shot_dict in shot_to_model_acc.values():
        all_models.update(shot_dict.keys())

    # Filter to models with known parameters and sort by size
    models_with_params = [
        (m, QWEN3_BASE_PARAMS_N[m]) for m in all_models if m in QWEN3_BASE_PARAMS_N
    ]
    models_with_params.sort(key=lambda x: x[1])  # Sort by parameter count

    plt.figure(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (model, n_params) in enumerate(models_with_params):
        accuracies = []
        shots_for_model = []

        for shot in shots:
            if model in shot_to_model_acc[shot]:
                acc = shot_to_model_acc[shot][model]
                accuracies.append(acc)
                shots_for_model.append(shot)

        if accuracies:
            color = colors[idx % len(colors)]
            label = QWEN3_MODEL_SIZES.get(model, f"{n_params / 1e9:.1f}B")

            # Add light dotted lines for 0.6B and 1.7B models
            model_size = QWEN3_MODEL_SIZES.get(model, "")
            if model_size in ["0.6B", "1.7B"]:
                plt.plot(
                    shots_for_model,
                    accuracies,
                    "o:",
                    label=label,
                    color=color,
                    markersize=8,
                    alpha=0.8,
                    linewidth=1.5,
                )
            else:
                plt.plot(
                    shots_for_model,
                    accuracies,
                    "o",
                    label=label,
                    color=color,
                    markersize=8,
                    alpha=0.8,
                )

    plt.xlabel("Number of shots")
    plt.ylabel("MMLU accuracy (%)")
    plt.title("MMLU Performance vs Number of Shots")
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.legend(title="Model Size", loc="best")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def load_val_losses(results_json: Path) -> Dict[str, Dict[str, float]]:
    """Load validation losses/perplexities and return mapping model -> metrics.

    Expected JSON schema: list of entries with keys 'model', 'loss', 'perplexity'.
    """
    data = json.loads(Path(results_json).read_text())
    model_to_metrics: Dict[str, Dict[str, float]] = {}
    if isinstance(data, list):
        for entry in data:
            try:
                model = entry.get("model")
                loss = (
                    float(entry.get("loss")) if entry.get("loss") is not None else None
                )
                ppl = (
                    float(entry.get("perplexity"))
                    if entry.get("perplexity") is not None
                    else None
                )
                if model is None:
                    continue
                metrics: Dict[str, float] = {}
                if loss is not None:
                    metrics["loss"] = loss
                if ppl is not None:
                    metrics["perplexity"] = ppl
                if metrics:
                    model_to_metrics[model] = metrics
            except Exception:
                continue
    return model_to_metrics


def plot_val_scaling(
    results_path: Path,
    y_metric: str,
    x_axis: str,
    save_path: Path | None,
    target_params_b: float | None = None,
) -> None:
    """Plot scaling for validation metrics (loss or perplexity) vs params/FLOPs."""
    model_to_metrics = load_val_losses(results_path)
    # Build mapping model -> metric value
    model_to_value: Dict[str, float] = {}
    for model_name, metrics in model_to_metrics.items():
        if y_metric in metrics:
            model_to_value[model_name] = float(metrics[y_metric])

    if not model_to_value:
        print(f"No '{y_metric}' data found in {results_path}")
        return

    X, Y, models = prepare_xy(model_to_value, x_axis)
    if len(X) == 0:
        print("No models with known parameter counts to plot.")
        return

    # Fit power-law y = A * x^alpha in log space
    alpha, A, r2 = fit_power_law(X, Y)

    plt.figure(figsize=(8, 6))
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    plt.scatter(X, Y, label=f"{y_metric}", alpha=0.85, color=color)

    x_line = np.logspace(np.log10(X.min()), np.log10(X.max()), 200)
    y_line = A * (x_line**alpha)
    plt.plot(
        x_line,
        y_line,
        color=color,
        linewidth=2,
        alpha=0.9,
        label=f"fit: y = {A:.3g} x^{alpha:.3f}, R^2={r2:.3f}",
    )

    # If requested and plotting vs params, project the loss at target parameter size
    if target_params_b is not None and x_axis == "params":
        x_target = float(target_params_b) * 1e9
        y_target = A * (x_target**alpha)
        guide_color = "teal"
        plt.axvline(x=x_target, color=guide_color, linewidth=1.2)
        plt.scatter(
            [x_target], [y_target], color=guide_color, marker="*", s=80, zorder=5
        )
        plt.annotate(
            f"{target_params_b}B → {y_metric}≈{y_target:.3f}",
            (x_target, y_target),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=8,
            color=guide_color,
        )

    # Annotate each point with model size and metric value
    for x_val, y_val, model_name in zip(X, Y, models):
        n_params = QWEN3_BASE_PARAMS_N.get(model_name)
        if n_params is None:
            continue
        model_size = QWEN3_MODEL_SIZES.get(model_name, f"{n_params / 1e9:.3g}B")
        if y_metric == "perplexity":
            label_txt = f"{model_size}, ppl={y_val:.3f}"
        else:
            label_txt = f"{model_size}, loss={y_val:.3f}"
        plt.annotate(
            label_txt,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(5, 4),
            ha="left",
            fontsize=8,
            color=color,
        )

    axis_label_x = (
        "FLOPs (C = 6ND)" if x_axis == "flops" else "Non-embedding parameters N"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(axis_label_x)
    plt.ylabel("Perplexity" if y_metric == "perplexity" else "Loss")
    plt.title(f"Power-law scaling: validation {y_metric} vs {axis_label_x}")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_avg_accuracy_scaling(
    results_path: Path,
    benchmark: str,
    save_path: Path | None,
    target_params_b: float | None = None,
    use_error_rate: bool = True,
) -> None:
    """Plot scaling law: average accuracy/error across shots vs model parameters.

    This function:
    1. Calculates average accuracy across all shots for each model
    2. Converts to error rate (100 - accuracy) for better scaling law fit
    3. Plots on log scale
    4. Fits power law: error = A * params^alpha (or accuracy if use_error_rate=False)
    5. Extrapolates to target parameter size if provided
    """
    # Discover all available shots for the benchmark
    shots = discover_available_shots(results_path, benchmark)
    if not shots:
        print(f"No shots found for {benchmark}")
        return

    # Load accuracy data for all shots
    shot_to_model_acc = load_accuracy_by_model(results_path, benchmark, shots)

    # Calculate average accuracy per model across all shots
    model_to_avg_acc: Dict[str, float] = {}
    for model in QWEN3_BASE_PARAMS_N:
        accuracies = []
        for shot in shots:
            if model in shot_to_model_acc[shot]:
                accuracies.append(shot_to_model_acc[shot][model])
        if accuracies:
            avg_acc = np.mean(accuracies)
            # Convert to error rate if requested
            if use_error_rate:
                model_to_avg_acc[model] = 100.0 - avg_acc
            else:
                model_to_avg_acc[model] = avg_acc

    if not model_to_avg_acc:
        print(f"No average accuracy data for {benchmark}")
        return

    # Prepare data for plotting (x-axis is params, not flops)
    X, Y, models = prepare_xy(model_to_avg_acc, "params")
    if len(X) == 0:
        print("No models with known parameter counts to plot.")
        return

    # Fit power law: y = A * x^alpha
    alpha, A, r2 = fit_power_law(X, Y)

    plt.figure(figsize=(10, 7))
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    # Plot data points
    plt.scatter(X, Y, s=100, alpha=0.85, color=color, zorder=3)

    # Plot fitted power law
    x_line = np.logspace(np.log10(X.min()), np.log10(X.max()), 200)
    y_line = A * (x_line**alpha)

    metric_name = "error rate" if use_error_rate else "accuracy"
    plt.plot(
        x_line,
        y_line,
        color=color,
        linewidth=2.5,
        alpha=0.9,
        label=f"Power law fit: {metric_name} = {A:.3g} × N^{{{alpha:.3f}}}\nR² = {r2:.3f}",
    )

    # Extrapolate to target parameter size
    if target_params_b is not None:
        x_target = float(target_params_b) * 1e9
        y_target = A * (x_target**alpha)

        # Extend the fitted line to target if needed
        if x_target > X.max():
            x_extended = np.logspace(np.log10(X.min()), np.log10(x_target * 1.1), 300)
            y_extended = A * (x_extended**alpha)
            plt.plot(
                x_extended[len(x_line) :],
                y_extended[len(x_line) :],
                color=color,
                linewidth=2.5,
                alpha=0.5,
                linestyle="--",
            )

        # Add visual guides for target
        guide_color = "red"
        plt.axvline(
            x=x_target, color=guide_color, linewidth=1.5, linestyle=":", alpha=0.7
        )
        plt.scatter(
            [x_target], [y_target], color=guide_color, marker="*", s=300, zorder=5
        )
        if use_error_rate:
            # Convert back to accuracy for display
            accuracy_at_target = 100.0 - y_target
            annotation_text = f"Estimated at {target_params_b}B:\nError rate ≈ {y_target:.2f}%\n(Accuracy ≈ {accuracy_at_target:.2f}%)"
        else:
            annotation_text = (
                f"Estimated at {target_params_b}B:\nAccuracy ≈ {y_target:.2f}%"
            )

        plt.annotate(
            annotation_text,
            (x_target, y_target),
            textcoords="offset points",
            xytext=(-80, -30),
            ha="center",
            fontsize=11,
            color=guide_color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=guide_color,
                alpha=0.8,
            ),
            arrowprops=dict(arrowstyle="->", color=guide_color, lw=1.5),
        )

    # Annotate each point with model size and metric value
    for x_val, y_val, model_name in zip(X, Y, models):
        model_size = QWEN3_MODEL_SIZES.get(
            model_name, f"{QWEN3_BASE_PARAMS_N[model_name] / 1e9:.1f}B"
        )
        if use_error_rate:
            label_text = f"{model_size}\n{y_val:.1f}% error"
        else:
            label_text = f"{model_size}\n{y_val:.1f}%"
        plt.annotate(
            label_text,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Non-embedding Parameters (N)", fontsize=12)
    if use_error_rate:
        plt.ylabel(f"Average {benchmark.upper()} Error Rate (%)", fontsize=12)
        plt.title(
            f"Scaling Law: {benchmark.upper()} Error Rate vs Model Size\n(Average across all shots)",
            fontsize=14,
        )
    else:
        plt.ylabel(f"Average {benchmark.upper()} Accuracy (%)", fontsize=12)
        plt.title(
            f"Scaling Law: {benchmark.upper()} Performance vs Model Size\n(Average across all shots)",
            fontsize=14,
        )
    plt.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved scaling law figure to {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-type",
        type=str,
        choices=["strict", "flexible", "flexible_custom"],
        default="strict",
        help="Choose baseline evaluation type: 'strict' (default) or 'flexible'",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(str(repo_root / "results" / "scaling_laws" / args.baseline_type))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select baseline file based on argument
    baselines_json = repo_root / "results" / f"baselines_{args.baseline_type}.json"
    val_losses_json = repo_root / "results" / "val_losses.json"

    # GSM8K: Plot accuracy vs shots for different model sizes
    try:
        shots = discover_available_shots(baselines_json, "gsm8k")
        if shots:
            fname = "gsm8k_shots_vs_accuracy.png"
            save_path = out_dir / fname
            plot_gsm8k_shots_vs_accuracy(
                results_path=baselines_json,
                shots=shots,
                save_path=save_path,
            )
    except Exception as e:
        print(f"Failed to generate GSM8K plot: {e}")

    # MMLU: Plot accuracy vs shots for different model sizes
    try:
        shots = discover_available_shots(baselines_json, "mmlu")
        if shots:
            fname = "mmlu_shots_vs_accuracy.png"
            save_path = out_dir / fname
            plot_mmlu_shots_vs_accuracy(
                results_path=baselines_json,
                shots=shots,
                save_path=save_path,
            )
    except Exception as e:
        print(f"Failed to generate MMLU plots: {e}")

    # For flexible_custom, also generate scaling law plots
    if args.baseline_type == "flexible_custom":
        # GSM8K scaling law plot
        try:
            fname = "gsm8k_scaling_law.png"
            save_path = out_dir / fname
            plot_avg_accuracy_scaling(
                results_path=baselines_json,
                benchmark="gsm8k",
                save_path=save_path,
                target_params_b=31.2,
            )
        except Exception as e:
            print(f"Failed to generate GSM8K scaling law plot: {e}")

        # MMLU scaling law plot - skipped as no shot data available in flexible_custom

    # Validation loss/perplexity plots
    out_dir = Path(str(repo_root / "results" / "scaling_laws"))
    if val_losses_json.exists():
        for metric in ["loss"]:  # "perplexity"
            for x_axis in ["params", "flops"]:
                fname = f"{val_losses_json.stem}_{metric}_{x_axis}.png"
                save_path = out_dir / fname
                plot_val_scaling(
                    results_path=val_losses_json,
                    y_metric=metric,
                    x_axis=x_axis,
                    target_params_b=31.2 if x_axis == "params" else None,
                    save_path=save_path,
                )
    else:
        print(
            f"No validation loss file found at {val_losses_json}; skipping validation plots."
        )


if __name__ == "__main__":
    main()
