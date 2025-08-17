#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


QWEN3_BASE_PARAMS_N: Dict[str, float] = {
    "Qwen/Qwen3-14B-base": 13.2e9,
    "Qwen/Qwen3-8B-base": 6.95e9,
    "Qwen/Qwen3-4B-base": 3.6e9,
    "Qwen/Qwen3-1.7B-base": 1.4e9,
    "Qwen/Qwen3-0.6B-base": 0.44e9,
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


def plot_scaling(
    results_path: Path,
    benchmark: str,
    shots: List[int],
    x_axis: str,
    save_path: Path | None,
) -> None:
    shot_to_model_acc = load_accuracy_by_model(results_path, benchmark, shots)

    plt.figure(figsize=(8, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, k in enumerate(shots):
        model_to_acc = shot_to_model_acc.get(k, {})
        if not model_to_acc:
            continue
        X, Y, models = prepare_xy(model_to_acc, x_axis)
        # Plot even if only one point; handle zeros on log-scale by clamping to a small epsilon
        if len(X) == 0:
            continue
        Y = np.asarray(Y, dtype=float)
        positive_mask = Y > 0
        if np.any(positive_mask):
            min_pos = float(np.min(Y[positive_mask]))
            eps = min_pos / 10.0
        else:
            eps = 0.01  # very small % so the point appears near the bottom
        Y_plot = Y.copy()
        Y_plot[~positive_mask] = eps
        # alpha, A, r2 = fit_power_law(X, Y)
        label = f"{k}-shot"  # y = {A:.3g} x^{alpha:.3f}, R^2={r2:.3f}
        color = colors[idx % len(colors)]
        plt.scatter(X, Y_plot, label=label, alpha=0.8, color=color)

        # x_line = np.logspace(np.log10(X.min()), np.log10(X.max()), 200)
        # y_line = A * (x_line**alpha)
        # plt.plot(x_line, y_line, color=color, linewidth=2, alpha=0.9)

        # Annotate each point with the model parameter size and raw accuracy (e.g., 0.6B, 42.1%)
        for x_val, y_plot_val, model_name, y_true_val in zip(X, Y_plot, models, Y):
            n_params = QWEN3_BASE_PARAMS_N.get(model_name)
            if n_params is None:
                continue
            label_txt = f"{n_params / 1e9:.3g}B, {y_true_val:.2f}%"
            plt.annotate(
                label_txt,
                (x_val, y_plot_val),
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
    plt.ylabel(f"{benchmark.upper()} accuracy (%)")
    plt.title(f"Power-law scaling: {benchmark.upper()} vs {axis_label_x}")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend()
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
        y_target = A * (x_target ** alpha)
        guide_color = "teal"
        plt.axvline(x=x_target, color=guide_color, linewidth=1.2)
        plt.scatter([x_target], [y_target], color=guide_color, marker="*", s=80, zorder=5)
        plt.annotate(
            f"{target_params_b:.1f}B → {y_metric}≈{y_target:.3f}",
            (x_target, y_target),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=8,
            color=guide_color,
        )

    # Annotate each point with N and metric value
    for x_val, y_val, model_name in zip(X, Y, models):
        n_params = QWEN3_BASE_PARAMS_N.get(model_name)
        if n_params is None:
            continue
        if y_metric == "perplexity":
            label_txt = f"{n_params / 1e9:.3g}B, ppl={y_val:.3f}"
        else:
            label_txt = f"{n_params / 1e9:.3g}B, loss={y_val:.3f}"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    # No required CLI arguments; defaults to generating all figures from repository result files
    parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(str(repo_root / "results" / "scaling_laws"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default input files under results/
    benchmarks_json = repo_root / "results" / "benchmarks.json"
    val_losses_json = repo_root / "results" / "val_losses.json"

    # Benchmark accuracy plots
    if benchmarks_json.exists():
        for benchmark in ["gsm8k", "mmlu"]:
            try:
                shots = discover_available_shots(benchmarks_json, benchmark)
            except Exception as e:
                print(f"Failed to read shots from {benchmarks_json}: {e}")
                shots = []
            if not shots:
                continue
            for x_axis in ["flops"]:  # "params",
                shots_part = "shots-" + "-".join(str(k) for k in shots)
                fname = f"{benchmarks_json.stem}_{benchmark}_{x_axis}_{shots_part}.png"
                save_path = out_dir / fname
                plot_scaling(
                    results_path=benchmarks_json,
                    benchmark=benchmark,
                    shots=shots,
                    x_axis=x_axis,
                    save_path=save_path,
                )
    else:
        print(
            f"No benchmark accuracy file found at {benchmarks_json}; skipping accuracy plots."
        )

    # Validation loss/perplexity plots
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
