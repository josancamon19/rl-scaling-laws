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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to JSON results produced by eval/run.py",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["mmlu", "gsm8k"],
        default="gsm8k",
        help="Benchmark to visualize",
    )
    parser.add_argument(
        "--x-axis",
        type=str,
        choices=["flops", "params"],
        default="params",
        help="X axis variable (flops uses C = 6ND; params uses N)",
    )
    parser.add_argument(
        "--shots",
        nargs="*",
        type=int,
        default=[0],  # , 1, 2, 3, 4, 5],
        help="Shot counts to include",
    )
    parser.add_argument("--save", type=str, default=None, help="(Deprecated) Use --out-dir.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "results"),
        help="Directory to store generated figures (default: ./results at repo root)",
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        default=True,
        help="Generate figures for all benchmarks and all available shots",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_all:
        for benchmark in ["gsm8k", "mmlu"]:
            shots = discover_available_shots(Path(args.results), benchmark)
            if not shots:
                continue
            for x_axis in ["params", "flops"]:
                shots_part = "shots-" + "-".join(str(k) for k in shots)
                fname = f"{Path(args.results).stem}_{benchmark}_{x_axis}_{shots_part}.png"
                save_path = out_dir / fname
                plot_scaling(
                    results_path=Path(args.results),
                    benchmark=benchmark,
                    shots=shots,
                    x_axis=x_axis,
                    save_path=save_path,
                )
    else:
        # Single figure mode
        shots_part = "shots-" + "-".join(str(k) for k in args.shots)
        fname = f"{Path(args.results).stem}_{args.benchmark}_{args.x_axis}_{shots_part}.png"
        save_path = out_dir / fname
        plot_scaling(
            results_path=Path(args.results),
            benchmark=args.benchmark,
            shots=args.shots,
            x_axis=args.x_axis,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
