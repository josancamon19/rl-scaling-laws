#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * (x**b)


def extract_gsm8k_data():
    """Extract GSM8K 0-shot accuracies for custom_flexible method."""
    sizes = ["0.6b", "1.7b", "4b", "8b", "14b"]

    baseline_data = {}
    trained_data = {}

    print("Extracting GSM8K data for custom_flexible method:")

    for size in sizes:
        results_file = f"/root/rl-scaling-laws/results/{size}.json"

        if not os.path.exists(results_file):
            print(f"Warning: Results file {results_file} not found, skipping {size}")
            continue

        # Load the JSON data
        with open(results_file, "r") as f:
            data = json.load(f)

        print(f"\nProcessing {size}.json:")

        # Extract data for this size
        for model in data["models"]:
            if "benchmarks" in model and "gsm8k" in model["benchmarks"]:
                gsm8k_data = model["benchmarks"]["gsm8k"]

                # Get 0-shot accuracy
                if "by_shot" in gsm8k_data and "0" in gsm8k_data["by_shot"]:
                    zero_shot_data = gsm8k_data["by_shot"]["0"]
                    if "accuracy" in zero_shot_data:
                        accuracy = zero_shot_data["accuracy"]
                        print(f"  Model: {model['model']}")
                        print(f"    0-shot accuracy: {accuracy:.2f}%")

                        # Check if this is a baseline model
                        is_baseline = (
                            "metadata" in model
                            and "evaluation_method" in model["metadata"]
                        )

                        if is_baseline:
                            eval_method = model["metadata"]["evaluation_method"]
                            print(f"    Evaluation method: {eval_method}")
                            if eval_method == "custom_flexible":
                                baseline_data[size] = accuracy
                                print(f"    ✓ Added baseline: {size} = {accuracy:.2f}%")
                        else:
                            # Check reward method for trained models
                            if (
                                "metadata" in model
                                and "reward_method" in model["metadata"]
                            ):
                                reward_method = model["metadata"]["reward_method"]
                                print(f"    Reward method: {reward_method}")
                                if reward_method == "custom_flexible":
                                    trained_data[size] = accuracy
                                    print(
                                        f"    ✓ Added trained: {size} = {accuracy:.2f}%"
                                    )
                            else:
                                print("    ✗ No reward_method in metadata")
    return baseline_data, trained_data


def plot_scaling_laws():
    """Create a log-log plot showing RL gains vs model size."""
    # Extract the data
    baseline_data, trained_data = extract_gsm8k_data()

    # Prepare data for plotting
    sizes = ["0.6b", "1.7b", "4b", "8b", "14b"]
    size_values = [0.6, 1.7, 4.0, 8.0, 14.0]  # Convert to billions
    gains = []

    print("\nCalculating RL gains:")
    valid_sizes = []
    valid_size_values = []

    for size, size_val in zip(sizes, size_values):
        baseline_acc = baseline_data.get(size)
        trained_acc = trained_data.get(size)

        if baseline_acc is not None and trained_acc is not None:
            gain = trained_acc - baseline_acc
            gains.append(gain)
            valid_sizes.append(size)
            valid_size_values.append(size_val)

            print(
                f"  {size}: Gain = {gain:.2f}% (baseline: {baseline_acc:.1f}%, trained: {trained_acc:.1f}%)"
            )
        else:
            print(
                f"  {size}: Missing data (baseline: {baseline_acc}, trained: {trained_acc})"
            )

    if len(gains) < 2:
        print("Not enough data points for scaling law analysis")
        return

    # Convert to numpy arrays for fitting
    x_data = np.array(valid_size_values)
    y_data = np.array(gains)

    print(f"\nFitting power law to {len(x_data)} data points...")

    # Fit power law
    try:
        # Use logarithmic fit to get better initial parameters
        log_x = np.log(x_data)
        log_y = np.log(y_data)

        # Linear fit in log space
        coeffs = np.polyfit(log_x, log_y, 1)
        a_fit = np.exp(coeffs[1])  # a = exp(intercept)
        b_fit = coeffs[0]  # b = slope

        print(f"    Power law: Gain = {a_fit:.3f} × (Model Size)^{b_fit:.3f}")

        # Generate points for the fitted curve
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = power_law(x_fit, a_fit, b_fit)

    except Exception as e:
        print(f"Power law fitting failed: {e}")
        print("Falling back to linear interpolation...")
        a_fit, b_fit = None, None
        x_fit, y_fit = None, None

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot the data points
    plt.scatter(
        x_data,
        y_data,
        color="#E63946",
        s=100,
        edgecolors="black",
        linewidth=2,
        label="Observed RL Gains",
        zorder=5,
    )

    # Add fitted curve if available
    if x_fit is not None and y_fit is not None:
        plt.plot(
            x_fit,
            y_fit,
            color="#2E86AB",
            linewidth=3,
            linestyle="--",
            label=f"Power Law Fit (exponent = {b_fit:.3f})",
            alpha=0.8,
        )

    # Add labels for each point
    for i, (size, x_val, gain) in enumerate(zip(valid_sizes, x_data, y_data)):
        plt.annotate(
            f"{size.upper()}\n+{gain:.1f}%",
            (x_val, gain),
            xytext=(0, 10 if i % 2 == 0 else -25),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Set log scales
    plt.xscale("log")
    plt.yscale("log")

    # Customize the plot
    plt.xlabel("Model Size (billions of parameters)", fontsize=14)
    plt.ylabel("RL Gain (trained - baseline accuracy, %)", fontsize=14)
    plt.title(
        "GSM8K RL Scaling Laws\n(custom_flexible method, 0-shot accuracy)",
        fontsize=16,
        fontweight="bold",
    )

    plt.grid(True, alpha=0.3, which="both")

    # Add power law equation if fitted
    if a_fit is not None and b_fit is not None:
        equation_text = f"Gain = {a_fit:.2f} × (Size)^{b_fit:.3f}"
        plt.text(
            0.02,
            0.98,
            equation_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )

    # Add legend
    plt.legend(loc="upper right", fontsize=12)

    plt.tight_layout()

    # Save the plot
    output_file = "/root/rl-scaling-laws/results/gsm8k_rl_scaling_laws.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nScaling laws plot saved to: {output_file}")

    plt.show()


def main():
    """Main function to run the scaling laws analysis."""
    plot_scaling_laws()


if __name__ == "__main__":
    main()
