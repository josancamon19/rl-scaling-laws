#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_cross_model_comparison(include_strict=True):
    """Create a cross-model comparison plot showing baseline vs trained for different sizes.

    Args:
        include_strict: Whether to include strict method baselines/trained models.
                       If False, only custom_flexible method will be shown.
    """
    # Available model sizes
    sizes = ['0.6b', '1.7b', '4b', '8b', '14b']

    # Data structure to hold results
    methods_to_include = ['custom_flexible']
    if include_strict:
        methods_to_include.append('strict')

    baseline_data = {}
    trained_data = {}
    for method in methods_to_include:
        baseline_data[method] = {}
        trained_data[method] = {}

    for size in sizes:
        results_file = f"/root/rl-scaling-laws/results/{size}.json"

        if not os.path.exists(results_file):
            print(f"Warning: Results file {results_file} not found, skipping {size}")
            continue

        # Load the JSON data
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Extract data for this size
        print(f"\nProcessing {size}.json:")
        for model in data['models']:
            print(f"  Model: {model['model']}")
            if 'benchmarks' in model and 'gsm8k' in model['benchmarks']:
                gsm8k_data = model['benchmarks']['gsm8k']['by_shot']

                # Get 0-shot accuracy
                if '0' in gsm8k_data and isinstance(gsm8k_data['0'], dict) and 'accuracy' in gsm8k_data['0']:
                    accuracy = gsm8k_data['0']['accuracy']
                    print(f"    0-shot accuracy: {accuracy}")

                    # Check if this is a baseline model
                    is_baseline = 'metadata' in model and 'evaluation_method' in model['metadata']

                    if is_baseline:
                        eval_method = model['metadata']['evaluation_method']
                        print(f"    Evaluation method: {eval_method}")
                        if eval_method in methods_to_include:
                            baseline_data[eval_method][size] = accuracy
                            print(f"    ✓ Added to baseline_data[{eval_method}][{size}] = {accuracy}")
                        else:
                            print(f"    ✗ Skipped: evaluation_method '{eval_method}' not in methods_to_include {methods_to_include}")
                    else:
                        # Check reward method for trained models
                        if 'metadata' in model and 'reward_method' in model['metadata']:
                            reward_method = model['metadata']['reward_method']
                            print(f"    Reward method: {reward_method}")
                            if reward_method in methods_to_include:
                                trained_data[reward_method][size] = accuracy
                                print(f"    ✓ Added to trained_data[{reward_method}][{size}] = {accuracy}")
                            else:
                                print(f"    ✗ Skipped: reward_method '{reward_method}' not in methods_to_include {methods_to_include}")
                        else:
                            print(f"    ✗ Skipped: no reward_method in metadata")
            else:
                print(f"    ✗ No GSM8k data found")

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Colors for different methods
    colors = {'strict': '#2E86AB', 'custom_flexible': '#E63946'}

    # Plot baselines
    x_positions = [0, 1]  # baseline, trained

    for i, (method, method_data) in enumerate(baseline_data.items()):
        color = colors[method]

        # Sort sizes for consistent vertical ordering
        sorted_sizes = sorted(sizes)
        vertical_offset = 0.8  # Space between points vertically

        for j, size in enumerate(sorted_sizes):
            if size in method_data:
                # All baselines align vertically at x=0
                x_pos = x_positions[0]
                y_pos = method_data[size] + (j * vertical_offset)

                plt.scatter(x_pos, y_pos, color=color, s=80,
                           marker='s', edgecolors='black', linewidth=2,
                           label=f'Baseline ({method})' if j == 0 else None, alpha=0.8)

                # Add size label with accuracy
                label_text = f"{size.upper()}\n({method_data[size]:.1f}%)"
                # Position tooltip based on model size
                if size == '0.6b':
                    # 0.6b models: tooltip to the left
                    plt.text(x_pos - 0.02, y_pos, label_text,
                            ha='right', va='center', fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                elif size == '1.7b':
                    # 1.7b models: tooltip to the right
                    plt.text(x_pos + 0.02, y_pos, label_text,
                            ha='left', va='center', fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                elif size == '8b':
                    # 8b models: tooltip to the right
                    plt.text(x_pos + 0.02, y_pos, label_text,
                            ha='left', va='center', fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                elif size == '4b':
                    # 4b models: tooltip to the left
                    plt.text(x_pos - 0.02, y_pos, label_text,
                            ha='right', va='center', fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                elif size == '14b':
                    # 14b models: tooltip to the left
                    plt.text(x_pos - 0.02, y_pos, label_text,
                            ha='right', va='center', fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                else:
                    # Default: tooltip to the right for other sizes
                    plt.text(x_pos + 0.02, y_pos, label_text,
                            ha='left', va='center', fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    # Plot trained models
    for i, (method, method_data) in enumerate(trained_data.items()):
        color = colors[method]

        # Sort sizes for consistent vertical ordering (same as baselines)
        sorted_sizes = sorted(sizes)
        vertical_offset = 0.8  # Space between points vertically

        for j, size in enumerate(sorted_sizes):
            if size in method_data:
                # All trained models align vertically at x=1
                x_pos = x_positions[1]
                y_pos = method_data[size] + (j * vertical_offset)

                plt.scatter(x_pos, y_pos, color=color, s=80,
                           marker='o', edgecolors='black', linewidth=2,
                           label=f'Trained ({method})' if j == 0 else None, alpha=0.8)

                # Add size label with accuracy
                label_text = f"{size.upper()}\n({method_data[size]:.1f}%)"

                # Position tooltip based on method type (vertical) and model size (horizontal)
                y_offset = 0.5  # Vertical offset for tooltips
                if method == 'strict':
                    # strict method: tooltip at top
                    tooltip_y = y_pos + y_offset
                    va_align = 'bottom'
                else:  # custom_flexible
                    # custom_flexible method: tooltip at bottom
                    tooltip_y = y_pos - y_offset
                    va_align = 'top'

                if size == '0.6b':
                    # 0.6b models: tooltip to the left
                    plt.text(x_pos - 0.02, tooltip_y, label_text,
                            ha='right', va=va_align, fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                elif size == '1.7b':
                    # 1.7b models: tooltip to the right
                    plt.text(x_pos + 0.02, tooltip_y, label_text,
                            ha='left', va=va_align, fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                elif size == '8b':
                    # 8b models: tooltip to the right
                    plt.text(x_pos + 0.02, tooltip_y, label_text,
                            ha='left', va=va_align, fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                elif size == '4b':
                    # 4b models: tooltip to the left
                    plt.text(x_pos - 0.02, tooltip_y, label_text,
                            ha='right', va=va_align, fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                elif size == '14b':
                    # 14b models: tooltip to the left
                    plt.text(x_pos - 0.02, tooltip_y, label_text,
                            ha='right', va=va_align, fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                else:
                    # Default: tooltip to the left for other sizes (original behavior for trained models)
                    plt.text(x_pos - 0.02, tooltip_y, label_text,
                            ha='right', va=va_align, fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    # Connect matching methods with dotted lines
    for method in methods_to_include:
        color = colors[method]
        sorted_sizes = sorted(sizes)
        vertical_offset = 0.8

        for j, size in enumerate(sorted_sizes):
            baseline_val = baseline_data[method].get(size)
            trained_val = trained_data[method].get(size)

            if baseline_val is not None and trained_val is not None:
                # Calculate the adjusted y-positions (accounting for vertical offset)
                baseline_y = baseline_val + (j * vertical_offset)
                trained_y = trained_val + (j * vertical_offset)

                plt.plot([x_positions[0], x_positions[1]], [baseline_y, trained_y],
                        color=color, linestyle=':', linewidth=1.5, alpha=0.6)

    # Customize plot
    plt.xlabel('Model Type', fontsize=14)
    plt.ylabel('GSM8k Accuracy (%)', fontsize=14)

    # Update title based on methods included
    method_desc = "Strict & Custom Flexible" if include_strict else "Custom Flexible Only"
    plt.title(f'Cross-Model Comparison: Baseline vs Trained\nGSM8k 0-shot Accuracy ({method_desc})', fontsize=16)

    plt.xticks(x_positions, ['Baseline', 'Trained'], fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    # Set y-axis limits with padding for vertical offsets
    plt.ylim(0, 100 + (len(sizes) * 0.8) + 5)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save the plot
    method_suffix = "_with_strict" if include_strict else "_custom_flexible_only"
    output_file = f"/root/rl-scaling-laws/results/cross_model_gsm8k_comparison{method_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Cross-model comparison plot saved to: {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot GSM8k accuracy vs shots for different models')
    parser.add_argument('size', nargs='?', choices=['0.6b', '1.7b', '4b', '8b', '14b'],
                       help='Model size to analyze (0.6b, 1.7b, or 4b). If not provided, creates cross-model comparison.')
    parser.add_argument('--cross-model', action='store_true',
                       help='Create cross-model comparison plot instead of single size plot')
    parser.add_argument('--include-strict', action='store_true',
                       help='Include strict method baselines/trained models in cross-model comparison. '
                            'If not specified, only custom_flexible method will be shown.')

    args = parser.parse_args()

    # If cross-model flag is set or no size provided, do cross-model comparison
    if args.cross_model or not args.size:
        plot_cross_model_comparison(include_strict=args.include_strict)
        return

    # Otherwise, create the single-size plot
    plot_single_size(args.size)


def plot_single_size(size):
    """Create the original single-size plot."""
    # Path to the results file
    results_file = f"/root/rl-scaling-laws/results/{size}.json"

    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        return

    # Load the JSON data
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Extract GSM8k data from all models
    baselines_data = {}  # Group baselines by evaluation method
    trained_models_data = {}  # Store trained models separately

    for model in data['models']:
        model_name = model['model']
        model_id = model.get('model_id', model_name)

        # Use a more readable name for display
        display_name = model_name.split('/')[-1] if '/' in model_name else model_name

        if 'benchmarks' in model and 'gsm8k' in model['benchmarks']:
            gsm8k_data = model['benchmarks']['gsm8k']['by_shot']

            # Extract shots and accuracies
            shots = []
            accuracies = []

            for shot_str, shot_data in gsm8k_data.items():
                if isinstance(shot_data, dict) and 'accuracy' in shot_data:
                    shots.append(int(shot_str))
                    accuracies.append(shot_data['accuracy'])

            if shots and accuracies:
                # Sort by shots
                sorted_indices = np.argsort(shots)
                shots = [shots[i] for i in sorted_indices]
                accuracies = [accuracies[i] for i in sorted_indices]

                # Check if this is a baseline model (has evaluation_method in metadata)
                is_baseline = 'metadata' in model and 'evaluation_method' in model['metadata']

                if is_baseline:
                    evaluation_method = model['metadata']['evaluation_method']
                    if evaluation_method not in baselines_data:
                        baselines_data[evaluation_method] = []
                    baselines_data[evaluation_method].append({
                        'display_name': display_name,
                        'shots': shots,
                        'accuracies': accuracies,
                        'model_id': model_id
                    })
                else:
                    trained_models_data[display_name] = {
                        'shots': shots,
                        'accuracies': accuracies,
                        'model_id': model_id
                    }

    if not baselines_data and not trained_models_data:
        print("No GSM8k data found in the results file")
        return

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Color maps for different types
    baseline_colors = plt.cm.Set1(np.linspace(0, 1, len(baselines_data)))
    trained_colors = plt.cm.tab10(np.linspace(0, 1, len(trained_models_data)))

    # Plot baselines first (with thicker lines and different markers)
    baseline_legend_handles = []
    for i, (eval_method, baseline_models) in enumerate(baselines_data.items()):
        color = baseline_colors[i % len(baseline_colors)]

        for j, model_data in enumerate(baseline_models):
            shots = model_data['shots']
            accuracies = model_data['accuracies']
            label = f"Baseline ({eval_method})" if j == 0 else None

            # Plot points with different marker for baselines
            scatter = plt.scatter(shots, accuracies, color=color, s=80,
                                marker='s', edgecolors='black', linewidth=2, label=label)

            # Connect with very light dotted lines (lighter for baselines)
            if len(shots) > 1:
                plt.plot(shots, accuracies, color=color, linestyle=':', linewidth=1.0, )

            if j == 0:  # Only add to legend once per evaluation method
                baseline_legend_handles.append(scatter)

    # Plot trained models
    trained_legend_handles = []
    for i, (model_name, model_data) in enumerate(trained_models_data.items()):
        color = trained_colors[i % len(trained_colors)]

        shots = model_data['shots']
        accuracies = model_data['accuracies']

        # Plot points with circles for trained models
        scatter = plt.scatter(shots, accuracies, color=color, s=60, alpha=0.7, label=model_name)

        # Connect with light dotted lines
        if len(shots) > 1:
            plt.plot(shots, accuracies, color=color, linestyle=':', alpha=0.4, linewidth=1.0)

        trained_legend_handles.append(scatter)

    plt.xlabel('Number of Shots', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'GSM8k Accuracy vs Shots - {size.upper()} Models\n(Baselines vs Trained Models)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Create combined legend
    all_handles = baseline_legend_handles + trained_legend_handles
    all_labels = [f"Baseline ({method})" for method in baselines_data.keys()] + list(trained_models_data.keys())

    # Set legend position based on model size
    if size == '0.6b':
        # Bottom legend for 0.6b models
        plt.legend(all_handles, all_labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=10)
    elif size == '1.7b':
        # Top legend for 1.7b models
        plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        # Default top legend for other sizes
        plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Set x-axis ticks for shots
    all_shots = set()

    # Collect shots from baselines
    for baseline_models in baselines_data.values():
        for model_data in baseline_models:
            all_shots.update(model_data['shots'])

    # Collect shots from trained models
    for model_data in trained_models_data.values():
        all_shots.update(model_data['shots'])

    plt.xticks(sorted(all_shots))

    plt.tight_layout()

    # Save the plot
    output_file = f"/root/rl-scaling-laws/results/{size}_gsm8k_accuracy_vs_shots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    plt.show()

if __name__ == "__main__":
    main()
