#!/usr/bin/env python3
"""
Plot MATH and MMLU performance for different model sizes.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os


def load_results(filepath: str) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_clean_model_name(model_name: str, size: str) -> str:
    """Get a clean, readable model name."""
    # Handle base models for different sizes
    if f'Qwen3-{size}-base' in model_name or f'Qwen/Qwen3-{size}-base' in model_name:
        return f'Qwen3-{size}-Base'

    # Handle different model types
    if 'flexible' in model_name:
        # Extract step number if present
        if 'global_step_' in model_name:
            step = model_name.split('global_step_')[-1]
            return f'Qwen3-{size}-GRPO-Flexible (step {step})'
        return f'Qwen3-{size}-GRPO-Flexible'
    elif 'strict' in model_name:
        # Extract step number if present
        if 'global_step_' in model_name:
            step = model_name.split('global_step_')[-1]
            return f'Qwen3-{size}-GRPO-Strict (step {step})'
        return f'Qwen3-{size}-GRPO-Strict'
    elif 'with-format' in model_name:
        # Extract step number if present
        if 'global_step_' in model_name:
            step = model_name.split('global_step_')[-1]
            return f'Qwen3-{size}-GRPO-Flexible-Format (step {step})'
        return f'Qwen3-{size}-GRPO-Flexible-Format'

    return model_name


def extract_subject_performance(results: dict):
    """
    Extract subject-wise performance for each model.
    Returns: (models, subjects, performance_matrix)
    """
    models = []
    all_subjects = set()

    # First pass: collect all models and subjects
    for model_data in results['models']:
        model_name = model_data['model']

        # Get MATH results (0-shot only)
        math_results = model_data.get('benchmarks', {}).get('math', {}).get('by_shot', {}).get('0', {})
        subject_stats = math_results.get('subject_stats', {})

        if subject_stats:
            models.append(model_name)
            all_subjects.update(subject_stats.keys())

    # Sort subjects for consistent ordering
    subjects = sorted(list(all_subjects))

    # Second pass: build performance matrix
    performance_matrix = []

    for model_data in results['models']:
        math_results = model_data.get('benchmarks', {}).get('math', {}).get('by_shot', {}).get('0', {})
        subject_stats = math_results.get('subject_stats', {})

        if subject_stats:
            model_performance = []
            for subject in subjects:
                if subject in subject_stats:
                    stats = subject_stats[subject]
                    accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
                    model_performance.append(accuracy)
                else:
                    model_performance.append(0)
            performance_matrix.append(model_performance)

    return models, subjects, performance_matrix


def extract_mmlu_performance(results: dict):
    """
    Extract MMLU performance for each model.
    Returns: (models, mmlu_scores)
    """
    models = []
    mmlu_scores = []

    for model_data in results['models']:
        model_name = model_data['model']

        # Get MMLU results (0-shot only)
        mmlu_results = model_data.get('benchmarks', {}).get('mmlu', {}).get('by_shot', {}).get('0', {})

        if mmlu_results and 'accuracy' in mmlu_results:
            models.append(model_name)
            mmlu_scores.append(mmlu_results['accuracy'])

    return models, mmlu_scores


def extract_math_average_performance(results: dict):
    """
    Extract average MATH performance for each model.
    Returns: (models, math_scores)
    """
    models = []
    math_scores = []

    for model_data in results['models']:
        model_name = model_data['model']

        # Get MATH results (0-shot only)
        math_results = model_data.get('benchmarks', {}).get('math', {}).get('by_shot', {}).get('0', {})
        subject_stats = math_results.get('subject_stats', {})

        if subject_stats:
            # Calculate average accuracy across all subjects
            total_correct = 0
            total_questions = 0

            for subject, stats in subject_stats.items():
                total_correct += stats.get('correct', 0)
                total_questions += stats.get('total', 0)

            if total_questions > 0:
                avg_accuracy = (total_correct / total_questions) * 100
                models.append(model_name)
                math_scores.append(avg_accuracy)

    return models, math_scores


def plot_math_subject_performance(size: str):
    """Create scatter plot of MATH performance by subject."""
    # Load results
    results_file = f"/root/rl-scaling-laws/results/{size}.json"

    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        return

    results = load_results(results_file)

    # Extract data
    models, subjects, performance_matrix = extract_subject_performance(results)

    if not models:
        print("No models with MATH results found!")
        return

    # Create figure
    plt.figure(figsize=(14, 8))

    # Color palette for different models
    colors = ['#2E86AB', '#E63946', '#42B883', '#F4A261', '#8E44AD']  # Extended color palette

    # Plot each model
    for i, (model, performance) in enumerate(zip(models, performance_matrix)):
        # Create x positions with small offset for each model to avoid overlap
        x_positions = np.arange(len(subjects)) + (i - len(models)/2) * 0.05

        plt.scatter(x_positions, performance,
                   color=colors[i % len(colors)],
                   s=80,  # Marker size
                   alpha=0.8,
                   label=get_clean_model_name(model, size),
                   edgecolors='black',
                   linewidth=1.5)

    # Customize plot
    plt.xlabel('Subject', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'MATH Performance by Subject - {size.upper()} Models', fontsize=14)

    # Set x-axis labels
    plt.xticks(range(len(subjects)), subjects, rotation=45, ha='right')

    # Add grid
    plt.grid(True, alpha=0.3, axis='y')

    # Set y-axis limits
    plt.ylim(0, 100)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Add horizontal line at base model average
    base_avg = performance_matrix[0]  # First model is base
    avg_score = np.mean(base_avg)
    plt.axhline(y=avg_score, color='gray', linestyle=':', alpha=0.4, linewidth=1.0, label=f'Base avg: {avg_score:.1f}%')

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_file = f"/root/rl-scaling-laws/results/{size}_math_by_subject.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"MATH plot saved to: {output_file}")

    # Also show plot
    plt.show()


def plot_mmlu_performance(size: str):
    """Create bar plot of MMLU performance."""
    # Load results
    results_file = f"/root/rl-scaling-laws/results/{size}.json"

    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        return

    results = load_results(results_file)

    # Extract data
    models, mmlu_scores = extract_mmlu_performance(results)

    if not models:
        print("No models with MMLU results found!")
        return

    # Create figure
    plt.figure(figsize=(12, 6))

    # Color palette for different models
    colors = ['#2E86AB', '#E63946', '#42B883', '#F4A261', '#8E44AD']  # Extended color palette

    # Create bar positions
    x_positions = np.arange(len(models))

    # Create bars
    bars = plt.bar(x_positions, mmlu_scores, color=[colors[i % len(colors)] for i in range(len(models))],
                    alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, mmlu_scores)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{score:.2f}%',
                ha='center', va='bottom', fontsize=11)

    # Customize plot
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('MMLU Accuracy (%)', fontsize=12)
    plt.title(f'MMLU Performance - {size.upper()} Models', fontsize=14)

    # Set x-axis labels
    plt.xticks(x_positions, [get_clean_model_name(m, size) for m in models], rotation=15, ha='right')

    # Add grid
    plt.grid(True, alpha=0.3, axis='y')

    # Set y-axis limits
    plt.ylim(0, max(mmlu_scores) * 1.15)  # 15% padding above max

    # Add horizontal line at base model score
    base_score = mmlu_scores[0]  # First model is base
    plt.axhline(y=base_score, color='gray', linestyle=':', alpha=0.4, linewidth=1.0, label=f'Base: {base_score:.2f}%')

    # Add legend
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_file = f"/root/rl-scaling-laws/results/{size}_mmlu_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"MMLU plot saved to: {output_file}")

    # Also show plot
    plt.show()


def plot_cross_model_math_comparison():
    """Create a cross-model comparison plot showing MATH performance for different sizes."""
    # Available model sizes
    sizes = ['0.6b', '1.7b', '4b', '8b', '14b']

    # Data structure to hold results
    baseline_data = {'strict': {}, 'custom_flexible': {}}
    trained_data = {'strict': {}, 'custom_flexible': {}}

    for size in sizes:
        results_file = f"/root/rl-scaling-laws/results/{size}.json"

        if not os.path.exists(results_file):
            print(f"Warning: Results file {results_file} not found, skipping {size}")
            continue

        # Load the JSON data
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Extract data for this size
        for model in data['models']:
            if 'benchmarks' in model and 'math' in model['benchmarks']:
                math_data = model['benchmarks']['math']
                if 'by_shot' in math_data:
                    math_results = math_data['by_shot'].get('0', {})
                    subject_stats = math_results.get('subject_stats', {})

                    if subject_stats:
                        # Calculate average accuracy across all subjects
                        total_correct = sum(stats.get('correct', 0) for stats in subject_stats.values())
                        total_questions = sum(stats.get('total', 0) for stats in subject_stats.values())

                        if total_questions > 0:
                            avg_accuracy = (total_correct / total_questions) * 100

                            # Check if this is a baseline model
                            is_baseline = 'metadata' in model and 'evaluation_method' in model['metadata']

                            if is_baseline:
                                eval_method = model['metadata']['evaluation_method']
                                if eval_method in ['strict', 'custom_flexible']:
                                    baseline_data[eval_method][size] = avg_accuracy
                            else:
                                # Check reward method for trained models
                                if 'metadata' in model and 'reward_method' in model['metadata']:
                                    reward_method = model['metadata']['reward_method']
                                    if reward_method in ['strict', 'custom_flexible']:
                                        trained_data[reward_method][size] = avg_accuracy

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
                else:
                    # Default: tooltip to the left for other sizes (original behavior for trained models)
                    plt.text(x_pos - 0.02, tooltip_y, label_text,
                            ha='right', va=va_align, fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    # Connect matching methods with dotted lines
    for method in ['strict', 'custom_flexible']:
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
    plt.ylabel('MATH Average Accuracy (%)', fontsize=14)
    plt.title('Cross-Model Comparison: Baseline vs Trained\nMATH Average 0-shot Accuracy', fontsize=16)

    plt.xticks(x_positions, ['Baseline', 'Trained'], fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    # Set y-axis limits with padding for vertical offsets
    plt.ylim(0, 100 + (len(sizes) * 0.8) + 5)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_file = "/root/rl-scaling-laws/results/cross_model_math_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Cross-model MATH comparison plot saved to: {output_file}")

    plt.show()


def plot_cross_model_mmlu_comparison():
    """Create a cross-model comparison plot showing MMLU performance for different sizes."""
    # Available model sizes
    sizes = ['0.6b', '1.7b', '4b']

    # Data structure to hold results
    baseline_data = {'strict': {}, 'custom_flexible': {}}
    trained_data = {'strict': {}, 'custom_flexible': {}}

    for size in sizes:
        results_file = f"/root/rl-scaling-laws/results/{size}.json"

        if not os.path.exists(results_file):
            print(f"Warning: Results file {results_file} not found, skipping {size}")
            continue

        # Load the JSON data
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Extract data for this size
        for model in data['models']:
            if 'benchmarks' in model and 'mmlu' in model['benchmarks']:
                mmlu_data = model['benchmarks']['mmlu']
                if 'by_shot' in mmlu_data:
                    mmlu_results = mmlu_data['by_shot'].get('0', {})

                    if mmlu_results and 'accuracy' in mmlu_results:
                        accuracy = mmlu_results['accuracy']

                        # Check if this is a baseline model
                        is_baseline = 'metadata' in model and 'evaluation_method' in model['metadata']

                        if is_baseline:
                            eval_method = model['metadata']['evaluation_method']
                            if eval_method in ['strict', 'custom_flexible']:
                                baseline_data[eval_method][size] = accuracy
                        else:
                            # Check reward method for trained models
                            if 'metadata' in model and 'reward_method' in model['metadata']:
                                reward_method = model['metadata']['reward_method']
                                if reward_method in ['strict', 'custom_flexible']:
                                    trained_data[reward_method][size] = accuracy

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
                else:
                    # Default: tooltip to the left for other sizes (original behavior for trained models)
                    plt.text(x_pos - 0.02, tooltip_y, label_text,
                            ha='right', va=va_align, fontsize=9, alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    # Connect matching methods with dotted lines
    for method in ['strict', 'custom_flexible']:
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
    plt.ylabel('MMLU Accuracy (%)', fontsize=14)
    plt.title('Cross-Model Comparison: Baseline vs Trained\nMMLU 0-shot Accuracy', fontsize=16)

    plt.xticks(x_positions, ['Baseline', 'Trained'], fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    # Set y-axis limits with padding for vertical offsets
    plt.ylim(0, 100 + (len(sizes) * 0.8) + 5)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_file = "/root/rl-scaling-laws/results/cross_model_mmlu_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Cross-model MMLU comparison plot saved to: {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot MATH and MMLU performance for different model sizes')
    parser.add_argument('size', nargs='?', choices=['0.6b', '1.7b', '4b', '8b', '14b'],
                       help='Model size to analyze (0.6b, 1.7b, or 4b). If not provided, creates cross-model comparison.')
    parser.add_argument('--cross-model', action='store_true',
                       help='Create cross-model comparison plot instead of single size plot')
    parser.add_argument('--benchmark', choices=['math', 'mmlu', 'both'], default='both',
                       help='Which benchmark to plot (default: both)')

    args = parser.parse_args()

    # If cross-model flag is set or no size provided, do cross-model comparison
    if args.cross_model or not args.size:
        if args.benchmark in ['math', 'both']:
            print("Generating cross-model MATH comparison plot...")
            plot_cross_model_math_comparison()
        if args.benchmark in ['mmlu', 'both']:
            print("Generating cross-model MMLU comparison plot...")
            plot_cross_model_mmlu_comparison()
        return

    # Otherwise, create the single-size plots
    if args.benchmark in ['math', 'both']:
        print(f"Generating MATH performance plot for {args.size}...")
        plot_math_subject_performance(args.size)

    if args.benchmark in ['mmlu', 'both']:
        print(f"\nGenerating MMLU performance plot for {args.size}...")
        plot_mmlu_performance(args.size)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
