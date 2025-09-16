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


def main():
    parser = argparse.ArgumentParser(description='Plot MATH and MMLU performance for different model sizes')
    parser.add_argument('size', choices=['0.6b', '1.7b', '4b'],
                       help='Model size to analyze (0.6b, 1.7b, or 4b)')

    args = parser.parse_args()

    print(f"Generating MATH performance plot for {args.size}...")
    plot_math_subject_performance(args.size)

    print(f"\nGenerating MMLU performance plot for {args.size}...")
    plot_mmlu_performance(args.size)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
