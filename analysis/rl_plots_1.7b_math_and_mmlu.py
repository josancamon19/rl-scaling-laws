"""
Plot MATH and MMLU performance for different 1.7B models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(filepath: Path) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_clean_model_name(model_name: str) -> str:
    """Get a clean, readable model name."""
    if 'Qwen3-1.7B-base' in model_name:
        return 'Qwen3-1.7B-Base'
    elif 'flexible' in model_name:
        # Extract step number if present
        if 'global_step_' in model_name:
            step = model_name.split('global_step_')[-1]
            return f'Qwen3-1.7B-GRPO-Flexible (step {step})'
        return 'Qwen3-1.7B-GRPO-Flexible'
    elif 'strict' in model_name:
        # Extract step number if present
        if 'global_step_' in model_name:
            step = model_name.split('global_step_')[-1]
            return f'Qwen3-1.7B-GRPO-Strict (step {step})'
        return 'Qwen3-1.7B-GRPO-Strict'
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


def plot_math_subject_performance():
    """Create scatter plot of MATH performance by subject."""
    # Load results
    results_path = Path('/root/rl-scaling-laws/results/rl_results_1.7b_math_and_mmlu.json')
    results = load_results(results_path)
    
    # Extract data
    models, subjects, performance_matrix = extract_subject_performance(results)
    
    if not models:
        print("No models with MATH results found!")
        return
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Color palette for different models
    colors = ['#2E86AB', '#E63946', '#42B883']  # Blue for base, Red for flexible, Green for strict
    
    # Plot each model
    for i, (model, performance) in enumerate(zip(models, performance_matrix)):
        # Create x positions with small offset for each model to avoid overlap
        x_positions = np.arange(len(subjects)) + (i - len(models)/2) * 0.05
        
        plt.scatter(x_positions, performance, 
                   color=colors[i], 
                   s=120,  # Marker size
                   alpha=0.8,
                   label=get_clean_model_name(model),
                   edgecolors='black',
                   linewidth=0.5)
    
    # Customize plot
    plt.xlabel('Subject', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('MATH Performance by Subject - 1.7B Models', fontsize=16)
    
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
    plt.axhline(y=avg_score, color='gray', linestyle='--', alpha=0.5, label=f'Base avg: {avg_score:.1f}%')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = Path('/root/rl-scaling-laws/results/rl_results_1.7b_math_by_subject.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"MATH plot saved to: {output_path}")
    
    # Also show plot
    plt.show()


def plot_mmlu_performance():
    """Create bar plot of MMLU performance."""
    # Load results
    results_path = Path('/root/rl-scaling-laws/results/rl_results_1.7b_math_and_mmlu.json')
    results = load_results(results_path)
    
    # Extract data
    models, mmlu_scores = extract_mmlu_performance(results)
    
    if not models:
        print("No models with MMLU results found!")
        return
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Color palette for different models
    colors = ['#2E86AB', '#E63946', '#42B883']  # Blue for base, Red for flexible, Green for strict
    
    # Create bar positions
    x_positions = np.arange(len(models))
    
    # Create bars
    bars = plt.bar(x_positions, mmlu_scores, color=colors[:len(models)], 
                    alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, mmlu_scores)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{score:.2f}%',
                ha='center', va='bottom', fontsize=12)
    
    # Customize plot
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('MMLU Accuracy (%)', fontsize=14)
    plt.title('MMLU Performance - 1.7B Models', fontsize=16)
    
    # Set x-axis labels
    plt.xticks(x_positions, [get_clean_model_name(m) for m in models], rotation=15, ha='right')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits
    plt.ylim(0, max(mmlu_scores) * 1.15)  # 15% padding above max
    
    # Add horizontal line at base model score
    base_score = mmlu_scores[0]  # First model is base
    plt.axhline(y=base_score, color='gray', linestyle='--', alpha=0.5, label=f'Base: {base_score:.2f}%')
    
    # Add legend
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = Path('/root/rl-scaling-laws/results/rl_results_1.7b_mmlu_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"MMLU plot saved to: {output_path}")
    
    # Also show plot
    plt.show()


def main():
    """Generate both MATH and MMLU plots."""
    print("Generating MATH performance plot...")
    plot_math_subject_performance()
    
    print("\nGenerating MMLU performance plot...")
    plot_mmlu_performance()
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
