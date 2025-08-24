"""
Plot MATH performance by subject for different 0.6B models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(filepath: Path) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


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


def get_clean_model_name(model_name: str) -> str:
    """Get a clean, readable model name."""
    if 'Qwen3-0.6B-base' in model_name:
        return 'Qwen3-0.6B-Base'
    elif 'qwen3-0.6b-grpo' in model_name:
        # Extract step number if present
        if 'global_step_' in model_name:
            step = model_name.split('global_step_')[-1]
            return f'Qwen3-0.6B-GRPO (step {step})'
        return 'Qwen3-0.6B-GRPO'
    return model_name


def plot_subject_performance():
    """Create scatter plot of performance by subject."""
    # Load results
    results_path = Path('/workspace/rl-scaling-laws/results/rl_results_0.6b_3_math_transfer.json')
    results = load_results(results_path)
    
    # Extract data
    models, subjects, performance_matrix = extract_subject_performance(results)
    
    if not models:
        print("No models with MATH results found!")
        return
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Color palette for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # Plot each model
    for i, (model, performance) in enumerate(zip(models, performance_matrix)):
        # Create x positions with small offset for each model to avoid overlap
        x_positions = np.arange(len(subjects)) + (i - len(models)/2) * 0.05
        
        plt.scatter(x_positions, performance, 
                   color=colors[i], 
                   s=100,  # Marker size
                   alpha=0.8,
                   label=get_clean_model_name(model),
                   edgecolors='black',
                   linewidth=0.5)
    
    # Customize plot
    plt.xlabel('Subject', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('MATH Performance by Subject - 0.6B Models', fontsize=16)
    
    # Set x-axis labels
    plt.xticks(range(len(subjects)), subjects, rotation=45, ha='right')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits
    plt.ylim(0, 100)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add horizontal line at 25% (average baseline performance)
    plt.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='25% baseline')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = Path('/workspace/rl-scaling-laws/results/math_performance_by_subject.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also show plot
    plt.show()


if __name__ == "__main__":
    plot_subject_performance()
