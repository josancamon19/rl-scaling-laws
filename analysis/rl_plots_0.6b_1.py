"""
Plot RL results alongside baseline results for 0.6B models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_results(filepath: Path) -> Dict:
    """Load results from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_performance_data(results: Dict) -> Dict[str, Dict[int, float]]:
    """
    Extract performance data from results.
    Returns dict mapping model name to shot->accuracy mapping.
    """
    performance_data = {}
    
    for model_data in results['models']:
        model_name = model_data['model']
        
        # Extract GSM8K accuracies for each shot
        if 'gsm8k' in model_data['benchmarks'] and 'by_shot' in model_data['benchmarks']['gsm8k']:
            shot_accuracies = {}
            for shot_str, shot_data in model_data['benchmarks']['gsm8k']['by_shot'].items():
                shot_num = int(shot_str)
                accuracy = shot_data['accuracy']
                shot_accuracies[shot_num] = accuracy
            
            performance_data[model_name] = shot_accuracies
    
    return performance_data


def get_model_display_name(model_name: str) -> str:
    """Get a clean display name for the model."""
    # Simplify model names for better readability
    if 'qwen3-0.6b-grpo-global_step_72' in model_name:
        return 'Qwen3-0.6B-GRPO (step 72)'
    elif 'qwen3-0-6b-grpo-flexible-with-format' in model_name:
        return 'Qwen3-0.6B-GRPO-Flexible-Format (step 56)'
    elif 'qwen3-0-6b-grpo-flexible-global_step_56' in model_name:
        return 'Qwen3-0.6B-GRPO-Flexible (step 56)'
    elif 'Qwen3-0.6B-base' in model_name:
        return 'Qwen3-0.6B-Base'
    else:
        return model_name


def plot_results():
    """Main plotting function."""
    # Define paths
    results_dir = Path('/workspace/rl-scaling-laws/results')
    
    # Define a list of distinct colors for each model
    color_palette = [
        '#e74c3c',  # Red
        '#3498db',  # Blue
        '#2ecc71',  # Green
        '#f39c12',  # Orange
        '#9b59b6',  # Purple
        '#1abc9c',  # Turquoise
        '#34495e',  # Dark Gray
        '#e67e22',  # Dark Orange
        '#16a085',  # Dark Turquoise
        '#8e44ad',  # Dark Purple
        '#2980b9',  # Dark Blue
        '#27ae60',  # Dark Green
    ]
    
    # Track color assignments
    color_index = 0
    model_colors = {}
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Load and plot RL results
    rl_results = load_results(results_dir / 'rl_results_0.6b_1.json')
    rl_data = extract_performance_data(rl_results)
    
    for model_name, shot_accuracies in rl_data.items():
        shots = sorted(shot_accuracies.keys())
        accuracies = [shot_accuracies[shot] for shot in shots]
        
        # Assign unique color
        if model_name not in model_colors:
            model_colors[model_name] = color_palette[color_index % len(color_palette)]
            color_index += 1
        
        plt.plot(shots, accuracies, 
                marker='o', 
                linestyle='--',  # Dashed line
                linewidth=1.5,
                markersize=6,
                color=model_colors[model_name],
                alpha=0.7,  # Light/transparent
                label=get_model_display_name(model_name))
    
    # Load and plot baseline results (only 0.6B models)
    baseline_files = ['baselines_strict.json', 'baselines_flexible.json', 'baselines_flexible_custom.json']
    
    for baseline_file in baseline_files:
        filepath = results_dir / baseline_file
        if filepath.exists():
            baseline_results = load_results(filepath)
            baseline_data = extract_performance_data(baseline_results)
            
            # Extract baseline type from filename
            baseline_type = baseline_file.replace('baselines_', '').replace('.json', '')
            
            # Only plot 0.6B models
            for model_name, shot_accuracies in baseline_data.items():
                if '0.6B' in model_name:
                    shots = sorted(shot_accuracies.keys())
                    accuracies = [shot_accuracies[shot] for shot in shots]
                    
                    # Create unique model identifier for color assignment
                    model_key = f"{model_name}_{baseline_type}"
                    
                    # Assign unique color
                    if model_key not in model_colors:
                        model_colors[model_key] = color_palette[color_index % len(color_palette)]
                        color_index += 1
                    
                    display_name = f'{get_model_display_name(model_name)} ({baseline_type})'
                    
                    plt.plot(shots, accuracies,
                            marker='s',
                            linestyle='--',  # Dashed line
                            linewidth=1.5,
                            markersize=6,
                            color=model_colors[model_key],
                            alpha=0.7,  # Light/transparent
                            label=display_name)
    
    # Customize plot
    plt.xlabel('Number of Shots', fontsize=14)
    plt.ylabel('GSM8K Accuracy (%)', fontsize=14)
    plt.title('GSM8K Performance: RL-trained vs Baseline 0.6B Models', fontsize=16)
    
    # Set x-axis to show integer shot values
    plt.xticks(range(6))
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Customize legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    output_path = Path('/workspace/rl-scaling-laws/results/rl_results_0.6b_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also display plot
    plt.show()


if __name__ == "__main__":
    plot_results()
