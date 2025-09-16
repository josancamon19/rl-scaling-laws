#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Plot GSM8k accuracy vs shots for different models')
    parser.add_argument('size', choices=['0.6b', '1.7b', '4b'],
                       help='Model size to analyze (0.6b, 1.7b, or 4b)')

    args = parser.parse_args()

    # Path to the results file
    results_file = f"/root/rl-scaling-laws/results/{args.size}.json"

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
    plt.title(f'GSM8k Accuracy vs Shots - {args.size.upper()} Models\n(Baselines vs Trained Models)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Create combined legend
    all_handles = baseline_legend_handles + trained_legend_handles
    all_labels = [f"Baseline ({method})" for method in baselines_data.keys()] + list(trained_models_data.keys())
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
    output_file = f"/root/rl-scaling-laws/results/{args.size}_gsm8k_accuracy_vs_shots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    plt.show()

if __name__ == "__main__":
    main()
