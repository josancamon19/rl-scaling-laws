#!/usr/bin/env python3
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple


def load_model_accuracies(benchmarks_json: Path, model_size: str, benchmark: str = "gsm8k", shot: int = 0) -> Tuple[float, List[Tuple[int, float]]]:
    """Load baseline and GRPO training accuracies for a specific model size.
    
    Returns:
        (baseline_accuracy, [(step, accuracy), ...])
    """
    data = json.loads(benchmarks_json.read_text())
    
    # Map model size to expected patterns
    size_map = {
        "0.6B": ("Qwen/Qwen3-0.6B-base", "qwen3-0.6b-grpo-global_step_"),
        "1.7B": ("Qwen/Qwen3-1.7B-base", "qwen3-1.7b-grpo-global_step_"),
        "4B": ("Qwen/Qwen3-4B-base", "qwen3-4b-grpo-global_step_"),
        "8B": ("Qwen/Qwen3-8B-base", "qwen3-8b-grpo-global_step_"),
        "14B": ("Qwen/Qwen3-14B-base", "qwen3-14b-grpo-global_step_"),
    }
    
    if model_size not in size_map:
        raise ValueError(f"Model size {model_size} not supported. Available: {list(size_map.keys())}")
    
    base_model_name, grpo_prefix = size_map[model_size]
    baseline_accuracy = None
    grpo_steps = []
    
    for model_entry in data.get("models", []):
        model_name = model_entry.get("model")
        benchmarks = model_entry.get("benchmarks", {})
        bench_data = benchmarks.get(benchmark, {})
        by_shot = bench_data.get("by_shot", {})
        shot_data = by_shot.get(str(shot))
        
        if not shot_data or "error" in shot_data:
            continue
            
        accuracy = shot_data.get("accuracy")
        if accuracy is None:
            continue
            
        # Check if this is the baseline model
        if model_name == base_model_name:
            baseline_accuracy = accuracy
        # Check if this is a GRPO model for our size
        elif model_name.startswith(grpo_prefix):
            try:
                step_num = int(model_name.split("global_step_")[-1])
                grpo_steps.append((step_num, accuracy))
            except (ValueError, IndexError):
                continue
    
    if baseline_accuracy is None:
        print(f"Warning: No baseline accuracy found for {base_model_name}")
        baseline_accuracy = 0.0
        
    # Sort GRPO steps by step number
    grpo_steps.sort(key=lambda x: x[0])
    
    return baseline_accuracy, grpo_steps


def plot_rl_progression(model_size: str, baseline_acc: float, grpo_steps: List[Tuple[int, float]], 
                       benchmark: str = "gsm8k", save_path: Path = None):
    """Plot RL training progression starting from baseline."""
    
    plt.figure(figsize=(10, 6))
    
    # Plot baseline point
    plt.scatter([0], [baseline_acc], color='blue', s=100, label=f'{model_size} Baseline', zorder=3)
    
    if grpo_steps:
        # Plot GRPO progression
        grpo_steps_only = [step for step, _ in grpo_steps]
        grpo_accs_only = [acc for _, acc in grpo_steps]
        
        plt.plot(grpo_steps_only, grpo_accs_only, 'ro-', linewidth=2, markersize=6, 
                label='GRPO Training', alpha=0.8)
        
        # Connect baseline to first GRPO step
        first_step, first_acc = grpo_steps[0]
        plt.plot([0, first_step], [baseline_acc, first_acc], 'r--', alpha=0.5, linewidth=1)
        
        # Set x-axis range to include all steps
        max_step = max(grpo_steps_only)
        plt.xlim(-max_step*0.05, max_step*1.05)
        
        # Show improvement
        final_step, final_acc = grpo_steps[-1]
        improvement = final_acc - baseline_acc
        plt.text(0.7 * final_step, 0.5 * (baseline_acc + final_acc), 
                f'+{improvement:.1f}%\nimprovement', 
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
                
        # Annotate final step
        plt.annotate(f'Step {final_step}: {final_acc:.1f}%', (final_step, final_acc), 
                    textcoords="offset points", xytext=(10, 10), ha="left", fontsize=9)
    else:
        # No GRPO steps available - show baseline only
        plt.xlim(-1, 5)
        plt.text(2.5, baseline_acc + 5, 'No RL training data available', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    # Annotations for baseline
    plt.annotate(f'Baseline: {baseline_acc:.1f}%', (0, baseline_acc), 
                textcoords="offset points", xytext=(10, 10), ha="left", fontsize=9)
    
    plt.xlabel('Training Step')
    plt.ylabel(f'{benchmark.upper()} Accuracy (%)')
    title = f'{model_size} Model: {benchmark.upper()} RL Training Progression'
    if not grpo_steps:
        title += ' (Baseline Only)'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set reasonable y-axis limits
    all_accs = [baseline_acc] + [acc for _, acc in grpo_steps]
    y_min = max(0, min(all_accs) - 5)
    y_max = max(all_accs) + 10
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot RL training progression for a specific model size")
    parser.add_argument("model_size", help="Model size (e.g., 0.6B, 1.7B, 4B, 8B, 14B)")
    parser.add_argument("--benchmark", default="gsm8k", help="Benchmark to plot (default: gsm8k)")
    parser.add_argument("--shot", type=int, default=0, help="Number of shots (default: 0)")
    parser.add_argument("--save", help="Save path for the plot (optional)")
    
    args = parser.parse_args()
    
    # Find benchmarks.json
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    benchmarks_json = repo_root / "results" / "benchmarks.json"
    
    if not benchmarks_json.exists():
        print(f"Error: {benchmarks_json} not found")
        return
    
    try:
        baseline_acc, grpo_steps = load_model_accuracies(benchmarks_json, args.model_size, args.benchmark, args.shot)
        
        print(f"Found baseline accuracy: {baseline_acc:.2f}%")
        print(f"Found {len(grpo_steps)} GRPO training steps")
        
        if grpo_steps:
            print(f"Steps: {[step for step, _ in grpo_steps]}")
            print(f"Final accuracy: {grpo_steps[-1][1]:.2f}%")
        
        save_path = Path(args.save) if args.save else None
        plot_rl_progression(args.model_size, baseline_acc, grpo_steps, args.benchmark, save_path)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
