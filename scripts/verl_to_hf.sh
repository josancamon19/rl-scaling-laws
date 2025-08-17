#!/bin/bash

set -euo pipefail

# Resolve repo root and python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        PYTHON_BIN="python"
    fi
fi

# Base checkpoint directory
CHECKPOINT_DIR="$REPO_ROOT/checkpoints/verl_grpo_gsm8k"

# Find all model directories (those containing latest_checkpointed_iteration.txt)
for model_dir in "$CHECKPOINT_DIR"/*; do
    if [ -d "$model_dir" ] && [ -f "$model_dir/latest_checkpointed_iteration.txt" ]; then
        # Get model name
        model_name=$(basename "$model_dir")
        echo "Processing model: $model_name"
        echo "  Looking for checkpoints in: $model_dir"
        
        found_any=false
        for step_dir in "$model_dir"/global_step_*; do
            if [ -d "$step_dir" ]; then
                found_any=true
                step_name=$(basename "$step_dir")
                echo "  Found checkpoint: $step_name"
                
                source_dir="$step_dir/actor"
                target_dir="$model_dir/merged_models/$step_name"
                
                # Check if source directory exists
                if [ ! -d "$source_dir" ]; then
                    echo "    WARN: Source directory not found: $source_dir"
                    continue
                fi
                
                # Ensure target directory exists
                mkdir -p "$target_dir"
                
                # Check if already merged
                if [ -f "$target_dir/model.safetensors" ] || [ -f "$target_dir/model.safetensors.index.json" ]; then
                    echo "    Already merged at: $target_dir"
                    echo "    Skipping..."
                    continue
                fi
                
                # Run the merge command
                echo "    Merging FSDP checkpoint to HuggingFace format..."
                echo "    Source: $source_dir"
                echo "    Target: $target_dir"
                
                "$PYTHON_BIN" -m verl.model_merger merge \
                    --backend fsdp \
                    --local_dir "$source_dir" \
                    --target_dir "$target_dir"
                
                if [ $? -eq 0 ]; then
                    echo "    ✓ Successfully merged $step_name"
                else
                    echo "    ✗ Failed to merge $step_name"
                fi
                echo ""
            fi
        done
        
        if [ "$found_any" = false ]; then
            echo "  No checkpoints found in $model_dir"
        fi
        
        echo ""
    fi
done

echo "All models processed!"