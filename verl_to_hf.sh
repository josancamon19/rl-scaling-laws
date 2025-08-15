#!/bin/bash

# Base checkpoint directory
CHECKPOINT_DIR="checkpoints/verl_grpo_gsm8k"

# Find all model directories (those containing latest_checkpointed_iteration.txt)
for model_dir in "$CHECKPOINT_DIR"/*; do
    if [ -d "$model_dir" ] && [ -f "$model_dir/latest_checkpointed_iteration.txt" ]; then
        # Get model name
        model_name=$(basename "$model_dir")
        echo "Processing model: $model_name"
        
        # Read the latest checkpoint iteration
        latest_step=$(cat "$model_dir/latest_checkpointed_iteration.txt")
        echo "  Latest checkpoint: global_step_$latest_step"
        
        # Define paths
        source_dir="$model_dir/global_step_$latest_step/actor"
        target_dir="$model_dir/merged_model"
        
        # Check if source directory exists
        if [ ! -d "$source_dir" ]; then
            echo "  ERROR: Source directory not found: $source_dir"
            continue
        fi
        
        # Check if already merged
        if [ -f "$target_dir/model.safetensors" ]; then
            echo "  Model already merged at: $target_dir"
            echo "  Skipping..."
            continue
        fi
        
        # Run the merge command
        echo "  Merging FSDP checkpoint to HuggingFace format..."
        echo "  Source: $source_dir"
        echo "  Target: $target_dir"
        
        .venv/bin/python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$source_dir" \
            --target_dir "$target_dir"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully merged $model_name"
        else
            echo "  ✗ Failed to merge $model_name"
        fi
        echo ""
    fi
done

echo "All models processed!"