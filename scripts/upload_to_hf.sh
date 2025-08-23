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

# Script to upload all models to HuggingFace as branches
# This script automatically:
#   1. Scans all directories in checkpoints/verl_grpo_gsm8k/
#   2. Uses the directory name as the HuggingFace repo name
#   3. Uploads all available checkpoint steps as branches
#
# Usage: ./upload_to_hf.sh [--prefix <prefix>]
# Examples:
#   ./upload_to_hf.sh                    # Upload with directory names as-is
#   ./upload_to_hf.sh --prefix myorg     # Upload with prefix (e.g., myorg-qwen3-0-6b-grpo-flexible)

# Parse optional prefix argument
PREFIX=""
if [ $# -eq 2 ] && [ "$1" = "--prefix" ]; then
    PREFIX="$2-"
fi

# Get HuggingFace username
HF_USERNAME=$("$PYTHON_BIN" -c "
from huggingface_hub import whoami
try:
    user_info = whoami()
    print(user_info['name'])
except Exception as e:
    print('ERROR')
    exit(1)
")

if [ "$HF_USERNAME" = "ERROR" ]; then
    echo "Error: Could not get HuggingFace username"
    exit 1
fi

# Define base dir
BASE_DIR="$REPO_ROOT/checkpoints/verl_grpo_gsm8k"

# Export env for embedded python blocks
export HF_USERNAME
export PREFIX
export BASE_DIR

# Check if user is logged in to HuggingFace
echo "Checking HuggingFace authentication..."
"$PYTHON_BIN" - <<'PYCODE'
from huggingface_hub import whoami
try:
    user_info = whoami()
    print('Logged in as: {}'.format(user_info['name']))
except Exception:
    print('Error: Not logged in to HuggingFace')
    print('Please run: huggingface-cli login')
    raise
PYCODE

if [ $? -ne 0 ]; then
    exit 1
fi

# Process all model directories
echo ""
echo "============================================"
echo "Uploading all models to HuggingFace"
echo "============================================"
echo "Base directory: $BASE_DIR"
echo ""

# Main upload logic
"$PYTHON_BIN" - <<'PYCODE'
import os
from huggingface_hub import HfApi, upload_folder
from pathlib import Path

base_dir = os.environ['BASE_DIR']
hf_username = os.environ['HF_USERNAME']
prefix = os.environ['PREFIX']

def has_model_files(path: str) -> bool:
    """Check if directory contains model files"""
    return os.path.isfile(os.path.join(path, 'model.safetensors')) or \
           os.path.isfile(os.path.join(path, 'model.safetensors.index.json'))

api = HfApi()

# Find all model directories
if not os.path.isdir(base_dir):
    raise RuntimeError('Base directory does not exist: {}'.format(base_dir))

model_dirs = []
for name in sorted(os.listdir(base_dir)):
    model_path = os.path.join(base_dir, name)
    merged_models_dir = os.path.join(model_path, 'merged_models')
    
    if os.path.isdir(model_path) and os.path.isdir(merged_models_dir):
        # Check if there are any model files in merged_models
        has_models = False
        for step_name in os.listdir(merged_models_dir):
            step_path = os.path.join(merged_models_dir, step_name)
            if os.path.isdir(step_path) and has_model_files(step_path):
                has_models = True
                break
        
        if has_models:
            model_dirs.append((name, model_path, merged_models_dir))

if not model_dirs:
    raise RuntimeError('No model directories with merged models found in {}'.format(base_dir))

print('Found {} model directories to upload:'.format(len(model_dirs)))
for model_name, _, _ in model_dirs:
    print('  - {}'.format(model_name))
print('')

# Upload each model
for model_name, model_path, merged_models_dir in model_dirs:
    # Create repo name from directory name
    repo_name = '{}{}'.format(prefix, model_name.replace('_', '-'))
    full_repo_name = '{}/{}'.format(hf_username, repo_name)
    
    print('----------------------------------------')
    print('Processing: {}'.format(model_name))
    print('Repository: {}'.format(full_repo_name))
    
    try:
        # Create repository
        api.create_repo(repo_id=full_repo_name, exist_ok=True, private=False)
        
        # Find all step directories
        step_dirs = []
        for step_name in sorted(os.listdir(merged_models_dir)):
            step_path = os.path.join(merged_models_dir, step_name)
            if os.path.isdir(step_path) and has_model_files(step_path):
                step_dirs.append((step_name, step_path))
        
        if not step_dirs:
            print('Warning: No model files found in {}, skipping...'.format(merged_models_dir))
            continue
        
        # Upload main branch (latest checkpoint)
        latest_step_name, latest_step_path = step_dirs[-1]
        print('  Uploading {} to main branch...'.format(latest_step_name))
        upload_folder(
            folder_path=latest_step_path,
            repo_id=full_repo_name,
            repo_type='model',
            commit_message='Upload latest checkpoint ({})'.format(latest_step_name)
        )
        
        # Upload each step as a separate branch
        for step_name, step_path in step_dirs:
            branch_name = step_name.replace('_', '-')
            print('  Uploading {} to branch {}...'.format(step_name, branch_name))
            
            try:
                api.create_branch(repo_id=full_repo_name, branch=branch_name, exist_ok=True)
                upload_folder(
                    folder_path=step_path,
                    repo_id=full_repo_name,
                    repo_type='model',
                    revision=branch_name,
                    commit_message='Upload checkpoint {}'.format(step_name)
                )
            except Exception as e:
                print('  Warning: Failed to upload {} to branch {}: {}'.format(step_name, branch_name, e))
        
        print('✓ Successfully uploaded {} to https://huggingface.co/{}'.format(model_name, full_repo_name))
        
    except Exception as e:
        print('✗ Failed to upload {}: {}'.format(model_name, e))

print('\n============================================')
print('Upload process completed!')
print('============================================')
PYCODE
