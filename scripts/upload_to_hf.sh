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

# Script to upload models to HuggingFace as branches
# Usage: ./upload_to_hf_branches.sh <model_size> [repo_name]
# Example: ./upload_to_hf_branches.sh 0.6b my-qwen3-0.6b-grpo

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_size> [repo_name]"
    echo ""
    echo "Available model sizes:"
    echo "  0.6b  - Qwen3 0.6B GRPO model"
    echo "  1.7b  - Qwen3 1.7B GRPO model" 
    echo "  4b    - Qwen3 4B GRPO model"
    echo "  7b    - Qwen3 7B GRPO model"
    echo "  14b   - Qwen3 14B GRPO model"
    echo ""
    echo "Examples:"
    echo "  $0 0.6b                     # Upload as qwen3-0.6b-grpo"
    echo "  $0 1.7b my-custom-name      # Upload as my-custom-name"
    exit 1
fi

MODEL_SIZE="$1"
REPO_NAME="${2-}"

# Map model size to directory name
case "$MODEL_SIZE" in
    "0.6b")
        MODEL_DIR="qwen3_0.6b_grpo"
        DEFAULT_REPO_NAME="qwen3-0.6b-grpo-gsm8k"
        ;;
    "1.7b")
        MODEL_DIR="qwen3_1.7b_grpo"
        DEFAULT_REPO_NAME="qwen3-1.7b-grpo-gsm8k"
        ;;
    "4b")
        MODEL_DIR="qwen3_4b_grpo"
        DEFAULT_REPO_NAME="qwen3-4b-grpo-gsm8k"
        ;;
    "7b")
        MODEL_DIR="qwen3_7b_grpo"
        DEFAULT_REPO_NAME="qwen3-7b-grpo-gsm8k"
        ;;
    "14b")
        MODEL_DIR="qwen3_14b_grpo"
        DEFAULT_REPO_NAME="qwen3-14b-grpo-gsm8k"
        ;;
    *)
        echo "Error: Unknown model size '$MODEL_SIZE'"
        echo "Available sizes: 0.6b, 1.7b, 4b, 7b, 14b"
        exit 1
        ;;
esac

# Use default repo name if not provided
if [ -z "$REPO_NAME" ]; then
    REPO_NAME="$DEFAULT_REPO_NAME"
fi

# Get HuggingFace username and create full repo path
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

# Create full repo path with username
FULL_REPO_NAME="$HF_USERNAME/$REPO_NAME"

# Define base dir and auto-detect actual model directory by size token
BASE_DIR="$REPO_ROOT/checkpoints/verl_grpo_gsm8k"
SIZE_TOKEN_DOT="$MODEL_SIZE"
SIZE_TOKEN_US="${MODEL_SIZE/./_}"

# Gather candidate dirs
CANDIDATES=()
for d in "$BASE_DIR"/*; do
    if [ -d "$d" ]; then
        name="$(basename "$d")"
        case "$name" in
            *"$SIZE_TOKEN_DOT"*|*"$SIZE_TOKEN_US"*)
                CANDIDATES+=("$d")
                ;;
        esac
    fi
done

if [ ${#CANDIDATES[@]} -eq 0 ]; then
    echo "Error: No model directory found in $BASE_DIR matching size '$MODEL_SIZE'"
    echo "Existing dirs:"
    ls -1 "$BASE_DIR"
    exit 1
fi

# Pick most recently modified candidate
MODEL_BASE_PATH=$(ls -td "${CANDIDATES[@]}" 2>/dev/null | head -n 1)

MERGED_MODELS_DIR="$MODEL_BASE_PATH/merged_models"

echo "============================================"
echo "Uploading model to HuggingFace as branches"
echo "============================================"
echo "Model size: $MODEL_SIZE"
echo "Model base: $MODEL_BASE_PATH"
echo "Repository: $FULL_REPO_NAME"
echo ""

# Export env for embedded python blocks
export FULL_REPO_NAME
export MERGED_MODELS_DIR

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

# Upload models per step as branches
echo ""
echo "Starting upload..."
"$PYTHON_BIN" - <<'PYCODE'
import os
from huggingface_hub import HfApi, upload_folder

repo_name = os.environ['FULL_REPO_NAME']
merged_models_dir = os.environ['MERGED_MODELS_DIR']

def has_model_files(path: str) -> bool:
    return os.path.isfile(os.path.join(path, 'model.safetensors')) or \
           os.path.isfile(os.path.join(path, 'model.safetensors.index.json'))

api = HfApi()
print('Creating/updating repository: {}'.format(repo_name))
api.create_repo(repo_id=repo_name, exist_ok=True, private=False)

if not os.path.isdir(merged_models_dir):
    raise RuntimeError('No merged models directory found. Please run ./scripts/verl_to_hf.sh first.')

step_dirs = []
for name in sorted(os.listdir(merged_models_dir)):
    local_path = os.path.join(merged_models_dir, name)
    if os.path.isdir(local_path) and has_model_files(local_path):
        step_dirs.append((name, local_path))

if not step_dirs:
    raise RuntimeError('No merged models found in {}. Please run ./scripts/verl_to_hf.sh first.'.format(merged_models_dir))

# Upload main branch first (latest checkpoint)
latest_step_name, latest_step_path = step_dirs[-1]
print('Uploading latest step {} to main branch...'.format(latest_step_name))
upload_folder(
    folder_path=latest_step_path,
    repo_id=repo_name,
    repo_type='model',
    commit_message='Upload latest GRPO model ({})'.format(latest_step_name)
)
print('✓ Uploaded {} to main branch'.format(latest_step_name))

# Upload each step as a separate branch
for step_name, local_path in step_dirs:
    branch_name = step_name.replace('_', '-')  # HF branches prefer hyphens
    print('Uploading {} to branch {} ...'.format(step_name, branch_name))
    
    try:
        # Create branch and upload
        api.create_branch(repo_id=repo_name, branch=branch_name, exist_ok=True)
        upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            repo_type='model',
            revision=branch_name,
            commit_message='Upload GRPO model checkpoint {}'.format(step_name)
        )
        print('✓ Uploaded {} to branch {}'.format(step_name, branch_name))
    except Exception as e:
        print('Warning: Failed to upload {} to branch {}: {}'.format(step_name, branch_name, e))

print('\n✓ All steps uploaded to: https://huggingface.co/{}'.format(repo_name))
print('\nTo load a specific checkpoint:')
print('  model = AutoModelForCausalLM.from_pretrained("{}", revision="global-step-100")'.format(repo_name))
PYCODE

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Upload completed successfully!"
    echo "Model available at: https://huggingface.co/$FULL_REPO_NAME"
    echo "============================================"
else
    echo ""
    echo "Upload failed. Please check the error messages above."
    exit 1
fi
