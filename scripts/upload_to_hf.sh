#!/bin/bash

# Script to upload models to HuggingFace
# Usage: ./upload_to_hf.sh <model_size> [repo_name]
# Example: ./upload_to_hf.sh 0.6b my-qwen3-0.6b-grpo

if [ $# -eq 0 ]; then
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
REPO_NAME="$2"

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
HF_USERNAME=$(.venv/bin/python -c "
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

# Define paths
BASE_DIR="checkpoints/verl_grpo_gsm8k"
MODEL_PATH="$BASE_DIR/$MODEL_DIR/merged_model"

# Check if merged model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Merged model not found at $MODEL_PATH"
    echo "Please run ./verl_to_hf.sh first to merge the model"
    exit 1
fi

if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    echo "Error: No model files found in $MODEL_PATH"
    echo "Please ensure the model is properly merged"
    exit 1
fi

echo "============================================"
echo "Uploading model to HuggingFace"
echo "============================================"
echo "Model size: $MODEL_SIZE"
echo "Source path: $MODEL_PATH"
echo "Repository: $FULL_REPO_NAME"
echo ""

# Check if user is logged in to HuggingFace
echo "Checking HuggingFace authentication..."
.venv/bin/python -c "
from huggingface_hub import whoami
try:
    user_info = whoami()
    print(f'Logged in as: {user_info[\"name\"]}')
except Exception as e:
    print('Error: Not logged in to HuggingFace')
    print('Please run: huggingface-cli login')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Upload the model
echo ""
echo "Starting upload..."
.venv/bin/python -c "
import os
from huggingface_hub import HfApi, Repository, upload_folder
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '$MODEL_PATH'
repo_name = '$FULL_REPO_NAME'

try:
    # Create repository if it doesn't exist
    api = HfApi()
    
    print(f'Creating/updating repository: {repo_name}')
    api.create_repo(repo_id=repo_name, exist_ok=True, private=False)
    
    # Upload the entire model directory
    print(f'Uploading model from {model_path}...')
    upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type='model',
        commit_message='Upload GRPO fine-tuned Qwen3 model'
    )
    
    print(f'✓ Model successfully uploaded to: https://huggingface.co/{repo_name}')
    
except Exception as e:
    print(f'✗ Upload failed: {e}')
    exit(1)
"

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
