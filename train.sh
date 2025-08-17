#!/usr/bin/env bash
set -euo pipefail
set -x

# Parse command line arguments
MODEL_SIZE="${1:-14B}" 
# tried 1e-5, follows a very similar shape
LEARNING_RATE="${2:-1e-6}"  # Default learning rate
BATCH_SIZE="${3:-512}"  # Default batch size

# echo "Usage: $0 [MODEL_SIZE] [LEARNING_RATE] [BATCH_SIZE]"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/gsm8k}"

# Validate model size and construct paths
VALID_SIZES=("0.6B" "1.7B" "4B" "8B" "14B")
SIZE_FOUND=false

for valid_size in "${VALID_SIZES[@]}"; do
    if [[ "${MODEL_SIZE}" == "${valid_size}" ]]; then
        SIZE_FOUND=true
        break
    fi
done

if [[ "${SIZE_FOUND}" == "false" ]]; then
    echo "Error: Invalid model size '${MODEL_SIZE}'"
    echo "Valid sizes: ${VALID_SIZES[*]}"
    exit 1
fi


# Construct model path and name using concatenation
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-${MODEL_SIZE}-Base}"
MODEL_NAME="qwen3_${MODEL_SIZE,,}"  # Convert to lowercase
MODEL_NAME="${MODEL_NAME//./_}"  # Replace . with _

PROJECT_NAME="${PROJECT_NAME:-verl_grpo_gsm8k}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_NAME}_grpo_lr${LEARNING_RATE}_bs${BATCH_SIZE}}" 
export HF_HOME="${HF_HOME:-/tmp/huggingface}"

echo "Training configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Experiment: ${EXPERIMENT_NAME}"

# Ray optimization settings
# export RAY_DEDUP_LOGS=0
# export CUDA_LAUNCH_BLOCKING=0
# export NCCL_DEBUG=INFO

# Pre-download model if not cached
(
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE  
  unset HF_DATASETS_OFFLINE 
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
print('Checking model cache...')
try:
    # Try to load from cache first
    model = AutoModelForCausalLM.from_pretrained('${MODEL_PATH}', 
                                                torch_dtype=torch.bfloat16, 
                                                cache_dir='${HF_HOME}',
                                                local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained('${MODEL_PATH}', 
                                            cache_dir='${HF_HOME}',
                                            local_files_only=True)
    print('Model already in cache')
except Exception as e:
    print(f'Downloading model: {e}')
    # Download if not in cache
    model = AutoModelForCausalLM.from_pretrained('${MODEL_PATH}', 
                                                torch_dtype=torch.bfloat16, 
                                                cache_dir='${HF_HOME}')
    tokenizer = AutoTokenizer.from_pretrained('${MODEL_PATH}', 
                                            cache_dir='${HF_HOME}')
    print('Model downloaded and cached')
"
)

# export HF_HUB_OFFLINE=1  # Use cached models only, don't check online
# export TRANSFORMERS_OFFLINE=1  # Prevent transformers from checking online
# export HF_DATASETS_OFFLINE=1  # Prevent datasets from checking online

# Detect JSONL and convert to Parquet if needed
ALT_DATA_DIR="${ROOT_DIR}/data"
if [[ ! -f "${DATA_DIR}/train.jsonl" && -f "${ALT_DATA_DIR}/train.jsonl" ]]; then
  DATA_DIR="${ALT_DATA_DIR}"
fi

TRAIN_JSON="${DATA_DIR}/train.jsonl"
TRAIN_PARQUET="${DATA_DIR}/train.parquet"
TEST_JSON="${DATA_DIR}/test.jsonl"
TEST_PARQUET="${DATA_DIR}/test.parquet"

if [[ -f "${TRAIN_JSON}" && ! -f "${TRAIN_PARQUET}" ]]; then
  echo "Converting ${TRAIN_JSON} -> ${TRAIN_PARQUET}"
  python3 "${ROOT_DIR}/jsonl_to_parquet.py" "$TRAIN_JSON" "$TRAIN_PARQUET"
fi

if [[ -f "${TEST_JSON}" && ! -f "${TEST_PARQUET}" ]]; then
  echo "Converting ${TEST_JSON} -> ${TEST_PARQUET}"
  python3 "${ROOT_DIR}/jsonl_to_parquet.py" "$TEST_JSON" "$TEST_PARQUET"
fi


ARGS=(
  # actor refers to policy being trained, rollout to policy sampling
  # grpo is just ppo with a diff way of estimating adv, thus is considered still ppo in verl
  algorithm.adv_estimator=grpo
  # basic node config
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  trainer.save_freq=4
  trainer.test_freq=2 # for 7B, 1 for 14B, 4 for all else
  trainer.resume_mode=disable # Force new run, don't recover from checkpoints
  # dumb settings
  "data.train_files=${TRAIN_PARQUET}" 
  "data.val_files=${TEST_PARQUET}"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True # grad checkpointing, to fit bigger batch sizes, not FLOPs constrained in general
  actor_rollout_ref.model.trust_remote_code=True # hf models thing
  trainer.logger=['console, wandb']
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  # batch settings
  trainer.total_epochs=8 # passes over the data
  data.train_batch_size=${BATCH_SIZE} # gsm8k 7474 examples / this * epochs, 512 by default
  actor_rollout_ref.rollout.n=3 # batch_size generates n sized groups per prompt
  # we have now to process batch_size*3 to be process before .step() is called
  # actor_rollout_ref.actor.ppo_mini_batch_size=$((BATCH_SIZE / 4)) # .backward() called
  actor_rollout_ref.actor.ppo_mini_batch_size=512
  # each gpu process 1/n of mini batch_size ideally, then we call .backward()
  # for 8B 6 size, is using 85GB peak/95GB, with no grad checkpoint, 8 is using 42GB with grad cheeckpointing
  # takes half the time, try 16, and 64 for size_per_gpu now, 16 45GB wtf, nothing, 32 takes 56GB, going 64, only 65GB
  # could fit higher but nah, 64*8*3 fits nicely
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 # accumulate ppo_mini_batch_size/this*n_gpus times
  # making the log probs in parallel as well, consume less memory, cause no activations needed
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 # is sharded + no activations, can fit anything here, lol
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64
  
  data.max_prompt_length=512 # might cut a few prompts short
  data.max_response_length=512 # limit responses length to

  actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} # 1e-6 default
  actor_rollout_ref.actor.use_kl_loss=True # enable kl penalty
  actor_rollout_ref.actor.kl_loss_coef=0.001 # kl beta coefficient
  actor_rollout_ref.actor.kl_loss_type=low_var_kl # kl estimation method
  actor_rollout_ref.actor.entropy_coeff=0 # entropy bonus to encourage diverse responses
  actor_rollout_ref.rollout.name=vllm # backend for rollout's, ofc vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 # inference needs a lot of memory
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 # tensor parallelism, if n_gpus > batch size
  actor_rollout_ref.rollout.enforce_eager=False # False, if True for debugging compile results
  actor_rollout_ref.rollout.dtype=bfloat16 # dtype, ofc bf16, maybe fp8 for some stuff
  actor_rollout_ref.model.use_fused_kernels=True  # Enable fused kernels
  algorithm.use_kl_in_reward=False # kl penalty into the reward signal not only loss, nah, not a clean reward
  trainer.critic_warmup=0 
)

python3 -m verl.trainer.main_ppo "${ARGS[@]}"

# TODO: squeezing more memory, check each process in nvidia-smi, has different mem consumption, tweak params that way
# should run 4B again, and then 14B.