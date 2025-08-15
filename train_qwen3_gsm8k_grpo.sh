#!/usr/bin/env bash
set -euo pipefail
set -x

# Optimized GRPO training on GSM8K with Verl + Qwen/Qwen3-1.7B-Base

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/gsm8k}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B-Base}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_gsm8k}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_0.6b_grpo}"

# Ensure model is cached
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
# export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/huggingface/transformers}"

# Ray optimization settings
# export RAY_DEDUP_LOGS=0
# export CUDA_LAUNCH_BLOCKING=0
# export NCCL_DEBUG=INFO

# Pre-download model if not cached
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Checking model cache...')
try:
    model = AutoModelForCausalLM.from_pretrained('${MODEL_PATH}', torch_dtype=torch.bfloat16, cache_dir='${HF_HOME}')
    tokenizer = AutoTokenizer.from_pretrained('${MODEL_PATH}', cache_dir='${HF_HOME}')
    print('Model ready in cache')
except Exception as e:
    print(f'Model download needed: {e}')
"

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
  trainer.n_gpus_per_node=2
  trainer.nnodes=1
  trainer.save_freq=10
  trainer.test_freq=3
  # dumb settings
  "data.train_files=${TRAIN_PARQUET}" 
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  actor_rollout_ref.model.enable_gradient_checkpointing=False # grad checkpointing
  actor_rollout_ref.model.trust_remote_code=True # hf models thing
  trainer.logger=['console, wandb']
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  # batch settings
  trainer.total_epochs=4 # passes over the data
  data.train_batch_size=512 # gsm8k 7474 examples / this * epochs
  actor_rollout_ref.rollout.n=3 # batch_size generates n sized groups per prompt
  # we have now to process 256*3 to call optimizer.step()
  actor_rollout_ref.actor.ppo_mini_batch_size=128 # .backward() called
  # each gpu process 1/n of mini batch_size ideally, then we call .backward()
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32
  # making the log probs in parallel as well
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32
  
  data.max_prompt_length=512 # might cut a few prompts short
  data.max_response_length=512 # limit responses length to

  actor_rollout_ref.actor.optim.lr=1e-6 # learning rate
  actor_rollout_ref.actor.use_kl_loss=True # enable kl penalty
  actor_rollout_ref.actor.kl_loss_coef=0.001 # beta coefficient
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

if [[ -f "${TEST_PARQUET}" ]]; then
  ARGS+=( "data.val_files=${TEST_PARQUET}" )
fi

python3 -m verl.trainer.main_ppo "${ARGS[@]}"

# TODO: understand what each of the params does
# TODO: set it up dynamic on num gpus, automatically, detecting them all and using them all
# TODO: how to squeeze the most
# TODO: upload to hugginface/ or save the best checkpoint? limit gpu 