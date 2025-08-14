#!/usr/bin/env bash
set -euo pipefail
set -x

# Minimal GRPO training on GSM8K with Verl + Qwen/Qwen3-0.6B-Base
# If JSONL files are present (train.jsonl / test.jsonl), they will be converted to Parquet automatically.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/gsm8k}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B-Base}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_gsm8k}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_0.6b_grpo}"

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
  algorithm.adv_estimator=grpo
  "data.train_files=${TRAIN_PARQUET}"
  data.train_batch_size=1024
  data.max_prompt_length=512
  data.max_response_length=512
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=80
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n=3
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20
  algorithm.use_kl_in_reward=False
  trainer.critic_warmup=0
  trainer.logger=['console','wandb']
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.save_freq=-1
  trainer.test_freq=5
  trainer.total_epochs=15
)

if [[ -f "${TEST_PARQUET}" ]]; then
  ARGS+=( "data.val_files=${TEST_PARQUET}" )
fi

python3 -m verl.trainer.main_ppo "${ARGS[@]}"


