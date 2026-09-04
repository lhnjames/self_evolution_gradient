#!/usr/bin/env bash
# CoEvoKG training example launcher.
#
# This is the only training shell entry point kept in the release package.
# It uses coevokg/config/train_example.yaml by default. Pass extra Hydra
# overrides after the script for model- or hardware-specific changes.

set -euo pipefail
ulimit -n 65535

export COEVOKG_ROOT=${COEVOKG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PYTHONPATH=${COEVOKG_ROOT}/verl:${COEVOKG_ROOT}
export HYDRA_FULL_ERROR=1

# -----------------------------------------------------------------------------
# User-provided paths. DATA_PATH and TEST_DATA_PATH omit the .parquet suffix.
# -----------------------------------------------------------------------------
export MODEL_PATH=${MODEL_PATH:-/path/to/base_model}
export DATA_PATH=${DATA_PATH:-${COEVOKG_ROOT}/examples/data/example_seed_question_pool}
export TEST_DATA_PATH=${TEST_DATA_PATH:-${COEVOKG_ROOT}/examples/data/example_validation_question_pool}
export CHAIN_DATA_NAS=${CHAIN_DATA_NAS:-${COEVOKG_ROOT}/examples/data/example_chain_pool.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-${COEVOKG_ROOT}/outputs/output_coevokg_example}

# -----------------------------------------------------------------------------
# External services.
# -----------------------------------------------------------------------------
export SEARCH_IP=${SEARCH_IP:-127.0.0.1}
export COEVOKG_BASE_URL=${COEVOKG_BASE_URL:-http://judge-host:5000/v1}
export COEVOKG_MODEL=${COEVOKG_MODEL:-judge-model}
export COEVOKG_API_KEY=${COEVOKG_API_KEY:-}
export COEVOKG_API_KEY_2=${COEVOKG_API_KEY_2:-}
export COEVOKG_MODEL_SLOT_TOTALS=${COEVOKG_MODEL_SLOT_TOTALS:-${COEVOKG_MODEL}:16}

# -----------------------------------------------------------------------------
# Run selection and hardware.
# -----------------------------------------------------------------------------
export COEVOKG_CONFIG_NAME=${COEVOKG_CONFIG_NAME:-train_example}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-coevokg_example_$(date +%Y%m%d_%H%M)}
export NNODES=${NNODES:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-${#_GPU_IDS[@]}}

# -----------------------------------------------------------------------------
# Runtime defaults. Override these from the environment when needed.
# -----------------------------------------------------------------------------
export RAY_MAX_CPUS=${RAY_MAX_CPUS:-64}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp}
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/tmp/pycache}
export CHAIN_DATA_LOCAL=${CHAIN_DATA_LOCAL:-/tmp/$(basename "${CHAIN_DATA_NAS}")}
export CACHE_MODEL_LOCAL=${CACHE_MODEL_LOCAL:-1}
export SWANLAB_MODE=${SWANLAB_MODE:-offline}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-}
export SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-workspace}
export SWANLAB_LOG_DIR=${OUTPUT_DIR}/swanlog

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-4}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export RAY_MAX_WORKERS=${RAY_MAX_WORKERS:-8}
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=${SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK:-true}
export SGLANG_BLOCK_NONZERO_RANK_CHILDREN=${SGLANG_BLOCK_NONZERO_RANK_CHILDREN:-0}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.9}
export RAY_DISABLE_METRICS=${RAY_DISABLE_METRICS:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export COEVOKG_SEARCH_CHAT_TEMPLATE=${COEVOKG_SEARCH_CHAT_TEMPLATE:-qwen2p5}
export COEVOKG_POOL_ACQUIRE_WAIT=${COEVOKG_POOL_ACQUIRE_WAIT:-90}
export COEVOKG_JUDGE_TIMEOUT=${COEVOKG_JUDGE_TIMEOUT:-120}
export COEVOKG_TOOL_CALL_TIMEOUT=${COEVOKG_TOOL_CALL_TIMEOUT:-120}
export COEVOKG_PROVIDER_MAX_ATTEMPTS=${COEVOKG_PROVIDER_MAX_ATTEMPTS:-4}
export COEVOKG_PROVIDER_COOLDOWN_TRANSIENT_S=${COEVOKG_PROVIDER_COOLDOWN_TRANSIENT_S:-0.5}
export COEVOKG_PROVIDER_COOLDOWN_429_S=${COEVOKG_PROVIDER_COOLDOWN_429_S:-0.3}
export COEVOKG_PROVIDER_COOLDOWN_QUOTA_S=${COEVOKG_PROVIDER_COOLDOWN_QUOTA_S:-21}
export COEVOKG_POOL_FATAL_AFTER_S=${COEVOKG_POOL_FATAL_AFTER_S:-300}
export COEVOKG_POOL_FATAL_AFTER_COUNT=${COEVOKG_POOL_FATAL_AFTER_COUNT:-20000}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export SELF_PLAY_DEBUG=${SELF_PLAY_DEBUG:-False}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-72000000}
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-72000}
export NCCL_WORK_FIFO_DEPTH=${NCCL_WORK_FIFO_DEPTH:-4194304}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_TIMEOUT=${GLOO_SOCKET_TIMEOUT:-7200}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-10485760}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export LANG=${LANG:-C.UTF-8}
export LANGUAGE=${LANGUAGE:-C.UTF-8}

mkdir -p "${PYTHONPYCACHEPREFIX}" "${RAY_TMPDIR}" "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/val" "${OUTPUT_DIR}/rollout"

# -----------------------------------------------------------------------------
# Basic input checks.
# -----------------------------------------------------------------------------
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[error] MODEL_PATH does not exist or is not a directory: ${MODEL_PATH}" >&2
    exit 1
fi
if [ ! -f "${DATA_PATH}.parquet" ]; then
    echo "[error] DATA_PATH parquet not found: ${DATA_PATH}.parquet" >&2
    exit 1
fi
if [ ! -f "${TEST_DATA_PATH}.parquet" ]; then
    echo "[error] TEST_DATA_PATH parquet not found: ${TEST_DATA_PATH}.parquet" >&2
    exit 1
fi
if [ ! -f "${CHAIN_DATA_NAS}" ]; then
    echo "[error] CHAIN_DATA_NAS not found: ${CHAIN_DATA_NAS}" >&2
    exit 1
fi

TRAIN_FILES_STR="[${DATA_PATH}.parquet]"
TEST_FILES_STR="[${TEST_DATA_PATH}.parquet]"

if [ ! -f "${CHAIN_DATA_LOCAL}" ]; then
    echo "[setup] Copying chain data to ${CHAIN_DATA_LOCAL} ..."
    cp "${CHAIN_DATA_NAS}" "${CHAIN_DATA_LOCAL}"
fi
export COEVOKG_CHAIN_DATA_LOCAL=${CHAIN_DATA_LOCAL}

if [ "${CACHE_MODEL_LOCAL}" = "1" ]; then
    MODEL_PATH_LOCAL=/tmp/$(basename "${MODEL_PATH}")
    if [ ! -d "${MODEL_PATH_LOCAL}" ]; then
        echo "[setup] Copying model weights to ${MODEL_PATH_LOCAL} ..."
        cp -r "${MODEL_PATH}" "${MODEL_PATH_LOCAL}"
    fi
    export MODEL_PATH=${MODEL_PATH_LOCAL}
fi

TOOL_CONFIG=${COEVOKG_ROOT}/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml
SEARCH_PORT=${SEARCH_PORT:-8000}
sed -i "/wiki:/s|http://[0-9.\+]*:${SEARCH_PORT}/retrieve|http://${SEARCH_IP}:${SEARCH_PORT}/retrieve|" "${TOOL_CONFIG}"
sed -i "/default:/s|http://[0-9.\+]*:${SEARCH_PORT}/retrieve|http://${SEARCH_IP}:${SEARCH_PORT}/retrieve|" "${TOOL_CONFIG}"

export NODE_RANK=${RANK}
export RAY_ADDRESS="${MASTER_ADDR}:6379"

if [ "${NODE_RANK}" -eq 0 ]; then
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray start" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true

    RETRIEVAL_PIDS=$(ss -tlnp 2>/dev/null | grep ":${SEARCH_PORT}" | grep -oP 'pid=\K[0-9]+' || true)
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ,'); do
        [[ "${RETRIEVAL_PIDS}" == *"${pid}"* ]] && continue
        kill -9 "${pid}" 2>/dev/null || true
    done
    pkill -9 -f "coevokg.main_rl\|setup_worker\|default_worker" 2>/dev/null || true
    sleep 3

    for i in $(seq 1 10); do
        ss -tlnp 2>/dev/null | grep -q ':6379' || break
        echo "[setup] Waiting for port 6379 to be released (${i}/10)..."
        sleep 2
    done

    ray start --block --head --port=6379 \
        --num-cpus="${RAY_MAX_CPUS}" --num-gpus="${N_GPUS_PER_NODE}" \
        --temp-dir="${RAY_TMPDIR}" &

    echo "[setup] Waiting for Ray GCS to be ready..."
    for i in $(seq 1 30); do
        ray status 2>/dev/null | grep -q "Autoscaler status" && break
        sleep 2
        echo "[setup] Ray not ready yet, retrying (${i}/30)..."
    done

    python3 -m coevokg.main_rl \
        --config-name="${COEVOKG_CONFIG_NAME}" \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
        trainer.nnodes="${NNODES}" \
        data.train_files="${TRAIN_FILES_STR}" \
        data.val_files="${TEST_FILES_STR}" \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
        "$@"
else
    ray start --block --address="${MASTER_ADDR}:6379" --temp-dir="${RAY_TMPDIR}"
    sleep 120
fi
