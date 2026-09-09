#!/usr/bin/env bash
#SBATCH --account=02360605-spr26
#SBATCH --job-name=caa-qwen3-eval
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ahmadg/caa-qwen3-eval-%j.log

set -euo pipefail

export PATH=/home/ahmadg/miniconda3/envs/xmix/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/ahmadg/miniconda3/envs/xmix/lib
export CUDA_HOME=/home/ahmadg/miniconda3/envs/xmix
export HF_HOME=/home/ahmadg/.cache/huggingface
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export CAA_VECTOR_PATH=/home/ahmadg/xMIx/artifacts/caa/qwen3-0.6b/sycophancy/layer_18.pt

cd /home/ahmadg/xMIx

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for setting in negative:-10 baseline:0 positive:10; do
    label=${setting%%:*}
    coefficient=${setting##*:}
    export CAA_COEFFICIENT=$coefficient

    vllm serve Qwen/Qwen3-0.6B \
        --dtype half \
        --max-model-len 2048 \
        --enforce-eager \
        --gpu-memory-utilization 0.85 \
        --host 127.0.0.1 \
        --port 8000 &
    server_pid=$!

    healthy=0
    for _ in $(seq 1 90); do
        if curl --silent --fail http://127.0.0.1:8000/health >/dev/null; then
            healthy=1
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            wait "$server_pid"
            exit $?
        fi
        sleep 10
    done
    if [[ "$healthy" != 1 ]]; then
        echo "server did not become healthy for $label" >&2
        exit 1
    fi

    python vllm/activations_extractor/applications/caa/evaluate_caa_ab.py \
        --dataset /home/ahmadg/CAA/datasets/test/sycophancy/test_dataset_ab.json \
        --output "/home/ahmadg/xMIx/artifacts/caa/qwen3-0.6b/sycophancy/eval_${label}_${SLURM_JOB_ID}.json" \
        --model Qwen/Qwen3-0.6B \
        --label "$label" \
        --coefficient "$coefficient"

    kill "$server_pid"
    wait "$server_pid" || true
    server_pid=""
done
