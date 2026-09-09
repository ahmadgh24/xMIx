#!/usr/bin/env bash
#SBATCH --account=02360605-spr26
#SBATCH --job-name=caa-qwen3-server
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ahmadg/caa-qwen3-server-%j.log

set -euo pipefail

export PATH=/home/ahmadg/miniconda3/envs/xmix/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/ahmadg/miniconda3/envs/xmix/lib
export CUDA_HOME=/home/ahmadg/miniconda3/envs/xmix
export HF_HOME=/home/ahmadg/.cache/huggingface
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

cd /home/ahmadg/xMIx

vllm serve Qwen/Qwen3-0.6B \
    --dtype half \
    --max-model-len 2048 \
    --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000 &
server_pid=$!

cleanup() {
    kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 90); do
    if curl --silent --fail http://127.0.0.1:8000/health >/dev/null; then
        curl --silent --fail \
            -H "Content-Type: application/json" \
            -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Say hello in one short sentence.","max_tokens":16,"temperature":0}' \
            http://127.0.0.1:8000/v1/completions \
            | tee "/home/ahmadg/caa-qwen3-smoke-${SLURM_JOB_ID}.json"
        kill "$server_pid"
        wait "$server_pid" || true
        exit 0
    fi

    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid"
        exit $?
    fi
    sleep 10
done

echo "server did not become healthy within 15 minutes" >&2
exit 1
