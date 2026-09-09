#!/usr/bin/env bash
#SBATCH --account=02360605-spr26
#SBATCH --job-name=caa-qwen3-vector
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ahmadg/caa-qwen3-vector-%j.log

set -euo pipefail

export PATH=/home/ahmadg/miniconda3/envs/xmix/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/ahmadg/miniconda3/envs/xmix/lib
export HF_HOME=/home/ahmadg/.cache/huggingface

cd /home/ahmadg/xMIx

python vllm/activations_extractor/applications/caa/generate_caa_vectors.py \
    --dataset /home/ahmadg/CAA/datasets/generate/sycophancy/generate_dataset.json \
    --output-dir /home/ahmadg/xMIx/artifacts/caa/qwen3-0.6b/sycophancy \
    --model Qwen/Qwen3-0.6B \
    --behavior sycophancy \
    --layers 6,9,12,15,18,21 \
    --train-size 900 \
    --validation-size 100 \
    --batch-size 8 \
    --seed 42
