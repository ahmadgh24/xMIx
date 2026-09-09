#!/usr/bin/env bash
#SBATCH --account=02360605-spr26
#SBATCH --job-name=caa-qwen3-select
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ahmadg/caa-qwen3-select-%j.log

set -euo pipefail

export PATH=/home/ahmadg/miniconda3/envs/xmix/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/ahmadg/miniconda3/envs/xmix/lib
export HF_HOME=/home/ahmadg/.cache/huggingface

cd /home/ahmadg/xMIx

python vllm/activations_extractor/applications/caa/select_caa_hyperparameters.py \
    --dataset /home/ahmadg/CAA/datasets/generate/sycophancy/generate_dataset.json \
    --artifact-dir /home/ahmadg/xMIx/artifacts/caa/qwen3-0.6b/sycophancy \
    --output /home/ahmadg/xMIx/artifacts/caa/qwen3-0.6b/sycophancy/selection_refined.json \
    --model Qwen/Qwen3-0.6B \
    --layers 15,18,21 \
    --coefficients=-20,-10,-5,0,5,10,20 \
    --batch-size 8
