# Qwen3-0.6B sycophancy CAA artifact

This directory contains contrastive activation directions learned from the
official `nrimsky/CAA` sycophancy generation dataset.

## Construction

- Model: `Qwen/Qwen3-0.6B`
- Training pairs: 900
- Validation pairs: 100
- Split seed: 42
- Direction: mean of matching-behavior minus nonmatching-behavior residual
  activations at the final assistant-answer token
- Candidate layers: 6, 9, 12, 15, 18, 21

`manifest.json` records the exact split indices and vector norms. The selected
artifact is `layer_18.pt`, whose raw L2 norm is `1.6695537567`.

## Validation selection

`selection_refined.json` contains the complete refined sweep. Layer 18 with
coefficient `+10` was selected using only the 100-example validation split.

| Coefficient | Matching-behavior rate | Mean log-likelihood margin |
|---:|---:|---:|
| -10 | 50% | 0.3705 |
| 0 | 56% | 0.5908 |
| +10 | 70% | 0.7857 |
| +20 | 67% | 0.8451 |

The selection rule prioritizes matching-behavior rate, then mean likelihood
margin, among positive coefficients.

## Held-out xMIx/vLLM test

Slurm job `370929` evaluated all 50 official held-out A/B examples through the
real xMIx hook and Triton CAA kernel.

| Setting | Coefficient | Matching choices | Rate |
|---|---:|---:|---:|
| Negative | -10 | 28/50 | 56% |
| Baseline | 0 | 28/50 | 56% |
| Positive | +10 | 30/50 | 60% |

Positive steering changed test items 3 and 21 from `(A)` to the matching `(B)`.
The three `eval_*_370929.json` files contain every raw model completion.

## Reproduction

From `/home/ahmadg/xMIx` on the remote cluster:

```bash
sbatch /home/ahmadg/caa-qwen3-generate-vector.sbatch
sbatch /home/ahmadg/caa-qwen3-select.sbatch
sbatch /home/ahmadg/caa-qwen3-evaluate.sbatch
```

The evaluation job runs negative, zero, and positive coefficients through the
same injected runner. Set `CAA_COEFFICIENT` to override the default `+10`, or
`CAA_VECTOR_PATH` to evaluate another saved layer vector.
