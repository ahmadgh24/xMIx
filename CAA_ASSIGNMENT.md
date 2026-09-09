# Contrastive Activation Addition in xMIx

This repository is my implementation of the Contrastive Activation Addition
(CAA) assignment. It adds a CUDA/Triton CAA primitive to xMIx, registers it in
the `.xmix` compiler, injects it into Qwen3, constructs a real behavioral
steering vector, and compares the model with and without steering.

## Result in one paragraph

I learned a sycophancy direction for `Qwen/Qwen3-0.6B` by averaging the
difference between 900 pairs of internal activations: one activation from an
answer matching the behavior and one from an answer not matching it. A separate
100-example validation split selected layer 18 and coefficient `+10`. On the 50
official held-out CAA questions, the unsteered model selected the
behavior-matching answer 28 times (56%), while positive steering selected it 30
times (60%). Thus, the injected Triton operation changed two answers in the
intended direction. This is a visible end-to-end effect, although 50 examples
are too few to claim strong statistical evidence.

## What CAA does

For each contrast pair, I record the model state associated with the final
assistant-answer token:

```text
difference_i = matching_activation_i - nonmatching_activation_i
```

The steering direction is their mean:

```text
CAA_vector = mean(difference_i)
```

At inference time, the new Triton kernel applies this update to each enabled
activation row:

```text
activation[row, :] += coefficient * CAA_vector
```

Positive coefficients move activations toward the learned behavior. Negative
coefficients move them in the opposite direction. A per-row `input_map` gate
allows rows to be enabled or disabled without reallocating GPU buffers. The
current experiment fills this gate with ones, so every activation row is
enabled.

## Assignment task mapping

### Task 1: Python wrapper, Triton kernel, and tests

- `caa_add_kernel` and `ContrastiveActivationAdder` are implemented in
  [`vllm/activations_extractor/write_activations.py`](vllm/activations_extractor/write_activations.py).
- The wrapper owns address-stable GPU tensors for the direction, coefficient,
  and row gate.
- The Triton launch treats any leading dimensions as activation rows and the
  last dimension as the model hidden dimension.
- Standalone coverage is in
  [`vllm/activations_extractor/test_caa_add.py`](vllm/activations_extractor/test_caa_add.py).
  All 23 GPU tests passed on an RTX 2080 Ti.

### Task 2: xMIx compiler registration

- `ContrastiveActivationAdder` is registered in
  [`vllm/activations_extractor/xmix/spec.py`](vllm/activations_extractor/xmix/spec.py).
- Compiler integration coverage is in
  [`vllm/activations_extractor/test_caa_xmix.py`](vllm/activations_extractor/test_caa_xmix.py).

### Task 3: `.xmix` application and compilation

- Source application:
  [`vllm/activations_extractor/applications/xmix_examples/caa.xmix`](vllm/activations_extractor/applications/xmix_examples/caa.xmix)
- Compiler output:
  [`vllm/activations_extractor/applications/xmix_examples/caa_synth.py`](vllm/activations_extractor/applications/xmix_examples/caa_synth.py)
- The generated application loads the learned layer-18 vector, reads the
  coefficient from `CAA_COEFFICIENT`, and installs a write hook at `mlp.post`.

### Task 4: model injection and measurable difference

- The generated application was injected into
  [`vllm/v1/worker/gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py).
- Qwen3-0.6B was served successfully through the patched vLLM server.
- The held-out evaluation ran the same injected path at coefficients `-10`,
  `0`, and `+10`.

## Generating the real vector

The scripts are in
[`vllm/activations_extractor/applications/caa/`](vllm/activations_extractor/applications/caa/):

- `generate_caa_vectors.py` renders Qwen chat conversations, locates the real
  answer-token position from tokenizer offsets, and generates vectors for six
  candidate layers in one extraction pass.
- `select_caa_hyperparameters.py` evaluates answer log-likelihood on the
  reserved validation split and selects the layer and coefficient without
  reading the test split.
- `evaluate_caa_ab.py` sends all held-out questions through the injected vLLM
  server and saves every raw completion.

The data comes from the official
[`nrimsky/CAA`](https://github.com/nrimsky/CAA) sycophancy dataset. The data is
not duplicated in this repository.

The generated vectors, exact split indices, parameter sweeps, raw completions,
and a more detailed results note are under
[`artifacts/caa/qwen3-0.6b/sycophancy/`](artifacts/caa/qwen3-0.6b/sycophancy/).

## Evaluation results

Validation selection used 100 examples that were excluded from vector
construction:

| Coefficient at layer 18 | Matching-behavior rate | Mean log-likelihood margin |
|---:|---:|---:|
| -10 | 50% | 0.3705 |
| 0 | 56% | 0.5908 |
| +10 | 70% | 0.7857 |
| +20 | 67% | 0.8451 |

The selection rule prioritized matching-behavior rate and then mean likelihood
margin, so it selected `+10` instead of `+20`.

The final 50-example test was executed through xMIx and the Triton kernel:

| Setting | Coefficient | Matching answers | Rate |
|---|---:|---:|---:|
| Negative | -10 | 28/50 | 56% |
| Baseline | 0 | 28/50 | 56% |
| Positive | +10 | 30/50 | 60% |

Test items 3 and 21 changed from the nonmatching `(A)` answer to the matching
`(B)` answer under positive steering.

## Reproduction on the Technion Lambda cluster

The root-level Slurm files record the exact jobs used for the experiment. They
contain cluster-specific account and home-directory paths that should be
adjusted for another user.

First clone the official dataset next to this repository:

```bash
git clone --depth 1 https://github.com/nrimsky/CAA.git ../CAA
```

Then generate vectors, select parameters, and run the final comparison:

```bash
sbatch caa-qwen3-generate-vector.sbatch
sbatch caa-qwen3-select.sbatch
sbatch caa-qwen3-evaluate.sbatch
```

The completed remote jobs were:

- `370923`: vector generation, completed in 35 seconds.
- `370925` and `370928`: initial and refined validation sweeps.
- `370929`: final negative/baseline/positive xMIx evaluation, completed in
  2 minutes 45 seconds.

The RTX 2080 Ti has compute capability 7.5, so the serving jobs explicitly use
`VLLM_ATTENTION_BACKEND=TRITON_ATTN`; the bundled FlashAttention 2 backend
requires a newer GPU.

## Additional runner cleanup

The original runner eagerly loaded several unrelated demo artifacts using
another developer's absolute paths. This prevented Qwen from starting even
when those applications were inactive. I removed those inactive initializers;
each active `.xmix` application now owns the tensors it actually requires.
