# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Standalone GPU tests for Contrastive Activation Addition.

Run with:
    pytest vllm/activations_extractor/test_caa_add.py -v
"""

import pytest
import torch

from vllm.activations_extractor.write_activations import (
    ContrastiveActivationAdder,
    caa_add_kernel,
)
from vllm.triton_utils import triton


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CAA's Triton kernel requires a CUDA GPU",
)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 2e-3, 2e-3
    return 2e-2, 2e-2


CUDA_DTYPES = [
    torch.float32,
    torch.float16,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            not torch.cuda.is_bf16_supported(),
            reason="This GPU does not support bfloat16",
        ),
    ),
]


@pytest.mark.parametrize("n_rows,hidden_size", [
    (1, 17),
    (4, 1024),
    (3, 1500),
])
@pytest.mark.parametrize("dtype", CUDA_DTYPES)
def test_kernel_matches_pytorch_reference(
    n_rows: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """The direct Triton launch must match scaled PyTorch addition."""
    torch.manual_seed(0)
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    steering_vector = torch.randn(hidden_size, device="cuda", dtype=dtype)
    coefficient = torch.tensor([-.75], device="cuda", dtype=dtype)
    input_map = torch.ones(n_rows, device="cuda", dtype=torch.int32)

    actual = x.clone()
    expected = x + coefficient * steering_vector
    block_size = 1024
    grid = (n_rows, triton.cdiv(hidden_size, block_size))

    caa_add_kernel[grid](
        actual,
        steering_vector,
        coefficient,
        input_map,
        hidden_size,
        BLOCK_SIZE=block_size,
    )

    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("coefficient", [0.0, 0.5, -2.0])
def test_wrapper_is_in_place_and_matches_reference(coefficient: float) -> None:
    """The wrapper must return and modify the original activation tensor."""
    torch.manual_seed(1)
    shape = (2, 3, 65)
    steering_vector = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
    adder = ContrastiveActivationAdder(
        steering_vector,
        coefficient,
        max_tokens=shape[0] * shape[1],
    )
    adder.load_mask(torch.ones(shape[0] * shape[1], device="cuda"))

    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    expected = x.clone() + coefficient * steering_vector
    original_address = x.data_ptr()

    result = adder(x)

    assert result is x
    assert result.data_ptr() == original_address
    torch.testing.assert_close(result, expected, atol=2e-3, rtol=2e-3)


def test_run_uses_write_hook_contract_and_row_gate() -> None:
    """run accepts xMIx's hook arguments and steers only enabled rows."""
    hidden_size = 33
    steering_vector = torch.ones(hidden_size, device="cuda")
    adder = ContrastiveActivationAdder(
        steering_vector,
        coefficient=2.0,
        max_tokens=2,
    )
    adder.load_mask(torch.tensor([1, 0], dtype=torch.int32, device="cuda"))
    x = torch.zeros(2, hidden_size, device="cuda")
    unused_runtime_vector = torch.empty_like(steering_vector)
    unused_selected_count = torch.zeros(1, dtype=torch.int32, device="cuda")

    result = adder.run(x, unused_runtime_vector, unused_selected_count)

    expected = torch.zeros_like(x)
    expected[0].fill_(2.0)
    torch.testing.assert_close(result, expected)


def test_loaders_update_values_without_replacing_storage() -> None:
    """Hot-swapping state must preserve addresses used by CUDA graphs."""
    hidden_size = 71
    adder = ContrastiveActivationAdder(
        torch.zeros(hidden_size, device="cuda", dtype=torch.float16),
        coefficient=0.0,
        max_tokens=3,
    )
    vector_address = adder.steering_vector.data_ptr()
    coefficient_address = adder.coefficient.data_ptr()
    input_map_address = adder.input_map.data_ptr()

    adder.load_vector(
        torch.full((hidden_size,), 0.25, device="cuda", dtype=torch.float16)
    )
    adder.load_coefficient(-4.0)
    adder.load_mask(torch.ones(3, dtype=torch.bool, device="cuda"))

    assert adder.steering_vector.data_ptr() == vector_address
    assert adder.coefficient.data_ptr() == coefficient_address
    assert adder.input_map.data_ptr() == input_map_address

    x = torch.ones(3, hidden_size, device="cuda", dtype=torch.float16)
    adder(x)
    torch.testing.assert_close(x, torch.zeros_like(x), atol=1e-3, rtol=0)


def test_constructor_rejects_non_vector_direction() -> None:
    with pytest.raises(AssertionError, match="must be 1-D"):
        ContrastiveActivationAdder(
            torch.zeros(2, 8, device="cuda"),
            coefficient=1.0,
            max_tokens=2,
        )


def test_constructor_rejects_non_scalar_coefficient() -> None:
    with pytest.raises(AssertionError, match="must contain one value"):
        ContrastiveActivationAdder(
            torch.zeros(8, device="cuda"),
            coefficient=torch.ones(2, device="cuda"),
            max_tokens=2,
        )


def test_loaders_reject_shape_mismatches() -> None:
    adder = ContrastiveActivationAdder(
        torch.zeros(8, device="cuda"),
        coefficient=1.0,
        max_tokens=2,
    )

    with pytest.raises(AssertionError, match="Shape mismatch"):
        adder.load_vector(torch.zeros(9, device="cuda"))
    with pytest.raises(AssertionError, match="must contain one value"):
        adder.load_coefficient(torch.ones(2, device="cuda"))
    with pytest.raises(AssertionError, match="must be 1-D"):
        adder.load_mask(torch.ones(1, 2, device="cuda"))
    with pytest.raises(AssertionError, match="max_tokens"):
        adder.load_mask(torch.ones(3, device="cuda"))


def test_short_mask_clears_unused_rows() -> None:
    """Loading a shorter mask must not leave stale enabled rows behind."""
    hidden_size = 19
    adder = ContrastiveActivationAdder(
        torch.ones(hidden_size, device="cuda"),
        coefficient=1.0,
        max_tokens=4,
    )
    adder.load_mask(torch.ones(4, device="cuda"))
    adder.load_mask(torch.tensor([1, 0], device="cuda"))

    x = torch.zeros(4, hidden_size, device="cuda")
    adder(x)

    expected = torch.zeros_like(x)
    expected[0].fill_(1.0)
    torch.testing.assert_close(x, expected)


def test_forward_rejects_more_rows_than_mask_capacity() -> None:
    adder = ContrastiveActivationAdder(
        torch.zeros(8, device="cuda"),
        coefficient=1.0,
        max_tokens=2,
    )

    with pytest.raises(AssertionError, match="max_tokens"):
        adder(torch.zeros(3, 8, device="cuda"))


@pytest.mark.parametrize("problem", ["hidden_size", "dtype", "device", "layout"])
def test_forward_rejects_incompatible_activations(problem: str) -> None:
    steering_vector = torch.zeros(8, device="cuda", dtype=torch.float16)
    adder = ContrastiveActivationAdder(
        steering_vector,
        coefficient=1.0,
        max_tokens=2,
    )

    if problem == "hidden_size":
        x = torch.zeros(2, 9, device="cuda", dtype=torch.float16)
        message = "Hidden-size mismatch"
    elif problem == "dtype":
        x = torch.zeros(2, 8, device="cuda", dtype=torch.float32)
        message = "Dtype mismatch"
    elif problem == "device":
        x = torch.zeros(2, 8, device="cpu", dtype=torch.float16)
        message = "Device mismatch"
    else:
        x = torch.zeros(8, 2, device="cuda", dtype=torch.float16).T
        assert not x.is_contiguous()
        message = "must be contiguous"

    with pytest.raises(AssertionError, match=message):
        adder(x)
