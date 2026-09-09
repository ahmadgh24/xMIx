# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Static integration tests for the CAA xMIx registration."""

from pathlib import Path

from vllm.activations_extractor.xmix.codegen import compile_xmix


def test_caa_preamble_instance_compiles(tmp_path: Path) -> None:
    app = tmp_path / "caa.xmix"
    app.write_text(
        "\n".join([
            "#model: qwen",
            "self.caa_vector = torch.ones(self.model_config.hidden_size, "
            "device=self.device, dtype=self.model_config.dtype)",
            "self.caa = ContrastiveActivationAdder(",
            "    steering_vector=self.caa_vector, coefficient=1.5)",
            "m.write(self.caa.run).layer([12]).submodule(\"mlp.post\")",
        ]),
        encoding="utf-8",
    )

    generated = compile_xmix(str(app))

    assert "model: qwen" in generated
    assert (
        "self.caa = ContrastiveActivationAdder("
        "steering_vector=self.caa_vector, coefficient=1.5, max_tokens=2048)"
        in generated
    )
    assert "if idx in (12,):" in generated
    assert 'layer.set_post_mlp_hook(self.caa.run, "w")' in generated
