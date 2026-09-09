# xmix synthesized from vllm/activations_extractor/applications/xmix_examples/caa.xmix  (model: qwen)
# assumes in scope: torch
#
# Excerpts are grouped by destination. Each '# >>> XMIX-APPLY file=… anchor=… <<<'
# tag is read by the apply tool to splice the block below it at that anchor;
# the '# >>> DESTINATION: … <<<' banner is the human-readable label.
# >>> XMIX-FOOTPRINT flag=w submodule=mlp.post layers=18 <<<

# === SETUP (construct once — see destinations) ===

# --- user preamble (verbatim) ---

# >>> XMIX-APPLY file=vllm/v1/worker/gpu_model_runner.py anchor=runner_setup <<<
# >>> DESTINATION: gpu_model_runner — constructed once (e.g. in __init__/setup) <<<
_caa_default_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "artifacts",
    "caa",
    "qwen3-0.6b",
    "sycophancy",
    "layer_18.pt",
)
_caa_artifact_path = os.environ.get("CAA_VECTOR_PATH", _caa_default_path)
_caa_artifact = torch.load(
    _caa_artifact_path,
    map_location="cpu",
    weights_only=True,
)
self.caa_vector = _caa_artifact["vector"].to(
    device=self.device,
    dtype=self.model_config.dtype,
)
self.caa = ContrastiveActivationAdder(
    steering_vector=self.caa_vector,
    coefficient=float(os.environ.get("CAA_COEFFICIENT", "10.0")),
    max_tokens=2048,
)
self.caa.input_map.fill_(1)

# === INSTALL (after load_model) ===

# >>> XMIX-APPLY file=vllm/v1/worker/gpu_model_runner.py anchor=runner_postload <<<
# >>> DESTINATION: gpu_model_runner — after load_model (install hooks + wire buffers) <<<

# vllm/activations_extractor/applications/xmix_examples/caa.xmix:30  m.write(self.caa.run).layers([18]).submodule("mlp.post")
# steer ContrastiveActivationAdder on layers [18] at 'mlp.post' (flag 'w')
for layer in self.model.model.layers:
    idx = layer.self_attn.layer_idx
    if idx in (18,):
        layer.set_post_mlp_hook(self.caa.run, "w")
