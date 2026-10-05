"""Exercise the Eagle API consumed by the Marin online trainer."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, PreTrainedModel, Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig


@pytest.mark.parametrize("config_class", [LlamaConfig, Qwen3Config])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gated", [False, True])
def test_eagle3_training_and_checkpoint(config_class, dtype, gated, tmp_path: Path):
    """Real safetensors import, exact-mask attention, KL gradients, and HF reload."""
    torch.manual_seed(19)
    verifier_dir = tmp_path / "verifier"
    layer_config = config_class(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="sdpa",
    )
    layer_config.save_pretrained(verifier_dir)
    weights = {
        "model.embed_tokens.weight": torch.randn(64, 16),
        "lm_head.weight": torch.randn(64, 16),
        "model.norm.weight": torch.rand(16) + 0.5,
    }
    if gated:
        weights.update(
            {
                "model.final_gated_norm.down_proj.weight": torch.randn(128, 16) * 0.1,
                "model.final_gated_norm.up_proj.weight": torch.randn(16, 128) * 0.1,
            }
        )
    save_file(weights, verifier_dir / "model.safetensors")
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=layer_config,
        draft_vocab_size=32,
        eagle_aux_hidden_state_layer_ids=[0, 1, 2],
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=str(verifier_dir),
                architectures=["GrugMoeForCausalLM"] if gated else [],
            ),
        ),
    )
    model = Eagle3DraftModel(config).to(dtype=dtype)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[::2] = True
    model.load_vocab_mappings(mask, torch.arange(0, 64, 2))
    model.load_verifier_weights()
    assert torch.equal(
        model.embed_tokens.weight, weights["model.embed_tokens.weight"].to(dtype)
    )
    assert torch.equal(
        model.verifier_lm_head.weight, weights["lm_head.weight"][::2].to(dtype)
    )
    hidden = torch.randn(1, 9, 48, dtype=dtype)
    final_hidden = torch.randn(1, 9, 16, dtype=dtype)
    arguments = {
        "hidden_states": hidden,
        "input_ids": torch.randint(0, 64, (1, 9)),
        "document_ids": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1]]),
        "verifier_last_hidden_states": final_hidden,
        "ttt_steps": 3,
        "loss_config": resolve_loss_config("kl_div", "eager"),
    }
    target_logits = []
    draft_logits = []
    target_hook = model.verifier_lm_head.register_forward_hook(
        lambda _module, _inputs, output: target_logits.append(output.detach())
    )
    draft_hook = model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: draft_logits.append(output)
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    initial_fc = model.fc.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        target_logits.clear()
        draft_logits.clear()
        tokens, loss, metrics = model(**arguments)
        assert len(tokens) == 3
        assert torch.isfinite(loss)
        assert torch.isfinite(metrics["loss_sum"])
        # Compute verifier RMS normalization and rank128 gate independently.
        normed = (
            final_hidden.float()
            * torch.rsqrt(
                final_hidden.float().square().mean(-1, keepdim=True)
                + layer_config.rms_norm_eps
            )
        ).to(dtype)
        normed = normed * weights["model.norm.weight"].to(dtype)
        if gated:
            down = weights["model.final_gated_norm.down_proj.weight"].to(dtype)
            up = weights["model.final_gated_norm.up_proj.weight"].to(dtype)
            normed = normed * torch.sigmoid(
                torch.nn.functional.silu(normed @ down.T) @ up.T
            )
        expected_target = normed @ weights["lm_head.weight"][::2].to(dtype).T
        torch.testing.assert_close(target_logits[0], expected_target, rtol=0, atol=0)
        reference_loss = torch.zeros(())
        for step, logits in enumerate(draft_logits):
            logits = logits[:, :-step] if step else logits
            logq = logits.float().log_softmax(-1)
            logp = expected_target[:, step:].float().log_softmax(-1)
            kl = (logp.exp() * (logp - logq)).sum(-1)
            reference_loss = reference_loss + kl.sum() / (kl.numel() + 1e-5)
        torch.testing.assert_close(loss, reference_loss, rtol=2e-5, atol=2e-6)
        loss.backward()
        assert model.fc.weight.grad is not None
        assert torch.isfinite(model.fc.weight.grad).all()
        optimizer.step()
    target_hook.remove()
    draft_hook.remove()
    assert not torch.equal(model.fc.weight, initial_fc)
    draft_dir = tmp_path / "draft"
    model.save_pretrained(draft_dir)
    saved = load_file(draft_dir / "model.safetensors")
    assert not any("verifier_gate" in name for name in saved)
    loaded_config = Eagle3SpeculatorConfig.from_pretrained(draft_dir)
    loaded_config.transformer_layer_config._attn_implementation = "sdpa"
    # This is the exact generic HF path used by the online trainer.
    loaded = PreTrainedModel.from_pretrained.__func__(
        Eagle3DraftModel, draft_dir, config=loaded_config, dtype=dtype
    )
    loaded.load_verifier_weights()
    # The online trainer applies its serving dtype after HF loading, including
    # non-persistent rotary buffers which HF can reconstruct in float32.
    loaded = loaded.to(dtype=dtype)
    for name, tensor in saved.items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
    with torch.no_grad():
        before = model(**arguments)
        after = loaded(**arguments)
    for left, right in zip(before[0], after[0], strict=True):
        assert torch.equal(left, right)
    torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
