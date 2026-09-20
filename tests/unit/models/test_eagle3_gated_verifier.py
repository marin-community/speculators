"""EAGLE target logits for verifiers with a gated final norm."""

from __future__ import annotations

import torch
from torch.nn import functional
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _model() -> Eagle3DraftModel:
    transformer_config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=32,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="verifier",
                architectures=["GrugMoeForCausalLM"],
            ),
        ),
    )
    return Eagle3DraftModel(config)


def test_loads_gated_verifier_head(monkeypatch):
    model = _model()
    down = torch.randn(4, 16)
    up = torch.randn(16, 4)

    monkeypatch.setattr(DraftVocabMixin, "load_verifier_weights", lambda _self: None)
    monkeypatch.setattr(
        "speculators.models.eagle3.core.AutoConfig.from_pretrained",
        lambda _path: LlamaConfig(hidden_size=16, num_attention_heads=2),
    )
    monkeypatch.setattr(
        "speculators.utils.loading.load_model_layers",
        lambda _names, _path: {
            "model.final_gated_norm.down_proj.weight": down,
            "model.final_gated_norm.up_proj.weight": up,
        },
    )

    model.load_verifier_weights()

    assert torch.equal(model.verifier_gate_down_weight, down)
    assert torch.equal(model.verifier_gate_up_weight, up)


def test_gated_verifier_logits_match_target_head():
    model = _model()
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    model.verifier_gate_down_weight = torch.randn(4, 16)
    model.verifier_gate_up_weight = torch.randn(16, 4)
    hidden_states = torch.randn(2, 3, 16)

    normalized = model.verifier_norm(hidden_states)
    gate = functional.linear(normalized, model.verifier_gate_down_weight)
    gate = functional.linear(functional.silu(gate), model.verifier_gate_up_weight)
    expected = model.verifier_lm_head(normalized * torch.sigmoid(gate))

    assert torch.equal(model._verifier_logits(hidden_states), expected)
