"""The firing accumulator (``neuron_lens.firings``) against hand captures on tiny random models."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from neuron_lens import arch
from neuron_lens.firings import Accumulator, firing_plan


def tiny_qwen():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64, attn_implementation="eager")
    return Qwen3Model(cfg).float().eval().requires_grad_(False)


def tiny_gemma():
    pytest.importorskip("transformers.models.gemma4_unified.modeling_gemma4_unified")
    from test_gemma4 import tiny_model
    return tiny_model()


@pytest.mark.parametrize("make", [tiny_qwen, tiny_gemma])
def test_means_match_hand_captures_and_after_is_additive(make):
    model = make()
    f, d = model.config.intermediate_size, model.config.hidden_size
    ids = torch.randint(0, 50, (2, 12), generator=torch.Generator().manual_seed(1))
    # Three firings at layer 1: two poles, doc 0 and doc 1; document a=0 and the batch's rows are the docs.
    doc, pos, val = torch.zeros(2 * f, 2, dtype=torch.long), torch.zeros(2 * f, 2, dtype=torch.long), -torch.ones(2 * f, 2)
    doc[3, 0], pos[3, 0], val[3, 0] = 0, 4, 2.0
    doc[3, 1], pos[3, 1], val[3, 1] = 1, 5, 1.0
    doc[f + 1, 0], pos[f + 1, 0], val[f + 1, 0] = 1, 7, 3.0
    acc = Accumulator({1: firing_plan(doc, pos, val)}, f, d, "cpu")
    cap = {}
    lay = model.layers[1]
    h = [arch.mid_norm(lay).register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0])),
         lay.register_forward_hook(lambda m, a, o: cap.__setitem__("after", o[0] if isinstance(o, tuple) else o)),
         model.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("final", a[0]))]
    hooks = acc.hooks(model)
    acc.begin(0, 2)
    with torch.no_grad():
        model(input_ids=ids)
    for x in h + hooks:
        x.remove()
    m = acc.means(1)
    rms = lambda v: v.pow(2).mean(-1, keepdim=True).sqrt()
    row = {3: [(0, 4), (1, 5)], f + 1: [(1, 7)]}
    for r, places in row.items():
        got_b = sum(cap["x"][b, p] / rms(cap["x"][b, p]) for b, p in places) / len(places)
        assert torch.allclose(m["before"][r], got_b, atol=1e-5)
        got_a = sum(cap["after"][b, p] / rms(cap["x"][b, p]) for b, p in places) / len(places)
        assert torch.allclose(m["after"][r], got_a, atol=1e-5), "after must equal scalar·before + mlp"
        got_p = sum(cap["final"][b, p] / rms(cap["final"][b, p]) for b, p in places) / len(places)
        assert torch.allclose(m["pred"][r], got_p, atol=1e-5)
    assert acc.count[1][3] == 2 and acc.count[1][f + 1] == 1 and acc.count[1].sum() == 3


def test_firing_plan_is_sorted_by_document_and_keeps_fired_positions():
    doc = torch.tensor([[5, 2], [2, 9]]); pos = torch.tensor([[1, 2], [3, 4]]); val = torch.tensor([[1.0, -1.0], [0.5, 0.2]])
    d, p, r = firing_plan(doc, pos, val)
    assert d.tolist() == [2, 5, 9] and p.tolist() == [3, 1, 4] and r.tolist() == [1, 0, 1]
