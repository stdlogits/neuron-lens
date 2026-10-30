import pytest
import torch

from neuron_lens.lenses import merge
from neuron_lens.readout import layer_logit_stats, layer_reads, read, read_z


def test_read_matches_direct_softmax():
    torch.manual_seed(0)
    d, n, vocab = 8, 5, 20
    w, unembed, gamma = torch.randn(d, n), torch.randn(vocab, d), torch.rand(d) + 0.5
    jac = torch.randn(d, d)
    probs, ids = read(w, unembed, gamma, jac, k=3, chunk=2)
    t = jac @ w
    t = t / t.pow(2).mean(0, keepdim=True).add(1e-6).sqrt() * gamma[:, None]
    p = torch.softmax(unembed @ t, dim=0).T
    top, idx = p.topk(3, dim=1)
    assert torch.equal(ids, idx)
    assert torch.allclose(probs, top, atol=1e-5)


def test_read_is_scale_invariant_and_sign_sensitive():
    torch.manual_seed(1)
    w, unembed, gamma = torch.randn(8, 4), torch.randn(30, 8), torch.ones(8)
    assert torch.equal(read(w, unembed, gamma)[1], read(7.0 * w, unembed, gamma)[1])
    assert not torch.equal(read(w, unembed, gamma)[1], read(-w, unembed, gamma)[1])


def test_layer_stats_mean_matches_full_matrix():
    torch.manual_seed(2)
    w, unembed, gamma = torch.randn(8, 9), torch.randn(30, 8), torch.ones(8)
    mean, std = layer_logit_stats(w, unembed, gamma, chunk=4)
    t = w / w.pow(2).mean(0, keepdim=True).add(1e-6).sqrt()
    logits = unembed @ t
    assert torch.allclose(mean, logits.mean(1), atol=1e-5)
    assert torch.allclose(std, logits.std(1, correction=0), atol=1e-4)


def test_read_z_ranks_by_layer_zscore():
    torch.manual_seed(3)
    w, unembed, gamma = torch.randn(8, 6), torch.randn(30, 8), torch.ones(8)
    mean, std = layer_logit_stats(w, unembed, gamma)
    t = w / w.pow(2).mean(0, keepdim=True).add(1e-6).sqrt()
    z = (unembed @ t - mean[:, None]) / std[:, None]
    assert torch.equal(read_z(w, unembed, gamma, None, mean, std, k=4), z.topk(4, dim=0).indices.T)


def test_layer_reads_matches_read_and_read_z_on_both_poles():
    torch.manual_seed(4)
    w, unembed, gamma = torch.randn(8, 7), torch.randn(30, 8), torch.rand(8) + 0.5
    jac = torch.randn(8, 8)
    out = layer_reads(w, unembed, gamma, jac, k=3, kz=4, chunk=3)
    mean, std = layer_logit_stats(w, unembed, gamma, jac)
    for sign in (1, -1):
        probs, ids = read(sign * w, unembed, gamma, jac, k=3)
        assert torch.equal(out["ids"][sign], ids)
        assert torch.allclose(out["probs"][sign], probs, atol=1e-5)
        z = read_z(sign * w, unembed, gamma, jac, sign * mean, std, k=4)
        assert torch.equal(out["zids"][sign], z)
    t = jac @ w
    logits = unembed @ (t / t.pow(2).mean(0, keepdim=True).add(1e-6).sqrt() * gamma[:, None])
    c = logits - logits.mean(0)
    assert torch.allclose(out["skew"], c.pow(3).mean(0) / c.pow(2).mean(0).pow(1.5), atol=1e-4)


def test_merge_is_prompt_weighted():
    JacobianLens = pytest.importorskip("drowse.lens.jlens").JacobianLens
    a = JacobianLens({0: torch.ones(2, 2)}, n_prompts=1, d_model=2)
    b = JacobianLens({0: torch.zeros(2, 2)}, n_prompts=3, d_model=2)
    m = merge([a, b])
    assert m.n_prompts == 4
    assert torch.allclose(m.jacobians[0], torch.full((2, 2), 0.25))
