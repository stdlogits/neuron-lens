"""Pole geometry: the orthogonal write, the write cosines and the negative-lobe flip (``neuron_lens.neurons``)."""
import torch

from neuron_lens.neurons import orthogonal_write, read_rows, write_cosines


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def _random(f=5, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, f, generator=g), _unit(torch.randn(f, d, generator=g)), _unit(torch.randn(f, d, generator=g))


def test_orthogonal_write_removes_the_gate_up_plane():
    w, a, b = _random()
    o = orthogonal_write(w, a, b).T
    assert (o * a).sum(1).abs().max() < 1e-5
    assert (o * b).sum(1).abs().max() < 1e-5
    # what it removes lies in the plane: the projection onto the plane's complement of w equals o
    for i in range(len(a)):
        q, _ = torch.linalg.qr(torch.stack([a[i], b[i]], 1))
        assert torch.allclose(o[i], w.T[i] - q @ (q.T @ w.T[i]), atol=1e-5)


def test_write_cosines_follow_the_pole_and_the_lobe():
    w, a, b = _random()
    plain = torch.ones(2 * len(a))
    gate, up = write_cosines(w, a, b, plain)
    u = _unit(w.T)
    x, y = (u * a).sum(1), (u * b).sum(1)
    assert torch.allclose(gate, torch.cat([x, -x]), atol=1e-6) and torch.allclose(up, torch.cat([y, y]), atol=1e-6)
    # a pole on the negative lobe reads the opposite direction: both cosines flip
    lobe = torch.ones(2 * len(a)); lobe[1] = -1; lobe[len(a) + 2] = -1
    g2, u2 = write_cosines(w, a, b, lobe)
    assert torch.allclose(g2 * lobe, gate, atol=1e-6) and torch.allclose(u2 * lobe, up, atol=1e-6)


def test_read_rows_negate_the_flipped_poles():
    _, a, b = _random()
    lobe = torch.ones(2 * len(a)); lobe[0] = -1
    plain, flipped = read_rows(a, b, torch.ones(2 * len(a))), read_rows(a, b, lobe)
    assert torch.allclose(flipped[1:], plain[1:]) and torch.allclose(flipped[0], -plain[0])
    assert torch.allclose(plain[:len(a)], a + b) and torch.allclose(plain[len(a):], a - b)
