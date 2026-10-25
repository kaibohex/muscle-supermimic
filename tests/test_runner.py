"""Runner pieces that need no environment."""

import torch

from muscle_supermimic.rl.supermimic.runner import ClipRowGrad


def test_clip_row_grad_bounds_each_row_over_the_group():
    a = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(3, 2, 2, requires_grad=True)
    stats = []
    a1, b1 = ClipRowGrad.apply(1.0, 2.0, stats, a, b)
    torch.testing.assert_close(a1, a, rtol=0, atol=0)
    torch.testing.assert_close(b1, b, rtol=0, atol=0)
    w = torch.tensor([10.0, 0.01, 0.3])
    ((a1.sum(-1) + b1.flatten(1).sum(-1)) * w).sum().backward()
    # row i's gradient is w_i over 8 entries: scaled norm 2 * w_i * sqrt(8) -> rows 0 and 2 exceed 1, row 1 does not
    norm = 2.0 * torch.sqrt(a.grad.square().sum(-1) + b.grad.flatten(1).square().sum(-1))
    torch.testing.assert_close(norm, torch.tensor([1.0, 2 * 0.01 * 8**0.5, 1.0]))
    torch.testing.assert_close(a.grad[1], torch.full((4,), 0.01))  # unclipped row untouched
    assert len(stats) == 1 and abs(float(stats[0]) - 2 / 3) < 1e-6
