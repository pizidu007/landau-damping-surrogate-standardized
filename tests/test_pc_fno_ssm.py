import torch
import pytest
from landau_surrogate.models.pc_fno_ssm import SmallFNOSSM
from landau_surrogate.fluid.pc_ssm import advance_ssm


def model(memory=True, residual=True):
    torch.manual_seed(1)
    norm = {"input_mean": [1., 0., 1., 0.], "input_std": [.1]*4,
            "gradient_std": .03, "k_mean": .4, "k_std": .1, "hp_scale": .7}
    result = SmallFNOSSM(norm, width=4, layers=1, memory_size=3, maximum_mode=2,
                         use_memory=memory, use_residual=residual)
    # Exercise learned memory effects, not only the zero-head initializer.
    torch.nn.init.normal_(result.head[-1].weight, std=.1)
    return result


def states():
    x = (torch.arange(16) + .5) * 2 * torch.pi / 16
    value = torch.zeros(1, 5, 4, 16)
    value[:, :, 0] = 1 + .01 * x.cos()
    value[:, :, 2] = 1 + .012 * x.cos()
    value[:, :, 3] = -.01/.4 * x.sin()
    value[:, :, 1] = torch.arange(5)[None, :, None] * .001 * x.sin()
    return value


def test_ssm_future_cannot_change_past_and_chunking_is_consistent():
    m = model().eval(); u = states(); k = torch.tensor([.4])
    with torch.no_grad():
        full, memory = m(u, k)
        changed = u.clone(); changed[:, 3:] += .2
        other, _ = m(changed, k)
        torch.testing.assert_close(full[:, :3], other[:, :3])
        first, h = m(u[:, :2], k)
        second, h = m(u[:, 2:], k, h)
        torch.testing.assert_close(full, torch.cat((first, second), dim=1))
        torch.testing.assert_close(memory, h)


def test_invalid_padding_does_not_advance_memory():
    m = model(); u = states(); k = torch.tensor([.4])
    valid = torch.zeros(1, 5, dtype=torch.bool)
    pred, hidden = m(u, k, valid=valid)
    assert torch.count_nonzero(hidden) == 0
    assert torch.isfinite(pred).all()


@pytest.mark.parametrize("memory,residual", [(False,False),(True,False),(False,True),(True,True)])
def test_variants_backward_bandlimit_and_zero_mean(memory, residual):
    m = model(memory, residual)
    pred, _ = m(states(), torch.tensor([.4]))
    assert pred.shape == (1, 5, 16)
    assert pred.mean(-1).abs().max() < 1e-7
    assert torch.fft.rfft(pred).abs()[..., 3:].max() < 1e-6
    pred.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_rk_substages_commit_memory_once_and_do_not_mutate_input(monkeypatch):
    m = model().eval(); u = states()[:, 0].clone(); original = u.clone()
    commits = []
    actual = m.advance_memory
    def record(hidden, features, dt, valid=None):
        commits.append(dt)
        return actual(hidden, features, dt, valid)
    monkeypatch.setattr(m, "advance_memory", record)
    with torch.no_grad():
        pred, hidden = advance_ssm(m, u, torch.tensor([.4]), horizon=.1)
    assert len(commits) == 5 and pred.shape == (1, 2, 4, 16)
    torch.testing.assert_close(u, original)
    assert torch.isfinite(hidden).all()


def test_rejected_rk_stage_does_not_commit_memory(monkeypatch):
    m = model(); commits = []
    monkeypatch.setattr(m, "advance_memory", lambda *args, **kw: commits.append(1))
    def reject(now, state):
        raise RuntimeError("reject provisional state")
    with pytest.raises(RuntimeError, match="reject"):
        advance_ssm(m, states()[:, 0], torch.tensor([.4]), horizon=.1, observer=reject)
    assert not commits
