from types import SimpleNamespace

import numpy as np
import torch

from landau_surrogate.data.continuum_history import HistoryCache, supervised_history
from landau_surrogate.fluid.history_closure_1d import advance_history
from landau_surrogate.models.history_closure import HistoryClosureFNO


def fixture():
    theta = torch.arange(16) * (2*torch.pi/16)
    states = torch.zeros(1, 31, 4, 16)
    states[:, :, 0] = 1 + .02*torch.cos(theta)
    states[:, :, 2] = states[:, :, 0]
    states[:, :, 3] = -.02/.35*torch.sin(theta)
    return HistoryCache([SimpleNamespace(K=.35, alpha=.02, split="train")], states,
                        torch.zeros(1,31,16), .02, [])


def model():
    return HistoryClosureFNO(maximum_mode=3, width=8, layers=1, history_span=.08,
                            normalization={"input_mean":[1,0,1,0],"input_std":[.1]*4,
                            "gradient_std":.01,"k_mean":.4,"k_std":.1,"alpha_mean":.1,"alpha_std":.05})


def test_history_mask_and_initial_padding_are_causal():
    cache = fixture()
    cache.state[:, 1:] += 10  # future data must not influence t=0 history
    history, valid = supervised_history(cache, torch.tensor([0]), torch.tensor([0.]), .5)
    assert valid.tolist() == [[False]*7+[True]]
    torch.testing.assert_close(history, cache.state[:, 0, None].expand(-1,8,-1,-1))
    sampled=cache.sample(torch.tensor([0]),torch.tensor([.01],dtype=torch.float64))
    assert sampled.dtype==cache.state.dtype


def test_alpha_is_disabled_and_output_is_zero_mean_bandlimited():
    torch.manual_seed(1)
    cache, net = fixture(), model()
    history, valid = supervised_history(cache, torch.tensor([0]), torch.tensor([0.]), .08)
    first = net(history, valid, torch.tensor([.35]), torch.tensor([.02]))
    second = net(history, valid, torch.tensor([.35]), torch.tensor([.2]))
    torch.testing.assert_close(first, second)
    assert abs(float(first.mean())) < 1e-8
    assert float(torch.fft.rfft(first)[...,4:].abs().max()) < 1e-8


def test_checkpointed_history_matches_uncheckpointed_values_and_full_gradients():
    torch.manual_seed(2)
    cache, net = fixture(), model()
    kwargs = dict(horizon=.08, dt=.01, output_dt=.02)
    start = cache.state[:,0].clone().requires_grad_()
    out = advance_history(net, cache, torch.tensor([0]), torch.tensor([0.]), initial=start,
                          use_checkpoint=False, **kwargs)
    loss = out[:,-1].square().mean()
    loss.backward()
    expected = {k:p.grad.clone() for k,p in net.named_parameters() if p.grad is not None}
    start_gradient = start.grad.clone()
    net.zero_grad(set_to_none=True)
    initial = start.detach().clone().requires_grad_()
    checked = advance_history(net, cache, torch.tensor([0]), torch.tensor([0.]), initial=initial,
                              use_checkpoint=True, **kwargs)
    checked[:,-1].square().mean().backward()
    torch.testing.assert_close(checked, out)
    torch.testing.assert_close(initial.grad, start_gradient)
    assert float(initial.grad.norm()) > 0
    for name, parameter in net.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=1e-4, atol=1e-8)


def test_generated_memory_handoff_matches_continuous_integration():
    torch.manual_seed(3)
    cache, net=fixture(),model()
    ids=torch.tensor([0]);starts=torch.tensor([0.])
    with torch.no_grad():
        whole=advance_history(net,cache,ids,starts,horizon=.14,dt=.01,output_dt=.02)
        _,memory=advance_history(net,cache,ids,starts,horizon=.06,dt=.01,output_dt=.02,return_memory=True)
        remainder=advance_history(net,cache,ids,starts+.06,horizon=.08,dt=.01,output_dt=.02,memory=memory)
    torch.testing.assert_close(remainder[:,-1],whole[:,-1],atol=1e-7,rtol=1e-6)


def test_low_frequency_closure_reduces_calls_and_remains_differentiable():
    torch.manual_seed(4)
    cache,net=fixture(),model()
    net.closure_interval=.02
    calls=[]
    hook=net.register_forward_hook(lambda *_:calls.append(1))
    out=advance_history(net,cache,torch.tensor([0]),torch.tensor([0.]),horizon=.08,
                        dt=.01,output_dt=.02,use_checkpoint=False)
    assert len(calls)==4
    out[:,-1].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    hook.remove()


def test_reflection_equivariance_for_arbitrary_states_preserves_scalar_gradient():
    torch.manual_seed(5)
    net=model();net.enforce_reflection=True
    history=torch.randn(2,8,4,16)*.03
    history[:,:,[0,2]]+=1
    valid=torch.ones(2,8,dtype=torch.bool)
    k=torch.tensor([.32,.48]);alpha=torch.tensor([.02,.1])
    parity=history.new_tensor([1,-1,1,-1])[None,None,:,None]
    direct=net(history,valid,k,alpha)
    reflected=net(history.flip(-1)*parity,valid,k,alpha)
    torch.testing.assert_close(reflected,direct.flip(-1),atol=1e-8,rtol=1e-5)


def test_closure_recording_uses_existing_calls_and_current_accepted_state():
    torch.manual_seed(6)
    cache,net=fixture(),model()
    calls=[];observed=[]
    hook=net.register_forward_hook(lambda *_:calls.append(1))
    with torch.no_grad():
        advance_history(net,cache,torch.tensor([0]),torch.tensor([0.]),horizon=.04,
                        dt=.01,output_dt=.02,use_checkpoint=False,
                        closure_observer=lambda t,g:observed.append((t,g.clone())))
    assert len(calls)==16 and len(observed)==4
    np.testing.assert_allclose([t for t,_ in observed],[0,.01,.02,.03])
    history,valid=supervised_history(cache,torch.tensor([0]),torch.tensor([0.]),net.history_span)
    expected=net(history,valid,torch.tensor([.35]),torch.tensor([.02]))
    torch.testing.assert_close(observed[0][1],expected)
    hook.remove()


def test_autonomous_rollout_does_not_read_future_kinetic_frames():
    torch.manual_seed(7)
    cache,net=fixture(),model()
    ids=torch.tensor([0]);starts=torch.tensor([0.])
    with torch.no_grad():
        expected=advance_history(net,cache,ids,starts,horizon=.08,dt=.01,output_dt=.02,use_checkpoint=False)
        cache.state[:,1:]=float("nan")
        cache.gradient[:]=float("nan")
        actual=advance_history(net,cache,ids,starts,horizon=.08,dt=.01,output_dt=.02,use_checkpoint=False)
    torch.testing.assert_close(actual,expected)


def test_validity_observer_sees_all_rk_stages_and_accepted_states():
    cache,net=fixture(),model();times=[]
    with torch.no_grad():
        advance_history(net,cache,torch.tensor([0]),torch.tensor([0.]),horizon=.02,dt=.01,
                        output_dt=.02,use_checkpoint=False,step_observer=lambda t,_:times.append(t))
    np.testing.assert_allclose(times,[0,.005,.005,.01,.01,.01,.015,.015,.02,.02])
