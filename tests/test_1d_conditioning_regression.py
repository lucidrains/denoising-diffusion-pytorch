"""Exercise conditioning through complete 1-D training and sampling paths."""
import pytest
import torch
from torch import nn
import denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d as implementation
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Unet1D

class ConditionalDenoiser(nn.Module):
    channels = 1

    def __init__(self, self_condition=False):
        super().__init__()
        self.self_condition = self_condition
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.calls = []

    def forward(self, x, t, *, condition, x_self_cond=None):
        self.calls.append((condition, x_self_cond, torch.is_grad_enabled()))
        output = self.scale * x + condition.reshape(-1, 1, 1)
        return output if x_self_cond is None else output + 0.1 * x_self_cond

def diffusion(self_condition=False, sampling_timesteps=None, objective='pred_noise'):
    model = ConditionalDenoiser(self_condition)
    return GaussianDiffusion1D(model, seq_length=8, timesteps=8, sampling_timesteps=sampling_timesteps, objective=objective, auto_normalize=False)

@pytest.mark.parametrize('objective', ['pred_noise', 'pred_x0', 'pred_v'])
def test_self_conditioning_keeps_external_condition_in_both_training_passes(monkeypatch, objective):
    monkeypatch.setattr(implementation, 'random', lambda: 0.0)
    model = diffusion(True, objective=objective)
    x = torch.ones(2, 1, 8) * 0.2
    t = torch.tensor([2, 4])
    condition = torch.tensor([0.1, 0.2])
    kwargs = {'condition': condition}
    loss = model.p_losses(x, t, noise=torch.zeros_like(x), model_forward_kwargs=kwargs)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.model.scale.grad)
    assert len(model.model.calls) == 2
    first, second = model.model.calls
    assert first[0] is condition and second[0] is condition
    assert first[2] is False and second[2] is True
    assert second[1] is not None and (not second[1].requires_grad)
    assert list(kwargs) == ['condition']

@pytest.mark.parametrize('self_condition', [False, True])
def test_ddpm_sampling_forwards_condition_and_self_condition(self_condition):
    model = diffusion(self_condition)
    condition = torch.tensor([0.1, 0.2])
    kwargs = {'condition': condition}
    result = model.sample(batch_size=2, model_forward_kwargs=kwargs)
    assert result.shape == (2, 1, 8) and torch.isfinite(result).all()
    assert len(model.model.calls) == 8
    assert all((call[0] is condition for call in model.model.calls))
    assert all((not call[2] for call in model.model.calls))
    if self_condition:
        assert all((call[1] is not None for call in model.model.calls[1:]))
    assert list(kwargs) == ['condition']

@pytest.mark.parametrize('self_condition', [False, True])
def test_ddim_sampling_forwards_condition_and_self_condition(self_condition):
    model = diffusion(self_condition, sampling_timesteps=3)
    condition = torch.tensor([0.1])
    result = model.sample(batch_size=1, model_forward_kwargs={'condition': condition})
    assert result.shape == (1, 1, 8) and torch.isfinite(result).all()
    assert all((call[0] is condition for call in model.model.calls))
    if self_condition:
        assert all((call[1] is not None for call in model.model.calls[1:]))

def test_no_self_conditioning_training_control(monkeypatch):
    monkeypatch.setattr(implementation, 'random', lambda: 1.0)
    model = diffusion(True)
    x = torch.zeros(1, 1, 8)
    condition = torch.zeros(1)
    loss = model.p_losses(x, torch.tensor([2]), noise=torch.ones_like(x), model_forward_kwargs={'condition': condition})
    assert torch.isfinite(loss)
    assert len(model.model.calls) == 1

@pytest.mark.parametrize('self_condition', [False, True])
def test_real_unet_forward_backward_and_ddpm_sampling(monkeypatch, self_condition):
    monkeypatch.setattr(implementation, 'random', lambda: 0.0)
    torch.manual_seed(0)
    denoiser = Unet1D(dim=8, dim_mults=(1, 2), channels=1, self_condition=self_condition)
    model = GaussianDiffusion1D(denoiser, seq_length=8, timesteps=4, auto_normalize=False)
    loss = model(torch.randn(1, 1, 8))
    loss.backward()
    assert torch.isfinite(loss)
    assert any((p.grad is not None and torch.isfinite(p.grad).all() for p in denoiser.parameters()))
    sample = model.sample(batch_size=1)
    assert sample.shape == (1, 1, 8) and torch.isfinite(sample).all()

def test_prediction_helper_accepts_the_builtin_self_condition_argument():
    torch.manual_seed(1)
    unet = Unet1D(dim=8, dim_mults=(1, 2), channels=1, self_condition=True)
    model = GaussianDiffusion1D(unet, seq_length=8, timesteps=4)
    x = torch.zeros(1, 1, 8)
    result = model.model_predictions(x, torch.tensor([1]), x_self_cond=torch.ones_like(x))
    assert torch.isfinite(result.pred_x_start).all()
