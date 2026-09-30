"""Loss targets must follow the permutation actually used to corrupt each image."""
import numpy as np
import pytest
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, extract


class ExactTarget(nn.Module):
    channels = out_dim = 1
    self_condition = False

    def __init__(self, clean, objective):
        super().__init__()
        self.clean = clean
        self.objective = objective
        self.error = nn.Parameter(torch.zeros_like(clean))
        self.seen = []

    def forward(self, x, t, x_self_cond=None):
        self.seen.append(x.detach().clone())
        alpha = extract(self.alpha, t, x.shape)
        sigma = extract(self.sigma, t, x.shape)
        noise = (x - alpha * self.clean) / sigma
        target = {"pred_noise": noise, "pred_x0": self.clean,
                  "pred_v": alpha * noise - sigma * self.clean}[self.objective]
        return target + self.error


def build(objective, immiscible=True, minimum_snr=False):
    clean = torch.tensor([-0.75, 0.75]).reshape(2, 1, 1, 1).expand(2, 1, 4, 4).clone()
    noise = torch.tensor([0.25, -0.25]).reshape(2, 1, 1, 1).expand_as(clean).clone()
    model = ExactTarget(clean, objective)
    diffusion = GaussianDiffusion(model, image_size=4, timesteps=8, beta_schedule="cosine",
                                  objective=objective, immiscible=immiscible,
                                  min_snr_loss_weight=minimum_snr)
    model.alpha, model.sigma = diffusion.sqrt_alphas_cumprod, diffusion.sqrt_one_minus_alphas_cumprod
    return diffusion, clean, noise, torch.tensor([2, 5])


@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
@pytest.mark.parametrize("minimum_snr", [False, True])
def test_perfect_assigned_target_has_zero_loss_and_gradient(objective, minimum_snr):
    diffusion, clean, noise, times = build(objective, minimum_snr=minimum_snr)
    expected_order = linear_sum_assignment(torch.cdist(clean.flatten(1), noise.flatten(1)).numpy())[1]
    np.testing.assert_array_equal(expected_order, [1, 0])
    losses = diffusion((clean + 1) / 2, times=times, noise=noise.clone(), loss_reduction="none")
    assert losses.shape == (2,)
    assert losses.max().item() < 1e-10
    losses.mean().backward()
    assert torch.isfinite(diffusion.model.error.grad).all()
    assert diffusion.model.error.grad.abs().max().item() < 1e-6
    expected_input = extract(diffusion.sqrt_alphas_cumprod, times, clean.shape) * clean
    expected_input += extract(diffusion.sqrt_one_minus_alphas_cumprod, times, clean.shape) * noise[expected_order]
    torch.testing.assert_close(diffusion.model.seen[0], expected_input)


@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
def test_disabled_immiscible_keeps_existing_targets(objective):
    diffusion, clean, noise, times = build(objective, immiscible=False)
    assert diffusion.p_losses(clean, times, noise=noise).item() < 1e-10


@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
def test_identity_assignment_remains_correct(objective):
    diffusion, clean, noise, times = build(objective)
    assert diffusion.p_losses(clean, times, noise=noise.flip(0)).item() < 1e-10


def test_q_sample_default_api_and_caller_input_are_unchanged():
    diffusion, clean, noise, times = build("pred_noise")
    original = noise.clone()
    expected_noise = noise.flip(0)
    result = diffusion.q_sample(clean, times, noise=noise)
    assert isinstance(result, torch.Tensor)
    expected = extract(diffusion.sqrt_alphas_cumprod, times, clean.shape) * clean
    expected += extract(diffusion.sqrt_one_minus_alphas_cumprod, times, clean.shape) * expected_noise
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(noise, original, rtol=0, atol=0)


def test_assignment_runs_once_per_training_call(monkeypatch):
    diffusion, clean, noise, times = build("pred_v")
    original = diffusion.noise_assignment
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(diffusion, "noise_assignment", counted)
    loss = diffusion.p_losses(clean, times, noise=noise)
    assert calls == [1]
    assert loss.item() < 1e-10


def test_model_error_gradient_matches_assigned_target():
    diffusion, clean, noise, times = build("pred_noise")
    with torch.no_grad():
        diffusion.model.error.fill_(0.125)
    loss = diffusion.p_losses(clean, times, noise=noise)
    loss.backward()
    weights = diffusion.loss_weight[times]
    expected_loss = (weights * 0.125 ** 2).mean()
    expected_grad = (2 * 0.125 * weights[:, None, None, None] / clean.numel()).expand_as(clean)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(diffusion.model.error.grad, expected_grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize('objective', ['pred_noise', 'pred_v'])
def test_real_unet_loss_matches_explicit_assigned_target(objective):
    from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet
    torch.manual_seed(103)
    denoiser = Unet(dim=8, dim_mults=(1, 2), channels=1)
    diffusion = GaussianDiffusion(denoiser, image_size=4, timesteps=8,
        beta_schedule='cosine', objective=objective, immiscible=True)
    _, clean, noise, times = build(objective)
    assigned_noise = noise.flip(0)
    noisy = extract(diffusion.sqrt_alphas_cumprod, times, clean.shape) * clean
    noisy += extract(diffusion.sqrt_one_minus_alphas_cumprod, times, clean.shape) * assigned_noise
    expected_output = denoiser(noisy, times)
    target = assigned_noise if objective == 'pred_noise' else diffusion.predict_v(clean, times, assigned_noise)
    per_sample = (expected_output - target).square().mean((1, 2, 3))
    expected_loss = (per_sample * diffusion.loss_weight[times]).mean()
    actual_loss = diffusion.p_losses(clean, times, noise=noise)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
    actual_loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in denoiser.parameters() if p.grad is not None)


def test_nonimmiscible_subclass_keeps_legacy_q_sample_signature(monkeypatch):
    diffusion, clean, noise, times = build("pred_noise", immiscible=False)
    original = diffusion.q_sample
    calls = []
    def legacy_q_sample(x_start, t, noise=None):
        calls.append(1)
        return original(x_start, t, noise=noise)
    monkeypatch.setattr(diffusion, "q_sample", legacy_q_sample)
    assert diffusion.p_losses(clean, times, noise=noise).item() < 1e-10
    assert calls == [1]
