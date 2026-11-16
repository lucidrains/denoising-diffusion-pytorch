import importlib
from math import sqrt

import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch.elucidated_diffusion import ElucidatedDiffusion

MODULES = [
    "denoising_diffusion_pytorch",
    "denoising_diffusion_pytorch_1d",
    "guided_diffusion",
    "repaint",
    "classifier_free_guidance",
]


class ConstantDenoiser(nn.Module):
    channels = out_dim = 1
    self_condition = False
    random_or_learned_sinusoidal_cond = False

    def __init__(self, sequence=False):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1) if sequence else nn.Conv2d(1, 1, 1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.fill_(0.2)
        self.seen = []

    def forward(self, x, times, *args, **kwargs):
        self.seen.append(x.clone())
        return self.conv(x)

    def forward_with_cond_scale(self, x, times, classes, **kwargs):
        output = self.forward(x, times)
        return output, output


def make_diffusion(module_name, eta):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{module_name}")
    sequence = module_name.endswith("_1d")
    model = ConstantDenoiser(sequence)
    diffusion_type = (
        module.GaussianDiffusion1D if sequence else module.GaussianDiffusion
    )
    dimensions = {"seq_length": 4} if sequence else {"image_size": 4}
    diffusion = diffusion_type(
        model,
        timesteps=8,
        sampling_timesteps=4,
        beta_schedule="cosine",
        objective="pred_x0",
        ddim_sampling_eta=eta,
        **dimensions,
    )
    shape = (2, 1, 4) if sequence else (2, 1, 4, 4)
    return diffusion, model, shape


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize("eta", [0.0, 0.6])
def test_ddim_only_draws_used_noise(monkeypatch, module_name, eta):
    diffusion, model, shape = make_diffusion(module_name, eta)
    generator = torch.Generator().manual_seed(37)
    initial = torch.randn(shape, generator=generator)
    first_step_noise = torch.randn(shape, generator=generator) if eta else 0.0
    alpha, next_alpha = diffusion.alphas_cumprod[[7, 5]]
    clean = torch.full(shape, 0.2)
    epsilon = (initial - alpha.sqrt() * clean) / (1 - alpha).sqrt()
    variance = eta**2 * (1 - alpha / next_alpha) * (1 - next_alpha) / (1 - alpha)
    expected_second = (
        next_alpha.sqrt() * clean
        + (1 - next_alpha - variance).sqrt() * epsilon
        + variance.sqrt() * first_step_noise
    )
    calls = []
    native_randn_like = torch.randn_like

    def record_noise(tensor, *args, **kwargs):
        calls.append(tensor.numel())
        return native_randn_like(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "randn_like", record_noise)
    torch.manual_seed(37)
    if module_name == "classifier_free_guidance":
        output = diffusion.ddim_sample(
            torch.tensor([0, 1]), shape, cond_scale=2, rescaled_phi=0
        )
    else:
        output = diffusion.ddim_sample(shape)

    assert len(model.seen) == 4
    torch.testing.assert_close(model.seen[0], initial, rtol=0, atol=0)
    torch.testing.assert_close(model.seen[1], expected_second, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(output, torch.full(shape, 0.6), rtol=1e-6, atol=1e-6)
    assert calls == ([initial.numel()] * 3 if eta else [])
    if not eta:
        # Deterministic inference should consume only its initial latent.
        generator.manual_seed(37)
        torch.randn(shape, generator=generator)
        assert torch.equal(torch.get_rng_state(), generator.get_state())


@pytest.mark.parametrize("eta", [0.0, 0.6])
def test_1d_auxiliary_noise_return_keeps_existing_draws(monkeypatch, eta):
    diffusion, _, shape = make_diffusion("denoising_diffusion_pytorch_1d", eta)
    generator = torch.Generator().manual_seed(37)
    for _ in range(4):
        expected_last_noise = torch.randn(shape, generator=generator)
    calls = []
    native_randn_like = torch.randn_like

    def record_noise(tensor, *args, **kwargs):
        calls.append(tensor.numel())
        return native_randn_like(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "randn_like", record_noise)
    torch.manual_seed(37)
    output, returned_noise = diffusion.ddim_sample(shape, return_noise=True)
    torch.testing.assert_close(returned_noise, expected_last_noise, rtol=0, atol=0)
    torch.testing.assert_close(output, torch.full(shape, 0.6))
    assert calls == [returned_noise.numel()] * 3


@pytest.mark.parametrize(
    "churn,lower,upper",
    [(0.0, 0.0, 10.0), (0.5, 10.0, 20.0), (0.5, 0.0, 10.0), (0.5, 0.1, 0.8)],
)
def test_edm_only_draws_noise_during_active_churn(monkeypatch, churn, lower, upper):
    model = ConstantDenoiser()
    model.random_or_learned_sinusoidal_cond = True
    with torch.no_grad():
        model.conv.bias.zero_()
    diffusion = ElucidatedDiffusion(
        model,
        channels=1,
        image_size=4,
        num_sample_steps=4,
        sigma_min=0.1,
        sigma_max=2.0,
        sigma_data=0.5,
        S_churn=churn,
        S_tmin=lower,
        S_tmax=upper,
    )
    sigmas = diffusion.sample_schedule().tolist()
    shape = (2, 1, 4, 4)
    generator = torch.Generator().manual_seed(43)
    expected = sigmas[0] * torch.randn(shape, generator=generator)
    expected_draws = 1
    for sigma, next_sigma in zip(sigmas, sigmas[1:]):
        gamma = min(churn / 4, sqrt(2) - 1) if lower <= sigma <= upper else 0.0
        sigma_hat = sigma * (1 + gamma)
        if gamma:
            expected += (
                sqrt(sigma_hat**2 - sigma**2)
                * diffusion.S_noise
                * torch.randn(shape, generator=generator)
            )
            expected_draws += 1
        # For a zero denoiser, EDM's ODE is x' = sigma*x/(sigma**2+sigma_data**2).
        slope = sigma_hat / (sigma_hat**2 + 0.25)
        step = next_sigma - sigma_hat
        multiplier = 1 + step * slope
        if next_sigma:
            next_slope = next_sigma / (next_sigma**2 + 0.25)
            multiplier = 1 + 0.5 * step * (slope + next_slope * multiplier)
        expected = multiplier * expected
    expected = (expected.clamp(-1, 1) + 1) * 0.5
    calls = []
    native_randn = torch.randn

    def record_noise(*args, **kwargs):
        result = native_randn(*args, **kwargs)
        calls.append(result.numel())
        return result

    monkeypatch.setattr(torch, "randn", record_noise)
    torch.manual_seed(43)
    output = diffusion.sample(batch_size=2, clamp=False)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)
    assert len(model.seen) == 7
    assert calls == [expected.numel()] * expected_draws
    assert torch.equal(torch.get_rng_state(), generator.get_state())
