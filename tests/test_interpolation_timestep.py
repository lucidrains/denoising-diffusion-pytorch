"""Interpolation must reverse the noise level used to construct its initial state."""

import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch import GaussianDiffusion, Unet


class AnalyticDenoiser(nn.Module):
    channels = out_dim = 1

    def __init__(self, self_condition=False):
        super().__init__()
        self.self_condition = self_condition
        self.scale = nn.Parameter(torch.tensor(0.17))
        self.calls = []

    def forward(self, x, times, x_self_cond=None):
        self.calls.append(
            (times.detach().clone(), x_self_cond, torch.is_grad_enabled())
        )
        result = self.scale * x + times[:, None, None, None] * 0.003
        if x_self_cond is not None:
            result = result + 0.11 * x_self_cond
        return result


def make_diffusion(
    schedule, dtype, self_condition=False, model=None, objective="pred_noise"
):
    model = AnalyticDenoiser(self_condition) if model is None else model
    return GaussianDiffusion(
        model,
        image_size=4,
        timesteps=32,
        beta_schedule=schedule,
        objective=objective,
        auto_normalize=False,
    ).to(dtype=dtype)


def inputs(dtype, batch=2):
    first = torch.linspace(-0.6, 0.7, batch * 16, dtype=dtype).reshape(batch, 1, 4, 4)
    second = torch.linspace(0.4, -0.5, batch * 16, dtype=dtype).reshape_as(first)
    return first, second


def forward_noise(diffusion, clean, timestep, noise):
    alpha_bar = diffusion.alphas_cumprod[timestep]
    return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise


def posterior_step(diffusion, noisy, timestep, self_cond=None):
    times = torch.full(
        (noisy.shape[0],), timestep, dtype=torch.long, device=noisy.device
    )
    prediction = diffusion.model(noisy, times, self_cond)
    alpha_bar = diffusion.alphas_cumprod[timestep]
    alpha_previous = diffusion.alphas_cumprod_prev[timestep]
    beta = diffusion.betas[timestep]
    if diffusion.objective == "pred_noise":
        clean = (noisy - (1 - alpha_bar).sqrt() * prediction) / alpha_bar.sqrt()
    elif diffusion.objective == "pred_x0":
        clean = prediction
    else:
        clean = alpha_bar.sqrt() * noisy - (1 - alpha_bar).sqrt() * prediction
    clean = clean.clamp(-1, 1)
    mean = beta * alpha_previous.sqrt() / (1 - alpha_bar) * clean
    mean = (
        mean
        + (alpha_bar / alpha_previous).sqrt()
        * (1 - alpha_previous)
        / (1 - alpha_bar)
        * noisy
    )
    if timestep > 0:
        variance = beta * (1 - alpha_previous) / (1 - alpha_bar)
        mean = mean + variance.sqrt() * torch.randn_like(noisy)
    return mean, clean


def interpolation_reference(diffusion, first, second, timestep, lam):
    timestep = diffusion.num_timesteps - 1 if timestep is None else timestep
    noisy_first = forward_noise(diffusion, first, timestep, torch.randn_like(first))
    noisy_second = forward_noise(diffusion, second, timestep, torch.randn_like(second))
    noisy = (1 - lam) * noisy_first + lam * noisy_second
    clean = None
    for current in range(timestep, -1, -1):
        condition = clean if diffusion.self_condition else None
        noisy, clean = posterior_step(diffusion, noisy, current, condition)
    return noisy


def assert_values(actual, expected, dtype):
    # The constructor caches each independently calculated coefficient in float32,
    # including when the complete diffusion module is subsequently moved to float64.
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize("schedule", ("linear", "cosine", "sigmoid"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("timestep", (0, 5, 31))
def test_q_sample_corrupts_at_the_requested_timestep(schedule, dtype, timestep):
    diffusion = make_diffusion(schedule, dtype)
    first, noise = inputs(dtype)
    times = torch.full((2,), timestep, dtype=torch.long)
    actual = diffusion.q_sample(first, times, noise=noise)
    assert_values(actual, forward_noise(diffusion, first, timestep, noise), dtype)
    assert not torch.equal(actual, first)


@pytest.mark.parametrize("schedule", ("linear", "cosine", "sigmoid"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("timestep", (0, 5, 31))
def test_p_sample_matches_the_requested_posterior_transition(schedule, dtype, timestep):
    diffusion = make_diffusion(schedule, dtype)
    noisy, _ = inputs(dtype)
    torch.manual_seed(215)
    with torch.no_grad():
        expected, expected_clean = posterior_step(diffusion, noisy, timestep)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(215)
    actual, clean = diffusion.p_sample(noisy, timestep)
    assert_values(actual, expected, dtype)
    assert_values(clean, expected_clean, dtype)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)


@pytest.mark.parametrize("schedule", ("linear", "cosine", "sigmoid"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("timestep", (0, 5, None))
@pytest.mark.parametrize("self_condition", (False, True))
def test_interpolation_matches_independent_complete_posterior_chain(
    schedule, dtype, timestep, self_condition
):
    diffusion = make_diffusion(schedule, dtype, self_condition)
    first, second = inputs(dtype, batch=3)
    original_first, original_second = first.clone(), second.clone()
    torch.manual_seed(186)
    with torch.no_grad():
        expected = interpolation_reference(diffusion, first, second, timestep, 0.35)
    expected_rng = torch.get_rng_state()
    diffusion.model.calls.clear()
    torch.manual_seed(186)
    actual = diffusion.interpolate(first, second, t=timestep, lam=0.35)
    assert_values(actual, expected, dtype)
    start = diffusion.num_timesteps - 1 if timestep is None else timestep
    assert [int(call[0][0]) for call in diffusion.model.calls] == list(
        range(start, -1, -1)
    )
    assert all(not call[2] for call in diffusion.model.calls)
    if self_condition:
        assert diffusion.model.calls[0][1] is None
        assert all(call[1] is not None for call in diffusion.model.calls[1:])
    else:
        assert all(call[1] is None for call in diffusion.model.calls)
    assert actual.device == first.device and actual.dtype == first.dtype
    assert actual.shape == first.shape and torch.isfinite(actual).all()
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    torch.testing.assert_close(first, original_first, rtol=0, atol=0)
    torch.testing.assert_close(second, original_second, rtol=0, atol=0)


@pytest.mark.parametrize("objective", ("pred_noise", "pred_x0", "pred_v"))
@pytest.mark.parametrize("self_condition", (False, True))
@pytest.mark.parametrize("timestep", (0, 3, None))
def test_real_unet_interpolation_matches_posterior_reference(
    objective, self_condition, timestep
):
    torch.manual_seed(608)
    model = Unet(
        dim=8,
        dim_mults=(1, 2),
        channels=1,
        self_condition=self_condition,
        attn_heads=2,
        attn_dim_head=4,
        flash_attn=False,
    ).eval()
    diffusion = make_diffusion(
        "cosine", torch.float32, model=model, objective=objective
    )
    first, second = inputs(torch.float32)
    torch.manual_seed(952)
    with torch.no_grad():
        expected = interpolation_reference(diffusion, first, second, timestep, 0.5)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(952)
    actual = diffusion.interpolate(first, second, t=timestep)
    assert_values(actual, expected, torch.float32)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert torch.isfinite(actual).all() and not actual.requires_grad


@pytest.mark.parametrize("lam", (0.0, 0.5, 1.0))
def test_singleton_interpolation_endpoints(lam):
    diffusion = make_diffusion("sigmoid", torch.float32)
    first, second = inputs(torch.float32, batch=1)
    torch.manual_seed(59)
    with torch.no_grad():
        expected = interpolation_reference(diffusion, first, second, 1, lam)
    torch.manual_seed(59)
    actual = diffusion.interpolate(first, second, t=1, lam=lam)
    assert_values(actual, expected, torch.float32)


def test_ordinary_training_and_sampling_remain_native():
    torch.manual_seed(401)
    model = Unet(dim=8, dim_mults=(1, 2), channels=1, attn_heads=2, attn_dim_head=4)
    diffusion = make_diffusion("cosine", torch.float32, model=model)
    first, _ = inputs(torch.float32)
    loss = diffusion(first, times=torch.tensor([0, 5]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    model.eval()
    torch.manual_seed(730)
    result = diffusion.sample(batch_size=1)
    assert result.shape == (1, 1, 4, 4) and torch.isfinite(result).all()
