import math

import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch import LearnedGaussianDiffusion, Unet

OBJECTIVES = ("pred_noise", "pred_x0", "pred_v")
DTYPES = (torch.float32, torch.float64)


class TrainableHeads(nn.Module):
    channels = 1
    out_dim = 2
    self_condition = False

    def __init__(self, dtype):
        super().__init__()
        self.prediction = nn.Parameter(torch.zeros(2, 1, 4, 4, dtype=dtype))
        self.variance = nn.Parameter(torch.zeros_like(self.prediction))
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return torch.cat((self.prediction, self.variance), dim=1)


def setup_case(objective, dtype, *, vb_loss_weight=0.0, amplitude=1.6):
    model = TrainableHeads(dtype)
    diffusion = LearnedGaussianDiffusion(
        model,
        image_size=4,
        timesteps=8,
        objective=objective,
        vb_loss_weight=vb_loss_weight,
        auto_normalize=False,
    ).to(dtype=dtype)
    clean = torch.linspace(-amplitude, amplitude, 32, dtype=dtype).reshape(2, 1, 4, 4)
    noise = torch.linspace(0.9, -0.7, 32, dtype=dtype).reshape_as(clean)
    times = torch.tensor([1, 6])
    alpha = diffusion.alphas_cumprod[times].sqrt().reshape(2, 1, 1, 1)
    sigma = (1 - diffusion.alphas_cumprod[times]).sqrt().reshape(2, 1, 1, 1)
    noisy = alpha * clean + sigma * noise
    if objective == "pred_noise":
        target = noise
    elif objective == "pred_x0":
        target = clean
    else:
        target = alpha * noise - sigma * clean
    with torch.no_grad():
        model.prediction.copy_(target)
        model.variance.copy_(torch.linspace(-0.6, 0.4, 32).reshape_as(clean))
    return model, diffusion, clean, noise, times, noisy, alpha, sigma


def tolerance(dtype):
    return (
        {"rtol": 2e-5, "atol": 2e-6}
        if dtype == torch.float32
        else {"rtol": 2e-6, "atol": 2e-7}
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_inherited_default_is_a_working_velocity_objective(dtype):
    model, _, clean, noise, times, noisy, *_ = setup_case("pred_v", dtype)
    diffusion = LearnedGaussianDiffusion(
        model, image_size=4, timesteps=8, vb_loss_weight=0.0, auto_normalize=False
    ).to(dtype=dtype)
    assert diffusion.objective == "pred_v"
    loss = diffusion.p_losses(clean, times, noise=noise)
    torch.testing.assert_close(loss, torch.zeros_like(loss), **tolerance(dtype))
    *_, reconstructed = diffusion.p_mean_variance(x=noisy, t=times, clip_denoised=False)
    torch.testing.assert_close(reconstructed, clean, **tolerance(dtype))


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("clip_denoised", (False, True))
@pytest.mark.parametrize("reduction", ("mean", "none"))
def test_selected_target_loss_and_gradient(objective, dtype, clip_denoised, reduction):
    model, diffusion, clean, noise, times, *_ = setup_case(objective, dtype)
    offset = torch.tensor([0.3, -0.4], dtype=dtype).reshape(2, 1, 1, 1)
    with torch.no_grad():
        model.prediction.add_(offset)
    original_clean, original_noise = clean.clone(), noise.clone()
    original_head = model.prediction.detach().clone()

    loss = diffusion.p_losses(
        clean, times, noise=noise, clip_denoised=clip_denoised, loss_reduction=reduction
    )
    expected = offset.flatten().square()
    if reduction == "mean":
        expected = expected.mean()
    torch.testing.assert_close(loss, expected, **tolerance(dtype))
    head_gradient, variance_gradient = torch.autograd.grad(
        loss.sum(), (model.prediction, model.variance)
    )
    expected_gradient = (2 * offset / 16).expand_as(model.prediction)
    if reduction == "mean":
        expected_gradient = expected_gradient / 2
    torch.testing.assert_close(head_gradient, expected_gradient, **tolerance(dtype))
    torch.testing.assert_close(variance_gradient, torch.zeros_like(variance_gradient))
    torch.testing.assert_close(model.prediction, original_head, rtol=0, atol=0)
    torch.testing.assert_close(clean, original_clean, rtol=0, atol=0)
    torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)
    assert model.calls == 1


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("clip_denoised", (False, True))
@pytest.mark.parametrize("supplied_output", (False, True))
def test_posterior_uses_selected_head_and_preserves_variance(
    objective, dtype, clip_denoised, supplied_output
):
    model, diffusion, clean, _, times, noisy, *_ = setup_case(objective, dtype)
    output = model(noisy, times) if supplied_output else None
    original_output = output.detach().clone() if output is not None else None
    model.calls = 0
    mean, variance, log_variance, reconstructed = diffusion.p_mean_variance(
        x=noisy, t=times, clip_denoised=clip_denoised, model_output=output
    )
    expected_clean = clean.clamp(-1, 1) if clip_denoised else clean
    beta = diffusion.betas[times].reshape(2, 1, 1, 1)
    alpha = diffusion.alphas_cumprod[times].reshape(2, 1, 1, 1)
    previous_alpha = diffusion.alphas_cumprod_prev[times].reshape(2, 1, 1, 1)
    coefficient1 = beta * previous_alpha.sqrt() / (1 - alpha)
    coefficient2 = (alpha / previous_alpha).sqrt() * (1 - previous_alpha) / (1 - alpha)
    expected_mean = coefficient1 * expected_clean + coefficient2 * noisy
    minimum_log = (beta * (1 - previous_alpha) / (1 - alpha)).clamp_min(1e-20).log()
    fraction = (model.variance + 1) / 2
    expected_log = fraction * beta.log() + (1 - fraction) * minimum_log
    torch.testing.assert_close(reconstructed, expected_clean, **tolerance(dtype))
    torch.testing.assert_close(mean, expected_mean, **tolerance(dtype))
    torch.testing.assert_close(log_variance, expected_log, **tolerance(dtype))
    torch.testing.assert_close(variance, expected_log.exp(), **tolerance(dtype))
    if supplied_output:
        torch.testing.assert_close(output, original_output, rtol=0, atol=0)
    assert model.calls == (0 if supplied_output else 1)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_posterior_input_and_mean_head_derivatives(objective, dtype):
    model, diffusion, _, _, times, noisy, alpha, sigma = setup_case(
        objective, dtype, amplitude=0.4
    )
    noisy.requires_grad_()
    mean, *_ = diffusion.p_mean_variance(x=noisy, t=times, clip_denoised=True)
    input_gradient, head_gradient = torch.autograd.grad(
        mean.sum(), (noisy, model.prediction)
    )
    coefficient1 = diffusion.posterior_mean_coef1[times].reshape(2, 1, 1, 1)
    coefficient2 = diffusion.posterior_mean_coef2[times].reshape(2, 1, 1, 1)
    if objective == "pred_noise":
        input_scale, head_scale = 1 / alpha, -sigma / alpha
    elif objective == "pred_x0":
        input_scale, head_scale = torch.zeros_like(alpha), torch.ones_like(alpha)
    else:
        input_scale, head_scale = alpha, -sigma
    torch.testing.assert_close(
        input_gradient,
        (coefficient1 * input_scale + coefficient2).expand_as(noisy),
        **tolerance(dtype),
    )
    torch.testing.assert_close(
        head_gradient,
        (coefficient1 * head_scale).expand_as(model.prediction),
        **tolerance(dtype),
    )


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_hybrid_gaussian_kl_and_detached_mean_gradient(objective, dtype):
    weight = 0.013
    model, diffusion, clean, noise, times, *_ = setup_case(
        objective, dtype, vb_loss_weight=weight, amplitude=0.4
    )
    losses = diffusion.p_losses(clean, times, noise=noise, loss_reduction="none")
    minimum_log = diffusion.posterior_log_variance_clipped[times].reshape(2, 1, 1, 1)
    maximum_log = diffusion.betas[times].log().reshape(2, 1, 1, 1)
    model_log = minimum_log + (model.variance + 1) * (maximum_log - minimum_log) / 2
    ratio = (minimum_log - model_log).exp()
    gaussian_kl = (model_log - minimum_log + ratio - 1) / (2 * math.log(2))
    expected_losses = gaussian_kl.flatten(1).mean(1) * weight
    torch.testing.assert_close(losses, expected_losses, **tolerance(dtype))
    head_gradient, variance_gradient = torch.autograd.grad(
        losses.sum(), (model.prediction, model.variance)
    )
    expected_gradient = (
        weight * (1 - ratio) * (maximum_log - minimum_log) / (64 * math.log(2))
    )
    torch.testing.assert_close(
        head_gradient, torch.zeros_like(head_gradient), **tolerance(dtype)
    )
    torch.testing.assert_close(variance_gradient, expected_gradient, **tolerance(dtype))


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("clip_x_start", (False, True))
@pytest.mark.parametrize("rederive_noise", (False, True))
def test_prediction_mean_noise_consistency(
    objective, dtype, clip_x_start, rederive_noise
):
    model, diffusion, clean, noise, times, noisy, alpha, sigma = setup_case(
        objective, dtype
    )
    prediction = diffusion.model_predictions(
        noisy, times, clip_x_start=clip_x_start, rederive_pred_noise=rederive_noise
    )
    expected_clean = clean.clamp(-1, 1) if clip_x_start else clean
    expected_noise = noise
    if objective != "pred_noise" or (clip_x_start and rederive_noise):
        expected_noise = (noisy - alpha * expected_clean) / sigma
    torch.testing.assert_close(
        prediction.pred_x_start, expected_clean, **tolerance(dtype)
    )
    torch.testing.assert_close(
        prediction.pred_noise, expected_noise, **tolerance(dtype)
    )
    torch.testing.assert_close(prediction.pred_variance, model.variance, rtol=0, atol=0)
    assert model.calls == 1


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_real_unet_training_loss_backward_and_hybrid_decoder(objective):
    torch.manual_seed(451)
    model = Unet(
        dim=8,
        dim_mults=(1, 2),
        channels=1,
        learned_variance=True,
        attn_heads=2,
        attn_dim_head=8,
        flash_attn=False,
    )
    diffusion = LearnedGaussianDiffusion(
        model, image_size=8, timesteps=8, objective=objective, vb_loss_weight=0.0
    )
    clean = torch.randn(2, 1, 8, 8).clamp(-0.8, 0.8)
    noise = torch.randn_like(clean)
    times = torch.tensor([0, 5])
    alpha = diffusion.alphas_cumprod[times].sqrt().reshape(2, 1, 1, 1)
    sigma = (1 - diffusion.alphas_cumprod[times]).sqrt().reshape(2, 1, 1, 1)
    noisy = alpha * clean + sigma * noise
    head, _ = model(noisy, times).chunk(2, dim=1)
    target = {
        "pred_noise": noise,
        "pred_x0": clean,
        "pred_v": alpha * noise - sigma * clean,
    }[objective]
    expected_loss = (head - target).square().mean()
    loss = diffusion.p_losses(clean, times, noise=noise)
    torch.testing.assert_close(loss, expected_loss, rtol=2e-5, atol=2e-6)
    loss.backward()
    assert torch.isfinite(model.final_conv.weight.grad).all()
    assert model.final_conv.weight.grad[:1].norm() > 0

    model.zero_grad(set_to_none=True)
    diffusion.vb_loss_weight = 0.001
    hybrid_loss = diffusion.p_losses(clean, times, noise=noise, clip_denoised=True)
    hybrid_loss.backward()
    assert torch.isfinite(hybrid_loss)
    assert torch.isfinite(model.final_conv.weight.grad).all()
    assert model.final_conv.weight.grad[1:].norm() > 0


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("sampling_steps", (2, 8))
def test_real_unet_ddim_and_ddpm_sampling(objective, sampling_steps):
    torch.manual_seed(816)
    model = Unet(
        dim=8,
        dim_mults=(1, 2),
        channels=1,
        learned_variance=True,
        attn_heads=2,
        attn_dim_head=8,
        flash_attn=False,
    ).eval()
    diffusion = LearnedGaussianDiffusion(
        model,
        image_size=8,
        timesteps=8,
        sampling_timesteps=sampling_steps,
        objective=objective,
    )
    images = diffusion.sample(batch_size=2)
    assert images.shape == (2, 1, 8, 8)
    assert torch.isfinite(images).all()
    assert images.min() >= 0 and images.max() <= 1
