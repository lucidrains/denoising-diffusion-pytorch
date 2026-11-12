import math
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from denoising_diffusion_pytorch import LearnedGaussianDiffusion, Unet
from denoising_diffusion_pytorch.learned_gaussian_diffusion import normal_kl, discretized_gaussian_log_likelihood


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_half_hybrid_training(dtype):
    torch.manual_seed(3)
    model = Unet(dim=8, dim_mults=(1,), channels=1, attn_heads=1, attn_dim_head=8, learned_variance=True).to(dtype)
    diffusion = LearnedGaussianDiffusion(model, image_size=8, timesteps=8, objective="pred_noise").to(dtype)
    image = torch.rand(2, 1, 8, 8, dtype=dtype) * 2 - 1
    noise = torch.randn_like(image)
    times = torch.tensor([0, 1])
    original = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    outputs = []
    model.register_forward_hook(lambda module, args, output: outputs.append(output.detach()))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        autocast = torch.autocast("cpu", dtype=dtype) if dtype != torch.float32 else nullcontext()
        with autocast:
            loss = diffusion.p_losses(image, times, noise=noise)
        assert torch.isfinite(loss), loss
        expected = native_hybrid_reference(diffusion, image, times, noise, outputs[-1])
        assert loss.item() == pytest.approx(expected, rel=3e-5, abs=3e-6)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.final_conv.weight.grad[1:].abs().sum() > 0
        optimizer.step()
    assert any(not torch.equal(before, after) for before, after in zip(original, model.parameters()))


def test_half_kl_zero_mean_difference():
    zero = torch.zeros(2, dtype=torch.float16, requires_grad=True)
    log_variance = torch.full((2,), -40., dtype=torch.float16, requires_grad=True)
    result = normal_kl(zero, log_variance, zero, log_variance)
    assert torch.isfinite(result).all(), result
    result.sum().backward()
    assert torch.isfinite(zero.grad).all()
    assert torch.isfinite(log_variance.grad).all()


def test_half_likelihood_tail():
    value = torch.tensor([0., 0.5, -0.5], dtype=torch.float16)
    mean = torch.tensor([0., -1., 1.], dtype=torch.float16, requires_grad=True)
    scale = torch.tensor([-20., -2., -2.], dtype=torch.float16, requires_grad=True)
    result = discretized_gaussian_log_likelihood(value, means=mean, log_scales=scale)
    assert torch.isfinite(result).all(), result
    result.sum().backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(scale.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_kl_matches_distribution_reference_and_gradients(dtype):
    values = [[0.2, 0., 0., -0.3], [-2., -20., -40., 1.], [0.1, 0., 0., -0.2], [-1.5, -19.5, -39.5, 0.8]]
    arguments = [torch.tensor(value, dtype=dtype, requires_grad=True) for value in values]
    reference_args = [value.detach().double().requires_grad_() for value in arguments]
    mean1, logvar1, mean2, logvar2 = reference_args
    reference = torch.distributions.kl_divergence(
        torch.distributions.Normal(mean1, (logvar1 / 2).exp()),
        torch.distributions.Normal(mean2, (logvar2 / 2).exp()),
    )
    result = normal_kl(*arguments)
    expected_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    assert result.dtype == expected_dtype
    torch.testing.assert_close(result.double(), reference, atol=2e-6, rtol=2e-5)
    actual_gradients = torch.autograd.grad(result.sum(), arguments)
    reference_gradients = torch.autograd.grad(reference.sum(), reference_args)
    for actual, expected in zip(actual_gradients, reference_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected.to(dtype), atol=2e-6, rtol=2e-5)


def likelihood_reference_and_derivatives(value, mean, log_scale):
    inverse = math.exp(-log_scale)
    upper = (value - mean + 1 / 255) * inverse
    lower = (value - mean - 1 / 255) * inverse

    def cdf_and_derivative(x):
        z = math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
        tanh = math.tanh(z)
        derivative = 0.5 * (1 - tanh**2) * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x**2)
        return 0.5 * (1 + tanh), derivative

    high, high_derivative = cdf_and_derivative(upper)
    low, low_derivative = cdf_and_derivative(lower)
    if value < -0.999:
        probability = high
        mean_derivative = -inverse * high_derivative
        scale_derivative = -upper * high_derivative
    elif value > 0.999:
        probability = 1 - low
        mean_derivative = inverse * low_derivative
        scale_derivative = lower * low_derivative
    else:
        probability = high - low
        mean_derivative = inverse * (low_derivative - high_derivative)
        scale_derivative = lower * low_derivative - upper * high_derivative
    if probability < 1e-15:
        return math.log(1e-15), 0., 0.
    return math.log(probability), mean_derivative / probability, scale_derivative / probability


def native_hybrid_reference(diffusion, image, times, noise, output):
    with torch.no_grad():
        noisy = diffusion.q_sample(image, times, noise=noise)
        true_mean, _, true_logvar = diffusion.q_posterior(image, noisy, times)
        model_mean, _, model_logvar, _ = diffusion.p_mean_variance(
            x=noisy, t=times, clip_denoised=False, model_output=output,
        )
        kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(true_mean.double(), (true_logvar.double() / 2).exp()),
            torch.distributions.Normal(model_mean.double(), (model_logvar.double() / 2).exp()),
        ).flatten(1).mean(1) / math.log(2)
        log_scales = 0.5 * model_logvar
        decoder = torch.tensor([
            -likelihood_reference_and_derivatives(value, mean, scale)[0]
            for value, mean, scale in zip(image.flatten().tolist(), model_mean.flatten().tolist(), log_scales.flatten().tolist())
        ], dtype=torch.float64).reshape_as(image).flatten(1).mean(1) / math.log(2)
        simple = (output[:, :1].float() - noise.float()).square().flatten(1).mean(1)
        return (simple.double() + torch.where(times == 0, decoder, kl) * diffusion.vb_loss_weight).mean().item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_likelihood_matches_scalar_bin_probabilities_and_derivatives(dtype):
    x = torch.tensor([0., 0.5, -0.5, -1., 1., -0.1, 0.2], dtype=dtype, requires_grad=True)
    means = torch.tensor([0., -1., 1., -0.8, 0.8, 0., -0.1], dtype=dtype, requires_grad=True)
    scales = torch.tensor([-20., -2., -2., -0.5, -0.5, -0.1, 0.2], dtype=dtype, requires_grad=True)
    expected = np.array([likelihood_reference_and_derivatives(*values) for values in zip(x.tolist(), means.tolist(), scales.tolist())])
    result = discretized_gaussian_log_likelihood(x, means=means, log_scales=scales)
    expected_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    assert result.dtype == expected_dtype
    np.testing.assert_allclose(result.detach().double().numpy(), expected[:, 0], atol=2e-5, rtol=3e-6)
    x_gradient, mean_gradient, scale_gradient = torch.autograd.grad(result.sum(), (x, means, scales))
    for actual, reference in [(x_gradient, -expected[:, 1]), (mean_gradient, expected[:, 1]), (scale_gradient, expected[:, 2])]:
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, torch.tensor(reference, dtype=dtype), atol=3e-5, rtol=3e-4)
