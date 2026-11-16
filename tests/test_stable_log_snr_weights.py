import copy
from decimal import Decimal, localcontext

import pytest
import torch

from denoising_diffusion_pytorch.continuous_time_gaussian_diffusion import (
    ContinuousTimeGaussianDiffusion,
)
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.simple_diffusion import GaussianDiffusion, UViT


def decimal_weights(log_snr, objective, gamma):
    values = []
    with localcontext() as context:
        context.prec = 80
        for value in log_snr.tolist():
            snr = Decimal(str(value)).exp()
            numerator = snr if gamma is None else min(snr, Decimal(str(gamma)))
            denominator = snr + 1 if objective == "v" else snr
            values.append(float(numerator / denominator))
    return torch.tensor(values, dtype=torch.float32)


def check_native_loss_and_gradients(diffusion, times, objective, gamma):
    reference = copy.deepcopy(diffusion.model)
    shape = (len(times), 1, 4, 4)
    inputs = torch.linspace(-0.8, 0.9, len(times) * 16).reshape(shape)
    noise = torch.linspace(0.7, -0.6, len(times) * 16).reshape(shape)
    noised, log_snr = diffusion.q_sample(inputs, times, noise)
    prediction = reference(noised, log_snr)
    if objective == "v":
        alpha = log_snr.sigmoid().sqrt().reshape(-1, 1, 1, 1)
        sigma = (-log_snr).sigmoid().sqrt().reshape(-1, 1, 1, 1)
        target = alpha * noise - sigma * inputs
    else:
        target = noise
    expected = (prediction - target).square().flatten(1).mean(1)
    expected = expected * decimal_weights(log_snr, objective, gamma)
    actual = diffusion.p_losses(inputs, times, noise, loss_reduction="none")
    assert torch.isfinite(actual).all()
    tolerance = 0.015 if times.dtype == torch.float16 else 3e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=1e-10)
    actual.mean().backward()
    expected.mean().backward()
    for parameter, oracle in zip(diffusion.model.parameters(), reference.parameters()):
        if oracle.grad is None:
            assert parameter.grad is None
            continue
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(
            parameter.grad,
            oracle.grad,
            rtol=tolerance,
            atol=2e-6 if times.dtype == torch.float16 else 2e-7,
        )


@pytest.mark.parametrize("time_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("limit", [5.0, 1000.0])
@pytest.mark.parametrize("objective", ["eps", "v"])
@pytest.mark.parametrize("gamma", [None, 0.0, 5.0])
def test_native_uvit_loss_matches_high_precision_weights(
    time_dtype, limit, objective, gamma
):
    torch.manual_seed(43)
    model = UViT(
        dim=8,
        dim_mults=(1,),
        channels=1,
        vit_depth=1,
        vit_dropout=0,
        attn_heads=1,
        attn_dim_head=4,
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=4,
        channels=1,
        pred_objective=objective,
        noise_schedule=lambda t: limit * (1 - 2 * t),
        min_snr_loss_weight=gamma is not None,
        min_snr_gamma=5.0 if gamma is None else gamma,
    )
    times = torch.tensor([0.0, 0.45, 0.5, 0.55, 1.0], dtype=time_dtype)
    check_native_loss_and_gradients(diffusion, times, objective, gamma)


@pytest.mark.parametrize("time_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("gamma", [0.0, 5.0])
def test_continuous_cosine_weight_does_not_drop_high_snr_samples(time_dtype, gamma):
    torch.manual_seed(47)
    model = Unet(
        dim=8,
        dim_mults=(1,),
        channels=1,
        random_fourier_features=True,
        attn_heads=1,
        attn_dim_head=4,
    )
    diffusion = ContinuousTimeGaussianDiffusion(
        model,
        image_size=4,
        channels=1,
        noise_schedule="cosine",
        min_snr_loss_weight=True,
        min_snr_gamma=gamma,
    )
    times = torch.tensor([0.0, 0.0001, 0.01, 0.1, 0.5, 0.8], dtype=time_dtype)
    check_native_loss_and_gradients(diffusion, times, "eps", gamma)
