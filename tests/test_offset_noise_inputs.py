from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from denoising_diffusion_pytorch import denoising_diffusion_pytorch as ordinary
from denoising_diffusion_pytorch import classifier_free_guidance as conditional
from denoising_diffusion_pytorch import repaint


def make_diffusion(variant, objective, strength):
    module = {"ordinary": ordinary, "conditional": conditional, "repaint": repaint}[variant]
    options = dict(dim=8, dim_mults=(1, 2), channels=1, attn_heads=1, attn_dim_head=8)
    if variant == "conditional":
        options.update(num_classes=3, cond_drop_prob=0)
    model = module.Unet(**options)
    return module.GaussianDiffusion(
        model, image_size=8, timesteps=20, beta_schedule="cosine",
        objective=objective, offset_noise_strength=strength,
    )


def model_call(model, variant, x, times):
    if variant == "conditional":
        return model(x, times, torch.tensor([0, 2]), cond_keep_mask=torch.ones(2, dtype=torch.bool))
    return model(x, times, None)


def training_loss(diffusion, variant, x, times, noise):
    kwargs = {}
    if variant == "conditional":
        kwargs.update(classes=torch.tensor([0, 2]), cond_keep_mask=torch.ones(2, dtype=torch.bool))
    return diffusion.p_losses(x, times, noise=noise, **kwargs)


def reference_loss(diffusion, model, variant, x, times, noise, strength):
    noise = torch.randn_like(x) if noise is None else noise
    if strength:
        noise = noise + strength * torch.randn(x.shape[:2])[:, :, None, None]
    alpha = diffusion.sqrt_alphas_cumprod[times, None, None, None]
    sigma = diffusion.sqrt_one_minus_alphas_cumprod[times, None, None, None]
    prediction = model_call(model, variant, alpha * x + sigma * noise, times)
    target = {
        "pred_noise": noise,
        "pred_x0": x,
        "pred_v": alpha * noise - sigma * x,
    }[diffusion.objective]
    losses = F.mse_loss(prediction, target, reduction="none").flatten(1).mean(1)
    return (losses * diffusion.loss_weight[times]).mean()


@pytest.mark.parametrize("variant", ["ordinary", "conditional", "repaint"])
@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
@pytest.mark.parametrize("strength", [0.0, 0.4])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_offset_training_preserves_noise_and_matches_reference(variant, objective, strength, requires_grad):
    torch.manual_seed(16)
    diffusion = make_diffusion(variant, objective, strength)
    reference_model = deepcopy(diffusion.model)
    x = torch.randn(2, 1, 8, 8)
    times = torch.tensor([1, 8])
    noise = torch.randn_like(x).requires_grad_(requires_grad)
    original_noise = noise.detach().clone()
    reference_noise = noise.detach().clone().requires_grad_(requires_grad)

    torch.manual_seed(42)
    expected = reference_loss(diffusion, reference_model, variant, x, times, reference_noise, strength)
    expected.backward()

    torch.manual_seed(42)
    actual = training_loss(diffusion, variant, x, times, noise)
    actual.backward()

    torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected)
    for actual_parameter, expected_parameter in zip(diffusion.model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
    if requires_grad:
        assert torch.isfinite(noise.grad).all()
        torch.testing.assert_close(noise.grad, reference_noise.grad)

    torch.manual_seed(42)
    repeated = training_loss(diffusion, variant, x, times, noise)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("strength", [0.0, 0.4])
def test_conditional_offset_noise_retains_original_rounding(dtype, strength):
    torch.manual_seed(16)
    diffusion = make_diffusion("conditional", "pred_noise", strength)
    x = torch.randn(2, 1, 8, 8)
    times = torch.tensor([1, 8])
    noise = torch.randn_like(x).to(dtype)
    torch.manual_seed(42)
    augmented_noise = (noise + strength * torch.randn(2, 1, 1, 1)).to(dtype)
    expected = (
        diffusion.sqrt_alphas_cumprod[times, None, None, None] * x
        + diffusion.sqrt_one_minus_alphas_cumprod[times, None, None, None] * augmented_noise
    )

    torch.manual_seed(42)
    actual = diffusion.q_sample(x, times, noise=noise)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["ordinary", "conditional", "repaint"])
@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
def test_internally_generated_offset_noise_preserves_loss_targets(variant, objective):
    torch.manual_seed(16)
    diffusion = make_diffusion(variant, objective, 0.4)
    reference_model = deepcopy(diffusion.model)
    x = torch.randn(2, 1, 8, 8)
    times = torch.tensor([1, 8])

    torch.manual_seed(42)
    expected = reference_loss(diffusion, reference_model, variant, x, times, None, 0.4)
    torch.manual_seed(42)
    actual = training_loss(diffusion, variant, x, times, None)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("strength", [0.0, 0.4])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_conditional_q_sample_preserves_supplied_noise(strength, requires_grad):
    torch.manual_seed(16)
    diffusion = make_diffusion("conditional", "pred_noise", strength)
    x = torch.randn(2, 1, 8, 8)
    times = torch.tensor([1, 8])
    noise = torch.randn_like(x).requires_grad_(requires_grad)
    original_noise = noise.detach().clone()
    torch.manual_seed(42)
    augmented_noise = noise + strength * torch.randn(2, 1, 1, 1)
    expected = (
        diffusion.sqrt_alphas_cumprod[times, None, None, None] * x
        + diffusion.sqrt_one_minus_alphas_cumprod[times, None, None, None] * augmented_noise
    )

    torch.manual_seed(42)
    actual = diffusion.q_sample(x, times, noise=noise)

    torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected)
    if requires_grad:
        actual.sum().backward()
        assert torch.isfinite(noise.grad).all()
