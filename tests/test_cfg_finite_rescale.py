"""Guidance rescaling must be finite without concealing invalid network output."""
import pytest
import torch
from torch import nn
from denoising_diffusion_pytorch.classifier_free_guidance import Unet


class PredictionPair(nn.Module):
    def __init__(self, conditional, unconditional):
        super().__init__()
        self.conditional = nn.Parameter(conditional)
        self.unconditional = nn.Parameter(unconditional)

    def forward(self, *args, cond_drop_prob=0.0, **kwargs):
        return self.conditional if cond_drop_prob == 0 else self.unconditional

    def guided(self, phi):
        return Unet.forward_with_cond_scale(self, cond_scale=6.0, rescaled_phi=phi,
                                           remove_parallel_component=False)[0]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_constant_predictions_fall_back_to_unrescaled_output(dtype, value):
    model = PredictionPair(torch.full((2, 1, 4, 4), value, dtype=dtype),
                           torch.full((2, 1, 4, 4), value / 2, dtype=dtype))
    reference = model.guided(0)
    actual = model.guided(0.7)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference)
    actual.float().sum().backward()
    assert torch.isfinite(model.conditional.grad).all()
    assert torch.isfinite(model.unconditional.grad).all()


def test_single_value_per_sample_has_finite_gradient():
    model = PredictionPair(torch.tensor([1., 2.]).reshape(2, 1, 1, 1),
                           torch.tensor([0., 1.]).reshape(2, 1, 1, 1))
    result = model.guided(0.7)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, model.guided(0))
    result.sum().backward()
    assert torch.isfinite(model.conditional.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_regular_prediction_matches_independent_original_formula(dtype):
    generator = torch.Generator().manual_seed(7)
    conditional = torch.randn((2, 2, 5, 7), generator=generator).to(dtype)
    unconditional = torch.randn((2, 2, 5, 7), generator=generator).to(dtype)
    model = PredictionPair(conditional, unconditional)
    # Use the same guidance tensor, then independently evaluate the original std ratio.
    scaled = model.guided(0).double()
    dims = (1, 2, 3)
    expected = (0.7 * scaled * (conditional.double().std(dims, keepdim=True)
                / scaled.std(dims, keepdim=True)) + 0.3 * scaled).to(dtype)
    result = model.guided(0.7)
    tolerance = 0.07 if dtype == torch.bfloat16 else 0.01 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance)
    result.float().square().mean().backward()
    assert torch.isfinite(model.conditional.grad).all()


def test_fallback_is_per_sample_not_per_batch():
    conditional = torch.stack([torch.zeros(1, 4, 4), torch.arange(16.).reshape(1, 4, 4)])
    model = PredictionPair(conditional, conditional * 0.1)
    result = model.guided(0.7)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0], torch.zeros_like(result[0]))
    expected = model.guided(0)[1]
    expected = 0.7 * expected * (conditional[1].std() / expected.std()) + 0.3 * expected
    torch.testing.assert_close(result[1], expected)


def test_real_zero_output_unet_and_backward():
    torch.manual_seed(18)
    model = Unet(dim=8, num_classes=3, dim_mults=(1, 2), channels=1)
    with torch.no_grad():
        model.final_conv.weight.zero_()
        model.final_conv.bias.zero_()
    result, null = model.forward_with_cond_scale(torch.ones(2, 1, 8, 8),
        torch.tensor([1, 2]), torch.tensor([0, 1]), cond_scale=6, rescaled_phi=0.7)
    assert result.shape == (2, 1, 8, 8)
    assert torch.isfinite(result).all() and torch.isfinite(null).all()
    result.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_nonfinite_model_output_is_not_silently_sanitized():
    model = PredictionPair(torch.full((1, 1, 4, 4), float('nan')), torch.zeros(1, 1, 4, 4))
    assert not torch.isfinite(model.guided(0.7)).all()


def test_real_zero_output_unet_ddim_sampling_remains_finite():
    from denoising_diffusion_pytorch.classifier_free_guidance import GaussianDiffusion
    torch.manual_seed(41)
    model = Unet(dim=8, num_classes=3, dim_mults=(1, 2), channels=1)
    with torch.no_grad():
        model.final_conv.weight.zero_()
        model.final_conv.bias.zero_()
    diffusion = GaussianDiffusion(model, image_size=8, timesteps=4, sampling_timesteps=2)
    result = diffusion.sample(classes=torch.tensor([0, 1]), cond_scale=6., rescaled_phi=0.7)
    assert result.shape == (2, 1, 8, 8)
    assert torch.isfinite(result).all()
