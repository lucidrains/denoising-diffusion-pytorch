import importlib

import pytest
import torch


@pytest.mark.parametrize(
    "module_name",
    [
        "denoising_diffusion_pytorch",
        "denoising_diffusion_pytorch_1d",
        "classifier_free_guidance",
        "guided_diffusion",
        "repaint",
        "simple_diffusion",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("value", [0.0, 40000.0])
def test_rms_normalization_preserves_finite_extreme_inputs(module_name, dtype, value):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{module_name}")
    layer = module.RMSNorm(8).to(dtype)
    if module_name == "simple_diffusion":
        shape = (2, 3, 8)
    elif module_name == "denoising_diffusion_pytorch_1d":
        shape = (2, 8, 4)
    else:
        shape = (2, 8, 2, 2)
    inputs = torch.full(shape, value, dtype=dtype, requires_grad=True)

    actual = layer(inputs)
    expected_value = (
        0.0 if value == 0.0 else (0.5 if module_name == "guided_diffusion" else 1.0)
    )
    expected = torch.full_like(actual, expected_value)
    tolerance = 1e-3 if dtype == torch.float16 else 1e-6
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    actual.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    torch.testing.assert_close(
        layer.g.grad,
        torch.full_like(layer.g.grad, 2 * expected_value**2 / 8),
        rtol=tolerance,
        atol=tolerance,
    )
