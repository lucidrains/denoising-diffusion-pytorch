import pytest
import torch

from denoising_diffusion_pytorch.attend import Attend


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [1, 4, 16])
@pytest.mark.parametrize(
    "scale_kind", ["default", "zero", "small", "large", "explicit_default"]
)
def test_sdpa_preserves_attention_scale(dim, scale_kind, dtype):
    scale = {
        "default": None,
        "zero": 0.,
        "small": 0.125,
        "large": 2.,
        "explicit_default": dim ** -0.5,
    }[scale_kind]
    generator = torch.Generator().manual_seed(42)
    q = torch.randn(2, 2, 3, dim, generator=generator, dtype=dtype)
    k = torch.randn(2, 2, 5, dim, generator=generator, dtype=dtype)
    v = torch.randn(2, 2, 5, 3, generator=generator, dtype=dtype)
    effective_scale = dim ** -0.5 if scale is None else scale
    expected = ((q @ k.transpose(-2, -1)) * effective_scale).softmax(-1) @ v

    eager = Attend(flash=False, scale=scale).eval()(q, k, v)
    sdpa = Attend(flash=True, scale=scale).eval()(q, k, v)
    tolerance = (
        dict(rtol=1e-10, atol=1e-10) if dtype == torch.float64
        else dict(rtol=1e-5, atol=1e-6)
    )
    torch.testing.assert_close(eager, expected, **tolerance)
    torch.testing.assert_close(sdpa, expected, **tolerance)


@pytest.mark.parametrize("dim", [4, 16])
@pytest.mark.parametrize("scale", [0.25, 1.])
def test_sdpa_preserves_attention_scale_gradients(dim, scale):
    generator = torch.Generator().manual_seed(17)
    q, k, v = [
        torch.randn(
            1, 2, length, dim, generator=generator,
            dtype=torch.float64, requires_grad=True
        )
        for length in [3, 5, 5]
    ]
    eager = Attend(flash=False, scale=scale).eval()(q, k, v)
    sdpa = Attend(flash=True, scale=scale).eval()(q, k, v)
    eager_gradients = torch.autograd.grad(eager.square().sum(), (q, k, v))
    sdpa_gradients = torch.autograd.grad(sdpa.square().sum(), (q, k, v))
    for actual, expected in zip(sdpa_gradients, eager_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
