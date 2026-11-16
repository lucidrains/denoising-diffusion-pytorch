import copy
from decimal import Decimal, localcontext

import pytest
import torch

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.elucidated_diffusion import ElucidatedDiffusion


def decimal_coefficients(value, sigma_data=0.5):
    with localcontext() as context:
        context.prec = 80
        sigma, data = Decimal(str(value)), Decimal(str(sigma_data))
        variance = sigma * sigma + data * data
        return {
            "c_skip": float(data * data / variance),
            "c_out": float(sigma * data / variance.sqrt()),
            "c_in": float(1 / variance.sqrt()),
            "loss_weight": float(variance / ((sigma * data) ** 2)),
        }


def make_diffusion():
    torch.manual_seed(107)
    model = Unet(
        dim=8,
        dim_mults=(1,),
        channels=1,
        random_fourier_features=True,
        attn_heads=1,
        attn_dim_head=4,
    )
    return ElucidatedDiffusion(model, image_size=4, channels=1)


@pytest.mark.parametrize(
    "value,dtype",
    [
        (0.5, None),
        (1000.0, None),
        (1e200, None),
        (0.5, torch.float32),
        (1000.0, torch.float32),
        (1e20, torch.float32),
        (0.5, torch.float16),
        (1000.0, torch.float16),
    ],
)
def test_coefficients_match_high_precision_formula_for_finite_sigmas(value, dtype):
    diffusion = make_diffusion()
    sigma = value if dtype is None else torch.tensor(value, dtype=dtype)
    stored_value = sigma if dtype is None else sigma.item()
    expected = decimal_coefficients(stored_value)
    for name, wanted in expected.items():
        actual = torch.as_tensor(getattr(diffusion, name)(sigma), dtype=torch.float64)
        assert torch.isfinite(actual)
        torch.testing.assert_close(
            actual, torch.tensor(wanted, dtype=torch.float64), rtol=2e-4, atol=1e-45
        )


@pytest.mark.parametrize(
    "dtype,values",
    [
        (torch.float32, [0.002, 0.02, 0.5, 80.0]),
        (torch.float16, [0.002, 0.02, 0.5, 80.0]),
        (torch.float32, [100.0, 1e10, 1e20, 3e20]),
        (torch.float16, [100.0, 500.0, 1000.0, 5000.0]),
    ],
)
def test_native_edm_loss_and_model_gradients_match_decimal_reference(dtype, values):
    diffusion = make_diffusion()
    reference_model = copy.deepcopy(diffusion.net)
    sigmas = torch.tensor(values, dtype=dtype)
    images = torch.linspace(0.1, 0.9, 4 * 16).reshape(4, 1, 4, 4)
    normalized = images * 2 - 1
    torch.manual_seed(109)
    noise = torch.randn_like(images)
    noised = normalized + sigmas.reshape(-1, 1, 1, 1) * noise
    rows = [decimal_coefficients(value) for value in sigmas.tolist()]
    coefficient = lambda name: torch.tensor([row[name] for row in rows]).reshape(
        -1, 1, 1, 1
    )
    prediction = reference_model(
        coefficient("c_in") * noised, diffusion.c_noise(sigmas)
    )
    denoised = coefficient("c_skip") * noised + coefficient("c_out") * prediction
    expected = (denoised - normalized).square().flatten(1).mean(1)
    expected = expected * coefficient("loss_weight").flatten()
    torch.manual_seed(109)
    actual = diffusion(images, sigma=sigmas, loss_reduction="none")
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1e-5)
    actual.mean().backward()
    expected.mean().backward()
    for parameter, oracle in zip(
        diffusion.net.parameters(), reference_model.parameters()
    ):
        if oracle.grad is None:
            assert parameter.grad is None
            continue
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, oracle.grad, rtol=5e-3, atol=2e-5)


@pytest.mark.parametrize("sampler", ["sample", "sample_using_dpmpp"])
@pytest.mark.parametrize("sigma_max", [80.0, 1e20])
def test_native_sampling_matches_decimal_preconditioning(
    monkeypatch, sampler, sigma_max
):
    diffusion = make_diffusion().eval()
    diffusion.sigma_max = sigma_max
    diffusion.num_sample_steps = 4
    reference = copy.deepcopy(diffusion)

    def oracle_coefficient(sigma, name):
        values = [
            decimal_coefficients(value)[name] for value in sigma.flatten().tolist()
        ]
        return sigma.new_tensor(values).reshape(sigma.shape)

    for name in ("c_skip", "c_out", "c_in"):
        monkeypatch.setattr(
            reference, name, lambda sigma, name=name: oracle_coefficient(sigma, name)
        )

    torch.manual_seed(113)
    expected = getattr(reference, sampler)(batch_size=2)
    torch.manual_seed(113)
    actual = getattr(diffusion, sampler)(batch_size=2)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=5e-6)
