import pytest
import torch

from denoising_diffusion_pytorch import ElucidatedDiffusion, Unet


def make_diffusion(steps):
    torch.manual_seed(16)
    model = Unet(
        dim=8, dim_mults=(1, 2), channels=1, attn_heads=1,
        attn_dim_head=8, learned_sinusoidal_cond=True,
    )
    return ElucidatedDiffusion(
        model, image_size=8, channels=1, num_sample_steps=steps,
        sigma_min=0.002, sigma_max=2.0, S_churn=0,
    )


@pytest.mark.parametrize("steps", [1, 2, 5, 32])
def test_schedule_is_finite_and_keeps_sampling_endpoints(steps):
    diffusion = make_diffusion(steps)

    schedule = diffusion.sample_schedule()

    assert schedule.shape == (steps + 1,)
    assert torch.isfinite(schedule).all()
    assert (schedule[:-1] > schedule[1:]).all()
    torch.testing.assert_close(schedule[0], torch.tensor(diffusion.sigma_max))
    assert schedule[-1] == 0
    if steps > 1:
        torch.testing.assert_close(schedule[-2], torch.tensor(diffusion.sigma_min))
        positions = torch.linspace(0, 1, steps)
        expected = (
            diffusion.sigma_max ** (1 / diffusion.rho)
            + positions * (diffusion.sigma_min ** (1 / diffusion.rho) - diffusion.sigma_max ** (1 / diffusion.rho))
        ) ** diffusion.rho
        torch.testing.assert_close(schedule[:-1], expected)


@pytest.mark.parametrize("method", ["sample", "sample_using_dpmpp"])
@pytest.mark.parametrize("override", [False, True])
def test_single_step_native_unet_matches_terminal_denoising(method, override):
    diffusion = make_diffusion(32 if override else 1)
    torch.manual_seed(42)
    initial_noise = diffusion.sigma_max * torch.randn(2, 1, 8, 8)
    with torch.no_grad():
        expected = diffusion.preconditioned_network_forward(
            initial_noise, diffusion.sigma_max, clamp=method == "sample"
        ).clamp(-1, 1) * 0.5 + 0.5
    calls = []
    handle = diffusion.net.register_forward_hook(lambda _module, _inputs, output: calls.append(output.shape))
    kwargs = {"num_sample_steps": 1} if override else {}
    try:
        torch.manual_seed(42)
        actual = getattr(diffusion, method)(batch_size=2, **kwargs)
    finally:
        handle.remove()

    assert calls == [torch.Size([2, 1, 8, 8])]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("method", ["sample", "sample_using_dpmpp", "sample_schedule"])
@pytest.mark.parametrize("steps", [0, -1])
def test_sampling_requires_at_least_one_step(method, steps):
    diffusion = make_diffusion(32)

    with pytest.raises(ValueError, match="at least 1"):
        getattr(diffusion, method)(num_sample_steps=steps)
