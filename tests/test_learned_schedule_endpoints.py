import copy

import pytest
import torch

from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.continuous_time_gaussian_diffusion import (
    ContinuousTimeGaussianDiffusion,
    learned_noise_schedule,
)


class FullBatchEndpointSchedule(learned_noise_schedule):
    def forward(self, x):
        zero = self.net(torch.zeros_like(x))
        one = self.net(torch.ones_like(x))
        value = self.slope * (self.net(x) - zero) / (one - zero) + self.intercept
        return value * self.frac_gradient + value.detach() * (1 - self.frac_gradient)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(), (1,), (32,), (2, 4), (0,)])
@pytest.mark.parametrize("fraction", [0., 0.3, 1.])
def test_schedule_matches_full_batch_values_and_derivatives(dtype, shape, fraction):
    torch.manual_seed(52)
    schedule = learned_noise_schedule(log_snr_max=9., log_snr_min=-10., hidden_dim=32, frac_gradient=fraction).to(dtype)
    reference = FullBatchEndpointSchedule(log_snr_max=9., log_snr_min=-10., hidden_dim=32, frac_gradient=fraction).to(dtype)
    reference.load_state_dict(copy.deepcopy(schedule.state_dict()))
    x = torch.rand(shape, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    weights = torch.rand_like(x)
    actual = schedule(x)
    expected = reference(reference_x)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), (x, *schedule.parameters()))
    expected_gradients = torch.autograd.grad((expected * weights).sum(), (reference_x, *reference.parameters()))
    for gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, expected_gradient, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("fraction", [0.3, 1.])
def test_native_learned_schedule_training_matches_full_batch_reference(fraction):
    torch.manual_seed(9)
    model = Unet(dim=8, dim_mults=(1,), channels=1, learned_sinusoidal_cond=True, attn_heads=1, attn_dim_head=8)
    diffusion = ContinuousTimeGaussianDiffusion(
        model, image_size=8, channels=1, noise_schedule="learned",
        learned_schedule_net_hidden_dim=32, learned_noise_schedule_frac_gradient=fraction, num_sample_steps=2,
    ).double()
    reference = copy.deepcopy(diffusion)
    reference.log_snr = FullBatchEndpointSchedule(
        log_snr_max=diffusion.log_snr.intercept,
        log_snr_min=diffusion.log_snr.intercept + diffusion.log_snr.slope,
        hidden_dim=32, frac_gradient=fraction,
    ).double()
    reference.log_snr.load_state_dict(copy.deepcopy(diffusion.log_snr.state_dict()))
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.001)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    original = [p.detach().clone() for p in diffusion.log_snr.parameters()]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        image = torch.rand(2, 1, 8, 8, dtype=torch.float64) * 2 - 1
        noise = torch.randn_like(image)
        times = torch.tensor([0.2, 0.8], dtype=torch.float64)
        loss = diffusion.p_losses(image, times, noise=noise)
        expected = reference.p_losses(image, times, noise=noise)
        torch.testing.assert_close(loss, expected, rtol=1e-10, atol=1e-12)
        loss.backward()
        expected.backward()
        for parameter, expected_parameter in zip(diffusion.parameters(), reference.parameters()):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=1e-9, atol=1e-11)
        optimizer.step()
        reference_optimizer.step()
    assert any(not torch.equal(before, after) for before, after in zip(original, diffusion.log_snr.parameters()))
    for parameter, expected_parameter in zip(diffusion.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected_parameter, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_schedule_stays_monotonic_and_has_fixed_endpoints(dtype):
    torch.manual_seed(29)
    schedule = learned_noise_schedule(log_snr_max=9., log_snr_min=-10., hidden_dim=32).to(dtype)
    values = schedule(torch.linspace(0, 1, 129, dtype=dtype))
    torch.testing.assert_close(values[[0, -1]], torch.tensor([9., -10.], dtype=dtype), rtol=1e-6, atol=1e-6)
    assert (values.diff() < 0).all()
