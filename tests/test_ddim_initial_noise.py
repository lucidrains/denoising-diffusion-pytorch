import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import GaussianDiffusion1D


class Denoiser(nn.Module):
    channels = 1
    self_condition = False

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, time):
        return x * self.scale


@pytest.mark.parametrize('sampling_steps', (8, 4, 1))
@pytest.mark.parametrize('eta', (0., 1.))
@pytest.mark.parametrize('channel_first', (False, True))
@pytest.mark.parametrize('batch_size', (1, 2))
def test_returned_noise_is_the_initial_latent(
    sampling_steps, eta, channel_first, batch_size
):
    diffusion = GaussianDiffusion1D(
        Denoiser(),
        seq_length=8,
        timesteps=8,
        sampling_timesteps=sampling_steps,
        ddim_sampling_eta=eta,
        channel_first=channel_first,
    )
    shape = (batch_size, 1, 8) if channel_first else (batch_size, 8, 1)
    torch.manual_seed(42)
    initial_noise = torch.randn(shape)
    torch.manual_seed(42)
    sample, returned_noise = diffusion.sample(batch_size=batch_size, return_noise=True)
    rng_state = torch.get_rng_state()

    assert torch.equal(returned_noise, initial_noise)
    torch.manual_seed(42)
    sample_only = diffusion.sample(batch_size=batch_size)
    assert torch.equal(sample, sample_only)
    assert torch.equal(torch.get_rng_state(), rng_state)
