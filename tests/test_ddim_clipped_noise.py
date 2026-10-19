import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import GaussianDiffusion1D


class Denoiser(nn.Module):
    channels = 1
    self_condition = False

    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(3.))

    def forward(self, x, time):
        return torch.ones_like(x) * self.value


@pytest.mark.parametrize('objective', ('pred_noise', 'pred_x0', 'pred_v'))
@pytest.mark.parametrize('clip_denoised', (False, True))
@pytest.mark.parametrize('channel_first', (False, True))
def test_ddim_predictions_reconstruct_the_current_noisy_sample(
    objective, clip_denoised, channel_first
):
    diffusion = GaussianDiffusion1D(
        Denoiser(),
        seq_length=8,
        timesteps=8,
        sampling_timesteps=4,
        objective=objective,
        channel_first=channel_first,
    )
    history = []
    predict = diffusion.model_predictions

    def record(x, times, *args, **kwargs):
        prediction = predict(x, times, *args, **kwargs)
        history.append((
            x.clone(), times.clone(),
            prediction.pred_noise.clone(), prediction.pred_x_start.clone(),
        ))
        return prediction

    diffusion.model_predictions = record
    shape = (2, 1, 8) if channel_first else (2, 8, 1)
    torch.manual_seed(0)
    output = diffusion.ddim_sample(shape, clip_denoised=clip_denoised)

    assert output.shape == shape
    assert len(history) == 4
    for noisy_sample, times, noise, clean_sample in history:
        alpha = diffusion.sqrt_alphas_cumprod[times].reshape(-1, 1, 1)
        sigma = diffusion.sqrt_one_minus_alphas_cumprod[times].reshape(-1, 1, 1)
        reconstructed = alpha * clean_sample + sigma * noise
        torch.testing.assert_close(reconstructed, noisy_sample, atol=1e-4, rtol=1e-5)
        if clip_denoised:
            assert clean_sample.abs().max() <= 1.
        elif objective == 'pred_noise':
            torch.testing.assert_close(noise, torch.full_like(noise, 3.))
