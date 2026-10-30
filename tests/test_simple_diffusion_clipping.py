import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch.continuous_time_gaussian_diffusion import (
    ContinuousTimeGaussianDiffusion,
    alpha_cosine_log_snr,
)
from denoising_diffusion_pytorch.simple_diffusion import GaussianDiffusion, UViT
from denoising_diffusion_pytorch.v_param_continuous_time_gaussian_diffusion import (
    VParamContinuousTimeGaussianDiffusion,
)


class ConstantPrediction(nn.Module):
    random_or_learned_sinusoidal_cond = True
    self_condition = False

    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, x, time):
        return torch.ones_like(x) * self.value


@pytest.mark.parametrize('objective', ['eps', 'v'])
@pytest.mark.parametrize('clip', [False, True])
@pytest.mark.parametrize('prediction', [-4., 4.])
def test_posterior_matches_continuous_diffusion(objective, clip, prediction):
    model = ConstantPrediction(prediction)
    diffusion = GaussianDiffusion(
        model, image_size = 4, channels = 1, pred_objective = objective,
        noise_schedule = alpha_cosine_log_snr, clip_sample_denoised = clip,
    )
    reference_class = (
        ContinuousTimeGaussianDiffusion if objective == 'eps'
        else VParamContinuousTimeGaussianDiffusion
    )
    kwargs = {'noise_schedule': 'cosine'} if objective == 'eps' else {}
    reference = reference_class(
        model, image_size = 4, channels = 1,
        clip_sample_denoised = clip, **kwargs,
    )
    x = torch.linspace(-0.5, 0.5, 32).reshape(2, 1, 4, 4)
    time, time_next = torch.tensor(0.7), torch.tensor(0.2)
    actual_mean, actual_variance = diffusion.p_mean_variance(x, time, time_next)
    expected_mean, expected_variance = reference.p_mean_variance(x, time, time_next)
    torch.testing.assert_close(actual_mean, expected_mean)
    torch.testing.assert_close(actual_variance, expected_variance)


@pytest.mark.parametrize('clip', [False, True])
def test_uvit_sampling_stays_finite(clip):
    model = UViT(
        dim = 8, dim_mults = (1,), channels = 1, vit_depth = 1,
        attn_heads = 2, attn_dim_head = 4, vit_dropout = 0.,
    ).eval()
    diffusion = GaussianDiffusion(
        model, image_size = 4, channels = 1,
        num_sample_steps = 2, clip_sample_denoised = clip,
    )
    samples = diffusion.sample(batch_size = 2)
    assert samples.shape == (2, 1, 4, 4)
    assert torch.isfinite(samples).all()
    assert ((samples >= 0) & (samples <= 1)).all()
