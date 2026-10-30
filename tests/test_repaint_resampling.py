import pytest
import torch

from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.repaint import GaussianDiffusion


class RecordingDiffusion(GaussianDiffusion):
    def __init__(self, timesteps = 4, self_condition = False):
        model = Unet(
            dim = 8, dim_mults = (1,), channels = 1,
            attn_heads = 2, attn_dim_head = 4,
            self_condition = self_condition,
        ).eval()
        super().__init__(model, image_size = 4, timesteps = timesteps)
        self.calls = []

    def p_sample(self, x, t, x_self_cond = None, **kwargs):
        result = super().p_sample(x, t, x_self_cond, **kwargs)
        self.calls.append((t, x.clone(), result[0].clone(), x_self_cond))
        return result


def run_inpainting(diffusion, **kwargs):
    gt = torch.full((1, 1, 4, 4), 0.25)
    mask = torch.zeros_like(gt)
    return diffusion.sample(gt = gt, mask = mask, **kwargs)


def test_resampling_returns_to_current_timestep():
    diffusion = RecordingDiffusion()
    samples = run_inpainting(
        diffusion, resample_jump = 2, resample_iter = 2, resample_every = 2,
    )
    assert [call[0] for call in diffusion.calls] == [
        3, 2, 3, 2, 3, 2, 1, 2, 1, 2, 1, 0,
    ]
    assert torch.isfinite(samples).all()


def test_jump_is_bounded_by_noise_schedule():
    diffusion = RecordingDiffusion(timesteps = 3)
    samples = run_inpainting(
        diffusion, resample_jump = 10, resample_iter = 1, resample_every = 1,
    )
    assert [call[0] for call in diffusion.calls] == [2, 2, 1, 2, 1, 0]
    assert torch.isfinite(samples).all()


def test_forward_jump_has_correct_signal_attenuation(monkeypatch):
    diffusion = RecordingDiffusion()
    monkeypatch.setattr(torch, 'randn_like', lambda x, **kwargs: torch.zeros_like(x))
    run_inpainting(
        diffusion, resample_jump = 2, resample_iter = 1, resample_every = 2,
    )
    # The first jump starts at x_1 and ends at x_3. With zero noise,
    # q(x_3 | x_1) scales the signal by sqrt(alpha_bar_3 / alpha_bar_1).
    before_jump = diffusion.calls[1][2]
    after_jump = diffusion.calls[2][1]
    attenuation = (diffusion.alphas_cumprod[3] / diffusion.alphas_cumprod[1]).sqrt()
    torch.testing.assert_close(after_jump, before_jump * attenuation)


def test_resampling_keeps_self_conditioning():
    diffusion = RecordingDiffusion(self_condition = True)
    run_inpainting(
        diffusion, resample_jump = 2, resample_iter = 1, resample_every = 2,
    )
    assert diffusion.calls[0][3] is None
    assert all(call[3] is not None for call in diffusion.calls[1:])


@pytest.mark.parametrize('kwargs', [{'resample': False}, {'resample_iter': 0}])
def test_no_resampling_keeps_original_reverse_schedule(kwargs):
    diffusion = RecordingDiffusion()
    run_inpainting(diffusion, **kwargs)
    assert [call[0] for call in diffusion.calls] == [3, 2, 1, 0]
