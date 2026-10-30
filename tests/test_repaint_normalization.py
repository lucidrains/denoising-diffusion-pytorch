import pytest
import torch

from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.repaint import GaussianDiffusion


def make_diffusion(auto_normalize):
    model = Unet(
        dim = 8, dim_mults = (1,), channels = 1,
        attn_heads = 2, attn_dim_head = 4,
    ).eval()
    return GaussianDiffusion(
        model, image_size = 4, timesteps = 3,
        auto_normalize = auto_normalize,
    )


@pytest.mark.parametrize('auto_normalize', [False, True])
@pytest.mark.parametrize('full_mask', [False, True])
def test_sample_preserves_known_pixels_in_input_range(auto_normalize, full_mask):
    diffusion = make_diffusion(auto_normalize)
    gt = torch.linspace(0, 1, 32).reshape(2, 1, 4, 4)
    if not auto_normalize:
        gt = gt * 2 - 1
    original = gt.clone()
    mask = torch.ones_like(gt)
    if not full_mask:
        mask[..., 2:] = 0

    sample = diffusion.sample(gt = gt, mask = mask, resample = False)

    torch.testing.assert_close(sample[mask.bool()], gt[mask.bool()])
    torch.testing.assert_close(gt, original)
    assert torch.isfinite(sample).all()


@pytest.mark.parametrize('auto_normalize', [False, True])
def test_final_step_uses_configured_normalization(auto_normalize):
    diffusion = make_diffusion(auto_normalize)
    gt = torch.full((1, 1, 4, 4), 0.25)
    sample, _ = diffusion.p_sample(
        torch.zeros_like(gt), 0, gt = gt, mask = torch.ones_like(gt),
    )
    expected = diffusion.normalize(gt)
    torch.testing.assert_close(sample, expected)
