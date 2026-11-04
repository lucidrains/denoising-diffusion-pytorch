import pytest
import torch

from denoising_diffusion_pytorch.guided_diffusion import GaussianDiffusion, Unet


def make_diffusion(sampling_timesteps, objective, self_condition):
    torch.manual_seed(0)
    model = Unet(dim=8, dim_mults=(1, 2), channels=1, self_condition=self_condition)
    return GaussianDiffusion(
        model,
        image_size=8,
        timesteps=6,
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        ddim_sampling_eta=0.5,
    ).eval()


@pytest.mark.parametrize("sampling_timesteps", [1, 3, 6])
@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
@pytest.mark.parametrize("self_condition", [False, True])
def test_history_ends_with_the_sample(sampling_timesteps, objective, self_condition):
    diffusion = make_diffusion(sampling_timesteps, objective, self_condition)
    torch.manual_seed(17)
    sample = diffusion.sample(batch_size=2)
    torch.manual_seed(17)
    history = diffusion.sample(batch_size=2, return_all_timesteps=True)
    assert history.shape == (2, sampling_timesteps + 1, 1, 8, 8)
    torch.testing.assert_close(history[:, -1], sample, rtol=0, atol=0)
    torch.manual_seed(17)
    initial_noise = torch.randn(2, 1, 8, 8)
    torch.testing.assert_close(
        history[:, 0], diffusion.unnormalize(initial_noise), rtol=0, atol=0
    )
    assert not torch.equal(history[:, 0], history[:, 1])
