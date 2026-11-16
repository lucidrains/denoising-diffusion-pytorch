import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch import GaussianDiffusion


@pytest.mark.parametrize(
    "image_grad,noise_grad",
    [(True, False), (False, True), (True, True), (False, False)],
)
def test_immiscible_assignment_preserves_input_gradients(image_grad, noise_grad):
    class AffineModel(nn.Module):
        channels = out_dim = 1
        self_condition = False

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.4))

        def forward(self, x, times, self_condition=None):
            return self.scale * x + 0.1

    model = AffineModel()
    diffusion = GaussianDiffusion(
        model,
        image_size=2,
        timesteps=10,
        objective="pred_x0",
        immiscible=True,
        auto_normalize=False,
    )
    images = (
        torch.tensor([1.0, -1.0])
        .reshape(2, 1, 1, 1)
        .expand(2, 1, 2, 2)
        .clone()
        .requires_grad_(image_grad)
    )
    noise = (
        torch.tensor([-0.9, 0.9])
        .reshape(2, 1, 1, 1)
        .expand(2, 1, 2, 2)
        .clone()
        .requires_grad_(noise_grad)
    )
    times = torch.tensor([2, 5])
    alpha = diffusion.sqrt_alphas_cumprod[times, None, None, None]
    sigma = diffusion.sqrt_one_minus_alphas_cumprod[times, None, None, None]
    expected_noised = alpha * images + sigma * noise.flip(0)
    expected_predictions = model(expected_noised, times)
    expected = (
        (expected_predictions - images).square().mean(dim=(1, 2, 3))
        * diffusion.loss_weight[times]
    ).mean()

    actual = diffusion.p_losses(images, times, noise=noise)
    variables = (
        [model.scale]
        + ([images] if image_grad else [])
        + ([noise] if noise_grad else [])
    )
    expected_grads = torch.autograd.grad(expected, variables)
    actual_grads = torch.autograd.grad(actual, variables)

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        assert torch.count_nonzero(actual_grad) > 0
        torch.testing.assert_close(actual_grad, expected_grad)
