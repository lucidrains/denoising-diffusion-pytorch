import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch import GaussianDiffusion1D, XMWrapper


class FeatureConditionedModel(nn.Module):
    channels = 1
    self_condition = False

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.4))

    def forward(self, x, times, *, feature_scale, condition):
        return x * feature_scale + self.weight * condition[:, None, None]


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 6])
def test_xm_preserves_static_condition_tensors_when_chunking(chunk_size):
    torch.manual_seed(0)
    model = FeatureConditionedModel()
    diffusion = GaussianDiffusion1D(
        model, seq_length=6, timesteps=10, auto_normalize=False
    )
    wrapper = XMWrapper(diffusion, candidates=3, max_batch_size=chunk_size)
    images = torch.randn(2, 1, 6)
    times = torch.tensor([2, 5])
    noise = torch.randn_like(images)
    feature_scale = torch.linspace(0.1, 0.6, 6, requires_grad=True)
    conditions = torch.tensor([0.2, 0.9])
    model_kwargs = {"feature_scale": feature_scale, "condition": conditions}

    expected = diffusion(
        images, times=times, noise=noise, model_forward_kwargs=model_kwargs
    )
    expected_grads = torch.autograd.grad(expected, (model.weight, feature_scale))
    actual = wrapper(
        images, times=times, noise=noise, model_forward_kwargs=model_kwargs
    )
    actual_grads = torch.autograd.grad(actual, (model.weight, feature_scale))

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert model_kwargs["feature_scale"] is feature_scale
    assert model_kwargs["condition"] is conditions
