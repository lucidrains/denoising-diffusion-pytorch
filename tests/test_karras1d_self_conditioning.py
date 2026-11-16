import copy

import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch import (
    denoising_diffusion_pytorch_1d as diffusion_module,
)
from denoising_diffusion_pytorch.karras_unet_1d import KarrasUnet1D


def make_model(self_condition):
    torch.manual_seed(13)
    model = KarrasUnet1D(
        seq_len=8,
        channels=1,
        dim=8,
        dim_max=16,
        num_downsamples=1,
        num_blocks_per_stage=1,
        attn_res=(),
        dropout=0,
        self_condition=self_condition,
    )
    with torch.no_grad():
        model.output_block[-1].gain.fill_(1)
    return model


def assert_gradients_match(actual, expected):
    for parameter, reference in zip(actual.parameters(), expected.parameters()):
        if reference.grad is None:
            assert parameter.grad is None
        else:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)


class LegacyKeywordAdapter(nn.Module):
    channels = out_dim = 1

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.self_condition = model.self_condition

    def forward(self, x, time, x_self_cond=None):
        return self.model(x, time, self_cond=x_self_cond)


@pytest.mark.parametrize("enabled", [False, True])
def test_native_training_matches_legacy_keyword_adapter(monkeypatch, enabled):
    actual_model = make_model(enabled)
    expected_model = copy.deepcopy(actual_model)
    actual = diffusion_module.GaussianDiffusion1D(
        actual_model, seq_length=8, timesteps=4
    )
    expected = diffusion_module.GaussianDiffusion1D(
        LegacyKeywordAdapter(expected_model), seq_length=8, timesteps=4
    )
    images = torch.linspace(0.1, 0.9, 16).reshape(2, 1, 8)
    noise = torch.linspace(-1, 1, 16).reshape_as(images)
    times = torch.tensor([1, 3])
    monkeypatch.setattr(diffusion_module, "random", lambda: 0.0)
    loss = actual(images, times=times, noise=noise)
    reference_loss = expected(images, times=times, noise=noise)
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    loss.backward()
    reference_loss.backward()
    assert_gradients_match(actual_model, expected_model)
    assert actual_model.output_block[-1].gain.grad.abs() > 0


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("steps", [2, 4])
def test_native_ddpm_and_ddim_match_legacy_keyword_adapter(enabled, steps):
    actual_model = make_model(enabled).eval()
    expected_model = copy.deepcopy(actual_model)
    actual = diffusion_module.GaussianDiffusion1D(
        actual_model, seq_length=8, timesteps=4, sampling_timesteps=steps
    )
    expected = diffusion_module.GaussianDiffusion1D(
        LegacyKeywordAdapter(expected_model),
        seq_length=8,
        timesteps=4,
        sampling_timesteps=steps,
    )
    torch.manual_seed(17)
    result = actual.sample(batch_size=2)
    torch.manual_seed(17)
    reference = expected.sample(batch_size=2)
    assert result.shape == (2, 1, 8)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("positional", [False, True])
def test_self_condition_alias_preserves_legacy_forward_and_gradients(positional):
    model = make_model(True).eval()
    reference = copy.deepcopy(model)
    x = torch.linspace(-1, 1, 16).reshape(2, 1, 8)
    condition = torch.linspace(0.8, -0.3, 16).reshape_as(x).requires_grad_()
    reference_condition = condition.detach().clone().requires_grad_()
    times = torch.tensor([1, 3])
    actual = model(x, times, x_self_cond=condition)
    expected = (
        reference(x, times, reference_condition)
        if positional
        else reference(x, times, self_cond=reference_condition)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(condition.grad, reference_condition.grad, rtol=0, atol=0)
    assert condition.grad.abs().sum() > 0
    assert_gradients_match(model, reference)


def test_rejects_ambiguous_self_condition_aliases():
    model = make_model(True)
    x = torch.randn(2, 1, 8)
    with pytest.raises(ValueError, match="self_cond.*x_self_cond"):
        model(x, torch.tensor([1, 3]), self_cond=x, x_self_cond=x)
