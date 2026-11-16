import copy
import importlib

import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch import classifier_free_guidance as cfg
from denoising_diffusion_pytorch.xm import XMWrapper

MODULES = [
    "denoising_diffusion_pytorch",
    "denoising_diffusion_pytorch_1d",
    "continuous_time_gaussian_diffusion",
    "v_param_continuous_time_gaussian_diffusion",
]


class Denoiser(nn.Module):
    channels = out_dim = 1
    self_condition = False

    def __init__(self, sequence, continuous):
        super().__init__()
        self.random_or_learned_sinusoidal_cond = continuous
        self.conv = nn.Conv1d(1, 1, 1) if sequence else nn.Conv2d(1, 1, 1)
        self.seen_times = []

    def forward(self, x, times, *args, **kwargs):
        self.seen_times.append(times.detach().clone())
        return self.conv(x)


def compare_loss_and_gradients(
    actual, expected, images, actual_kwargs, expected_kwargs, chunk
):
    torch.manual_seed(7)
    result = XMWrapper(actual, candidates=3, max_batch_size=chunk)(
        images, **actual_kwargs
    )
    result.backward()
    torch.manual_seed(7)
    reference = XMWrapper(expected, candidates=3, max_batch_size=chunk)(
        images, **expected_kwargs
    )
    reference.backward()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for actual_parameter, expected_parameter in zip(
        actual.parameters(), expected.parameters()
    ):
        if expected_parameter.grad is None:
            assert actual_parameter.grad is None
        else:
            torch.testing.assert_close(
                actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0
            )


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize("chunk", [2, 6])
@pytest.mark.parametrize("provided", [False, True])
def test_none_times_matches_omitted_shared_times(module_name, chunk, provided):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{module_name}")
    sequence = module_name.endswith("_1d")
    continuous = "continuous_time" in module_name
    torch.manual_seed(31)
    model = Denoiser(sequence, continuous)
    if sequence:
        diffusion = module.GaussianDiffusion1D(model, seq_length=4, timesteps=8)
        images = torch.linspace(0.1, 0.9, 8).reshape(2, 1, 4)
    elif continuous:
        diffusion_type = (
            module.VParamContinuousTimeGaussianDiffusion
            if module_name.startswith("v_param")
            else module.ContinuousTimeGaussianDiffusion
        )
        diffusion = diffusion_type(model, image_size=4, channels=1)
        images = torch.linspace(0.1, 0.9, 32).reshape(2, 1, 4, 4)
    else:
        diffusion = module.GaussianDiffusion(
            model, image_size=4, timesteps=8, beta_schedule="cosine"
        )
        images = torch.linspace(0.1, 0.9, 32).reshape(2, 1, 4, 4)
    reference = copy.deepcopy(diffusion)
    times = torch.tensor([0.2, 0.7]) if continuous else torch.tensor([1, 5])
    kwargs = {"times": times if provided else None}
    reference_kwargs = {"times": times} if provided else {}
    compare_loss_and_gradients(
        diffusion, reference, images, kwargs, reference_kwargs, chunk
    )
    observed = torch.cat(model.seen_times).reshape(2, 3)
    torch.testing.assert_close(
        observed, observed[:, :1].expand_as(observed), rtol=0, atol=0
    )
    if not provided:
        assert kwargs["times"] is None
    else:
        assert kwargs["times"] is times


@pytest.mark.parametrize("chunk", [2, 6])
@pytest.mark.parametrize("setting", ["times", "cond_keep_mask", "both", "provided"])
def test_none_cfg_shared_inputs_match_omitted_defaults(chunk, setting):
    torch.manual_seed(31)
    model = cfg.Unet(
        dim=8, dim_mults=(1,), channels=1, num_classes=2, cond_drop_prob=0.5
    )
    diffusion = cfg.GaussianDiffusion(
        model, image_size=4, timesteps=8, beta_schedule="cosine"
    )
    reference = copy.deepcopy(diffusion)
    images = torch.linspace(0.1, 0.9, 32).reshape(2, 1, 4, 4)
    classes = torch.tensor([0, 1])
    kwargs = {"classes": classes}
    reference_kwargs = {"classes": classes}
    if setting == "provided":
        kwargs.update(
            times=torch.tensor([1, 5]), cond_keep_mask=torch.tensor([True, False])
        )
        reference_kwargs.update(kwargs)
    else:
        if setting in ("times", "both"):
            kwargs["times"] = None
        if setting in ("cond_keep_mask", "both"):
            kwargs["cond_keep_mask"] = None
    observed = []

    def record_inputs(module, args, keyword_args):
        mask = keyword_args.get("cond_keep_mask")
        observed.append((args[1].clone(), None if mask is None else mask.clone()))

    handle = model.register_forward_pre_hook(record_inputs, with_kwargs=True)
    compare_loss_and_gradients(
        diffusion, reference, images, kwargs, reference_kwargs, chunk
    )
    handle.remove()
    times = torch.cat([row[0] for row in observed]).reshape(2, 3)
    torch.testing.assert_close(times, times[:, :1].expand_as(times), rtol=0, atol=0)
    assert all(row[1] is not None for row in observed)
    masks = torch.cat([row[1] for row in observed]).reshape(2, 3)
    assert torch.equal(masks, masks[:, :1].expand_as(masks))
    if setting == "provided":
        assert torch.equal(times[:, 0], kwargs["times"])
        assert torch.equal(masks[:, 0], kwargs["cond_keep_mask"])
