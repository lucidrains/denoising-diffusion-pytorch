import importlib

import pytest
import torch
from torch import nn

MODULES = [
    "denoising_diffusion_pytorch",
    "denoising_diffusion_pytorch_1d",
    "guided_diffusion",
    "repaint",
    "classifier_free_guidance",
]


class ConvolutionalDenoiser(nn.Module):
    channels = out_dim = 1
    self_condition = False
    random_or_learned_sinusoidal_cond = False

    def __init__(self, sequence):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1) if sequence else nn.Conv2d(1, 1, 1)
        with torch.no_grad():
            self.conv.weight.fill_(0.1)
            self.conv.bias.fill_(0.1)
        self.seen_inputs = []

    def forward(self, x, times, *args, **kwargs):
        self.seen_inputs.append(x.detach().clone())
        return self.conv(x)

    def forward_with_cond_scale(self, x, times, classes, **kwargs):
        output = self.forward(x, times)
        return output, output


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize("sampler", ["p_sample_loop", "ddim_sample"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_sampler_initial_noise_matches_diffusion_dtype(module_name, sampler, dtype):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{module_name}")
    sequence = module_name.endswith("_1d")
    model = ConvolutionalDenoiser(sequence)
    diffusion_type = (
        module.GaussianDiffusion1D if sequence else module.GaussianDiffusion
    )
    dimensions = {"seq_length": 4} if sequence else {"image_size": 4}
    diffusion = diffusion_type(
        model,
        timesteps=4,
        sampling_timesteps=2,
        beta_schedule="cosine",
        objective="pred_v",
        **dimensions,
    ).to(dtype)
    shape = (2, 1, 4) if sequence else (2, 1, 4, 4)
    generator = torch.Generator().manual_seed(19)
    expected_noise = torch.randn(shape, dtype=dtype, generator=generator)
    torch.manual_seed(19)
    if module_name == "classifier_free_guidance":
        output = getattr(diffusion, sampler)(
            torch.tensor([0, 1]), shape, cond_scale=2, rescaled_phi=0
        )
    else:
        output = getattr(diffusion, sampler)(shape)

    torch.testing.assert_close(model.seen_inputs[0], expected_noise, rtol=0, atol=0)
    expected_steps = 4 if sampler == "p_sample_loop" else 2
    assert len(model.seen_inputs) == expected_steps
    assert all(inputs.dtype == dtype for inputs in model.seen_inputs)
    assert output.shape == shape
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
