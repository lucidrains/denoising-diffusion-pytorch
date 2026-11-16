import importlib

import pytest
import torch


def make_model(name, **kwargs):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{name}")
    extra = {"num_classes": 2} if name == "classifier_free_guidance" else {}
    cls = module.Unet if extra else module.Unet1D
    return module, cls(dim=8, dim_mults=(1, 2), channels=1, **extra, **kwargs)


@pytest.mark.parametrize(
    "name", ["denoising_diffusion_pytorch_1d", "classifier_free_guidance"]
)
@pytest.mark.parametrize("heads,dim_head", [(1, 8), (2, 16), (4, 32)])
def test_linear_stages_use_the_requested_attention_width_during_training(
    name, heads, dim_head
):
    torch.manual_seed(73)
    module, model = make_model(name, attn_heads=heads, attn_dim_head=dim_head)
    linear_layers = [
        layer for layer in model.modules() if isinstance(layer, module.LinearAttention)
    ]
    projected_widths = []
    handles = [
        layer.to_qkv.register_forward_hook(
            lambda layer, inputs, output: projected_widths.append(output.shape[1])
        )
        for layer in linear_layers
    ]
    sequence = name.endswith("_1d")
    if sequence:
        diffusion = module.GaussianDiffusion1D(model, seq_length=8, timesteps=4)
        inputs = torch.linspace(0.1, 0.9, 16).reshape(2, 1, 8)
        kwargs = {}
    else:
        diffusion = module.GaussianDiffusion(
            model, image_size=8, timesteps=4, beta_schedule="cosine"
        )
        inputs = torch.linspace(0.1, 0.9, 128).reshape(2, 1, 8, 8)
        kwargs = {
            "classes": torch.tensor([0, 1]),
            "cond_keep_mask": torch.ones(2, dtype=torch.bool),
        }
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3)
    before = [layer.to_qkv.weight.detach().clone() for layer in linear_layers]
    loss = diffusion(inputs, times=torch.tensor([1, 2]), **kwargs)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    for old, layer, handle in zip(before, linear_layers, handles):
        handle.remove()
        assert not torch.equal(old, layer.to_qkv.weight)
        assert torch.isfinite(layer.to_qkv.weight).all()
    assert len(linear_layers) == 4
    assert projected_widths == [3 * heads * dim_head] * len(linear_layers)
    assert all(layer.heads == heads for layer in linear_layers)


@pytest.mark.parametrize(
    "name", ["denoising_diffusion_pytorch_1d", "classifier_free_guidance"]
)
def test_default_architecture_and_outputs_are_unchanged(name):
    torch.manual_seed(79)
    _, default = make_model(name)
    torch.manual_seed(79)
    _, explicit = make_model(name, attn_heads=4, attn_dim_head=32)
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    inputs = torch.randn((2, 1, 8) if name.endswith("_1d") else (2, 1, 8, 8))
    kwargs = (
        {}
        if name.endswith("_1d")
        else {
            "classes": torch.tensor([0, 1]),
            "cond_keep_mask": torch.ones(2, dtype=torch.bool),
        }
    )
    times = torch.tensor([1, 2])
    torch.testing.assert_close(
        default(inputs, times, **kwargs),
        explicit(inputs, times, **kwargs),
        rtol=0,
        atol=0,
    )
