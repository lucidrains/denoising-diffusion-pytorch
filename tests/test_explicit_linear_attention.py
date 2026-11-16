import importlib

import pytest
import torch


@pytest.mark.parametrize("module_name", ["denoising_diffusion_pytorch", "repaint"])
@pytest.mark.parametrize("dim_mults", [(1,), (1, 2)])
@pytest.mark.parametrize("full_attention", [False, True, None])
def test_scalar_attention_setting_matches_explicit_stages(
    module_name, dim_mults, full_attention
):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{module_name}")
    count = len(dim_mults)
    explicit = (
        (False,) * (count - 1) + (True,)
        if full_attention is None
        else (full_attention,) * count
    )
    options = dict(
        dim=8, channels=1, dim_mults=dim_mults, attn_heads=1, attn_dim_head=4
    )
    torch.manual_seed(31)
    model = module.Unet(full_attn=full_attention, **options)
    torch.manual_seed(31)
    reference = module.Unet(full_attn=explicit, **options)

    for stage, expected_full in zip(model.downs, explicit):
        expected_type = module.Attention if expected_full else module.LinearAttention
        assert isinstance(stage[2], expected_type)
    for stage, expected_full in zip(model.ups, reversed(explicit)):
        expected_type = module.Attention if expected_full else module.LinearAttention
        assert isinstance(stage[2], expected_type)

    inputs = torch.randn(2, 1, 4, 4)
    times = torch.tensor([1, 3])
    actual = model(inputs, times)
    expected = reference(inputs, times)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for parameter, expected_parameter in zip(
        model.parameters(), reference.parameters()
    ):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(
            parameter.grad, expected_parameter.grad, rtol=0, atol=0
        )
