import copy

import pytest
import torch

from denoising_diffusion_pytorch.classifier_free_guidance import Unet


@pytest.mark.parametrize("mask_kind", ["none", "all_keep", "all_drop", "mixed"])
@pytest.mark.parametrize("scale", [0.0, 2.0])
@pytest.mark.parametrize("rescaled_phi", [0.0, 0.5])
def test_null_guidance_branch_ignores_conditional_keep_mask(
    mask_kind, scale, rescaled_phi
):
    torch.manual_seed(23)
    model = Unet(dim=8, dim_mults=(1,), channels=1, num_classes=3)
    reference = copy.deepcopy(model)
    inputs = torch.linspace(-0.8, 0.9, 32).reshape(2, 1, 4, 4)
    times = torch.tensor([1, 3])
    classes = torch.tensor([0, 2])
    masks = {
        "none": None,
        "all_keep": torch.tensor([True, True]),
        "all_drop": torch.tensor([False, False]),
        "mixed": torch.tensor([True, False]),
    }
    kwargs = {"cond_keep_mask": masks[mask_kind]}

    conditional = reference(inputs, times, classes, cond_drop_prob=0.0, **kwargs)
    unconditional = reference(inputs, times, classes, cond_drop_prob=1.0)
    expected = conditional + (conditional - unconditional) * (scale - 1.0)
    if rescaled_phi:
        axes = tuple(range(1, conditional.ndim))
        ratio = conditional.std(dim=axes, keepdim=True) / expected.std(
            dim=axes, keepdim=True
        )
        expected = expected * ratio * rescaled_phi + expected * (1 - rescaled_phi)

    guided, null_logits = model.forward_with_cond_scale(
        inputs,
        times,
        classes,
        cond_scale=scale,
        rescaled_phi=rescaled_phi,
        remove_parallel_component=False,
        **kwargs,
    )
    torch.testing.assert_close(null_logits, unconditional, rtol=0, atol=0)
    torch.testing.assert_close(guided, expected, rtol=0, atol=0)
    if mask_kind in ("all_keep", "mixed"):
        assert not torch.equal(conditional[0], unconditional[0])

    (guided.square().mean() + null_logits.square().mean()).backward()
    (expected.square().mean() + unconditional.square().mean()).backward()
    for parameter, expected_parameter in zip(
        model.parameters(), reference.parameters()
    ):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(
            parameter.grad, expected_parameter.grad, rtol=0, atol=0
        )
    assert kwargs["cond_keep_mask"] is masks[mask_kind]
