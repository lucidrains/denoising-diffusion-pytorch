from copy import deepcopy

import pytest
import torch

from denoising_diffusion_pytorch.karras_unet import KarrasUnet
from denoising_diffusion_pytorch.karras_unet_1d import KarrasUnet1D
from denoising_diffusion_pytorch.karras_unet_3d import KarrasUnet3D


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("label_dtype", [torch.int32, torch.int64])
def test_integer_class_labels_match_long_labels(dimension, label_dtype):
    torch.manual_seed(0)
    kwargs = dict(
        dim=4,
        dim_max=8,
        channels=1,
        num_classes=3,
        num_downsamples=1,
        num_blocks_per_stage=1,
        attn_res=(),
        dropout=0.0,
    )
    if dimension == 1:
        model = KarrasUnet1D(seq_len=4, **kwargs)
        shape = (2, 1, 4)
    elif dimension == 2:
        model = KarrasUnet(image_size=4, **kwargs)
        shape = (2, 1, 4, 4)
    else:
        model = KarrasUnet3D(image_size=4, frames=2, **kwargs)
        shape = (2, 1, 2, 4, 4)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".gain"):
                parameter.fill_(1.0)
    reference = deepcopy(model)
    inputs = torch.randn(shape)
    times = torch.tensor([0.2, 0.8])
    labels = torch.tensor([0, 2], dtype=label_dtype)

    actual = model(inputs, times, class_labels=labels)
    expected = reference(inputs, times, class_labels=labels.long())
    actual.square().mean().backward()
    expected.square().mean().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(model.to_class_emb.weight.grad) > 0
    for (name, parameter), (_, expected_parameter) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        if not parameter.requires_grad:
            continue
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(
            parameter.grad, expected_parameter.grad, rtol=0, atol=0
        )
