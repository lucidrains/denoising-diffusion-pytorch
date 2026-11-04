import pytest
import torch
import torch.nn.functional as F

from denoising_diffusion_pytorch.karras_unet import KarrasUnet
from denoising_diffusion_pytorch.karras_unet_1d import KarrasUnet1D
from denoising_diffusion_pytorch.karras_unet_3d import KarrasUnet3D


@pytest.mark.parametrize(
    "model_type, shape",
    [
        (KarrasUnet, (2, 1, 8, 8)),
        (KarrasUnet1D, (2, 1, 8)),
        (KarrasUnet3D, (2, 1, 4, 8, 8)),
    ],
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_class_conditioning_preserves_model_dtype(model_type, shape, dtype):
    torch.manual_seed(9)
    kwargs = dict(
        dim=8,
        dim_max=16,
        channels=1,
        num_classes=3,
        num_downsamples=1,
        num_blocks_per_stage=1,
        attn_res=(),
        dropout=0,
    )
    if model_type is KarrasUnet1D:
        kwargs["seq_len"] = 8
    else:
        kwargs["image_size"] = 8
    if model_type is KarrasUnet3D:
        kwargs["frames"] = 4
    model = model_type(**kwargs).to(dtype=dtype).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("gain"):
                parameter.fill_(0.5)
    images = torch.randn(shape, dtype=dtype, requires_grad=True)
    times = torch.tensor([0.2, 0.8], dtype=dtype)
    labels = torch.tensor([0, 2])
    output = model(images, times, class_labels=labels)
    one_hot_output = model(images, times, class_labels=F.one_hot(labels, 3).to(dtype))
    assert output.dtype == dtype
    torch.testing.assert_close(output, one_hot_output, rtol=0, atol=0)
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert torch.isfinite(images.grad).all()
    assert model.to_class_emb.weight.grad is not None
    assert torch.isfinite(model.to_class_emb.weight.grad).all()
