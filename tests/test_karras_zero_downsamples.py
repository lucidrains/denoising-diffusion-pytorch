import pytest
import torch

from denoising_diffusion_pytorch.karras_unet_3d import Gain, KarrasUnet3D


def make_model(downsamples, blocks, self_condition=False, downsample_types=None):
    model = KarrasUnet3D(
        frames=4, image_size=4, channels=1, dim=8, dim_max=16,
        num_downsamples=downsamples, num_blocks_per_stage=blocks,
        self_condition=self_condition, downsample_types=downsample_types,
        dropout=0., attn_dim_head=4, attn_res=(1, 2, 4),
    )
    with torch.no_grad():
        for child in model.modules():
            if isinstance(child, Gain):
                child.gain.fill_(0.5)
    return model


@pytest.mark.parametrize("downsamples", [0, 1, 2])
@pytest.mark.parametrize("blocks", [1, 2])
@pytest.mark.parametrize("self_condition", [False, True])
def test_scalar_block_count_matches_explicit_stages(downsamples, blocks, self_condition):
    torch.manual_seed(3)
    model = make_model(downsamples, blocks, self_condition)
    torch.manual_seed(3)
    reference = make_model(downsamples, (blocks,) * (downsamples + 1), self_condition)
    assert model.state_dict().keys() == reference.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    original = model.input_block.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        value = torch.randn(2, 1, 4, 4, 4, requires_grad=True)
        reference_value = value.detach().clone().requires_grad_()
        times = torch.rand(2)
        condition = torch.randn_like(value) if self_condition else None
        target = torch.randn_like(value)
        actual = model(value, times, self_cond=condition)
        expected = reference(reference_value, times, self_cond=condition)
        assert actual.shape == value.shape
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (actual - target).square().mean().backward()
        (expected - target).square().mean().backward()
        torch.testing.assert_close(value.grad, reference_value.grad, rtol=0, atol=0)
        for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
    assert not torch.equal(model.input_block.weight, original)
    for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected_parameter, rtol=0, atol=0)


@pytest.mark.parametrize("downsample_types", [("all", "all"), ("image", "frame"), ("frame", "image")])
def test_short_stage_tuple_retains_legacy_expansion(downsample_types):
    torch.manual_seed(14)
    legacy = make_model(2, (2, 1), downsample_types=downsample_types)
    torch.manual_seed(14)
    explicit = make_model(2, (2, 2, 1), downsample_types=downsample_types)
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    values = torch.randn(2, 1, 4, 4, 4)
    times = torch.rand(2)
    with torch.no_grad():
        torch.testing.assert_close(legacy(values, times), explicit(values, times), rtol=0, atol=0)


@pytest.mark.parametrize("blocks", [(1, 0), (1, 2, 3), 0, -1])
def test_invalid_block_counts_remain_rejected(blocks):
    with pytest.raises(AssertionError):
        make_model(1, blocks)
