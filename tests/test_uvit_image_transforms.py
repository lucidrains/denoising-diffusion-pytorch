import copy

import pytest
import torch

from denoising_diffusion_pytorch.simple_diffusion import GaussianDiffusion, UViT


def transform_pair(channels):
    if channels == 3:
        return lambda x: x[:, [2, 0, 1]], lambda x: x[:, [1, 2, 0]]
    return lambda x: x.flip(-1), lambda x: x.flip(-1)


def make_model(channels, patch_size=1, dual_patchnorm=False, **kwargs):
    return UViT(
        dim=8, dim_mults=(1,), channels=channels, vit_depth=1, vit_dropout=0.,
        attn_heads=1, attn_dim_head=8, patch_size=patch_size,
        dual_patchnorm=dual_patchnorm, **kwargs,
    )


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("patch_size,dual_patchnorm", [(1, False), (2, False), (2, True)])
def test_paired_transforms_match_explicit_composition(channels, patch_size, dual_patchnorm):
    torch.manual_seed(19)
    forward, inverse = transform_pair(channels)
    model = make_model(channels, patch_size, dual_patchnorm,
                       init_img_transform=forward, final_img_itransform=inverse)
    reference = make_model(channels, patch_size, dual_patchnorm)
    reference.load_state_dict(copy.deepcopy(model.state_dict()))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    times = torch.tensor([-1., 1.])
    original = next(model.parameters()).detach().clone()

    for _ in range(2):
        image = torch.randn(2, channels, 8, 8, requires_grad=True)
        reference_image = image.detach().clone().requires_grad_()
        target = torch.randn_like(image)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual = model(image, times)
        expected = inverse(reference(forward(reference_image), times))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (actual - target).square().mean().backward()
        (expected - target).square().mean().backward()
        torch.testing.assert_close(image.grad, reference_image.grad, rtol=0, atol=0)
        for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
    assert not torch.equal(next(model.parameters()), original)
    for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected_parameter, rtol=0, atol=0)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("patch_size", [1, 2])
def test_paired_transforms_support_native_diffusion_training_and_sampling(channels, patch_size):
    torch.manual_seed(31)
    forward, inverse = transform_pair(channels)
    model = make_model(channels, patch_size, init_img_transform=forward, final_img_itransform=inverse)
    diffusion = GaussianDiffusion(model, image_size=8, channels=channels, num_sample_steps=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion(torch.rand(2, channels, 8, 8))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    samples = diffusion.sample(batch_size=2)
    assert samples.shape == (2, channels, 8, 8)
    assert torch.isfinite(samples).all()
    assert ((samples >= 0) & (samples <= 1)).all()


def test_noninvertible_shape_is_still_rejected():
    with pytest.raises(AssertionError):
        make_model(3, init_img_transform=lambda x: x, final_img_itransform=lambda x: x[..., :-1])


@pytest.mark.parametrize("hook", [None, "init_img_transform", "final_img_itransform"])
def test_missing_transform_preserves_existing_path(hook):
    def transform(x):
        return x.flip(-1)

    kwargs = {hook: transform} if hook is not None else {}
    torch.manual_seed(5)
    model = make_model(3, **kwargs)
    torch.manual_seed(5)
    reference = make_model(3)
    image = torch.randn(2, 3, 8, 8)
    times = torch.tensor([0., 1.])
    expected = reference(transform(image) if hook == "init_img_transform" else image, times)
    if hook == "final_img_itransform":
        expected = transform(expected)
    torch.testing.assert_close(model(image, times), expected, rtol=0, atol=0)
