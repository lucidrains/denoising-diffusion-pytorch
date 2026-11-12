from copy import deepcopy
import importlib

import pytest
import torch


def make_diffusion(variant, auto_normalize=True):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{variant}")
    torch.manual_seed(16)
    model_kwargs = dict(dim=8, dim_mults=(1, 2), channels=1)
    if variant != "guided_diffusion":
        model_kwargs.update(attn_heads=1, attn_dim_head=8)
    if variant == "classifier_free_guidance":
        model = module.Unet(**model_kwargs, num_classes=3, cond_drop_prob=0)
        diffusion = module.GaussianDiffusion(model, image_size=8, timesteps=4, beta_schedule="cosine")
    elif variant == "denoising_diffusion_pytorch_1d":
        model = module.Unet1D(**model_kwargs, self_condition=True)
        diffusion = module.GaussianDiffusion1D(
            model, seq_length=8, timesteps=4, beta_schedule="cosine", auto_normalize=auto_normalize
        )
    else:
        model = module.Unet(**model_kwargs, self_condition=True)
        diffusion = module.GaussianDiffusion(
            model, image_size=8, timesteps=4, beta_schedule="cosine", auto_normalize=auto_normalize
        )
    return module, diffusion.eval()


@pytest.mark.parametrize("variant", [
    "denoising_diffusion_pytorch", "denoising_diffusion_pytorch_1d", "repaint", "guided_diffusion"
])
@pytest.mark.parametrize("auto_normalize", [False, True])
@pytest.mark.parametrize("time", [0, 2])
def test_interpolation_matches_manual_model_space_conversion(variant, auto_normalize, time):
    _, diffusion = make_diffusion(variant, auto_normalize)
    reference = deepcopy(diffusion)
    reference.normalize = lambda value: value
    reference.unnormalize = lambda value: value
    shape = (2, 1, 8) if variant.endswith("_1d") else (2, 1, 8, 8)
    first = torch.rand(shape)
    second = torch.rand(shape)
    originals = (first.clone(), second.clone())
    model_inputs = tuple(value * 2 - 1 if auto_normalize else value for value in (first, second))

    torch.manual_seed(42)
    expected = reference.interpolate(*model_inputs, t=time, lam=0.3)
    if auto_normalize:
        expected = (expected + 1) * 0.5
    expected_random_state = torch.random.get_rng_state()
    torch.manual_seed(42)
    actual = diffusion.interpolate(first, second, t=time, lam=0.3)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(torch.random.get_rng_state(), expected_random_state)
    for value, original in zip((first, second), originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("time", [0, 2])
def test_class_conditioned_interpolation_uses_the_image_convention(time, monkeypatch):
    module, diffusion = make_diffusion("classifier_free_guidance")
    reference = deepcopy(diffusion)
    first, second = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    originals = (first.clone(), second.clone())
    classes = torch.tensor([0, 2])

    # The classifier-conditioned variant always normalizes, so disable only its boundary
    # transforms for the independent model-space reference. Native denoising still runs.
    with monkeypatch.context() as context:
        context.setattr(module, "normalize_to_neg_one_to_one", lambda value: value)
        context.setattr(module, "unnormalize_to_zero_to_one", lambda value: value)
        torch.manual_seed(42)
        expected = (reference.interpolate(first * 2 - 1, second * 2 - 1, classes, t=time, lam=0.7) + 1) * 0.5
        expected_random_state = torch.random.get_rng_state()
    torch.manual_seed(42)
    actual = diffusion.interpolate(first, second, classes, t=time, lam=0.7)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(torch.random.get_rng_state(), expected_random_state)
    for value, original in zip((first, second), originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
