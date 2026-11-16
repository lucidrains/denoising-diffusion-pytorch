import copy

import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch.guided_diffusion import GaussianDiffusion, Unet


def make_diffusion(steps):
    torch.manual_seed(89)
    model = Unet(dim=8, dim_mults=(1,), channels=1)
    return GaussianDiffusion(
        model, image_size=4, timesteps=steps, beta_schedule="cosine"
    ).eval()


def classifier_guidance(classifier, calls):
    def condition(images, times, scale=1.0):
        calls.append(times.clone())
        with torch.enable_grad():
            inputs = images.detach().requires_grad_(True)
            selected_log_prob = classifier(inputs).log_softmax(-1)[:, 1]
            return torch.autograd.grad(selected_log_prob.sum(), inputs)[0] * scale

    return condition


@pytest.mark.parametrize("steps", [2, 4])
@pytest.mark.parametrize("arguments", ["omitted", "none", "empty", "nonempty"])
def test_ddpm_uses_classifier_closures_without_requiring_extra_kwargs(steps, arguments):
    actual = make_diffusion(steps)
    reference = copy.deepcopy(actual)
    torch.manual_seed(97)
    classifier = nn.Sequential(nn.Flatten(), nn.Linear(16, 2))
    actual_calls, reference_calls = [], []
    actual_condition = classifier_guidance(classifier, actual_calls)
    reference_condition = classifier_guidance(classifier, reference_calls)
    mapping = {"scale": 0.75} if arguments == "nonempty" else {}
    optional = (
        {}
        if arguments == "omitted"
        else {"guidance_kwargs": None if arguments == "none" else mapping}
    )
    torch.manual_seed(101)
    result = actual.sample(
        batch_size=2, return_all_timesteps=True, cond_fn=actual_condition, **optional
    )
    torch.manual_seed(101)
    expected = reference.sample(
        batch_size=2,
        return_all_timesteps=True,
        cond_fn=reference_condition,
        guidance_kwargs=mapping,
    )
    assert len(actual_calls) == len(reference_calls) == steps
    for observed, wanted in zip(actual_calls, reference_calls):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert mapping == ({"scale": 0.75} if arguments == "nonempty" else {})


@pytest.mark.parametrize("steps", [2, 4])
def test_sampling_without_a_guidance_callback_is_unchanged(steps):
    diffusion = make_diffusion(steps)
    torch.manual_seed(103)
    expected = diffusion.sample(batch_size=2)
    torch.manual_seed(103)
    actual = diffusion.sample(
        batch_size=2, cond_fn=None, guidance_kwargs={"unused": True}
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
