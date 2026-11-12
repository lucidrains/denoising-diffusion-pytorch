import random

import pytest
import torch

from denoising_diffusion_pytorch import (
    ContinuousTimeGaussianDiffusion,
    ElucidatedDiffusion,
    KarrasUnet,
    Unet,
    VParamContinuousTimeGaussianDiffusion,
)
from denoising_diffusion_pytorch.karras_unet import Gain


WRAPPERS = [ElucidatedDiffusion, ContinuousTimeGaussianDiffusion, VParamContinuousTimeGaussianDiffusion]


def make_model(channels, self_condition=False):
    model = KarrasUnet(
        image_size=4, channels=channels, dim=8, dim_max=16,
        num_downsamples=1, num_blocks_per_stage=1, attn_dim_head=4,
        attn_res=(2, 4), dropout=0., self_condition=self_condition,
    )
    with torch.no_grad():
        for child in model.modules():
            if isinstance(child, Gain):
                child.gain.fill_(0.5)
    return model


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("channels", [1, 3])
def test_karras_model_trains_and_samples_with_continuous_wrappers(wrapper, channels):
    torch.manual_seed(26)
    model = make_model(channels)
    diffusion = wrapper(model, image_size=4, channels=channels, num_sample_steps=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    original = model.input_block.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion(torch.rand(2, channels, 4, 4))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.input_block.weight.grad.abs().sum() > 0
        assert model.to_time_emb[1].weight.grad.abs().sum() > 0
        assert model.to_time_emb[0].weights.grad is None
        optimizer.step()
    assert not torch.equal(model.input_block.weight, original)
    model.eval()
    samplers = [diffusion.sample]
    if wrapper is ElucidatedDiffusion:
        samplers.append(diffusion.sample_using_dpmpp)
    for sampler in samplers:
        samples = sampler(batch_size=2)
        assert samples.shape == (2, channels, 4, 4)
        assert torch.isfinite(samples).all()
        assert ((samples >= 0) & (samples <= 1)).all()


@pytest.mark.parametrize("channels", [1, 3])
def test_elucidated_wrapper_preserves_karras_self_conditioning(channels):
    torch.manual_seed(8)
    random.seed(1)
    model = make_model(channels, self_condition=True)
    diffusion = ElucidatedDiffusion(model, image_size=4, channels=channels, num_sample_steps=2)
    conditions = []
    model.register_forward_pre_hook(lambda module, args: conditions.append(args[2]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion(torch.rand(2, channels, 4, 4), sigma=torch.tensor([0.2, 0.8]))
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(model.input_block.weight.grad).all()
        optimizer.step()
    assert any(condition is not None for condition in conditions)
    model.eval()
    samples = diffusion.sample(batch_size=2)
    assert samples.shape == (2, channels, 4, 4)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("wrapper", [ContinuousTimeGaussianDiffusion, VParamContinuousTimeGaussianDiffusion])
def test_wrappers_without_self_conditioning_still_reject_it(wrapper):
    with pytest.raises(AssertionError, match="not supported yet"):
        wrapper(make_model(1, self_condition=True), image_size=4, channels=1, num_sample_steps=2)


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_wrappers_still_require_random_or_learned_fourier_features(wrapper):
    model = Unet(dim=8, dim_mults=(1,), channels=1, attn_heads=1, attn_dim_head=8)
    with pytest.raises(AssertionError):
        wrapper(model, image_size=4, channels=1, num_sample_steps=2)
