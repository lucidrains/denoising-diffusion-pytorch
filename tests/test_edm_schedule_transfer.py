import copy

import pytest
import torch

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.elucidated_diffusion import ElucidatedDiffusion


@pytest.mark.parametrize("steps", [2, 4, 8])
@pytest.mark.parametrize("churn", ["none", "all", "partial"])
@pytest.mark.parametrize("self_condition", [False, True])
def test_heun_sampling_avoids_per_step_scalar_transfers(
    monkeypatch, steps, churn, self_condition
):
    torch.manual_seed(53)
    model = Unet(
        dim=8,
        dim_mults=(1,),
        channels=1,
        self_condition=self_condition,
        random_fourier_features=True,
        attn_heads=1,
        attn_dim_head=4,
    )
    diffusion = ElucidatedDiffusion(
        model,
        image_size=4,
        channels=1,
        sigma_min=0.1,
        sigma_max=2,
        num_sample_steps=steps,
        S_churn=0 if churn == "none" else 1,
        S_tmin=0.3 if churn == "partial" else 0,
        S_tmax=1 if churn == "partial" else 3,
    )
    reference = copy.deepcopy(diffusion)
    item_calls = []
    original_item = torch.Tensor.item
    original_tolist = torch.Tensor.tolist

    def counted_item(tensor, *args, **kwargs):
        item_calls.append(tensor.shape)
        return original_item(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", counted_item)
        torch.manual_seed(59)
        actual = diffusion.sample(batch_size=2, clamp=False)
        actual_rng = torch.get_rng_state()

    # Recreate the original scalar extraction only; use the same native model
    # and complete Heun sampler for the numerical and RNG comparison.
    def scalar_tolist(tensor):
        if tensor.ndim == 2 and tensor.shape[1] == 3:
            return [[original_item(value) for value in row] for row in tensor]
        return original_tolist(tensor)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", scalar_tolist)
        torch.manual_seed(59)
        expected = reference.sample(batch_size=2, clamp=False)
        expected_rng = torch.get_rng_state()

    assert torch.isfinite(actual).all()
    assert actual.shape == (2, 1, 4, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_rng, expected_rng)
    assert item_calls == [], (
        f"sampling extracted {len(item_calls)} scalars individually"
    )
