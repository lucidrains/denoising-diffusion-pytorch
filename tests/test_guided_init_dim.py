import io
import random

import pytest
import torch

from denoising_diffusion_pytorch.guided_diffusion import GaussianDiffusion, Unet


@pytest.mark.parametrize("init_dim", [4, 8, 16])
@pytest.mark.parametrize("self_condition", [False, True])
@pytest.mark.parametrize("dim_mults", [(1,), (1, 2)])
def test_custom_width_supports_native_training(init_dim, self_condition, dim_mults):
    torch.manual_seed(91)
    random.seed(1)
    model = Unet(dim=8, init_dim=init_dim, dim_mults=dim_mults, channels=1, self_condition=self_condition)
    diffusion = GaussianDiffusion(model, image_size=8, timesteps=8)
    image = torch.rand(2, 1, 8, 8)
    original = model.init_conv.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = []
    model.final_res_block.register_forward_pre_hook(lambda module, args: inputs.append(args[0].shape[1]))

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion(image)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.init_conv.weight.grad.abs().sum() > 0
        assert model.final_conv.weight.grad.abs().sum() > 0
        optimizer.step()

    assert inputs and all(width == 2 * init_dim for width in inputs)
    assert not torch.equal(model.init_conv.weight, original)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = Unet(dim=8, init_dim=init_dim, dim_mults=dim_mults, channels=1, self_condition=self_condition)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    times = torch.tensor([1, 5])
    condition = torch.randn_like(image) if self_condition else None
    torch.testing.assert_close(restored(image, times, condition), model(image, times, condition), rtol=0, atol=0)


@pytest.mark.parametrize("init_dim", [4, 16])
@pytest.mark.parametrize("sampling_timesteps", [2, 4])
def test_custom_width_supports_native_sampling(init_dim, sampling_timesteps):
    torch.manual_seed(12)
    model = Unet(dim=8, init_dim=init_dim, dim_mults=(1, 2), channels=3, self_condition=True)
    diffusion = GaussianDiffusion(model, image_size=8, timesteps=4, sampling_timesteps=sampling_timesteps)
    samples = diffusion.sample(batch_size=2)
    assert samples.shape == (2, 3, 8, 8)
    assert torch.isfinite(samples).all()
    assert ((samples >= 0) & (samples <= 1)).all()


@pytest.mark.parametrize("self_condition", [False, True])
def test_default_width_preserves_checkpoint_and_outputs(self_condition):
    torch.manual_seed(21)
    default = Unet(dim=8, dim_mults=(1, 2), channels=1, self_condition=self_condition)
    torch.manual_seed(21)
    explicit = Unet(dim=8, init_dim=8, dim_mults=(1, 2), channels=1, self_condition=self_condition)
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    image = torch.randn(2, 1, 8, 8)
    condition = torch.randn_like(image) if self_condition else None
    times = torch.tensor([0, 3])
    torch.testing.assert_close(default(image, times, condition), explicit(image, times, condition), rtol=0, atol=0)
