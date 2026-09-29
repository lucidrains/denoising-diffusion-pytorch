"""Batch and CFG++ regressions using a real, randomly initialized CPU Unet."""
import pytest
import torch

from denoising_diffusion_pytorch.classifier_free_guidance import GaussianDiffusion, Unet, project


@pytest.fixture
def model():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(178)
    yield Unet(dim=8, num_classes=3, dim_mults=(1, 2), channels=1).eval()
    torch.set_num_threads(threads)


@pytest.mark.parametrize('batch', [1, 2, 3, 4])
@pytest.mark.parametrize('objective', ['pred_noise', 'pred_x0', 'pred_v'])
@pytest.mark.parametrize('cfg_plus_plus', [False, True])
@pytest.mark.parametrize('scale', [1., 3.])
def test_prediction_batch_and_unconditional_branch(model, batch, objective, cfg_plus_plus, scale):
    diffusion = GaussianDiffusion(model, image_size=8, timesteps=8,
                                  objective=objective, use_cfg_plus_plus=cfg_plus_plus)
    x = torch.randn(batch, 1, 8, 8)
    t = torch.ones(batch, dtype=torch.long)
    classes = torch.arange(batch) % 3
    with torch.no_grad():
        conditional = model(x, t, classes, cond_drop_prob=0.)
        unconditional = model(x, t, classes, cond_drop_prob=1.)
        assert not torch.allclose(conditional, unconditional)
        _, orthogonal = project(conditional - unconditional, conditional)
        guided = conditional if scale == 1. else conditional + orthogonal * (scale - 1.)
        noise_output = unconditional if cfg_plus_plus else guided

        def to_start(output):
            if objective == 'pred_noise':
                return diffusion.predict_start_from_noise(x, t, output)
            if objective == 'pred_v':
                return diffusion.predict_start_from_v(x, t, output)
            return output

        expected_start = to_start(guided)
        expected_noise = noise_output if objective == 'pred_noise' else diffusion.predict_noise_from_start(x, t, to_start(noise_output))
        prediction = diffusion.model_predictions(x, t, classes, cond_scale=scale, rescaled_phi=0.)
        assert prediction.pred_noise.shape == x.shape
        assert prediction.pred_x_start.shape == x.shape
        torch.testing.assert_close(prediction.pred_noise, expected_noise)
        torch.testing.assert_close(prediction.pred_x_start, expected_start)
        permutation = torch.arange(batch - 1, -1, -1)
        permuted = diffusion.model_predictions(x[permutation], t[permutation], classes[permutation], cond_scale=scale, rescaled_phi=0.)
        torch.testing.assert_close(permuted.pred_noise, prediction.pred_noise[permutation])
        torch.testing.assert_close(permuted.pred_x_start, prediction.pred_x_start[permutation])


@pytest.mark.parametrize('sampling_steps', [2, 4])
@pytest.mark.parametrize('cfg_plus_plus', [False, True])
@pytest.mark.parametrize('objective', ['pred_noise', 'pred_x0', 'pred_v'])
def test_unit_scale_ddim_and_ddpm(model, sampling_steps, cfg_plus_plus, objective):
    diffusion = GaussianDiffusion(model, image_size=8, timesteps=4, sampling_timesteps=sampling_steps,
                                  objective=objective, use_cfg_plus_plus=cfg_plus_plus)
    result = diffusion.sample(classes=torch.tensor([0, 1, 2]), cond_scale=1., rescaled_phi=0.)
    assert result.shape == (3, 1, 8, 8)
    assert torch.isfinite(result).all()


def test_unit_scale_helper_returns_both_predictions_with_finite_gradients(model):
    x = torch.randn(3, 1, 8, 8, requires_grad=True)
    t = torch.ones(3, dtype=torch.long)
    classes = torch.arange(3)
    outputs = model.forward_with_cond_scale(x, t, classes, cond_scale=1.)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    conditional, unconditional = outputs
    torch.testing.assert_close(conditional, model(x, t, classes, cond_drop_prob=0.))
    torch.testing.assert_close(unconditional, model(x, t, classes, cond_drop_prob=1.))
    (conditional.square().mean() + unconditional.square().mean()).backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
