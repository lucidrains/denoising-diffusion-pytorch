import unittest

import torch
from torch import nn

from denoising_diffusion_pytorch.elucidated_diffusion import ElucidatedDiffusion


class RecordingDenoiser(nn.Module):
    random_or_learned_sinusoidal_cond = True

    def __init__(self, self_condition):
        super().__init__()
        self.self_condition = self_condition
        self.anchor = nn.Parameter(torch.tensor(.1))
        self.inputs = []

    def forward(self, image, time, self_cond=None):
        self.inputs.append(None if self_cond is None else self_cond.detach().clone())
        # An explicitly self-conditioned network whose arithmetic remains native.
        output = image * 0 + self.anchor
        if self_cond is not None:
            output = output + .2 * self_cond
        return output


class TestDPMppSelfCondition(unittest.TestCase):
    def test_uses_previous_denoised_prediction(self):
        torch.manual_seed(14)
        net = RecordingDenoiser(True)
        diffusion = ElucidatedDiffusion(
            net, image_size=2, channels=1, num_sample_steps=4, sigma_min=.1, sigma_max=1.,
        )
        denoised = []
        original_forward = diffusion.preconditioned_network_forward
        def record(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            denoised.append(result.detach().clone())
            return result
        # Observe complete native preconditioned outputs; sampler arithmetic is unchanged.
        diffusion.preconditioned_network_forward = record
        result = diffusion.sample_using_dpmpp(batch_size=2)
        self.assertEqual(len(net.inputs), 4)
        self.assertIsNone(net.inputs[0])
        for index in range(1, 4):
            self.assertIsNotNone(net.inputs[index])
            torch.testing.assert_close(net.inputs[index], denoised[index - 1])
        self.assertTrue(torch.isfinite(result).all())

    def test_non_self_conditioned_model_receives_no_condition(self):
        net = RecordingDenoiser(False)
        diffusion = ElucidatedDiffusion(
            net, image_size=2, channels=1, num_sample_steps=3, sigma_min=.1, sigma_max=1.,
        )
        diffusion.sample_using_dpmpp(batch_size=2)
        self.assertEqual(len(net.inputs), 3)
        self.assertTrue(all(value is None for value in net.inputs))


if __name__ == "__main__":
    unittest.main()
