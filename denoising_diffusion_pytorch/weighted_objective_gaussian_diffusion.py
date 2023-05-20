import torch
from inspect import isfunction
from torch import nn, einsum
from einops import rearrange

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion

# helper functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# some improvisation on my end
# where i have the model learn to both predict noise and x0
# and learn the weighted sum for each depending on time step

class WeightedObjectiveGaussianDiffusion(GaussianDiffusion):
    def __init__(
        self,
        model,
        *args,
        pred_noise_loss_weight = 0.1,
        pred_x_start_loss_weight = 0.1,
        **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        channels = model.channels
        assert model.out_dim == (channels * 2 + 2), 'dimension out (out_dim) of unet must be twice the number of channels + 2 (for the softmax weighted sum) - for channels of 3, this should be (3 * 2) + 2 = 8'
        assert not model.self_condition, 'not supported yet'
        assert not self.is_ddim_sampling, 'ddim sampling cannot be used'

        self.split_dims = (channels, channels, 2)
        self.pred_noise_loss_weight = pred_noise_loss_weight
        self.pred_x_start_loss_weight = pred_x_start_loss_weight

    def p_mean_variance(self, *, x, t, clip_denoised, model_output = None):
        model_output = self.model(x, t)

        pred_noise, pred_x_start, weights = model_output.split(self.split_dims, dim = 1)
        normalized_weights = weights.softmax(dim = 1)

        x_start_from_noise = self.predict_start_from_noise(x, t = t, noise = pred_noise)
        
        x_starts = torch.stack((x_start_from_noise, pred_x_start), dim = 1)
        weighted_x_start = einsum('b j h w, b j c h w -> b c h w', normalized_weights, x_starts)

        if clip_denoised:
            weighted_x_start.clamp_(-1., 1.)

        model_mean, model_variance, model_log_variance = self.q_posterior(weighted_x_start, x, t)

        return model_mean, model_variance, model_log_variance

    def p_losses(self, x_start, t, noise = None, clip_denoised = False):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start = x_start, t = t, noise = noise)

        model_output = self.model(x_t, t)
        pred_noise, pred_x_start, weights = model_output.split(self.split_dims, dim = 1)

        # get loss for predicted noise and x_start
        # with the loss weight given at initialization

        noise_loss = F.mse_loss(noise, pred_noise) * self.pred_noise_loss_weight
        x_start_loss = F.mse_loss(x_start, pred_x_start) * self.pred_x_start_loss_weight

        # calculate x_start from predicted noise
        # then do a weighted sum of the x_start prediction, weights also predicted by the model (softmax normalized)

        x_start_from_pred_noise = self.predict_start_from_noise(x_t, t, pred_noise)
        x_start_from_pred_noise = x_start_from_pred_noise.clamp(-2., 2.)
        weighted_x_start = einsum('b j h w, b j c h w -> b c h w', weights.softmax(dim = 1), torch.stack((x_start_from_pred_noise, pred_x_start), dim = 1))

        # main loss to x_start with the weighted one

        weighted_x_start_loss = F.mse_loss(x_start, weighted_x_start)
        return weighted_x_start_loss + x_start_loss + noise_loss
