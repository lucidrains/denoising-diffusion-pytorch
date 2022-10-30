from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# main class

class ElucidatedDiffusion(nn.Module):
    def __init__(
        self,
        net,
        *,
        image_size,
        channels = 3,
        num_sample_steps = 32, # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
    ):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.self_condition = net.self_condition

        self.net = net

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_images, sigma, self_cond = None, clamp = False):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_images,
            self.c_noise(sigma),
            self_cond
        )

        out = self.c_skip(padded_sigma) * noised_images +  self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps = None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(self, batch_size = 16, num_sample_steps = None, clamp = True):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device = self.device)

        # for self conditioning

        x_start = None

        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            self_cond = x_start if self.self_condition else None

            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond, clamp = clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None

                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, self_cond, clamp = clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next
            x_start = model_output

        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)

    # training

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(self, images):
        batch_size, c, h, w, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels

        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'
        assert c == channels, 'mismatch of image channels'

        images = normalize_to_neg_one_to_one(images)

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        noise = torch.randn_like(images)

        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        self_cond = None

        if self.self_condition and random() < 0.5:
            # from hinton's group's bit diffusion paper
            with torch.no_grad():
                self_cond = self.preconditioned_network_forward(noised_images, sigmas)
                self_cond.detach_()

        denoised = self.preconditioned_network_forward(noised_images, sigmas, self_cond)

        losses = F.mse_loss(denoised, images, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()
