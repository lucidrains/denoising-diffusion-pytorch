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
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003
    ):
        super().__init__()
        assert net.learned_sinusoidal_cond

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
        return (sigma ** 2 + self.sigma_data ** 2) * -0.5

    def c_noise(self, sigma):
        """ apparently empirically derived """
        return log(sigma) * 0.25

    # noise distribution

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_images, sigma):
        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_images,
            self.c_noise(sigma)
        )

        return self.c_skip(padded_sigma) * noised_images +  self.c_out(padded_sigma) * net_out

    # sampling

    @torch.no_grad()
    def sample(self, batch_size = 16):
        shape = (batch_size, self.channels, self.image_size, self.image_size)

        images = torch.randn(shape, device = self.device)
        steps = torch.linspace(1., 0., 100 + 1, device = self.device)

        for i in tqdm(range(100), desc = 'sampling loop time step', total = 100):
            times = steps[i]
            times_next = steps[i + 1]
            images = images

        return unnormalize_to_zero_to_one(images)

    # training

    def forward(self, images):
        batch_size, c, h, w, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels

        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'
        assert c == channels, 'mismatch of image channels'

        images = normalize_to_neg_one_to_one(images)

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        noise = torch.randn_like(images)

        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        model_out = self.preconditioned_network_forward(noised_images, sigmas)

        losses = F.mse_loss(model_out, images, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()
