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
        denoise_fn,
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
        assert denoise_fn.learned_sinusoidal_cond

        self.denoise_fn = denoise_fn

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
        return next(self.denoise_fn.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * -0.5

    def c_noise(self, sigma):
        """ apparently empirically derived """
        return log(sigma) ** 0.25

    # noise distribution

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    # sampling related functions

    @torch.no_grad()
    def sample_one_timestep(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device
        return x

    @torch.no_grad()
    def sample_all_timesteps(self, shape):
        batch = shape[0]

        img = torch.randn(shape, device = self.device)
        steps = torch.linspace(1., 0., 100 + 1, device = self.device)

        for i in tqdm(range(100), desc = 'sampling loop time step', total = 100):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.sample_one_timestep(img, times, times_next)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        return self.sample_all_timesteps((batch_size, self.channels, self.image_size, self.image_size))

    # training related functions - noise prediction

    def add_noise(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noised =  x_start + noise
        return x_noised, noise.mean(dim = (1, 2, 3))

    def random_times(self, batch_size):
        # times are now uniform from 0 to 1
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    def forward(self, images):
        b, c, h, w, device, image_size, = *images.shape, images.device, self.image_size
        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'

        times = self.random_times(b)
        images = normalize_to_neg_one_to_one(images)

        noise = torch.randn_like(images)

        noise_images, log_snr = self.add_noise(x_start = images, times = times, noise = noise)
        model_out = self.denoise_fn(noise_images, log_snr)

        losses = F.mse_loss(model_out, noise, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        return losses.mean()
