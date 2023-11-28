import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_for_residual import Unet, GaussianDiffusion, Trainer


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/home/linfeng/Diffusion Model/datasets/ITS/imgH',
    train_batch_size = 16,
    train_lr = 8e-5,
    train_num_steps = 30000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
)

trainer.train()