<img src="./denoising-diffusion.png" width="500px"></img>

## Denoising Diffusion Probabilistic Model, in Pytorch

Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> in Pytorch. It is a new approach to generative modeling that may <a href="https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/">have the potential</a> to rival GANs. It uses denoising score matching to estimate the gradient of the data distribution, followed by Langevin sampling to sample from the true distribution. This implementation was transcribed from the official Tensorflow version <a href="https://github.com/hojonathanho/diffusion">here</a>. 

## Install

```bash
$ pip install denoising_diffusion_pytorch
```

## Usage

```python
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    beta_start = 0.0001,
    beta_end = 0.02,
    num_diffusion_timesteps = 1000,   # number of steps
    loss_type = 'l1'                  # L1 or L2 (wavegrad paper claims l1 is better?)
)

training_images = torch.randn(8, 3, 128, 128)
loss = diffusion(training_images)
loss.backward()
# after a lot of training

sampled_images = diffusion.p_sample_loop((1, 3, 128, 128))
sampled_images.shape # (1, 3, 128, 128)
```

Or, if you simply want to pass in a folder name and the desired image dimensions, you can use the `Trainer` class to easily train a model.

```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    beta_start = 0.0001,
    beta_end = 0.02,
    num_diffusion_timesteps = 1000,   # number of steps
    loss_type = 'l1'                  # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    image_size = 128,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 100000,
    gradient_accumulate_every = 2
)

trainer.train()
```

Todo: Command line tool for one-line training

## Citations

```bibtex
@misc{ho2020denoising,
    title={Denoising Diffusion Probabilistic Models},
    author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year={2020},
    eprint={2006.11239},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
