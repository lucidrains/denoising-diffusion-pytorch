## Denoising Diffusion Probabilistic Model, in Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> in Pytorch.

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

```bibtex
@misc{chen2020wavegrad,
    title={WaveGrad: Estimating Gradients for Waveform Generation},
    author={Nanxin Chen and Yu Zhang and Heiga Zen and Ron J. Weiss and Mohammad Norouzi and William Chan},
    year={2020},
    eprint={2009.00713},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
