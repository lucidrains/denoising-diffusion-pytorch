import logging
import os.path

from denoising_diffusion_pytorch import GaussianDiffusion, Unet
import torch

DEBUG_DATA_PATH = "data"
# based on the code line denoising-diffusion-pytorch/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py:43

# logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

if __name__ == '__main__':
    time_steps = 1000
    device = torch.device('cuda')
    image_size = 32
    num_images = 1
    num_channels = 1

    model = Unet(
        dim=64,
        channels=num_channels,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=image_size,
        timesteps=time_steps,  # number of steps
        auto_normalize=False
    ).to(device)

    # Double-checking if models are actually on cuda
    #   https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/2
    is_unet_model_on_cuda = next(model.parameters()).is_cuda
    logger.info(f'If core model is on cuda ? : {is_unet_model_on_cuda}')

    is_diffusion_model_on_cuda = next(diffusion.parameters()).is_cuda
    logger.info(f'Is diffusion model on cuda ? : {is_diffusion_model_on_cuda}')

    ###

    # data from this data dump
    data = torch.load(f=os.path.join(DEBUG_DATA_PATH, "data_tensor_1.pkl"))
    logger.debug("data loaded")
    # Some data debugging info
    logger.debug(f"Data is of type {type(data)}, with dtype {data.dtype} and shape {data.shape}")
    logger.debug(f"Data has nan ? {torch.any(torch.isnan(data))}")
    logger.debug(f"Data max = {torch.max(data)} Min {torch.min(data)} avg {torch.mean(data)}")
    # Should give close results to
    #   here denoising-diffusion-pytorch/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py:1115
    loss = diffusion(data)
    print(loss)
