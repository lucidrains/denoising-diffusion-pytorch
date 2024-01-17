"""
This script is for testing diffusion models trained by this training script
denoising-diffusion-pytorch/denoising_diffusion_pytorch/ddpm_sandbox/Trainer2.py
and which are saved here
denoising-diffusion-pytorch/denoising_diffusion_pytorch/models
"""
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import logging
from PIL import Image

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def tensor_to_images(images_tensor: torch.Tensor):
    # Assume images of the shape is B X C X H X W where N is the number of images
    assert len(images_tensor.shape) == 4, "Images tensor must have dims B X C X H X W"
    # assert that C = 1, i.e. one channel and the image is BW
    assert images_tensor.shape[1] == 1, "Supporting BW images only with one channel, i.e. dim C = 1"
    images_tensor = torch.clamp(input=images_tensor, min=0, max=1) * 255.0
    images_tensor = images_tensor.squeeze()
    logger.info(
        f"After clamping and reverse-normalizing , min and max = {torch.min(images_tensor)}, {torch.max(images_tensor)}")
    images_tensors_list = list(images_tensor)
    prefix = "generated_img"
    for i, image_tensor in enumerate(images_tensors_list):
        img = Image.fromarray(image_tensor.detach().cpu().numpy()).convert("L")
        img.save(f"{prefix}_{i}.png")


if __name__ == '__main__':
    time_steps = 1000
    device = torch.device('cuda')
    image_size = 32
    num_images = 1
    num_channels = 1
    batch_size = 64
    num_train_step = 20_000
    model_path = "../models/diffusion_mnist_8_n_train_steps_50000.pkl"

    # Test if cuda is available
    logger.info(f"Cuda checks")
    logger.info(f'Is cuda available ? : {torch.cuda.is_available()}')
    logger.info(f'Cuda device count = {torch.cuda.device_count()}')
    # https://github.com/mbaddar1/denoising-diffusion-pytorch?tab=readme-ov-file#usage

    logger.info(f"Creating UNet and Diffusion Models ")
    # UNet
    unet_model = Unet(
        dim=64,
        channels=num_channels,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    ).to(device)

    # Double-checking if models are actually on cuda
    #   https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/2
    is_unet_model_on_cuda = next(unet_model.parameters()).is_cuda
    logger.info(f'If core model is on cuda ? : {is_unet_model_on_cuda}')

    diffusion = GaussianDiffusion(
        model=unet_model,
        image_size=image_size,
        timesteps=time_steps,  # number of steps
        auto_normalize=False
    ).to(device)

    is_diffusion_model_on_cuda = next(diffusion.parameters()).is_cuda
    logger.info(f'Is diffusion model on cuda? : {is_diffusion_model_on_cuda}')
    logger.info(f"loading model weights from file {model_path}")
    diffusion.load_state_dict(torch.load(model_path))
    logger.info(f"Successfully loaded model weights")
    #
    logger.info("Sampling images")
    sampled_images = diffusion.sample(batch_size=4)
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    logger.info(
        f"quantiles of levels {quantiles} = {torch.quantile(input=sampled_images, q=torch.tensor(quantiles).to(device))}")
    logger.info(f"Average of the sampled images {torch.mean(sampled_images)}")
    tensor_to_images(images_tensor=sampled_images)
    logger.info("Testing script finished")
