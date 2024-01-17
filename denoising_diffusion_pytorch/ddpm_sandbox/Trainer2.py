"""
This script is an attempt to write my own trainer for DDPM based on the classes and methods in
denoising-diffusion-pytorch/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py.
The trainer will be also based on the Trainer class in denoising_diffusion_pytorch.denoising_diffusion_pytorch.Trainer
"""
import numpy as np
from PIL import Image
from denoising_diffusion_pytorch import GaussianDiffusion, Unet
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import transforms as T
from torch.optim import Optimizer, Adam
from tqdm import tqdm
from datetime import datetime

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()



# similar to
class Dataset2(Dataset):
    EXTENSIONS = ['jpg', 'png']

    def __init__(self, folder: str, image_size: int, debug_flag: bool = False):
        super().__init__()
        self.debug_flag = debug_flag
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in Dataset2.EXTENSIONS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        # based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L838
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        # Line for learning purposes. Raw values are from 0 to 255.
        #   See this post https://discuss.pytorch.org/t/pil-image-and-its-normalisation/78812/4?u=mbaddar
        raw_img_data = np.array(img)  # FIXME for debugging only , remove later
        if self.debug_flag:
            old_level = logger.level
            logger.setLevel(logging.DEBUG)
            logger.debug(f"raw img data =\n {raw_img_data}")
            logger.setLevel(old_level)
        transformed_image = self.transform(img)
        return transformed_image

        # Some coding notes:
        # --------------------
        # img = read_image(str(path))
        # transformed_image = torch.squeeze(img)  # Just squeezing dims with value = 1 , no actual transformation
        # Cannot do squeezing as code is designed to each image with channel, height and width : see this error
        # File "/home/mbaddar/Documents/mbaddar/phd/genmodel/denoising-diffusion-pytorch/denoising_diffusion_pytorch/
        #   denoising_diffusion_pytorch.py", line 854, in forward
        #     b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # ValueError: not enough values to unpack (expected 6, got 5)
        # As long as we have validated loaded data against passed num_channel and image size , it should be fine


class Trainer2:
    def __init__(self, diffusion_model: GaussianDiffusion, batch_size: int, train_num_steps: int,
                 device: torch.device,
                 dataset: Dataset,
                 optimizer: Optimizer,
                 debug_flag: bool = False):
        self.diffusion_model = diffusion_model
        self.debug_flag = debug_flag
        self.device = device
        self.train_num_steps = train_num_steps
        self.optimizer = optimizer
        self.step = 0
        #
        logger.info(f"Creating data loader for the dataset")
        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        if self.debug_flag:
            self.__validate_dataset()  # FIXME remove later

    def train(self):
        # Use manual control for tqdm and progress bar update
        # See https://github.com/tqdm/tqdm#usage
        # and   https://stackoverflow.com/a/45808255
        self.step = 0
        update_frequency = 100
        # Exponential smoothing for loss reporting
        # Called Exponential Smoothing or Exponential Moving Average
        ema_loss = 0.0  # Just for initialization
        ema_loss_alpha = 0.9
        start_timestamp = datetime.now()
        with tqdm(initial=self.step, total=self.train_num_steps) as progress_bar:
            while self.step < self.train_num_steps:
                self.optimizer.zero_grad()
                data = next(iter(self.dataloader)).to(device)
                loss = self.diffusion_model(data)
                if self.step == 0:
                    ema_loss = loss.item()
                else:
                    ema_loss = ema_loss_alpha * loss.item() + (1 - ema_loss_alpha) * ema_loss
                loss.backward()
                self.optimizer.step()
                # TODO add
                #   1. loss curve tracking
                #   2. sampling code
                #   3. quality measure for samples , along with loss (FID  , WD , etc..)
                if self.step % update_frequency == 0:
                    progress_bar.set_description(f'loss: {ema_loss:.4f}')
                self.step += 1
                progress_bar.update(1)
        end_datetime = datetime.now()
        elapsed_time = (end_datetime - start_timestamp).seconds
        logger.info(f"Training time = {elapsed_time} seconds")

    # private method
    # https://www.geeksforgeeks.org/private-methods-in-python/
    def __validate_dataset(self):
        """
        Test the coherence of loaded data to diffusion model data-related parameters
        """

        num_test_iters = 10
        quantiles = [0., 0.25, 0.5, 0.75, 1.0]
        for i in range(num_test_iters):
            data = next(iter(self.dataloader))
            old_level = logger.level
            logger.setLevel(logging.DEBUG)  # would it work ?
            logger.debug(f"Data batch # {i + 1} loaded with dimensions : {data.shape}")
            logger.debug(
                f"For quantiles at levels {quantiles} = {torch.quantile(input=data, q=torch.tensor(quantiles))}")
            logger.setLevel(old_level)

            # data should have dimension : batch, channels , height , width
            # height should be = width
            assert self.diffusion_model.image_size == data.shape[2], \
                "loaded data height must be equal to diffusion model image size property"
            assert self.diffusion_model.image_size == data.shape[3], \
                "loaded data width must be equal to diffusion model image size property"


if __name__ == '__main__':
    # Params and constants
    time_steps = 1000
    device = torch.device('cuda')
    image_size = 32
    num_images = 1
    num_channels = 1
    batch_size = 64
    num_train_step = 50_000
    mnist_number = 8
    # Test if cuda is available
    logger.info(f"Cuda checks")
    logger.info(f'Is cuda available ? : {torch.cuda.is_available()}')
    logger.info(f'Cuda device count = {torch.cuda.device_count()}')
    # https://github.com/mbaddar1/denoising-diffusion-pytorch?tab=readme-ov-file#usage
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

    # Dataset
    logger.info("Setting up MNIST dataset")
    mnist_data_path = f"../mnist_image_samples/{mnist_number}"
    mnist_dataset = Dataset2(folder=mnist_data_path, image_size=image_size)

    # Trainer
    opt = Adam(params=diffusion.parameters(), lr=1e-4)
    trainer = Trainer2(diffusion_model=diffusion, batch_size=batch_size, dataset=mnist_dataset, debug_flag=True,
                       optimizer=opt, device=device, train_num_steps=num_train_step)
    trainer.train()
    logger.info(f"Saving the diffusion model...")
    model_path = f"../models/diffusion_mnist_{mnist_number}_n_train_steps_{num_train_step}.pkl"
    torch.save(diffusion.state_dict(), model_path)
    logger.info(f"Successfully model saved to {model_path}")
    logger.info(f"Training script finished")

    #######################
    """
        Note
        =====
        In the original code , I have tried to use cuda and had the following error
        "RuntimeError: CUDA error: out of memory"?
        I have tried
            1. using gc collect and torch empty cache
            2. os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
        But none of them worked
        What worked is to reduce the image size passed for data and mode from 128 to 64
            ...
            device = torch.device('cuda')
            image_size=64
            diffusion = GaussianDiffusion( model,
                image_size=image_size,
                timesteps=time_steps  # number of steps
                ).to(device)
            training_images = torch.rand(num_images, num_channels, image_size, image_size).to(device)  
            # images are normalized from 0 to 1
        This is the snippet that works.
        """
