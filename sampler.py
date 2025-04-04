import argparse
import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def main(num_samples, image_path, train_load_num, output_folder, dim, dim_mults, sampling_timesteps=100, batch_size=4, image_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = Path(image_path)

    model = Unet(
        dim=dim,
        dim_mults=dim_mults,
        flash_attn=True
    ).to(device)

    print(f'Model parameter count: {sum(p.numel() for p in model.parameters()):,}')

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps
    ).to(device)

    trainer = Trainer(
        diffusion,
        paths,
        train_batch_size=batch_size,
        train_lr=5e-4,
        train_num_steps=60000,
        gradient_accumulate_every=5,
        ema_decay=0.995,
        amp=True,
        calculate_fid=False,
        save_and_sample_every=2000,
        num_samples=4,
        results_folder='./results',
    )

    trainer.load(train_load_num)
    trainer.model.eval()

    os.makedirs(output_folder, exist_ok=True)

    full_batches = num_samples // batch_size
    remainder = num_samples % batch_size

    for batch_num in range(full_batches):
        generate_and_save_images(diffusion, batch_size, sampling_timesteps, output_folder, batch_num * batch_size, device)

    if remainder > 0:
        generate_and_save_images(diffusion, remainder, sampling_timesteps, output_folder, full_batches * batch_size, device)

def generate_and_save_images(diffusion, batch_size, sampling_timesteps, output_folder, start_idx, device):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            gen_images = diffusion.sample(batch_size=batch_size).to(device)
    save_images(gen_images, output_folder, start_idx)

def save_images(gen_images, output_folder, start_idx):
    for i, img_tensor in enumerate(gen_images):
        np_array = (img_tensor.cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(np_array).save(os.path.join(output_folder, f"generated_image_{start_idx + i}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a denoising diffusion model.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of images to generate.")
    parser.add_argument("--sampling_timesteps", type=int, default=100, help="Number of sampling timesteps (<= 1000).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--image_size", type=int, default=128, help="Image size for generation.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image directory.")
    parser.add_argument("--train_load_num", type=int, required=True, help="Training load checkpoint number.")
    parser.add_argument("--output_folder", type=str, default="gen_images", help="Output folder for generated images.")
    parser.add_argument("--dim", type=int, required=True, help="Dimension size for the model.")
    parser.add_argument("--dim_mults", type=int, nargs='+', required=True, help="List of dimension multipliers for the model.")
    args = parser.parse_args()

    if args.sampling_timesteps > 1000:
        raise ValueError("sampling_timesteps must be <= 1000.")

    main(args.num_samples, args.image_path, args.train_load_num, args.output_folder, args.dim, args.dim_mults, args.sampling_timesteps, args.batch_size, args.image_size)
