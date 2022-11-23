from pathlib import Path

from denoising_diffusion_pytorch import Unet, GaussianDiffusionSegmentationMapping, TrainerSegmentation

def test(images_folder, segmentation_folder, results_folder):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ) #.cuda()

    diffusion = GaussianDiffusionSegmentationMapping(
        model,
        image_size=128,
        margin=1.0,
        loss_type="triplet",
        timesteps=1000,           
        sampling_timesteps=250
    ) #.cuda()

    trainer = TrainerSegmentation(
        diffusion,
        images_folder,
        segmentation_folder,
        augment_horizontal_flip=False,
        results_folder=results_folder,
        save_and_sample_every=10,
        num_samples=25,
        train_batch_size=32,
        train_lr=8e-5,
        train_num_steps=100,    
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=True
    )

    trainer.train()


def main():
    full_folder = Path("/mnt/c/Users/ivayl/OneDrive/Documents/Bachelor_Thesis/dataset")
    images_folder = full_folder / "images"
    segmentation_folder = full_folder / "segmentations"
    results_folder = full_folder / "results"

    test(images_folder, segmentation_folder, results_folder)


if __name__ == "__main__":
    main()