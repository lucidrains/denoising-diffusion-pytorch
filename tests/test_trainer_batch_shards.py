import importlib

import pytest
import torch
from accelerate.data_loader import prepare_data_loader
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ImageSamples(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 128

    def __getitem__(self, index):
        return torch.full((1, 4, 4), index / 127)


class ConvolutionalDenoiser(nn.Module):
    channels = out_dim = 1
    self_condition = random_or_learned_sinusoidal_cond = False

    def __init__(self, sequence):
        super().__init__()
        self.conv = (nn.Conv1d if sequence else nn.Conv2d)(1, 1, 1)

    def forward(self, x, time, self_condition=None):
        return self.conv(x)


@pytest.mark.parametrize("split", [None, False, True])
@pytest.mark.parametrize(
    "name",
    [
        "denoising_diffusion_pytorch",
        "denoising_diffusion_pytorch_1d",
        "guided_diffusion",
        "repaint",
    ],
)
def test_trainer_configuration_controls_actual_batch_shards(
    monkeypatch, tmp_path, name, split
):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{name}")
    monkeypatch.setattr(module, "cpu_count", lambda: 0)
    sequence = name.endswith("_1d")
    model = ConvolutionalDenoiser(sequence)
    if sequence:
        diffusion = module.GaussianDiffusion1D(model, seq_length=8, timesteps=4)
        dataset = module.Dataset1D(
            torch.arange(128).float().reshape(128, 1, 1).expand(-1, -1, 8) / 127
        )
        trainer_class, data = module.Trainer1D, dataset
    else:
        monkeypatch.setattr(module, "Dataset", ImageSamples)
        dataset = ImageSamples()
        diffusion = module.GaussianDiffusion(
            model, image_size=4, timesteps=4, beta_schedule="cosine"
        )
        trainer_class, data = module.Trainer, str(tmp_path)
    extra = (
        {"calculate_fid": False}
        if name in ("denoising_diffusion_pytorch", "repaint")
        else {}
    )
    if split is not None:
        extra["split_batches"] = split
    trainer = trainer_class(
        diffusion,
        data,
        train_batch_size=16,
        train_num_steps=1,
        num_samples=4,
        save_and_sample_every=1000,
        results_folder=str(tmp_path),
        **extra,
    )
    # Exercise Accelerate's real sharding implementation without requiring sockets.
    shards = [
        prepare_data_loader(
            DataLoader(dataset, batch_size=16),
            num_processes=2,
            process_index=rank,
            split_batches=trainer.accelerator.split_batches,
            rng_types=[],
        )
        for rank in range(2)
    ]
    batches = [next(iter(shard)) for shard in shards]
    before = [p.detach().clone() for p in model.parameters()]
    trainer.dl = module.cycle(shards[0])
    trainer.train()
    assert trainer.step == 1
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters())
    )
    expected_local = 16 if split is False else 8
    assert [len(batch) for batch in batches] == [expected_local, expected_local]
    ids = torch.cat(batches).flatten(1)[:, 0] * 127
    assert ids.round().int().tolist() == list(range(expected_local * 2))
