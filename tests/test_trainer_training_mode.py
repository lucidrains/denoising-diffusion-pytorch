import copy
import importlib

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset


class Images(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 128

    def __getitem__(self, index):
        return torch.linspace(0.0, 0.5, 16).reshape(1, 4, 4) + index / 256


class Denoiser(nn.Module):
    channels = out_dim = 1
    self_condition = random_or_learned_sinusoidal_cond = False

    def __init__(self, sequence):
        super().__init__()
        self.conv = (nn.Conv1d if sequence else nn.Conv2d)(1, 1, 1)
        self.norm = (nn.BatchNorm1d if sequence else nn.BatchNorm2d)(1)
        self.dropout = nn.Dropout(0.5)
        self.modes = []

    def forward(self, x, time, self_condition=None):
        self.modes.append((self.norm.training, self.dropout.training))
        return self.dropout(self.norm(self.conv(x)))


def make_pair(monkeypatch, tmp_path, name, initially_eval):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{name}")
    monkeypatch.setattr(module, "cpu_count", lambda: 0)
    sequence = name.endswith("_1d")
    torch.manual_seed(31)
    actual_model = Denoiser(sequence)
    actual_model.train(not initially_eval)
    reference_model = copy.deepcopy(actual_model).train()
    extra = (
        {"calculate_fid": False}
        if name in ("denoising_diffusion_pytorch", "repaint")
        else {}
    )
    trainers = []
    for label, model in [("actual", actual_model), ("reference", reference_model)]:
        if sequence:
            diffusion = module.GaussianDiffusion1D(model, seq_length=8, timesteps=4)
            data = module.Dataset1D(
                torch.linspace(0.1, 0.9, 128 * 8).reshape(128, 1, 8)
            )
            trainer_class = module.Trainer1D
        else:
            monkeypatch.setattr(module, "Dataset", Images)
            diffusion = module.GaussianDiffusion(
                model, image_size=4, timesteps=4, beta_schedule="cosine"
            )
            data, trainer_class = str(tmp_path), module.Trainer
        trainers.append(
            trainer_class(
                diffusion,
                data,
                train_batch_size=16,
                train_num_steps=1,
                save_and_sample_every=1000,
                num_samples=4,
                results_folder=str(tmp_path / label),
                **extra,
            )
        )
    return trainers, (actual_model, reference_model)


def compare_training(trainers, models):
    for trainer in trainers:
        torch.manual_seed(97)
        trainer.train()
    actual, reference = models
    assert actual.modes == reference.modes
    assert all(norm and dropout for norm, dropout in actual.modes)
    assert actual.norm.num_batches_tracked > 0
    for key, expected in reference.state_dict().items():
        value = actual.state_dict()[key]
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name",
    [
        "denoising_diffusion_pytorch",
        "denoising_diffusion_pytorch_1d",
        "guided_diffusion",
        "repaint",
    ],
)
@pytest.mark.parametrize("initially_eval", [False, True])
def test_training_restores_model_mode(monkeypatch, tmp_path, name, initially_eval):
    trainers, models = make_pair(monkeypatch, tmp_path, name, initially_eval)
    compare_training(trainers, models)


@pytest.mark.parametrize("name", ["guided_diffusion", "repaint"])
def test_resuming_training_after_external_evaluation(monkeypatch, tmp_path, name):
    trainers, models = make_pair(monkeypatch, tmp_path, name, False)
    compare_training(trainers, models)
    trainers[0].model.eval()
    for trainer in trainers:
        trainer.train_num_steps = 2
    compare_training(trainers, models)
