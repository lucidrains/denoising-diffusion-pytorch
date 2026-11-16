import importlib
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from denoising_diffusion_pytorch.version import __version__


class Images(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 128

    def __getitem__(self, index):
        return torch.linspace(0.1, 0.9, 16).reshape(1, 4, 4)


class Denoiser(nn.Module):
    channels = out_dim = 1
    self_condition = random_or_learned_sinusoidal_cond = False

    def __init__(self, sequence):
        super().__init__()
        self.conv = (nn.Conv1d if sequence else nn.Conv2d)(1, 1, 1)

    def forward(self, x, time, self_condition=None):
        return self.conv(x)


@pytest.fixture(
    params=[
        "denoising_diffusion_pytorch",
        "denoising_diffusion_pytorch_1d",
        "guided_diffusion",
        "repaint",
    ]
)
def trainer(request, monkeypatch, tmp_path):
    name = request.param
    module = importlib.import_module(f"denoising_diffusion_pytorch.{name}")
    monkeypatch.setattr(module, "cpu_count", lambda: 0)
    # The separate missing-import defect in guided diffusion is covered by #372.
    # Supply that metadata here so this test isolates checkpoint publication.
    monkeypatch.setattr(module, "__version__", __version__, raising=False)
    sequence = name.endswith("_1d")
    torch.manual_seed(61)
    model = Denoiser(sequence)
    if sequence:
        diffusion = module.GaussianDiffusion1D(model, seq_length=8, timesteps=4)
        data = module.Dataset1D(torch.linspace(0.1, 0.9, 128 * 8).reshape(128, 1, 8))
        trainer_class = module.Trainer1D
    else:
        monkeypatch.setattr(module, "Dataset", Images)
        diffusion = module.GaussianDiffusion(
            model, image_size=4, timesteps=4, beta_schedule="cosine"
        )
        data, trainer_class = str(tmp_path), module.Trainer
    extra = (
        {"calculate_fid": False}
        if name in ("denoising_diffusion_pytorch", "repaint")
        else {}
    )
    instance = trainer_class(
        diffusion,
        data,
        train_batch_size=16,
        train_num_steps=1,
        save_and_sample_every=1000,
        num_samples=4,
        results_folder=str(tmp_path / "checkpoints"),
        **extra,
    )
    instance.train()
    return instance


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("existing", [False, True])
def test_failed_serialization_never_publishes_a_partial_checkpoint(
    trainer, monkeypatch, existing
):
    path = trainer.results_folder / "model-latest.pt"
    if existing:
        trainer.save("latest")
        previous = path.read_bytes()
        saved = torch.load(path, weights_only=True)

    def interrupted_save(data, file, *args, **kwargs):
        if hasattr(file, "write"):
            file.write(b"incomplete checkpoint")
            file.flush()
        else:
            Path(file).write_bytes(b"incomplete checkpoint")
        raise OSError("simulated short checkpoint write")

    trainer.step += 1
    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", interrupted_save)
        with pytest.raises(OSError, match="short checkpoint write"):
            trainer.save("latest")

    if existing:
        assert path.read_bytes() == previous
        trainer.load("latest")
        assert trainer.step == saved["step"]
        assert_nested_equal(trainer.model.state_dict(), saved["model"])
        assert_nested_equal(trainer.opt.state_dict(), saved["opt"])
        assert_nested_equal(trainer.ema.state_dict(), saved["ema"])
    else:
        assert not path.exists()
    assert set(trainer.results_folder.iterdir()) == ({path} if existing else set())


def test_successful_replacement_restores_complete_native_training_state(trainer):
    trainer.save("latest")
    trainer.train_num_steps = 2
    trainer.train()
    trainer.save("latest")
    path = trainer.results_folder / "model-latest.pt"
    saved = torch.load(path, weights_only=True)
    assert saved["step"] == 2
    assert saved["opt"]["state"]
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()
    trainer.step = 0
    trainer.load("latest")
    assert trainer.step == 2
    assert_nested_equal(trainer.model.state_dict(), saved["model"])
    assert_nested_equal(trainer.opt.state_dict(), saved["opt"])
    assert_nested_equal(trainer.ema.state_dict(), saved["ema"])
    assert set(trainer.results_folder.iterdir()) == {path}
