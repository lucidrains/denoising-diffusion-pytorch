import importlib

import pytest
import torch
from PIL import Image

from denoising_diffusion_pytorch import fid_evaluation


@pytest.fixture
def image_folder(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index in range(100):
        Image.new("L", (8, 8), color=index * 2).save(folder / f"{index}.png")
    return folder


def make_trainer(module, monkeypatch, images, results, score, steps, best_only=True):
    # Metric values are controlled to exercise checkpoint decisions without downloading Inception weights.
    class FixedMetric:
        def __init__(self, **kwargs):
            self.calls = 0

        def fid_score(self):
            self.calls += 1
            return score

    monkeypatch.setattr(fid_evaluation, "FIDEvaluation", FixedMetric)
    if hasattr(module, "FIDEvaluation"):
        monkeypatch.setattr(module, "FIDEvaluation", FixedMetric)
    monkeypatch.setattr(module, "cpu_count", lambda: 0)
    torch.manual_seed(16)
    model = module.Unet(dim=8, dim_mults=(1, 2), channels=1, attn_heads=1, attn_dim_head=8)
    diffusion = module.GaussianDiffusion(
        model, image_size=8, timesteps=4, sampling_timesteps=2, beta_schedule="cosine"
    )
    return module.Trainer(
        diffusion, str(images), train_batch_size=16, train_num_steps=steps,
        save_and_sample_every=1, num_samples=1, ema_update_every=1,
        augment_horizontal_flip=False, results_folder=str(results),
        calculate_fid=best_only, save_best_and_latest_only=best_only,
    )


@pytest.mark.parametrize("variant", ["denoising_diffusion_pytorch", "repaint"])
@pytest.mark.parametrize("next_score", [5.0, 10.0, 20.0])
def test_resuming_preserves_best_checkpoint_selection(variant, next_score, monkeypatch, image_folder, tmp_path):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{variant}")
    results = tmp_path / "results"
    first = make_trainer(module, monkeypatch, image_folder, results, score=10.0, steps=1)
    first.train()
    best_path = results / "model-best.pt"
    original_best = best_path.read_bytes()
    first_parameters = {name: value.detach().clone() for name, value in first.model.named_parameters()}

    resumed = make_trainer(module, monkeypatch, image_folder, results, score=next_score, steps=2)
    resumed.load("latest")
    assert resumed.best_fid == 10.0
    assert resumed.step == 1
    for name, value in resumed.model.named_parameters():
        torch.testing.assert_close(value, first_parameters[name], rtol=0, atol=0)
    resumed.train()

    assert resumed.step == 2
    assert resumed.fid_scorer.calls == 1
    assert any(not torch.equal(value, first_parameters[name]) for name, value in resumed.model.named_parameters())
    assert resumed.best_fid == min(10.0, next_score)
    latest = torch.load(results / "model-latest.pt", weights_only=True)
    best = torch.load(best_path, weights_only=True)
    assert latest["step"] == 2
    assert latest["best_fid"] == min(10.0, next_score)
    if next_score < 10.0:
        assert best["step"] == 2
        assert best["best_fid"] == next_score
    else:
        assert best_path.read_bytes() == original_best
        assert best["step"] == 1


@pytest.mark.parametrize("variant", ["denoising_diffusion_pytorch", "repaint"])
def test_legacy_checkpoint_keeps_the_existing_fallback(variant, monkeypatch, image_folder, tmp_path):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{variant}")
    results = tmp_path / "results"
    trainer = make_trainer(module, monkeypatch, image_folder, results, score=10.0, steps=1)
    trainer.save("legacy")
    path = results / "model-legacy.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint.pop("best_fid", None)
    torch.save(checkpoint, path)
    trainer.best_fid = 17.0
    trainer.load("legacy")
    assert trainer.best_fid == 17.0


@pytest.mark.parametrize("variant", ["denoising_diffusion_pytorch", "repaint"])
def test_regular_checkpoint_mode_does_not_require_a_best_score(variant, monkeypatch, image_folder, tmp_path):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{variant}")
    results = tmp_path / "results"
    trainer = make_trainer(module, monkeypatch, image_folder, results, score=None, steps=1, best_only=False)
    trainer.save("regular")
    trainer.load("regular")
    checkpoint = torch.load(results / "model-regular.pt", weights_only=True)
    assert "best_fid" not in checkpoint
    assert not hasattr(trainer, "best_fid")
