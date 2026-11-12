import math
import weakref

import numpy as np
import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch import GaussianDiffusion, Unet, fid_evaluation


class IdentityFeatures(nn.Module):
    """Controlled feature populations; no pretrained network is downloaded."""

    BLOCK_INDEX_BY_DIM = {8: 0}

    def __init__(self, blocks):
        super().__init__()

    def forward(self, samples):
        return [samples]


class PopulationSampler(nn.Module):
    def __init__(self, population):
        super().__init__()
        self.population = population
        self.offset = 0

    def sample(self, batch_size):
        samples = self.population[self.offset:self.offset + batch_size]
        self.offset += batch_size
        return samples


def evaluator(monkeypatch, tmp_path, real, fake, batch_size):
    monkeypatch.setattr(fid_evaluation, "InceptionV3", IdentityFeatures)
    return fid_evaluation.FIDEvaluation(
        batch_size=batch_size,
        dl=iter(real.split(batch_size)),
        sampler=PopulationSampler(fake),
        channels=8,
        stats_dir=str(tmp_path),
        device="cpu",
        num_fid_samples=len(real),
        inception_block_idx=8,
    )


def moments(population):
    array = population.reshape(len(population), -1).double().numpy()
    return array.mean(axis=0), np.cov(array, rowvar=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [2, 17, 65])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_fid_matches_full_population_and_cached_statistics(dtype, count, batch_size, monkeypatch, tmp_path):
    generator = torch.Generator().manual_seed(105)
    real = torch.randn(count, 8, 1, 1, dtype=dtype, generator=generator)
    fake = torch.randn(count, 8, 1, 1, dtype=dtype, generator=generator) * 1.3 + 0.2
    calculation = evaluator(monkeypatch, tmp_path, real, fake, batch_size)
    expected_real = moments(real)
    expected_fake = moments(fake)
    expected_fid = fid_evaluation.calculate_frechet_distance(*expected_fake, *expected_real)
    score = calculation.fid_score()
    np.testing.assert_allclose(calculation.m2, expected_real[0], atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(calculation.s2, expected_real[1], atol=2e-7, rtol=2e-6)
    assert score == pytest.approx(expected_fid, abs=3e-6, rel=3e-6)
    with np.load(tmp_path / "dataset_stats.npz") as cached:
        np.testing.assert_array_equal(cached["m2"], calculation.m2)
        np.testing.assert_array_equal(cached["s2"], calculation.s2)

    reloaded = evaluator(monkeypatch, tmp_path, real[:0], fake, batch_size)
    reloaded.n_samples = count
    assert reloaded.fid_score() == pytest.approx(score, abs=1e-10, rel=1e-10)


@pytest.mark.parametrize("phase", ["real", "fake"])
def test_completed_feature_batches_are_released(phase, monkeypatch, tmp_path):
    population = torch.arange(65 * 8, dtype=torch.float32).reshape(65, 8, 1, 1) / 100
    calculation = evaluator(monkeypatch, tmp_path, population, population + 0.1, 8)
    if phase == "fake":
        calculation.load_or_precalc_dataset_stats()
    references = []
    previous_batches_alive = []
    original = calculation.calculate_inception_features

    def track_features(samples):
        previous_batches_alive.append(sum(reference() is not None for reference in references))
        features = original(samples)
        references.append(weakref.ref(features))
        return features

    calculation.calculate_inception_features = track_features
    if phase == "real":
        calculation.load_or_precalc_dataset_stats()
    else:
        calculation.fid_score()
    assert len(references) == math.ceil(len(population) / 8)
    assert max(previous_batches_alive) <= 1, previous_batches_alive
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("offset", [0.0, 1e8])
def test_float64_moments_preserve_small_variation_around_large_means(offset, monkeypatch, tmp_path):
    generator = torch.Generator().manual_seed(15)
    real = (torch.randn(101, 8, 1, 1, dtype=torch.float64, generator=generator) + offset)
    calculation = evaluator(monkeypatch, tmp_path, real, real, 7)
    calculation.load_or_precalc_dataset_stats()
    mean, covariance = moments(real)
    np.testing.assert_allclose(calculation.m2, mean, atol=1e-7, rtol=1e-14)
    np.testing.assert_allclose(calculation.s2, covariance, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("count", [0, 1])
def test_insufficient_populations_do_not_create_invalid_caches(count, monkeypatch, tmp_path):
    real = torch.zeros(count, 8, 1, 1)
    calculation = evaluator(monkeypatch, tmp_path, real, real, 8)
    with pytest.raises(ValueError, match="at least two"):
        calculation.load_or_precalc_dataset_stats()
    assert not (tmp_path / "dataset_stats.npz").exists()


class SmallConvFeatures(nn.Module):
    """A native feature network isolates statistics from pretrained Inception."""

    BLOCK_INDEX_BY_DIM = {8: 0}

    def __init__(self, blocks):
        super().__init__()
        self.projection = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, samples):
        return [self.projection(samples).relu()]


@pytest.mark.parametrize("channels", [1, 3])
def test_native_diffusion_sampling_and_feature_pooling(channels, monkeypatch, tmp_path):
    torch.manual_seed(88)
    model = Unet(dim=8, dim_mults=(1,), channels=channels, attn_heads=1, attn_dim_head=8)
    sampler = GaussianDiffusion(model, image_size=8, timesteps=4, sampling_timesteps=2).eval()
    real = torch.rand(8, channels, 8, 8)
    monkeypatch.setattr(fid_evaluation, "InceptionV3", SmallConvFeatures)
    calculation = fid_evaluation.FIDEvaluation(
        batch_size=3, dl=iter(real.split(3)), sampler=sampler,
        channels=channels, stats_dir=str(tmp_path), device="cpu",
        num_fid_samples=8, inception_block_idx=8,
    )
    with torch.inference_mode():
        expected_real = moments(calculation.calculate_inception_features(real))
        torch.manual_seed(42)
        features = [calculation.calculate_inception_features(sampler.sample(batch_size=n)) for n in (3, 3, 2)]
        expected_fake = moments(torch.cat(features))
        expected_score = fid_evaluation.calculate_frechet_distance(*expected_fake, *expected_real)
        torch.manual_seed(42)
        actual_score = calculation.fid_score()
    assert actual_score == pytest.approx(expected_score, abs=1e-7, rel=1e-5)
    np.testing.assert_allclose(calculation.m2, expected_real[0], atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(calculation.s2, expected_real[1], atol=1e-7, rtol=1e-5)
