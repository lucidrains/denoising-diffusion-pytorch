import numpy as np
import pytest
import torch
from torch import nn

import denoising_diffusion_pytorch.fid_evaluation as fid


class LocalFeatures(nn.Module):
    BLOCK_INDEX_BY_DIM = {3: 0}

    def __init__(self, blocks):
        super().__init__()

    def forward(self, samples):
        return (samples.mean(dim=(-2, -1), keepdim=True),)


@pytest.mark.parametrize(
    "sizes, sample_count",
    [((4, 4), 5), ((2, 2, 2), 5), ((7,), 5), ((1, 3, 4), 5), ((4, 4), 8)],
)
def test_real_statistics_use_exact_sample_budget(
    monkeypatch, tmp_path, sizes, sample_count
):
    monkeypatch.setattr(fid, "InceptionV3", LocalFeatures)
    samples = (
        torch.arange(sum(sizes) * 12, dtype=torch.float32).reshape(-1, 3, 2, 2) / 100
    )
    evaluation = fid.FIDEvaluation(
        batch_size=4,
        dl=iter(samples.split(sizes)),
        sampler=nn.Identity(),
        device="cpu",
        num_fid_samples=sample_count,
        stats_dir=tmp_path,
        inception_block_idx=3,
    )
    evaluation.load_or_precalc_dataset_stats()
    reference = samples[:sample_count].mean(dim=(-2, -1)).numpy()
    np.testing.assert_allclose(evaluation.m2, reference.mean(axis=0))
    np.testing.assert_allclose(evaluation.s2, np.cov(reference, rowvar=False))
    with np.load(tmp_path / "dataset_stats.npz") as cached:
        np.testing.assert_array_equal(cached["m2"], evaluation.m2)
        np.testing.assert_array_equal(cached["s2"], evaluation.s2)


def test_short_finite_dataset_remains_usable(monkeypatch, tmp_path):
    monkeypatch.setattr(fid, "InceptionV3", LocalFeatures)
    samples = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2) / 100
    evaluation = fid.FIDEvaluation(
        batch_size=4,
        dl=iter(samples.split(2)),
        sampler=nn.Identity(),
        device="cpu",
        num_fid_samples=9,
        stats_dir=tmp_path,
        inception_block_idx=3,
    )
    evaluation.load_or_precalc_dataset_stats()
    reference = samples.mean(dim=(-2, -1)).numpy()
    np.testing.assert_allclose(evaluation.m2, reference.mean(axis=0))
    np.testing.assert_allclose(evaluation.s2, np.cov(reference, rowvar=False))
