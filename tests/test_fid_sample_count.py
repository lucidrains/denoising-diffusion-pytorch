import numpy as np
import pytest
import torch
from torch import nn

import denoising_diffusion_pytorch.fid_evaluation as fid


class TinyInception(nn.Module):
    BLOCK_INDEX_BY_DIM = {64: 0}

    def __init__(self, blocks):
        super().__init__()
        self.batch_sizes = []

    def forward(self, samples):
        self.batch_sizes.append(len(samples))
        return [samples.mean(dim=(-2, -1), keepdim=True)]


@pytest.mark.parametrize(
    ('sample_count', 'real_batch_size'),
    [(5, 4), (5, 2), (5, 8), (8, 4), (4, 4), (7, 3), (12, 4)],
)
def test_real_statistics_respect_sample_count(
    tmp_path, monkeypatch, sample_count, real_batch_size
):
    monkeypatch.setattr(fid, 'InceptionV3', TinyInception)
    values = torch.arange(8, dtype=torch.float32) / 10
    images = values[:, None, None, None].repeat(1, 3, 2, 2)
    batches = iter(images.split(real_batch_size))
    evaluator = fid.FIDEvaluation(
        batch_size=4,
        dl=batches,
        sampler=nn.Identity(),
        device='cpu',
        stats_dir=str(tmp_path),
        num_fid_samples=sample_count,
        inception_block_idx=64,
    )
    evaluator.load_or_precalc_dataset_stats()

    expected_count = min(sample_count, len(images))
    expected_features = images[:expected_count].mean(dim=(-2, -1)).numpy()
    np.testing.assert_allclose(evaluator.m2, expected_features.mean(axis=0))
    np.testing.assert_allclose(evaluator.s2, np.cov(expected_features, rowvar=False))
    assert sum(evaluator.inception_v3.batch_sizes) == expected_count
    assert evaluator.dataset_stats_loaded
    with np.load(tmp_path / 'dataset_stats.npz') as cache:
        np.testing.assert_allclose(cache['m2'], evaluator.m2)
        np.testing.assert_allclose(cache['s2'], evaluator.s2)
    if sample_count == 4 and real_batch_size == 4:
        assert torch.equal(next(batches), images[4:])


@pytest.mark.parametrize('sample_count', (0, -1))
def test_nonpositive_sample_count_still_fails(tmp_path, monkeypatch, sample_count):
    monkeypatch.setattr(fid, 'InceptionV3', TinyInception)
    images = torch.ones(4, 3, 2, 2)
    evaluator = fid.FIDEvaluation(
        batch_size=4,
        dl=iter([images]),
        sampler=nn.Identity(),
        device='cpu',
        stats_dir=str(tmp_path),
        num_fid_samples=sample_count,
        inception_block_idx=64,
    )

    with pytest.raises((ValueError, RuntimeError)):
        evaluator.load_or_precalc_dataset_stats()
    assert not evaluator.dataset_stats_loaded
    assert not (tmp_path / 'dataset_stats.npz').exists()


def test_empty_batches_fail_without_unbounded_iteration(tmp_path, monkeypatch):
    monkeypatch.setattr(fid, 'InceptionV3', TinyInception)

    def batches():
        yield torch.empty(0, 3, 2, 2)
        raise AssertionError('an empty batch must fail before another read')

    evaluator = fid.FIDEvaluation(
        batch_size=4,
        dl=batches(),
        sampler=nn.Identity(),
        device='cpu',
        stats_dir=str(tmp_path),
        num_fid_samples=5,
        inception_block_idx=64,
    )
    with pytest.raises(ValueError, match='empty batch'):
        evaluator.load_or_precalc_dataset_stats()
    assert not evaluator.dataset_stats_loaded
    assert evaluator.inception_v3.batch_sizes == []
