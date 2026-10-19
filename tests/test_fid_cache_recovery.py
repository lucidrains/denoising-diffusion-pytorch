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
    'cache_kind',
    ['missing', 'valid', 'truncated', 'empty', 'missing_mean', 'missing_covariance', 'text'],
)
def test_dataset_stats_cache_recovery(tmp_path, monkeypatch, cache_kind):
    monkeypatch.setattr(fid, 'InceptionV3', TinyInception)
    path = tmp_path / 'dataset_stats.npz'
    cached_mean = np.full(3, 0.25)
    cached_covariance = np.eye(3)
    if cache_kind in ['valid', 'truncated']:
        np.savez_compressed(path, m2=cached_mean, s2=cached_covariance)
        if cache_kind == 'truncated':
            data = path.read_bytes()
            path.write_bytes(data[:len(data) // 2])
    elif cache_kind == 'empty':
        path.write_bytes(b'')
    elif cache_kind == 'text':
        path.write_bytes(b'interrupted statistics archive')
    elif cache_kind == 'missing_mean':
        np.savez_compressed(path, s2=cached_covariance)
    elif cache_kind == 'missing_covariance':
        np.savez_compressed(path, m2=cached_mean)

    original_load = np.load
    opened_archives = []

    def load(*args, **kwargs):
        archive = original_load(*args, **kwargs)
        opened_archives.append(archive)
        return archive

    monkeypatch.setattr(fid.np, 'load', load)
    images = torch.arange(8, dtype=torch.float32)[:, None, None, None].repeat(1, 3, 2, 2) / 10
    evaluator = fid.FIDEvaluation(
        batch_size=4,
        dl=iter(images.split(4)),
        sampler=nn.Identity(),
        device='cpu',
        stats_dir=str(tmp_path),
        num_fid_samples=8,
        inception_block_idx=64,
    )
    evaluator.load_or_precalc_dataset_stats()

    if cache_kind == 'valid':
        np.testing.assert_allclose(evaluator.m2, cached_mean)
        np.testing.assert_allclose(evaluator.s2, cached_covariance)
        assert evaluator.inception_v3.batch_sizes == []
    else:
        features = images.mean(dim=(-2, -1)).numpy()
        np.testing.assert_allclose(evaluator.m2, features.mean(axis=0))
        np.testing.assert_allclose(evaluator.s2, np.cov(features, rowvar=False))
        assert evaluator.inception_v3.batch_sizes == [4, 4]
    assert evaluator.dataset_stats_loaded
    assert all(archive.zip is None for archive in opened_archives)
    with original_load(path) as cache:
        np.testing.assert_allclose(cache['m2'], evaluator.m2)
        np.testing.assert_allclose(cache['s2'], evaluator.s2)
