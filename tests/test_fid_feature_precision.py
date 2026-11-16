from contextlib import nullcontext

import numpy as np
import pytest
import torch
import torchvision
from pytorch_fid import inception
from pytorch_fid.fid_score import calculate_frechet_distance
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d

from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation


class ReplaySampler(nn.Module):
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.offset = 0

    def sample(self, batch_size):
        result = self.images[self.offset : self.offset + batch_size]
        self.offset += batch_size
        return result


@pytest.fixture
def evaluator(monkeypatch, tmp_path):
    # Exercise the native Inception wrapper and first convolutional block.
    # Random weights suffice for dtype behavior and avoid a network download.
    monkeypatch.setattr(
        inception,
        "fid_inception_v3",
        lambda: torchvision.models.inception_v3(
            weights=None, aux_logits=False, init_weights=False
        ),
    )
    torch.manual_seed(83)
    return FIDEvaluation(
        batch_size=2,
        dl=iter(()),
        sampler=ReplaySampler(torch.empty(0)),
        stats_dir=str(tmp_path),
        device="cpu",
        num_fid_samples=4,
        inception_block_idx=64,
    )


def reference_features(evaluator, images):
    images = images.float()
    if evaluator.channels == 1:
        images = images.repeat(1, 3, 1, 1)
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        features = evaluator.inception_v3.eval()(images)[0]
        return adaptive_avg_pool2d(features, 1).flatten(1)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("autocast", [False, True])
def test_features_use_inception_precision_without_changing_input(
    evaluator, dtype, channels, autocast
):
    evaluator.channels = channels
    images = (
        torch.linspace(0.05, 0.95, 2 * channels * 32 * 32)
        .reshape(2, channels, 32, 32)
        .to(dtype)
    )
    original = images.clone()
    expected = reference_features(evaluator, images)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
    with context:
        actual = evaluator.calculate_inception_features(images)
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert images.dtype == dtype
    torch.testing.assert_close(images, original, rtol=0, atol=0)


@pytest.mark.parametrize("sample_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_complete_fid_evaluation_inside_autocast_matches_float32_reference(
    evaluator, sample_dtype
):
    real = torch.linspace(0.0, 1.0, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)
    fake = real.flip(-1).mul(0.8).add(0.1).to(sample_dtype)
    evaluator.dl = iter(real.split(2))
    evaluator.sampler = ReplaySampler(fake)
    real_features = torch.cat(
        [reference_features(evaluator, batch) for batch in real.split(2)]
    ).numpy()
    fake_features = torch.cat(
        [reference_features(evaluator, batch) for batch in fake.split(2)]
    ).numpy()
    expected = calculate_frechet_distance(
        fake_features.mean(0),
        np.cov(fake_features, rowvar=False),
        real_features.mean(0),
        np.cov(real_features, rowvar=False),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = evaluator.fid_score()
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=1e-8, abs=1e-8)
    assert evaluator.sampler.offset == 4
