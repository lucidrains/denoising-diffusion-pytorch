import copy

import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from denoising_diffusion_pytorch import guided_diffusion as guided


@pytest.mark.parametrize(
    "channels,explicit_mode",
    [
        (1, None),
        (3, None),
        (4, None),
        (1, "L"),
        (3, "RGB"),
        (4, "RGBA"),
        (3, "YCbCr"),
        (2, None),
    ],
)
def test_guided_training_matches_explicit_channel_conversion(
    monkeypatch, tmp_path, channels, explicit_mode
):
    folder = tmp_path / "images"
    folder.mkdir()
    modes = [("L", 177), ("RGB", (55, 121, 200)), ("RGBA", (31, 103, 213, 120))]
    if channels == 2:
        modes = [("LA", (value, 180)) for value in (40, 90, 170)]
    for index, (mode, color) in enumerate(modes):
        Image.new(mode, (4, 4), color).save(folder / f"{index}.png")
    expected_mode = explicit_mode or {1: "L", 3: "RGB", 4: "RGBA"}.get(channels)
    monkeypatch.setattr(guided, "cpu_count", lambda: 0)
    torch.manual_seed(67)
    original = guided.Unet(dim=8, dim_mults=(1,), channels=channels)
    trainers = []
    for label, mode in [("actual", explicit_mode), ("reference", expected_mode)]:
        diffusion = guided.GaussianDiffusion(
            copy.deepcopy(original), image_size=4, timesteps=4, beta_schedule="cosine"
        )
        trainers.append(
            guided.Trainer(
                diffusion,
                str(folder),
                train_batch_size=3,
                train_num_steps=1,
                augment_horizontal_flip=False,
                convert_image_to=mode,
                num_samples=4,
                save_and_sample_every=1000,
                results_folder=str(tmp_path / label),
            )
        )

    for trainer in trainers:
        torch.manual_seed(71)
        trainer.train()

    actual, reference = trainers
    for index, path in enumerate(actual.ds.paths):
        with Image.open(path) as image:
            expected = to_tensor(
                image.convert(expected_mode) if expected_mode else image
            )
        torch.testing.assert_close(actual.ds[index], expected, rtol=0, atol=0)
    for name, parameter in actual.model.state_dict().items():
        assert torch.isfinite(parameter).all()
        torch.testing.assert_close(
            parameter, reference.model.state_dict()[name], rtol=0, atol=0
        )
    assert actual.step == reference.step == 1
    assert any(
        not torch.equal(parameter, original.state_dict()[name])
        for name, parameter in actual.model.model.state_dict().items()
    )
