import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset

VARIANTS = (
    "denoising_diffusion_pytorch",
    "denoising_diffusion_pytorch_1d",
    "guided_diffusion",
    "repaint",
)


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


def distributed_worker(output):
    records = {}
    for name in VARIANTS:
        module = importlib.import_module(f"denoising_diffusion_pytorch.{name}")
        module.cpu_count = lambda: 0
        sequence = name.endswith("_1d")
        if not sequence:
            # Keep real DataLoader sharding; replace only image-file I/O.
            module.Dataset = ImageSamples
        for split in (False, True):
            torch.manual_seed(41)
            model = ConvolutionalDenoiser(sequence)
            if sequence:
                diffusion = module.GaussianDiffusion1D(model, seq_length=8, timesteps=4)
                dataset = module.Dataset1D(
                    torch.arange(128).float().reshape(128, 1, 1).expand(-1, -1, 8) / 127
                )
                trainer_class, data = module.Trainer1D, dataset
            else:
                diffusion = module.GaussianDiffusion(
                    model, image_size=4, timesteps=4, beta_schedule="cosine"
                )
                trainer_class, data = module.Trainer, str(output)
            extra = {"calculate_fid": False} if name in (VARIANTS[0], "repaint") else {}
            trainer = trainer_class(
                diffusion,
                data,
                train_batch_size=16,
                train_num_steps=1,
                num_samples=4,
                save_and_sample_every=1000,
                results_folder=str(output / f"{name}-{split}"),
                split_batches=split,
                **extra,
            )
            batch = next(trainer.dl)
            ids = (batch.flatten(1)[:, 0] * 127).round().int().tolist()
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, ids)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            trainer.train()
            records[f"{name}-{split}"] = {
                "rank": dist.get_rank(),
                "world_size": dist.get_world_size(),
                "backend": dist.get_backend(),
                "native_ddp": isinstance(trainer.model, DistributedDataParallel),
                "local_ids": ids,
                "global_ids": sum(gathered, []),
                "step": trainer.step,
                "changed": any(
                    not torch.equal(old, new)
                    for old, new in zip(before, model.parameters())
                ),
                "finite": all(
                    torch.isfinite(p).all().item() for p in model.parameters()
                ),
                "parameters": [
                    p.detach().flatten().tolist() for p in model.parameters()
                ],
            }
            del trainer
    (output / f"rank-{dist.get_rank()}.json").write_text(json.dumps(records))
    dist.barrier()
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def distributed_results(tmp_path_factory):
    output = tmp_path_factory.mktemp("split-batches")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(ACCELERATE_USE_CPU="true", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).resolve()),
            "--distributed-worker",
            str(output),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(2)]


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("split", [False, True])
def test_native_distributed_training_respects_split_batches(
    distributed_results, name, split
):
    rows = [rank[f"{name}-{split}"] for rank in distributed_results]
    expected_local = 8 if split else 16
    for row in rows:
        assert row["world_size"] == 2 and row["backend"] == "gloo"
        assert row["native_ddp"] and row["step"] == 1
        assert row["changed"] and row["finite"]
        assert len(row["local_ids"]) == expected_local
        assert len(row["global_ids"]) == expected_local * 2
        assert len(set(row["global_ids"])) == expected_local * 2
    assert rows[0]["parameters"] == rows[1]["parameters"]


if __name__ == "__main__":
    assert sys.argv[1] == "--distributed-worker"
    distributed_worker(Path(sys.argv[2]))
