import copy
import importlib
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch import nn


VARIANTS = [
    "denoising_diffusion_pytorch",
    "denoising_diffusion_pytorch_1d",
    "guided_diffusion",
    "repaint",
]


def make_trainer(variant, folder, results, accumulation):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{variant}")
    # Avoid worker subprocesses inside the two distributed test workers.
    module.cpu_count = lambda: 0
    kwargs = dict(
        train_batch_size=16,
        gradient_accumulate_every=accumulation,
        train_num_steps=2,
        save_and_sample_every=100,
        num_samples=1,
        results_folder=str(results),
        ema_update_every=1,
    )
    torch.manual_seed(120)
    if variant == "denoising_diffusion_pytorch_1d":
        model = module.Unet1D(dim=8, dim_mults=(1,), channels=1)
        diffusion = module.GaussianDiffusion1D(model, seq_length=8, timesteps=4)
        dataset = torch.linspace(0.0, 1.0, 128 * 8).reshape(128, 1, 8)
        trainer = module.Trainer1D(diffusion, dataset, **kwargs)
    else:
        model_kwargs = dict(dim=8, dim_mults=(1,), channels=1)
        if variant != "guided_diffusion":
            model_kwargs.update(attn_heads=1, attn_dim_head=8)
        model = module.Unet(**model_kwargs)
        diffusion = module.GaussianDiffusion(
            model, image_size=8, timesteps=4, sampling_timesteps=2, beta_schedule="cosine"
        )
        kwargs.update(augment_horizontal_flip=False, convert_image_to="L")
        if variant != "guided_diffusion":
            kwargs["calculate_fid"] = False
        trainer = module.Trainer(diffusion, str(folder), **kwargs)
    return trainer


def count_and_reduce(state, bucket):
    state.append(bucket.buffer().numel())
    future = dist.all_reduce(bucket.buffer(), async_op=True).get_future()
    return future.then(lambda result: result.value()[0] / dist.get_world_size())


def run_training_case(rank, world_size, variant, accumulation, root):
    torch.set_num_threads(1)
    root = Path(root)
    if world_size > 1:
        os.environ.update(
            RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
            LOCAL_WORLD_SIZE=str(world_size), MASTER_ADDR="127.0.0.1", OMP_NUM_THREADS="1",
        )
        dist.init_process_group(
            "gloo", init_method=(root / "rendezvous").as_uri(),
            rank=rank, world_size=world_size, timeout=timedelta(seconds=90),
        )
    try:
        trainer = make_trainer(variant, root / "images", root / "results", accumulation)
        native_model = trainer.accelerator.unwrap_model(trainer.model)
        reference = copy.deepcopy(native_model)
        initial_parameters = [parameter.detach().clone() for parameter in native_model.parameters()]
        reduction_calls = []
        if world_size > 1:
            assert isinstance(trainer.model, nn.parallel.DistributedDataParallel)
            trainer.model.register_comm_hook(reduction_calls, count_and_reduce)
        else:
            assert not isinstance(trainer.model, nn.parallel.DistributedDataParallel)

        inputs = []
        gradients = []
        parameters = []
        reduction_boundaries = []

        def record_input(module, args):
            inputs.append((args[0].detach().clone(), torch.random.get_rng_state()))

        def record_step(optimizer, args, kwargs):
            gradients.append([parameter.grad.detach().clone() for parameter in native_model.parameters()])
            reduction_boundaries.append(len(reduction_calls))

        def record_parameters(optimizer, args, kwargs):
            parameters.append([parameter.detach().clone() for parameter in native_model.parameters()])

        native_model.register_forward_pre_hook(record_input)
        trainer.opt.optimizer.register_step_pre_hook(record_step)
        trainer.opt.optimizer.register_step_post_hook(record_parameters)
        torch.manual_seed(500 + rank)
        trainer.train()
        assert trainer.step == 2
        assert len(inputs) == 2 * accumulation
        assert len(gradients) == len(parameters) == 2

        optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4, betas=(0.9, 0.99))
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for data, rng in inputs[step * accumulation:(step + 1) * accumulation]:
                torch.random.set_rng_state(rng)
                (reference(data) / accumulation).backward()
            if world_size > 1:
                for parameter in reference.parameters():
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(world_size)
            nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            for actual, parameter in zip(gradients[step], reference.parameters()):
                torch.testing.assert_close(actual, parameter.grad, atol=3e-6, rtol=3e-5)
            optimizer.step()
            for actual, parameter in zip(parameters[step], reference.parameters()):
                torch.testing.assert_close(actual, parameter, atol=3e-6, rtol=3e-5)

        assert any(not torch.equal(before, after) for before, after in zip(initial_parameters, parameters[-1]))
        if world_size > 1:
            # These small models fit into one DDP bucket. Only one reduction is
            # needed per optimizer update, including after DDP rebuilds buckets.
            assert reduction_boundaries == [1, 2], reduction_boundaries
            for parameter in native_model.parameters():
                from_rank_zero = parameter.detach().clone()
                dist.broadcast(from_rank_zero, src=0)
                torch.testing.assert_close(parameter, from_rank_zero, atol=0, rtol=0)
        else:
            assert reduction_boundaries == [0, 0]
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("accumulation", [1, 3])
@pytest.mark.parametrize("world_size", [1, 2])
def test_training_accumulation_reduces_only_at_update_boundary(variant, accumulation, world_size, tmp_path):
    if world_size > 1 and not dist.is_gloo_available():
        pytest.skip("Gloo is required for the native two-process CPU regression")
    folder = tmp_path / "images"
    folder.mkdir()
    for index in range(128):
        Image.new("L", (8, 8), color=index * 2).save(folder / f"{index}.png")
    mp.spawn(
        run_training_case,
        args=(world_size, variant, accumulation, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
