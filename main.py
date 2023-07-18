import time
import wandb
import argparse
import pickle
import torch
import os 
import pandas as pd

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import data
import paths

def main(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'

    device = torch.device(device_name)
    print(f"Device : {device_name}")

    train_setting = f'seed_{args.seed}_sampling_method_{args.sampling_method}_num_samples_{args.num_samples}-\
                      diffusion_time_steps_{args.diffusion_time_steps}-train_num_steps_{args.train_num_steps}'
    if not args.ignore_wandb:
        wandb.init(project='check_time',
                   entity='ppg-diffusion')
        wandb_run_name = train_setting
        wandb.run.name = wandb_run_name


    train_set_root = paths.TRAINSET_ROOT
    train_set_name = f'sampling_method_{args.sampling_method}-num_samples_{args.num_samples}'
    # TODO: 다운로드 다 되면 paths에서 관리
    training_seq = data.get_data(sampling_method=args.sampling_method,
                                 num_samples=args.num_samples,
                                 data_root=paths.DATA_ROOT)
    os.makedirs(train_set_root, exist_ok=True)
    with open(os.path.join(train_set_root, train_set_name), 'wb') as f:
        pickle.dump(training_seq, f)

    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 1000,
        timesteps = args.diffusion_time_steps,
        objective = 'pred_v'
    )

    dataset = Dataset1D(training_seq)
    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = args.train_num_steps,          # total training steps
        gradient_accumulate_every = 2,                   # gradient accumulation steps
        ema_decay = 0.995,                               # exponential moving average decay
        amp = True,                                      # turn on mixed precision
    )

    if not args.sample_only:
        train_start_time = time.time()
        trainer.train()
        train_time = time.time() - train_start_time
        if not args.ignore_wandb:
            wandb.log({'train_time': train_time})

    # TODO: 가중치 불러오는 코드 짜기
    sampling_root = paths.SAMPLING_ROOT
    sampling_dir = train_setting + f'_sampling_batch_size_{args.sampling_batch_size}'
    sample_start_time = time.time()
    sampled_seq = diffusion.sample(batch_size=args.sampling_batch_size)
    sample_time = time.time() - sample_start_time
    if not args.ignore_wandb:
        wandb.log({'sample_time': sample_time})
    os.makedirs(sampling_dir, exist_ok=True)
    with open(f'{sampling_dir}/sample.pkl', 'wb') as f:
        pickle.dump(sampled_seq, f)

    # TODO: sampled_seq 하나하나 분리해서 plot으로 변환 후 저장하는 코드 작성

if __name__ == '__main__':
    ## COMMON --------------------------------------------------
    parser = argparse.ArgumentParser(description="gp-regression for the confirmation and dead prediction")
    parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")

    ## DATA ----------------------------------------------------
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sampling_method", type=str, default='first_k')

    ## Training ------------------------------------------------
    parser.add_argument("--diffusion_time_steps", type=int, default=1000)
    parser.add_argument("--train_num_steps", type=int, default=5000)
    parser.add_argument("--init_lr", type=float, default=0.1)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--tolerance", type=int, default=500)
    parser.add_argument("--eval_only", action='store_true',
        help = "Stop using wandb (Default : False)")

    ## Sampling
    parser.add_argument("--sample_only", action='store_true',
        help = "Stop using wandb (Default : False)")
    parser.add_argument("--sampling_batch_size", type=int, default=16)

    args = parser.parse_args()

    main(args)