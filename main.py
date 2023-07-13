import argparse
import pickle
import torch
import os 
import pandas as pd
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

def main(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'

    device = torch.device(device_name)
    print(f"Device : {device_name}")

    # TODO: 다운로드 다 되면 paths에서 관리
    data_root = '/data1/data_ETRI/p09/p093486'
    col_names = ['time', 'PPG', 'abp']

    sample_list = []
    for sample_path in os.listdir(data_root):
        sample = pd.read_csv(f'{data_root}/{sample_path}', names=col_names)
        sample_tensor = torch.tensor(sample['PPG'].values)
        sample_list.append(sample_tensor)
    
    training_seq = torch.stack(sample_list).unsqueeze(1).half()
    
    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 1000,
        timesteps = 1000,
        objective = 'pred_v'
    )

    dataset = Dataset1D(training_seq)
    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 10000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
    )
    trainer.train()

    sampled_seq = diffusion.sample(batch_size = 16)
    with open('sample2.pkl', 'wb') as f:
        pickle.dump(sampled_seq, f)

if __name__ == '__main__':
    ## COMMON --------------------------------------------------
    parser = argparse.ArgumentParser(description="gp-regression for the confirmation and dead prediction")
    parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")

    ## DATA ----------------------------------------------------
    ## Training ------------------------------------------------
    parser.add_argument("--max_epoch", type=int, default=5000)
    parser.add_argument("--init_lr", type=float, default=0.1)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--tolerance", type=int, default=500)
    parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()


    main(args)