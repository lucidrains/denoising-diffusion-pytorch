import argparse
from denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import numpy as np
import random
import os

def set_seed(seed):
   torch.manual_seed(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   np.random.seed(seed)
   random.seed(seed)


def run_training(rank,world_size,seed,folder,batch_size,num_train_steps,img_size):
   is_main = rank == 0
   is_dpp = world_size > 1
   if is_dpp:
      set_seed(seed)
      torch.cuda.set_device(rank)
      os.environ['MASTER_ADDR'] = 'localhost'
      os.environ['MASTER_PORT'] = '12355'
      dist.init_process_group('nccl', rank=rank, world_size=world_size)

   model = Unet(channels=3,dim=128,dim_mults = (1,1,2,4)).cuda(rank)
   diffusion = GaussianDiffusion(model,image_size=img_size,timesteps = 500, loss_type = 'l2', beta_schedule='cosine').cuda(rank)
   trainer = Trainer(diffusion,folder,  train_batch_size = batch_size, amp = True, train_lr = 1e-4, train_num_steps=num_train_steps, gradient_accumulate_every = 4, ema_decay = 0.995, rank = rank, world_size = world_size)
   print("Start trainer")
   trainer.train()

if __name__ == "__main__":
  world_size = torch.cuda.device_count()
  print('World size: ',world_size)
  parser = argparse.ArgumentParser(description="Train Denoising Diffusion")
  parser.add_argument('--seed', type=int, default=9900, help='Random seed number')
  parser.add_argument('--batchsize', type=int, default=16, help='Batch size')
  parser.add_argument('--folder', type=str, help='Folder of images', required=True)
  parser.add_argument('--steps', type=int, default=700000, help='Number of training iterations')
  parser.add_argument('--imgsize', type=int, default=256, help='Number of training iterations')

  args = parser.parse_args()
  if (world_size == 1):
      run_training(0,1,args.seed,args.folder,args.batchsize,args.steps,args.imgsize)
  else:
      mp.spawn(run_training,args=(world_size,args.seed,args.folder,args.batchsize,args.steps,args.imgsize),
               nprocs=world_size,
               join=True)
