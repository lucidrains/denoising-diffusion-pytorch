import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion

class Diffusion_agent(object):
    def __init__(self,
             device,
             model,
             step_start_ema=1000,
             lr=3e-4,
             # lr_decay=False,
             grad_norm=1.0,
             ):
        self.model = model
        self.diffusion=GaussianDiffusion(model=self.model,image_size=(1,10),timesteps=1000,loss_type="l2",).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        self.step = 0
        # self.step_start_ema = step_start_ema
        # if lr_decay:
        #     self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=lr_maxt, eta_min=0.)

        self.device = device

    def train(self, target,cifar_10_input,  log_writer=None):# batch_size=128,
        bc_loss = self.diffusion.forward(target, cifar_10_input)
        self.optimizer.zero_grad()
        bc_loss.backward()
        if self.grad_norm > 0:
            grad_norms = nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.optimizer.step()


        """ Log """
        if log_writer is not None:
            self.step+=1
            if self.grad_norm > 0:
                log_writer.add_scalar('Grad Norm', grad_norms.max().item(), self.step)
            log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)


        # if self.lr_decay:
        #     self.actor_lr_scheduler.step()
        #     self.critic_lr_scheduler.step()

        return bc_loss.item()

    def sample_action(self, condition):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        # condition=condition.to(self.device)
        with torch.no_grad():
            generated_probablity = self.diffusion.sample(condition,batch_size=condition.shape[0])

        return generated_probablity.cpu().data.numpy()

    # def save_model(self, dir, id=None):
    #     if id is not None:
    #         torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
    #         torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
    #     else:
    #         torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
    #         torch.save(self.critic.state_dict(), f'{dir}/critic.pth')
    #
    # def load_model(self, dir, id=None):
    #     if id is not None:
    #         self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
    #         self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
    #     else:
    #         self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
    #         self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))