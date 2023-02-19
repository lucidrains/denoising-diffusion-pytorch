'''Train CIFAR10 with PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tqdm import tqdm

'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from denoising_diffusion_pytorch import denoising_diffusion_pytorch
from model import ResNet18

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LeNet(nn.Module):
    def __init__(self,
                 t_dim=16):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # nn.init.xavier_uniform_(self.conv1.weight,gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.conv2.weight,gain=nn.init.calculate_gain('relu'))
        self.resnet=ResNet18()

        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(256+16, 128)
        self.fc3 = nn.Linear(128, 10)
        self.fc4 = nn.Linear(10, 128)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.activate_function=F.leaky_relu

    def forward(self, x, condition, t):
        # out = self.activate_function(self.conv1(condition))
        # out = F.max_pool2d(out, 2)
        # out = self.activate_function(self.conv2(out))
        # out = F.max_pool2d(out, 2)
        # out = out.view(out.size(0), -1)  # n,25*16
        out1=self.resnet(condition)#n,128



        # out1 = self.activate_function(self.fc1(out))  # n ,16
        out2 = self.activate_function(self.fc4(x.squeeze(1)))  # n ,128
        t = self.time_mlp(t)
        out = torch.concat((out1, out2, t), dim=1)  # n, 240

        out = self.activate_function(self.fc2(out))
        out = self.fc3(out)
        return out


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 diffusion Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

model = LeNet()
from diffusion_agent import Diffusion_agent as Agent
from torch.utils.tensorboard import SummaryWriter
writer =  SummaryWriter("./data")
agent = Agent(device=device,
              model=model,
              lr=args.lr,
              )


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    num_class = 10
    for batch_idx, (inputs, label) in tqdm(enumerate(trainloader)):
        targets = torch.zeros((len(label), 10)).scatter_(1, label.long().reshape(-1, 1), 1).unsqueeze(1).to(device)
        inputs = inputs.to(device)
        loss = agent.train(target=targets,
                                  cifar_10_input=inputs,
                                  # batch_size=args.batch_size,
                                    log_writer=writer
                                  )

def test(epoch):
    global best_acc
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
            condition, targets = inputs.to(device), targets.to(device)
            generateed_picture=agent.sample_action(condition)
            predicted=generateed_picture.squeeze(1).argmax(1)
            total += targets.size(0)
            correct += torch.tensor(predicted).eq(targets.cpu()).sum().item()
            break



    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:

        print('Saving..best_acc:'+f"{acc}")
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

haha=1
for epoch in range(start_epoch, start_epoch + 20000):
    train(epoch)
    if epoch> start_epoch+50 and epoch%50==0:
        test(epoch)
    if haha == 0:
        test(epoch)
    # scheduler.step()
