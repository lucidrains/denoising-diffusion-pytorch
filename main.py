import os
import time
import os.path
import pickle
from PIL import Image
import numpy as np
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from tqdm.auto import tqdm

import torch
import torchvision
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision.datasets.utils import check_integrity
from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


from ema_pytorch import EMA
from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.version import __version__
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

import argparse

parser = argparse.ArgumentParser(description="Denoising Diffusion Probabilistic Models")

parser.add_argument("--image_size", type=int, default=32, help="Image size")
parser.add_argument("--timestep_respacing", type=str, default="1000", help="Timestep respacing")
parser.add_argument("--num_channels", type=int, default=3, help="Number of channels")
parser.add_argument("--num_fid_samples", type=int, default=10000, help="Number of FID samples")
parser.add_argument("--train_batch_size", type=int, default=512, help="Training batch size")
parser.add_argument("--train_num_steps", type=int, default=10000, help="Number of training steps")
parser.add_argument("--gradient_accumulate_every", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--ema_decay", type=float, default=0.995, help="EMA decay")
parser.add_argument("--model_dim", type=int, default=64, help="Model dimension")
parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate")
parser.add_argument("--timesteps", type=int, default=1000, help="Number of timesteps")

args = parser.parse_args()


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer, denom):
    return (numer % denom) == 0


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class Custom_Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=["jpg", "jpeg", "png", "tiff"],
        augment_horizontal_flip=False,
        convert_image_to=None,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]

        maybe_convert_fn = (
            partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        )

        self.transform = T.Compose(
            [
                T.Lambda(maybe_convert_fn),
                T.Resize(image_size),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class ImageNetDS(Custom_Dataset):
    """

    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    base_folder = ""
    train_list = [
        ["train_data_batch_1", ""],
        ["train_data_batch_2", ""],
        ["train_data_batch_3", ""],
        ["train_data_batch_4", ""],
        ["train_data_batch_5", ""],
        ["train_data_batch_6", ""],
        ["train_data_batch_7", ""],
        ["train_data_batch_8", ""],
        ["train_data_batch_9", ""],
        ["train_data_batch_10", ""],
    ]

    test_list = [
        ["val_data", ""],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)
        self.label_map = self._load_label_map()
        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, "rb") as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry["data"])
                    self.train_labels += [label - 1 for label in entry["labels"]]
                    self.mean = entry["mean"]

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, img_size, img_size))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, "rb")
            entry = pickle.load(fo)
            self.test_data = entry["data"]
            self.test_labels = [label - 1 for label in entry["labels"]]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, img_size, img_size))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target_name = self.label_map[target]
        # print(f"Index: {index}, Label: {target}, Name: {target_name}")  # 디버깅을 위한 출력

        return img

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def _load_label_map(self):
        label_map = {}
        synset_file = os.path.join(self.root, "map_clsloc.txt")
        with open(synset_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) == 3:
                    label_map[int(parts[1]) - 1] = parts[2]
                    # print(f"Mapping: {int(parts[1]) - 1} -> {parts[2]}")  # 디버깅을 위한 출력
        return label_map


datetime = time.strftime("%Y%m%d")
if not os.path.exists(f"./results/{datetime}"):
    os.makedirs(f"./results/{datetime}")


def train(trainset):
    model = Unet(dim=args.model_dim, dim_mults=(1, 2, 4, 8), flash_attn=False)

    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,  # number of steps
        sampling_timesteps=500,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        trainset,
        train_batch_size=args.train_batch_size,
        train_lr=args.lr,  # learning rate
        train_num_steps=args.train_num_steps,  # total training steps
        # gradient_accumulate_every=args.gradient_accumulate_every,  # gradient accumulation steps
        ema_decay=args.ema_decay,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=True,  # whether to calculate fid during training
        num_fid_samples=args.num_fid_samples,  # number of samples for calculating fid
        results_folder=f"./results/{datetime}",
    )

    trainer.train()


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    trainset = ImageNetDS(root="../ImageNet32", img_size=32, train=True, transform=transform)
    train(trainset)
