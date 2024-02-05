import math
from math import sqrt
from random import random
from functools import partial

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
import torch.nn.functional as F

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from denoising_diffusion_pytorch.attend import Attend

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# in paper, they use eps 1e-4 for pixelnorm

def l2norm(t, dim = -1, eps = 1e-12):
    return F.normalize(t, dim = dim, eps = eps)

# small helper modules

def Upsample(dim):
    return nn.Upsample(scale_factor = 2, mode = 'bilinear')

class Downsample(Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = WeightNormedConv2d(dim, dim, 1)
        self.pixel_norm = PixelNorm(dim = 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        assert all([divisible_by(_, 2) for _ in (h, w)])

        x = F.interpolate(x, (h // 2, w // 2), mode = 'bilinear')
        x = self.conv(x)
        x = self.pixel_norm(x)
        return x

# mp activations
# section 2.5

class MPSiLU(Module):
    def forward(self, x):
        return F.silu(x) / 0.596

# gain - layer scaling

class Gain(Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return x * self.gain

# magnitude preserving concat
# equation (103) - default to 0.5, which they recommended

class MPCat(Module):
    def __init__(self, t = 0.5, dim = -1):
        super().__init__()
        self.t = t
        self.dim = dim

    def forward(self, a, b):
        dim, t = self.dim, self.t
        Na, Nb = a.shape[dim], b.shape[dim]

        C = sqrt((Na + Nb) / ((1. - t) ** 2 + t ** 2))

        a = a * (1. - t) / sqrt(Na)
        b = b * t / sqrt(Nb)

        return C * torch.cat((a, b), dim = dim)

# magnitude preserving sum
# equation (88)
# empirically, they found t=0.3 for encoder / decoder / attention residuals
# and for embedding, t=0.5

class MPSum(Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, res):
        a, b, t = x, res, self.t
        num = a * (1. - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den

# pixelnorm
# equation (30)

class PixelNorm(Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        # high epsilon for the pixel norm in the paper
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return l2norm(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])

# forced weight normed conv2d and linear
# algorithm 1 in paper

class WeightNormedConv2d(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        concat_ones_to_input = False   # they use this in the input block to protect against loss of expressivity due to removal of all biases, even though they claim they observed none
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 2
        self.concat_ones_to_input = concat_ones_to_input

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                weight, ps = pack_one(self.weight, 'o *')
                normed_weight = l2norm(weight, eps = self.eps)
                normed_weight = unpack_one(normed_weight, ps, 'o *')
                self.weight.copy_(normed_weight)

        weight = l2norm(self.weight, eps = self.eps) / sqrt(self.fan_in)

        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 0, 0, 1, 0), value = 1.)

        return F.conv2d(x, weight, padding='same')

class WeightNormedLinear(Module):
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = l2norm(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = l2norm(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.linear(x, weight)

# mp fourier embeds

class MPFourierEmbedding(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * (2 ** 0.5)

# building block modules

class Block(Module):
    def __init__(
        self,
        dim,
        dim_out,
        mp_sum_t = 0.3
    ):
        super().__init__()
        self.proj = WeightNormedConv2d(dim, dim_out, 3)
        self.act = MPSiLU()
        self.mp_add = MPSum(t = mp_sum_t)

    def forward(self, x, scale = None):
        res = x

        x = self.proj(x)

        if exists(scale):
            x = x * (scale + 1)

        x = self.act(x)

        return mp_add(x, res)

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim = None
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            MPSiLU(),
            nn.Linear(time_emb_dim, dim_out, bias = False)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = WeightNormedConv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale = None
        if exists(self.mlp) and exists(time_emb):
            scale = self.mlp(time_emb)
            scale = rearrange(scale, 'b c -> b c 1 1')

        h = self.block1(x, scale = scale)

        h = self.block2(h)

        return h + self.res_conv(x)

class CosineSimAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False,
        mp_sum_t = 0.3
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.pixel_norm = PixelNorm(dim = 1)

        # equation (34) - they used cosine sim of queries and keys with a fixed scale of sqrt(Nc)
        self.attend = Attend(flash = flash, scale = dim_head ** 0.5)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = WeightNormedConv2d(dim, hidden_dim * 3, 1)
        self.to_out = WeightNormedConv2d(hidden_dim, dim, 1)

        self.mp_add = MPSum(t = mp_sum_t)

    def forward(self, x):
        res, b, c, h, w = x, *x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        q, k, v = map(self.pixel_norm, (q, k, v))

        q, k = map(l2norm, (q, k))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        return self.mp_add(out, res)

# model

class KarrasUnet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        sinusoidal_dim = 16,
        fourier_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False,
        mp_cat_t = 0.5
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = WeightNormedConv2d(input_channels, init_dim, 7, concat_ones_to_input = True)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = MPFourierEmbedding(sinusoidal_dim)
        fourier_dim = sinusoidal_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim, bias = False),
            nn.GELU(),
            nn.Linear(time_dim, time_dim, bias = False)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        self.mp_cat = MPCat(t = mp_cat_t, dim = 1)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                CosineSimAttention(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads, flash = flash_attn),
                Downsample(dim_in, dim_out) if not is_last else WeightNormedConv2d(dim_in, dim_out, 3)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = CosineSimAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                CosineSimAttention(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads, flash = flash_attn),
                Upsample(dim_out, dim_in) if not is_last else  WeightNormedConv2d(dim_out, dim_in, 3)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = WeightNormedConv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = self.mp_cat(x, h.pop())
            x = block1(x, t)

            x = self.mp_cat(x, h.pop())
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
