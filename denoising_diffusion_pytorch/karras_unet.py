"""
the magnitude-preserving unet proposed in https://arxiv.org/abs/2312.02696 by Karras et al.
"""

import math
from math import sqrt, ceil
from functools import partial

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack

from denoising_diffusion_pytorch.attend import Attend

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def xnor(x, y):
    return not (x ^ y)

def append(arr, el):
    arr.append(el)

def prepend(arr, el):
    arr.insert(0, el)

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

class MPAdd(Module):
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

def normalize_weight(weight, eps = 1e-4):
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps = eps)
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')

class Conv2d(Module):
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
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)

        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 0, 0, 1, 0), value = 1.)

        return F.conv2d(x, weight, padding='same')

class Linear(Module):
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
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
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * sqrt(2)

# building block modules

class Encoder(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        attn_flash = False,
        downsample = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.downsample = downsample
        self.downsample_conv = None

        curr_dim = dim
        if downsample:
            self.downsample_conv = Conv2d(curr_dim, dim_out, 1)
            curr_dim = dim_out

        self.pixel_norm = PixelNorm(dim = 1)

        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv2d(curr_dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv2d(dim_out, dim_out, 3)
        )

        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        if has_attn:
            self.attn = Attention(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

    def forward(
        self,
        x,
        emb = None
    ):
        if self.downsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h // 2, w // 2), mode = 'bilinear')
            x = self.downsample_conv(x)

        x = self.pixel_norm(x)

        res = x.clone()

        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1')

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x

class Decoder(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        attn_flash = False,
        upsample = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.upsample = upsample
        self.needs_skip = not upsample

        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv2d(dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv2d(dim_out, dim_out, 3)
        )

        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        if has_attn:
            self.attn = Attention(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

    def forward(
        self,
        x,
        emb = None
    ):
        if self.upsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h * 2, w * 2), mode = 'bilinear')

        res = self.res_conv(x)

        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1')

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        num_mem_kv = 4,
        flash = False,
        mp_add_t = 0.3
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.pixel_norm = PixelNorm(dim = -1)

        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1)
        self.to_out = Conv2d(hidden_dim, dim, 1)

        self.mp_add = MPAdd(t = mp_add_t)

    def forward(self, x):
        res, b, c, h, w = x, *x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        q, k, v = map(self.pixel_norm, (q, k, v))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        return self.mp_add(out, res)

# unet proposed by karras
# bias-less, no group-norms, with magnitude preserving operations

class KarrasUnet(Module):
    """
    going by figure 21. config G
    """

    def __init__(
        self,
        *,
        image_size,
        dim = 192,
        dim_max = 768,            # channels will double every downsample and cap out to this value
        num_classes = None,       # in paper, they do 1000 classes for a popular benchmark
        channels = 4,             # 4 channels in paper for some reason, must be alpha channel?
        num_downsamples = 3,
        num_blocks_per_stage = 4,
        attn_res = (16, 8),
        fourier_dim = 16,
        attn_dim_head = 64,
        attn_flash = False,
        mp_cat_t = 0.5,
        mp_add_emb_t = 0.5,
        attn_res_mp_add_t = 0.3,
        resnet_mp_add_t = 0.3,
        dropout = 0.1,
        self_condition = False
    ):
        super().__init__()

        self.self_condition = self_condition

        # determine dimensions

        self.channels = channels
        self.image_size = image_size
        input_channels = channels * (2 if self_condition else 1)

        # input and output blocks

        self.input_block = Conv2d(input_channels, dim, 3, concat_ones_to_input = True)

        self.output_block = nn.Sequential(
            Conv2d(dim, channels, 3),
            Gain()
        )

        # time embedding

        emb_dim = dim * 4

        self.to_time_emb = nn.Sequential(
            MPFourierEmbedding(fourier_dim),
            Linear(fourier_dim, emb_dim)
        )

        # class embedding

        self.needs_class_labels = exists(num_classes)
        self.num_classes = num_classes

        if self.needs_class_labels:
            self.to_class_emb = Linear(num_classes, 4 * dim)
            self.add_class_emb = MPAdd(t = mp_add_emb_t)

        # final embedding activations

        self.emb_activation = MPSiLU()

        # number of downsamples

        self.num_downsamples = num_downsamples

        # attention

        attn_res = set(cast_tuple(attn_res))

        # resnet block

        block_kwargs = dict(
            dropout = dropout,
            emb_dim = emb_dim,
            attn_dim_head = attn_dim_head,
            attn_res_mp_add_t = attn_res_mp_add_t,
            attn_flash = attn_flash
        )

        # unet encoder and decoders

        self.downs = ModuleList([])
        self.ups = ModuleList([])

        curr_dim = dim
        curr_res = image_size

        self.skip_mp_cat = MPCat(t = mp_cat_t, dim = 1)

        # take care of skip connection for initial input block and first three encoder blocks

        prepend(self.ups, Decoder(dim * 2, dim, **block_kwargs))

        assert num_blocks_per_stage >= 1

        for _ in range(num_blocks_per_stage):
            enc = Encoder(curr_dim, curr_dim, **block_kwargs)
            dec = Decoder(curr_dim * 2, curr_dim, **block_kwargs)

            append(self.downs, enc)
            prepend(self.ups, dec)

        # stages

        for _ in range(self.num_downsamples):
            dim_out = min(dim_max, curr_dim * 2)
            upsample = Decoder(dim_out, curr_dim, has_attn = curr_res in attn_res, upsample = True, **block_kwargs)

            curr_res //= 2
            has_attn = curr_res in attn_res

            downsample = Encoder(curr_dim, dim_out, downsample = True, has_attn = has_attn, **block_kwargs)

            append(self.downs, downsample)
            prepend(self.ups, upsample)
            prepend(self.ups, Decoder(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs))

            for _ in range(num_blocks_per_stage):
                enc = Encoder(dim_out, dim_out, has_attn = has_attn, **block_kwargs)
                dec = Decoder(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs)

                append(self.downs, enc)
                prepend(self.ups, dec)

            curr_dim = dim_out

        # take care of the two middle decoders

        mid_has_attn = curr_res in attn_res

        self.mids = ModuleList([
            Decoder(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
            Decoder(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
        ])

        self.out_dim = channels

    @property
    def downsample_factor(self):
        return 2 ** self.num_downsamples

    def forward(
        self,
        x,
        time,
        self_cond = None,
        class_labels = None
    ):
        # validate image shape

        assert x.shape[1:] == (self.channels, self.image_size, self.image_size)

        # self conditioning

        if self.self_condition:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim = 1)
        else:
            assert not exists(self_cond)

        # time condition

        time_emb = self.to_time_emb(time)

        # class condition

        assert xnor(exists(class_labels), self.needs_class_labels)

        if self.needs_class_labels:
            if class_labels.dtype in (torch.int, torch.long):
                class_labels = F.one_hot(class_labels, self.num_classes)

            assert class_labels.shape[-1] == self.num_classes
            class_labels = class_labels.float() * sqrt(self.num_classes)

            class_emb = self.to_class_emb(class_labels)

            time_emb = self.add_class_emb(time_emb, class_emb)

        # final mp-silu for embedding

        emb = self.emb_activation(time_emb)

        # skip connections

        skips = []

        # input block

        x = self.input_block(x)

        skips.append(x)

        # down

        for encoder in self.downs:
            x = encoder(x, emb = emb)
            skips.append(x)

        # mid

        for decoder in self.mids:
            x = decoder(x, emb = emb)

        # up

        for decoder in self.ups:
            if decoder.needs_skip:
                skip = skips.pop()
                x = self.skip_mp_cat(x, skip)

            x = decoder(x, emb = emb)

        # output block

        return self.output_block(x)

# improvised MP Transformer

class MPFeedForward(Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        mp_add_t = 0.3
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            PixelNorm(dim = 1),
            Conv2d(dim, dim_inner, 1),
            MPSiLU(),
            Conv2d(dim_inner, dim, 1)
        )

        self.mp_add = MPAdd(t = mp_add_t)

    def forward(self, x):
        res = x
        out = self.net(x)
        return self.mp_add(out, res)

class MPImageTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_mem_kv = 4,
        ff_mult = 4,
        attn_flash = False,
        residual_mp_add_t = 0.3
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, num_mem_kv = num_mem_kv, flash = attn_flash, mp_add_t = residual_mp_add_t),
                MPFeedForward(dim = dim, mult = ff_mult, mp_add_t = residual_mp_add_t)
            ]))

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x

# works best with inverse square root decay schedule

def InvSqrtDecayLRSched(
    optimizer,
    t_ref = 70000,
    sigma_ref = 0.01
):
    """
    refer to equation 67 and Table1
    """
    def inv_sqrt_decay_fn(t: int):
        return sigma_ref / sqrt(max(t / t_ref, 1.))

    return LambdaLR(optimizer, lr_lambda = inv_sqrt_decay_fn)

# example

if __name__ == '__main__':
    unet = KarrasUnet(
        image_size = 64,
        dim = 192,
        dim_max = 768,
        num_classes = 1000,
    )

    images = torch.randn(2, 4, 64, 64)

    denoised_images = unet(
        images,
        time = torch.ones(2,),
        class_labels = torch.randint(0, 1000, (2,))
    )

    assert denoised_images.shape == images.shape
