import torch
from torch.distributions.normal import Normal
from torch import nn, einsum, Tensor
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import default, partial, ResnetBlock, RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb, Residual, PreNorm, Downsample, Upsample, LinearAttention, Attention
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# model
class DenseUnitNorm(nn.Module):
    def __init__(self, features, embedding_dim, seq_length):  # 2 512 125
        # TODO : to device
        super(DenseUnitNorm, self).__init__()
        self.linear_mean = nn.Linear(features, embedding_dim)
        self.linear_mean2 = nn.Linear(1, seq_length)
        self.linear_mean2 = spectral_norm(self.linear_mean2)
        self.linear_var = nn.Linear(features, embedding_dim)
        self.linear_var2 = nn.Linear(1, seq_length)
        self.linear_var2 = spectral_norm(self.linear_var2)

    def forward(self, x):  # x shape: [32, 2]
        mean = self.linear_mean(x) # 32 512
        var = self.linear_var(x) # 32 512
        
        mean = mean.unsqueeze(-1)
        var = var.unsqueeze(-1)
        
        mean = self.linear_mean2(mean)
        var = self.linear_var2(var)


        return mean, var
    
def sample_norm(mean, variance):
    stddev = torch.sqrt(variance)
    dist = Normal(0, 1)
    sample = dist.sample()
    sample = sample*mean + stddev
    return sample

class RegressorGuidanceUnet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = True, # 기존 False
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions
        seq_length = 1000
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        
        # self.Regressor = MLPRegressor(input_size=seq_length, hidden_size=128, num_layers=3, output_size=2)
        self.Regressor = MLPRegressorConv(input_size=1, hidden_channels=128, num_layers=3, output_channels=2, seq_length=1000)
        self.DenseUnit = DenseUnitNorm(2, 512, 125) # 하드코딩 수정 필요
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.z_mean = nn.Linear(mid_dim, mid_dim)
        self.z_variance = nn.Linear(mid_dim, mid_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels # * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        # TODO 0 -> t
        r = self.Regressor(x, 0)
        pz_mu, pz_sig = self.DenseUnit(r)
        
        x = self.init_conv(x)
        ori_r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        # TODO : pz_mean distance loss
        #        
        x = self.mid_block2(x, t) # 32 512 125
        
        # added -> embedding to mu / sigma
        x_permuted = x.permute(0, 2, 1)
        z_mu = self.z_mean(x_permuted)
        z_log_var = self.z_variance(x_permuted)

        z_mu = z_mu.permute(0, 2, 1)
        z_log_var = z_log_var.permute(0, 2, 1)
        z_sig = torch.exp(z_log_var)

        z = sample_norm(z_mu, z_sig)
        for block1, block2, attn, upsample in self.ups:
            z = torch.cat((z, h.pop()), dim = 1)
            z = block1(z, t)

            z = torch.cat((z, h.pop()), dim = 1)
            z = block2(z, t)
            z = attn(z)

            z = upsample(z)

        x = torch.cat((z, ori_r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x), (pz_mu,pz_sig, z_mu, z_sig)


class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, drop_out=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=True)])
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
        self.layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
        self.dropout = nn.Dropout(drop_out)

        self.r_mean = nn.Linear(hidden_size, output_size * input_size)
        self.r_variance = nn.Linear(hidden_size, output_size * input_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                x = layer(x)  
            else:
                x = self.dropout(F.relu(layer(x)))
        r_mu = self.r_mean(x)
        r_log_var = self.r_variance(x)
        r_sig = torch.exp(r_log_var)
        r = sample_norm(r_mu, r_sig)
        return r
    
class MLPRegressorConv(nn.Module):
    def __init__(self, input_size, hidden_channels, num_layers, output_channels, seq_length, drop_out=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.Conv1d(input_size, hidden_channels, kernel_size=1, bias=True)])
        for _ in range(num_layers - 2):
            self.layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=True))
        self.layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=True))
        self.dropout = nn.Dropout(drop_out)

        self.r_mean = nn.Conv1d(hidden_channels, output_channels, kernel_size=seq_length)
        self.r_log_variance = nn.Conv1d(hidden_channels, output_channels, kernel_size=seq_length)
    

    # TODO : t 추가 활용 필요
    def forward(self, x, t):
        # x is of shape: (batch_size, input_channels, seq_length)
        for i, layer in enumerate(self.layers):
            x = self.dropout(F.relu(layer(x)))
        r_mu = self.r_mean(x).squeeze(-1)
        r_log_var = self.r_log_variance(x).squeeze(-1)
        r_sig = torch.exp(r_log_var)
        r = sample_norm(r_mu, r_sig)
        return r

