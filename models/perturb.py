import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F


class ChannelEquivarientOp(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.padding = kernel_size // 2
        self.w_identity = nn.Parameter(torch.ones(1, 1, kernel_size, kernel_size))
        self.w_all = nn.Parameter(torch.ones(1, 1, kernel_size, kernel_size))
        nn.init.normal_(self.w_identity, 0, 0.01)
        nn.init.normal_(self.w_all, 0, 0.01)

    def forward(self, x):
        C = x.size(1)

        _w_identity = self.w_identity.expand(C, -1, -1, -1)
        _x = x.mean(dim=1, keepdim=True)

        alpha_identity = F.conv2d(x, _w_identity, padding=self.padding, groups=C)
        alpha_all = F.conv2d(_x, self.w_all, padding=self.padding).expand(-1, C, -1, -1)
        return alpha_identity + alpha_all


class Perturb(nn.Module):
    def __init__(self, channel_norm_factor, spatial_norm_factor, h_dim=4, kernel_size=3):
        super().__init__()
        # Noise generator
        self.phi_1 = ChannelEquivarientOp(kernel_size)
        self.phi_2 = ChannelEquivarientOp(kernel_size)

        # Inference scale (& shift)
        self.h_dim = h_dim
        self.kernel = nn.Parameter(torch.ones(self.h_dim, 1, 3, 3))
        self.fc = nn.Linear(2 * self.h_dim + 2, 1, bias=False)
        self.channel_norm_factor = channel_norm_factor
        self.spatial_norm_factor = spatial_norm_factor

        # Parmeter initialize
        nn.init.normal_(self.kernel, 0, 0.01)
        nn.init.normal_(self.fc.weight, 0, 0.01)

        # Initilaizing running_mean & running_var for each perturb layer
        self.momentum = 0.1
        self.running_mean = []
        self.running_var = []

    def add_running_stats(self, channels):
        self.running_mean.append(torch.zeros(channels, self.h_dim).cuda())
        self.running_var.append(torch.ones(channels, self.h_dim).cuda())

    def forward(self, x, clipval=None, noise_coeff=1.0, perturb_idx=None):
        # Noise generation
        alpha = self.phi_1(x)
        alpha = F.relu(alpha, inplace=True)
        alpha = self.phi_2(alpha)
        dist = tdist.Normal(alpha, torch.ones_like(alpha))
        noise = dist.rsample()
        noise = F.softplus(noise)
        metrics = {
            "max_noise": torch.max(noise),
            "avg_noise": torch.mean(noise),
            "max_input": torch.max(x),
            "avg_input": torch.mean(x),
        }

        # Scale inference
        B, C, H, W = x.size()
        if self.training:
            kernel = self.kernel.repeat(C, 1, 1, 1)
            _x = F.relu(F.conv2d(x, kernel, padding=1, groups=C)).mean(dim=[-1, -2]).view(B, C, self.h_dim)
            _x_mean = _x.mean(dim=0)
            _x_var = _x.var(dim=0)
            with torch.no_grad():
                self.running_mean[perturb_idx] = (
                    self.momentum * _x_mean + (1 - self.momentum) * self.running_mean[perturb_idx]
                )
                self.running_var[perturb_idx] = (
                    self.momentum * _x_var * B / (B - 1) + (1 - self.momentum) * self.running_var[perturb_idx]
                )
        else:
            _x_mean = self.running_mean[perturb_idx]
            _x_var = self.running_var[perturb_idx]

        channel = 1.0 * C / self.channel_norm_factor * torch.ones(C, 1).cuda()
        size = 1.0 * H / self.spatial_norm_factor * torch.ones(C, 1).cuda()
        x_vec = torch.cat([_x_mean, _x_var, channel, size], dim=-1)
        _x = self.fc(x_vec)
        scale = _x.view(1, -1, 1, 1)
        scale = torch.sigmoid(scale)

        # Apply sacle to noise & Noise annealing
        noise = ((scale * noise) - 1) * noise_coeff + 1
        out = noise * x
        metrics["max_output"] = torch.max(out)
        metrics["avg_output"] = torch.mean(out)
        metrics["norm_output"] = torch.norm(out, p=2, dim=[-1, -2]).mean()
        if clipval is not None:
            out = torch.clamp(out, max=clipval)
        return out, metrics
