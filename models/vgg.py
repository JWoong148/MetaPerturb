import torch.nn as nn
import torch.nn.functional as F

from .perturb import Perturb


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, perturb, kernel_size=3, stride=1, padding=1, do_maxpool=False, perturb_idx=None
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.do_maxpool = do_maxpool

        self.perturb = perturb
        if self.perturb:
            self.perturb.add_running_stats(out_channels)
        self.perturb_idx = perturb_idx

    def forward(self, x, clipval=None, noise_coeff=1.0):
        out = self.bn(self.conv(x))
        metrics = None
        if self.perturb:
            out, metrics = self.perturb(out, clipval, noise_coeff, self.perturb_idx)
        out = F.relu(out, inplace=True)
        if self.do_maxpool:
            out = F.max_pool2d(out, 2)
        return out, metrics


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList()

    def forward(self, x, clipval=None, noise_coeff=1.0):
        metrics_all = []
        out = x
        for layer in self.conv_layers:
            out, metrics = layer(out, clipval, noise_coeff)
            metrics_all.append(metrics)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, metrics_all


class ConvNet4(ConvNet):
    def __init__(self, num_classes, conv_channels=32, img_size=32, do_perturb=False):
        super().__init__()
        sz = (((img_size // 2) // 2) // 2) // 2
        self.perturb = None
        if do_perturb:
            self.perturb = Perturb(channel_norm_factor=conv_channels, spatial_norm_factor=img_size)
        self.conv_layers = nn.ModuleList(
            [
                ConvBlock(3, conv_channels, self.perturb, do_maxpool=True, perturb_idx=0),
                ConvBlock(conv_channels, conv_channels, self.perturb, do_maxpool=True, perturb_idx=1),
                ConvBlock(conv_channels, conv_channels, self.perturb, do_maxpool=True, perturb_idx=2),
                ConvBlock(conv_channels, conv_channels, self.perturb, do_maxpool=True, perturb_idx=3),
            ]
        )
        self.fc = nn.Linear(sz * sz * conv_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ConvNet6(nn.Module):
    def __init__(self, num_classes, conv_channels=32, img_size=32, do_perturb=False):
        super().__init__()
        sz = (((img_size // 2) // 2) // 2) // 2
        self.perturb = None
        if do_perturb:
            self.perturb = Perturb(channel_norm_factor=conv_channels, spatial_norm_factor=img_size)
        self.conv_layers = nn.ModuleList(
            [
                ConvBlock(3, conv_channels, self.perturb, do_maxpool=True, perturb_idx=0),
                ConvBlock(conv_channels, conv_channels, self.perturb, do_maxpool=True, perturb_idx=1),
                ConvBlock(conv_channels, conv_channels, self.perturb, do_maxpool=False, perturb_idx=2),
                ConvBlock(conv_channels, conv_channels, self.perturb, do_maxpool=True, perturb_idx=3),
                ConvBlock(conv_channels, conv_channels, self.perturb, do_maxpool=False, perturb_idx=4),
                ConvBlock(conv_channels, conv_channels, self.perturb, do_maxpool=True, perturb_idx=5),
            ]
        )
        self.fc = nn.Linear(sz * sz * conv_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class VGG9(nn.Module):
    def __init__(self, num_classes, img_size=32, do_perturb=False):
        super().__init__()
        sz = ((((img_size // 2) // 2) // 2) // 2) // 2
        self.perturb = None
        if do_perturb:
            self.perturb = Perturb(channel_norm_factor=512, spatial_norm_factor=img_size)
        self.conv_layers = nn.ModuleList(
            [
                ConvBlock(3, 64, self.perturb, do_maxpool=True, perturb_idx=0),
                ConvBlock(64, 128, self.perturb, do_maxpool=True, perturb_idx=1),
                ConvBlock(128, 256, self.perturb, do_maxpool=False, perturb_idx=2),
                ConvBlock(256, 256, self.perturb, do_maxpool=True, perturb_idx=3),
                ConvBlock(256, 512, self.perturb, do_maxpool=False, perturb_idx=4),
                ConvBlock(512, 512, self.perturb, do_maxpool=True, perturb_idx=5),
                ConvBlock(512, 512, self.perturb, do_maxpool=False, perturb_idx=6),
                ConvBlock(512, 512, self.perturb, do_maxpool=True, perturb_idx=7),
            ]
        )
        self.fc = nn.Linear(sz * sz * 512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
