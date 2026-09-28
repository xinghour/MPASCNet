import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input: [B, N, T]
        B, N, T = input.shape
        assert N % self.groups == 0

        input_float = input.reshape(B, self.groups, -1, T).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(-2, keepdim=True) + self.eps)

        return input_norm.type_as(input).reshape(B, N, T) * self.weight.reshape(1, -1, 1)


class BandPatchEncoder(nn.Module):
    def __init__(self, band_width, feature_dim, input_channel=2):
        super(BandPatchEncoder, self).__init__()
        self.band_width = band_width
        self.proj = nn.Conv2d(input_channel, feature_dim, (1, band_width))
        self.norm = RMSNorm(feature_dim)

    def forward(self, x):
        # x: [B, 2, T, BW]
        assert x.shape[-1] == self.band_width

        x = self.proj(x).squeeze(-1)
        x = self.norm(x)

        return x


class BandSplit(nn.Module):
    def __init__(self, h):
        super(BandSplit, self).__init__()
        self.n_fft = h.n_fft
        self.enc_dim = h.n_fft // 2 + 1
        self.feature_dim = h.feature_dim

        bandwidth = int(h.n_fft / 160)
        self.band_width = [bandwidth] * 79
        self.band_width.append(self.enc_dim - sum(self.band_width))
        self.nband = len(self.band_width)

        self.band_encoder = nn.ModuleList([])
        for i in range(self.nband):
            self.band_encoder.append(BandPatchEncoder(self.band_width[i], self.feature_dim))

    def forward(self, noisy_mag, noisy_pha):
        # noisy_mag/noisy_pha: [B*C, F, T]
        assert noisy_mag.shape == noisy_pha.shape
        assert noisy_mag.shape[1] == self.enc_dim

        features = []
        noisy_mag_bands = []
        noisy_pha_bands = []
        band_idx = 0

        for i in range(self.nband):
            band_width = self.band_width[i]
            mag = noisy_mag[:, band_idx:band_idx+band_width]
            pha = noisy_pha[:, band_idx:band_idx+band_width]
            x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2).contiguous()
            x = self.band_encoder[i](x)

            features.append(x)
            noisy_mag_bands.append(mag)
            noisy_pha_bands.append(pha)
            band_idx += band_width

        features = torch.stack(features, dim=1)

        return features, noisy_mag_bands, noisy_pha_bands
