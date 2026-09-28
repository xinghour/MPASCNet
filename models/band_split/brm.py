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


class MagnitudeHead(nn.Module):
    def __init__(self, band_width, feature_dim):
        super(MagnitudeHead, self).__init__()
        self.band_width = band_width
        self.output = nn.Sequential(RMSNorm(feature_dim),
                                    nn.Conv1d(feature_dim, band_width * 2, 1),
                                    nn.GLU(dim=1))

    def forward(self, input):
        # input: [B, D, T]
        output = self.output(input)

        return output


class PhaseHead(nn.Module):
    def __init__(self, band_width, feature_dim):
        super(PhaseHead, self).__init__()
        self.band_width = band_width
        self.output = nn.Sequential(RMSNorm(feature_dim),
                                    nn.Conv1d(feature_dim, band_width * 4, 1),
                                    nn.GLU(dim=1))

    def forward(self, input):
        # input: [B, D, T]
        output = self.output(input)

        return output


class BandReconstruction(nn.Module):
    def __init__(self, h):
        super(BandReconstruction, self).__init__()
        self.n_fft = h.n_fft
        self.enc_dim = h.n_fft // 2 + 1
        self.feature_dim = h.feature_dim
        self.eps = 1e-8

        bandwidth = int(h.n_fft / 160)
        self.band_width = [bandwidth] * 79
        self.band_width.append(self.enc_dim - sum(self.band_width))
        self.nband = len(self.band_width)

        self.mag_head = nn.ModuleList([])
        self.phase_head = nn.ModuleList([])
        for i in range(self.nband):
            self.mag_head.append(MagnitudeHead(self.band_width[i], self.feature_dim))
            self.phase_head.append(PhaseHead(self.band_width[i], self.feature_dim))


    def forward(self, features, noisy_mag_bands, noisy_pha_bands):
        # features: [B, nband, D, T]
        assert features.shape[1] == self.nband
        assert len(noisy_mag_bands) == self.nband
        assert len(noisy_pha_bands) == self.nband

        pred_mag = []
        pred_pha = []

        for i in range(self.nband):
            band_width = self.band_width[i]
            feature = features[:, i]

            delta_mag = self.mag_head[i](feature)
            mag = torch.clamp(noisy_mag_bands[i] + delta_mag, min=self.eps)

            phase_vec = self.phase_head[i](feature).view(feature.shape[0], 2, band_width, feature.shape[-1])
            pha = torch.atan2(phase_vec[:, 1], phase_vec[:, 0])

            pred_mag.append(mag)
            pred_pha.append(pha)

        pred_mag = torch.cat(pred_mag, dim=1)
        pred_pha = torch.cat(pred_pha, dim=1)
        pred_com = torch.stack((pred_mag * torch.cos(pred_pha), pred_mag * torch.sin(pred_pha)), dim=-1)

        return pred_mag, pred_pha, pred_com
