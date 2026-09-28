import torch
import torch.nn as nn


class BandBias(nn.Module):
    def __init__(self, n_fft, num_head, decoder=False):
        super(BandBias, self).__init__()
        self.n_fft = n_fft
        self.enc_dim = n_fft // 2 + 1
        self.num_head = num_head

        bandwidth = int(n_fft / 160)
        self.band_width = [bandwidth] * 79
        self.band_width.append(self.enc_dim - sum(self.band_width))
        self.nband = len(self.band_width)

        assert self.band_width[-1] > 0

        self.band_indices = []
        band_idx = 0
        for i in range(self.nband):
            self.band_indices.append((band_idx, band_idx + self.band_width[i]))
            band_idx += self.band_width[i]

        assert band_idx == self.enc_dim

        position_bias = self.prepare_position_bias()
        if decoder:
            position_bias = position_bias.transpose(-1, -2).contiguous()

        self.position_bias = nn.Parameter(position_bias)

    def prepare_position_bias(self):
        position_bias = torch.zeros(self.nband, self.enc_dim)

        for i in range(self.nband):
            start, end = self.band_indices[i]
            center = (start + end) // 2
            denominator = (end - start) // 2 + 1

            for j in range(self.enc_dim):
                if j < start:
                    position_bias[i, j] = j - start
                elif j > end - 1:
                    position_bias[i, j] = end - 1 - j
                else:
                    position_bias[i, j] = -abs(center - j) / denominator

        position_bias = position_bias.unsqueeze(0).repeat(self.num_head, 1, 1)
        position_bias = position_bias.unsqueeze(0)

        return position_bias

    def forward(self, dtype=None):
        position_bias = self.position_bias
        if dtype is not None:
            position_bias = position_bias.to(dtype=dtype)

        return position_bias