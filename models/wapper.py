import torch
import torch.nn as nn
from models.mp_asc import MPASCEncoder, MPASCDecoder
from models.tsroformer import TSRoformer


class MPASCNet(nn.Module):
    def __init__(self, h):
        super(MPASCNet, self).__init__()
        self.h = h
        self.mp_asc_encoder = MPASCEncoder(h)
        self.tsroformer = TSRoformer(h)
        self.mp_asc_decoder = MPASCDecoder(h)

    def forward(self, noisy_mag, noisy_pha):
        feature = self.mp_asc_encoder(noisy_mag, noisy_pha)
        feature = self.tsroformer(feature)
        denoised_mag, denoised_pha, denoised_com = self.mp_asc_decoder(feature, noisy_mag, noisy_pha)

        return denoised_mag, denoised_pha, denoised_com