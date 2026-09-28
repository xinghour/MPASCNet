import torch
import torch.nn.functional as F
import numpy as np


def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_losses(phase_r, phase_g):

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss, gd_loss, iaf_loss


def generator_losses(
    clean_audio, audio_g, clean_mag, clean_pha, clean_com,
    mag_g, pha_g, com_g, com_g_hat,
    mag_weight=1.2, pha_weight=0.3, com_weight=0.1,
    stft_weight=0.1, time_weight=0.2
):

    loss_mag = F.mse_loss(mag_g, clean_mag)
    loss_ip, loss_gd, loss_iaf = phase_losses(pha_g, clean_pha)
    loss_pha = loss_ip + loss_gd + loss_iaf
    loss_com = F.mse_loss(com_g, clean_com) * 2
    loss_stft = F.mse_loss(com_g, com_g_hat) * 2
    loss_time = F.l1_loss(audio_g, clean_audio)

    loss_gen = loss_mag * mag_weight + loss_pha * pha_weight + loss_com * com_weight
    loss_gen = loss_gen + loss_stft * stft_weight + loss_time * time_weight

    return {
        "loss_gen": loss_gen,
        "loss_mag": loss_mag,
        "loss_pha": loss_pha,
        "loss_ip": loss_ip,
        "loss_gd": loss_gd,
        "loss_iaf": loss_iaf,
        "loss_com": loss_com,
        "loss_stft": loss_stft,
        "loss_time": loss_time
    }