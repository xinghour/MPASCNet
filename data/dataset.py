import os
import random
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from torchaudio.functional import apply_codec


def _flatten_audio_channels(y):
    if y.dim() == 2:
        return y

    if y.dim() == 3:
        B, C, T = y.shape
        return y.reshape(B * C, T)

    raise ValueError(f"Expected audio shape [B, T] or [B, C, T], got {tuple(y.shape)}")


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    y = _flatten_audio_channels(y)
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1]+(1e-10), stft_spec[:, :, :, 0]+(1e-5))
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True, length=None):
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, length=length)

    return wav


def match2(x, d):
    assert x.dim() == 2, x.shape
    assert d.dim() == 2, d.shape
    minlen = min(x.shape[-1], d.shape[-1])
    x, d = x[:, 0:minlen], d[:, 0:minlen]
    Fx = torch.fft.rfft(x, dim=-1)
    Fd = torch.fft.rfft(d, dim=-1)
    Phi = Fd * Fx.conj()
    Phi = Phi / (Phi.abs() + 1e-3)
    Phi[:, 0] = 0
    tmp = torch.fft.irfft(Phi, dim=-1)
    tau = torch.argmax(tmp.abs(), dim=-1).tolist()

    return tau


def codec_simu(wav, sr=44100, options={'bitrate':'random','compression':'random', 'complexity':'random', 'vbr':'random'}):
    options = options.copy()
    if options['bitrate'] == 'random':
        options['bitrate'] = random.choice([24000, 32000, 48000, 64000, 96000, 128000])
    compression = int(options['bitrate']//1000)
    param = {'format': "mp3", "compression": compression}
    wav_encdec = apply_codec(wav, sr, **param)
    if wav_encdec.shape[-1] >= wav.shape[-1]:
        wav_encdec = wav_encdec[..., :wav.shape[-1]]
    else:
        wav_encdec = torch.cat([wav_encdec, wav[..., wav_encdec.shape[-1]:]], -1)
    tau = match2(wav, wav_encdec)
    wav_encdec = torch.roll(wav_encdec, -tau[0], -1)

    return wav_encdec


def _list_h5_files(directory):
    if not os.path.isdir(directory):
        return []

    files = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith('.h5') or name.endswith('.hdf5')]

    return sorted(files)


class MusicCodecDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir, codec_type, codec_options, sr=44100,
        segments=10, num_stems=4, snr_range=(-10, 10), num_samples=40000
    ):
        self.data_dir = data_dir
        self.codec_type = codec_type
        self.codec_options = codec_options
        self.segments = int(segments * sr)
        self.sr = sr
        self.num_stems = num_stems
        self.snr_range = snr_range
        self.num_samples = num_samples

        self.instruments = [
            "bass",
            "bowed_strings",
            "drums",
            "guitar",
            "other",
            "other_keys",
            "other_plucked",
            "percussion",
            "piano",
            "vocals",
            "wind"
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if random.random() > 0.5:
            select_stems = random.randint(1, self.num_stems)
            select_stems = random.choices(self.instruments, k=select_stems)
            ori_wav = []
            for stem in select_stems:
                h5path = random.choice(os.listdir(os.path.join(self.data_dir, stem)))
                datas = h5py.File(os.path.join(self.data_dir, stem, h5path), 'r')['data']
                random_index = random.randint(0, datas.shape[0]-1)
                music_wav = torch.FloatTensor(datas[random_index])
                start = random.randint(0, music_wav.shape[-1] - self.segments)
                music_wav = music_wav[:, start:start+self.segments]

                rescale_snr = random.randint(self.snr_range[0], self.snr_range[1])
                music_wav = music_wav * np.sqrt(10**(rescale_snr/10))
                ori_wav.append(music_wav)
            ori_wav = torch.stack(ori_wav).sum(0)
        else:
            h5path = random.choice(os.listdir(os.path.join(self.data_dir, "mixture")))
            datas = h5py.File(os.path.join(self.data_dir, "mixture", h5path), 'r')['data']
            random_index = random.randint(0, datas.shape[0]-1)
            music_wav = torch.FloatTensor(datas[random_index])
            start = random.randint(0, music_wav.shape[-1] - self.segments)
            ori_wav = music_wav[:, start:start+self.segments]

        codec_wav = codec_simu(ori_wav, sr=self.sr, options=self.codec_options)

        max_scale = max(ori_wav.abs().max(), codec_wav.abs().max())

        if max_scale > 0:
            ori_wav = ori_wav / max_scale
            codec_wav = codec_wav / max_scale

        return ori_wav, codec_wav


class MusicCodecEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_path = os.listdir(data_dir)
        self.data_path = [os.path.join(data_dir, i) for i in self.data_path]

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        ori_wav = torchaudio.load(self.data_path[idx]+"/ori_wav.wav")[0]
        codec_wav = torchaudio.load(self.data_path[idx]+"/codec_wav.wav")[0]

        return ori_wav, codec_wav, self.data_path[idx]
