import os
import argparse
import random
import h5py
import torch
import torchaudio
from torchaudio.functional import apply_codec


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


def list_h5_files(data_dir):
    files = [name for name in os.listdir(data_dir) if name.endswith(".h5") or name.endswith(".hdf5")]

    if len(files) == 0:
        raise FileNotFoundError("No h5 or hdf5 files found in {}".format(data_dir))

    return files


def load_random_h5_segment(data_dir, segments):
    h5path = random.choice(list_h5_files(data_dir))

    with h5py.File(os.path.join(data_dir, h5path), "r") as f:
        datas = f["data"]

        if datas.shape[0] == 0:
            raise ValueError("{} contains no valid segments".format(h5path))

        random_index = random.randint(0, datas.shape[0] - 1)
        music_wav = torch.FloatTensor(datas[random_index])

    if music_wav.shape[-1] < segments:
        raise ValueError("{} is shorter than {} samples".format(h5path, segments))

    start = random.randint(0, music_wav.shape[-1] - segments)
    music_wav = music_wav[:, start:start + segments]

    return music_wav


def create_sample(data_dir, segments):
    ori_wav = load_random_h5_segment(data_dir, segments)

    return ori_wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--sampling_rate", default=44100, type=int)
    parser.add_argument("--segments", default=3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    codec_options = {
        "bitrate": "random",
        "compression": "random",
        "complexity": "random",
        "vbr": "random"
    }

    os.makedirs(args.output_dir, exist_ok=True)

    segments = int(args.segments * args.sampling_rate)

    for i in range(args.num_samples):
        ori_wav = create_sample(args.data_dir, segments)
        codec_wav = codec_simu(ori_wav, sr=args.sampling_rate, options=codec_options)

        max_scale = max(ori_wav.abs().max(), codec_wav.abs().max())

        if max_scale > 0:
            ori_wav = ori_wav / max_scale
            codec_wav = codec_wav / max_scale

        sample_dir = os.path.join(args.output_dir, "{:05d}".format(i))
        os.makedirs(sample_dir, exist_ok=True)

        torchaudio.save(os.path.join(sample_dir, "ori_wav.wav"), ori_wav, args.sampling_rate)
        torchaudio.save(os.path.join(sample_dir, "codec_wav.wav"), codec_wav, args.sampling_rate)

    print("Generated {} validation samples at {}".format(args.num_samples, args.output_dir))


if __name__ == "__main__":
    main()