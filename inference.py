import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import csv
import argparse
import json
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader
from env import AttrDict
from data.dataset import MusicCodecEvalDataset, mag_pha_stft, mag_pha_istft
from models.wapper import MPASCNet
from utils import load_checkpoint


torch.backends.cudnn.benchmark = True


AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg')


def stft_audio(audio, h):
    B, C, T = audio.shape
    audio = audio.reshape(B * C, T)
    mag, pha, com = mag_pha_stft(audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

    return mag, pha, com


def istft_audio(mag, pha, h, shape):
    batch_size, channels, length = shape
    audio = mag_pha_istft(mag, pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor, length=length)
    audio = audio.reshape(batch_size, channels, audio.shape[-1])

    return audio


def find_checkpoint(checkpoint_path, checkpoint_name):
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    checkpoint_file = os.path.join(checkpoint_path, checkpoint_name)

    if os.path.isfile(checkpoint_file):
        return checkpoint_file

    raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_file))


def match_length(clean_audio, audio):
    min_len = min(clean_audio.shape[-1], audio.shape[-1])
    clean_audio = clean_audio[..., :min_len]
    audio = audio[..., :min_len]

    return clean_audio, audio


def cal_sdr(clean_audio, audio, eps=1e-8):
    noise = clean_audio - audio
    clean_power = clean_audio.pow(2).sum(dim=-1)
    noise_power = noise.pow(2).sum(dim=-1)
    sdr = 10 * torch.log10((clean_power + eps) / (noise_power + eps))

    return sdr


def cal_si_snr(clean_audio, audio, eps=1e-8):
    clean_audio = clean_audio - clean_audio.mean(dim=-1, keepdim=True)
    audio = audio - audio.mean(dim=-1, keepdim=True)

    clean_power = torch.sum(clean_audio ** 2, dim=-1, keepdim=True)
    target = torch.sum(audio * clean_audio, dim=-1, keepdim=True) * clean_audio / (clean_power + eps)
    noise = audio - target

    target_power = target.pow(2).sum(dim=-1)
    noise_power = noise.pow(2).sum(dim=-1)
    si_snr = 10 * torch.log10((target_power + eps) / (noise_power + eps))

    return si_snr


def write_metrics(csv_path, rows):
    fields = [
        "name",
        "codec_sdr",
        "enhanced_sdr",
        "delta_sdr",
        "codec_si_snr",
        "enhanced_si_snr",
        "delta_si_snr"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def get_audio_files(input_path):
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(AUDIO_EXTENSIONS):
            raise ValueError("Unsupported audio file: {}".format(input_path))

        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError("Input path not found: {}".format(input_path))

    audio_files = []

    for name in os.listdir(input_path):
        path = os.path.join(input_path, name)

        if os.path.isfile(path) and name.lower().endswith(AUDIO_EXTENSIONS):
            audio_files.append(path)

    return sorted(audio_files)


def enhance_audio(audio, generator, h, device):
    audio = audio.unsqueeze(0).to(device)
    batch_size, channels, length = audio.shape
    mag, pha, _ = stft_audio(audio, h)
    mag_g, pha_g, _ = generator(mag, pha)
    audio_g = istft_audio(mag_g, pha_g, h, (batch_size, channels, length))

    return audio_g.squeeze(0).detach().cpu()


def inference_audio_input(a, h, generator, device):
    audio_files = get_audio_files(a.input)

    if len(audio_files) == 0:
        raise RuntimeError("No supported audio files found in {}".format(a.input))

    audio_bar = tqdm(audio_files, desc="Inference", dynamic_ncols=True)

    with torch.no_grad():
        for audio_path in audio_bar:
            audio, sampling_rate = torchaudio.load(audio_path)
            original_length = audio.shape[-1]

            if sampling_rate != h.sampling_rate:
                audio = torchaudio.functional.resample(audio, sampling_rate, h.sampling_rate)

            enhanced_audio = enhance_audio(audio, generator, h, device)

            if sampling_rate != h.sampling_rate:
                enhanced_audio = torchaudio.functional.resample(enhanced_audio, h.sampling_rate, sampling_rate)

                if enhanced_audio.shape[-1] >= original_length:
                    enhanced_audio = enhanced_audio[..., :original_length]
                else:
                    enhanced_audio = F.pad(enhanced_audio, (0, original_length - enhanced_audio.shape[-1]))

            name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(a.output_dir, name + a.suffix + '.wav')
            torchaudio.save(output_path, enhanced_audio.clamp(-1.0, 1.0), sampling_rate)
            audio_bar.set_postfix({"file": os.path.basename(audio_path)})

    print("Saved enhanced audio to {}".format(a.output_dir))


def inference_eval(a, h, generator, device):
    validset = MusicCodecEvalDataset(a.eval_dir)

    validation_loader = DataLoader(
        validset,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=False
    )

    rows = []

    codec_sdr_tot = 0
    enhanced_sdr_tot = 0
    codec_si_snr_tot = 0
    enhanced_si_snr_tot = 0

    valid_bar = tqdm(validation_loader, desc="Inference", dynamic_ncols=True)

    with torch.no_grad():
        for _, batch in enumerate(valid_bar):
            clean_audio, noisy_audio, data_path = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)

            batch_size, channels, length = noisy_audio.shape
            noisy_mag, noisy_pha, _ = stft_audio(noisy_audio, h)

            mag_g, pha_g, _ = generator(noisy_mag, noisy_pha)
            shape = (batch_size, channels, length)
            audio_g = istft_audio(mag_g, pha_g, h, shape)

            clean_for_codec, codec_for_metric = match_length(clean_audio, noisy_audio)
            clean_for_enhanced, enhanced_for_metric = match_length(clean_audio, audio_g)

            codec_sdr = cal_sdr(clean_for_codec, codec_for_metric).mean().item()
            enhanced_sdr = cal_sdr(clean_for_enhanced, enhanced_for_metric).mean().item()
            codec_si_snr = cal_si_snr(clean_for_codec, codec_for_metric).mean().item()
            enhanced_si_snr = cal_si_snr(clean_for_enhanced, enhanced_for_metric).mean().item()

            codec_sdr_tot += codec_sdr
            enhanced_sdr_tot += enhanced_sdr
            codec_si_snr_tot += codec_si_snr
            enhanced_si_snr_tot += enhanced_si_snr

            data_path = data_path[0]
            name = os.path.basename(os.path.normpath(data_path))
            save_dir = os.path.join(a.output_dir, name)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, a.save_name)
            save_audio = enhanced_for_metric.squeeze(0).detach().cpu().clamp(-1.0, 1.0)
            torchaudio.save(save_path, save_audio, h.sampling_rate)

            rows.append({
                "name": name,
                "codec_sdr": codec_sdr,
                "enhanced_sdr": enhanced_sdr,
                "delta_sdr": enhanced_sdr - codec_sdr,
                "codec_si_snr": codec_si_snr,
                "enhanced_si_snr": enhanced_si_snr,
                "delta_si_snr": enhanced_si_snr - codec_si_snr
            })

            valid_bar.set_postfix({
                "codec_sdr": "{:.3f}".format(codec_sdr),
                "enhanced_sdr": "{:.3f}".format(enhanced_sdr),
                "codec_sisnr": "{:.3f}".format(codec_si_snr),
                "enhanced_sisnr": "{:.3f}".format(enhanced_si_snr)
            })

    length = len(rows)

    if length == 0:
        raise RuntimeError("No validation samples found in {}".format(a.eval_dir))

    codec_sdr_avg = codec_sdr_tot / length
    enhanced_sdr_avg = enhanced_sdr_tot / length
    codec_si_snr_avg = codec_si_snr_tot / length
    enhanced_si_snr_avg = enhanced_si_snr_tot / length

    rows.append({
        "name": "avg",
        "codec_sdr": codec_sdr_avg,
        "enhanced_sdr": enhanced_sdr_avg,
        "delta_sdr": enhanced_sdr_avg - codec_sdr_avg,
        "codec_si_snr": codec_si_snr_avg,
        "enhanced_si_snr": enhanced_si_snr_avg,
        "delta_si_snr": enhanced_si_snr_avg - codec_si_snr_avg
    })

    csv_path = os.path.join(a.output_dir, "metrics.csv")
    write_metrics(csv_path, rows)

    print("Codec SDR: {:.4f}, Enhanced SDR: {:.4f}, Delta SDR: {:.4f}".format(codec_sdr_avg, enhanced_sdr_avg, enhanced_sdr_avg - codec_sdr_avg))
    print("Codec SI-SNR: {:.4f}, Enhanced SI-SNR: {:.4f}, Delta SI-SNR: {:.4f}".format(codec_si_snr_avg, enhanced_si_snr_avg, enhanced_si_snr_avg - codec_si_snr_avg))
    print("Saved metrics to {}".format(csv_path))


def inference(a, h):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    generator = MPASCNet(h).to(device)

    checkpoint_file = find_checkpoint(a.checkpoint_path, a.checkpoint_name)
    print("Loading checkpoint: {}".format(checkpoint_file))

    state_dict = load_checkpoint(checkpoint_file, device)

    if "generator" in state_dict:
        generator.load_state_dict(state_dict["generator"])
    else:
        generator.load_state_dict(state_dict)

    generator.eval()
    os.makedirs(a.output_dir, exist_ok=True)

    if a.input is not None:
        inference_audio_input(a, h, generator, device)
    else:
        inference_eval(a, h, generator, device)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input')
    input_group.add_argument('--eval_dir')
    parser.add_argument('--checkpoint_path', default='cp_model')
    parser.add_argument('--checkpoint_name', default='best')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output_dir', default='inference_results')
    parser.add_argument('--save_name', default='enhanced_wav.wav')
    parser.add_argument('--suffix', default='_enhanced')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)

    inference(a, h)


if __name__ == '__main__':
    main()