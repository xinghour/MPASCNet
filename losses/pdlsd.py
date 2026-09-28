import os
import csv
import argparse
import torch
import torchaudio
from tqdm import tqdm


def match_length(clean_audio, audio):
    min_len = min(clean_audio.shape[-1], audio.shape[-1])
    clean_audio = clean_audio[..., :min_len]
    audio = audio[..., :min_len]

    return clean_audio, audio


def load_audio(path, sampling_rate):
    audio, sr = torchaudio.load(path)

    if sr != sampling_rate:
        raise ValueError("{}: {} != {}".format(path, sr, sampling_rate))

    return audio


def stft_audio(audio, n_fft, hop_size, win_size):
    window = torch.hann_window(win_size, device=audio.device)
    spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size, window=window, center=True, return_complex=True)

    return spec


def cal_pd(clean_spec, audio_spec, eps=1e-8):
    clean_mag = clean_spec.abs()
    phase_diff = torch.angle(clean_spec * audio_spec.conj()).abs() * 180.0 / torch.pi
    pd = torch.sum(clean_mag * phase_diff, dim=(-2, -1)) / (torch.sum(clean_mag, dim=(-2, -1)) + eps)

    return pd.mean().item()


def cal_lsd(clean_spec, audio_spec, eps=1e-8):
    clean_log_power = torch.log10(clean_spec.abs().pow(2) + eps)
    audio_log_power = torch.log10(audio_spec.abs().pow(2) + eps)
    frame_lsd = torch.sqrt(torch.mean((clean_log_power - audio_log_power).pow(2), dim=-2))
    lsd = frame_lsd.mean(dim=-1)

    return lsd.mean().item()


def get_samples(eval_dir, enhanced_dir, clean_name, codec_name, enhanced_name):
    samples = []

    for name in sorted(os.listdir(eval_dir)):
        eval_path = os.path.join(eval_dir, name)
        enhanced_path = os.path.join(enhanced_dir, name)

        if not os.path.isdir(eval_path):
            continue

        clean_file = os.path.join(eval_path, clean_name)
        codec_file = os.path.join(eval_path, codec_name)
        enhanced_file = os.path.join(enhanced_path, enhanced_name)

        if not os.path.isfile(clean_file):
            continue

        if not os.path.isfile(codec_file):
            continue

        if not os.path.isfile(enhanced_file):
            continue

        samples.append((name, clean_file, codec_file, enhanced_file))

    return samples


def write_metrics(csv_path, rows):
    fields = [
        "name",
        "codec_pd",
        "enhanced_pd",
        "delta_pd",
        "codec_lsd",
        "enhanced_lsd",
        "delta_lsd"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def calculate(a):
    samples = get_samples(a.eval_dir, a.enhanced_dir, a.clean_name, a.codec_name, a.enhanced_name)

    if len(samples) == 0:
        raise RuntimeError("No evaluation samples found")

    rows = []

    codec_pd_tot = 0.
    enhanced_pd_tot = 0.
    codec_lsd_tot = 0.
    enhanced_lsd_tot = 0.

    sample_bar = tqdm(samples, desc="Calculating PD/LSD", dynamic_ncols=True)

    for name, clean_path, codec_path, enhanced_path in sample_bar:
        clean_audio = load_audio(clean_path, a.sampling_rate)
        codec_audio = load_audio(codec_path, a.sampling_rate)
        enhanced_audio = load_audio(enhanced_path, a.sampling_rate)

        clean_for_codec, codec_audio = match_length(clean_audio, codec_audio)
        clean_for_enhanced, enhanced_audio = match_length(clean_audio, enhanced_audio)

        clean_codec_spec = stft_audio(clean_for_codec, a.n_fft, a.hop_size, a.win_size)
        codec_spec = stft_audio(codec_audio, a.n_fft, a.hop_size, a.win_size)
        clean_enhanced_spec = stft_audio(clean_for_enhanced, a.n_fft, a.hop_size, a.win_size)
        enhanced_spec = stft_audio(enhanced_audio, a.n_fft, a.hop_size, a.win_size)

        codec_pd = cal_pd(clean_codec_spec, codec_spec)
        enhanced_pd = cal_pd(clean_enhanced_spec, enhanced_spec)
        codec_lsd = cal_lsd(clean_codec_spec, codec_spec)
        enhanced_lsd = cal_lsd(clean_enhanced_spec, enhanced_spec)

        codec_pd_tot += codec_pd
        enhanced_pd_tot += enhanced_pd
        codec_lsd_tot += codec_lsd
        enhanced_lsd_tot += enhanced_lsd

        rows.append({
            "name": name,
            "codec_pd": codec_pd,
            "enhanced_pd": enhanced_pd,
            "delta_pd": codec_pd - enhanced_pd,
            "codec_lsd": codec_lsd,
            "enhanced_lsd": enhanced_lsd,
            "delta_lsd": codec_lsd - enhanced_lsd
        })

        sample_bar.set_postfix({
            "codec_pd": "{:.3f}".format(codec_pd),
            "enhanced_pd": "{:.3f}".format(enhanced_pd),
            "codec_lsd": "{:.3f}".format(codec_lsd),
            "enhanced_lsd": "{:.3f}".format(enhanced_lsd)
        })

    length = len(rows)

    codec_pd_avg = codec_pd_tot / length
    enhanced_pd_avg = enhanced_pd_tot / length
    codec_lsd_avg = codec_lsd_tot / length
    enhanced_lsd_avg = enhanced_lsd_tot / length

    rows.append({
        "name": "avg",
        "codec_pd": codec_pd_avg,
        "enhanced_pd": enhanced_pd_avg,
        "delta_pd": codec_pd_avg - enhanced_pd_avg,
        "codec_lsd": codec_lsd_avg,
        "enhanced_lsd": enhanced_lsd_avg,
        "delta_lsd": codec_lsd_avg - enhanced_lsd_avg
    })

    write_metrics(a.output_path, rows)

    print("Codec PD: {:.4f}, Enhanced PD: {:.4f}, Delta PD: {:.4f}".format(codec_pd_avg, enhanced_pd_avg, codec_pd_avg - enhanced_pd_avg))
    print("Codec LSD: {:.4f}, Enhanced LSD: {:.4f}, Delta LSD: {:.4f}".format(codec_lsd_avg, enhanced_lsd_avg, codec_lsd_avg - enhanced_lsd_avg))
    print("Saved metrics to {}".format(a.output_path))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval_dir', required=True)
    parser.add_argument('--enhanced_dir', required=True)
    parser.add_argument('--output_path', default='pd_lsd.csv')
    parser.add_argument('--clean_name', default='ori_wav.wav')
    parser.add_argument('--codec_name', default='codec_wav.wav')
    parser.add_argument('--enhanced_name', default='enhanced_wav.wav')
    parser.add_argument('--sampling_rate', default=44100, type=int)
    parser.add_argument('--n_fft', default=882, type=int)
    parser.add_argument('--hop_size', default=441, type=int)
    parser.add_argument('--win_size', default=882, type=int)

    a = parser.parse_args()

    calculate(a)


if __name__ == '__main__':
    main()