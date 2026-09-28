import os
import re
import shutil
import argparse
import random
import yaml
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from torchaudio.functional import apply_codec


VOCAL_INSTRUMENTS = {
    "male singer",
    "female singer",
    "vocalists",
    "choir"
}


def to_stereo(x):
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return x


def VAD(x, sr=44100, win=441, session_length=6):
    x = to_stereo(x)
    x_len = x.shape[0] // sr * sr
    segment_len = sr * session_length
    step = segment_len // 2

    if x_len < segment_len:
        return []

    x = x[:x_len]
    x_p = x.reshape(-1, win, 2)
    x_pow = np.sum(np.power(x_p, 2), axis=(1, 2))
    x_valid_pow = x_pow[x_pow > 1e-3]

    if len(x_valid_pow) == 0:
        return []

    threshold = np.quantile(x_valid_pow, 0.15)
    valid_segment = []

    for start in range(0, x_len - segment_len + 1, step):
        segment = x[start:start + segment_len]
        segment_pow = segment.reshape(-1, win, 2)
        segment_pow = np.sum(np.power(segment_pow, 2), axis=(1, 2))
        pow_ratio = np.mean(segment_pow > threshold)

        if pow_ratio > 0.5:
            valid_segment.append(segment)

    return valid_segment


def match2(x, d):
    assert x.dim() == 2, x.shape
    assert d.dim() == 2, d.shape
    minlen = min(x.shape[-1], d.shape[-1])
    x, d = x[:, :minlen], d[:, :minlen]
    Fx = torch.fft.rfft(x, dim=-1)
    Fd = torch.fft.rfft(d, dim=-1)
    Phi = Fd * Fx.conj()
    Phi = Phi / (Phi.abs() + 1e-3)
    Phi[:, 0] = 0
    tmp = torch.fft.irfft(Phi, dim=-1)
    tau = torch.argmax(tmp.abs(), dim=-1).tolist()
    return tau


def codec_simu(wav, sr=44100, options={'bitrate':'random','compression':'random','complexity':'random','vbr':'random'}):
    options = options.copy()

    if options["bitrate"] == "random":
        options["bitrate"] = random.choice([24000, 32000, 48000, 64000, 96000, 128000])

    compression = int(options["bitrate"] // 1000)
    param = {"format": "mp3", "compression": compression}
    wav_encdec = apply_codec(wav, sr, **param)

    if wav_encdec.shape[-1] >= wav.shape[-1]:
        wav_encdec = wav_encdec[..., :wav.shape[-1]]
    else:
        wav_encdec = torch.cat([wav_encdec, wav[..., wav_encdec.shape[-1]:]], -1)

    tau = match2(wav, wav_encdec)
    wav_encdec = torch.roll(wav_encdec, -tau[0], -1)

    return wav_encdec


def normalize_instrument_name(name):
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def is_vocal_instrument(instrument):
    if isinstance(instrument, (list, tuple)):
        return any(is_vocal_instrument(name) for name in instrument)

    return normalize_instrument_name(instrument) in VOCAL_INSTRUMENTS


def get_tracks(root_dir):
    tracks = []

    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)

        if os.path.isdir(path):
            tracks.append((name, path))

    return sorted(tracks)


def load_metadata(track, metadata_dir):
    yaml_path = os.path.join(metadata_dir, f"{track}_METADATA.yaml")
    yml_path = os.path.join(metadata_dir, f"{track}_METADATA.yml")

    if os.path.isfile(yaml_path):
        metadata_path = yaml_path
    elif os.path.isfile(yml_path):
        metadata_path = yml_path
    else:
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    return metadata


def get_track_infos(root_dir, metadata_dir):
    track_infos = []

    for track, track_dir in get_tracks(root_dir):
        metadata = load_metadata(track, metadata_dir)

        if metadata is None:
            continue

        mix_path = os.path.join(track_dir, f"{track}_MIX.wav")
        stem_dir = os.path.join(track_dir, f"{track}_STEMS")
        stem_files = []
        vocal_files = []

        if os.path.isdir(stem_dir):
            for stem_id, stem_info in sorted(metadata.get("stems", {}).items()):
                if not isinstance(stem_info, dict):
                    continue

                stem_file = stem_info.get("filename", "")

                if not stem_file:
                    match = re.search(r"(\d+)", str(stem_id))

                    if match is None:
                        continue

                    stem_file = f"{track}_STEM_{int(match.group(1)):02d}.wav"

                stem_path = os.path.abspath(os.path.join(stem_dir, stem_file))

                if not os.path.isfile(stem_path):
                    continue

                stem_files.append(stem_path)

                if is_vocal_instrument(stem_info.get("instrument", "")):
                    vocal_files.append(stem_path)

        track_infos.append({"track": track, "mix": os.path.abspath(mix_path) if os.path.isfile(mix_path) else "", "stems": stem_files, "vocals": vocal_files})

    return track_infos


def load_audio(path, sr):
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)

    if file_sr != sr:
        raise ValueError(f"{path}: {file_sr} != {sr}")

    return to_stereo(x)


def load_vocal_track(vocal_files, sr):
    sources = []
    max_len = 0

    for path in vocal_files:
        x = load_audio(path, sr)
        sources.append(x)
        max_len = max(max_len, x.shape[0])

    if len(sources) == 0:
        return None

    vocal = np.zeros((max_len, 2), dtype=np.float32)

    for x in sources:
        vocal[:x.shape[0]] += x

    return vocal


def load_valid_sources(track_info, mode, sr, vad_segments):
    if mode == "multi":
        if not track_info["mix"]:
            return []

        x = load_audio(track_info["mix"], sr)
        segments = VAD(x, sr, 441, vad_segments)

        if len(segments) == 0:
            return []

        return [segments]

    if mode == "vocal":
        if len(track_info["vocals"]) == 0:
            return []

        x = load_vocal_track(track_info["vocals"], sr)
        segments = VAD(x, sr, 441, vad_segments)

        if len(segments) == 0:
            return []

        return [segments]

    if mode == "single":
        valid_sources = []

        for path in track_info["stems"]:
            x = load_audio(path, sr)
            segments = VAD(x, sr, 441, vad_segments)

            if len(segments) > 0:
                valid_sources.append(segments)

        return valid_sources

    raise ValueError(f"Unknown mode: {mode}")


def save_sample(ori_wav, output_dir, index, sr, codec_options):
    codec_wav = codec_simu(ori_wav, sr, codec_options)
    max_scale = max(ori_wav.abs().max(), codec_wav.abs().max())

    if max_scale > 0:
        ori_wav = ori_wav / max_scale
        codec_wav = codec_wav / max_scale

    sample_dir = os.path.join(output_dir, "{:05d}".format(index))
    os.makedirs(sample_dir, exist_ok=True)
    torchaudio.save(os.path.join(sample_dir, "ori_wav.wav"), ori_wav, sr)
    torchaudio.save(os.path.join(sample_dir, "codec_wav.wav"), codec_wav, sr)


def create_track_samples(track_info, mode, output_dir, index, num_samples, sr, segments, vad_segments, codec_options):
    valid_sources = load_valid_sources(track_info, mode, sr, vad_segments)

    if len(valid_sources) == 0:
        return index, 0

    segment_len = int(segments * sr)
    created = 0

    for _ in range(num_samples):
        source_segments = random.choice(valid_sources)
        segment = random.choice(source_segments)

        if segment_len > segment.shape[0]:
            raise ValueError("segments must not be longer than vad_segments")

        start = random.randint(0, segment.shape[0] - segment_len)
        ori_wav = torch.FloatTensor(segment[start:start + segment_len].T.copy())
        save_sample(ori_wav, output_dir, index, sr, codec_options)
        index += 1
        created += 1

    return index, created


def prepare_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)

        if os.path.isdir(path) and re.fullmatch(r"\d{5}", name):
            shutil.rmtree(path)


def get_mode_tracks(track_infos, mode):
    if mode == "multi":
        return [track_info for track_info in track_infos if track_info["mix"]]

    if mode == "single":
        return [track_info for track_info in track_infos if len(track_info["stems"]) > 0]

    if mode == "vocal":
        return [track_info for track_info in track_infos if len(track_info["vocals"]) > 0]

    raise ValueError(f"Unknown mode: {mode}")


def create_dataset(track_infos, mode, output_dir, num_samples, sr, segments, vad_segments, codec_options):
    prepare_output_dir(output_dir)
    mode_tracks = get_mode_tracks(track_infos, mode)

    if len(mode_tracks) == 0:
        raise RuntimeError(f"No tracks found for {mode}")

    random.shuffle(mode_tracks)
    base_count = num_samples // len(mode_tracks)
    remainder = num_samples % len(mode_tracks)
    sample_index = 0
    valid_tracks = []

    for i, track_info in enumerate(tqdm(mode_tracks, desc=f"Processing {mode.upper()}", dynamic_ncols=True)):
        count = base_count + int(i < remainder)

        if count == 0:
            continue

        sample_index, created = create_track_samples(track_info, mode, output_dir, sample_index, count, sr, segments, vad_segments, codec_options)

        if created > 0:
            valid_tracks.append(track_info)

    if len(valid_tracks) == 0:
        raise RuntimeError(f"No valid tracks found for {mode}")

    while sample_index < num_samples:
        track_info = random.choice(valid_tracks)
        sample_index, _ = create_track_samples(track_info, mode, output_dir, sample_index, 1, sr, segments, vad_segments, codec_options)

    print("{} tracks: {}".format(mode.upper(), len(mode_tracks)))
    print("{} valid tracks: {}".format(mode.upper(), len(valid_tracks)))
    print("{} samples: {}".format(mode.upper(), sample_index))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--metadata_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--sampling_rate", default=44100, type=int)
    parser.add_argument("--segments", default=3, type=float)
    parser.add_argument("--vad_segments", default=6, type=int)
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

    track_infos = get_track_infos(args.data_dir, args.metadata_dir)

    if len(track_infos) == 0:
        raise RuntimeError("No MedleyDB tracks found")

    multi_output_dir = os.path.join(args.output_dir, "multi")
    single_output_dir = os.path.join(args.output_dir, "single")
    vocal_output_dir = os.path.join(args.output_dir, "vocal")

    create_dataset(track_infos, "multi", multi_output_dir, args.num_samples, args.sampling_rate, args.segments, args.vad_segments, codec_options)
    create_dataset(track_infos, "single", single_output_dir, args.num_samples, args.sampling_rate, args.segments, args.vad_segments, codec_options)
    create_dataset(track_infos, "vocal", vocal_output_dir, args.num_samples, args.sampling_rate, args.segments, args.vad_segments, codec_options)


if __name__ == "__main__":
    main()