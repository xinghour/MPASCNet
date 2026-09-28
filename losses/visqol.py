import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import sys
import os
import csv
import argparse
import json
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader
from env import AttrDict
from data.dataset import MusicCodecEvalDataset
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2


def match_length(clean_audio, audio):
    min_len = min(clean_audio.shape[-1], audio.shape[-1])
    clean_audio = clean_audio[..., :min_len]
    audio = audio[..., :min_len]

    return clean_audio, audio


def create_visqol_api():
    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = 48000
    config.options.use_speech_scoring = False

    svr_model_path = "libsvm_nu_svr_model.txt"
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__),
        "model",
        svr_model_path
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    return api


def prepare_visqol_audio(audio, sampling_rate):
    if audio.dim() == 3:
        audio = audio.squeeze(0)

    if audio.dim() == 2:
        audio = audio.mean(dim=0)

    audio = audio.detach().cpu()

    if sampling_rate != 48000:
        audio = torchaudio.functional.resample(
            audio,
            sampling_rate,
            48000
        )

    audio = audio.contiguous().numpy().astype(np.float64)

    return audio


def cal_visqol(api, clean_audio, audio, clean_sampling_rate, audio_sampling_rate):
    clean_audio = prepare_visqol_audio(clean_audio, clean_sampling_rate)
    audio = prepare_visqol_audio(audio, audio_sampling_rate)

    clean_audio, audio = match_length(clean_audio, audio)

    similarity_result = api.Measure(clean_audio, audio)
    visqol = float(similarity_result.moslqo)

    return visqol


def write_metrics(csv_path, rows):
    fields = [
        "name",
        "codec_visqol",
        "enhanced_visqol",
        "delta_visqol"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def evaluation(a, h):
    os.makedirs(a.output_dir, exist_ok=True)

    validset = MusicCodecEvalDataset(a.eval_dir)

    validation_loader = DataLoader(
        validset,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=False
    )

    visqol_api = create_visqol_api()

    rows = []

    codec_visqol_tot = 0
    enhanced_visqol_tot = 0

    valid_bar = tqdm(validation_loader, desc="ViSQOL", dynamic_ncols=True)

    for _, batch in enumerate(valid_bar):
        clean_audio, noisy_audio, data_path = batch

        data_path = data_path[0]
        name = os.path.basename(os.path.normpath(data_path))

        enhanced_path = os.path.join(
            a.enhanced_dir,
            name,
            a.enhanced_name
        )

        if not os.path.isfile(enhanced_path):
            raise FileNotFoundError("Enhanced audio not found: {}".format(enhanced_path))

        enhanced_audio, enhanced_sampling_rate = torchaudio.load(enhanced_path)

        try:
            codec_visqol = cal_visqol(
                visqol_api,
                clean_audio,
                noisy_audio,
                h.sampling_rate,
                h.sampling_rate
            )

            enhanced_visqol = cal_visqol(
                visqol_api,
                clean_audio,
                enhanced_audio,
                h.sampling_rate,
                enhanced_sampling_rate
            )
        except Exception as e:
            raise RuntimeError("ViSQOL failed for {}: {}".format(name, e)) from e

        codec_visqol_tot += codec_visqol
        enhanced_visqol_tot += enhanced_visqol

        rows.append({
            "name": name,
            "codec_visqol": codec_visqol,
            "enhanced_visqol": enhanced_visqol,
            "delta_visqol": enhanced_visqol - codec_visqol
        })

        valid_bar.set_postfix({
            "codec": "{:.3f}".format(codec_visqol),
            "enhanced": "{:.3f}".format(enhanced_visqol),
            "delta": "{:.3f}".format(enhanced_visqol - codec_visqol)
        })

    length = len(rows)

    if length == 0:
        raise RuntimeError("No validation samples found in {}".format(a.eval_dir))

    codec_visqol_avg = codec_visqol_tot / length
    enhanced_visqol_avg = enhanced_visqol_tot / length

    rows.append({
        "name": "avg",
        "codec_visqol": codec_visqol_avg,
        "enhanced_visqol": enhanced_visqol_avg,
        "delta_visqol": enhanced_visqol_avg - codec_visqol_avg
    })

    csv_path = os.path.join(a.output_dir, "metrics.csv")
    write_metrics(csv_path, rows)

    print("Codec ViSQOL: {:.4f}, Enhanced ViSQOL: {:.4f}, Delta ViSQOL: {:.4f}".format(
        codec_visqol_avg,
        enhanced_visqol_avg,
        enhanced_visqol_avg - codec_visqol_avg
    ))
    print("Saved metrics to {}".format(csv_path))


def main():
    print('Initializing ViSQOL Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--eval_dir', required=True)
    parser.add_argument('--enhanced_dir', required=True)
    parser.add_argument('--enhanced_name', default='enhanced_wav.wav')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output_dir', default='visqol_results')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)

    evaluation(a, h)


if __name__ == '__main__':
    main()