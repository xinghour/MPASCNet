import os
import torch
import numpy as np
import librosa
import fast_bss_eval
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2


class VisqolCalculator(object):
    def __init__(self, sample_rate=44100, target_sample_rate=48000, normalize=True, norm_factor=5.0):
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self.norm_factor = norm_factor

        self.visqol_config = visqol_config_pb2.VisqolConfig()
        self.visqol_config.audio.sample_rate = target_sample_rate
        self.visqol_config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
        self.visqol_config.options.svr_model_path = os.path.join(os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
        self.visqol_api = visqol_lib_py.VisqolApi()
        self.visqol_api.Create(self.visqol_config)

    def prepare_audio(self, audio):
        # audio: [C, T]
        audio = audio.detach().cpu().float().mean(0).numpy()
        if self.sample_rate != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=self.target_sample_rate)
        audio = audio.astype(np.float64)

        return audio

    def measure(self, clean, estimate):
        clean = self.prepare_audio(clean)
        estimate = self.prepare_audio(estimate)
        score = self.visqol_api.Measure(clean, estimate).moslqo

        if self.normalize:
            score = score / self.norm_factor

        return score

    def measure_batch(self, clean, estimate):
        # clean/estimate: [B, C, T]
        assert clean.shape == estimate.shape
        assert clean.dim() == 3

        scores = []
        for i in range(clean.shape[0]):
            scores.append(self.measure(clean[i], estimate[i]))
        scores = torch.tensor(scores, device=clean.device, dtype=clean.dtype)

        return scores


class AudioMetrics(object):
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.visqol = VisqolCalculator(sample_rate=sample_rate, normalize=False)

    def sdr(self, clean, estimate):
        # clean/estimate: [B, C, T]
        assert clean.shape == estimate.shape
        assert clean.dim() == 3

        scores = []
        for i in range(clean.shape[0]):
            score = fast_bss_eval.sdr(clean[i].unsqueeze(0), estimate[i].unsqueeze(0), zero_mean=True).mean()
            scores.append(score)
        scores = torch.stack(scores)

        return scores

    def si_snr(self, clean, estimate):
        # clean/estimate: [B, C, T]
        assert clean.shape == estimate.shape
        assert clean.dim() == 3

        scores = []
        for i in range(clean.shape[0]):
            score = fast_bss_eval.si_sdr(clean[i].unsqueeze(0), estimate[i].unsqueeze(0), zero_mean=True).mean()
            scores.append(score)
        scores = torch.stack(scores)

        return scores

    def visqol_score(self, clean, estimate):
        return self.visqol.measure_batch(clean, estimate)

    def __call__(self, clean, estimate):
        sdr = self.sdr(clean, estimate)
        si_snr = self.si_snr(clean, estimate)
        visqol = self.visqol_score(clean, estimate)

        return {"sdr": sdr, "si_snr": si_snr, "visqol": visqol}


class MetricsTracker(object):
    def __init__(self, sample_rate=44100):
        self.metrics = AudioMetrics(sample_rate=sample_rate)
        self.all_sdrs = []
        self.all_si_snrs = []
        self.all_visqols = []

    def __call__(self, clean, estimate):
        scores = self.metrics(clean, estimate)
        self.all_sdrs += scores["sdr"].detach().cpu().tolist()
        self.all_si_snrs += scores["si_snr"].detach().cpu().tolist()
        self.all_visqols += scores["visqol"].detach().cpu().tolist()

        return scores

    def update(self):
        return {"sdr": np.array(self.all_sdrs).mean(),
                "si_snr": np.array(self.all_si_snrs).mean(),
                "visqol": np.array(self.all_visqols).mean()}

    def final(self):
        return {"sdr": np.array(self.all_sdrs).mean(),
                "si_snr": np.array(self.all_si_snrs).mean(),
                "visqol": np.array(self.all_visqols).mean()}
