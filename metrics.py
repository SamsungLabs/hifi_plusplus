"""
Based on
    https://github.com/AndreevP/speech_distances
"""
import itertools
from abc import ABC, abstractmethod

import librosa
import numpy as np
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

import log_utils
from metric_denoising import composite_eval
from metric_nets import Wav2Vec2MOS


class Metric(ABC):
    name = "Abstract Metric"

    def __init__(self, num_splits=5, device="cuda", big_val_size=500):
        self.num_splits = num_splits
        self.device = device
        self.duration = None  # calculated in Metric.compute()
        self.val_size = None
        self.result = dict()
        self.big_val_size = big_val_size

    @abstractmethod
    def better(self, first, second):
        pass

    @abstractmethod
    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        pass

    def compute(self, samples, real_samples, epoch_num, epoch_info):
        with log_utils.Timer() as timer:
            self._compute(samples, real_samples, epoch_num, epoch_info)
        self.result["dur"] = timer.duration
        self.result["val_size"] = samples.shape[0]

        if "best_mean" not in self.result or self.better(
            self.result["mean"], self.result["best_mean"]
        ):
            self.result["best_mean"] = self.result["mean"]
            self.result["best_std"] = self.result["std"]
            self.result["best_epoch"] = epoch_num

        if self.result["val_size"] >= 200:  # for now
            self.result["big_val_mean"] = self.result["mean"]
            if "best_big_val_mean" not in self.result or self.better(
                self.result["big_val_mean"], self.result["best_big_val_mean"]
            ):
                self.result["best_big_val_mean"] = self.result["big_val_mean"]

    def save_result(self, epoch_info):
        metric_name = self.name
        for key, value in self.result.items():
            epoch_info[f"metrics_{key}/{metric_name}"] = value


class MOSNet(Metric):
    name = "MOSNet"

    def __init__(self, sr=22050, **kwargs):
        super().__init__(**kwargs)

        self.mos_net = Wav2Vec2MOS("weights/wave2vec2mos.pth")
        self.sr = sr

    def better(self, first, second):
        return first > second

    def _compute_per_split(self, split):
        return self.mos_net.calculate(split)

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        required_sr = self.mos_net.sample_rate
        resample = torchaudio.transforms.Resample(
            orig_freq=self.sr, new_freq=required_sr
        ).to("cuda")

        samples /= samples.abs().max(-1, keepdim=True)[0]
        samples = [resample(s).squeeze() for s in samples]

        splits = [
            samples[i : i + self.num_splits]
            for i in range(0, len(samples), self.num_splits)
        ]
        fid_per_splits = [self._compute_per_split(split) for split in splits]
        self.result["mean"] = np.mean(fid_per_splits)
        self.result["std"] = np.std(fid_per_splits)


class ScaleInvariantSignalToDistortionRatio(Metric):
    """
    See https://arxiv.org/pdf/1811.02508.pdf
    """

    name = "SISDR"

    def better(self, first, second):
        return first > second

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        real_samples, samples = real_samples.squeeze(), samples.squeeze()
        if real_samples.dim() == 1:
            real_samples = real_samples[None]
            samples = samples[None]
        alpha = (samples * real_samples).sum(
            dim=1, keepdim=True
        ) / real_samples.square().sum(dim=1, keepdim=True)
        real_samples_scaled = alpha * real_samples
        e_target = real_samples_scaled.square().sum(dim=1)
        e_res = (samples - real_samples_scaled).square().sum(dim=1)
        si_sdr = 10 * torch.log10(e_target / e_res).cpu().numpy()

        self.result["mean"] = np.mean(si_sdr)
        self.result["std"] = np.std(si_sdr)


class SignalToNoiseRatio(Metric):

    name = "SNR"

    def better(self, first, second):
        return first > second

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        real_samples, samples = real_samples.squeeze(), samples.squeeze()
        if real_samples.dim() == 1:
            real_samples = real_samples[None]
            samples = samples[None]

        e_target = real_samples.square().sum(dim=1)
        e_res = (samples - real_samples).square().sum(dim=1)
        snr = 10 * torch.log10(e_target / e_res).cpu().numpy()

        self.result["mean"] = np.mean(snr)
        self.result["std"] = np.std(snr)


class VGGDistance(Metric):

    name = "VGG_dist"

    def __init__(self, sr=22050, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        self.model.eval()

    def better(self, first, second):
        return first < second

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        real_samples = real_samples.squeeze().cpu().numpy()
        samples = samples.squeeze().cpu().numpy()
        assert (
            samples.shape[1] >= 16384
        ), "too small segment size, everything will fall in this function"
        real_embs, fake_embs = [], []
        for real_s, fake_s in zip(real_samples, samples):
            real_embs.append(self.model(real_s, self.sr))
            fake_embs.append(self.model(fake_s, self.sr))
        real_embs = torch.stack(real_embs, dim=0)
        fake_embs = torch.stack(fake_embs, dim=0)
        dist = (real_embs - fake_embs).square().mean(dim=1)
        dist = dist.cpu().detach().numpy()

        self.result["mean"] = np.mean(dist)
        self.result["std"] = np.std(dist)


class LSD(Metric):
    name = "LSD"

    def better(self, first, second):
        return first < second

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        real_samples, samples = real_samples.squeeze(), samples.squeeze()
        if real_samples.dim() == 1:
            real_samples = real_samples[None]
            samples = samples[None]

        lsd = (samples - real_samples).square().mean(dim=1).sqrt()
        lsd = lsd.cpu().numpy()
        self.result["mean"] = np.mean(lsd)
        self.result["std"] = np.std(lsd)


class STOI(Metric):
    name = "STOI"

    def better(self, first, second):
        return first > second

    def __init__(self, sr=22050, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        real_samples = real_samples.squeeze().cpu().numpy()
        samples = samples.squeeze().cpu().numpy()
        if real_samples.ndim == 1:
            real_samples = real_samples[None]
            samples = samples[None]

        stois = []
        for s_real, s_fake in zip(real_samples, samples):
            s = stoi(s_real, s_fake, self.sr, extended=True)
            stois.append(s)
        self.result["mean"] = np.mean(stois)
        self.result["std"] = np.std(stois)


class PESQ(Metric):
    name = "PESQ"

    def better(self, first, second):
        return first > second

    def __init__(self, sr=22050, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        samples /= samples.abs().max(-1, keepdim=True)[0]
        real_samples /= real_samples.abs().max(-1, keepdim=True)[0]
        real_samples = real_samples.squeeze().cpu().numpy()
        samples = samples.squeeze().cpu().numpy()
        if real_samples.ndim == 1:
            real_samples = real_samples[None]
            samples = samples[None]

        pesqs = []
        for s_real, s_fake in zip(real_samples, samples):
            try:
                p = pesq(self.sr, s_real, s_fake, mode="wb")
            except:
                p = 1
            pesqs.append(p)

        self.result["mean"] = np.mean(pesqs)
        self.result["std"] = np.std(pesqs)


class CSEMetric(Metric):
    # sampling rate is 16000
    def better(self, first, second):
        return first > second

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = None

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        samples /= samples.abs().max(-1, keepdim=True)[0]
        real_samples /= real_samples.abs().max(-1, keepdim=True)[0]
        real_samples = real_samples.squeeze().cpu().numpy()
        samples = samples.squeeze().cpu().numpy()
        if real_samples.ndim == 1:
            real_samples = real_samples[None]
            samples = samples[None]

        res = list()
        for s_real, s_fake in zip(real_samples, samples):
            r = self.func(s_real, s_fake)
            res.append(r)

        self.result["mean"] = np.mean(res)
        self.result["std"] = np.std(res)


class CSIG(CSEMetric):
    # sampling rate is 16000
    name = "CSIG"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = lambda x, y: composite_eval(x, y)["csig"]


class CBAK(CSEMetric):
    # sampling rate is 16000
    name = "CBAK"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = lambda x, y: composite_eval(x, y)["cbak"]


class COVL(CSEMetric):
    # sampling rate is 16000
    name = "COVL"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = lambda x, y: composite_eval(x, y)["covl"]


def calculate_all_metrics(wavs, reference_wavs, metrics, n_max_files=None):
    scores = {metric.name: [] for metric in metrics}
    for x, y in tqdm(
        itertools.islice(zip(wavs, reference_wavs), n_max_files),
        total=n_max_files if n_max_files is not None else len(wavs),
        desc="Calculating metrics",
    ):
        x = librosa.util.normalize(x[: min(len(x), len(y))])
        y = librosa.util.normalize(y[: min(len(x), len(y))])
        x = torch.from_numpy(x)[None, None]
        y = torch.from_numpy(y)[None, None]
        for metric in metrics:
            metric._compute(x, y, None, None)
            scores[metric.name] += [metric.result["mean"]]
    scores = {k: (np.mean(v), np.std(v)) for k, v in scores.items()}
    return scores
