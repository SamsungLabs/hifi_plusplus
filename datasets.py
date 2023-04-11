import librosa
import torch.utils.data
import torch.distributions
import numpy as np
import random
import os
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
import utils
import scipy


datasets = utils.ClassRegistry()
loaders = utils.ClassRegistry()


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
    return_mel_and_spec=False,
):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                             fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    mel = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    mel = spectral_normalize_torch(mel)
    result = mel.squeeze()

    if return_mel_and_spec:
        spec = spectral_normalize_torch(spec)
        return result, spec
    else:
        return result


def get_dataset_filelist(dataset_split_file, input_wavs_dir):
    with open(dataset_split_file, "r", encoding="utf-8") as fi:
        files = [
            os.path.join(input_wavs_dir, fn)
            for fn in fi.read().split("\n")
            if len(fn) > 0
        ]
    return files


@loaders.add_to_registry("infinite", ("train", "val", "test"))
class InfiniteLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        *args,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        infinite=True,
        device=None,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        **kwargs
    ):
        super().__init__(
            *args,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            **kwargs
        )
        self.infinite = infinite
        self.device = device
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = next(self.dataset_iterator)
        except StopIteration:
            if self.infinite:
                self.dataset_iterator = super().__iter__()
                x = next(self.dataset_iterator)
            else:
                raise
        if self.device is not None:
            x = utils.move_to_device(x, self.device)
        return x


def split_audios(audios, segment_size, split):
    audios = [torch.FloatTensor(audio).unsqueeze(0) for audio in audios]
    if split:
        if audios[0].size(1) >= segment_size:
            max_audio_start = audios[0].size(1) - segment_size
            audio_start = random.randint(0, max_audio_start)
            audios = [
                audio[:, audio_start : audio_start + segment_size]
                for audio in audios
            ]
        else:
            audios = [
                torch.nn.functional.pad(
                    audio,
                    (0, segment_size - audio.size(1)),
                    "constant",
                )
                for audio in audios
            ]
    audios = [audio.squeeze(0).numpy() for audio in audios]
    return audios


@datasets.add_to_registry("vctk", ("train", "val", "test"))
class VCTKDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_split_file,
        vctk_wavs_dir,
        segment_size=8192,
        sampling_rate=16000,
        split=True,
        shuffle=False,
        device=None,
        input_freq=None,
        lowpass="default",
    ):
        self.audio_files = get_dataset_filelist(dataset_split_file,
                                                vctk_wavs_dir)
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device
        self.input_freq = input_freq
        self.lowpass = lowpass
        self.clean_wavs_dir = vctk_wavs_dir

    def __getitem__(self, index):
        vctk_fn = self.audio_files[index]

        vctk_audio = librosa.load(
            vctk_fn,
            sr=self.sampling_rate,
            res_type="polyphase",
        )[0]
        (vctk_audio, ) = split_audios([vctk_audio],
                                      self.segment_size, self.split)

        lp_inp = low_pass_filter(
            vctk_audio, self.input_freq,
            lp_type=self.lowpass, orig_sr=self.sampling_rate
        )
        input_audio = normalize(lp_inp)[None] * 0.95
        assert input_audio.shape[1] == vctk_audio.size

        input_audio = torch.FloatTensor(input_audio)
        audio = torch.FloatTensor(normalize(vctk_audio) * 0.95)
        audio = audio.unsqueeze(0)

        return input_audio, audio

    def __len__(self):
        return len(self.audio_files)


def low_pass_filter(audio: np.ndarray, max_freq,
                    lp_type="default", orig_sr=16000):
    if lp_type == "default":
        tmp = librosa.resample(
            audio, orig_sr=orig_sr, target_sr=max_freq * 2, res_type="polyphase"
        )
    elif lp_type == "decimate":
        sub = orig_sr / (max_freq * 2)
        assert int(sub) == sub
        tmp = scipy.signal.decimate(audio, int(sub))
    else:
        raise NotImplementedError
    # soxr_hq is faster and better than polyphase,
    # but requires additional libraries installed
    # the speed difference is only 4 times, we can live with that
    tmp = librosa.resample(tmp, orig_sr=max_freq * 2, target_sr=16000, res_type="polyphase")
    return tmp[: audio.size]


@datasets.add_to_registry("voicebank", ("train", "val", "test"))
class VoicebankDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        noisy_wavs_dir,
        clean_wavs_dir=None,
        path_prefix=None,
        segment_size=8192,
        sampling_rate=16000,
        split=True,
        shuffle=False,
        device=None,
        input_freq=None,
    ):
        if path_prefix:
            if clean_wavs_dir:
                clean_wavs_dir = os.path.join(path_prefix, clean_wavs_dir)
            noisy_wavs_dir = os.path.join(path_prefix, noisy_wavs_dir)
        self.audio_files = self.read_files_list(clean_wavs_dir, noisy_wavs_dir)
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device
        self.input_freq = input_freq

    @staticmethod
    def read_files_list(clean_wavs_dir, noisy_wavs_dir):
        fn_lst_clean = os.listdir(clean_wavs_dir)
        fn_lst_noisy = os.listdir(noisy_wavs_dir)
        assert set(fn_lst_clean) == set(fn_lst_noisy)
        return sorted(fn_lst_clean)

    def __getitem__(self, index):
        fn = self.audio_files[index]

        clean_audio = librosa.load(
            os.path.join(self.clean_wavs_dir, fn),
            sr=self.sampling_rate,
            res_type="polyphase",
        )[0]
        noisy_audio = librosa.load(
            os.path.join(self.noisy_wavs_dir, fn),
            sr=self.sampling_rate,
            res_type="polyphase",
        )[0]
        clean_audio, noisy_audio = split_audios(
            [clean_audio, noisy_audio],
            self.segment_size, self.split
        )

        input_audio = normalize(noisy_audio)[None] * 0.95
        assert input_audio.shape[1] == clean_audio.size

        input_audio = torch.FloatTensor(input_audio)
        audio = torch.FloatTensor(normalize(clean_audio) * 0.95)
        audio = audio.unsqueeze(0)

        return input_audio, audio

    def __len__(self):
        return len(self.audio_files)
