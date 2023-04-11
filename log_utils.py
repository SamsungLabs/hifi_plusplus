import collections
import copy
import datetime
import glob
import heapq
import logging
import os
import time

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from ptflops import get_model_complexity_info

import datasets
import utils
import wandb


def print_network(net, net_name, x_input, logger):
    print("input shape:", tuple(x_input.shape)[1:])
    print(net)
    macs, params = get_model_complexity_info(
        net,
        tuple(x_input.shape)[1:],
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    logger.exp_logger.log({f"{net_name}_size": f"{params}"})
    logger.exp_logger.log({f"{net_name}_macs": f"{macs}"})
    print(f"Number of parameters: {params}")


def strf_time_delta(td):
    td_str = ""
    if td.days > 0:
        td_str += f"{td.days} days, " if td.days > 1 else f"{td.days} day, "
    hours = td.seconds // 3600
    if hours > 0:
        td_str += f"{hours}h "
    minutes = (td.seconds // 60) % 60
    if minutes > 0:
        td_str += f"{minutes}m "
    seconds = td.seconds % 60 + td.microseconds * 1e-6
    td_str += f"{seconds:.1f}s"
    return td_str


def draw_spec(x, title, sr, ax, mel=False, compute=True):
    plt.sca(ax)

    n_fft = 1024
    num_mels = 80
    hop_size = 256
    win_size = 1024
    fmin = 0
    fmax = 8000
    if compute:
        if mel:
            f = datasets.mel_spectrogram(
                x,
                n_fft=n_fft,
                num_mels=num_mels,
                sampling_rate=sr,
                hop_size=hop_size,
                win_size=win_size,
                fmin=fmin,
                fmax=fmax,
            )
            f = f.squeeze().numpy()
        else:
            f = librosa.core.stft(x, n_fft=n_fft, hop_length=hop_size)
            f = np.abs(f)
    else:
        f = x
    librosa.display.specshow(
        librosa.amplitude_to_db(f, ref=np.max),
        y_axis="hz",
        x_axis="time",
        sr=2 * fmax if mel else sr,
        hop_length=hop_size,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()


class LoggingManager:
    def __init__(self, exp_logger, console_logger):
        self.exp_logger = exp_logger
        self.console_logger = console_logger

    def log_iter(self, iter_num, epoch_num, num_iters, iter_info, **kwargs):
        self.console_logger.log_iter(
            epoch_num, iter_num, num_iters, iter_info, **kwargs
        )

    def log_epoch(self, epoch_num, epoch_info, test_predictions=None):
        self.exp_logger.log(dict(epoch=epoch_num, **epoch_info.to_dict()))
        self.console_logger.log_epoch(epoch_num, epoch_info)

        if test_predictions:
            for idx, pred_dict in enumerate(test_predictions):
                self.exp_logger.log_pred(idx, pred_dict, epoch_num)

    def log_info(self, output_info):
        self.console_logger.logger.info(output_info)


class WandbLogger:
    def __init__(self, **kwargs):
        wandb.init(**kwargs)
        self.run_dir = wandb.run.dir
        code = wandb.Artifact("project-source", type="code")

        # exclude venv if located in project dir
        exclude_env = os.getenv("VIRTUAL_ENV", "ffc_se_env").split("/")[-1]
        for path in glob.glob("**/*.py", recursive=True):
            if not path.startswith("wandb") and not path.startswith(exclude_env):
                if os.path.basename(path) != path:
                    code.add_dir(os.path.dirname(path), name=os.path.dirname(path))
                else:
                    code.add_file(os.path.basename(path), name=path)
        wandb.run.log_artifact(code)

    def log(self, info):
        wandb.log(info)

    def log_pred(self, idx, pred_dict, epoch_num):
        name = f"test_sample_{idx}"
        for key in pred_dict["audios"].keys():
            wandb.log(
                {
                    f"{name}/{key}": wandb.Audio(
                        pred_dict["audios"][key],
                        sample_rate=pred_dict["sr"],
                        caption=f"epoch = {epoch_num}",
                    )
                }
            )
        for key in pred_dict["figures"].keys():
            wandb.log(
                {
                    f"{name}/{key}": wandb.Image(
                        pred_dict["figures"][key],
                        caption=f"epoch = {epoch_num}",
                    )
                }
            )


class ConsoleLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

        self.logger.propagate = False

    @staticmethod
    def format_info(info):
        if not info:
            return str(info)
        log_groups = collections.defaultdict(dict)
        for k, v in info.to_dict().items():
            prefix, suffix = k.split("/", 1)
            log_groups[prefix][suffix] = f"{v:.3f}" if isinstance(v, float) else str(v)
        formatted_info = ""
        max_group_size = len(max(log_groups, key=len)) + 2
        max_k_size = max([len(max(g, key=len)) for g in log_groups.values()]) + 1
        max_v_size = (
            max([len(max(g.values(), key=len)) for g in log_groups.values()]) + 1
        )
        for group, group_info in log_groups.items():
            group_str = [
                f"{k:<{max_k_size}}={v:>{max_v_size}}" for k, v in group_info.items()
            ]
            max_g_size = len(max(group_str, key=len)) + 2
            group_str = "".join([f"{g:>{max_g_size}}" for g in group_str])
            formatted_info += f"\n{group + ':':<{max_group_size}}{group_str}"
        return formatted_info

    def log_iter(self, epoch_num, iter_num, num_iters, iter_info, event="epoch"):
        output_info = f"{event.upper()} {epoch_num}, ITER {iter_num}/{num_iters}:"
        output_info += self.format_info(iter_info)
        self.logger.info(output_info)

    def log_epoch(self, epoch_num, epoch_info):
        output_info = f"EPOCH {epoch_num}:"
        output_info += self.format_info(epoch_info)
        self.logger.info(output_info)


class Timer:
    def __init__(self, info=None, log_event=None):
        self.info = info
        self.log_event = log_event

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        self.duration = self.start.elapsed_time(self.end) / 1000
        if self.info:
            self.info[f"duration/{self.log_event}"] = self.duration


class TimeLog:
    def __init__(self, logger, total_num, event):
        self.logger = logger
        self.total_num = total_num
        self.event = event.upper()
        self.start = time.time()

    def now(self, current_num):
        elapsed = time.time() - self.start
        left = self.total_num * elapsed / (current_num + 1) - elapsed
        elapsed = strf_time_delta(datetime.timedelta(seconds=elapsed))
        left = strf_time_delta(datetime.timedelta(seconds=left))
        self.logger.log_info(f"TIME ELAPSED SINCE {self.event} START: {elapsed}")
        self.logger.log_info(f"TIME LEFT UNTIL {self.event} END: {left}")

    def end(self):
        elapsed = time.time() - self.start
        elapsed = strf_time_delta(datetime.timedelta(seconds=elapsed))
        self.logger.log_info(f"TIME ELAPSED SINCE {self.event} START: {elapsed}")
        self.logger.log_info(f"{self.event} ENDS")


class _StreamingMean:
    def __init__(self, val=None, counts=None):
        if val is None:
            self.mean = 0.0
            self.counts = 0
        else:
            if isinstance(val, torch.Tensor):
                val = val.data.cpu().numpy()
            self.mean = val
            if counts is not None:
                self.counts = counts
            else:
                self.counts = 1

    def update(self, mean, counts=1):
        if isinstance(mean, torch.Tensor):
            mean = mean.data.cpu().numpy()
        elif isinstance(mean, _StreamingMean):
            mean, counts = mean.mean, mean.counts * counts
        assert counts >= 0
        if counts == 0:
            return
        total = self.counts + counts
        self.mean = self.counts / total * self.mean + counts / total * mean
        self.counts = total

    def __add__(self, other):
        new = self.__class__(self.mean, self.counts)
        if isinstance(other, _StreamingMean):
            if other.counts == 0:
                return new
            else:
                new.update(other.mean, other.counts)
        else:
            new.update(other)
        return new


class StreamingMeans(collections.defaultdict):
    def __init__(self):
        super().__init__(_StreamingMean)

    def __setitem__(self, key, value):
        if isinstance(value, _StreamingMean):
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, _StreamingMean(value))

    def update(self, *args, **kwargs):
        for_update = dict(*args, **kwargs)
        for k, v in for_update.items():
            self[k].update(v)

    def to_dict(self, prefix=""):
        return dict((prefix + k, v.mean) for k, v in self.items())

    def to_str(self):
        return ", ".join([f"{k} = {v:.3f}" for k, v in self.to_dict().items()])


def save_checkpoint(
    checkpoint_dir, exp_name, epoch_num, nets, optims, device, save_full=False
):
    for name, net in nets.items():
        if name == "gen" or save_full:
            net = net.cpu()
            torch.save(
                net.state_dict(),
                os.path.join(checkpoint_dir, f"{name}_{epoch_num:04d}.pth"),
            )
            net.to(device)
    if save_full:
        for name, optim in optims.items():
            if optim is not None:
                torch.save(
                    optim.state_dict(),
                    os.path.join(checkpoint_dir, f"{name}_{epoch_num:04d}.pth"),
                )
    # save exp_name in exp dir (parent of the checkpoints) to make restoring
    # from checkpoint for the same exp_name possible
    with open(os.path.join(checkpoint_dir, "..", "exp_name.txt"), "w") as f:
        f.write(exp_name)


def restore_checkpoint_from_dir(checkpoint_dir, epoch_num, nets, optims, device):
    for name, net in nets.items():
        net.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, f"{name}_{epoch_num:04d}.pth"),
                map_location=device,
            )
        )
    for name, optim in optims.items():
        optim.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, f"{name}_{epoch_num:04d}.pth"),
                map_location=device,
            )
        )


def restore_checkpoint(
    run_dir, exp_name, nets, optims, device, logger=logging.getLogger("root")
):
    """
    Load latest checkpoint for the experiment with the same name.
    If there are multiple latest checkpoints, load any one of them.
    Return the epoch for the latest checkpoint that was loaded,
    or -1 if no valid checkpoint was found.
    """
    # find wandb experiment run with the same name
    checkpoint_dir_candidates = glob.glob(
        os.path.join(run_dir, "..", "..", "*", "files")
    )
    checkpoint_dirs = list()
    for dn in checkpoint_dir_candidates:
        exp_name_fn = os.path.join(dn, "exp_name.txt")
        if not os.path.exists(exp_name_fn):
            continue
        with open(exp_name_fn, "r") as f:
            candidate_exp_name = f.read()
        if candidate_exp_name == exp_name:
            checkpoint_dir = os.path.realpath(os.path.join(dn, "checkpoints"))
            checkpoint_dirs.append(checkpoint_dir)
    if len(checkpoint_dirs) == 0:
        return -1
    logger.info("Restore checkpoint: directories found: " + ", ".join(checkpoint_dirs))

    # make a sorted list of epochs to try loading
    epochs = list()
    for checkpoint_dir in checkpoint_dirs:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_*.pth"))
        epochs.extend(
            [int(os.path.basename(c).split("_")[-1][:-4]) for c in checkpoints]
        )
    epochs = list(set(epochs))
    epochs = list(reversed(sorted(epochs)))
    if len(epochs) == 0:
        logger.warning("Restore checkpoint: no suitable checkpoints were found")
        return -1

    # we want to restore initial initialization if something goes wrong
    backup_nets = dict()
    for name, net in nets.items():
        backup_nets[name] = copy.deepcopy(net.state_dict())
    backup_optims = dict()
    for name, optim in optims.items():
        backup_optims[name] = copy.deepcopy(optim.state_dict())
    for epoch_num in epochs:
        for checkpoint_dir in checkpoint_dirs:
            try:
                restore_checkpoint_from_dir(
                    checkpoint_dir, epoch_num, nets, optims, device
                )
                # if we are here, we successfully loaded the checkpoint
                logger.info(
                    f"Restore checkpoint: loaded checkpoint for epoch "
                    f"{epoch_num:04d} from {checkpoint_dir}"
                )
                return epoch_num
            except Exception as e:
                # just suppress because there might be a lot of such cases
                pass
        logger.warning(
            "Restore checkpoint: failed to load checkpoint for epoch "
            + f"{epoch_num:04d}"
        )

    # here we failed to load any checkpoint; hence restoring from backups
    for name, net in nets.items():
        net.load_state_dict(backup_nets[name])
    for name, optim in optims.items():
        optim.load_state_dict(backup_optims[name])
    logger.warning(
        "Restore checkpoint: failed to load any checkpoint, "
        "using standard initialization"
    )
    return -1


def remove_checkpoint(checkpoint_dir, epoch_num, nets, optims, remove_full):
    for name in list(nets.keys()) + list(optims.keys()):
        if name == "gen" and not remove_full:
            continue
        fn = os.path.join(checkpoint_dir, f"{name}_{epoch_num:04d}.pth")
        if os.path.exists(fn):
            os.remove(fn)
