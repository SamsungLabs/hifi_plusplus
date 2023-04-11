import os

import matplotlib.pyplot as plt
import omegaconf
import soundfile as sf
import torch
import torch.nn.functional
import torch.utils.data

import datasets
import log_utils
import metrics
import models
import utils

trainers = utils.ClassRegistry()


@trainers.add_to_registry(name="base")
class BaseInferencer:
    def __init__(self, config):
        # initialized in setup_logger()
        self.logger = None
        self.run_dir = None

        # initialized in setup_dataset()
        self.dataset = None

        # initialized in setup_loaders()
        self.loaders = None

        # initialized in setup_networks()
        self.gen = None
        self.checkpoint_dir = None

        # initialized in setup_metrics()
        self.metrics = None

        self.config = config

    def setup_logger(self):
        config_for_logger = omegaconf.OmegaConf.to_container(self.config)
        config_for_logger["PID"] = os.getpid()
        exp_logger = log_utils.WandbLogger(
            project=self.config.exp.project,
            name=self.config.exp.name,
            dir=self.config.exp.root,
            tags=tuple(self.config.exp.tags) if self.config.exp.tags else None,
            notes=self.config.exp.notes,
            config=config_for_logger,
        )
        self.run_dir = exp_logger.run_dir
        console_logger = log_utils.ConsoleLogger(self.config.exp.name)
        self.logger = log_utils.LoggingManager(exp_logger, console_logger)

    def setup_dataset(self):
        self.dataset = dict()
        for data_type in ["val"]:
            dataset = datasets.datasets[self.config.data.name](
                **self.config.dataset[self.config.data.name][data_type]
            )
            self.dataset[data_type] = dataset

    def setup_loaders(self):
        self.loaders = dict()
        loader = self.config.data.loader
        loader_args = self.config.loader[loader]
        for data_type in ["val"]:
            self.loaders[data_type] = datasets.loaders[loader](
                self.dataset[data_type],
                **loader_args[data_type],
            )

    def setup_networks(self):
        x, _ = next(self.loaders["val"])
        print("\nGenerator:")
        self.gen = models.generators[self.config.gen.model](
            **self.config.gennets[self.config.gen.model],
        )
        log_utils.print_network(self.gen, "gen", x, self.logger)
        self.gen = self.gen.to(self.config.training.device)


@trainers.add_to_registry(name="audio2audio_infer")
class Audio2AudioHiFiInferencer(BaseInferencer):
    def setup_metrics(self):
        self.metrics = (
            metrics.MOSNet(sr=self.config.data.sampling_rate),
            metrics.ScaleInvariantSignalToDistortionRatio(),
            metrics.SignalToNoiseRatio(),
            metrics.LSD(),
            metrics.STOI(sr=self.config.data.sampling_rate),
            metrics.PESQ(sr=self.config.data.sampling_rate),
            metrics.CSIG(),
            metrics.CBAK(),
            metrics.COVL(),
        )

    def train_loop(self):
        ckpt_state_dict = torch.load(
            self.config.checkpoint.checkpoint4inference,
            map_location=self.config.training.device,
        )

        self.gen.load_state_dict(ckpt_state_dict)

        self.setup_loaders()
        self.gen = self.gen.eval()
        epoch_info = log_utils.StreamingMeans()
        epoch_num = 0

        val_predictions = self.compute_metrics(epoch_num, epoch_info)

        self.logger.log_epoch(epoch_num, epoch_info)

        os.makedirs(self.config.data.dir4inference, exist_ok=True)
        for audio, names in zip(val_predictions, self.dataset["val"].audio_files):
            if isinstance(names, list) or isinstance(names, tuple):
                name = names[0]
            else:
                name = names
            print(name)
            sf.write(
                os.path.join(self.config.data.dir4inference, name.split("/")[-1]),
                audio,
                self.config.data.sampling_rate,
            )

    def compute_metrics(self, epoch_num, epoch_info):
        real_samples = []
        fake_samples = []
        for i in range(len(self.dataset["val"])):
            x, y = self.dataset["val"][i]
            x = x[None].to(self.config.training.device)
            y = y[None].to(self.config.training.device)
            pad_size = utils.closest_power_of_two(x.shape[-1]) - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_size))

            with torch.no_grad():
                y_hat = self.gen(x)
            real_samples.append(y.squeeze().cpu().numpy())
            fake_samples.append(y_hat.squeeze().cpu().numpy())

        if self.dataset["val"].clean_wavs_dir:
            scores = metrics.calculate_all_metrics(
                fake_samples, real_samples, self.metrics
            )
            for metric in self.metrics:
                epoch_info[f"inference_metrics_mean/{metric.name}"] = scores[
                    metric.name
                ][0]
                epoch_info[f"inference_metrics_std/{metric.name}"] = scores[
                    metric.name
                ][1]
        return fake_samples
