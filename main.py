import os

import torch

import arguments
import trainer


def setup_seed(seed, cudnn_benchmark_off):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_benchmark_off:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main(config):
    setup_seed(config.exp.seed, config.exp.cudnn_benchmark_off)

    current_trainer = trainer.trainers[config.training.trainer](config)

    current_trainer.setup_logger()
    current_trainer.setup_dataset()
    current_trainer.setup_loaders()
    current_trainer.setup_networks()
    current_trainer.setup_metrics()

    current_trainer.train_loop()


if __name__ == "__main__":
    # limit CPU usage
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OPENMP_NUM_THREADS"] = "1"
    torch.multiprocessing.set_start_method("spawn")

    config = arguments.load_config()
    main(config)
