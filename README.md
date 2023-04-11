# HiFi++: a Unified Framework for Bandwidth Extension and Speech Enhancement

This is an official implementation of paper [HiFi++: a Unified Framework for Bandwidth Extension and Speech Enhancement](https://arxiv.org/pdf/2203.13086.pdf).

## Installation

Clone the repo and install requirements:

```
git clone https://github.com/SamsungLabs/hifi_plusplus.git 
cd ffc_se
python -m venv hifi_env
. hifi_env/bin/activate
pip install -r requirements.txt
```

## Downloading weights for computing metrics
You need to download weights of Wave2Vec2Mos network which is used to compute MOSNet metrics. These weights should
be located in **"weights"** directory. Run `download_extract_weights.sh` script, which does everything for you:
```
chmod +x download_extract_weights.sh
./download_extract_weights.sh
```

## Inference
The inference can be launched with the following command:
```
python main.py exp.config_dir=${config_dir} exp.config=${config_name} exp.name=${exp_name} \
data.dir4inference=${inference_dir} checkpoint.checkpoint4inference=${checkpoint_path}
```
Mandatory Args:
- `exp.config_dir`: directory containing yaml config (see [Configs](#configs) below)
- `exp.config`: config name in yaml format (see [Configs](#configs) below)
- `exp.name`: Name of experiment for wandb
- `data.dir4inference`: Inference directory where denoised audios are stored

Optional Args:
- `data.voicebank_dir`: path to dir where Voicebank-DEMAND dataset is stored, it assumed that noisy audios (for the model to be tested on) are in `${data.voicebank_dir}/noisy_testset_wav` and clean audious are in `${data.voicebank_dir}/clean_testset_wav` (required for metrics computation). Use this argument when inferencing denoising checkpoint.
- `data.vctk_wavs_dir`: path to dir where VCTK dataset wav files are stored. Use this argument when inferencing bandwidth extension checkpoints.

We use W&B for logging, so if you want to disable it just put `WANDB_MODE=disabled` before the command.

## Checkpoints 

You can find and download checkpoints in Releases tab.


## Configs
Available configs can be found in `configs` directory:
```
configs/
├── bwe_1khz.yaml
├── bwe_2khz.yaml
├── bwe_4khz.yaml
└── denoising.yaml
```

## Citation
If you find this work useful in your research, please cite:
```
@article{andreev2022hifi++,
  title={Hifi++: a unified framework for bandwidth extension and speech enhancement},
  author={Andreev, Pavel and Alanov, Aibek and Ivanov, Oleg and Vetrov, Dmitry},
  journal={arXiv preprint arXiv:2203.13086},
  year={2022}
}
```

Copyright (c) 2023 Samsung Electronics
