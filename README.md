# FFC-SE: Fast Fourier Convolution for Speech Enhancement

This is an official implementation of paper [FFC-SE: Fast Fourier Convolution for Speech Enhancement](https://www.isca-speech.org/archive/pdfs/interspeech_2022/shchekotov22_interspeech.pdf).

## Installation

Clone the repo and install requirements:

```
git clone https://github.com/SamsungLabs/ffc_se.git 
cd ffc_se
python -m venv ffc_se_env
. ffc_se_env/bin/activate
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
data.dir4inference=${inference_dir} checkpoint.checkpoint4inference=${checkpoint_path} \
data.root_dir=${path_to_root_dir} dataset.inference_1ch.val.noisy_wavs_dir=${path_to_noisy_wavs} \
dataset.inference_1ch.val.clean_wavs_dir=${path_to_clean_wavs}
```
Mandatory Args:
- `exp.config_dir`: directory containing yaml config (see [Configs](#configs) below)
- `exp.config`: config name in yaml format (see [Configs](#configs) below)
- `exp.name`: Name of experiment for wandb
- `data.dir4inference`: Inference directory where denoised audios are stored
- `checkpoint.checkpoint4inference`: Checkpoint path to corresponding to config model weights (see [Checkpoints](#checkpoints) below)
- `dataset.inference_1ch.val.noisy_wavs_dir`: path to noisy audios; if used in conjunction with `data.root_dir` the final path to noisy audio is: `${path_to_root_dir}/${path_to_noisy_wavs}`

Optional Args:
- `data.root_dir`: path to root dir for convenience if clean and noisy audios are stored in the same directory. None by default.
- `dataset.inference_1ch.val.clean_wavs_dir`: path to clean audios; adding this argument also **enables metric computation**; if provided, the folders with noisy and clean wavs should have the same structure (i.e. audios in both noisy and clean directories should be named the same); 
the final path to clean audios if used with `data.root_dir` is: `${path_to_root_dir}/${path_to_clean_wavs}`

We use W&B for logging, so if you want to disable it just put `WANDB_MODE=disabled` before the command.

If you wand to compute WV-MOS score for unlabeled data (i.e. without clean reference), you can pass the same value as in `dataset.inference_1ch.val.noisy_wavs_dir` to `dataset.inference_1ch.val.clean_wavs_dir`. However, other metrics would also be computed (incorrectly, though).

If you launch full inference with metric computation, inference on voicebank dataset should take approximately 30 min, on dns-blind ~ 1h 50 min.

## Checkpoints 

You can find and download checkpoints in Releases tab.

`checkpoints` folder contents are as follows:
```
checkpoints/
├── dns_ckpt
│   ├── ffc_ae_v0_dns_ckpt.pth
│   ├── ffc_ae_v1_dns_ckpt.pth
│   └── ffc_unet_dns_ckpt.pth
└── vb_ckpt
    ├── ffc_ae_v0_vb_ckpt.pth
    ├── ffc_ae_v1_vb_ckpt.pth
    └── ffc_unet_vb_ckpt.pth
```

## Configs
Available configs can be found in `configs` directory:
```
configs/
├── ffc_se_ae_v0.yaml
├── ffc_se_ae_v1.yaml
└── ffc_se_unet.yaml
```

## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{shchekotov22_interspeech,
  author={Ivan Shchekotov and Pavel K. Andreev and Oleg Ivanov and Aibek Alanov and Dmitry Vetrov},
  title={{FFC-SE: Fast Fourier Convolution for Speech Enhancement}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={1188--1192},
  doi={10.21437/Interspeech.2022-603}
}
```

Copyright (c) 2022 Samsung Electronics
