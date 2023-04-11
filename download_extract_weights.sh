#!/bin/bash

WEIGHTS_DIR="weights"
mkdir ./${WEIGHTS_DIR}

WAV2VEC_WEIGHTS="https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1"

wget ${WAV2VEC_WEIGHTS} -O "${WEIGHTS_DIR}/wave2vec2mos.pth"

