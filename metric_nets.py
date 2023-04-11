"""
Based on
    https://github.com/AndreevP/speech_distances
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


def extract_prefix(prefix, weights):
    result = OrderedDict()
    for key in weights:
        if key.find(prefix) == 0:
            result[key[len(prefix) :]] = weights[key]
    return result


class Wav2Vec2MOS(nn.Module):
    sample_rate = 16_000

    def __init__(self, path, freeze=True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.freeze = freeze

        self.dense = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

        if self.freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.load_state_dict(extract_prefix("model.", torch.load(path)["state_dict"]))
        self.eval()
        self.cuda()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def forward(self, x):
        x = self.encoder(x)["last_hidden_state"]  # [Batch, time, feats]
        x = self.dense(x)  # [batch, time, 1]
        x = x.mean(dim=[1, 2], keepdims=True)  # [batch, 1, 1]
        return x

    def train(self, mode):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()

    def calculate(self, samples):
        pred_mos = []
        for s in samples:
            x = self.processor(
                s.cpu(),
                return_tensors="pt",
                padding=True,
                sampling_rate=self.sample_rate,
            ).input_values
            with torch.no_grad():
                res = self.forward(x.cuda()).mean()
            pred_mos.append(res.item())
        return np.mean(pred_mos)
