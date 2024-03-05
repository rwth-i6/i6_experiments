#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:59:45 2020

@author: krishna

"""


import torch.nn as nn
from .tdnn_cnn import TDNN
import torch
import torch.nn.functional as F
import numpy as np

from .feature_extraction import DbMelFeatureExtraction
from ..x_vectors.feature_config import DbMelFeatureExtractionConfig


class Model(nn.Module):
    def __init__(self, input_dim=40, num_classes=8, **kwargs):
        super(Model, self).__init__()
        self.tdnn1 = TDNN(
            input_dim=input_dim,
            output_dim=512,
            context_size=5,
            dilation=1,
            dropout_p=0.5,
            batch_norm=kwargs["batch_norm"],
        )
        self.tdnn2 = TDNN(
            input_dim=512, output_dim=512, context_size=3, dilation=2, dropout_p=0.5, batch_norm=kwargs["batch_norm"]
        )
        self.tdnn3 = TDNN(
            input_dim=512, output_dim=512, context_size=2, dilation=3, dropout_p=0.5, batch_norm=kwargs["batch_norm"]
        )
        self.tdnn4 = TDNN(
            input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0.5, batch_norm=kwargs["batch_norm"]
        )
        self.tdnn5 = TDNN(
            input_dim=512, output_dim=1500, context_size=1, dilation=1, dropout_p=0.5, batch_norm=kwargs["batch_norm"]
        )
        #### Frame levelPooling
        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

        fe_config = DbMelFeatureExtractionConfig.from_dict(kwargs["fe_config"])
        self.feature_extraction = DbMelFeatureExtraction(config=fe_config)

    def forward(self, raw_audio, raw_audio_lengths):
        with torch.no_grad():
            squeezed_audio = torch.squeeze(raw_audio)
            x, x_lengths = self.feature_extraction(squeezed_audio, raw_audio_lengths)  # [B, T, F]

        x = x.transpose(1, 2)
        tdnn1_out = self.tdnn1(x)
        # return tdnn1_out
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        mean = torch.mean(tdnn5_out, 2)
        std = torch.std(tdnn5_out, 2)
        stat_pooling = torch.cat((mean, std), 1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        output = self.output(x_vec)
        predictions = self.softmax(output)
        return output, predictions, x_vec


def train_step(*, model: Model, data, run_ctx, **kwargs):
    tags = data["seq_tag"]
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    # phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    # phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    speaker_labels = data["speaker_labels"][indices, :].long()  # [B, 1] (sparse)
    tags = list(np.array(tags)[indices.detach().cpu().numpy()])

    logits, _, _ = model(audio_features, audio_features_len)
    loss = F.cross_entropy(logits, speaker_labels.squeeze(1))

    run_ctx.mark_as_loss(name="ce", loss=loss)
