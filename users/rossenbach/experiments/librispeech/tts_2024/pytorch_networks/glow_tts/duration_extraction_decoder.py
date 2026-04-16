import os

import numpy as np
import torch
import torchaudio
import subprocess
import multiprocessing as mp
from ..vocoder.simple_gl.blstm_gl_predictor import Model
from ..tts_shared.corpus import Corpus, Recording, Segment

from returnn.datasets.util.hdf import SimpleHDFWriter



def forward_init_hook(run_ctx, **kwargs):
    run_ctx.hdf_writer = SimpleHDFWriter("durations.hdf", dim=None, ndim=1)


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.hdf_writer.close()


def forward_step(*, model: Model, data, run_ctx, **kwargs):
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)
    
    raw_samples = data["raw_audio"]  # [B, T, F]
    audio_features_len = data["raw_audio:size1"]  # [B]

    tags = data["seq_tag"]

    (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (y_lengths, feature_lengths), (attn, logw, logw_) = model(
        phonemes,
        phonemes_len,
        raw_audio=raw_samples,
        raw_audio_lengths=audio_features_len,
        g=speaker_labels,
    )

    # attn is of shape B, 1, N, T
    durations = torch.sum(attn, dim=-1)  # B, 1, N
    durations = durations * x_mask
    durations = durations.squeeze(dim=1)  # B, N

    durations = durations.cpu().numpy()
    phonemes_len = phonemes_len.cpu().numpy()
    feature_lengths = feature_lengths.cpu().numpy()

    for duration, duration_len, feature_len, tag in zip(durations, phonemes_len, feature_lengths, tags):
        shorted_duration = duration[:duration_len]
        length_adjustment = feature_len - np.sum(shorted_duration)
        shorted_duration[-1] += length_adjustment

        shorted_duration = np.expand_dims(shorted_duration, 0)
        duration_len = np.expand_dims(duration_len, 0)
        run_ctx.hdf_writer.insert_batch(inputs=shorted_duration, seq_len=duration_len, seq_tag=[tag])
