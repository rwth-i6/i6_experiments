"""
Greedy CTC decoder without any extras
"""

import time
import numpy as np
import torch
from torch import nn


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(
        vocab_file=kwargs["returnn_vocab"], unknown_label=None)
    run_ctx.labels = vocab.labels

    run_ctx.running_audio_len_s = 0
    run_ctx.total_time = 0

def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    print("Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s))


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
    run_ctx.running_audio_len_s += audio_len_batch

    am_start = time.time()
    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    batch_indices = []
    for lp, l in zip(logprobs, audio_features_len):
        batch_indices.append(torch.unique_consecutive(torch.argmax(lp[:l], dim=-1), dim=0).detach().cpu().numpy())

    am_time = time.time() - am_start
    run_ctx.total_time += am_time
    print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time, am_time / audio_len_batch))

    tags = data["seq_tag"]


    for indices, tag in zip(batch_indices, tags):
        print(indices)
        sequence = [run_ctx.labels[idx] for idx in indices if idx < len(run_ctx.labels)]
        sequence = [s for s in sequence if (not s.startswith("<") and not s.startswith("["))]
        text = " ".join(sequence).replace("@@ ","")
        print(text)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))