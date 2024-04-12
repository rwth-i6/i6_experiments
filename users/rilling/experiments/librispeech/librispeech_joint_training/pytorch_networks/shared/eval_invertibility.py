import multiprocessing
import torch
import numpy as np
from returnn.datasets.hdf import SimpleHDFWriter

def forward_init_hook_invertibility(run_ctx, **kwargs):
    run_ctx.batch_mae = []


def forward_finish_hook_invertibility(run_ctx, **kwargs):
    all_batch_mse = torch.Tensor(run_ctx.batch_mae).mean()
    with open("output.hdf", "w+") as f:
        f.write(str(all_batch_mse))

def forward_step_invertibility(*, model, data, run_ctx, **kwargs):
    raw_audio = data["audio_features"]  # [B, N] (sparse)
    raw_audio_len = data["audio_features:size1"]  # [B]
    phonemes = data["phonemes"]
    phonemes_len = data["phonemes:size1"]

    if "xvectors" in data:
        g = data["xvectors"]
    elif "speaker_labels" in data:
        g = data["speaker_labels"]
    else:
        raise Exception("Missing speaker embedding!")

    tags = data["seq_tag"]

    y_hat, y = model(
        x=phonemes,
        x_lengths=phonemes_len,
        raw_audio=raw_audio,
        raw_audio_lengths=raw_audio_len,
        g=g,
        invertibility_check=True,
    )

    mse = torch.nn.functional.l1_loss(y_hat, y)
    run_ctx.batch_mae.append(mse)
