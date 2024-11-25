import multiprocessing
import torch
import numpy as np
from returnn.datasets.hdf import SimpleHDFWriter
from . import commons


def forward_init_hook_invertibility(run_ctx, **kwargs):
    run_ctx.total_mae = 0
    run_ctx.total_ae_var = 0
    run_ctx.total_ae_max = torch.tensor(-np.inf)
    run_ctx.total_ae_min = torch.tensor(np.inf)
    run_ctx.num_of_obs = 0


def forward_finish_hook_invertibility(run_ctx, **kwargs):
    with open("output.hdf", "w+") as f:
        f.write("total, mean, var, max, min \n")
        f.write(
            f"{run_ctx.num_of_obs}, {str(float(run_ctx.total_mae))}, {str(float(run_ctx.total_ae_var))}, {str(float(run_ctx.total_ae_max))}, {str(float(run_ctx.total_ae_min))}"
        )


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

    squeezed_audio = torch.squeeze(raw_audio)
    y, y_lengths = model.feature_extraction(squeezed_audio, raw_audio_len)  # [B, T, F]
    y = y.transpose(1, 2)  # [B, F, T]

    if hasattr(model, "x_vector"):
        _, _, g = model.x_vector(y, y_lengths)

        if hasattr(model, "x_vector_bottleneck"):
            g = model.x_vector_bottleneck(g)
    elif hasattr(model, "emb_g"):
        g = torch.nn.functional.normalize(model.emb_g(g.squeeze(-1))).unsqueeze(-1)
    else:
        g = None

    y_max_length = y.size(2)

    y, y_lengths, y_max_length = model.preprocess(y, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(torch.int32)

    z, _ = model.decoder(y, z_mask, g=g, reverse=False)
    y_hat, _ = model.decoder(z, z_mask, g=g, reverse=True)

    mae = torch.nn.functional.l1_loss(y_hat * z_mask, y * z_mask, reduction="none")  # [B, F, T]

    current_num_of_obs = y_hat.shape[1] * y_lengths.sum()  # F * total_number_of_frames_in_batch

    old_mae = run_ctx.total_mae

    current_mae = (
        mae.sum() / current_num_of_obs
    )  # This considers the masking by only using the mean over all unmasked elements

    current_var = (mae - current_mae).pow(2).sum() / (
        current_num_of_obs - 1
    )  # Variance over unmasked elements with bias correction 1

    run_ctx.total_mae = ((run_ctx.num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * old_mae) + (
        (current_num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * current_mae
    )

    run_ctx.total_ae_var = (
        (run_ctx.num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * run_ctx.total_ae_var
        + ((current_num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * current_var)
        + ((run_ctx.num_of_obs * current_num_of_obs) / (run_ctx.num_of_obs + current_num_of_obs) ** 2)
        * (old_mae - current_mae) ** 2
    )

    run_ctx.total_ae_max = torch.max(run_ctx.total_ae_max, mae.max())

    run_ctx.total_ae_min = torch.min(
        run_ctx.total_ae_min, (mae + (-1 * z_mask + 1) * torch.tensor(float("inf")).nan_to_num(0.0)).min()
    )  # Masked Min operation

    run_ctx.num_of_obs += current_num_of_obs

def forward_init_hook_asr_invertibility(run_ctx, **kwargs):
    run_ctx.total_mae = 0
    run_ctx.total_ae_var = 0
    run_ctx.total_ae_max = torch.tensor(-np.inf)
    run_ctx.total_ae_min = torch.tensor(np.inf)
    run_ctx.num_of_obs = 0


def forward_finish_hook_asr_invertibility(run_ctx, **kwargs):
    with open("output.hdf", "w+") as f:
        f.write("total, mean, var, max, min \n")
        f.write(
            f"{run_ctx.num_of_obs}, {str(float(run_ctx.total_mae))}, {str(float(run_ctx.total_ae_var))}, {str(float(run_ctx.total_ae_max))}, {str(float(run_ctx.total_ae_min))}"
        )


def forward_step_asr_invertibility(*, model, data, run_ctx, **kwargs):
    raw_audio = data["audio_features"]  # [B, N] (sparse)
    raw_audio_len = data["audio_features:size1"]  # [B]
    phonemes = data["phonemes"]
    phonemes_len = data["phonemes:size1"]

    squeezed_audio = torch.squeeze(raw_audio)
    y, y_lengths = model.feature_extraction(squeezed_audio, raw_audio_len)  # [B, T, F]
    y = y.transpose(1, 2)  # [B, F, T]

    g = None

    y_max_length = y.size(2)

    y, y_lengths, y_max_length = model.preprocess(y, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(torch.int32)

    z, _ = model.decoder(y, z_mask, g=g, reverse=False)
    y_hat, _ = model.decoder(z, z_mask, g=g, reverse=True)

    mae = torch.nn.functional.l1_loss(y_hat * z_mask, y * z_mask, reduction="none")  # [B, F, T]

    current_num_of_obs = y_hat.shape[1] * y_lengths.sum()  # F * total_number_of_frames_in_batch

    old_mae = run_ctx.total_mae

    current_mae = (
        mae.sum() / current_num_of_obs
    )  # This considers the masking by only using the mean over all unmasked elements

    current_var = (mae - current_mae).pow(2).sum() / (
        current_num_of_obs - 1
    )  # Variance over unmasked elements with bias correction 1

    run_ctx.total_mae = ((run_ctx.num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * old_mae) + (
        (current_num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * current_mae
    )

    run_ctx.total_ae_var = (
        (run_ctx.num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * run_ctx.total_ae_var
        + ((current_num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * current_var)
        + ((run_ctx.num_of_obs * current_num_of_obs) / (run_ctx.num_of_obs + current_num_of_obs) ** 2)
        * (old_mae - current_mae) ** 2
    )

    run_ctx.total_ae_max = torch.max(run_ctx.total_ae_max, mae.max())

    run_ctx.total_ae_min = torch.min(
        run_ctx.total_ae_min, (mae + (-1 * z_mask + 1) * torch.tensor(float("inf")).nan_to_num(0.0)).min()
    )  # Masked Min operation

    run_ctx.num_of_obs += current_num_of_obs
