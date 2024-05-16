import multiprocessing
import torch
import numpy as np
from returnn.datasets.hdf import SimpleHDFWriter
from . import commons

def forward_init_hook_statistics(run_ctx, **kwargs):
    run_ctx.total_mean = 0
    run_ctx.total_var = 0
    run_ctx.total_max = torch.tensor(-np.inf)
    run_ctx.total_min = torch.tensor(np.inf)
    run_ctx.num_of_obs = 0

def forward_finish_hook_statistics(run_ctx, **kwargs):
    with open("output.hdf", "w+") as f:
        f.write("total, mean, var, max, min \n")
        f.write(
            f"{run_ctx.num_of_obs}, {str(float(run_ctx.total_mean))}, {str(float(run_ctx.total_var))}, {str(float(run_ctx.total_max))}, {str(float(run_ctx.total_min))}"
        )

def forward_step_statistics(*, model, data, run_ctx, **kwargs):
    raw_audio = data["audio_features"]  # [B, N] (sparse)
    raw_audio_len = data["audio_features:size1"]  # [B]
    

    squeezed_audio = torch.squeeze(raw_audio)
    y, y_lengths = model.feature_extraction(squeezed_audio, raw_audio_len)  # [B, T, F]
    y = y.transpose(1,2)
    y_max_length = y.size(2)

    y, y_lengths, y_max_length = model.preprocess(y, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(torch.int32)

    # y = torch.nn.functional.l1_loss(y_hat * z_mask, y * z_mask, reduction="none")  # [B, F, T]
    current_num_of_obs = y.shape[1] * y_lengths.sum()  # F * total_number_of_frames_in_batch

    old_mean = run_ctx.total_mean

    masked_y = y * z_mask
    current_mean = (
        masked_y.sum() / current_num_of_obs
    )  # This considers the masking by only using the mean over all unmasked elements

    current_var = (masked_y - current_mean).pow(2).sum() / (
        current_num_of_obs - 1
    )  # Variance over unmasked elements with bias correction 1

    run_ctx.total_mean = ((run_ctx.num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * old_mean) + (
        (current_num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * current_mean
    )

    run_ctx.total_var = (
        (run_ctx.num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * run_ctx.total_var
        + ((current_num_of_obs / (run_ctx.num_of_obs + current_num_of_obs)) * current_var)
        + ((run_ctx.num_of_obs * current_num_of_obs) / (run_ctx.num_of_obs + current_num_of_obs) ** 2)
        * (old_mean - current_mean) ** 2
    )

    run_ctx.total_max = torch.max(run_ctx.total_max, masked_y.max())

    run_ctx.total_min = torch.min(
        run_ctx.total_min, (masked_y + (-1 * z_mask + 1) * torch.tensor(float("inf")).nan_to_num(0.0)).min()
    )  # Masked Min operation

    run_ctx.num_of_obs += current_num_of_obs
