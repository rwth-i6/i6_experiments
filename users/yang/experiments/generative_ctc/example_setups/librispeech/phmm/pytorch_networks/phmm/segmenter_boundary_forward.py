from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ForwardConfig:
    output_mode: Literal["scores", "boundaries"] = "scores"
    threshold: float = 0.0
    output_filename: str = "output.hdf"
    boundary_frame_offset: int = 1


def forward_init_hook(run_ctx, **kwargs):
    from returnn.datasets.hdf import SimpleHDFWriter

    run_ctx.config = ForwardConfig(**kwargs.get("config", {}))
    if run_ctx.config.output_mode == "scores":
        run_ctx.hdf_writer = SimpleHDFWriter(run_ctx.config.output_filename, dim=1, ndim=2)
    elif run_ctx.config.output_mode == "boundaries":
        run_ctx.hdf_writer = SimpleHDFWriter(run_ctx.config.output_filename, dim=None, ndim=1)
    else:
        raise ValueError(f"Unsupported output_mode {run_ctx.config.output_mode!r}")


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.hdf_writer.close()


def _normalize_seq_tags(seq_tags):
    normalized = []
    for tag in seq_tags:
        if isinstance(tag, (bytes, bytearray)):
            normalized.append(tag.decode("utf8"))
        else:
            normalized.append(str(tag))
    return normalized


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)

    with torch.no_grad():
        features, feature_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
        features = F.normalize(features.float(), p=2.0, dim=-1)
        scores = -(features[:, :-1] * features[:, 1:]).sum(dim=-1)  # [B, T-1]

    seq_tags = _normalize_seq_tags(data["seq_tag"])
    score_lens = torch.clamp(feature_len - 1, min=0).detach().cpu().tolist()

    if run_ctx.config.output_mode == "scores":
        scores_np = scores.detach().cpu().numpy().astype("float32")[..., None]
        run_ctx.hdf_writer.insert_batch(
            inputs=scores_np,
            seq_len=score_lens,
            seq_tag=seq_tags,
        )
        return

    boundary_sequences = []
    for seq_scores, seq_len in zip(scores, score_lens):
        valid_scores = seq_scores[:seq_len]
        boundary_frames = torch.nonzero(valid_scores > run_ctx.config.threshold, as_tuple=False).flatten()
        boundary_frames = boundary_frames + run_ctx.config.boundary_frame_offset
        boundary_sequences.append(boundary_frames.detach().cpu().numpy().astype("int32"))

    max_boundary_len = max(1, max((len(boundaries) for boundaries in boundary_sequences), default=0))
    padded = np.zeros((len(boundary_sequences), max_boundary_len), dtype="int32")
    boundary_lens = []
    for batch_idx, boundaries in enumerate(boundary_sequences):
        boundary_lens.append(len(boundaries))
        if len(boundaries) > 0:
            padded[batch_idx, : len(boundaries)] = boundaries

    run_ctx.hdf_writer.insert_batch(
        inputs=padded,
        seq_len=boundary_lens,
        seq_tag=seq_tags,
    )
