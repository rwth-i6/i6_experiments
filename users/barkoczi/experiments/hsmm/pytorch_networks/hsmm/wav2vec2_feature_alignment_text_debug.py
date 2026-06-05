from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ForwardConfig:
    output_filename: str = "feature_alignment_debug.txt"
    target_layer: int = -1
    max_seqs_to_dump: int = 8
    max_frames_to_dump: int = 12
    feature_precision: int = 4
    alignment_subsample_factor: int = 2


def _subsample_alignment(labels: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return labels
    return labels[::factor]


def forward_init_hook(run_ctx, **kwargs):
    config = ForwardConfig(**kwargs.get("config", {}))
    run_ctx.target_layer = config.target_layer
    run_ctx.max_seqs_to_dump = config.max_seqs_to_dump
    run_ctx.max_frames_to_dump = config.max_frames_to_dump
    run_ctx.feature_precision = config.feature_precision
    run_ctx.alignment_subsample_factor = config.alignment_subsample_factor
    run_ctx.num_dumped = 0
    run_ctx.output_file = open(config.output_filename, "wt", encoding="utf-8")


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.output_file.close()


def forward_step(*, model, data, run_ctx, **kwargs):
    if run_ctx.num_dumped >= run_ctx.max_seqs_to_dump:
        return

    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)
    alignments = data.get("alignments")
    if alignments is None:
        raise ValueError("Alignments not found in data stream. Check hdf_stream_name in the dataset builder.")
    alignments_len = data["alignments:size1"].to(torch.long)
    tags = data["seq_tag"]

    all_hidden_states, output_lengths = model.extract_hidden_states(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transformed_hidden_states = model.transform_hidden_states(
        all_hidden_states,
        output_lengths,
        update_pca=False,
    )

    layer_idx = model.return_layers.index(run_ctx.target_layer)
    layer_features = transformed_hidden_states[layer_idx]

    for batch_idx, tag in enumerate(tags):
        if run_ctx.num_dumped >= run_ctx.max_seqs_to_dump:
            break

        feature_len = int(output_lengths[batch_idx].item())
        alignment_len = int(alignments_len[batch_idx].item())
        raw_alignments = alignments[batch_idx, :alignment_len].detach().cpu().to(torch.long)
        subsampled_alignments = _subsample_alignment(raw_alignments, run_ctx.alignment_subsample_factor)
        effective_len = min(feature_len, int(subsampled_alignments.shape[0]))
        feature_frames = layer_features[batch_idx, :effective_len].detach().cpu().numpy()
        printed_frames = feature_frames[: run_ctx.max_frames_to_dump]
        alignments_np = subsampled_alignments[:effective_len].numpy()

        feature_str = np.array2string(
            printed_frames,
            precision=run_ctx.feature_precision,
            threshold=printed_frames.size,
            max_line_width=200,
        )

        run_ctx.output_file.write(f"seq_tag: {tag}\n")
        run_ctx.output_file.write(f"feature_shape: {tuple(feature_frames.shape)}\n")
        run_ctx.output_file.write(
            f"alignment_lengths: raw={alignment_len}, subsampled={int(subsampled_alignments.shape[0])}, used={effective_len}\n"
        )
        run_ctx.output_file.write(
            f"feature_frames_shown: {printed_frames.shape[0]} of {feature_frames.shape[0]}\n"
        )
        run_ctx.output_file.write("features:\n")
        run_ctx.output_file.write(feature_str)
        run_ctx.output_file.write("\n")
        run_ctx.output_file.write(f"alignments: {alignments_np.tolist()}\n")
        run_ctx.output_file.write("\n")

        run_ctx.num_dumped += 1
