from dataclasses import dataclass

from torch import nn


class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


@dataclass
class ForwardConfig:
    output_filename: str = "dataset_debug.txt"
    alignment_key: str = "alignments"


def forward_init_hook(run_ctx, **kwargs):
    config = ForwardConfig(**kwargs.get("config", {}))
    run_ctx.output_filename = config.output_filename
    run_ctx.alignment_key = config.alignment_key
    run_ctx.output_file = open(config.output_filename, "wt")


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.output_file.close()


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio_len = data["raw_audio:size1"]
    labels = data.get("labels")
    labels_len = data.get("labels:size1")
    alignments = data.get(run_ctx.alignment_key)
    alignments_len = data.get(f"{run_ctx.alignment_key}:size1")
    tags = data["seq_tag"]

    for batch_idx, tag in enumerate(tags):
        run_ctx.output_file.write(f"{tag}\n")
        run_ctx.output_file.write(f"input_len: {int(raw_audio_len[batch_idx])}\n")
        if labels is not None:
            label_len = int(labels_len[batch_idx])
            run_ctx.output_file.write(f"labels: {labels[batch_idx, :label_len].detach().cpu().tolist()}\n")
        else:
            run_ctx.output_file.write("labels: None\n")
        if alignments is not None:
            if alignments_len is not None:
                alignment_len = int(alignments_len[batch_idx])
                alignment_values = alignments[batch_idx, :alignment_len]
            else:
                alignment_values = alignments[batch_idx]
            run_ctx.output_file.write(f"{run_ctx.alignment_key}: {alignment_values.detach().cpu().tolist()}\n")
        else:
            run_ctx.output_file.write(f"{run_ctx.alignment_key}: None\n")
        run_ctx.output_file.write("\n")
