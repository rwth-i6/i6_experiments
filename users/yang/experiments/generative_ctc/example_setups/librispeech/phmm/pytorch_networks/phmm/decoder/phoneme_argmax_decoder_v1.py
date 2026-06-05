"""
Simple phoneme-level argmax decoder.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DecoderConfig:
    phoneme_vocab: Any
    silence_label: str = "[SILENCE]"
    data_key: str = "data"


def _read_vocab(path):
    from returnn.util.basic import cf

    labels = []
    with open(cf(path), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Unexpected vocab line: {line!r}")
            idx = int(parts[0])
            label = parts[1]
            if idx != len(labels):
                raise ValueError(f"Expected vocab index {len(labels)}, got {idx} in {line!r}")
            labels.append(label)
    return labels


def _collapse(labels):
    out = []
    prev = None
    for label in labels:
        if label == prev:
            continue
        out.append(label)
        prev = label
    return out


def forward_init_hook(run_ctx, **kwargs):
    config = DecoderConfig(**kwargs["config"])
    run_ctx.labels = _read_vocab(config.phoneme_vocab)
    if config.silence_label not in run_ctx.labels:
        raise ValueError(f"Silence label {config.silence_label!r} not found in {config.phoneme_vocab!r}")
    run_ctx.silence_label = config.silence_label
    run_ctx.data_key = config.data_key
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()


def forward_step(*, model, data, run_ctx, **kwargs):
    import torch

    model_input = data[run_ctx.data_key]
    input_len = data[f"{run_ctx.data_key}:size1"].to(torch.long)
    model_out = model(model_input, input_len)
    if isinstance(model_out, tuple):
        log_probs, output_len = model_out
    else:
        log_probs = model_out
        output_len = torch.clamp(input_len, max=log_probs.shape[1])

    if log_probs.shape[-1] != len(run_ctx.labels):
        raise ValueError(f"Model output dim {log_probs.shape[-1]} does not match vocab dim {len(run_ctx.labels)}")

    pred = torch.argmax(log_probs, dim=-1).detach().cpu().tolist()
    output_len = torch.clamp(output_len, max=log_probs.shape[1]).detach().cpu().tolist()
    tags = data["seq_tag"]

    for seq_pred, seq_len, tag in zip(pred, output_len, tags):
        labels = [run_ctx.labels[int(idx)] for idx in seq_pred[: int(seq_len)]]
        labels = [label for label in labels if label != run_ctx.silence_label]
        hyp = " ".join(_collapse(labels))
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(hyp)))
