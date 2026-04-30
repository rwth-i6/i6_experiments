from typing import Any, Dict

import torch


def forward_init_hook(run_ctx, **kwargs):
    config = dict(kwargs.get("config", {}))
    run_ctx.output_filename = config.get("output_filename", "init_logprobs_debug.txt")
    run_ctx.output_file = open(run_ctx.output_filename, "wt")
    run_ctx.did_write = False


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.output_file.close()


def _format_topk(values: torch.Tensor, indices: torch.Tensor) -> str:
    pairs = [f"({int(index)}, {float(value):.6f})" for value, index in zip(values.tolist(), indices.tolist())]
    return "[" + ", ".join(pairs) + "]"


def forward_step(*, model, data: Dict[str, Any], run_ctx, **kwargs):

    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"]

    with torch.no_grad():
        log_probs_list, output_lengths = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

    seq_tags = data.get("seq_tag")
    seq_tag_0 = seq_tags[0] if seq_tags is not None else "<unknown>"
    if isinstance(seq_tag_0, (bytes, bytearray)):
        seq_tag_0 = seq_tag_0.decode("utf8")
    else:
        seq_tag_0 = str(seq_tag_0)

    output_len_0 = int(output_lengths[0])
    run_ctx.output_file.write(f"seq_tag={seq_tag_0}\n")
    run_ctx.output_file.write(f"output_len={output_len_0}\n")

    for layer_index, layer_log_probs in zip(model.return_layers, log_probs_list):
        run_ctx.output_file.write(f"layer={layer_index}\n")
        seq_log_probs = layer_log_probs[0]
        num_frames = min(5, seq_log_probs.shape[0], output_len_0)
        for frame_index in range(num_frames):
            top_values, top_indices = torch.topk(seq_log_probs[frame_index], k=min(5, seq_log_probs.shape[-1]))

            print(layer_index, top_values)
            run_ctx.output_file.write(
                f"frame={frame_index} top5={_format_topk(top_values.detach().cpu(), top_indices.detach().cpu())}\n"
            )

