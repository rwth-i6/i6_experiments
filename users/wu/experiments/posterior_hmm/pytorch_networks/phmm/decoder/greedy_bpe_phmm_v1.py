"""
Greedy decoder for posterior HMM BPE models.

Uses the phoneme inventory order from the Bliss lexicon as the label inventory.
The non-emitting silence state is expected to be present in that inventory and is
filtered after collapsing consecutive frame-level argmax labels.
"""

from dataclasses import dataclass
import time
import torch


@dataclass
class DecoderConfig:
    lexicon: str
    silence_label: str = "[SILENCE]"


@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True


def forward_init_hook(run_ctx, **kwargs):
    config = DecoderConfig(**kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    from i6_core.lib import lexicon

    lex = lexicon.Lexicon()
    lex.load(config.lexicon)
    run_ctx.labels = list(lex.phonemes.keys())
    run_ctx.label_by_index = dict(enumerate(run_ctx.labels))
    run_ctx.silence_label = config.silence_label

    if config.silence_label not in lex.phonemes:
        raise ValueError(f"Silence label {config.silence_label!r} not found in lexicon {config.lexicon!r}")
    run_ctx.silence_idx = run_ctx.labels.index(config.silence_label)

    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_time = 0

    run_ctx.print_hypothesis = extra_config.print_hypothesis


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    if run_ctx.print_rtf:
        print("Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s))


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:seq_len"]  # [B]

    audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000

    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s += audio_len_batch
        am_start = time.time()

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    if isinstance(logprobs, list):
        logprobs = logprobs[-1]

    batch_indices = []
    for lp, l in zip(logprobs, audio_features_len):
        frame_labels = torch.argmax(lp[:l], dim=-1)
        batch_indices.append(torch.unique_consecutive(frame_labels, dim=0).detach().cpu().tolist())

    if run_ctx.print_rtf:
        am_time = time.time() - am_start
        run_ctx.total_time += am_time
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time, am_time / audio_len_batch))

    tags = data["seq_tag"]

    for indices, tag in zip(batch_indices, tags):
        sequence = [run_ctx.label_by_index[idx] for idx in indices if idx in run_ctx.label_by_index]
        sequence = [
            s
            for s in sequence
            if s != run_ctx.silence_label and not s.startswith("<") and not s.startswith("[")
        ]
        text = " ".join(sequence).replace("@@ ", "")
        if run_ctx.print_hypothesis:
            print(text)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))
