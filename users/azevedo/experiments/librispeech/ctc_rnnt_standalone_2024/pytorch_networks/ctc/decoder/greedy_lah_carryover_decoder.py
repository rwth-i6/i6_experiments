"""
Greedy CTC decoder without any extras

v3: add config objects
"""
import math
import collections
from dataclasses import dataclass
import time
import torch
from typing import Optional

from ...rnnt.decoder.chunk_handler import AudioStreamer
from ...rnnt.auxil.functional import Mode
from .greedy_bpe_ctc_v3 import forward_step as forward_offline
from .lah_carryover_decoder import infer



@dataclass
class DecoderConfig:
    returnn_vocab: str

    # chunk size definitions for streaming in #samples
    chunk_size: Optional[int] = None
    lookahead_size: Optional[int] = None
    stride: Optional[int] = None

    carry_over_size: Optional[float] = None

    pad_value: Optional[float] = None

    test_version: Optional[float] = 0.0


@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    config = DecoderConfig(**kwargs["config"])
    run_ctx.config = config
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    from returnn.datasets.util.vocabulary import Vocabulary

    vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, unknown_label=None)
    run_ctx.labels = vocab.labels

    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_time = 0

    run_ctx.print_hypothesis = extra_config.print_hypothesis


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    print("Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s))


def forward_online(model, raw_audio, raw_audio_len, run_ctx):
    config: DecoderConfig = run_ctx.config
    batch_indices = []

    for i in range(raw_audio.shape[0]):
        # init generator for chunks of our raw_audio according to DecoderConfig
        chunk_streamer = AudioStreamer(
            raw_audio=raw_audio[i],
            raw_audio_len=raw_audio_len[i],
            left_size=config.chunk_size,
            right_size=config.lookahead_size,
            stride=config.stride,
            pad_value=config.pad_value,
        )

        states = collections.deque(maxlen=math.ceil(config.carry_over_size))

        indices = []
        for chunk, eff_chunk_len in chunk_streamer:
            logprobs, audio_features_len, state = infer(
                model=model,
                input=chunk.unsqueeze(0),
                lengths=torch.tensor(eff_chunk_len).unsqueeze(0),
                states=tuple(states) if len(states) > 0 else None,
                chunk_size=config.chunk_size
            )
            # print(f"{logprobs.shape = }, {audio_features_len.shape = }")
            states.append(state)

            # FIXME: loop unnecessary because we go batch for batch
            for lp, l in zip(logprobs, audio_features_len):
                indices.extend(
                    torch.unique_consecutive(torch.argmax(lp[:l], dim=-1), dim=0).detach().cpu().numpy())

        batch_indices.append(indices)

    return batch_indices


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000

    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s += audio_len_batch
        am_start = time.time()

    if run_ctx.config.chunk_size is None or run_ctx.config.chunk_size <= 0:
        model.mode = Mode.OFFLINE
        forward_offline(model=model, data=data, run_ctx=run_ctx)
        return
    else:
        model.mode = Mode.STREAMING
        batch_indices = forward_online(model, raw_audio, raw_audio_len, run_ctx)

    if run_ctx.print_rtf:
        am_time = time.time() - am_start
        run_ctx.total_time += am_time
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time, am_time / audio_len_batch))

    tags = data["seq_tag"]

    for indices, tag in zip(batch_indices, tags):
        sequence = [run_ctx.labels[idx] for idx in indices if idx < len(run_ctx.labels)]
        sequence = [s for s in sequence if (not s.startswith("<") and not s.startswith("["))]
        text = " ".join(sequence).replace("@@ ", "")
        if run_ctx.print_hypothesis:
            print(text)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))