"""
Base decoder
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import time
import torch
import copy
import pickle
import sys

from .chunk_handler import AudioStreamer
from ..streamable_module import StreamableModule
from ..common import Mode, _Hypothesis, insert_trie, follow_trie
from ..base_config import BaseConfig
from ._base_decoder import BaseDecoderModule, DecoderConfig, ExtraConfig


def forward_init_hook(run_ctx, **kwargs):
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    # initializing configs and decoder
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)
    config = DecoderConfig.from_dict(kwargs["config"])

    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, unknown_label=None)
    run_ctx.labels = vocab.labels

    # get decoder dynamically based on config
    decoder: BaseDecoderModule = config.search_config.module()(run_ctx)
    print("initializing decoder...")
    decoder.init_decoder(decoder_config=config, extra_config=extra_config)
    print("done!")
    run_ctx.decoder = decoder

    # RTF stuff
    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_time = 0

    # saves hypos returned after each chunk (e.g. for latency analysis)
    run_ctx.hypo_trie = dict()  # we use tries to save memory


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    if run_ctx.print_rtf:
        print(
            "Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s)
        )

    # save trie
    sys.setrecursionlimit(50000)  # tries can get deep
    with open("hypo_trie.pickle", "wb") as handle:
        pickle.dump(run_ctx.hypo_trie, handle, protocol=pickle.HIGHEST_PROTOCOL)


def forward_step(*, model: StreamableModule, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', 1]
    raw_audio_len = data["raw_audio:size1"].cpu()  # [B]

    decoder: BaseDecoderModule = run_ctx.decoder
    mode = decoder.decoder_config.mode

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

        start = time.time()

    # decode sequences in mode defined in config
    hyps: List[_Hypothesis] = []
    for i in range(raw_audio.shape[0]):
        decoder.reset()  # resets internal state of decoder

        if mode == Mode.OFFLINE:
            hyp = process_offline_sample(
                raw_audio=raw_audio[i], raw_audio_len=raw_audio_len[i], run_ctx=run_ctx, tag=data["seq_tag"][i]
            )
        elif mode == Mode.STREAMING:
            hyp = process_streaming_sample(
                raw_audio=raw_audio[i], raw_audio_len=raw_audio_len[i], run_ctx=run_ctx, tag=data["seq_tag"][i]
            )
        else: 
            raise ValueError(f"Invalid decoding mode: {mode}")
        hyps.append(hyp)

    if run_ctx.print_rtf:
        total_time = time.time() - start
        run_ctx.total_time += total_time
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (total_time, total_time / audio_len_batch))

    # hypos to text + printing
    for hyp, tag in zip(hyps, data["seq_tag"]):
        text = decoder.get_text(hyp)

        # FIXME: this is only viable for rnnt alignments (ctc needs torch.unique_consecutive)
        if decoder.extra_config.print_hypothesis:  
            if mode == Mode.STREAMING and run_ctx.hypo_trie:
                # prints each chunk in new line
                try: 
                    path = follow_trie(run_ctx.hypo_trie, [tag]+hyp.alignment)
                    path[0] = path[0][1:]  # remove tag
                    path_seq = [[run_ctx.labels[idx] for idx in hyp_chunk if idx not in [decoder.blank, decoder.sos]] for hyp_chunk in path]
                    path_seq = map(lambda x: " ".join(x), path_seq)
                    path_seq = "\n".join(list(path_seq)).replace("@@ ", "").replace("@@", "-")  # we replace "@@" at chunk boundaries with "-"
                    print(path_seq)
                except ValueError:
                    print(text)
            else:
                print(text)

            print()

        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))


def process_offline_sample(raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, run_ctx, tag: str) -> _Hypothesis:
    decoder: BaseDecoderModule = run_ctx.decoder
    hypotheses = decoder(
        raw_audio=raw_audio.unsqueeze(0),
        raw_audio_len=raw_audio_len.unsqueeze(0),
    )
    for i, hypo in enumerate(hypotheses):
        insert_trie(run_ctx.hypo_trie, [tag]+hypo.alignment, priority=i)

    return hypotheses[0]


def process_streaming_sample(raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, run_ctx, tag: str) -> _Hypothesis:
    decoder: BaseDecoderModule = run_ctx.decoder

    # object responsible for returning chunks of raw_audio
    chunk_streamer = AudioStreamer(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,

        left_size=decoder.decoder_config.chunk_size,
        right_size=decoder.decoder_config.lookahead_size,
        stride=decoder.decoder_config.stride,
        pad_value=decoder.decoder_config.pad_value,
    )

    hypotheses: List[_Hypothesis] = None
    for chunk, eff_chunk_sz in chunk_streamer:
        hypotheses, _ = decoder.step(audio_chunk=chunk, chunk_len=eff_chunk_sz)

        # save hypos of current beam, can be used to analyze latency etc...
        for i, hypo in enumerate(hypotheses):
            # print(hypo.tokens)
            run_ctx.hypo_trie, _ = insert_trie(run_ctx.hypo_trie, [tag]+hypo.alignment, priority=i)

    return decoder.get_final_hypotheses()[0]
