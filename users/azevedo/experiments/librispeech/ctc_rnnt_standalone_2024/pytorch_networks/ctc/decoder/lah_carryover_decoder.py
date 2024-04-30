"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation
"""

import math
import collections
from dataclasses import dataclass
import time
import torch
import numpy as np
from typing import Any, Dict, Optional, List, Tuple

from ...rnnt.decoder.chunk_handler import AudioStreamer
from ...rnnt.auxil.functional import Mode, num_samples_to_frames
from .greedy_bpe_ctc_v3 import forward_step as forward_offline


@dataclass
class DecoderConfig:
    # search related options:
    beam_size: int
    beam_size_token: int
    beam_threshold: float


    # needed files
    lexicon: str
    returnn_vocab: str

    # chunk size definitions for streaming in #samples
    chunk_size: Optional[int] = None
    lookahead_size: Optional[int] = None
    stride: Optional[int] = None

    carry_over_size: Optional[float] = None
    
    pad_value: Optional[float] = None

    test_version: Optional[float] = 0.0
    
    # additional search options
    lm_weight: float = 0.0
    sil_score: float = 0.0
    word_score: float = 0.0

    # prior correction
    blank_log_penalty: Optional[float] = None
    prior_scale: float = 0.0
    prior_file: Optional[str] = None

    arpa_lm: Optional[str] = None

    use_torch_compile: bool = False
    torch_compile_options: Optional[Dict[str, Any]] = None


@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True


def forward_init_hook(run_ctx, **kwargs):
    """

    :param run_ctx:
    :param kwargs:
    :return:
    """
    import torch
    from torchaudio.models.decoder import ctc_decoder

    from returnn.datasets.util.vocabulary import Vocabulary
    from returnn.util.basic import cf

    config = DecoderConfig(**kwargs["config"])
    run_ctx.config = config
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    if config.arpa_lm is not None:
        lm = cf(config.arpa_lm)
    else:
        lm = None

    vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, unknown_label=None)
    labels = vocab.labels

    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=config.lexicon,
        lm=lm,
        lm_weight=config.lm_weight,
        tokens=labels + ["[blank]"],
        blank_token="[blank]",
        sil_token="[blank]",
        unk_word="[unknown]",
        nbest=1,
        beam_size=config.beam_size,
        beam_size_token=config.beam_size_token,
        beam_threshold=config.beam_threshold,
        sil_score=config.sil_score,
        word_score=config.word_score,
    )
    run_ctx.labels = labels
    run_ctx.blank_log_penalty = config.blank_log_penalty

    if config.prior_file:
        run_ctx.prior = np.loadtxt(config.prior_file, dtype="float32")
        run_ctx.prior_scale = config.prior_scale
    else:
        run_ctx.prior = None

    if config.use_torch_compile:
        options = config.torch_compile_options or {}
        run_ctx.engine._model = torch.compile(run_ctx.engine._model, **options)

    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_am_time = 0
        run_ctx.total_search_time = 0

    run_ctx.print_hypothesis = extra_config.print_hypothesis


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    if run_ctx.print_rtf:
        print(
            "Total-AM-Time: %.2fs, AM-RTF: %.3f"
            % (run_ctx.total_am_time, run_ctx.total_am_time / run_ctx.running_audio_len_s)
        )
        print(
            "Total-Search-Time: %.2fs, Search-RTF: %.3f"
            % (run_ctx.total_search_time, run_ctx.total_search_time / run_ctx.running_audio_len_s)
        )
        total_proc_time = run_ctx.total_am_time + run_ctx.total_search_time
        print("Total-time: %.2f, Batch-RTF: %.3f" % (total_proc_time, total_proc_time / run_ctx.running_audio_len_s))


def forward_online(model, raw_audio, raw_audio_len, run_ctx):
    config: DecoderConfig = run_ctx.config
    hypothesis = []

    search_start = time.time()
    for i in range(raw_audio.shape[0]):
        run_ctx.ctc_decoder.decode_begin()

        # init generator for chunks of our raw_audio according to DecoderConfig
        chunk_streamer = AudioStreamer(
            raw_audio=raw_audio[i],
            raw_audio_len=raw_audio_len[i],
            left_size=config.chunk_size,
            right_size=config.lookahead_size,
            stride=config.stride,
            pad_value=config.pad_value,
        )

        state = None
        states = collections.deque(maxlen=math.ceil(config.carry_over_size))

        for chunk, eff_chunk_len in chunk_streamer:
                
            logprobs, audio_features_len, state = infer(
                model=model, 
                input=chunk.unsqueeze(0), 
                lengths=torch.tensor(eff_chunk_len).unsqueeze(0),
                states=tuple(states) if len(states) > 0 else None,
                chunk_size=config.chunk_size
            )
            states.append(state)

            logprobs_cpu, audio_features_len_cpu = logprobs.cpu(), audio_features_len.cpu()
            if run_ctx.blank_log_penalty is not None:
                # assumes blank is last
                logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty
            if run_ctx.prior is not None:
                logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior

            run_ctx.ctc_decoder.decode_step(logprobs_cpu[0, :audio_features_len_cpu[0]])

        run_ctx.ctc_decoder.decode_end()

        hypos = run_ctx.ctc_decoder.get_final_hypothesis()
        hypothesis.append(hypos)

    search_time = time.time() - search_start
    run_ctx.total_search_time += search_time
    
    return hypothesis, search_time, search_time


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]
    tags = data["seq_tag"]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

    if run_ctx.config.chunk_size is None or run_ctx.config.chunk_size <= 0:
        model.mode = Mode.OFFLINE
        forward_offline(model=model, data=data, run_ctx=run_ctx)
        return
    else:
        model.mode = Mode.STREAMING
        hypothesis, am_time, search_time = forward_online(model, raw_audio, raw_audio_len, run_ctx)

    if run_ctx.print_rtf:
        print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.3f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))

    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        if run_ctx.print_hypothesis:
            print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))


def infer(
    model,
    input: torch.Tensor,
    lengths: torch.Tensor,
    states: Optional[List[List[torch.Tensor]]],
    chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:

    # encoder implements own infer method
    infer_func = getattr(model.conformer, "infer", None)
    assert infer_func is not None and callable(infer_func), "Encoder requires function 'infer' for decoding."

    squeezed_features = torch.squeeze(input)
    with torch.no_grad():
        audio_features, audio_features_len = model.feature_extraction(squeezed_features, lengths)

        chunk_size_frames = num_samples_to_frames(
            n_fft=model.feature_extraction.n_fft, 
            hop_length=model.feature_extraction.hop_length,
            center=model.feature_extraction.center,
            num_samples=int(chunk_size)
        )

        time_dim_pad = -audio_features.size(1) % chunk_size_frames
        audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, time_dim_pad), "constant", 0)

        encoder_out, out_mask, state = model.conformer.infer(audio_features, audio_features_len, 
                                                             states,
                                                             chunk_size=chunk_size_frames, 
                                                             lookahead_size=model.cfg.lookahead_size)
        
        encoder_out = model.final_linear(encoder_out) # (1, C', V+1)
        encoder_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

    return encoder_out, encoder_out_lengths, [state]
