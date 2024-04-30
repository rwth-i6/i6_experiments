"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation
"""

from dataclasses import dataclass
import time
import torch
import numpy as np
from typing import Any, Dict, Optional, List, Tuple

from ...rnnt.decoder.chunk_handler import AudioStreamer


@dataclass
class DecoderConfig:
    # search related options:
    beam_size: int
    beam_size_token: int
    beam_threshold: float


    # needed files
    lexicon: str
    returnn_vocab: str

    # stream options:
    left_size: Optional[int] = None
    right_size: Optional[int] = None
    stride: Optional[int] = None

    
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



def forward_offline(model, raw_audio, raw_audio_len, run_ctx):
    am_start = time.time()
    audio_features_len, logprobs, _ = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )[1]

    logprobs_cpu = logprobs.cpu()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty
    if run_ctx.prior is not None:
        logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior

    am_time = time.time() - am_start
    run_ctx.total_am_time += am_time

    search_start = time.time()
    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())
    search_time = time.time() - search_start
    run_ctx.total_search_time += search_time

    return hypothesis, am_time, search_time


def forward_online(model, raw_audio, raw_audio_len, run_ctx):
    config = run_ctx.config
    hypothesis = []

    search_start = time.time()
    for i in range(raw_audio.shape[0]):
        run_ctx.ctc_decoder.decode_begin()

        # init generator for chunks of our raw_audio according to DecoderConfig
        chunk_streamer = AudioStreamer(
            raw_audio=raw_audio[i],
            raw_audio_len=raw_audio_len[i],
            left_size=config.left_size,
            right_size=config.right_size,
            stride=config.stride,
            pad_value=config.pad_value,
        )

        state = None

        for chunk, eff_chunk_len in chunk_streamer:
            if config.right_size is not None:
                lah_samples = int(config.right_size*0.06 * 16e3)   # e.g. 8 frames * 60ms = 480ms
                chunk[config.left_size+lah_samples:] = 0 if config.pad_value is None else config.pad_value
                
            logprobs, audio_features_len, state = infer(
                model=model, 
                input=chunk.unsqueeze(0), 
                lengths=torch.tensor(eff_chunk_len).unsqueeze(0),
                states=state
            )

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

    if run_ctx.config.left_size is None:
        model.set(streaming=False)
        hypothesis, am_time, search_time = forward_offline(model, raw_audio, raw_audio_len, run_ctx)
    else:
        model.set(streaming=True)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:

    # encoder implements own infer method
    infer_func = getattr(model.conformer, "infer", None)
    assert infer_func is not None and callable(infer_func), "Encoder requires function 'infer' for decoding."

    squeezed_features = torch.squeeze(input)
    with torch.no_grad():
        audio_features, audio_features_len = model.feature_extraction(squeezed_features, lengths)

        encoder_out, out_mask, state = model.conformer.infer(audio_features, audio_features_len, 
                                                            states, lookahead_size=model.cfg.lookahead_size)
        
        encoder_out = model.final_linear(encoder_out) # (1, C', V+1)
        encoder_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

    return encoder_out, encoder_out_lengths, [state]
