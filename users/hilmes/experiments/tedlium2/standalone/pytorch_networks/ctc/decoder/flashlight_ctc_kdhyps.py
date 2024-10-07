"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation
v2 includes the option to output probabilities for n-best
"""

from dataclasses import dataclass
from sisyphus import tk
import time
import numpy as np
from typing import Any, Dict, Optional, Union


@dataclass
class DecoderConfig:
    # search related options:
    beam_size: int
    beam_size_token: int
    beam_threshold: float

    # needed files
    lexicon: Union[str, tk.Path]
    returnn_vocab: Union[str, tk.Path]

    # additional search options
    lm_weight: float = 0.0
    sil_score: float = 0.0
    word_score: float = 0.0

    # prior correction
    blank_log_penalty: Optional[float] = None
    prior_scale: float = 0.0
    prior_file: Optional[str] = None

    arpa_lm: Optional[Union[str, tk.Path]] = None

    use_torch_compile: bool = False
    torch_compile_options: Optional[Dict[str, Any]] = None

    n_best_probs: Optional[int] = None
    add_reference: bool = False
    length_norm: bool = False


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
    from torchaudio.models.decoder._ctc_decoder import ctc_decoder

    from returnn.datasets.util.vocabulary import Vocabulary
    from returnn.util.basic import cf

    config = DecoderConfig(**kwargs["config"])
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
        nbest=1 if config.n_best_probs is None else config.n_best_probs * 10,
        beam_size=config.beam_size,
        beam_size_token=config.beam_size_token,
        beam_threshold=config.beam_threshold,
        sil_score=config.sil_score,
        word_score=config.word_score,
    )
    run_ctx.labels = labels
    run_ctx.blank_log_penalty = config.blank_log_penalty
    run_ctx.n_best_probs = config.n_best_probs
    if run_ctx.n_best_probs is not None:
        run_ctx.n_best_probs_file = open("n_best_probs.py", "wt")
        run_ctx.n_best_probs_file.write("{\n")

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
    run_ctx.add_reference = config.add_reference
    run_ctx.length_norm = config.length_norm


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    if run_ctx.n_best_probs is not None:
        run_ctx.n_best_probs_file.write("}\n")
        run_ctx.n_best_probs_file.close()

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


def forward_step(*, model, data, run_ctx, **kwargs):
    import torch

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    targets = data['labels']
    targets_len = data['labels:size1']

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

    am_start = time.time()
    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    tags = data["seq_tag"]

    logprobs_cpu = logprobs.cpu()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty
    if run_ctx.prior is not None:
        logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior



    search_start = time.time()
    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())
    search_time = time.time() - search_start


    if run_ctx.print_rtf:
        run_ctx.total_search_time += search_time
        am_time = time.time() - am_start
        run_ctx.total_am_time += am_time
        print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.3f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))

    from torch import nn
    for hyp, tag, length, logprobs, target, target_len in zip(hypothesis, tags, audio_features_len, logprobs, targets, targets_len):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        if run_ctx.print_hypothesis:
            print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))
        if run_ctx.n_best_probs is not None:
            dc = {}
            sm = 0
            for seq in hyp:
                ctc_loss = nn.functional.ctc_loss(
                    logprobs,
                    seq.tokens,
                    input_lengths=length,
                    target_lengths=torch.LongTensor([len(seq.tokens)]).to(device="cpu"),
                    blank=model.cfg.label_target_size,
                    reduction="sum",
                    zero_infinity=True,
                )
                if repr(seq.tokens) not in dc.keys():
                    if run_ctx.length_norm is True:
                         ctc_loss = ctc_loss / target_len
                    ctc_loss = ctc_loss.cpu()
                    score = np.exp(-1 * ctc_loss)
                    sm += score
                    assert not score == float('nan')
                    dc[repr(seq.tokens)] = score
                if len(dc) == run_ctx.n_best_probs:
                    break
            #if not len(dc) >= run_ctx.n_best_probs / 2:
                #tmp = {}
                #for seq in dc:
                #    if dc[seq] > 0.01:
                #        tmp[seq] = dc[seq]
                #assert len(dc) >= run_ctx.n_best_probs / 2, (len(hyp), len(dc), len(tmp), dc)
                #f"Not enough unique sequences {len(dc)}, try rerunning with different params {len(hyp)}"
            if run_ctx.add_reference is True:
                if repr(target) not in dc:
                    ctc_loss = nn.functional.ctc_loss(
                        logprobs,
                        target[:target_len],
                        input_lengths=length,
                        target_lengths=target_len,
                        blank=model.cfg.label_target_size,
                        reduction="sum",
                        zero_infinity=True,
                    )
                    if run_ctx.length_norm is True:
                        ctc_loss = ctc_loss / target_len
                    ctc_loss = ctc_loss.cpu()
                    score = np.exp(-1 * ctc_loss)
                    sm += score
                    dc[repr(target)] = score
            for seq in dc:
                assert not dc[seq] / sm == torch.tensor(float('nan'))
                dc[seq] = dc[seq] / sm  # normalize
            run_ctx.n_best_probs_file.write("%s: %s,\n" % (repr(tag), repr(dc)))
