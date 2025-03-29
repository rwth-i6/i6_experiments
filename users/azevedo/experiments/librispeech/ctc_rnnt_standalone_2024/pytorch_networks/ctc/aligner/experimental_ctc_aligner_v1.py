"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation
"""

from dataclasses import dataclass
import json
import time
import numpy as np
from typing import Any, Dict, Optional
import torchaudio.functional as F

from ...rnnt.auxil.functional import Mode


@dataclass
class DecoderConfig:
    # search related options:
    beam_size: int
    beam_size_token: int
    beam_threshold: float

    # needed files
    lexicon: str
    returnn_vocab: str

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
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.alignments = {}

    if config.arpa_lm is not None:
        lm = cf(config.arpa_lm)
    else:
        lm = None

    vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, unknown_label=None)
    labels = vocab.labels
    run_ctx.labels = labels + ["[blank]"]
    print(f"Size of vocabulary: {len(run_ctx.labels)-1}")
    print(f"{run_ctx.labels[-2]} {run_ctx.labels[-1]}")

    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=config.lexicon,
        lm=lm,
        lm_weight=config.lm_weight,
        tokens=run_ctx.labels,
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
    with open("aligns_out.json", "w") as f:
        json.dump(run_ctx.alignments, f, indent=4)

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

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

    if hasattr(model, "mode"):
        model.mode = Mode.OFFLINE

    am_start = time.time()
    logprobs, _ = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    tags = data["seq_tag"]

    if isinstance(logprobs, list):
        logprobs = logprobs[-1]

    logprobs_cpu = logprobs.cpu()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty
    if run_ctx.prior is not None:
        logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior

    am_time = time.time() - am_start
    run_ctx.total_am_time += am_time

    #
    # calc forced alignments
    #
    def align(emissions, tokens):
        als, scores = [], []
        for i in range(emissions.size(0)):
            target_seq = tokens[i, :labels_len[i]].unsqueeze(0)
            alignment, score = F.forced_align(emissions[[i]], target_seq, blank=len(run_ctx.labels)-1)

            alignment, score = alignment[0], score[0]
            als.append(alignment)
            scores.append(score)
        
        # scores = scores.exp()
        return als, scores

    search_start = time.time()
    aligned_tokens, alignment_scores = align(emissions=logprobs_cpu, tokens=labels)    
    search_time = time.time() - search_start
    run_ctx.total_search_time += search_time

    # create dict w/ alignment info
    alignments = {}
    for i, (seq, score) in enumerate(zip(aligned_tokens, alignment_scores)):
        tag = repr(tags[i])

        token_spans = F.merge_tokens(seq, score, blank=len(run_ctx.labels)-1)
        alignments[tag] = []
        for span in token_spans:
            alignments[tag].append({
                "token": span.token,
                "start": span.start,
                "end": span.end,
                "score": span.score
            })

    run_ctx.alignments.update(alignments)

    if run_ctx.print_rtf:
        print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.3f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))
