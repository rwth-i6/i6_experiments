"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation
"""

from dataclasses import dataclass
import json
import time
import numpy as np
from typing import Any, Dict, Optional, Union
import torchaudio.functional as F
import collections
import math

from ...rnnt.auxil.functional import Mode
from ...rnnt.decoder.chunk_handler import AudioStreamer, StreamingASRContextManager
from ..decoder.lah_carryover_decoder import infer


@dataclass
class AlignerConfig:
    # needed files
    returnn_vocab: str
    
    mode: Union[Mode, str] = None

    # streaming definitions if mode == Mode.STREAMING
    chunk_size: Optional[int] = None
    carry_over_size: Optional[float] = None
    lookahead_size: Optional[int] = None
    stride: Optional[int] = None

    pad_value: Optional[float] = None

    test_version: Optional[float] = 0.0

    rnnt_hypo_path: Optional[str] = None

    # prior correction
    prior_scale: float = 0.0
    prior_file: Optional[str] = None

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

    config = AlignerConfig(**kwargs["config"])
    config.mode = {str(m): m for m in Mode}[config.mode]
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.config = config
    run_ctx.alignments = {}

    vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, unknown_label=None)
    labels = vocab.labels
    run_ctx.labels = labels + ["[blank]"]
    print(f"Size of vocabulary: {len(run_ctx.labels)-1}")
    print(f"{run_ctx.labels[-2]} {run_ctx.labels[-1]}")

    if config.prior_file:
        run_ctx.prior = np.loadtxt(config.prior_file, dtype="float32")
        run_ctx.prior_scale = config.prior_scale
    else:
        run_ctx.prior = None

    if config.rnnt_hypo_path:
        with open(config.rnnt_hypo_path) as f:
            run_ctx.rnnt_hypos = json.load(f)

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
    
    config: AlignerConfig = run_ctx.config

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    tags = data["seq_tag"]
    if config.rnnt_hypo_path:
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(run_ctx.rnnt_hypos[tag]["labels"]) for tag in tags],
            batch_first=True, padding_value=config.pad_value if config.pad_value else 0.0
        )
        labels_len = torch.tensor([run_ctx.rnnt_hypos[tag]["labels_len"] for tag in tags])
    else:
        labels = data["labels"]  # [B, N] (sparse)
        labels_len = data["labels:size1"]  # [B, N]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

    am_start = time.time()
    if config.mode == Mode.OFFLINE:
        model.mode = Mode.OFFLINE
        logprobs, logprobs_lengths = model(
            raw_audio=raw_audio,
            raw_audio_len=raw_audio_len,
        )
    else:
        model.mode = Mode.STREAMING
        logprobs = []
        logprobs_lengths = torch.zeros(raw_audio.size(0))
        for i in range(raw_audio.size(0)):
            chunk_streamer = AudioStreamer(
                raw_audio=raw_audio[i],
                raw_audio_len=raw_audio_len[i],
                left_size=config.chunk_size,
                right_size=config.lookahead_size,
                stride=config.stride,
                pad_value=config.pad_value,
            )

            curr_logprobs = []
            states = collections.deque(maxlen=math.ceil(config.carry_over_size))
            for chunk, eff_chunk_len in chunk_streamer:
                logsprobs_chunk, encoder_out_lengths, state = infer(
                    model=model, 
                    input=chunk.unsqueeze(0), 
                    lengths=torch.tensor(eff_chunk_len).unsqueeze(0),
                    states=tuple(states) if len(states) > 0 else None,
                    chunk_size=config.chunk_size
                )
                states.append(state)

                if isinstance(logsprobs_chunk, list):
                    logsprobs_chunk = logsprobs_chunk[-1]

                curr_logprobs.append(logsprobs_chunk)
                logprobs_lengths[i] += encoder_out_lengths[0]

            logprobs.append(torch.cat(curr_logprobs, dim=1))
            
        logprobs_padded = torch.full(
            size=(raw_audio.size(0), max(l.size(1) for l in logprobs), *logprobs[0].shape[2:]), 
            fill_value=float("-inf"), device=logprobs[0].device
        )
        for i, logs in enumerate(logprobs):
            logprobs_padded[i, :logs.size(1)] = logs[0]
        logprobs = logprobs_padded
        print(f"> {logprobs.shape = }, {raw_audio.shape = }")
        print(f"> {logprobs_lengths}\n")

    if isinstance(logprobs, list):
        logprobs = logprobs[-1]

    logprobs_cpu, logprobs_lengths = logprobs.cpu(), logprobs_lengths.cpu().int()
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
            if config.mode == Mode.STREAMING:
                emission = emissions[i, :logprobs_lengths[i]].unsqueeze(0)
            else:
                emission = emissions[[i]]

            target_seq = tokens[i, :labels_len[i]].unsqueeze(0)
            alignment, score = F.forced_align(emission, target_seq, blank=len(run_ctx.labels)-1)

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
