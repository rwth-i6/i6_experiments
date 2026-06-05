"""
LibRASR decoder for phoneme posterior HMM models.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Protocol, Union

import numpy as np
from sisyphus import tk


class TracebackItem(Protocol):
    lemma: str
    am_score: float
    lm_score: float
    start_time: int
    end_time: int


def _traceback_to_string(traceback: List[TracebackItem]) -> str:
    traceback_str = " ".join(item.lemma for item in traceback)
    for token in ("<s>", "</s>", "<blank>", "[BLANK]", "[SILENCE]", "<silence>", "[SENTENCE-END]"):
        traceback_str = traceback_str.replace(token, "")
    return " ".join(traceback_str.split())


@dataclass
class DecoderConfig:
    rasr_config_file: Union[str, tk.Path]
    lexicon: Union[str, tk.Path]
    silence_label: str = "[SILENCE]"
    data_key: str = "data"
    segment_starts_key: str = "segment_starts"
    frame_lengths_key: str = "frame_lengths"
    logprob_mode: str = "discriminative"
    lm_image_file: Optional[Union[str, tk.Path]] = None
    decode_layer_index: Optional[int] = None
    prior_file: Optional[Union[str, tk.Path]] = None
    prior_scale: float = 0.0


@dataclass
class ExtraConfig:
    print_rtf: bool = True
    sample_rate: int = 16000
    print_hypothesis: bool = True


def forward_init_hook(run_ctx, **kwargs):
    from i6_core.lib import lexicon
    from librasr import Configuration, SearchAlgorithm
    from returnn.util.basic import cf

    config = DecoderConfig(**kwargs["config"])
    extra_config = ExtraConfig(**kwargs.get("extra_config", {}))

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    lex = lexicon.Lexicon()
    lex.load(cf(config.lexicon))
    run_ctx.label_inventory = list(lex.phonemes.keys())
    run_ctx.num_labels = len(run_ctx.label_inventory)
    if config.silence_label not in lex.phonemes:
        raise ValueError(f"Silence label {config.silence_label!r} not found in lexicon {config.lexicon!r}")

    rasr_config = Configuration()
    rasr_config.set_from_file(cf(config.rasr_config_file))
    run_ctx.search_algorithm = SearchAlgorithm(config=rasr_config)

    run_ctx.sample_rate = extra_config.sample_rate
    run_ctx.print_rtf = extra_config.print_rtf
    run_ctx.print_hypothesis = extra_config.print_hypothesis
    if config.logprob_mode not in {"generative", "discriminative"}:
        raise ValueError(f"Unsupported logprob_mode {config.logprob_mode!r}")
    run_ctx.data_key = config.data_key
    run_ctx.segment_starts_key = config.segment_starts_key
    run_ctx.frame_lengths_key = config.frame_lengths_key
    run_ctx.logprob_mode = config.logprob_mode
    run_ctx.lm_image_file = config.lm_image_file  # kept as explicit Sisyphus dependency for prebuilt LM images
    run_ctx.prior_scale = config.prior_scale
    run_ctx.prior = None
    if config.prior_file is not None:
        run_ctx.prior = np.loadtxt(cf(config.prior_file), dtype="float32")

    # if run_ctx.print_rtf:
    #     run_ctx.running_audio_len_s = 0.0
    #     run_ctx.total_am_time = 0.0
    #     run_ctx.total_search_time = 0.0


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    # if run_ctx.print_rtf and run_ctx.running_audio_len_s > 0:
    #     print(
    #         "Total-AM-Time: %.2fs, AM-RTF: %.3f"
    #         % (run_ctx.total_am_time, run_ctx.total_am_time / run_ctx.running_audio_len_s)
    #     )
    #     print(
    #         "Total-Search-Time: %.2fs, Search-RTF: %.3f"
    #         % (run_ctx.total_search_time, run_ctx.total_search_time / run_ctx.running_audio_len_s)
    #     )
    #     total_proc_time = run_ctx.total_am_time + run_ctx.total_search_time
    #     print("Total-time: %.2f, Batch-RTF: %.3f" % (total_proc_time, total_proc_time / run_ctx.running_audio_len_s))


def forward_step(*, model, data, run_ctx, **kwargs):
    import torch

    cluster_ids = data[run_ctx.data_key]
    cluster_ids_len = data[f"{run_ctx.data_key}:size1"]
    segment_starts = data.get(run_ctx.segment_starts_key)
    segment_starts_len = data.get(f"{run_ctx.segment_starts_key}:size1")
    frame_lengths = data.get(run_ctx.frame_lengths_key)

    # if run_ctx.print_rtf:
    #     audio_len_batch = torch.sum(cluster_ids_len).detach().cpu().numpy() / run_ctx.sample_rate
    #     run_ctx.running_audio_len_s += audio_len_batch

    am_start = perf_counter()
    log_probs, audio_features_len = model(
        raw_audio=cluster_ids,
        raw_audio_len=cluster_ids_len,
        logprob_mode=run_ctx.logprob_mode,
        output_label_order=run_ctx.label_inventory,
        segment_starts=segment_starts,
        segment_starts_len=segment_starts_len,
        frame_lengths=frame_lengths,
    )
    am_time = perf_counter() - am_start

    search_start = perf_counter()
    hypotheses = []
    for seq_logprobs, seq_len in zip(log_probs, audio_features_len):
        seq_logprobs = seq_logprobs[:seq_len]
        if seq_logprobs.shape[-1] != run_ctx.num_labels:
            raise ValueError(
                f"Unexpected label dimension {seq_logprobs.shape[-1]} for lexicon inventory size {run_ctx.num_labels}"
            )
        if run_ctx.prior is not None:
            if seq_logprobs.shape[-1] != run_ctx.prior.shape[0]:
                raise ValueError(
                    f"Prior dimension {run_ctx.prior.shape[0]} does not match label dimension {seq_logprobs.shape[-1]}"
                )
            seq_logprobs = seq_logprobs - run_ctx.prior_scale * torch.as_tensor(
                run_ctx.prior, device=seq_logprobs.device, dtype=seq_logprobs.dtype
            )

        features = (-seq_logprobs).detach().cpu().numpy().astype("float32", copy=False)
        features = np.ascontiguousarray(features)
        traceback = run_ctx.search_algorithm.recognize_segment(features=features)
        hypotheses.append(_traceback_to_string(traceback))
    search_time = perf_counter() - search_start
    print("Batch-Search-Time: %.2fs" %search_time)

    # if run_ctx.print_rtf:
    #     run_ctx.total_am_time += am_time
    #     run_ctx.total_search_time += search_time
    #     print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (am_time, am_time / audio_len_batch))
    #     print("Batch-Search-Time: %.2fs, Search-RTF: %.3f" % (search_time, search_time / audio_len_batch))
    #     print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))

    tags = data["seq_tag"]
    for hyp, tag in zip(hypotheses, tags):
        if run_ctx.print_hypothesis:
            print(hyp)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(hyp)))
