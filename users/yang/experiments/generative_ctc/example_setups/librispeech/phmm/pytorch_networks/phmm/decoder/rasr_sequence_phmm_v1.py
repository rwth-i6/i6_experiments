"""
LibRASR decoder for HDF-input supervised generative pHMM models.
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
    decode_layer_index: Optional[int] = None
    input_key: str = "data"
    prior_file: Optional[Union[str, tk.Path]] = None
    prior_scale: float = 0.0


@dataclass
class ExtraConfig:
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

    run_ctx.decode_layer_index = config.decode_layer_index
    run_ctx.input_key = config.input_key
    run_ctx.print_hypothesis = extra_config.print_hypothesis
    run_ctx.prior_scale = config.prior_scale
    run_ctx.prior = None
    if config.prior_file is not None:
        run_ctx.prior = np.loadtxt(cf(config.prior_file), dtype="float32")


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()


def forward_step(*, model, data, run_ctx, **kwargs):
    import torch

    model_input = data[run_ctx.input_key]
    model_input_len = data[f"{run_ctx.input_key}:size1"].to(torch.long)

    am_start = perf_counter()
    log_probs_list, output_len = model(model_input=model_input, model_input_len=model_input_len)
    logprobs = model.get_log_probs_by_layer(log_probs_list, decode_layer_index=run_ctx.decode_layer_index)
    am_time = perf_counter() - am_start

    search_start = perf_counter()
    hypotheses = []
    for seq_logprobs, seq_len in zip(logprobs, output_len):
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
    print("Batch-AM-Time: %.2fs, Batch-Search-Time: %.2fs" % (am_time, search_time))

    tags = data["seq_tag"]
    for hyp, tag in zip(hypotheses, tags):
        if run_ctx.print_hypothesis:
            print(hyp)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(hyp)))
