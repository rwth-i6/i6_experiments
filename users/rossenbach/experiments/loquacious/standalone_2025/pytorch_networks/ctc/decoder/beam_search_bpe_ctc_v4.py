"""
Greedy CTC decoder without any extras

v4 proper LSTM/Trafo support
"""
from dataclasses import dataclass
import time
import torch
from typing import Any, Dict, Optional
import numpy as np


from .search.documented_ctc_beam_search_v4 import CTCBeamSearch

@dataclass
class DecoderConfig:
    returnn_vocab: str

    beam_size: int
    # e.g. "lm.lstm.some_lstm_variant_file.Model"
    lm_module: Optional[str]
    lm_model_args: Dict[str, Any]
    lm_checkpoint: Optional[str]
    lm_states_need_label_axis: bool

    prior_scale: float = 0.0
    prior_file: Optional[str] = None

    lm_scale: float = 0.0
    

@dataclass
class DecoderExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True
    
    # LM model package path
    lm_package: Optional[str] = None


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    config = DecoderConfig(**kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = DecoderExtraConfig(**extra_config_dict)
    
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(
        vocab_file=config.returnn_vocab, unknown_label=None)
    run_ctx.labels = vocab.labels

    model = run_ctx.engine._model
    run_ctx.beam_size = config.beam_size
    
    if config.prior_file:
        run_ctx.prior = torch.tensor(np.loadtxt(config.prior_file, dtype="float32"), device=run_ctx.device)
        run_ctx.prior_scale = config.prior_scale
    else:
        run_ctx.prior = None

    lm_model = None
    if config.lm_module is not None:
        # load LM
        assert extra_config.lm_package is not None
        lm_module_prefix = ".".join(config.lm_module.split(".")[:-1])
        lm_module_class = config.lm_module.split(".")[-1]

        LmModule = __import__(
            ".".join([extra_config.lm_package, lm_module_prefix]),
            fromlist=[lm_module_class],
        )
        LmClass = getattr(LmModule, lm_module_class)

        lm_model = LmClass(**config.lm_model_args)
        checkpoint_state = torch.load(
            config.lm_checkpoint,
            map_location=run_ctx.device,
        )
        lm_model.load_state_dict(checkpoint_state["model"])
        lm_model.to(device=run_ctx.device)
        lm_model.eval()
        run_ctx.lm_model = lm_model

        print("loaded external LM")
        print()

    run_ctx.ctc_decoder = CTCBeamSearch(
        model=model,
        blank=model.cfg.label_target_size,
        device=run_ctx.device,
        lm_model=lm_model,
        lm_scale=config.lm_scale,
        lm_sos_token_index=0,
        lm_states_need_label_axis=config.lm_states_need_label_axis,
        prior=run_ctx.prior,
        prior_scale=run_ctx.prior_scale,

    )
    print("done!")
    

    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_time = 0

    run_ctx.print_hypothesis = extra_config.print_hypothesis

def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    print("Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s))


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000

    tags = data["seq_tag"]

    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s += audio_len_batch
        am_start = time.time()

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    if isinstance(logprobs, list):
        logprobs = logprobs[-1]


    if run_ctx.print_rtf:
        if torch.cuda.is_available():
            torch.cuda.synchronize(run_ctx.device)
        am_time = time.time() - am_start
        search_start = time.time()

    hyps = []
    for lp, l in zip(logprobs, audio_features_len):
        hypothesis = run_ctx.ctc_decoder.forward(lp, l, run_ctx.beam_size)
        hyps.append(hypothesis[0].tokens[1:])
        # hyps = [hypothesis[0].tokens for hypothesis in batched_hypotheses]  # exclude last sentence end token

    if run_ctx.print_rtf:
        search_time = (time.time() - search_start)
        run_ctx.total_time += (am_time + search_time)
        print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.3f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))

    for hyp, tag in zip(hyps, tags):
        sequence = [run_ctx.labels[idx] for idx in hyp if idx < len(run_ctx.labels)]
        text = " ".join(sequence).replace("@@ ","")
        if run_ctx.print_hypothesis:
            print(text)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))