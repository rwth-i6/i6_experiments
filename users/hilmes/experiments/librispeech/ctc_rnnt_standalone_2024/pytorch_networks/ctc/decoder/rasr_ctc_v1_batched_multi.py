"""
Flashlight/Torchaudio CTC decoder -- multi-scale variant

Runs the (slow) model forward exactly once and then applies a whole grid of
(lm_scale, prior_scale) combinations to the same posteriors. One transcript
file is written per combination (``search_out_lm{lm}_prior{prior}.py``), so a
downstream job can score each combination independently without recomputing
the model forward.

The lm scale is baked into the RASR config file (librasr has no runtime scale
setter), so one RASR config file / SearchAlgorithm is needed per lm scale. The
search over the grid is parallelized over lm scales -- one SearchAlgorithm (and
thus one LM) per CPU worker.

includes handling of prior computation
"""

from dataclasses import dataclass
from sisyphus import tk
import time
from time import perf_counter
import numpy as np
from typing import Optional, Union, List, Protocol


class TracebackItem(Protocol):
    lemma: str
    am_score: float
    lm_score: float
    start_time: int
    end_time: int

def _traceback_to_string(traceback: List[TracebackItem]) -> str:
    traceback_str = " ".join(item.lemma for item in traceback)
    traceback_str = traceback_str.replace("<s>", "")
    traceback_str = traceback_str.replace("</s>", "")
    traceback_str = traceback_str.replace("<blank>", "")
    traceback_str = traceback_str.replace("[BLANK] [1]", "")
    traceback_str = traceback_str.replace("[BLANK]", "")
    traceback_str = traceback_str.replace("<silence>", "")
    traceback_str = traceback_str.replace("[SILENCE]", "")
    traceback_str = traceback_str.replace("[SENTENCE-END]", "")
    traceback_str = " ".join(traceback_str.split())
    return traceback_str


def get_search_out_filename(lm_scale, prior_scale) -> str:
    """
    Single source of truth for the per-combination output file name. Imported by
    the pipeline to declare the ReturnnForwardJobV2 output_files, and used by the
    decoder to open the matching files, so both sides never diverge.

    :param lm_scale: lm scale of the combination
    :param prior_scale: prior scale of the combination
    :return: basename of the recognition output file for this combination
    """
    return f"search_out_lm{lm_scale}_prior{prior_scale}.py"


@dataclass
class DecoderConfig:
    # search related options:
    from i6_core.rasr import RasrConfig
    # one written RASR config per lm scale, parallel to lm_scales (lm scale baked into the file)
    rasr_config_files: List[Union[str, tk.Path, RasrConfig]]
    lm_scales: List[float]
    prior_scales: List[float]

    # number of parallel search workers (one SearchAlgorithm / LM per CPU)
    num_search_workers: int = 8

    rasr_post_config: Optional[Union[str, tk.Path, RasrConfig]] = None

    # prior correction
    blank_log_penalty: Optional[float] = None
    prior_file: Optional[str] = None

    turn_off_quant: Union[
        bool, str
    ] = False  # parameter for sanity checks, call self.prep_dequant instead of self.prep_quant

@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True


def _search_one_lm(args):
    """
    Worker entry point: run RASR search for a single lm scale over all stored
    posteriors and all prior scales, writing one output file per prior scale.

    Runs in a separate process (spawn), so it must not rely on any parent state
    besides its arguments. Posteriors are read from a transient .npz file.

    :param args: tuple of
        (lm_scale, rasr_config_file, prior_scales, prior, prior_file_present, out_dir, posteriors_path)
    :return: (lm_scale, search_time)
    """
    (lm_scale, rasr_config_file, prior_scales, prior, out_dir, posteriors_path) = args

    import torch
    from librasr import Configuration, SearchAlgorithm

    rasr_config = Configuration()
    rasr_config.set_from_file(rasr_config_file)
    search_algorithm = SearchAlgorithm(config=rasr_config)

    # open one output file per prior scale for this lm scale
    files = {}
    for prior_scale in prior_scales:
        f = open(f"{out_dir}/{get_search_out_filename(lm_scale, prior_scale)}", "wt")
        f.write("{\n")
        files[prior_scale] = f

    data = np.load(posteriors_path, allow_pickle=True)
    tags = data["tags"]
    # posteriors stored as an object array of [T, F] float arrays (already blank-penalized)
    posteriors = data["posteriors"]

    search_start = perf_counter()
    for tag, post in zip(tags, posteriors):
        for prior_scale in prior_scales:
            if prior is not None:
                adj = post - prior_scale * prior
            else:
                adj = post
            adj = -adj  # RASR wants negative logprobs
            # pass a torch CPU tensor to match the original (non-multi) decoder call exactly
            traceback = search_algorithm.recognize_segment(features=torch.from_numpy(np.ascontiguousarray(adj)))
            recog_str = _traceback_to_string(traceback)
            files[prior_scale].write("%s: %s,\n" % (repr(str(tag)), repr(recog_str)))
    search_time = perf_counter() - search_start

    for f in files.values():
        f.write("}\n")
        f.close()

    return lm_scale, search_time


def forward_init_hook(run_ctx, **kwargs):
    """

    :param run_ctx:
    :param kwargs:
    :return:
    """
    import torch
    from returnn.util.basic import cf

    config = DecoderConfig(**kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.rasr_config_files = list(config.rasr_config_files)
    run_ctx.lm_scales = list(config.lm_scales)
    run_ctx.prior_scales = list(config.prior_scales)
    run_ctx.num_search_workers = config.num_search_workers
    assert len(run_ctx.rasr_config_files) == len(run_ctx.lm_scales), (
        "rasr_config_files and lm_scales must be parallel (one config per lm scale)"
    )
    run_ctx.sample_rate = extra_config.sample_rate

    run_ctx.blank_log_penalty = config.blank_log_penalty
    if config.prior_file:
        run_ctx.prior = np.loadtxt(config.prior_file, dtype="float32")
        run_ctx.prior_scale = None  # prior scales are applied per-combination in the workers
    else:
        run_ctx.prior = None

    # accumulate posteriors here, run all searches in forward_finish_hook
    run_ctx.stored_tags = []
    run_ctx.stored_posteriors = []

    run_ctx.print_rtf = extra_config.print_rtf
    # always initialized: forward_step / forward_finish_hook accumulate these unconditionally
    run_ctx.running_audio_len_s = 0
    run_ctx.total_am_time = 0
    run_ctx.total_search_time = 0

    run_ctx.print_hypothesis = extra_config.print_hypothesis

    if config.turn_off_quant is False:
        print("Run Prep quantization")
        run_ctx.engine._model.prep_quant()
    elif config.turn_off_quant == "torch":
        print("Run Torch Quantization")
        run_ctx.engine._model.prep_torch_quant()
    elif config.turn_off_quant == "decomposed":
        run_ctx.engine._model.prep_quant(decompose=True)
        print("Use decomposed version, should match training")
    elif config.turn_off_quant == "leave_as_is":
        print("Use same version as in training")
    else:
        raise NotImplementedError
        run_ctx.engine._model.prep_dequant()  # TODO: needs fix
    run_ctx.engine._model.to(device=run_ctx.device)


def forward_finish_hook(run_ctx, **kwargs):
    import os
    import tempfile
    import concurrent.futures
    import multiprocessing

    # dump accumulated posteriors to a transient local .npz for the workers.
    # build 1-D object arrays explicitly so uniform-shape posteriors are not
    # collapsed into a single dense N-D array.
    tags_arr = np.empty(len(run_ctx.stored_tags), dtype=object)
    for i, t in enumerate(run_ctx.stored_tags):
        tags_arr[i] = str(t)
    post_arr = np.empty(len(run_ctx.stored_posteriors), dtype=object)
    for i, p in enumerate(run_ctx.stored_posteriors):
        post_arr[i] = p

    tmp_dir = tempfile.mkdtemp(prefix="ctc_multi_post_")
    posteriors_path = os.path.join(tmp_dir, "posteriors.npz")
    np.savez(posteriors_path, tags=tags_arr, posteriors=post_arr)
    # free the in-memory copy now that it is on disk
    run_ctx.stored_posteriors = []

    out_dir = os.getcwd()
    tasks = [
        (lm_scale, rasr_file, run_ctx.prior_scales, run_ctx.prior, out_dir, posteriors_path)
        for lm_scale, rasr_file in zip(run_ctx.lm_scales, run_ctx.rasr_config_files)
    ]

    num_workers = max(1, min(run_ctx.num_search_workers, len(tasks)))
    search_start = perf_counter()
    if num_workers == 1:
        for task in tasks:
            lm_scale, _ = _search_one_lm(task)
            print("Finished search for lm_scale %s" % lm_scale)
    else:
        # spawn to avoid fork-after-CUDA hangs (the forward used the GPU)
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            for lm_scale, worker_search_time in executor.map(_search_one_lm, tasks):
                print("Finished search for lm_scale %s (worker search time %.2fs)" % (lm_scale, worker_search_time))
    wallclock_search_time = perf_counter() - search_start
    run_ctx.total_search_time += wallclock_search_time

    # cleanup transient posteriors
    try:
        os.remove(posteriors_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    if run_ctx.print_rtf and run_ctx.running_audio_len_s > 0:
        print(
            "Total-AM-Time: %.2fs, AM-RTF: %.3f"
            % (run_ctx.total_am_time, run_ctx.total_am_time / run_ctx.running_audio_len_s)
        )
        print(
            "Total-Search-Time (wallclock, %d workers): %.2fs, Search-RTF: %.3f"
            % (num_workers, run_ctx.total_search_time, run_ctx.total_search_time / run_ctx.running_audio_len_s)
        )
        total_proc_time = run_ctx.total_am_time + run_ctx.total_search_time
        print("Total-time: %.2f, Batch-RTF: %.3f" % (total_proc_time, total_proc_time / run_ctx.running_audio_len_s))


def forward_step(*, model, data, run_ctx, **kwargs):
    import torch

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

    encoder_start = perf_counter()
    encoder_states, audio_features_len = model.forward(raw_audio, raw_audio_len)
    if not isinstance(encoder_states, list):
        encoder_states_cpu = encoder_states.cpu()
    else:
        assert len(encoder_states) == 1
        encoder_states_cpu = encoder_states[0].cpu()

    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last; scale-independent, apply once
        encoder_states_cpu[:, :, -1] -= run_ctx.blank_log_penalty

    encoder_time = perf_counter() - encoder_start
    run_ctx.total_am_time += encoder_time

    tags = data["seq_tag"]
    # store per-sequence log-probs (before prior subtraction / negation) for the search phase
    for encoder_state, audio_len, tag in zip(encoder_states_cpu, audio_features_len, tags):
        post = encoder_state[:audio_len].contiguous().numpy().astype("float32")
        run_ctx.stored_tags.append(tag)
        run_ctx.stored_posteriors.append(post)

    if run_ctx.print_rtf:
        print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (encoder_time, encoder_time / audio_len_batch))
