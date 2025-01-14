"""
Flashlight for CTC with neural LM
"""

import sys
import time
from typing import Any, Tuple, Dict, List

from returnn.tensor import Tensor, Dim, single_step_dim, batch_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import RecogDef

from ..ctc import Model


def model_recog_flashlight(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import tracemalloc
    from dataclasses import dataclass
    import torch
    from flashlight.lib.text.decoder import LM, LMState
    from i6_experiments.users.zeyer.utils.lru_cache import lru_cache
    from returnn.config import get_global_config
    from returnn.util import basic as util

    config = get_global_config()
    n_best = config.int("n_best", 1)
    beam_size = config.typed_value("beam_size", None)
    beam_size_token = config.typed_value("beam_size_token", None)
    beam_threshold = config.typed_value("beam_threshold", None)

    # Eager-mode implementation of beam search using Flashlight.

    debug_tracemalloc = config.bool("debug_tracemalloc", False)
    if debug_tracemalloc:
        tracemalloc.start()

    # noinspection PyUnresolvedReferences
    lm: TransformerDecoder = model.lm
    # noinspection PyUnresolvedReferences
    lm_scale: float = model.lm_scale

    dev_s = rf.get_default_device()
    dev = torch.device(dev_s)

    total_mem = None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        _, total_mem = torch.cuda.mem_get_info(dev if dev.index is not None else None)

    def _collect_mem_stats():
        if dev.type == "cuda":
            return [
                f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
            ]
        return ["(unknown)"]

    print(
        f"Memory usage {dev_s} before encoder forward:",
        " ".join(_collect_mem_stats()),
        "total:",
        util.human_bytes_size(total_mem) if total_mem else "(unknown)",
    )

    lm_initial_state = lm.default_initial_state(batch_dims=[])

    # https://github.com/flashlight/text/tree/main/bindings/python#decoding-with-your-own-language-model
    # https://github.com/facebookresearch/fairseq/blob/main/examples/speech_recognition/new/decoders/flashlight_decoder.py
    # https://github.com/pytorch/audio/blob/main/src/torchaudio/models/decoder/_ctc_decoder.py

    # The current implementation of FlashlightLM below assumes we can just use the token_idx as-is for the LM.
    assert model.blank_idx == model.target_dim.dimension

    @dataclass
    class FlashlightLMState:
        label_seq: List[int]
        prev_state: LMState

    # Use LRU cache for the LM states (on GPU) and log probs.
    # Note that additionally to the cache size limit here,
    # we free more when we run out of CUDA memory.
    start_lru_cache_size = config.int("lm_state_lru_initial_cache_size", 1024)
    max_used_mem_fraction = 0.9

    class FlashlightLM(LM):
        def __init__(self):
            super().__init__()
            # Cannot use weakrefs because the LMState object will always be recreated on-the-fly,
            # i.e. the Python object does not persist.
            self.mapping_states: Dict[LMState, FlashlightLMState] = {}
            self._count_recalc_whole_seq = 0
            self._recent_debug_log_time = -sys.maxsize
            self._max_used_mem_fraction = max_used_mem_fraction

        def reset(self):
            self.mapping_states.clear()
            self._count_recalc_whole_seq = 0
            self._recent_debug_log_time = -sys.maxsize
            self._max_used_mem_fraction = max_used_mem_fraction
            self._calc_next_lm_state.cache_clear()
            self._calc_next_lm_state.cache_set_maxsize(start_lru_cache_size)

        @lru_cache(maxsize=start_lru_cache_size)
        def _calc_next_lm_state(self, state: LMState) -> Tuple[Any, torch.Tensor]:
            """
            :return: LM state, log probs [Vocab]
            """
            state_ = self.mapping_states[state]

            if state_.label_seq == [model.bos_idx]:
                prev_lm_state = lm_initial_state
            else:
                prev_lm_state, _ = self._calc_next_lm_state.cache_peek(state_.prev_state, fallback=(None, None))
            lm_logits, lm_state = None, None
            while True:
                self._cache_maybe_free_memory()
                try:
                    if prev_lm_state is not None or lm_initial_state is None:
                        # We have the prev state, or there is no state at all.
                        # So we can do a single step.
                        lm_logits, lm_state = lm(
                            rf.constant(state_.label_seq[-1], dims=[], sparse_dim=model.target_dim),
                            spatial_dim=single_step_dim,
                            state=prev_lm_state,
                        )  # Vocab / ...
                    else:
                        # We don't have the prev state. So recalculate it now, but directly on the whole given seq.
                        self._count_recalc_whole_seq += 1
                        spatial_dim = Dim(len(state_.label_seq), name="seq")
                        lm_logits, lm_state = lm(
                            rf.convert_to_tensor(state_.label_seq, dims=[spatial_dim], sparse_dim=model.target_dim),
                            spatial_dim=spatial_dim,
                            state=lm_initial_state,
                            output_only_last_frame=True,
                        )  # Vocab / ...
                except torch.cuda.OutOfMemoryError as exc:
                    if self._calc_next_lm_state.cache_len() == 0:
                        raise  # cannot free more
                    print(f"{type(exc).__name__}: {exc}")
                    new_max_used_mem_fraction = max(0.2, self._max_used_mem_fraction - 0.1)
                    if new_max_used_mem_fraction != self._max_used_mem_fraction:
                        print(f"Reduce max used mem fraction to {new_max_used_mem_fraction:.0%}")
                    continue  # try again
                break
            assert lm_logits.dims == (model.target_dim,)
            lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Vocab
            log_probs_raw = lm_log_probs.raw_tensor.cpu()
            return lm_state, log_probs_raw

        def _cache_maybe_free_memory(self):
            if dev.type == "cuda":
                # Maybe check if we should free some more memory.
                count_pop = 0
                used_mem = 0
                while self._calc_next_lm_state.cache_len() > 0:
                    used_mem = torch.cuda.memory_reserved(dev)
                    if used_mem / total_mem < self._max_used_mem_fraction:
                        break
                    # Check again after trying to empty the cache.
                    # Note: gc.collect() is problematic here because of how Flashlight handles the states:
                    # We have millions of Python objects in the mapping_states dict,
                    # which takes a very long time to go through.
                    torch.cuda.empty_cache()
                    used_mem = torch.cuda.memory_reserved(dev)
                    if used_mem / total_mem < self._max_used_mem_fraction:
                        break
                    self._calc_next_lm_state.cache_pop_oldest()
                    count_pop += 1
                if count_pop > 0:
                    print(
                        f"Pop {count_pop} states from cache,"
                        f" cache size {self._calc_next_lm_state.cache_len()},"
                        f" reached {used_mem / total_mem:.1%} of total mem,"
                        f" mem usage {dev_s}: {' '.join(_collect_mem_stats())}"
                    )
                    self._calc_next_lm_state.cache_set_maxsize(self._calc_next_lm_state.cache_len())

        def start(self, start_with_nothing: bool):
            """
            Parameters:
                start_with_nothing (bool): whether or not to start sentence with sil token.
            """
            start_with_nothing  # noqa  # not sure how to handle this?
            self.reset()
            state = LMState()
            self.mapping_states[state] = FlashlightLMState(label_seq=[model.bos_idx], prev_state=state)
            return state

        def score(self, state: LMState, token_index: int):
            """
            Evaluate language model based on the current lm state and new word

            Parameters:
                state: current lm state
                token_index: index of the word
                            (can be lexicon index then you should store inside LM the
                            mapping between indices of lexicon and lm, or lm index of a word)

            Returns:
                (LMState, float): pair of (new state, score for the current word)
            """
            state_ = self.mapping_states[state]
            if time.monotonic() - self._recent_debug_log_time > 1:
                print(
                    "LM prefix",
                    [model.target_dim.vocab.id_to_label(label_idx) for label_idx in state_.label_seq],
                    f"score {model.target_dim.vocab.id_to_label(token_index)!r}",
                    f"({len(self.mapping_states)} states seen)",
                    f"(cache info {self._calc_next_lm_state.cache_info()})",
                    f"(mem usage {dev_s}: {' '.join(_collect_mem_stats())})",
                )
                self._recent_debug_log_time = time.monotonic()
            outstate = state.child(token_index)
            if outstate not in self.mapping_states:
                self.mapping_states[outstate] = FlashlightLMState(
                    label_seq=state_.label_seq + [token_index], prev_state=state
                )

                if debug_tracemalloc and len(self.mapping_states) % 1_000_000 == 0:
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.compare_to(snapshot_start, "lineno")
                    print(f"[ {len(self.mapping_states)} states, top 100 mallocs ]")
                    for stat in top_stats[:100]:
                        print(stat)

            _, log_probs_raw = self._calc_next_lm_state(state)
            return outstate, log_probs_raw[token_index]

        def finish(self, state: LMState):
            """
            Evaluate eos for language model based on the current lm state

            Returns:
                (LMState, float): pair of (new state, score for the current word)
            """
            return self.score(state, model.eos_idx)

    fl_lm = FlashlightLM()

    from flashlight.lib.text.decoder import LexiconFreeDecoderOptions, LexiconFreeDecoder, CriterionType

    # Some values from hilmes:
    # beam_size=1024,  # Untuned
    # beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
    # beam_threshold=14,  # Untuned

    fl_decoder_opts = LexiconFreeDecoderOptions(
        beam_size=beam_size,
        beam_size_token=beam_size_token,
        beam_threshold=beam_threshold,
        lm_weight=lm_scale,
        sil_score=0.0,
        log_add=False,
        criterion_type=CriterionType.CTC,
    )
    sil_idx = -1  # no silence
    fl_decoder = LexiconFreeDecoder(fl_decoder_opts, fl_lm, sil_idx, model.blank_idx, [])

    assert data.dims_set == {batch_dim, data_spatial_dim, data.feature_dim}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert logits.dims_set == {batch_dim, enc_spatial_dim, model.wb_target_dim}

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob = label_log_prob.copy_transpose((batch_dim, enc_spatial_dim, model.wb_target_dim))
    batch_size, max_seq_len = label_log_prob.raw_tensor.shape[:2]
    assert enc_spatial_dim.dyn_size_ext.dims == (batch_dim,)

    label_log_prob = rf.cast(label_log_prob, "float32")
    label_log_prob = rf.copy_to_device(label_log_prob, "cpu")
    label_log_prob_raw = label_log_prob.raw_tensor.contiguous()
    float_bytes = 4

    print(f"Memory usage {dev_s} after encoder forward:", " ".join(_collect_mem_stats()))
    snapshot_start = tracemalloc.take_snapshot() if debug_tracemalloc else None

    hyps = []
    scores = []
    for batch_idx in range(batch_size):
        emissions_ptr = label_log_prob_raw.data_ptr() + float_bytes * batch_idx * label_log_prob_raw.stride(0)
        seq_len = enc_spatial_dim.dyn_size[batch_idx]
        assert seq_len <= max_seq_len
        results = fl_decoder.decode(emissions_ptr, seq_len, model.wb_target_dim.dimension)
        # I get -1 (silence label?) at the beginning and end in the tokens? Filter those away.
        # These are also additional frames which don't correspond to the input frames?
        # When removing those two frames, the len of tokens (align labels) matches the emission frames
        # (as it should be).
        hyps_per_batch = [[label for label in result.tokens if label >= 0] for result in results]
        scores_per_batch = [result.score for result in results]
        print(
            f"batch {batch_idx + 1}/{batch_size}: {len(results)} hyps,"
            f" best score: {scores_per_batch[0]},"
            f" best seq {_format_align_label_seq(results[0].tokens, model.wb_target_dim)},"
            f" worst score: {scores_per_batch[-1]},"
            f" LM cache info {fl_lm._calc_next_lm_state.cache_info()},"
            f" LM recalc whole seq count {fl_lm._count_recalc_whole_seq},"
            f" mem usage {dev_s}: {' '.join(_collect_mem_stats())}"
        )
        assert all(
            len(hyp) == seq_len for hyp in hyps_per_batch
        ), f"seq_len {seq_len}, hyps lens {[len(hyp) for hyp in hyps_per_batch]}"
        if len(results) >= n_best:
            hyps_per_batch = hyps_per_batch[:n_best]
            scores_per_batch = scores_per_batch[:n_best]
        else:
            hyps_per_batch += [[]] * (n_best - len(results))
            scores_per_batch += [-1e30] * (n_best - len(results))
        assert len(hyps_per_batch) == len(scores_per_batch) == n_best
        hyps_per_batch = [hyp + [model.blank_idx] * (max_seq_len - len(hyp)) for hyp in hyps_per_batch]
        assert all(len(hyp) == max_seq_len for hyp in hyps_per_batch)
        hyps.append(hyps_per_batch)
        scores.append(scores_per_batch)
    fl_lm.reset()
    hyps_pt = torch.tensor(hyps, dtype=torch.int32)
    assert hyps_pt.shape == (batch_size, n_best, max_seq_len)
    scores_pt = torch.tensor(scores, dtype=torch.float32)
    assert scores_pt.shape == (batch_size, n_best)

    beam_dim = Dim(n_best, name="beam")
    out_spatial_dim = enc_spatial_dim
    hyps_r = rf.convert_to_tensor(hyps_pt, dims=(batch_dim, beam_dim, out_spatial_dim), sparse_dim=model.wb_target_dim)
    scores_r = rf.convert_to_tensor(scores_pt, dims=(batch_dim, beam_dim))
    print(f"Memory usage ({dev_s}) after batch:", " ".join(_collect_mem_stats()))
    return hyps_r, scores_r, out_spatial_dim, beam_dim


# RecogDef API
model_recog_flashlight: RecogDef[Model]
model_recog_flashlight.output_with_beam = True
model_recog_flashlight.output_blank_label = "<blank>"
model_recog_flashlight.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def _format_align_label_seq(align_label_seq: List[int], wb_target_dim: Dim) -> str:
    seq_label: List[str] = []  # list of label
    seq_label_idx: List[int] = []  # list of label index
    seq_label_count: List[int] = []  # list of label count
    for align_label in align_label_seq:
        if seq_label_idx and seq_label_idx[-1] == align_label:
            seq_label_count[-1] += 1
        else:
            seq_label.append(wb_target_dim.vocab.id_to_label(align_label) if align_label >= 0 else str(align_label))
            seq_label_idx.append(align_label)
            seq_label_count.append(1)
    return " ".join(f"{label}*{count}" if count > 1 else label for label, count in zip(seq_label, seq_label_count))
