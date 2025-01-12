"""
CTC recognition with LM
"""
import sys
import time
from typing import Optional, Any, TypeVar, Sequence, Tuple, Dict, List
import functools

from returnn.tensor import Tensor, Dim, single_step_dim, batch_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.decoder.transformer import TransformerDecoder

from sisyphus import tk

from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, ModelWithCheckpoint

from i6_experiments.users.zeyer.recog import recog_model
from i6_experiments.users.zeyer.collect_model_dataset_stats import collect_statistics

from .ctc import Model, ctc_model_def, _batch_size_factor


_ctc_model_name = (
    "v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2"
    "-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001"
)

# trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b32_1k*: 34.03  -- still running...
# trafo-n96-d512-gelu-drop0-b32_1k: 34.96
# trafo-n24-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k-ep40: 35.60
# ...
# trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k
# _lm_name = "trafo-n96-d512-gelu-drop0-b32_1k"
_lms = {
    "n24-d512": "trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k",
    # "n96-d512": "trafo-n96-d512-gelu-drop0-b32_1k",
}


def py():
    """Sis entry point"""
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = f"{_sis_prefix}/ctc+lm"

    vocab = "spm10k"
    task = get_librispeech_task_raw_v2(vocab=vocab)
    ctc_model = _get_ctc_model(_ctc_model_name)
    prior = get_ctc_prior_probs(ctc_model, task.train_dataset.copy_train_as_static())
    tk.register_output(f"{prefix}/ctc-prior", prior)

    for lm_out_name, lm_name in _lms.items():
        lm = _get_lm_model(lm_name)

        # Our own beam search implementation.
        for beam_size, prior_scale, lm_scale in [
            # (12, 1.0, 1.0),
            (12, 0.0, 1.0),
            # (1, 0.0, 1.0),
            # (1, 1.0, 1.0),
        ]:
            model = get_ctc_with_lm(
                ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
            )
            res = recog_model(
                task=task,
                model=model,
                recog_def=model_recog,
                config={
                    "beam_size": beam_size,
                    "recog_version": 6,
                    "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                },
                search_rqmt={"time": 24},
            )
            tk.register_output(
                f"{prefix}/recog-beam{beam_size}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}", res.output
            )

        # Flashlight beam search implementation.
        # Play around with beam size here.
        for prior_scale, lm_scale in [
            (0.0, 1.0),
            # (0.2, 2.0),
        ]:
            model = get_ctc_with_lm(
                ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
            )
            for name, opts in [
                (
                    "beamToken16-cache0",
                    {
                        "n_best": 32,
                        "beam_size": 1024,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 0,
                    },
                ),
                (
                    "beam128-beamToken16-cache0",
                    {
                        "n_best": 32,
                        "beam_size": 128,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 0,
                    },
                ),
                (
                    "beam16-beamToken16-cache0",
                    {
                        "n_best": 16,
                        "beam_size": 16,
                        "beam_size_token": 16,
                        "beam_threshold": 14,
                        "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
                        "torch_amp": {"dtype": "bfloat16"},
                        "lm_state_lru_initial_cache_size": 0,
                    },
                ),
            ]:
                res = recog_model(
                    task=task,
                    model=model,
                    recog_def=model_recog_flashlight,
                    config=opts,
                    search_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
                )
                tk.register_output(
                    f"{prefix}/recog-fl-{name}-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}",
                    res.output,
                )

        # Flashlight beam search implementation.
        # Play around with scales.
        # for prior_scale, lm_scale in [
        #     (0.0, 1.0),
        #     # (0.2, 2.0),
        # ]:
        #     model = get_ctc_with_lm(
        #         ctc_model=ctc_model, prior=prior, prior_scale=prior_scale, language_model=lm, lm_scale=lm_scale
        #     )
        #     res = recog_model(
        #         task=task,
        #         model=model,
        #         recog_def=model_recog_flashlight,
        #         config={
        #             "n_best": 32,
        #             "beam_size": 1024,
        #             "beam_size_token": 128,
        #             "beam_threshold": 14,
        #             "batch_size": 5_000 * ctc_model.definition.batch_size_factor,
        #             "torch_amp": {"dtype": "bfloat16"},
        #         },
        #         search_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 24},
        #     )
        #     tk.register_output(
        #         f"{prefix}/recog-fl-beamToken128-lm_{lm_out_name}-lmScale{lm_scale}-priorScale{prior_scale}", res.output
        #     )


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


_called_ctc_py_once = False
_model_cache_by_name = {}


def _get_ctc_model(name: str) -> ModelWithCheckpoint:
    # noinspection PyProtectedMember
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
        py as ctc_py,
        _train_experiments as ctc_train_experiments,
    )

    global _called_ctc_py_once

    if name in _model_cache_by_name:
        return _model_cache_by_name[name]

    if not _called_ctc_py_once:
        from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

        with disable_register_output():
            ctc_py()
        _called_ctc_py_once = True

    exp = ctc_train_experiments[name]
    model = exp.get_last_fixed_epoch()
    _model_cache_by_name[name] = model
    return model


_lm_cache_by_name = {}
_called_lm_py_once = False


def _get_lm_model(name: str) -> ModelWithCheckpoint:
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import py as lm_py
    from i6_experiments.users.zeyer.train_v3 import train_models_by_prefix

    global _called_lm_py_once

    if name in _lm_cache_by_name:
        return _lm_cache_by_name[name]

    if not _called_lm_py_once:
        from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

        with disable_register_output():
            lm_py()
        _called_lm_py_once = True

    exp = train_models_by_prefix["lm/" + name]
    model = exp.get_last_fixed_epoch()
    _lm_cache_by_name[name] = model
    return model


def get_ctc_with_lm(
    *,
    ctc_model: ModelWithCheckpoint,
    prior: Optional[tk.Path] = None,
    prior_type: str = "prob",
    prior_scale: Optional[float] = None,
    language_model: ModelWithCheckpoint,
    lm_scale: float,
) -> ModelWithCheckpoint:
    # Keep CTC model config as-is, extend below for prior and LM.
    ctc_model_def = ctc_model.definition
    if isinstance(ctc_model_def, ModelDefWithCfg):
        config: Dict[str, Any] = ctc_model_def.config.copy()
    else:
        config = {}

    # Add prior.
    # Then the CTC Model log_probs_wb_from_logits will include the prior.
    if prior is not None:
        assert prior_scale is not None
    if prior_scale:
        assert prior is not None
        config.update(
            {
                "ctc_prior_type": "static",
                "ctc_prior_scale": prior_scale,
                "static_prior": {"type": prior_type, "file": prior},
            }
        )

    # Add LM.
    # LM has _model_def_dict in config. Put that as _lm_model_def_dict.
    config.update(
        {
            "_lm_model_def_dict": language_model.definition.config["_model_def_dict"],
            "lm_scale": lm_scale,
        }
    )
    config.setdefault("preload_from_files", {})["lm"] = {"prefix": "lm.", "filename": language_model.checkpoint}

    return ModelWithCheckpoint(
        definition=ModelDefWithCfg(model_def=ctc_model_ext_def, config=config),
        checkpoint=ctc_model.checkpoint,
    )


def ctc_model_ext_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    lm = rf.build_from_dict(config.typed_value("_lm_model_def_dict"), vocab_dim=target_dim)
    lm_scale = config.typed_value("lm_scale", None)
    assert isinstance(lm_scale, (int, float))
    model = ctc_model_def(epoch=epoch, in_dim=in_dim, target_dim=target_dim)
    model.lm = lm
    model.lm_scale = lm_scale
    return model


ctc_model_ext_def: ModelDef[Model]
ctc_model_ext_def.behavior_version = 21
ctc_model_ext_def.backend = "torch"
ctc_model_ext_def.batch_size_factor = _batch_size_factor


def model_recog(
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
    import tree
    from returnn.config import get_global_config

    config = get_global_config()
    beam_size = config.int("beam_size", 12)
    version = config.int("recog_version", 1)
    assert version == 6

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_ta = TensorArray.unstack(label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB

    # We usually have TransformerDecoder, but any other type would also be ok when it has the same API.
    # noinspection PyUnresolvedReferences
    lm: TransformerDecoder = model.lm
    # noinspection PyUnresolvedReferences
    lm_scale: float = model.lm_scale

    lm_state = lm.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
    lm_logits, lm_state = lm(
        target,
        spatial_dim=single_step_dim,
        state=lm_state,
    )  # Batch, InBeam, Vocab / ...
    lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
    lm_log_probs *= lm_scale

    lm_log_probs = lm_log_probs.copy_compatible_to_dims(batch_dims_ + [model.target_dim])
    print("* lm_log_probs initial:", lm_log_probs)
    print(
        f"* argmax LM begin: {model.target_dim.vocab.id_to_label(lm_log_probs.raw_tensor[0, 0].argmax().cpu().item())}"
    )

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        # print(f"* prev_target_wb {model.wb_target_dim.vocab.id_to_label(prev_target_wb.raw_tensor.cpu()[0, 0].item())}")

        seq_log_prob = seq_log_prob + label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        seq_log_prob = seq_log_prob.copy_compatible_to_dims(batch_dims + [beam_dim, model.wb_target_dim])
        print(
            f"* argmax seq_log_prob (before LM) t={t}:"
            f" {model.wb_target_dim.vocab.id_to_label(seq_log_prob.raw_tensor[0, 0].argmax().cpu().item())}"
        )

        # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
        seq_log_prob += rf.where(
            (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
            _target_dense_extend_blank(
                lm_log_probs,
                target_dim=model.target_dim,
                wb_target_dim=model.wb_target_dim,
                blank_idx=model.blank_idx,
                value=0.0,
            ),
            0.0,
        )  # Batch, InBeam -> VocabWB

        seq_log_prob = seq_log_prob.copy_compatible_to_dims(batch_dims + [beam_dim, model.wb_target_dim])
        print(
            f"* argmax seq_log_prob (with LM) t={t}:"
            f" {model.wb_target_dim.vocab.id_to_label(seq_log_prob.raw_tensor[0, 0].argmax().cpu().item())}"
        )

        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        target_wb = target_wb.copy_compatible_to_dims(batch_dims + [beam_dim])  # Batch, Beam -> VocabWB
        print(
            f"* target_wb t={t}:" f" {model.wb_target_dim.vocab.id_to_label(target_wb.raw_tensor[0, 0].cpu().item())}"
        )

        lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, Vocab
        lm_state = tree.map_structure(functools.partial(_gather_backrefs, backrefs=backrefs), lm_state)
        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB
        got_new_label = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _target_remove_blank(
                target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
            ),
            prev_target,
        )  # Batch, Beam -> Vocab

        (target_, lm_state_), packed_new_label_dim, packed_new_label_dim_map = _masked_select_tree(
            (target, lm_state),
            mask=got_new_label,
            dims=batch_dims + [beam_dim],
        )

        if packed_new_label_dim.get_dim_value() > 0:
            print(f"* feed target {model.target_dim.vocab.id_to_label(target_.raw_tensor.cpu()[0].item())}")

            lm_logits_, lm_state_ = lm(
                target_,
                spatial_dim=single_step_dim,
                state=lm_state_,
            )  # Flat_Batch_Beam, Vocab / ...
            lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
            lm_log_probs_ *= lm_scale

            lm_log_probs_ = lm_log_probs_.copy_compatible_to_dims([packed_new_label_dim, model.target_dim])
            print(
                f"* argmax LM after feed:"
                f" {model.target_dim.vocab.id_to_label(lm_log_probs_.raw_tensor[0].argmax().cpu().item())}"
            )

            lm_log_probs, lm_state = _masked_scatter_tree(
                (lm_log_probs_, lm_state_),
                (lm_log_probs, lm_state),
                mask=got_new_label,
                dims=batch_dims + [beam_dim],
                in_dim=packed_new_label_dim,
                dim_map=packed_new_label_dim_map,
            )  # Batch, Beam, Vocab / ...

    # seq_log_prob, lm_log_probs: Batch, Beam
    # Add LM EOS score at the end.
    seq_log_prob += rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)  # Batch, Beam -> VocabWB

    # Backtrack via backrefs, resolve beams.
    seq_targets_wb_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target_wb in zip(seq_backrefs[::-1], seq_targets_wb[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_wb_.insert(0, rf.gather(target_wb, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets_wb__ = TensorArray(seq_targets_wb_[0])
    for target_wb in seq_targets_wb_:
        seq_targets_wb__ = seq_targets_wb__.push_back(target_wb)
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_wb__.stack(axis=out_spatial_dim)

    return seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = True  # our models currently just are batch-size-dependent...


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


def get_ctc_prior_probs(
    ctc_model: ModelWithCheckpoint, dataset: DatasetConfig, config: Optional[Dict[str, Any]] = None
) -> tk.Path:
    """
    :return: CTC prior, in prob space (not log prob)
    """
    # Note: there is also compute_model_softmax_prior_statistics,
    # which assumes a slightly different model API though,
    # and we must call log_probs_wb_from_logits to have it correct in any case.
    return collect_statistics(
        model=ctc_model, dataset=dataset, forward_def=_ctc_model_softmax_prior_returnn_forward, config=config
    ).mean


def _ctc_model_softmax_prior_returnn_forward(
    source: Tensor, /, in_spatial_dim: Dim, model: Model
) -> Tuple[Tensor, Dim]:
    """ForwardDef API"""
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim

    logits, enc, enc_spatial_dim = model(source, in_spatial_dim=in_spatial_dim)
    assert isinstance(logits, Tensor) and isinstance(enc_spatial_dim, Dim)
    assert logits.feature_dim  # we expect a feature dim
    assert enc_spatial_dim in logits.dims
    log_probs = model.log_probs_wb_from_logits(logits)
    assert isinstance(log_probs, Tensor)
    probs = rf.exp(log_probs)  # the statistics take the average over this, thus prob space, not log prob
    return probs, enc_spatial_dim


T = TypeVar("T")


def _gather_backrefs(s: T, *, backrefs: Tensor) -> T:
    if isinstance(s, Tensor):
        if backrefs.sparse_dim in s.dims:
            return rf.gather(s, indices=backrefs)  # really the default case
        return s  # e.g. scalar or so, independent from beam
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        assert backrefs.sparse_dim not in s.dyn_size_ext.dims  # currently not supported, also not expected
        return s
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


def _masked_select_tree(s: T, *, mask: Tensor, dims: Sequence[Dim]) -> Tuple[T, Dim, Dict[Dim, Dim]]:
    import tree

    packed_new_label_dim = Dim(None, name="packed_new_label")  # Flat_Batch_InBeam
    packed_new_label_dim_map = {}
    tree.map_structure(
        functools.partial(
            _masked_select_prepare_dims,
            mask=mask,
            dims=dims,
            out_dim=packed_new_label_dim,
            dim_map=packed_new_label_dim_map,
        ),
        s,
    )
    s = tree.map_structure(
        functools.partial(
            _masked_select,
            mask=mask,
            dims=dims,
            out_dim=packed_new_label_dim,
            dim_map=packed_new_label_dim_map,
        ),
        s,
    )
    return s, packed_new_label_dim, packed_new_label_dim_map


def _masked_select_prepare_dims(s, *, mask: Tensor, dims: Sequence[Dim], out_dim: Dim, dim_map: Dict[Dim, Dim]):
    if isinstance(s, Tensor):
        return s  # ignored at this stage
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        if not any(d in s.dyn_size_ext.dims for d in dims):
            return s
        if s in dim_map:
            return dim_map[s]
        new_dyn_size = _masked_select(s.dyn_size_ext, mask=mask, dims=dims, out_dim=out_dim, dim_map=dim_map)
        new_dim = Dim(new_dyn_size, name=s.name + "_")
        dim_map[s] = new_dim
        return new_dim
    raise TypeError(f"_masked_select_prepare_dims: unexpected type ({type(s)})")


def _masked_select(s: T, *, mask: Tensor, dims: Sequence[Dim], out_dim: Dim, dim_map: Dict[Dim, Dim]) -> T:
    if isinstance(s, Tensor):
        if not any(d in s.dims for d in dims):
            return s  # e.g. scalar or so, independent from dims
        # For the masked_select, we need that all masked dims are present, so add them if not.
        # (E.g., when we mask [batch,beam], but we only have [batch], we need to add the beam dim.)
        if any(d not in s.dims for d in dims):
            s = rf.expand_dims(s, dims=[d for d in dims if d not in s.dims])
        # The packing itself (masked_select).
        s, _ = rf.masked_select(s, mask=mask, dims=dims, out_dim=out_dim)
        # In the resulting tensor, potentially replace dims.
        # In addition to the dim replacement, we also might need to slice, as the size might be smaller.
        if any(d in dim_map for d in s.dims):
            for d in s.dims:
                if d in dim_map:
                    s, _ = rf.slice(s, axis=d, size=dim_map[d])
        return s
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        if not any(d in s.dyn_size_ext.dims for d in dims):
            return s
        assert s in dim_map
        return dim_map[s]
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


def _masked_scatter_tree(
    s: T, backup: T, *, mask: Tensor, dims: Sequence[Dim], in_dim: Dim, dim_map: Dict[Dim, Dim]
) -> T:
    import tree

    reverse_dim_map = {v: k for k, v in dim_map.items()}
    merged_dim_map = {}

    tree.map_structure(
        functools.partial(
            _masked_scatter_merge_dims,
            mask=mask,
            dims=dims,
            in_dim=in_dim,
            reverse_dim_map=reverse_dim_map,
            merged_dim_map=merged_dim_map,
        ),
        s,
        backup,
    )
    s = tree.map_structure(
        functools.partial(
            _masked_scatter,
            mask=mask,
            dims=dims,
            in_dim=in_dim,
            reverse_dim_map=reverse_dim_map,
            merged_dim_map=merged_dim_map,
        ),
        s,
        backup,
    )
    return s


def _masked_scatter_merge_dims(
    s: T,
    backup: T,
    *,
    mask: Tensor,
    dims: Sequence[Dim],
    in_dim: Dim,
    reverse_dim_map: Dict[Dim, Dim],
    merged_dim_map: Dict[Dim, Dim],
) -> T:
    if isinstance(s, Tensor):
        return s
    if isinstance(s, Dim):
        # This is slightly more complex than in the _masked_select case:
        # We need to merge the s and backup depending on the mask.
        if s in reverse_dim_map:
            s = reverse_dim_map[s]
        if s == backup:
            return s
        if s in merged_dim_map:
            return merged_dim_map[s]
        # Note: s/backup might even be static dims.
        new_size = _masked_scatter(
            s.get_size_tensor(),
            backup.get_size_tensor(),
            mask=mask,
            dims=dims,
            in_dim=in_dim,
            reverse_dim_map=reverse_dim_map,
            merged_dim_map=merged_dim_map,
        )
        new_dim = Dim(new_size, name=s.name + "_")
        merged_dim_map[s] = new_dim
        merged_dim_map[backup] = new_dim
        return new_dim
    raise TypeError(f"_masked_scatter_merge_dims: unexpected type ({type(s)})")


def _masked_scatter(
    s: T,
    backup: T,
    *,
    mask: Tensor,
    dims: Sequence[Dim],
    in_dim: Dim,
    reverse_dim_map: Dict[Dim, Dim],
    merged_dim_map: Dict[Dim, Dim],
) -> T:
    if isinstance(s, Tensor):
        if in_dim not in s.dims:
            return s  # e.g. scalar or so, independent from masking
        assert isinstance(backup, Tensor)
        # Do the reverse of _masked_select above.
        # First replace the dims back.
        if any(d in reverse_dim_map for d in s.dims):
            for d in s.dims:
                if d in reverse_dim_map:
                    s = _expand_slice(s, axis=d, expanded_size=reverse_dim_map[d])
        # We also might need to replace newly merged dims, both in s and backup.
        for d in s.dims:
            if d in merged_dim_map:
                s, _ = rf.slice(s, axis=d, size=merged_dim_map[d])
        # There is currently the implicit assumption that the backup might need extra padding,
        # while the s needs slicing...
        # (We think of the hist_dim, where s should only have more frames than backup, or the same.)
        for d in backup.dims:
            if d in merged_dim_map:
                backup = _expand_slice(backup, axis=d, expanded_size=merged_dim_map[d])
        # The unpacking itself (reversing the masked_select, i.e. masked_scatter).
        s = rf.masked_scatter(s, backup, mask=mask, dims=dims, in_dim=in_dim)
        # Now remove potential added dims.
        for d in s.dims:
            if d not in backup.dims:
                s = rf.gather(s, axis=d, indices=0)
        return s
    if isinstance(s, Dim):
        # This is slightly more complex than in the _masked_select case:
        # We need to merge the s and backup depending on the mask.
        if s in reverse_dim_map:
            s = reverse_dim_map[s]
        if s in merged_dim_map:
            return merged_dim_map[s]
        return s
    raise TypeError(f"_masked_scatter: unexpected type ({type(s)})")


def _expand_slice(source: Tensor, axis: Dim, expanded_size: Dim) -> Tensor:
    res, _ = rf.pad(
        source,
        axes=[axis],
        padding=[(0, expanded_size.get_dim_value_tensor() - axis.get_dim_value_tensor())],
        out_dims=[expanded_size],
    )
    return res


def _target_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == wb_target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, target_dim)


def _target_dense_extend_blank(
    target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int, value: float
) -> Tensor:
    assert target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
    return res


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
