"""
log x = lim_{alpha -> 0} 1/alpha (x^alpha - 1).
Try alpha=1 first.
"""

from __future__ import annotations
from typing import Optional, Any, Tuple, Dict
import tree

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from sisyphus import tk
from i6_experiments.users.zeyer.recog import recog_model
from i6_experiments.users.zeyer.model_interfaces import RecogDef, ModelWithCheckpoint

from .configs import config_24gb_v6, _get_cfg_lrlin_oclr_by_bs_nep
from .conformer_import_moh_att_2023_06_30 import train_exp as lstm_train_exp, Model as LstmModel, _get_ls_task

_sis_prefix: Optional[str] = None


def py():
    from .sis_setup import get_prefix_for_config

    global _sis_prefix

    _sis_prefix = get_prefix_for_config(__file__)

    recog("lstm", "baseline", {})
    recog("lstm", "greedy", {"beam_size": 1})


_models_by_type: Dict[str, ModelWithCheckpoint] = {}


def _get_model(model_type: str) -> ModelWithCheckpoint:
    if model_type in _models_by_type:
        return _models_by_type[model_type]
    if model_type == "lstm":
        # {"dev-clean": 2.31, "dev-other": 5.44, "test-clean": 2.64, "test-other": 5.74}, "best_epoch": 1941
        lstm_model = lstm_train_exp(
            "base-24gb-v6-lrlin1e_5_450k", config_24gb_v6, config_updates=_get_cfg_lrlin_oclr_by_bs_nep(40_000, 2000)
        ).get_epoch(1941)
        _models_by_type[model_type] = lstm_model
    else:
        raise ValueError(f"not handled: model type {model_type!r}")
    return _models_by_type[model_type]


def recog(model_type: str, name: str, config: Dict[str, Any]):
    task = _get_ls_task()
    tk.register_output(
        f"{_sis_prefix}/{model_type}_{name}",
        recog_model(task, _get_model(model_type), recog_def=lstm_model_recog, config=config).output,
    )


def lstm_model_recog(
    *,
    model: LstmModel,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
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
    from returnn.config import get_global_config

    config = get_global_config()
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = config.int("beam_size", 12)
    length_normalization_exponent = config.float("length_normalization_exponent", 1.0)
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    decoder_state = model.decoder_default_initial_state(batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        if i == 0:
            input_embed = rf.zeros(batch_dims_ + [model.target_embed.out_dim], feature_dim=model.target_embed.out_dim)
        else:
            input_embed = model.target_embed(target)
        step_out, decoder_state = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=decoder_state,
        )
        logits = model.decode_logits(input_embed=input_embed, **step_out)
        label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=-1.0e30),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.target_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), decoder_state)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= rf.where(
                ended,
                (i / (i - 1)) ** length_normalization_exponent,
                1.0,
            )

    if i > 0 and length_normalization_exponent != 0:
        seq_log_prob *= (1 / i) ** length_normalization_exponent

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
lstm_model_recog: RecogDef[LstmModel]
lstm_model_recog.output_with_beam = True
lstm_model_recog.output_blank_label = None
lstm_model_recog.batch_size_dependent = False
