from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
import math
import numpy as np
import torch
import hashlib
import contextlib
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)

from i6_experiments.users.gaudino.models.asr.rf.conformer_rnnt.model_conformer_rnnt import Model

_batch_size_factor = 160

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import trafo_lm_kazuki_import

def model_recog(
    *,
    model: Model,
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
    assert (
        not model.language_model
    )  # not implemented here. use the pure PyTorch search instead

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12
    length_normalization_exponent = 1.0
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    decoder_state = model.decoder_default_initial_state(
        batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim
    )
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim_w_blank)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        # if i == 0:
        #     input_embed = rf.zeros(
        #         batch_dims_ + [model.target_embed.out_dim],
        #         feature_dim=model.target_embed.out_dim,
        #     )
        # else:
        #     input_embed = model.target_embed(target)
        step_out, decoder_state = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=target,
            state=decoder_state,
        )
        # logits = model.decode_logits(input_embed=input_embed, **step_out)
        label_log_prob = rf.log_softmax(step_out["output"], axis=model.target_dim)

        # TODO: implement rnnt search
        breakpoint()
        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(
                model.eos_idx,
                axis=model.target_dim,
                label_value=0.0,
                other_value=-1.0e30,
            ),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob,
            k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
            axis=[beam_dim, model.target_dim],
        )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        decoder_state = tree.map_structure(
            lambda s: rf.gather(s, indices=backrefs), decoder_state
        )
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
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
# output_blank_label=blank is actually wrong for AED, but now we don't change it anymore
# because it would change all recog hashes.
# Also, it does not matter too much -- it will just cause an extra SearchRemoveLabelJob,
# which will not have any effect here.
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False


def model_recog_pure_torch(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Optional[Tensor] = None,
    targets_spatial_dim: Optional[Dim] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        recog results info: key -> {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import torch
    import time
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_v5 import (
        BeamSearchOptsV5,
        beam_search_v5,
    )
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_sep_ended import (
        BeamSearchDynBeamOpts,
        beam_search_sep_ended,
    )
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_sep_ended_keep_v6 import (
        BeamSearchSepEndedKeepOpts,
        beam_search_sep_ended_keep_v6,
    )
    from i6_experiments.users.zeyer.decoding.beam_search_torch.scorers.length_reward import (
        LengthRewardScorer,
    )
    from i6_experiments.users.zeyer.decoding.beam_search_torch.scorers.shallow_fusion import (
        ShallowFusedLabelScorers,
    )
    from returnn.config import get_global_config

    config = get_global_config()

    torch.cuda.set_sync_debug_mode(
        1
    )  # debug CUDA sync. does not hurt too much to leave this always in?
    start_time = time.perf_counter_ns()

    data_concat_zeros = config.float("data_concat_zeros", 0)
    if data_concat_zeros:
        data_concat_zeros_dim = Dim(
            int(data_concat_zeros * _batch_size_factor * 100), name="data_concat_zeros"
        )
        data, data_spatial_dim = rf.concat(
            (data, data_spatial_dim),
            (rf.zeros([data_concat_zeros_dim]), data_concat_zeros_dim),
            allow_broadcast=True,
        )

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    assert (
        len(batch_dims) == 1
    ), batch_dims  # not implemented otherwise, simple to add...
    batch_dim = batch_dims[0]
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")

    if data.raw_tensor.device.type == "cuda":
        # Just so that timing of encoder is correct.
        torch.cuda.synchronize(data.raw_tensor.device)

    enc_end_time = time.perf_counter_ns()

    beam_search_version = config.typed_value("beam_search_version", 1)
    beam_search_func = {
        5: beam_search_v5,
        "sep_ended": beam_search_sep_ended,
        "sep_ended_keep_v6": beam_search_sep_ended_keep_v6,
    }[beam_search_version]
    if beam_search_version == "sep_ended":
        beam_search_opts_cls = BeamSearchDynBeamOpts
    elif isinstance(beam_search_version, str) and beam_search_version.startswith(
        "sep_ended_keep"
    ):
        beam_search_opts_cls = BeamSearchSepEndedKeepOpts
    elif isinstance(beam_search_version, int) and beam_search_version >= 5:
        beam_search_opts_cls = BeamSearchOptsV5
    else:
        raise ValueError(f"unexpected {beam_search_version=}")
    beam_search_opts = (config.typed_value("beam_search_opts", None) or {}).copy()
    if beam_search_opts.get("beam_size") is None:
        beam_search_opts["beam_size"] = config.int("beam_size", 12)
    if beam_search_opts.get("length_normalization_exponent") is None:
        beam_search_opts["length_normalization_exponent"] = config.float(
            "length_normalization_exponent", 1.0
        )
    if beam_search_opts.get("length_reward") is None:
        beam_search_opts["length_reward"] = config.float("length_reward", 0.0)
    extra = {}
    out_individual_seq_scores = None
    if config.bool("beam_search_collect_individual_seq_scores", False):
        out_individual_seq_scores = {}
        extra["out_individual_seq_scores"] = out_individual_seq_scores
    cheating = config.bool("cheating", False)
    if cheating:
        assert targets and targets_spatial_dim
        extra["cheating_targets"] = targets.copy_compatible_to_dims_raw(
            [batch_dim, targets_spatial_dim]
        )
        extra[
            "cheating_targets_seq_len"
        ] = targets_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw([batch_dim])
    coverage_scale = beam_search_opts.pop("attention_coverage_scale", 0.0)
    coverage_opts = beam_search_opts.pop("attention_coverage_opts", {})
    neg_coverage_scale = beam_search_opts.pop("neg_attention_coverage_scale", 0.0)
    neg_coverage_opts = beam_search_opts.pop("neg_attention_coverage_opts", {})
    monotonicity_scale = beam_search_opts.pop("attention_monotonicity_scale", 0.0)
    monotonicity_opts = beam_search_opts.pop("attention_monotonicity_opts", {})
    max_seq_len_factor = beam_search_opts.pop("max_seq_len_factor", 1)
    if max_seq_len_factor != 1:
        max_seq_len = rf.cast(max_seq_len * max_seq_len_factor, max_seq_len.dtype)
    label_scorer = ShallowFusedLabelScorers()
    if coverage_scale or neg_coverage_scale or cheating:
        label_scorer.label_scorers.update(
            get_label_scorer_and_coverage_scorer_pure_torch(
                model=model,
                batch_dim=batch_dim,
                enc=enc,
                enc_spatial_dim=enc_spatial_dim,
                coverage_opts=coverage_opts,
                coverage_scale=coverage_scale,
                neg_coverage_scale=neg_coverage_scale,
                neg_coverage_opts=neg_coverage_opts,
                monotonicity_scale=monotonicity_scale,
                monotonicity_opts=monotonicity_opts,
                always_add_scorers=cheating,
            )
        )
    else:
        label_scorer.label_scorers["decoder"] = (
            get_label_scorer_pure_torch(
                model=model,
                batch_dim=batch_dim,
                enc=enc,
                enc_spatial_dim=enc_spatial_dim,
            ),
            1.0,
        )
    if isinstance(beam_search_version, str) or beam_search_version >= 5:
        len_reward = beam_search_opts.pop("length_reward", 0.0)
        if len_reward or cheating:
            label_scorer.label_scorers["length_reward"] = (
                LengthRewardScorer(),
                len_reward,
            )
    if model.language_model:
        lm_scale = beam_search_opts.pop("lm_scale")  # must be defined with LM
        label_scorer.label_scorers["lm"] = (
            model.language_model_make_label_scorer(),
            lm_scale,
        )

    print("** max seq len:", max_seq_len.raw_tensor)

    # Beam search happening here:
    (
        seq_targets,  # [Batch,FinalBeam,OutSeqLen]
        seq_log_prob,  # [Batch,FinalBeam]
        out_seq_len,  # [Batch,FinalBeam]
    ) = beam_search_func(
        label_scorer,
        batch_size=int(batch_dim.get_dim_value()),
        max_seq_len=max_seq_len.copy_compatible_to_dims_raw([batch_dim]),
        device=data.raw_tensor.device,
        opts=beam_search_opts_cls(
            **beam_search_opts,
            bos_label=model.bos_idx,
            eos_label=model.eos_idx,
            num_labels=model.target_dim.dimension,
        ),
        **extra,
    )

    beam_dim = Dim(seq_log_prob.shape[1], name="beam")
    out_spatial_dim = Dim(
        rf.convert_to_tensor(
            out_seq_len, dims=[batch_dim, beam_dim], name="out_spatial"
        )
    )
    seq_targets_t = rf.convert_to_tensor(
        seq_targets,
        dims=[batch_dim, beam_dim, out_spatial_dim],
        sparse_dim=model.target_dim,
    )
    seq_log_prob_t = rf.convert_to_tensor(seq_log_prob, dims=[batch_dim, beam_dim])

    search_end_time = time.perf_counter_ns()
    data_seq_len_sum = rf.reduce_sum(
        data_spatial_dim.dyn_size_ext, axis=data_spatial_dim.dyn_size_ext.dims
    )
    data_seq_len_sum_secs = data_seq_len_sum.raw_tensor / _batch_size_factor / 100.0
    data_seq_len_max_seqs = (
        data_spatial_dim.get_dim_value() / _batch_size_factor / 100.0
    )
    out_len_longest_sum = rf.reduce_sum(
        rf.reduce_max(out_spatial_dim.dyn_size_ext, axis=beam_dim), axis=batch_dim
    )
    print(
        "TIMINGS:",
        ", ".join(
            (
                f"batch size {data.get_batch_dim_tag().get_dim_value()}",
                f"data len max {data_spatial_dim.get_dim_value()} ({data_seq_len_max_seqs:.2f} secs)",
                f"data len sum {data_seq_len_sum.raw_tensor} ({data_seq_len_sum_secs:.2f} secs)",
                f"enc {enc_end_time - start_time} ns",
                f"enc len max {enc_spatial_dim.get_dim_value()}",
                f"dec {search_end_time - enc_end_time} ns",
                f"out len max {out_spatial_dim.get_dim_value()}",
                f"out len longest sum {out_len_longest_sum.raw_tensor}",
            )
        ),
    )

    extra_recog_results = {}
    if out_individual_seq_scores:
        for k, v in out_individual_seq_scores.items():
            extra_recog_results[f"score:{k}"] = rf.convert_to_tensor(
                v.expand(batch_dim.get_dim_value(), beam_dim.get_dim_value()),
                dims=[batch_dim, beam_dim],
            )

    return seq_targets_t, seq_log_prob_t, extra_recog_results, out_spatial_dim, beam_dim


def get_label_scorer_pure_torch(
    *,
    model: Model,
    batch_dim: Dim,
    enc: Dict[str, Tensor],
    enc_spatial_dim: Dim,
):
    import torch
    import functools
    from i6_experiments.users.zeyer.decoding.beam_search_torch.interface import (
        LabelScorerIntf,
        StateObjTensorExt,
        StateObjIgnored,
    )

    class LabelScorer(LabelScorerIntf):
        """label scorer"""

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            beam_dim = Dim(1, name="initial-beam")
            batch_dims_ = [batch_dim, beam_dim]
            decoder_state = model.decoder_default_initial_state(
                batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim
            )
            return tree.map_structure(
                functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim),
                decoder_state,
            )

        def max_remaining_seq_score(
            self, *, state: Any, max_remaining_steps: torch.Tensor, device: torch.device
        ) -> torch.Tensor:
            """max remaining"""
            return torch.zeros((1, 1), device=device)

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            beam_dim = Dim(prev_label.shape[1], name="beam")

            def _map_raw_to_tensor(v):
                if isinstance(v, StateObjTensorExt):
                    tensor: Tensor = v.extra
                    tensor = tensor.copy_template_new_dim_tags(
                        (batch_dim, beam_dim) + tensor.dims[2:], keep_special_axes=True
                    )
                    tensor.raw_tensor = v.tensor
                    return tensor
                elif isinstance(v, StateObjIgnored):
                    return v.content
                else:
                    raise TypeError(
                        f"_map_raw_to_tensor: unexpected {v} ({type(v).__name__})"
                    )

            input_embed = model.target_embed(
                rf.convert_to_tensor(
                    prev_label, dims=[batch_dim, beam_dim], sparse_dim=model.target_dim
                )
            )
            decode_out, decoder_state = model.loop_step(
                **enc,
                enc_spatial_dim=enc_spatial_dim,
                input_embed=input_embed,
                state=tree.map_structure(_map_raw_to_tensor, prev_state),
            )
            logits = model.decode_logits(input_embed=input_embed, **decode_out)
            label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
            assert set(label_log_prob.dims) == {batch_dim, beam_dim, model.target_dim}

            return (
                self._map_tensor_to_raw(label_log_prob, beam_dim=beam_dim).tensor,
                tree.map_structure(
                    functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim),
                    decoder_state,
                ),
            )

        @staticmethod
        def _map_tensor_to_raw(v, *, beam_dim: Dim):
            if isinstance(v, Tensor):
                if beam_dim not in v.dims:
                    return StateObjIgnored(v)
                batch_dims_ = [batch_dim, beam_dim]
                v = v.copy_transpose(
                    batch_dims_ + [dim for dim in v.dims if dim not in batch_dims_]
                )
                raw = v.raw_tensor
                return StateObjTensorExt(raw, v.copy_template())
            elif isinstance(v, Dim):
                return StateObjIgnored(v)
            else:
                raise TypeError(
                    f"_map_tensor_to_raw: unexpected {v} ({type(v).__name__})"
                )

    return LabelScorer()


# RecogDef API
model_recog_pure_torch: RecogDef[Model]
model_recog_pure_torch.output_with_beam = True
model_recog_pure_torch.output_blank_label = None
model_recog_pure_torch.batch_size_dependent = False


def get_label_scorer_and_coverage_scorer_pure_torch(
    *,
    model: Model,
    batch_dim: Dim,
    enc: Dict[str, Tensor],
    enc_spatial_dim: Dim,
    coverage_scale: float = 0.0,
    coverage_opts: Optional[Dict[str, Any]] = None,
    neg_coverage_scale: float = 0.0,
    neg_coverage_opts: Optional[Dict[str, Any]] = None,
    monotonicity_scale: float = 0.0,
    monotonicity_opts: Optional[Dict[str, Any]] = None,
    always_add_scorers: bool = False,
):
    import torch
    import functools
    from returnn.frontend.decoder.transformer import TransformerDecoderLayer
    from i6_experiments.users.zeyer.decoding.beam_search_torch.interface import (
        LabelScorerIntf,
        StateObjTensorExt,
        StateObjIgnored,
    )

    accum_att_weights = rf.zeros(())  # [Batch,Beam,kv_axis]
    att_weights_dec_frame: Tensor  # [Batch,Beam,kv_axis]
    beam_dim: Dim

    raise NotImplementedError("need more work here")  # TODO...

    model_att_reduce_type = coverage_opts.get("model_att_reduce_type", "max")

    def hooked_cross_att(
        self: rf.CrossAttention, q: Tensor, k: Tensor, v: Tensor, *, kv_axis: Dim
    ) -> Tensor:
        """apply attention"""
        nonlocal att_weights_dec_frame
        # Standard dot attention, inline rf.dot_attention.
        q *= self.key_dim_per_head.dimension**-0.5
        energy = rf.matmul(q, k, reduce=self.key_dim_per_head)
        att_weights = rf.softmax(energy, axis=kv_axis)
        if model_att_reduce_type == "max":
            att_weights_dec_frame = rf.maximum(
                att_weights_dec_frame, rf.reduce_max(att_weights, axis=self.num_heads)
            )
        elif model_att_reduce_type == "avg":
            att_weights_dec_frame += rf.reduce_mean(
                att_weights, axis=self.num_heads
            ) * (1 / len(model.decoder.layers))
        else:
            raise ValueError(f"invalid model_att_reduce_type {model_att_reduce_type!r}")
        # Masking not needed because softmax should already have masked,
        # so we have 0.0 att weights for padded frames.
        att = rf.matmul(att_weights, v, reduce=kv_axis, use_mask=False)
        if v.feature_dim in att.dims:
            att.feature_dim = v.feature_dim
        output, _ = rf.merge_dims(
            att,
            dims=(self.num_heads, self.value_dim_per_head),
            out_dim=self.value_dim_total,
        )
        if self.proj:
            output = self.proj(output)
        return output

    for layer in model.decoder.layers:
        layer: TransformerDecoderLayer
        layer.cross_att.attention = functools.partial(
            hooked_cross_att, self=layer.cross_att
        )

    class LabelScorer(LabelScorerIntf):
        """label scorer"""

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            beam_dim = Dim(1, name="initial-beam")
            batch_dims_ = [batch_dim, beam_dim]
            decoder_state = model.decoder_default_initial_state(
                batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim
            )
            if coverage_scale or neg_coverage_scale or always_add_scorers:
                decoder_state["accum_att_weights"] = rf.zeros(batch_dims_)
            return tree.map_structure(
                functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim),
                decoder_state,
            )

        def max_remaining_seq_score(
            self, *, state: Any, max_remaining_steps: torch.Tensor, device: torch.device
        ) -> torch.Tensor:
            """max remaining"""
            return torch.zeros((1, 1), device=device)

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            nonlocal beam_dim
            beam_dim = Dim(prev_label.shape[1], name="beam")

            def _map_raw_to_tensor(v):
                if isinstance(v, StateObjTensorExt):
                    tensor: Tensor = v.extra
                    tensor = tensor.copy_template_new_dim_tags(
                        (batch_dim, beam_dim) + tensor.dims[2:], keep_special_axes=True
                    )
                    tensor.raw_tensor = v.tensor
                    return tensor
                elif isinstance(v, StateObjIgnored):
                    return v.content
                else:
                    raise TypeError(
                        f"_map_raw_to_tensor: unexpected {v} ({type(v).__name__})"
                    )

            prev_state = tree.map_structure(_map_raw_to_tensor, prev_state)

            nonlocal accum_att_weights, att_weights_dec_frame
            accum_att_weights = prev_state["accum_att_weights"]
            att_weights_dec_frame = rf.zeros(())
            logits, decoder_state = model.decoder(
                rf.convert_to_tensor(
                    prev_label, dims=[batch_dim, beam_dim], sparse_dim=model.target_dim
                ),
                spatial_dim=single_step_dim,
                encoder=enc,
                state=prev_state,
            )
            accum_att_weights += att_weights_dec_frame
            if coverage_scale or neg_coverage_scale or always_add_scorers:
                decoder_state["accum_att_weights"] = accum_att_weights
            label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
            assert set(label_log_prob.dims) == {batch_dim, beam_dim, model.target_dim}

            return (
                self._map_tensor_to_raw(label_log_prob, beam_dim=beam_dim).tensor,
                tree.map_structure(
                    functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim),
                    decoder_state,
                ),
            )

        @staticmethod
        def _map_tensor_to_raw(v, *, beam_dim: Dim):
            if isinstance(v, Tensor):
                if beam_dim not in v.dims:
                    return StateObjIgnored(v)
                batch_dims_ = [batch_dim, beam_dim]
                v = v.copy_transpose(
                    batch_dims_ + [dim for dim in v.dims if dim not in batch_dims_]
                )
                raw = v.raw_tensor
                return StateObjTensorExt(raw, v.copy_template())
            elif isinstance(v, Dim):
                return StateObjIgnored(v)
            else:
                raise TypeError(
                    f"_map_tensor_to_raw: unexpected {v} ({type(v).__name__})"
                )

    class CoverageScorer(LabelScorerIntf):
        """coverage

        Google NMT: https://arxiv.org/pdf/1609.08144.pdf
        Alternative: https://arxiv.org/abs/1612.02695
        Another alternative: https://arxiv.org/pdf/2105.00982.pdf
        """

        def __init__(self, opts: Dict[str, Any]):
            self.opts = opts

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            return {"prev_score": torch.zeros([batch_size, 1], device=device)}

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            prev_label  # noqa  # unused
            # We assume the label scorer has run before us (make sure by right ordering).
            accum_att_weights_ = accum_att_weights
            assert set(accum_att_weights_.dims) == {
                batch_dim,
                beam_dim,
                enc_spatial_dim,
            }
            cov_type = self.opts.get("type", "log1p")
            if self.opts.get("rescale", False):
                accum_att_weights_ /= rf.maximum(
                    rf.reduce_max(accum_att_weights_, axis=enc_spatial_dim), 1.0
                )
            if (
                cov_type == "log1p"
            ):  # log1p, to avoid having lots of negative numbers. So this starts more around 0.0.
                coverage_score = rf.log1p(rf.minimum(accum_att_weights_, 1.0))
            elif (
                cov_type == "log"
            ):  # orig Google NMT: https://arxiv.org/pdf/1609.08144.pdf, but clipped
                eps = self.opts.get("eps", 0.0)
                clip_min = self.opts.get("clip_min", 0.01)
                coverage_score = rf.log(
                    rf.clip_by_value(accum_att_weights_, clip_min, 1.0) + eps
                )
            elif cov_type == "indicator":
                threshold = self.opts.get("threshold", 0.5)
                coverage_score = rf.where(accum_att_weights_ >= threshold, 1.0, 0.0)
            elif cov_type == "relu_upper":
                threshold = self.opts.get("threshold", 0.5)
                coverage_score = rf.where(
                    accum_att_weights_ >= threshold, accum_att_weights_ - threshold, 0.0
                )
            else:
                raise ValueError(f"invalid coverage opts type {cov_type!r}")
            coverage_score = rf.reduce_sum(coverage_score, axis=enc_spatial_dim)
            coverage_score_raw = coverage_score.copy_compatible_to_dims_raw(
                (batch_dim, beam_dim)
            )
            state = {"prev_score": coverage_score_raw}
            return (coverage_score_raw - prev_state["prev_score"])[:, :, None], state

    class MonotonicityScorer(LabelScorerIntf):
        """score monotonicity"""

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            return {"att_pos": torch.zeros([batch_size, 1], device=device)}

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            prev_label  # noqa  # unused
            # We assume the label scorer has run before us (make sure by right ordering).
            assert set(att_weights_dec_frame.dims) == {
                batch_dim,
                beam_dim,
                enc_spatial_dim,
            }
            att_pos = rf.matmul(
                att_weights_dec_frame,
                rf.range_over_dim(enc_spatial_dim, dtype=att_weights_dec_frame.dtype),
                reduce=enc_spatial_dim,
                use_mask=False,  # not needed, att weights already 0 outside
            )  # [Batch,Beam]
            att_pos_raw = att_pos.copy_compatible_to_dims_raw((batch_dim, beam_dim))
            delta_raw = prev_state["att_pos"] - att_pos_raw
            threshold = monotonicity_opts.get("threshold", 1.0)
            # Penalize when below threshold. The more it is below (or even negative), the more.
            score_raw = torch.where(
                delta_raw < threshold, delta_raw - threshold, 0.0
            )  # [Batch,Beam]
            return score_raw[:, :, None], {"att_pos": att_pos_raw}

    # Note: insertion order matters here, we want that decoder is scored first.
    res = {"decoder": (LabelScorer(), 1.0)}
    if coverage_scale or always_add_scorers:
        res["attention_coverage"] = (
            CoverageScorer(coverage_opts or {}),
            coverage_scale,
        )
    if neg_coverage_scale or (neg_coverage_opts and always_add_scorers):
        # Idea: Too much attention on some frames (e.g. repetitions) is scored negatively.
        res["attention_neg_coverage"] = (
            CoverageScorer(neg_coverage_opts or {}),
            -neg_coverage_scale,
        )
    if monotonicity_scale or always_add_scorers:
        res["attention_monotonicity"] = (MonotonicityScorer(), monotonicity_scale)
    return res