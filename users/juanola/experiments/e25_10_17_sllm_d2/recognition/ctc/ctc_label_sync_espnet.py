"""
Copied from i6_experiments/users/zeyer/experiments/exp2024_04_23_baselines/recog_ext/ctc_label_sync_espnet.py and
slightly adapted to work with
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Tuple, Dict
from functools import partial

import tree
import torch

from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated
from i6_experiments.users.zeyer.nn_rf.top_k_and_random_choice_without_replacement import (
    top_k_and_random_choice_without_replacement,
)

from .ctc_prefix_scorer import CtcPrefixScorer
from ..beam_search import _gather_backrefs
from ...networks.interfaces.base_encoder_decoder_model import BaseEncoderDecoderModel
from ...networks.sllm_with_ext_modules import SllmV4

if TYPE_CHECKING:
    import torch


def ctc_label_sync_search_v1(
    *,
    model: BaseEncoderDecoderModel,
    data: torch.Tensor,
    data_seq_lens: torch.Tensor,
    beam_size: int,
    ctc_soft_collapse_threshold: Optional[float] = None,
    ctc_top_k_pruning: Optional[int] = None,
    ctc_top_k_pruning_reduce_func: str = "mean",
    ctc_scale: float = 1.0,  # TODO: unused
    prior_scale: float = 1.0,  # TODO: unused
    lm_scale: float = 1.0,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Code copied and adapted from
    :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc.model_recog`.

    Function is run within RETURNN.

    Note, some potential further improvements:
    There are many align label seqs which correspond to the same label seq,
    but the LM score is calculated for each of them.
    We could make this somehow unique depending on the label seq.
    (But unclear how exactly to do this in a GPU friendly, batched way.)

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    ## CONFIG INITS?
    from returnn.config import get_global_config

    config = get_global_config()
    version = config.int("recog_version", 1)
    assert version == 1, f"invalid recog_version {version}"
    ctc_soft_collapse_reduce_type = config.typed_value("ctc_soft_collapse_reduce_type", "logmeanexp")
    ctc_top_k_with_random_sampling = config.float(
        "ctc_top_k_with_random_sampling", 0.0
    )  # 0 disabled, 1 enabled. but a smooth transition is possible
    ctc_top_k_with_random_sampling_opts: Optional[Dict[str, Any]] = None
    if ctc_top_k_with_random_sampling:
        ctc_top_k_with_random_sampling_opts = {"max_noise_scale": ctc_top_k_with_random_sampling}
    ctc_top_p = config.typed_value("ctc_top_p", None)  # 1.0 picks all (no effect). e.g. use 0.9.
    if ctc_top_p is not None:
        assert ctc_top_k_with_random_sampling_opts is not None
        ctc_top_k_with_random_sampling_opts["top_p"] = ctc_top_p
    if config.typed_value("ctc_top_k_with_random_sampling_opts", None):
        ctc_top_k_with_random_sampling_opts.update(config.typed_value("ctc_top_k_with_random_sampling_opts", None))
    if ctc_top_k_with_random_sampling_opts:
        for k in ["top_p"]:
            v = ctc_top_k_with_random_sampling_opts.get(k, None)
            if v is not None:
                ctc_top_k_with_random_sampling_opts[k] = rf.convert_to_tensor(v, device=data.device)

    # BASE INITS
    ctc_beam_size = beam_size
    neg_inf = float("-inf")

    ## ENCODING
    decoder_state, aux_logits, encoder_lens = model.forward_encoder(
        data,
        data_seq_lens,
        initial_beam_size=1,
    )
    ctc_log_prob = torch.nn.functional.log_softmax(aux_logits[-1], dim=-1) # USING LAST CTC LAYER

    batch_dims: Dim = [batch_dim]
    enc_spatial_dim = Dim(rf.convert_to_tensor(encoder_lens, dims=[batch_dim]), name="enc_spatial_dim")
    target_dim = Dim(model.num_labels, name="target_dim")
    wb_target_dim = Dim(model.num_labels + 1, name="wb_target_dim")  # Using num_labels + 1...

    ## BEAM PREPARATION/INITIALIZATION
    if ctc_top_k_pruning is not None:
        reduce_func = getattr(torch, ctc_top_k_pruning_reduce_func)
        reduced_log_probs = reduce_func(ctc_log_prob[:, :, :-1], dim=1)
        if ctc_top_k_pruning_reduce_func in ("max", "min"):
            reduced_log_probs = reduced_log_probs[0]
        # get top k log probs for non-blank labels over reduced time frames
        _, pruned_indices = torch.topk(reduced_log_probs, k=ctc_top_k_pruning, dim=-1)
        # add EOS and blank to pruned indices
        pruned_indices = torch.cat(
            [
                pruned_indices,
                # EOS is needed for CTC prefix scoring
                torch.full((pruned_indices.size(0), 1), model.eos_idx, device=pruned_indices.device),
            ],
            dim=-1,
        )
        pruned_indices_wb = torch.cat(
            [
                pruned_indices,
                # EOS is needed for CTC prefix scoring
                torch.full((pruned_indices.size(0), 1), model.blank_idx, device=pruned_indices.device),
            ],
            dim=-1,
        )
        # gather selected log probs and re-normalize
        ctc_log_prob = torch.gather(
            ctc_log_prob, dim=-1, index=pruned_indices_wb.unsqueeze(1).expand(-1, ctc_log_prob.size(1), -1)
        )
        ctc_log_prob = torch.nn.functional.log_softmax(ctc_log_prob, dim=-1)
        pruned_eos_idx = pruned_indices.size(1) - 1  # last non-blank index
        pruned_bos_idx = pruned_eos_idx
        pruned_blank_idx = pruned_indices_wb.size(1) - 1  # last with blank idx
        pruned_wb_target_dim = Dim(pruned_indices_wb.size(1), name="pruned_wb_target_dim")
        wb_target_dim = pruned_wb_target_dim
        pruned_target_dim = Dim(pruned_indices.size(1), name="pruned_target_dim")
        pruned_indices_rf = rf.convert_to_tensor(
            pruned_indices, dims=[batch_dim, pruned_target_dim], sparse_dim=target_dim
        )
        ctc_log_prob = rf.convert_to_tensor(ctc_log_prob, dims=[batch_dim, enc_spatial_dim, pruned_wb_target_dim])
    else:
        ctc_log_prob = rf.convert_to_tensor(ctc_log_prob, dims=[batch_dim, enc_spatial_dim, wb_target_dim])

    # Eager-mode implementation of beam search.

    # The label log probs include the AM and the (scaled) prior.
    if ctc_soft_collapse_threshold is not None:
        ctc_log_prob, enc_spatial_dim = soft_collapse_repeated(
            ctc_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=wb_target_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )

    ctc_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        ctc_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=wb_target_dim, label_value=0.0, other_value=neg_inf),
    )

    ctc_beam_dim = Dim(1, name="ctc_initial_beam")
    ctc_prefix_scorer = CtcPrefixScorer(
        log_probs=ctc_log_prob,
        batch_dims=batch_dims,
        enc_spatial_dim=enc_spatial_dim,
        vocab_wb_dim=wb_target_dim,
        vocab_dim=target_dim if ctc_top_k_pruning is None else pruned_target_dim,
        blank_idx=model.blank_idx if ctc_top_k_pruning is None else pruned_blank_idx,
        eos_idx=model.eos_idx if ctc_top_k_pruning is None else pruned_eos_idx,
    )
    ctc_prefix_scorer_state = None
    ctc_seq_log_prob = rf.constant(0.0, dims=[ctc_beam_dim] + batch_dims)  # Batch, InBeam
    # differentiate between LM and CTC targets in case of pruning
    # in case of pruning, the lm target behaves as before but the ctc prefix scorer always gets the pruned indices
    target_lm = rf.constant(
        model.bos_idx,
        dims=[ctc_beam_dim] + batch_dims,
        sparse_dim=target_dim
    )  # Batch, InBeam -> Vocab
    target_ctc = rf.constant(
        model.bos_idx if ctc_top_k_pruning is None else pruned_bos_idx,
        dims=[ctc_beam_dim] + batch_dims,
        sparse_dim=target_dim if ctc_top_k_pruning is None else pruned_target_dim,
    )  # Batch, InBeam -> Vocab
    ended = rf.constant(False, dims=[ctc_beam_dim] + batch_dims)
    out_seq_len = rf.constant(0, dims=[ctc_beam_dim] + batch_dims)

    labelwise_prior: Optional[rf.Parameter] = getattr(model, "labelwise_prior", None)

    max_seq_len = enc_spatial_dim.get_size_tensor(device=data.device)

    lm_state_raw = decoder_state

    # BEAM SEARCH LOOP / STEP
    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:  # TODO: step could be extracted (probably would need a better beam struct
        label_log_prob, ctc_prefix_scorer_state = ctc_prefix_scorer.score_and_update_state(
            prev_state=ctc_prefix_scorer_state, prev_label=target_ctc, beam_dim=ctc_beam_dim
        )
        if ctc_top_k_pruning is not None:
            # scatter pruned log probs back to original vocab size with -inf for non-selected
            label_log_prob = rf.scatter(
                label_log_prob,
                fill_value=neg_inf,
                indices=pruned_indices_rf,
                indices_dim=pruned_target_dim,
                out_dim=target_dim,
                mode="max",
            )

        if lm_scale > 0:  # Added to avoid calling decoder if using the forward pass as a CTC greedy recognition
            targets_lm_raw = target_lm.copy_compatible_to_dims_raw(batch_dims + [ctc_beam_dim])

            lm_logits_raw, lm_state_raw = model.step_decoder(targets_lm_raw.unsqueeze(-1), lm_state_raw)
            lm_logits_raw = lm_logits_raw.squeeze(-2)  # squeeze singleton time dim
            lm_logits = rf.convert_to_tensor(lm_logits_raw, dims=batch_dims + [ctc_beam_dim, target_dim])
            if ctc_top_k_pruning is not None:
                # gather selected lm logits
                lm_logits = rf.gather(
                    lm_logits,
                    indices=pruned_indices_rf,
                    axis=target_dim,
                )
                # scatter back to original vocab size with -inf for non-selected
                lm_logits = rf.scatter(
                    lm_logits,
                    fill_value=neg_inf,
                    indices=pruned_indices_rf,
                    indices_dim=pruned_target_dim,
                    out_dim=target_dim,
                    mode="max",
                )
            lm_log_probs = rf.log_softmax(lm_logits, axis=target_dim)  # Batch, InBeam, Vocab
            lm_log_probs *= lm_scale
            label_log_prob += lm_log_probs  # Batch, InBeam, Vocab

        if labelwise_prior is not None:
            label_log_prob -= labelwise_prior  # prior scale already applied

        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=target_dim, label_value=0.0, other_value=neg_inf),
            label_log_prob,
        )
        ctc_seq_log_prob = ctc_seq_log_prob + label_log_prob  # Batch, InBeam, Vocab

        if ctc_top_k_with_random_sampling:
            assert ctc_top_k_pruning is None, "not implemented for pruning case"
            ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = top_k_and_random_choice_without_replacement(
                ctc_seq_log_prob,
                axis=[ctc_beam_dim, model.num_labels],
                k=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                **ctc_top_k_with_random_sampling_opts,
            )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
        else:
            k_dim = None
            if ctc_top_k_pruning is not None:
                _, (_, target_ctc), ctc_beam_dim_ = rf.top_k(
                    rf.gather(
                        ctc_seq_log_prob,
                        indices=pruned_indices_rf,
                        axis=target_dim,
                    ),
                    k_dim=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                    axis=[ctc_beam_dim, pruned_target_dim],
                )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
                k_dim = ctc_beam_dim_
            ctc_seq_log_prob, (backrefs, target_lm), ctc_beam_dim = rf.top_k(
                ctc_seq_log_prob,
                k_dim=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam") if k_dim is None else k_dim,
                axis=[ctc_beam_dim, target_dim],
            )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
            if ctc_top_k_pruning is None:
                target_ctc = target_lm

        target_lm = rf.cast(target_lm, dtype=rf.get_default_int_dtype())
        target_ctc = rf.cast(target_ctc, dtype=rf.get_default_int_dtype())
        seq_targets.append(target_lm)
        seq_backrefs.append(backrefs)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        ctc_prefix_scorer_state = rf.nested.gather_nested(ctc_prefix_scorer_state, indices=backrefs)

        lm_state_raw = tree.map_structure(
            partial(_gather_backrefs, backrefs=backrefs.raw_tensor, beam_size=beam_size), lm_state_raw
        )

        i += 1
        ended = rf.logical_or(ended, target_lm == model.eos_idx)
        ended = rf.logical_or(ended, i >= max_seq_len)
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

    # BACKTRACK SEQUENCES
    ctc_seq_targets, labels_spatial_dim = backtrack_sequences(out_seq_len, seq_backrefs, seq_targets, ctc_beam_dim)

    return ctc_seq_targets, ctc_seq_log_prob, labels_spatial_dim, ctc_beam_dim


def ctc_label_sync_search_v2( #TODO: in progress!!
        *,
        model: SllmV4,
        data: torch.Tensor,

        data_seq_lens: torch.Tensor,
        beam_size: int,
        ctc_soft_collapse_threshold: Optional[float] = None,
        ctc_top_k_pruning: Optional[int] = None,
        ctc_top_k_pruning_reduce_func: str = "mean",

        ctc_scale: float = 1.0,
        prior_scale: float = 0.0,  # TODO: unused
        sllm_scale: float = 0.0,
        external_lm_scale: float = 0.0,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Adds external LM
    """
    assert ctc_scale > 0.0 or sllm_scale > 0.0 or external_lm_scale > 0.0, "Either CTC, SLLM or LM scale needs to be positive"

    ## CONFIG INITS?
    from returnn.config import get_global_config

    config = get_global_config()
    version = config.int("recog_version", 1)
    assert version == 1, f"invalid recog_version {version}"
    ctc_soft_collapse_reduce_type = config.typed_value("ctc_soft_collapse_reduce_type", "logmeanexp")
    ctc_top_k_with_random_sampling = config.float(
        "ctc_top_k_with_random_sampling", 0.0
    )  # 0 disabled, 1 enabled. but a smooth transition is possible
    ctc_top_k_with_random_sampling_opts: Optional[Dict[str, Any]] = None
    if ctc_top_k_with_random_sampling:
        ctc_top_k_with_random_sampling_opts = {"max_noise_scale": ctc_top_k_with_random_sampling}
    ctc_top_p = config.typed_value("ctc_top_p", None)  # 1.0 picks all (no effect). e.g. use 0.9.
    if ctc_top_p is not None:
        assert ctc_top_k_with_random_sampling_opts is not None
        ctc_top_k_with_random_sampling_opts["top_p"] = ctc_top_p
    if config.typed_value("ctc_top_k_with_random_sampling_opts", None):
        ctc_top_k_with_random_sampling_opts.update(config.typed_value("ctc_top_k_with_random_sampling_opts", None))
    if ctc_top_k_with_random_sampling_opts:
        for k in ["top_p"]:
            v = ctc_top_k_with_random_sampling_opts.get(k, None)
            if v is not None:
                ctc_top_k_with_random_sampling_opts[k] = rf.convert_to_tensor(v, device=data.device)

    # BASE INITS
    ctc_beam_size = beam_size
    neg_inf = float("-inf")

    ## CTC LOGITS

    ### SLLM CALL (always necessary)
    decoder_state, aux_logits, encoder_lens = model.forward_encoder(
        data,
        data_seq_lens,
        initial_beam_size=1,
    )

    if model.has_external_ctc():
        ext_ctc_decoder_state, ext_ctc_aux_logits, ext_ctc_encoder_lens = model.external_ctc_forward_encoder(
            data,
            data_seq_lens,
            initial_beam_size=1,
        )
        ctc_log_prob = torch.nn.functional.log_softmax(ext_ctc_aux_logits[-1], dim=-1) # USING EXT CTC LOGITS
    else:
        ctc_log_prob = torch.nn.functional.log_softmax(aux_logits[-1], dim=-1) # USING LAST SLLM CTC LAYER

    batch_dims: Dim = [batch_dim]
    enc_spatial_dim = Dim(rf.convert_to_tensor(encoder_lens, dims=[batch_dim]), name="enc_spatial_dim") #TODO: this should be changed if ext CTC??
    target_dim = Dim(model.num_labels, name="target_dim")
    wb_target_dim = Dim(model.num_labels + 1, name="wb_target_dim")  # Using num_labels + 1...

    ## BEAM PREPARATION/INITIALIZATION
    if ctc_top_k_pruning is not None:
        reduce_func = getattr(torch, ctc_top_k_pruning_reduce_func)
        reduced_log_probs = reduce_func(ctc_log_prob[:, :, :-1], dim=1)
        if ctc_top_k_pruning_reduce_func in ("max", "min"):
            reduced_log_probs = reduced_log_probs[0]
        # get top k log probs for non-blank labels over reduced time frames
        _, pruned_indices = torch.topk(reduced_log_probs, k=ctc_top_k_pruning, dim=-1)
        # add EOS and blank to pruned indices
        pruned_indices = torch.cat(
            [
                pruned_indices,
                # EOS is needed for CTC prefix scoring
                torch.full((pruned_indices.size(0), 1), model.eos_idx, device=pruned_indices.device),
            ],
            dim=-1,
        )
        pruned_indices_wb = torch.cat(
            [
                pruned_indices,
                # EOS is needed for CTC prefix scoring
                torch.full((pruned_indices.size(0), 1), model.blank_idx, device=pruned_indices.device),
            ],
            dim=-1,
        )
        # gather selected log probs and re-normalize
        ctc_log_prob = torch.gather(
            ctc_log_prob, dim=-1, index=pruned_indices_wb.unsqueeze(1).expand(-1, ctc_log_prob.size(1), -1)
        )
        ctc_log_prob = torch.nn.functional.log_softmax(ctc_log_prob, dim=-1)
        pruned_eos_idx = pruned_indices.size(1) - 1  # last non-blank index
        pruned_bos_idx = pruned_eos_idx
        pruned_blank_idx = pruned_indices_wb.size(1) - 1  # last with blank idx
        pruned_wb_target_dim = Dim(pruned_indices_wb.size(1), name="pruned_wb_target_dim")
        wb_target_dim = pruned_wb_target_dim
        pruned_target_dim = Dim(pruned_indices.size(1), name="pruned_target_dim")
        pruned_indices_rf = rf.convert_to_tensor(
            pruned_indices, dims=[batch_dim, pruned_target_dim], sparse_dim=target_dim
        )
        ctc_log_prob = rf.convert_to_tensor(ctc_log_prob, dims=[batch_dim, enc_spatial_dim, pruned_wb_target_dim])
    else:
        ctc_log_prob = rf.convert_to_tensor(ctc_log_prob, dims=[batch_dim, enc_spatial_dim, wb_target_dim])

    # Eager-mode implementation of beam search.

    # The label log probs include the AM and the (scaled) prior.
    if ctc_soft_collapse_threshold is not None:
        ctc_log_prob, enc_spatial_dim = soft_collapse_repeated(
            ctc_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=wb_target_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )

    ctc_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        ctc_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=wb_target_dim, label_value=0.0, other_value=neg_inf),
    )

    ctc_beam_dim = Dim(1, name="ctc_initial_beam")
    ctc_prefix_scorer = CtcPrefixScorer(
        log_probs=ctc_log_prob,
        batch_dims=batch_dims,
        enc_spatial_dim=enc_spatial_dim,
        vocab_wb_dim=wb_target_dim,
        vocab_dim=target_dim if ctc_top_k_pruning is None else pruned_target_dim,
        blank_idx=model.blank_idx if ctc_top_k_pruning is None else pruned_blank_idx,
        eos_idx=model.eos_idx if ctc_top_k_pruning is None else pruned_eos_idx,
    )
    ctc_prefix_scorer_state = None
    ctc_seq_log_prob = rf.constant(0.0, dims=[ctc_beam_dim] + batch_dims)  # Batch, InBeam
    # differentiate between LM and CTC targets in case of pruning
    # in case of pruning, the lm target behaves as before but the ctc prefix scorer always gets the pruned indices
    target_lm = rf.constant(
        model.bos_idx,
        dims=[ctc_beam_dim] + batch_dims,
        sparse_dim=target_dim
    )  # Batch, InBeam -> Vocab
    target_ctc = rf.constant(
        model.bos_idx if ctc_top_k_pruning is None else pruned_bos_idx,
        dims=[ctc_beam_dim] + batch_dims,
        sparse_dim=target_dim if ctc_top_k_pruning is None else pruned_target_dim,
    )  # Batch, InBeam -> Vocab
    ended = rf.constant(False, dims=[ctc_beam_dim] + batch_dims)
    out_seq_len = rf.constant(0, dims=[ctc_beam_dim] + batch_dims)

    labelwise_prior: Optional[rf.Parameter] = getattr(model, "labelwise_prior", None)

    max_seq_len = enc_spatial_dim.get_size_tensor(device=data.device)

    lm_state_raw = decoder_state
    ext_lm_state = model.get_empty_qwen_input_embeds(decoder_state["input_embeds"], initial_beam_size=1)

    # BEAM SEARCH LOOP / STEP
    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:  # TODO: step could be extracted (probably would need a better beam struct

        # --- CTC SCORING (Calculated first as the anchor) ---
        label_log_prob, ctc_prefix_scorer_state = ctc_prefix_scorer.score_and_update_state(
            prev_state=ctc_prefix_scorer_state, prev_label=target_ctc, beam_dim=ctc_beam_dim
        )
        label_log_prob = label_log_prob * ctc_scale
        if ctc_top_k_pruning is not None:
            # scatter pruned log probs back to original vocab size with -inf for non-selected
            label_log_prob = rf.scatter(
                label_log_prob,
                fill_value=neg_inf,
                indices=pruned_indices_rf,
                indices_dim=pruned_target_dim,
                out_dim=target_dim,
                mode="max",
            )

        targets_lm_raw = target_lm.copy_compatible_to_dims_raw(batch_dims + [ctc_beam_dim])

        # --- SLLM DECODER SCORING ---
        if sllm_scale > 0:  # Added to avoid calling decoder if using the forward pass as a CTC greedy recognition
            lm_logits_raw, lm_state_raw = model.step_decoder(targets_lm_raw.unsqueeze(-1), lm_state_raw)
            lm_logits_raw = lm_logits_raw.squeeze(-2)  # squeeze singleton time dim
            lm_logits = rf.convert_to_tensor(lm_logits_raw, dims=batch_dims + [ctc_beam_dim, target_dim])
            if ctc_top_k_pruning is not None:
                # gather selected lm logits
                lm_logits = rf.gather(
                    lm_logits,
                    indices=pruned_indices_rf,
                    axis=target_dim,
                )
                # scatter back to original vocab size with -inf for non-selected
                lm_logits = rf.scatter(
                    lm_logits,
                    fill_value=neg_inf,
                    indices=pruned_indices_rf,
                    indices_dim=pruned_target_dim,
                    out_dim=target_dim,
                    mode="max",
                )
            lm_log_probs = rf.log_softmax(lm_logits, axis=target_dim)  # Batch, InBeam, Vocab
            lm_log_probs *= sllm_scale

            # MERGE POINT 1: Add SLLM to the accumulation
            label_log_prob += lm_log_probs  # Batch, InBeam, Vocab

        # --- EXTERNAL LM ---
        if external_lm_scale > 0:
            # Note: External LMs usually don't need encoder context
            # Assuming external_lm has a similar step interface
            ext_logits_raw, ext_lm_state = model.external_llm_step_decoder(targets_lm_raw.unsqueeze(-1), ext_lm_state)
            ext_logits_raw = ext_logits_raw.squeeze(-2)  # squeeze singleton time dim
            ext_logits = rf.convert_to_tensor(ext_logits_raw, dims=batch_dims + [ctc_beam_dim, target_dim])
            if ctc_top_k_pruning is not None:
                # gather selected lm logits
                ext_logits = rf.gather(
                    ext_logits,
                    indices=pruned_indices_rf,
                    axis=target_dim,
                )
                # scatter back to original vocab size with -inf for non-selected
                ext_logits = rf.scatter(
                    ext_logits,
                    fill_value=neg_inf,
                    indices=pruned_indices_rf,
                    indices_dim=pruned_target_dim,
                    out_dim=target_dim,
                    mode="max",
                )
            # Re-normalize external LM
            ext_log_probs = rf.log_softmax(ext_logits, axis=target_dim) # Batch, InBeam, Vocab

            # MERGE POINT 2: Add External LM to the accumulation
            label_log_prob += external_lm_scale * ext_log_probs

        # --- PRIOR SCORING ??? ---
        if labelwise_prior is not None:
            label_log_prob -= labelwise_prior  # prior scale already applied

        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=target_dim, label_value=0.0, other_value=neg_inf),
            label_log_prob,
        )
        ctc_seq_log_prob = ctc_seq_log_prob + label_log_prob  # Batch, InBeam, Vocab

        if ctc_top_k_with_random_sampling:
            assert ctc_top_k_pruning is None, "not implemented for pruning case"
            ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = top_k_and_random_choice_without_replacement(
                ctc_seq_log_prob,
                axis=[ctc_beam_dim, model.num_labels],
                k=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                **ctc_top_k_with_random_sampling_opts,
            )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
        else:
            k_dim = None
            if ctc_top_k_pruning is not None:
                _, (_, target_ctc), ctc_beam_dim_ = rf.top_k(
                    rf.gather(
                        ctc_seq_log_prob,
                        indices=pruned_indices_rf,
                        axis=target_dim,
                    ),
                    k_dim=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                    axis=[ctc_beam_dim, pruned_target_dim],
                )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
                k_dim = ctc_beam_dim_
            ctc_seq_log_prob, (backrefs, target_lm), ctc_beam_dim = rf.top_k(
                ctc_seq_log_prob,
                k_dim=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam") if k_dim is None else k_dim,
                axis=[ctc_beam_dim, target_dim],
            )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
            if ctc_top_k_pruning is None:
                target_ctc = target_lm

        target_lm = rf.cast(target_lm, dtype=rf.get_default_int_dtype())
        target_ctc = rf.cast(target_ctc, dtype=rf.get_default_int_dtype())
        seq_targets.append(target_lm)
        seq_backrefs.append(backrefs)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        ctc_prefix_scorer_state = rf.nested.gather_nested(ctc_prefix_scorer_state, indices=backrefs)

        if sllm_scale > 0:
            lm_state_raw = tree.map_structure(
                partial(_gather_backrefs, backrefs=backrefs.raw_tensor, beam_size=beam_size), lm_state_raw
            )
        if external_lm_scale > 0:
            ext_lm_state = tree.map_structure(
                partial(_gather_backrefs, backrefs=backrefs.raw_tensor, beam_size=beam_size), ext_lm_state
            )

        i += 1
        ended = rf.logical_or(ended, target_lm == model.eos_idx)
        ended = rf.logical_or(ended, i >= max_seq_len)
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

    # BACKTRACK SEQUENCES
    ctc_seq_targets, labels_spatial_dim = backtrack_sequences(out_seq_len, seq_backrefs, seq_targets, ctc_beam_dim)

    return ctc_seq_targets, ctc_seq_log_prob, labels_spatial_dim, ctc_beam_dim



def backtrack_sequences(out_seq_len, seq_backrefs: list[Any], seq_targets: list[Any], ctc_beam_dim) -> tuple[Any, Any]:
    """
    Backtrack via backrefs, resolve beams.
    :param ctc_beam_dim:
    :param out_seq_len:
    :param seq_backrefs:
    :param seq_targets:
    :return:
    """
    seq_targets_ = []
    indices = rf.range_over_dim(ctc_beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    labels_spatial_dim = Dim(out_seq_len, name="ctc_labels_spatial")
    ctc_seq_targets = seq_targets__.stack(axis=labels_spatial_dim)

    # Remove the remaining EOS labels.
    ctc_seq_targets, _ = rf.slice(ctc_seq_targets, axis=labels_spatial_dim, size=labels_spatial_dim)
    return ctc_seq_targets, labels_spatial_dim
