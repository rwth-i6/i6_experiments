from typing import Optional, Dict, Any, Tuple, Sequence
import tree

from returnn.tensor import Tensor, Dim, single_step_dim
from returnn.frontend.state import State
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.recog import (
    RecogDef,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import (
    _batch_size_factor,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import (
    SegmentalAttentionModel as Model,
)
# from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import Model
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import (
    recombination,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import (
    utils,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.beam_search import (
    utils as beam_search_utils,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
    BlankDecoderV1,
    BlankDecoderV3,
    BlankDecoderV4,
    BlankDecoderV5,
    BlankDecoderV6,
    BlankDecoderV7,
)


def update_state(
    model,
    update_state_mask: Tensor,
    backrefs: Tensor,
    label_decoder_state: State,
    label_decoder_state_updated: State,
    lm_state: Optional[State],
    lm_state_updated: Optional[State],
    ilm_state: Optional[State],
    ilm_state_updated: Optional[State],
) -> Tuple[State, Optional[State], Optional[State], Optional[State]]:

    # ------------------- update label decoder state and ILM state -------------------

    def _get_masked_state(old, new, mask):
        old = rf.gather(old, indices=backrefs)
        new = rf.gather(new, indices=backrefs)
        return rf.where(mask, new, old)

    # label decoder
    label_decoder_state = tree.map_structure(
        lambda s: rf.gather(s, indices=backrefs), label_decoder_state_updated
    )

    # ILM
    if ilm_state is not None:
        ilm_state = tree.map_structure(
            lambda old_state, new_state: _get_masked_state(
                old_state, new_state, update_state_mask
            ),
            ilm_state,
            ilm_state_updated,
        )

    # ------------------- update external LM state -------------------

    if lm_state is not None:
        for state in lm_state:
            if state == "pos":
                lm_state[state] = rf.where(
                    update_state_mask,
                    rf.gather(lm_state_updated[state], indices=backrefs),
                    rf.gather(lm_state[state], indices=backrefs),
                )
            else:
                updated_accum_axis = lm_state_updated[state].self_att.accum_axis

                updated_self_att_expand_dim_dyn_size_ext = rf.gather(
                    updated_accum_axis.dyn_size_ext, indices=backrefs
                )
                masked_self_att_expand_dim_dyn_size_ext = rf.where(
                    update_state_mask,
                    updated_self_att_expand_dim_dyn_size_ext,
                    updated_self_att_expand_dim_dyn_size_ext - 1,
                )
                masked_self_att_expand_dim = Dim(
                    masked_self_att_expand_dim_dyn_size_ext,
                    name="self_att_expand_dim_init",
                )
                lm_state[state].self_att.accum_axis = masked_self_att_expand_dim

                def _mask_lm_state(tensor: rf.Tensor):
                    tensor = rf.gather(tensor, indices=backrefs)
                    tensor = tensor.copy_transpose(
                        [updated_accum_axis] + tensor.remaining_dims(updated_accum_axis)
                    )
                    tensor_raw = tensor.raw_tensor
                    tensor_raw = tensor_raw[
                        : rf.reduce_max(
                            masked_self_att_expand_dim_dyn_size_ext,
                            axis=masked_self_att_expand_dim_dyn_size_ext.dims,
                        ).raw_tensor.item()
                    ]
                    tensor = tensor.copy_template_replace_dim_tag(
                        tensor.get_axis_from_description(updated_accum_axis),
                        masked_self_att_expand_dim,
                    )
                    tensor.raw_tensor = tensor_raw
                    return tensor

                lm_state[state].self_att.k_accum = _mask_lm_state(
                    lm_state_updated[state].self_att.k_accum
                )
                lm_state[state].self_att.v_accum = _mask_lm_state(
                    lm_state_updated[state].self_att.v_accum
                )

    return label_decoder_state, lm_state, ilm_state


def get_score(
    model,
    i: int,
    input_embed_label_model: Tensor,
    input_embed_blank_model: Optional[Tensor],
    nb_target: Tensor,
    emit_positions: Tensor,
    label_decoder_state: State,
    lm_state: Optional[State],
    ilm_state: Optional[State],
    enc_args: Dict[str, Tensor],
    enc_spatial_dim: Dim,
    beam_dim: Dim,
    batch_dims: Sequence[Dim],
    external_lm_scale: Optional[float] = None,
    ilm_correction_scale: Optional[float] = None,
    subtract_ilm_eos_score: bool = False,
) -> Tuple[Tensor, State, Optional[State], Optional[State], Optional[State]]:
    # ------------------- label step -------------------

    label_step_out, label_decoder_state = model.loop_step(
        **enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=input_embed_label_model,
        state=label_decoder_state,
    )
    label_step_out.pop("att_weights", None)

    label_logits = model.decode_logits(
        input_embed=input_embed_label_model, **label_step_out
    )
    label_log_prob = rf.log_softmax(label_logits, axis=model.target_dim)

    # ------------------- external LM step -------------------

    lm_eos_log_prob = rf.zeros(batch_dims, dtype="float32")
    if lm_state is not None:
        lm_logits, lm_state = model.language_model(
            nb_target,
            spatial_dim=single_step_dim,
            state=lm_state,
        )
        lm_label_log_prob = rf.log_softmax(lm_logits, axis=model.target_dim)

        # do not apply LM scores to blank
        lm_label_log_prob_ = rf.where(
            rf.range_over_dim(model.target_dim) == model.blank_idx,
            rf.zeros(batch_dims, dtype="float32"),
            lm_label_log_prob,
        )
        lm_label_log_prob = rf.where(
            rf.convert_to_tensor(
                i
                == rf.copy_to_device(
                    enc_spatial_dim.get_size_tensor(),
                    input_embed_label_model.device,
                )
                - 1
            ),
            lm_label_log_prob,
            lm_label_log_prob_,
        )

        label_log_prob += external_lm_scale * lm_label_log_prob

    # --------------------------------- ILM step ---------------------------------

    ilm_eos_log_prob = rf.zeros(batch_dims, dtype="float32")
    if ilm_state is not None:
        ilm_step_out, ilm_state = model.ilm(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed_label_model,
            state=ilm_state,
            use_mini_att=True,
        )
        ilm_label_log_prob = rf.log_softmax(ilm_step_out, axis=model.target_dim)

        # do not apply ILM correction to blank
    ilm_label_log_prob_ = rf.where(
        rf.range_over_dim(model.target_dim) == model.blank_idx,
        rf.zeros(batch_dims, dtype="float32"),
        ilm_label_log_prob,
    )
    if subtract_ilm_eos_score:
        ilm_label_log_prob = rf.where(
            rf.convert_to_tensor(
                i
                == rf.copy_to_device(
                    enc_spatial_dim.get_size_tensor(), input_embed_label_model.device
                )
                - 1
            ),
            ilm_label_log_prob,
            ilm_label_log_prob_,
        )
    else:
        ilm_label_log_prob = ilm_label_log_prob_.copy()

    label_log_prob -= ilm_correction_scale * ilm_label_log_prob

    output_log_prob = label_log_prob

    return (
        output_log_prob,
        label_decoder_state,
        lm_state,
        ilm_state,
    )


def model_recog(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
    search_args: Optional[Dict[str, Any]] = None,
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

    if hasattr(model, "search_args"):
        search_args = model.search_args

    beam_size = search_args.get("beam_size", 12)
    use_recombination = search_args.get("use_recombination", "sum")
    external_lm_scale = search_args.get("lm_scale", 0.0)
    ilm_correction_scale = search_args.get("ilm_scale", 0.0)
    subtract_ilm_eos_score = search_args.get("subtract_ilm_eos_score", False)

    # --------------------------------- init encoder, dims, etc ---------------------------------

    enc_args, enc_spatial_dim = model.encode(
        data, in_spatial_dim=data_spatial_dim
    )
    enc_args.pop("ctc", None)

    max_seq_len = enc_spatial_dim.get_size_tensor()
    max_seq_len = rf.reduce_max(max_seq_len, axis=max_seq_len.dims)

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    backrefs = rf.zeros(batch_dims_, dtype="int32")

    bos_idx = 0

    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    # for blank decoder v7
    emit_positions = rf.full(dims=batch_dims_, fill_value=-1, dtype="int32")

    if use_recombination:
        assert len(batch_dims) == 1
        assert use_recombination in {"sum", "max"}
        seq_hash = rf.constant(0, dims=batch_dims_, dtype="int64")
    else:
        seq_hash = None

    # lists of [B, beam] tensors
    seq_targets = []
    seq_backrefs = []

    update_state_mask = rf.constant(True, dims=batch_dims_)

    output_dim = model.target_dim

    # --------------------------------- init states ---------------------------------

    # label decoder
    label_decoder_state = model.decoder_default_initial_state(
        batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim
    )


    blank_decoder_state = None

    # external LM
    if model.language_model:
        lm_state = model.language_model.default_initial_state(batch_dims=batch_dims_)
        for state in lm_state:
            if state == "pos":
                lm_state[state] = rf.zeros(batch_dims_, dtype="int32")
            else:
                self_att_expand_dim = Dim(
                    rf.zeros(batch_dims_, dtype="int32"),
                    name="self_att_expand_dim_init",
                )
                lm_state[state].self_att.accum_axis = self_att_expand_dim

                k_accum = lm_state[state].self_att.k_accum  # type: rf.Tensor
                k_accum_raw = k_accum.raw_tensor
                lm_state[
                    state
                ].self_att.k_accum = k_accum.copy_template_replace_dim_tag(
                    k_accum.get_axis_from_description("stag:self_att_expand_dim_init"),
                    self_att_expand_dim,
                )
                lm_state[state].self_att.k_accum.raw_tensor = k_accum_raw

                v_accum = lm_state[state].self_att.v_accum  # type: rf.Tensor
                v_accum_raw = v_accum.raw_tensor
                lm_state[
                    state
                ].self_att.v_accum = v_accum.copy_template_replace_dim_tag(
                    v_accum.get_axis_from_description("stag:self_att_expand_dim_init"),
                    self_att_expand_dim,
                )
                lm_state[state].self_att.v_accum.raw_tensor = v_accum_raw
    else:
        lm_state = None

    # ILM
    if ilm_correction_scale > 0.0:
        ilm_state = model.ilm.default_initial_state(
            batch_dims=batch_dims_
        )
    else:
        ilm_state = None

    # --------------------------------- init targets, embeddings ---------------------------------

    target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    target_non_blank = target.copy()


    input_embed = rf.zeros(
        batch_dims_ + [model.target_embed.out_dim],
        feature_dim=model.target_embed.out_dim,
        dtype="float32",
    )

    input_embed_length_model = None

    # --------------------------------- main loop ---------------------------------

    i = 0
    while i < max_seq_len.raw_tensor:
        if i > 0:
            target_non_blank = rf.where(
                update_state_mask, target, rf.gather(target_non_blank, indices=backrefs)
            )
            target_non_blank.sparse_dim = model.target_embed.in_dim

            input_embed = model.target_embed(target)

            emit_positions = rf.where(
                update_state_mask,
                rf.full(dims=batch_dims, fill_value=i - 1, dtype="int32"),
                rf.gather(emit_positions, indices=backrefs),
            )

        (
            output_log_prob,
            label_decoder_state_updated,
            lm_state_updated,
            ilm_state_updated,
        ) = get_score(
            model=model,
            i=i,
            input_embed_label_model=input_embed,
            input_embed_blank_model=input_embed_length_model,
            nb_target=target_non_blank,
            emit_positions=emit_positions,
            label_decoder_state=label_decoder_state,
            lm_state=lm_state,
            ilm_state=ilm_state,
            enc_args=enc_args,
            enc_spatial_dim=enc_spatial_dim,
            beam_dim=beam_dim,
            batch_dims=batch_dims,
            external_lm_scale=external_lm_scale,
            ilm_correction_scale=ilm_correction_scale,
            subtract_ilm_eos_score=subtract_ilm_eos_score,
        )

        # for shorter seqs in the batch, set the blank score to zero and the others to ~-inf
        output_log_prob = rf.where(
            rf.convert_to_tensor(
                i >= rf.copy_to_device(enc_spatial_dim.get_size_tensor(), data.device)
            ),
            rf.sparse_to_dense(
                model.blank_idx, axis=output_dim, label_value=0.0, other_value=-1.0e30
            ),
            output_log_prob,
        )

        # ------------------- recombination -------------------

        if use_recombination:
            seq_log_prob = recombination.recombine_seqs(
                seq_targets,
                seq_log_prob,
                seq_hash,
                beam_dim,
                batch_dims[0],
                use_sum=use_recombination == "sum",
            )

        # ------------------- top-k -------------------

        seq_log_prob = seq_log_prob + output_log_prob  # Batch, InBeam, Vocab
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob,
            k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
            axis=[beam_dim, output_dim],
        )
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

        # ------------------- update hash for recombination -------------------

        if use_recombination:
            seq_hash = recombination.update_seq_hash(
                seq_hash, target, backrefs, model.blank_idx
            )

        # mask for updating label-sync states
        update_state_mask = rf.convert_to_tensor(target != model.blank_idx)

        label_decoder_state, blank_decoder_state, lm_state, ilm_state = update_state(
            model=model,
            update_state_mask=update_state_mask,
            backrefs=backrefs,
            label_decoder_state=label_decoder_state,
            label_decoder_state_updated=label_decoder_state_updated,
            blank_decoder_state=blank_decoder_state,
            lm_state=lm_state,
            lm_state_updated=lm_state_updated,
            ilm_state=ilm_state,
            ilm_state_updated=ilm_state_updated,
        )

        i += 1

    # last recombination
    if use_recombination:
        seq_log_prob = recombination.recombine_seqs(
            seq_targets,
            seq_log_prob,
            seq_hash,
            beam_dim,
            batch_dims[0],
            use_sum=use_recombination == "sum",
        )

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
    seq_targets = seq_targets__.stack(axis=enc_spatial_dim)

    non_blank_targets, non_blank_targets_spatial_dim = utils.get_masked(
        seq_targets,
        utils.get_non_blank_mask(seq_targets, model.blank_idx),
        enc_spatial_dim,
        [beam_dim] + batch_dims,
    )
    non_blank_targets.sparse_dim = model.target_dim

    return non_blank_targets, seq_log_prob, non_blank_targets_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False
