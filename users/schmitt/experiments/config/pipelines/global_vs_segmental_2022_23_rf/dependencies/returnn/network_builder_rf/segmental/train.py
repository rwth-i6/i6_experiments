from typing import Optional, Dict, List
import torch
import os

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import FramewiseTrainDef, FullSumTrainDef

from returnn.tensor import TensorDict
from returnn.datasets.hdf import SimpleHDFWriter
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.augmentation.alignment import shift_alignment_boundaries_batched
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  viterbi_training as label_model_viterbi_training
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  viterbi_training_efficient as label_model_viterbi_training_efficient
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  full_sum_training as label_model_full_sum_training
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.realignment import (
  model_realign_efficient, model_realign
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.train import (
  viterbi_training as blank_model_viterbi_training
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.train import (
  viterbi_training_v3 as blank_model_viterbi_training_v3
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.train import (
  viterbi_training_v4 as blank_model_viterbi_training_v4
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.train import (
  viterbi_training_v5 as blank_model_viterbi_training_v5
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.train import (
  viterbi_training_v6 as blank_model_viterbi_training_v6
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.train import (
  viterbi_training_v7 as blank_model_viterbi_training_v7
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.train import (
  viterbi_training_v8 as blank_model_viterbi_training_v8
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.train import (
  viterbi_training_v9 as blank_model_viterbi_training_v9
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
  BlankDecoderV1,
  BlankDecoderV3,
  BlankDecoderV4,
  BlankDecoderV5,
  BlankDecoderV6,
  BlankDecoderV7,
  BlankDecoderV8,
  BlankDecoderV9,
  BlankDecoderV10,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder,
  GlobalAttEfficientDecoder,
)

from returnn.tensor import Dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef


def _returnn_v2_train_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  targets = extern_data[default_target_key]
  targets_spatial_dim = targets.get_time_dim_tag()
  train_def: FramewiseTrainDef = config.typed_value("_train_def")
  train_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    align_targets=targets,
    align_targets_spatial_dim=targets_spatial_dim,
    seq_tags=extern_data["seq_tag"],
  )


def _returnn_v2_full_sum_train_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  targets = extern_data[default_target_key]
  targets_spatial_dim = targets.get_time_dim_tag()
  interpolation_alignment = extern_data.data.get("interpolation_alignment")
  if interpolation_alignment is not None:
    interpolation_alignment_spatial_dim = interpolation_alignment.get_time_dim_tag()
  else:
    interpolation_alignment_spatial_dim = None
  interpolation_alignment_scores = extern_data.data.get("interpolation_alignment_scores")

  train_def: FullSumTrainDef = config.typed_value("_train_def")
  train_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    non_blank_targets=targets,
    non_blank_targets_spatial_dim=targets_spatial_dim,
    seq_tags=extern_data["seq_tag"],
    interpolation_alignment=interpolation_alignment,
    interpolation_alignment_spatial_dim=interpolation_alignment_spatial_dim,
    interpolation_alignment_scores=interpolation_alignment_scores,
  )


def get_alignment_args(
        model: SegmentalAttentionModel,
        align_targets: rf.Tensor,
        align_targets_spatial_dim: Dim,
        batch_dims: List[Dim],
):
  if model.use_joint_model:
    # TODO: use rf.window() instead
    segment_starts, segment_lens, center_positions = utils.get_segment_starts_and_lens(
      non_blank_mask=rf.sequence_mask(align_targets.dims),  # this way, every frame is interpreted as non-blank
      align_targets_spatial_dim=align_targets_spatial_dim,
      center_window_size=model.center_window_size,
      batch_dims=batch_dims,
      out_spatial_dim=align_targets_spatial_dim
    )
    # set blank indices in alignment to 0 (= EOS index of imported global att model which is not used otherwise)
    align_targets.raw_tensor[align_targets.raw_tensor == model.target_dim.dimension] = 0
    align_targets.sparse_dim = model.target_dim

    non_blank_mask = utils.get_non_blank_mask(align_targets, model.blank_idx)
    non_blank_targets, non_blank_targets_spatial_dim = utils.get_masked(
      align_targets, non_blank_mask, align_targets_spatial_dim, batch_dims
    )
    non_blank_targets.sparse_dim = model.target_dim
  else:
    non_blank_mask = utils.get_non_blank_mask(align_targets, model.blank_idx)
    non_blank_targets, non_blank_targets_spatial_dim = utils.get_masked(
      align_targets, non_blank_mask, align_targets_spatial_dim, batch_dims
    )
    non_blank_targets.sparse_dim = model.target_dim

    segment_starts, segment_lens, center_positions = utils.get_segment_starts_and_lens(
      non_blank_mask,
      align_targets_spatial_dim,
      model.center_window_size,
      batch_dims,
      non_blank_targets_spatial_dim
    )

  return segment_starts, segment_lens, center_positions, non_blank_targets, non_blank_targets_spatial_dim, non_blank_mask


def viterbi_training(
        *,
        model: SegmentalAttentionModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        align_targets: rf.Tensor,
        align_targets_spatial_dim: Dim,
        enc_args: Optional[Dict[str, rf.Tensor]] = None,
        enc_spatial_dim: Optional[Dim] = None,
        batch_dims: Optional[List[Dim]] = None,
        beam_dim: Optional[Dim] = None,
        seq_tags: Optional[rf.Tensor] = None,
):
  if enc_args is not None:
    assert enc_spatial_dim is not None

  from returnn.config import get_global_config

  config = get_global_config()  # noqa
  aux_loss_layers = config.typed_value("aux_loss_layers")
  aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
  aux_loss_focal_loss_factors = config.typed_value("aux_loss_focal_loss_factors", ([0.0] * len(aux_loss_layers)) if aux_loss_layers else None)
  aux_loss_type = config.typed_value("aux_loss_type", "ctc")
  force_inefficient_loop = config.typed_value("force_inefficient_loop", False)

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

  if batch_dims is None:
    batch_dims = data.remaining_dims(data_spatial_dim)

  alignment_augmentation_opts = config.typed_value("alignment_augmentation_opts", None)
  if alignment_augmentation_opts is not None:
    for _ in range(alignment_augmentation_opts["num_iterations"]):
      align_targets = shift_alignment_boundaries_batched(
        alignment=align_targets,
        alignment_spatial_dim=align_targets_spatial_dim,
        batch_dims=batch_dims,
        blank_idx=model.blank_idx,
        max_shift=alignment_augmentation_opts["max_shift"],
      )

  (
    segment_starts, segment_lens, center_positions, non_blank_targets, non_blank_targets_spatial_dim, non_blank_mask
  ) = get_alignment_args(
    model=model,
    align_targets=align_targets,
    align_targets_spatial_dim=align_targets_spatial_dim,
    batch_dims=batch_dims,
  )

  if enc_args is None:
    # ------------------- encoder aux loss -------------------

    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encoder.encode(
      data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)

    if aux_loss_layers:
      for i, layer_idx in enumerate(aux_loss_layers):
        if layer_idx > len(model.encoder.layers):
          continue
        linear = getattr(model.encoder, f"enc_aux_logits_{layer_idx}")
        aux_logits = linear(collected_outputs[str(layer_idx - 1)])

        if aux_loss_type == "ctc":
          aux_loss = rf.ctc_loss(
            logits=aux_logits,
            targets=non_blank_targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=non_blank_targets_spatial_dim,
            blank_index=model.blank_idx,
          )
          loss_name = f"ctc_{layer_idx}"
        else:
          assert aux_loss_type == "ce"

          aux_logits_packed, pack_dim = rf.pack_padded(
            aux_logits, dims=batch_dims + [enc_spatial_dim], enforce_sorted=False)
          align_targets_packed, _ = rf.pack_padded(
            align_targets, dims=batch_dims + [align_targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim)

          enc_log_probs = rf.log_softmax(aux_logits_packed, axis=model.align_target_dim)
          # enc_log_probs = utils.copy_tensor_replace_dim_tag(enc_log_probs, enc_spatial_dim, align_targets_spatial_dim)
          aux_loss = rf.cross_entropy(
            target=align_targets_packed,
            estimated=enc_log_probs,
            axis=model.align_target_dim,
            estimated_type="log-probs",
          )
          loss_name = f"enc_ce_{layer_idx}"

          best = rf.reduce_argmax(aux_logits_packed, axis=model.align_target_dim)
          frame_error = best != align_targets_packed
          frame_error.mark_as_loss(name=f"{loss_name}_fer", as_error=True)

        if aux_loss_focal_loss_factors[i] > 0:
          aux_loss *= (1.0 - rf.exp(-aux_loss)) ** aux_loss_focal_loss_factors[i]
        aux_loss.mark_as_loss(
          loss_name,
          scale=aux_loss_scales[i],
          custom_inv_norm_factor=align_targets_spatial_dim.get_size_tensor(),
          use_normalized_loss=True,
        )

  chunking = config.typed_value("chunking", None)

  if chunking:
    print("align_targets_spatial_dim", align_targets_spatial_dim.dyn_size_ext.raw_tensor)
    print("data_spatial_dim", data_spatial_dim.dyn_size_ext.raw_tensor)
    print("enc_spatial_dim", enc_spatial_dim.dyn_size_ext.raw_tensor)
    exit()

  if model.use_joint_model:

    # ------------------- joint loop -------------------

    # isinstance() does not work here, since SegmentalAttEfficientJointLabelDecoder inherits from SegmentalAttLabelDecoder
    if type(model.label_decoder) is SegmentalAttLabelDecoder:
      assert model.label_decoder_state == "joint-lstm", "not implemented yet, simple to extend"
      targets = align_targets
      targets_spatial_dim = align_targets_spatial_dim
      label_logits, _, _ = label_model_viterbi_training(
        model=model.label_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        non_blank_targets=targets,
        non_blank_targets_spatial_dim=targets_spatial_dim,
        segment_starts=segment_starts,
        segment_lens=segment_lens,
        center_positions=center_positions,
        batch_dims=batch_dims,
        separate_blank_loss=model.use_joint_model,
      )
    else:
      assert type(model.label_decoder) is SegmentalAttEfficientLabelDecoder
      if model.label_decoder_state == "joint-lstm":
        targets = align_targets
        targets_spatial_dim = align_targets_spatial_dim
        non_blank_mask_ = None
        non_blank_mask_spatial_dim = None
      else:
        targets = non_blank_targets
        targets_spatial_dim = non_blank_targets_spatial_dim
        non_blank_mask_ = non_blank_mask
        non_blank_mask_spatial_dim = align_targets_spatial_dim

      label_logits, _, _ = label_model_viterbi_training_efficient(
        model=model.label_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
        segment_starts=segment_starts,
        segment_lens=segment_lens,
        center_positions=center_positions,
        non_blank_mask=non_blank_mask_,
        non_blank_mask_spatial_dim=non_blank_mask_spatial_dim,
        ce_targets=align_targets,
        ce_spatial_dim=align_targets_spatial_dim,
        batch_dims=batch_dims,
        beam_dim=beam_dim,
        separate_blank_loss=model.use_joint_model,
      )

      if rf.get_run_ctx().train_flag and config.bool("training_do_realignments", False):
        label_log_prob = rf.log_softmax(label_logits, axis=model.target_dim)
        align_target_scores = -rf.cross_entropy(
          target=align_targets, estimated=label_log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        align_target_scores = rf.reduce_sum(align_target_scores, axis=align_targets_spatial_dim)
  else:

    # ------------------- label loop -------------------

    if (
            type(model.label_decoder) is SegmentalAttLabelDecoder or
            type(model.label_decoder) is GlobalAttDecoder or
            force_inefficient_loop
    ):
      label_logits, label_decoder_outputs, energy_in = label_model_viterbi_training(
        model=model.label_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        non_blank_targets=non_blank_targets,
        non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
        segment_starts=segment_starts,
        segment_lens=segment_lens,
        center_positions=center_positions,
        batch_dims=batch_dims,
        return_label_model_states=model.blank_decoder.get_label_decoder_deps() is not None,
        separate_blank_loss=model.use_joint_model,
      )
    else:
      assert (
                     type(model.label_decoder) is SegmentalAttEfficientLabelDecoder or
                      type(model.label_decoder) is GlobalAttEfficientDecoder
      )
      label_logits, label_decoder_outputs, energy_in = label_model_viterbi_training_efficient(
        model=model.label_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        targets=non_blank_targets,
        targets_spatial_dim=non_blank_targets_spatial_dim,
        segment_starts=segment_starts,
        segment_lens=segment_lens,
        center_positions=center_positions,
        batch_dims=batch_dims,
        ce_targets=non_blank_targets,
        ce_spatial_dim=non_blank_targets_spatial_dim,
        return_label_model_states=model.blank_decoder.get_label_decoder_deps() is not None,
        separate_blank_loss=model.use_joint_model,
      )

    # ------------------- blank loop -------------------

    emit_ground_truth, emit_blank_target_dim = utils.get_emit_ground_truth(align_targets, model.blank_idx)
    if isinstance(model.blank_decoder, BlankDecoderV1):
      emit_log_prob, blank_log_prob = blank_model_viterbi_training(
        model=model.blank_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        align_targets=align_targets,
        align_targets_spatial_dim=align_targets_spatial_dim,
        emit_ground_truth=emit_ground_truth,
        emit_blank_target_dim=emit_blank_target_dim,
        batch_dims=batch_dims,
        beam_dim=beam_dim,
      )
    elif model.blank_decoder_version in (3, 4, 5, 6, 7, 8, 9):
      assert isinstance(
        model.blank_decoder, BlankDecoderV3) or isinstance(
        model.blank_decoder, BlankDecoderV4) or isinstance(
        model.blank_decoder, BlankDecoderV5) or isinstance(
        model.blank_decoder, BlankDecoderV6) or isinstance(
        model.blank_decoder, BlankDecoderV7) or isinstance(
        model.blank_decoder, BlankDecoderV8) or isinstance(
        model.blank_decoder, BlankDecoderV9) or isinstance(
        model.blank_decoder, BlankDecoderV10
      )

      label_states_unmasked = utils.get_unmasked(
        input=label_decoder_outputs[0],
        input_spatial_dim=label_decoder_outputs[1],
        mask=non_blank_mask,
        mask_spatial_dim=align_targets_spatial_dim
      )

      if model.blank_decoder_version == 3:
        emit_log_prob, blank_log_prob = blank_model_viterbi_training_v3(
          model=model.blank_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          label_states_unmasked=label_states_unmasked,
          label_states_unmasked_spatial_dim=align_targets_spatial_dim,
          emit_ground_truth=emit_ground_truth,
          emit_blank_target_dim=emit_blank_target_dim,
          batch_dims=batch_dims,
        )
      elif model.blank_decoder_version == 4:
        emit_log_prob, blank_log_prob = blank_model_viterbi_training_v4(
          model=model.blank_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          label_states_unmasked=label_states_unmasked,
          label_states_unmasked_spatial_dim=align_targets_spatial_dim,
          emit_ground_truth=emit_ground_truth,
          emit_blank_target_dim=emit_blank_target_dim,
          batch_dims=batch_dims,
        )
      elif model.blank_decoder_version == 5:
        emit_log_prob, blank_log_prob = blank_model_viterbi_training_v5(
          model=model.blank_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          label_states_unmasked=label_states_unmasked,
          label_states_unmasked_spatial_dim=align_targets_spatial_dim,
          emit_ground_truth=emit_ground_truth,
          emit_blank_target_dim=emit_blank_target_dim,
          batch_dims=batch_dims,
        )
      elif model.blank_decoder_version == 6:
        emit_log_prob, blank_log_prob = blank_model_viterbi_training_v6(
          model=model.blank_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          label_states_unmasked=label_states_unmasked,
          label_states_unmasked_spatial_dim=align_targets_spatial_dim,
          emit_ground_truth=emit_ground_truth,
          emit_blank_target_dim=emit_blank_target_dim,
          batch_dims=batch_dims,
        )
      elif model.blank_decoder_version == 7:
        # pad center positions to account for virtual label at position -1 to correctly calculate emit distances
        emit_positions, emit_positions_spatial_dim = rf.pad(
          center_positions,
          axes=[non_blank_targets_spatial_dim],
          padding=[(1, 0)],
          value=-1,
        )
        emit_positions_unmasked = utils.get_unmasked(
          emit_positions,
          emit_positions_spatial_dim[0],
          non_blank_mask,
          align_targets_spatial_dim
        )
        emit_log_prob, blank_log_prob = blank_model_viterbi_training_v7(
          model=model.blank_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          label_states_unmasked=label_states_unmasked,
          label_states_unmasked_spatial_dim=align_targets_spatial_dim,
          emit_positions_unmasked=emit_positions_unmasked,
          emit_ground_truth=emit_ground_truth,
          emit_blank_target_dim=emit_blank_target_dim,
          batch_dims=batch_dims,
        )
      elif model.blank_decoder_version == 8:
        emit_log_prob, blank_log_prob = blank_model_viterbi_training_v8(
          model=model.blank_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          emit_ground_truth=emit_ground_truth,
          emit_blank_target_dim=emit_blank_target_dim,
          batch_dims=batch_dims,
        )
      else:  # BlankDecoderV9 or V10
        emit_log_prob, blank_log_prob = blank_model_viterbi_training_v9(
          model=model.blank_decoder,
          energy_in=energy_in[0],  # Tuple[Tensor, Dim] -> Tensor
          energy_in_spatial_dim=energy_in[1],  # Tuple[Tensor, Dim] -> Dim
          enc_spatial_dim=enc_spatial_dim,
          non_blank_mask=non_blank_mask,
          emit_ground_truth=emit_ground_truth,
          emit_blank_target_dim=emit_blank_target_dim,
          batch_dims=batch_dims,
        )
    else:
      raise NotImplementedError

  if rf.get_run_ctx().train_flag and config.bool("training_do_realignments", False):
    train_partition_epoch = config.int("train_partition_epoch", 20)
    with torch.no_grad():
      viterbi_alignment_scores, viterbi_alignment, viterbi_alignment_spatial_dim = model_realign(
        model=model,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        non_blank_targets=non_blank_targets,
        non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
        batch_dims=batch_dims,
        beam_size=100,
        use_recombination="max",
      )

      # realignment has encoder dimension as spatial dim, replace it with align_targets_spatial_dim
      # in order to combine it with the given alignment
      viterbi_alignment = utils.copy_tensor_replace_dim_tag(
        viterbi_alignment, enc_spatial_dim, align_targets_spatial_dim)

      align_targets = rf.where(
        viterbi_alignment_scores > align_target_scores,
        viterbi_alignment,
        align_targets
      )

      for name, tensor, dim, ndim in zip(
              ("realignment",),
              (align_targets,),
              (model.target_dim.dimension,),
              (1,)
      ):
        # dump the new interpolation alignment to hdf
        hdf_filename = f"{name}_full-epoch-{(rf.get_run_ctx().epoch - 1) // train_partition_epoch + 1}.hdf"
        hdf_dataset = SimpleHDFWriter(
          filename=hdf_filename,
          dim=dim,
          ndim=ndim,
          extend_existing_file=os.path.exists(hdf_filename),
        )
        hdf.dump_hdf_rf(
          hdf_dataset=hdf_dataset,
          data=tensor,
          batch_dim=batch_dims[0],
          seq_tags=seq_tags,
        )

viterbi_training: TrainDef[SegmentalAttentionModel]
viterbi_training.learning_rate_control_error_measure = "dev_score_full_sum"


def full_sum_training(
        *,
        model: SegmentalAttentionModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: Dim,
        seq_tags: rf.Tensor,
        interpolation_alignment: Optional[rf.Tensor] = None,
        interpolation_alignment_spatial_dim: Optional[Dim] = None,
        interpolation_alignment_scores: Optional[rf.Tensor] = None,
):
  # assert model.use_joint_model
  # assert isinstance(model.label_decoder, SegmentalAttEfficientLabelDecoder)
  # assert model.label_decoder_state == "nb-lstm"
  if interpolation_alignment is not None:
    assert interpolation_alignment_scores is not None

  from returnn.config import get_global_config

  config = get_global_config()  # noqa
  aux_loss_layers = config.typed_value("aux_loss_layers")
  aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)

  full_sum_training_opts = config.typed_value("full_sum_training_opts", {})
  full_sum_beam_size = full_sum_training_opts.get("beam_size", None)

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

  batch_dims = data.remaining_dims(data_spatial_dim)
  assert len(batch_dims) == 1

  # ------------------- encoder aux loss -------------------

  collected_outputs = {}
  enc_args, enc_spatial_dim = model.encoder.encode(
    data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)

  # if config.bool("debug", False):
  #   print("\n\nseq_tags", seq_tags.raw_tensor)

  if aux_loss_layers:
    for i, layer_idx in enumerate(aux_loss_layers):
      if layer_idx > len(model.encoder.layers):
        continue
      linear = getattr(model, f"enc_aux_logits_{layer_idx}")
      aux_logits = linear(collected_outputs[str(layer_idx - 1)])
      aux_loss = rf.ctc_loss(
        logits=aux_logits,
        targets=non_blank_targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=non_blank_targets_spatial_dim,
        blank_index=model.blank_idx,
      )
      aux_loss.mark_as_loss(
        f"ctc_{layer_idx}",
        scale=aux_loss_scales[i],
        custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
        use_normalized_loss=True,
      )

  # for every frame position, get the corresponding window around it ([B,T,W])
  # TODO: use rf.window() instead
  segment_starts, segment_lens, center_positions = utils.get_segment_starts_and_lens(
    rf.sequence_mask(batch_dims + [enc_spatial_dim]),  # this way, every frame is interpreted as non-blank
    enc_spatial_dim,
    model.center_window_size,
    batch_dims,
    enc_spatial_dim
  )

  if full_sum_beam_size:
    downsampling = full_sum_training_opts.get("lattice_downsampling", 1)
    precompute_chunk_size = full_sum_training_opts.get("precompute_chunk_size", 10)
    interpolation_alignment_factor = full_sum_training_opts.get("alignment_interpolation_factor", 0.0)
    partition_epoch = config.int("train_partition_epoch", 20)
    train_on_viterbi_paths = full_sum_training_opts.get("train_on_viterbi_paths", False)
    only_use_blank_model = full_sum_training_opts.get("only_use_blank_model", False)

    # do not use interpolation alignment for eval datasets
    if not rf.get_run_ctx().train_flag or interpolation_alignment_factor == 0.0:
      interpolation_alignment_factor = 0.0
    # in the first full epoch (20 subepochs), use linear alignment for interpolation
    elif 1 <= rf.get_run_ctx().epoch <= partition_epoch:
      interpolation_alignment = utils.get_linear_alignment(
        non_blank_targets=non_blank_targets,
        non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
        enc_spatial_dim=enc_spatial_dim,
        batch_dims=batch_dims,
        blank_idx=model.blank_idx,
      )
      # log(uniform) * T
      interpolation_alignment_scores = rf.copy_to_device(
        enc_spatial_dim.get_size_tensor(), device=data.device) * rf.log(
        rf.convert_to_tensor(1 / model.target_dim.dimension))
    # otherwise, use given interpolation alignment
    else:
      # set spatial dim to enc_spatial_dim
      interpolation_alignment = utils.copy_tensor_replace_dim_tag(
        interpolation_alignment, interpolation_alignment_spatial_dim, enc_spatial_dim)
      interpolation_alignment_scores = rf.squeeze(
        interpolation_alignment_scores, axis=interpolation_alignment_scores.feature_dim)

    if train_on_viterbi_paths:
      with torch.no_grad():
        # viterbi_alignment_scores, viterbi_alignment, viterbi_alignment_spatial_dim = model_realign_efficient(
        #   model=model.label_decoder,
        #   enc=enc_args["enc"],
        #   enc_ctx=enc_args["enc_ctx"],
        #   enc_spatial_dim=enc_spatial_dim,
        #   non_blank_targets=non_blank_targets,
        #   non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
        #   segment_starts=segment_starts,
        #   segment_lens=segment_lens,
        #   batch_dims=batch_dims,
        #   beam_size=100 if full_sum_beam_size == 1 else full_sum_beam_size,
        #   downsampling=downsampling if rf.get_run_ctx().train_flag else 1,
        #   precompute_chunk_size=precompute_chunk_size,
        #   interpolation_alignment=interpolation_alignment,
        #   interpolation_alignment_factor=interpolation_alignment_factor,
        #   use_recombination="max" if full_sum_beam_size == 1 else None,
        #   return_realignment=True,
        # )
        viterbi_alignment_scores, viterbi_alignment, viterbi_alignment_spatial_dim = model_realign(
          model=model,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          non_blank_targets=non_blank_targets,
          non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
          batch_dims=batch_dims,
          beam_size=100 if full_sum_beam_size == 1 else full_sum_beam_size,
          use_recombination="max" if full_sum_beam_size == 1 else None,
          only_use_blank_model=only_use_blank_model,
        )

      if full_sum_beam_size > 1:
        beam_dim = viterbi_alignment.remaining_dims(batch_dims + [viterbi_alignment_spatial_dim])
        assert len(beam_dim) == 1
        beam_dim = beam_dim[0]
      else:
        beam_dim = None

      viterbi_training(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        align_targets=viterbi_alignment,
        align_targets_spatial_dim=viterbi_alignment_spatial_dim,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        batch_dims=(batch_dims + [beam_dim]) if beam_dim is not None else batch_dims,
        beam_dim=beam_dim,
      )
    else:
      # full-sum loss with beam search
      # seq_log_prob, _, _ = model_realign_efficient(
      #   model=model.label_decoder,
      #   enc=enc_args["enc"],
      #   enc_ctx=enc_args["enc_ctx"],
      #   enc_spatial_dim=enc_spatial_dim,
      #   non_blank_targets=non_blank_targets,
      #   non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
      #   segment_starts=segment_starts,
      #   segment_lens=segment_lens,
      #   batch_dims=batch_dims,
      #   beam_size=full_sum_beam_size,
      #   downsampling=downsampling if rf.get_run_ctx().train_flag else 1,
      #   precompute_chunk_size=precompute_chunk_size,
      #   interpolation_alignment=interpolation_alignment,
      #   interpolation_alignment_factor=interpolation_alignment_factor,
      #   use_recombination="sum",
      # )
      seq_log_prob, _, _ = model_realign(
        model=model,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        non_blank_targets=non_blank_targets,
        non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
        batch_dims=batch_dims,
        beam_size=100 if full_sum_beam_size == 1 else full_sum_beam_size,
        use_recombination="sum",
        only_use_blank_model=False,
      )
      loss = -1 * seq_log_prob
      loss.mark_as_loss("full_sum_loss", scale=1.0, use_normalized_loss=True)

    # in training, do realignment and update previous interpolation alignment if realignment is better
    if rf.get_run_ctx().train_flag and interpolation_alignment_factor > 0.0:
      with torch.no_grad():
        viterbi_alignment_scores, viterbi_alignment, viterbi_alignment_spatial_dim = model_realign_efficient(
          model=model.label_decoder,
          enc=enc_args["enc"],
          enc_ctx=enc_args["enc_ctx"],
          enc_spatial_dim=enc_spatial_dim,
          non_blank_targets=non_blank_targets,
          non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
          segment_starts=segment_starts,
          segment_lens=segment_lens,
          batch_dims=batch_dims,
          beam_size=100,
          downsampling=1,
          precompute_chunk_size=10,
          interpolation_alignment=None,
          interpolation_alignment_factor=0.0,
          use_recombination="max",
          return_realignment=True,
        )

        interpolation_alignment = rf.where(
          viterbi_alignment_scores > interpolation_alignment_scores,
          viterbi_alignment,
          interpolation_alignment
        )
        interpolation_alignment_scores = rf.where(
          viterbi_alignment_scores > interpolation_alignment_scores,
          viterbi_alignment_scores,
          interpolation_alignment_scores
        )

        for name, tensor, dim, ndim in zip(
                ("interpolation-alignment", "interpolation-alignment-scores"),
                (interpolation_alignment, interpolation_alignment_scores),
                (model.target_dim.dimension, 1),
                (1, 0)
        ):
          # dump the new interpolation alignment to hdf
          hdf_filename = f"{name}_full-epoch-{(rf.get_run_ctx().epoch - 1) // partition_epoch + 1}.hdf"
          hdf_dataset = SimpleHDFWriter(
            filename=hdf_filename,
            dim=dim,
            ndim=ndim,
            extend_existing_file=os.path.exists(hdf_filename),
          )
          hdf.dump_hdf_rf(
            hdf_dataset=hdf_dataset,
            data=tensor,
            batch_dim=batch_dims[0],
            seq_tags=seq_tags,
          )
  else:
    label_model_full_sum_training(
      model=model,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      non_blank_targets=non_blank_targets,
      non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
      segment_starts=segment_starts,
      segment_lens=segment_lens,
      center_positions=center_positions,
      batch_dims=batch_dims,
    )


full_sum_training: TrainDef[SegmentalAttentionModel]
full_sum_training.learning_rate_control_error_measure = "dev_score_full_sum"
