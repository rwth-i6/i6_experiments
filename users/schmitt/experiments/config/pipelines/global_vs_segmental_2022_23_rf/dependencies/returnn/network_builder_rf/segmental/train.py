import torch

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import FramewiseTrainDef, FullSumTrainDef

from returnn.tensor import TensorDict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
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
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  full_sum_training_w_beam as label_model_full_sum_training_w_beam
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  full_sum_training_w_beam_eff as label_model_full_sum_training_w_beam_eff
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  full_sum_training_w_beam_eff_w_recomb as label_model_full_sum_training_w_beam_eff_w_recomb
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
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
  BlankDecoderV1,
  BlankDecoderV3,
  BlankDecoderV5,
  BlankDecoderV6
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder
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
  train_def: FullSumTrainDef = config.typed_value("_train_def")
  train_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    non_blank_targets=targets,
    non_blank_targets_spatial_dim=targets_spatial_dim,
  )


def viterbi_training(
        *,
        model: SegmentalAttentionModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        align_targets: rf.Tensor,
        align_targets_spatial_dim: Dim
):
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  config = get_global_config()  # noqa
  aux_loss_layers = config.typed_value("aux_loss_layers")
  aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
  use_ctc_loss = config.typed_value("use_ctc_loss", True)
  generate_ctc_alignments_on_the_fly = config.typed_value("generate_ctc_alignments_on_the_fly", False)
  force_inefficient_loop = config.typed_value("force_inefficient_loop", False)

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

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

  if model.use_joint_model:
    # TODO: use rf.window() instead
    segment_starts, segment_lens = utils.get_segment_starts_and_lens(
      non_blank_mask=rf.sequence_mask(align_targets.dims),  # this way, every frame is interpreted as non-blank
      align_targets_spatial_dim=align_targets_spatial_dim,
      model=model,
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

    segment_starts, segment_lens = utils.get_segment_starts_and_lens(
      non_blank_mask,
      align_targets_spatial_dim,
      model,
      batch_dims,
      non_blank_targets_spatial_dim
    )

  # ------------------- encoder aux loss -------------------

  collected_outputs = {}
  enc_args, enc_spatial_dim = model.encoder.encode(
    data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)

  if aux_loss_layers:
    if use_ctc_loss:
      for i, layer_idx in enumerate(aux_loss_layers):
        if layer_idx > len(model.encoder.layers):
          continue
        linear = getattr(model.encoder, f"enc_aux_logits_{layer_idx}")
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
          custom_inv_norm_factor=align_targets_spatial_dim.get_size_tensor(),
          use_normalized_loss=True,
        )
    elif generate_ctc_alignments_on_the_fly:
      assert len(aux_loss_layers) == 1
      assert len(batch_dims) == 1
      assert model.blank_idx == model.target_dim.dimension
      print("Generating CTC alignments on the fly")
      layer_idx = aux_loss_layers[0]
      linear = getattr(model.encoder, f"enc_aux_logits_{layer_idx}")
      aux_logits = linear(collected_outputs[str(layer_idx - 1)])  # type: rf.Tensor
      print("aux_logits", aux_logits)

      from torchaudio.functional import forced_align
      rem_dims = aux_logits.remaining_dims(batch_dims + [enc_spatial_dim])
      ctc_align = forced_align(
        log_probs=aux_logits.copy_transpose(batch_dims + [enc_spatial_dim] + rem_dims).raw_tensor,
        targets=non_blank_targets.copy_transpose(batch_dims + [non_blank_targets_spatial_dim]).raw_tensor.contiguous(),
        input_lengths=enc_spatial_dim.get_size_tensor().raw_tensor,
        target_lengths=non_blank_targets_spatial_dim.get_size_tensor().raw_tensor,
        blank=model.blank_idx,
      )
      print("ctc_align", ctc_align.shape)
      exit()

  if model.use_joint_model:

    # ------------------- joint loop -------------------

    # isinstance() does not work here, since SegmentalAttEfficientJointLabelDecoder inherits from SegmentalAttLabelDecoder
    if type(model.label_decoder) is SegmentalAttLabelDecoder:
      assert model.label_decoder_state == "joint-lstm", "not implemented yet, simple to extend"
      label_model_viterbi_training(
        model=model.label_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        non_blank_targets=align_targets,
        non_blank_targets_spatial_dim=align_targets_spatial_dim,
        segment_starts=segment_starts,
        segment_lens=segment_lens,
        batch_dims=batch_dims,
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

      label_model_viterbi_training_efficient(
        model=model.label_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
        segment_starts=segment_starts,
        segment_lens=segment_lens,
        non_blank_mask=non_blank_mask_,
        non_blank_mask_spatial_dim=non_blank_mask_spatial_dim,
        ce_targets=align_targets,
        ce_spatial_dim=align_targets_spatial_dim,
        batch_dims=batch_dims,
      )
  else:

    # ------------------- label loop -------------------

    if type(model.label_decoder) is SegmentalAttLabelDecoder or force_inefficient_loop:
      label_decoder_outputs = label_model_viterbi_training(
        model=model.label_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        non_blank_targets=non_blank_targets,
        non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
        segment_starts=segment_starts,
        segment_lens=segment_lens,
        batch_dims=batch_dims,
        return_label_model_states=model.blank_decoder.get_label_decoder_deps() is not None,
      )
    else:
      assert type(model.label_decoder) is SegmentalAttEfficientLabelDecoder
      label_decoder_outputs = label_model_viterbi_training_efficient(
        model=model.label_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        targets=non_blank_targets,
        targets_spatial_dim=non_blank_targets_spatial_dim,
        segment_starts=segment_starts,
        segment_lens=segment_lens,
        batch_dims=batch_dims,
        ce_targets=non_blank_targets,
        ce_spatial_dim=non_blank_targets_spatial_dim,
        return_label_model_states=model.blank_decoder.get_label_decoder_deps() is not None,
      )

    # ------------------- blank loop -------------------

    emit_ground_truth, emit_blank_target_dim = utils.get_emit_ground_truth(align_targets, model.blank_idx)
    if isinstance(model.blank_decoder, BlankDecoderV1):
      blank_model_viterbi_training(
        model=model.blank_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        align_targets=align_targets,
        align_targets_spatial_dim=align_targets_spatial_dim,
        emit_ground_truth=emit_ground_truth,
        emit_blank_target_dim=emit_blank_target_dim,
        batch_dims=batch_dims,
      )
    elif model.blank_decoder_version in (3, 5, 6):
      assert isinstance(
        model.blank_decoder, BlankDecoderV3) or isinstance(
        model.blank_decoder, BlankDecoderV5) or isinstance(
        model.blank_decoder, BlankDecoderV6)

      label_states_unmasked = utils.get_unmasked(
        input=label_decoder_outputs[0],
        input_spatial_dim=label_decoder_outputs[1],
        mask=non_blank_mask,
        mask_spatial_dim=align_targets_spatial_dim
      )

      if model.blank_decoder_version == 3:
        blank_model_viterbi_training_v3(
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
        blank_model_viterbi_training_v5(
          model=model.blank_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          label_states_unmasked=label_states_unmasked,
          label_states_unmasked_spatial_dim=align_targets_spatial_dim,
          emit_ground_truth=emit_ground_truth,
          emit_blank_target_dim=emit_blank_target_dim,
          batch_dims=batch_dims,
        )
      else:
        blank_model_viterbi_training_v6(
          model=model.blank_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          label_states_unmasked=label_states_unmasked,
          label_states_unmasked_spatial_dim=align_targets_spatial_dim,
          emit_ground_truth=emit_ground_truth,
          emit_blank_target_dim=emit_blank_target_dim,
          batch_dims=batch_dims,
        )
    else:
      assert model.blank_decoder_version == 4 and isinstance(model.blank_decoder, BlankDecoderV3)
      blank_model_viterbi_training_v4(
        model=model.blank_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        label_states=label_decoder_outputs[0],
        label_states_spatial_dim=label_decoder_outputs[1],
        non_blank_mask=non_blank_mask,
        non_blank_mask_dim=align_targets_spatial_dim,
        non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
        emit_ground_truth=emit_ground_truth,
        emit_blank_target_dim=emit_blank_target_dim,
        batch_dims=batch_dims,
      )

viterbi_training: TrainDef[SegmentalAttentionModel]
viterbi_training.learning_rate_control_error_measure = "dev_score_full_sum"


def full_sum_training(
        *,
        model: SegmentalAttentionModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: Dim
):
  assert model.use_joint_model
  assert isinstance(model.label_decoder, SegmentalAttEfficientLabelDecoder)
  assert model.label_decoder_state == "nb-lstm"

  from returnn.config import get_global_config

  config = get_global_config()  # noqa
  aux_loss_layers = config.typed_value("aux_loss_layers")
  aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)

  full_sum_training_beam_size = config.int("full_sum_training_beam_size", None)

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

  batch_dims = data.remaining_dims(data_spatial_dim)

  # ------------------- encoder aux loss -------------------

  collected_outputs = {}
  enc_args, enc_spatial_dim = model.encoder.encode(
    data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)

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
  segment_starts, segment_lens = utils.get_segment_starts_and_lens(
    rf.sequence_mask(batch_dims + [enc_spatial_dim]),  # this way, every frame is interpreted as non-blank
    enc_spatial_dim,
    model,
    batch_dims,
    enc_spatial_dim
  )

  # ------------------- joint loop -------------------

  if full_sum_training_beam_size:
    label_model_full_sum_training_w_beam_eff_w_recomb(
      model=model.label_decoder,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      non_blank_targets=non_blank_targets,
      non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
      segment_starts=segment_starts,
      segment_lens=segment_lens,
      batch_dims=batch_dims,
      beam_size=full_sum_training_beam_size
    )
  else:
    label_model_full_sum_training(
      model=model.label_decoder,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      non_blank_targets=non_blank_targets,
      non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
      segment_starts=segment_starts,
      segment_lens=segment_lens,
      batch_dims=batch_dims,
    )


full_sum_training: TrainDef[SegmentalAttentionModel]
full_sum_training.learning_rate_control_error_measure = "dev_score_full_sum"
