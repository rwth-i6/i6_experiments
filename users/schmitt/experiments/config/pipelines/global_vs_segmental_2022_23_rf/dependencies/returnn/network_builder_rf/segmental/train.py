import torch

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import FramewiseTrainDef

from returnn.tensor import TensorDict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt.augmentation.alignment import shift_alignment_boundaries_batched
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  viterbi_training as label_model_viterbi_training
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.train import (
  viterbi_training as blank_model_viterbi_training
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.train import (
  viterbi_training_v3 as blank_model_viterbi_training_v3
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
  BlankDecoderV1,
  BlankDecoderV3,
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
    non_blank_targets, non_blank_targets_spatial_dim = None, None
    segment_starts, segment_lens = utils.get_segment_starts_and_lens(
      utils.get_non_blank_mask(align_targets, blank_idx=-1),  # this way, every frame is interpreted as non-blank
      align_targets,
      align_targets_spatial_dim,
      model,
      batch_dims,
      align_targets_spatial_dim
    )
    # set blank indices in alignment to 0 (= EOS index of imported global att model which is not used otherwise)
    align_targets.raw_tensor[align_targets.raw_tensor == model.target_dim.dimension] = 0
    align_targets.sparse_dim = model.target_dim
  else:
    non_blank_targets, non_blank_targets_spatial_dim = utils.get_masked(
      align_targets, utils.get_non_blank_mask(align_targets, model.blank_idx), align_targets_spatial_dim, batch_dims
    )
    non_blank_targets.sparse_dim = model.target_dim
    segment_starts, segment_lens = utils.get_segment_starts_and_lens(
      utils.get_non_blank_mask(align_targets, model.blank_idx),
      align_targets,
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
        custom_inv_norm_factor=align_targets_spatial_dim.get_size_tensor(),
        use_normalized_loss=True,
      )

  if model.use_joint_model:
    # ------------------- joint loop -------------------
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
    # ------------------- label loop -------------------

    label_decoder_outputs = label_model_viterbi_training(
      model=model.label_decoder,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      non_blank_targets=non_blank_targets,
      non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
      segment_starts=segment_starts,
      segment_lens=segment_lens,
      batch_dims=batch_dims,
      output_tensors=model.blank_decoder.get_label_decoder_deps(),
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
    else:
      assert isinstance(model.blank_decoder, BlankDecoderV3)
      assert "s" in label_decoder_outputs

      label_states_unmasked = utils.get_unmasked(
        input=label_decoder_outputs["s"][0],
        input_spatial_dim=label_decoder_outputs["s"][1],
        mask=utils.get_non_blank_mask(align_targets, model.blank_idx),
        mask_spatial_dim=align_targets_spatial_dim
      )
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

viterbi_training: TrainDef[SegmentalAttentionModel]
viterbi_training.learning_rate_control_error_measure = "dev_score_full_sum"
