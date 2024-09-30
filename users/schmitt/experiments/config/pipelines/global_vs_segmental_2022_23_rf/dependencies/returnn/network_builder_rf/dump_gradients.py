import os.path
from typing import Optional, Dict, List, Callable
import sys
import ast

import numpy as np
import torch
import matplotlib.pyplot as plt
import mpl_toolkits

from returnn.tensor import TensorDict
from returnn.frontend.tensor_array import TensorArray
import returnn.frontend as rf
from returnn.datasets.hdf import SimpleHDFWriter
from returnn.config import get_global_config
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.ctc.realignment import ctc_align_get_center_positions
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.ctc.model import CtcModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder,
  GlobalAttEfficientDecoder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.train import get_alignment_args
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  forward_sequence_efficient as forward_sequence_efficient_segmental,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  forward_sequence as forward_sequence_segmental,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.train import (
  forward_sequence as forward_sequence_global,
)
from .analyze_gradients import _plot_log_prob_gradient_wrt_to_input_batched

from .dump_att_weights import dump_hdfs



def dump_gradients(
        model: SegmentalAttentionModel,
        data: rf.Tensor,
        data_spatial_dim: rf.Dim,
        targets: rf.Tensor,
        targets_spatial_dim: rf.Dim,
        seq_tags: rf.Tensor,
):
  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio
  batch_dims = data.remaining_dims(data_spatial_dim)

  config = get_global_config()

  input_layer_name = config.typed_value("input_layer_name", "encoder_input")

  # set all params trainable, i.e. require gradients
  for name, param in model.named_parameters():
    param.trainable = True
  rf.set_requires_gradient(data)

  if isinstance(model, SegmentalAttentionModel):
    (
      segment_starts, segment_lens, center_positions, non_blank_targets, non_blank_targets_spatial_dim,
      non_blank_mask
    ) = get_alignment_args(
      model=model,
      align_targets=targets,
      align_targets_spatial_dim=targets_spatial_dim,
      batch_dims=batch_dims,
    )
  else:
    segment_starts = None
    segment_lens = None
    center_positions = None
    non_blank_targets = targets
    non_blank_targets_spatial_dim = targets_spatial_dim

  with torch.enable_grad():
    # ------------------- run encoder and capture encoder input -----------------
    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encoder.encode(
      data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)

    # ----------------------------------------------------------------------------------

    if isinstance(model, CtcModel):
      import torchaudio
      # dump targets to hdf file
      dump_hdfs(
        batch_dims=batch_dims,
        seq_tags=seq_tags,
        align_targets=non_blank_targets,
        align_target_dim=model.target_dim.dimension
      )

      layer_idx = len(model.encoder.layers)
      ctc_projection = getattr(model.encoder, f"enc_aux_logits_{layer_idx}")
      aux_logits = ctc_projection(collected_outputs[str(layer_idx - 1)])
      log_probs = rf.log_softmax(aux_logits, axis=model.align_target_dim)

      non_blank_positions = rf.zeros(dims=batch_dims + [non_blank_targets_spatial_dim], dtype="int32")
      enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext)
      non_blank_targets_spatial_sizes = rf.copy_to_device(non_blank_targets_spatial_dim.dyn_size_ext)
      for b in range(seq_tags.raw_tensor.size):
        input_len_b = enc_spatial_sizes.raw_tensor[b]
        target_len_b = non_blank_targets_spatial_sizes.raw_tensor[b]
        ctc_alignment, _ = torchaudio.functional.forced_align(
          log_probs=log_probs.raw_tensor[b, :input_len_b][None],
          targets=non_blank_targets.raw_tensor[b, :target_len_b][None],
          input_lengths=input_len_b[None],
          target_lengths=target_len_b[None],
          blank=model.blank_idx,
        )
        ctc_positions = ctc_align_get_center_positions(ctc_alignment, model.blank_idx)

        non_blank_positions.raw_tensor[b, :len(ctc_positions)] = ctc_positions

      log_probs = rf.gather(
        log_probs,
        indices=non_blank_positions,
        axis=enc_spatial_dim,
      )
    else:
      for enc_layer_idx in range(len(model.encoder.layers) - 1, len(model.encoder.layers)):
        enc = collected_outputs[str(enc_layer_idx)]
        enc_args = {
          "enc": enc,
        }

        if not isinstance(model.label_decoder, TransformerDecoder) and hasattr(model.encoder, "enc_ctx"):
          enc_args["enc_ctx"] = model.encoder.enc_ctx(enc)  # does not exist for transformer decoder

          if model.label_decoder.use_weight_feedback:
            enc_args["inv_fertility"] = rf.sigmoid(model.encoder.inv_fertility(enc))  # does not exist for transformer decoder
          else:
            enc_args["inv_fertility"] = None  # dummy value, not used

        if isinstance(model.label_decoder, SegmentalAttLabelDecoder):
          if type(model.label_decoder) is SegmentalAttLabelDecoder:
            assert not model.use_joint_model

            logits, _ = forward_sequence_segmental(
              model=model.label_decoder,
              enc_args=enc_args,
              enc_spatial_dim=enc_spatial_dim,
              non_blank_targets=non_blank_targets,
              non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
              segment_starts=segment_starts,
              segment_lens=segment_lens,
              center_positions=center_positions,
              batch_dims=batch_dims,
            )

            log_probs = rf.log_softmax(logits, axis=model.target_dim)
            log_probs = log_probs.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])

            # shapes
            # print("att_weights", att_weights)  # [B, T, S, 1]
            # print("logits", logits)  # [B, S, V]
            # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]
          else:
            assert type(model.label_decoder) is SegmentalAttEfficientLabelDecoder, "need to extend for other models"
            assert not model.use_joint_model

            logits, _ = forward_sequence_efficient_segmental(
              model=model.label_decoder,
              enc_args=enc_args,
              enc_spatial_dim=enc_spatial_dim,
              targets=non_blank_targets,
              targets_spatial_dim=non_blank_targets_spatial_dim,
              segment_starts=segment_starts,
              segment_lens=segment_lens,
              center_positions=center_positions,
              batch_dims=batch_dims,
            )
            log_probs = rf.log_softmax(logits, axis=model.target_dim)
            log_probs = log_probs.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])

            # shapes
            # print("att_weights", att_weights)  # [B, T, S, 1]
            # print("logits", logits)  # [B, S, V]
            # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]

        elif type(model.label_decoder) is GlobalAttEfficientDecoder and not model.label_decoder.trafo_att:
          logits, _ = forward_sequence_global(
            model=model.label_decoder,
            enc_args=enc_args,
            enc_spatial_dim=enc_spatial_dim,
            targets=non_blank_targets,
            targets_spatial_dim=non_blank_targets_spatial_dim,
            center_positions=center_positions,
            batch_dims=batch_dims,
          )

          log_probs = rf.log_softmax(logits, axis=model.target_dim)
          log_probs = log_probs.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])

          # shapes
          # print("att_weights", att_weights)  # [B, T, S, 1]
          # print("logits", logits)  # [B, S, V]
          # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]
        elif type(model.label_decoder) is GlobalAttDecoder:
          logits, _ = forward_sequence_global(
            model=model.label_decoder,
            enc_args=enc_args,
            enc_spatial_dim=enc_spatial_dim,
            targets=non_blank_targets,
            targets_spatial_dim=non_blank_targets_spatial_dim,
            center_positions=center_positions,
            batch_dims=batch_dims,
          )
          log_probs = rf.log_softmax(logits, axis=model.target_dim)
          log_probs = log_probs.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])

          # shapes
          # print("att_weights", att_weights)  # [B, T, S, 1]
          # print("logits", log_probs)  # [B, S, V]
          # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]
        else:
          assert isinstance(model.label_decoder, TransformerDecoder) or (
                    type(model.label_decoder) is GlobalAttEfficientDecoder and model.label_decoder.trafo_att)

          if isinstance(model.label_decoder, TransformerDecoder):
            logits, _, _ = model.label_decoder(
              non_blank_targets,
              spatial_dim=non_blank_targets_spatial_dim,
              encoder=model.label_decoder.transform_encoder(enc_args["enc"], axis=enc_spatial_dim),
              state=model.label_decoder.default_initial_state(batch_dims=batch_dims)
            )
          else:
            logits, _ = forward_sequence_global(
              model=model.label_decoder,
              enc_args=enc_args,
              enc_spatial_dim=enc_spatial_dim,
              targets=non_blank_targets,
              targets_spatial_dim=non_blank_targets_spatial_dim,
              center_positions=center_positions,
              batch_dims=batch_dims,
            )

          log_probs = rf.log_softmax(logits, axis=model.target_dim)

          # shapes for trafo decoder
          # print("att_weights", att_weights)  # [B, H, S, T] -> [B, T, S, 1]
          # print("energies", energies)  # [B, H, S, T]
          # print("logits", logits)  # [B, S, V]

          # print("x_linear", x_linear)  # [B, T, F]
          # print("logits", logits)  # [B, S, V]

    gradients = _plot_log_prob_gradient_wrt_to_input_batched(
      input_=collected_outputs[input_layer_name],
      log_probs=log_probs,
      targets=non_blank_targets,
      batch_dims=batch_dims,
      seq_tags=seq_tags,
      enc_spatial_dim=enc_spatial_dim,
      targets_spatial_dim=non_blank_targets_spatial_dim,
      dirname=f"x_linear/log-prob-grads_wrt_x_linear_log-space",
      return_gradients=True,
      ref_alignment_hdf=None,
      ref_alignment_blank_idx=None,
      ref_alignment_json_vocab_path=None,
      json_vocab_path=None,
      print_progress=False,
    )
    input_spatial_dim = collected_outputs[input_layer_name].remaining_dims(batch_dims + [collected_outputs[input_layer_name].feature_dim])[0]
    singleton_dim = gradients.remaining_dims(batch_dims + [non_blank_targets_spatial_dim, input_spatial_dim])[0]

    return gradients, non_blank_targets_spatial_dim, input_spatial_dim, singleton_dim


def _returnn_v2_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  import returnn.frontend as rf
  from returnn.tensor import Tensor, Dim, batch_dim
  from returnn.config import get_global_config

  if rf.is_executing_eagerly():
    batch_size = int(batch_dim.get_dim_value())
    for batch_idx in range(batch_size):
      seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
      print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  default_target_key = config.typed_value("target")
  align_targets = extern_data[default_target_key]
  align_targets_spatial_dim = align_targets.get_time_dim_tag()
  dump_gradients_def = config.typed_value("_dump_gradients_def")

  gradients, non_blank_targets_spatial_dim, enc_spatial_dim, singleton_dim = dump_gradients_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    targets=align_targets,
    targets_spatial_dim=align_targets_spatial_dim,
    seq_tags=extern_data["seq_tag"],
  )

  rf.get_run_ctx().mark_as_output(
    gradients,
    "gradients",
    dims=[batch_dim, non_blank_targets_spatial_dim, enc_spatial_dim, singleton_dim]
  )


def _returnn_v2_get_forward_callback():
  from returnn.tensor import TensorDict
  from returnn.forward_iface import ForwardCallbackIface
  from returnn.config import get_global_config

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      self.gradients_hdf: Optional[SimpleHDFWriter] = None

    def init(self, *, model):
      self.gradients_hdf = SimpleHDFWriter(filename="gradients.hdf", dim=1, ndim=3)

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
      gradients: rf.Tensor = outputs["gradients"]

      non_blank_targets_spatial_dim = gradients.dims[0]
      non_blank_target_len = non_blank_targets_spatial_dim.dyn_size_ext.raw_tensor
      enc_spatial_dim = gradients.dims[1]
      enc_spatial_len = enc_spatial_dim.dyn_size_ext.raw_tensor

      gradients.raw_tensor = gradients.raw_tensor[:non_blank_target_len, :enc_spatial_len]

      # gradients.raw_tensor = gradients.raw_tensor[:non_blank_targets_spatial_dim.dyn_size_ext.]

      hdf.dump_hdf_rf(
        hdf_dataset=self.gradients_hdf,
        data=gradients,
        batch_dim=None,
        seq_tags=[seq_tag],
      )
      # self.gradients_hdf.close()

    def finish(self):
      self.gradients_hdf.close()

  return _ReturnnRecogV2ForwardCallbackIface()
