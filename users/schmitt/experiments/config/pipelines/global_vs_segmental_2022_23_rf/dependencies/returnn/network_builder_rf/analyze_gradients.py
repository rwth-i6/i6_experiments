import os.path
from typing import Optional, Dict, List, Callable
import sys

import torch
import matplotlib.pyplot as plt

from returnn.tensor import TensorDict
from returnn.frontend.tensor_array import TensorArray
import returnn.frontend as rf
from returnn.datasets.hdf import SimpleHDFWriter
from returnn.config import get_global_config
from returnn.frontend.encoder.conformer import ConformerEncoder
from returnn.frontend.decoder.transformer import TransformerDecoder, TransformerDecoderLayer

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
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
  get_s_and_att_efficient as get_s_and_att_efficient_global,
  get_s_and_att as get_s_and_att_global,
)
from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.visualization.visualization import plot_att_weights

from .dump_att_weights import dump_hdfs, scatter_att_weights

from sisyphus import Path


def _get_scattered_grad_tensor(
        grad_list: List[torch.Tensor],
        att_weights: rf.Tensor,
        name: str,
        segment_starts: rf.Tensor,
        new_slice_dim: rf.Dim,
        align_targets_spatial_dim: rf.Dim,
):
  grads_raw = torch.stack(grad_list, dim=1)[None, :, :, None]
  grads_rf = att_weights.copy_template(name=name)
  grads_rf.raw_tensor = grads_raw

  grads_rf = scatter_att_weights(
    att_weights=grads_rf,
    segment_starts=segment_starts,
    new_slice_dim=new_slice_dim,
    align_targets_spatial_dim=align_targets_spatial_dim,
  )

  return grads_rf


def analyze_weights(model: SegmentalAttentionModel):
  s_size = model.label_decoder.get_lstm().out_dim.dimension
  embed_size = model.label_decoder.target_embed.out_dim.dimension
  enc_dim = model.encoder.out_dim.dimension

  readout_in_weights = model.label_decoder.readout_in_w_current_frame.weight.raw_tensor

  readout_in_att_weights = readout_in_weights[s_size + embed_size:s_size + embed_size + enc_dim]
  readout_in_cur_frame_weights = readout_in_weights[s_size + embed_size + enc_dim:]

  print("readout_in_att_weights mean", torch.mean(torch.abs(readout_in_att_weights)))
  print("readout_in_cur_frame_weights mean", torch.mean(torch.abs(readout_in_cur_frame_weights)))
  exit()


def _plot_log_prob_gradient_wrt_to_input(
        input: rf.Tensor,
        log_probs: rf.Tensor,
        targets: rf.Tensor,
        batch_dims: List[rf.Dim],
        seq_tags: rf.Tensor,
        enc_spatial_dim: rf.Dim,
        targets_spatial_dim: rf.Dim,
        input_name: str,
        json_vocab_path: Path
):
  input_raw = input.raw_tensor
  log_probs_raw = log_probs.raw_tensor
  B = log_probs_raw.size(0)  # noqa
  S = log_probs_raw.size(1)  # noqa

  batch_gradients = []
  for b in range(B):
    s_gradients = []
    for s in range(S):
      v = targets.raw_tensor[b, s]
      input_raw.retain_grad()
      log_probs_raw.retain_grad()
      log_probs_raw[b, s, v].backward(retain_graph=True)

      x_linear_grad_l2 = torch.linalg.vector_norm(input_raw.grad[b], dim=-1)
      s_gradients.append(x_linear_grad_l2)
    batch_gradients.append(torch.stack(s_gradients, dim=0))
  x_linear_grad_l2_raw = torch.stack(batch_gradients, dim=0)[:, :, :, None]
  x_linear_grad_l2 = rf.convert_to_tensor(
    x_linear_grad_l2_raw,
    dims=batch_dims + [targets_spatial_dim, enc_spatial_dim, rf.Dim(1, name="dummy")],
  )

  dirname = f"log-prob-grads_wrt_{input_name}"
  dump_hdfs(
    att_weights=x_linear_grad_l2,
    batch_dims=batch_dims,
    dirname=dirname,
    seq_tags=seq_tags,
  )

  # json_vocab_path = Path(
  #   "/work/asr4/zeyer/setups-data/combined/2021-05-31/work/i6_core/text/label/sentencepiece/vocab/ExtractSentencePieceVocabJob.mSRAzk018au3/output/spm.vocab")
  targets_hdf = Path("targets.hdf")
  ref_alignment_hdf = Path(
    "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2021_22/i6_core/returnn/forward/ReturnnForwardJob.1fohfY7LLczN/output/alignments.hdf")
  ref_alignment_json_vocab_path = Path(
    "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab")
  ref_alignment_blank_idx = 10025

  plot_att_weights(
    att_weight_hdf=Path(os.path.join(dirname, "att_weights.hdf")),
    targets_hdf=targets_hdf,
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=ref_alignment_blank_idx,
    ref_alignment_hdf=ref_alignment_hdf,
    ref_alignment_json_vocab_path=ref_alignment_json_vocab_path,
    json_vocab_path=json_vocab_path,
    segment_whitelist=list(seq_tags.raw_tensor),
    plot_name=dirname
  )


code_obj_to_func = None
captured_tensors = None  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor


def _trace_func(frame, event, arg):
  """
  Trace func to get intermediate outputs.
  """
  func = code_obj_to_func.get(frame.f_code)
  if func:
    if event == "call":
      captured_tensors.setdefault(func, []).append({})
    else:
      for k, v in frame.f_locals.items():
        if not isinstance(v, rf.Tensor):
          continue
        prev = captured_tensors[func][-1].get(k, None)
        if prev is None or prev[-1] is not v:
          print(f"{func.__qualname__} tensor var changed: {k} = {v}")
          captured_tensors[func][-1].setdefault(k, []).append(v)
    return _trace_func


def copy_lower_triangular_matrix_to_upper(matrix: torch.Tensor):
  # Get the indices of the lower triangular part (excluding the diagonal)
  lower_tri_indices = torch.tril_indices(row=matrix.size(0), col=matrix.size(1), offset=-1)
  # Get the values from the lower triangular part (excluding the diagonal)
  lower_tri_values = matrix[lower_tri_indices[0], lower_tri_indices[1]]
  # Swap the indices to get the positions for the upper triangular part
  upper_tri_indices = lower_tri_indices.flip(0)
  # Create a copy of the original matrix to keep it unchanged
  full_matrix = matrix.clone()
  # Place the values in the corresponding upper triangular positions
  full_matrix[upper_tri_indices[0], upper_tri_indices[1]] = lower_tri_values

  return full_matrix


def get_tensors_trafo_decoder(
        model: TransformerDecoder,
        enc_args: Dict[str, rf.Tensor],
        enc_spatial_dim: rf.Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: rf.Dim,
        batch_dims: List[rf.Dim],
        dec_layer_idx: int,
        get_cross_attentions: bool = True,
):
  funcs_to_trace_list = [
    TransformerDecoder.__call__,
    TransformerDecoderLayer.__call__,
    rf.CrossAttention.__call__,
    rf.CrossAttention.attention,
    rf.dot_attention,
  ]
  global code_obj_to_func
  global captured_tensors
  code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
  captured_tensors = {}

  _dec_layer_idx = 2 * dec_layer_idx + int(get_cross_attentions)

  _layer_mapping = {
    # dot_attention is called twice per decoder layer (self-attention and cross-attention)
    # we want the last cross-attention, therefore we use 11 as the second arg here
    "att_weights": (rf.dot_attention, _dec_layer_idx, "att_weights", -1),
    "energy": (rf.dot_attention, _dec_layer_idx, "energy", -1),
    "logits": (TransformerDecoder.__call__, 0, "logits", -1),
  }

  sys.settrace(_trace_func)
  model(
    non_blank_targets,
    spatial_dim=non_blank_targets_spatial_dim,
    encoder=model.transform_encoder(enc_args["enc"], axis=enc_spatial_dim),
    state=model.default_initial_state(batch_dims=batch_dims)
  )
  sys.settrace(None)

  att_weights = None
  energies = None
  logits = None

  for tensor_name, var_path in _layer_mapping.items():
    new_out = captured_tensors
    for k in var_path:
      new_out = new_out[k]

    if tensor_name == "att_weights":
      att_weights = new_out
    elif tensor_name == "energy":
      energies = new_out
    else:
      assert tensor_name == "logits"
      logits = new_out  # type: rf.Tensor

  return att_weights, energies, logits


def set_trace_variables(
        funcs_to_trace_list: List,
):
  funcs_to_trace_list = funcs_to_trace_list
  global code_obj_to_func
  global captured_tensors
  code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
  captured_tensors = {}


def process_captured_tensors(
        layer_mapping: Dict,
        process_func: Optional[Callable] = None
):
  tensor_list = []

  for tensor_name, var_path in list(layer_mapping.items()):
    new_out = captured_tensors
    for k in var_path:
      new_out = new_out[k]

    tensor_list.append(process_func(new_out) if process_func else new_out)
    del layer_mapping[tensor_name]

  return tensor_list if len(tensor_list) > 1 else tensor_list[0]


def analyze_gradients(
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

  # start by analyzing the weights
  # analyze_weights(model)

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
    non_blank_mask = None

  with torch.enable_grad():
    # ------------------- run encoder and capture encoder input -----------------
    set_trace_variables(
      funcs_to_trace_list=[
        model.encoder.encode,
        ConformerEncoder.__call__
      ]
    )

    collected_outputs = {}
    sys.settrace(_trace_func)
    enc_args, enc_spatial_dim = model.encoder.encode(
      data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    sys.settrace(None)

    x_linear = process_captured_tensors(
      layer_mapping={"x_linear": (ConformerEncoder.__call__, 0, "x_linear", -1)},
    )
    assert isinstance(x_linear, rf.Tensor)

    # ----------------------------------------------------------------------------------

    for enc_layer_idx in range(11, 12):
      enc = collected_outputs[str(enc_layer_idx)]
      enc_args = {
        "enc": enc,
      }

      if not isinstance(model.label_decoder, TransformerDecoder):
        enc_args["enc_ctx"] = model.encoder.enc_ctx(enc)  # does not exist for transformer decoder

      if model.label_decoder.use_weight_feedback:
        enc_args["inv_fertility"] = rf.sigmoid(model.encoder.inv_fertility(enc))  # does not exist for transformer decoder

      if isinstance(model.label_decoder, GlobalAttDecoder) or isinstance(model.label_decoder, TransformerDecoder):
        dump_hdfs(
          batch_dims=batch_dims,
          seq_tags=seq_tags,
          align_targets=non_blank_targets,
          align_target_dim=model.target_dim.dimension
        )

      if isinstance(model.label_decoder, SegmentalAttLabelDecoder):
        dump_hdfs(
          batch_dims=batch_dims,
          seq_tags=seq_tags,
          center_positions=center_positions,
          segment_starts=segment_starts,
          segment_lens=segment_lens,
          align_targets=non_blank_targets,
          align_target_dim=model.target_dim.dimension
        )

        max_num_frames = rf.reduce_max(enc_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()
        new_slice_dim = rf.Dim(min(model.center_window_size, max_num_frames), name="att_window")

        assert type(model.label_decoder) is SegmentalAttEfficientLabelDecoder, "need to extend for other models"
        assert not model.use_joint_model

        set_trace_variables(
          funcs_to_trace_list=[
            forward_sequence_efficient_segmental,
            SegmentalAttEfficientLabelDecoder.__call__,
            SegmentalAttEfficientLabelDecoder.decode_logits,
          ]
        )

        sys.settrace(_trace_func)
        forward_sequence_efficient_segmental(
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
        sys.settrace(None)

        def _replace_slice_dim(tensor: rf.Tensor):
          old_slice_dim = tensor.remaining_dims(
            batch_dims + [model.label_decoder.att_num_heads, non_blank_targets_spatial_dim])
          assert len(old_slice_dim) == 1
          return utils.copy_tensor_replace_dim_tag(tensor, old_slice_dim[0], new_slice_dim)

        att_weights = process_captured_tensors(
          layer_mapping={"att_weights": (SegmentalAttEfficientLabelDecoder.__call__, 0, "att_weights", -1)},
          process_func=_replace_slice_dim
        )
        assert isinstance(att_weights, rf.Tensor)

        if model.center_window_size != 1:
          energies = process_captured_tensors(
            layer_mapping={"energy": (SegmentalAttEfficientLabelDecoder.__call__, 0, "energy", -1)},
            process_func=_replace_slice_dim
          )
          assert isinstance(energies, rf.Tensor)
        else:
          energies = None  # no energies for center window size 1 (att weights = 1 for current frame)

        logits = process_captured_tensors(
          layer_mapping={"logits": (SegmentalAttLabelDecoder.decode_logits, 0, "logits", -1)}
        )
        assert isinstance(logits, rf.Tensor)
        log_probs = rf.log_softmax(logits, axis=model.target_dim)

        # shapes
        # print("att_weights", att_weights)  # [B, T, S, 1]
        # print("logits", logits)  # [B, S, V]
        # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]

        json_vocab_path = Path(
          "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab")

      elif type(model.label_decoder) is GlobalAttEfficientDecoder:
        set_trace_variables(
          funcs_to_trace_list=[
            forward_sequence_global,
            get_s_and_att_efficient_global,
            GlobalAttEfficientDecoder.__call__,
            GlobalAttEfficientDecoder.decode_logits,
          ]
        )

        sys.settrace(_trace_func)
        forward_sequence_global(
          model=model.label_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          targets=non_blank_targets,
          targets_spatial_dim=non_blank_targets_spatial_dim,
          center_positions=center_positions,
          batch_dims=batch_dims,
        )
        sys.settrace(None)

        att_weights = process_captured_tensors(
          layer_mapping={"att_weights": (GlobalAttEfficientDecoder.__call__, 0, "att_weights", -1)},
        )
        assert isinstance(att_weights, rf.Tensor)
        energies = process_captured_tensors(
          layer_mapping={"energy": (GlobalAttEfficientDecoder.__call__, 0, "energy", -1)},
        )
        assert isinstance(energies, rf.Tensor)
        logits = process_captured_tensors(
          layer_mapping={"logits": (GlobalAttEfficientDecoder.decode_logits, 0, "logits", -1)}
        )
        assert isinstance(logits, rf.Tensor)
        log_probs = rf.log_softmax(logits, axis=model.target_dim)

        # shapes
        # print("att_weights", att_weights)  # [B, T, S, 1]
        # print("logits", logits)  # [B, S, V]
        # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]

        json_vocab_path = Path(
          "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab")
      elif type(model.label_decoder) is GlobalAttDecoder:
        set_trace_variables(
          funcs_to_trace_list=[
            forward_sequence_global,
            get_s_and_att_global,
            GlobalAttDecoder.loop_step,
            GlobalAttDecoder.decode_logits,
          ]
        )

        sys.settrace(_trace_func)
        forward_sequence_global(
          model=model.label_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          targets=non_blank_targets,
          targets_spatial_dim=non_blank_targets_spatial_dim,
          center_positions=center_positions,
          batch_dims=batch_dims,
        )
        sys.settrace(None)

        max_num_labels = rf.reduce_max(non_blank_targets_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

        att_weights = process_captured_tensors(
          layer_mapping={
            f"att_weight_step{i}": (GlobalAttDecoder.loop_step, i, "att_weights", -1) for i in range(max_num_labels)},
        )
        att_weights = rf.stack(att_weights, out_dim=non_blank_targets_spatial_dim)

        energies = process_captured_tensors(
          layer_mapping={
            f"energy_step{i}": (GlobalAttDecoder.loop_step, i, "energy", -1) for i in range(max_num_labels)},
        )
        energies = rf.stack(energies, out_dim=non_blank_targets_spatial_dim)

        logits = process_captured_tensors(
          layer_mapping={"logits": (GlobalAttDecoder.decode_logits, 0, "logits", -1)}
        )
        assert isinstance(logits, rf.Tensor)
        log_probs = rf.log_softmax(logits, axis=model.target_dim)
        log_probs = log_probs.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])

        # shapes
        # print("att_weights", att_weights)  # [B, T, S, 1]
        # print("logits", log_probs)  # [B, S, V]
        # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]

        json_vocab_path = Path(
          "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab")
      else:
        assert isinstance(model.label_decoder, TransformerDecoder)

        for get_cross_attentions in [True, False]:
          for dec_layer_idx in range(6):
            set_trace_variables(
              funcs_to_trace_list=[
                TransformerDecoder.__call__,
                TransformerDecoderLayer.__call__,
                rf.CrossAttention.__call__,
                rf.CrossAttention.attention,
                rf.dot_attention,
              ]
            )

            sys.settrace(_trace_func)
            model.label_decoder(
              non_blank_targets,
              spatial_dim=non_blank_targets_spatial_dim,
              encoder=model.label_decoder.transform_encoder(enc_args["enc"], axis=enc_spatial_dim),
              state=model.label_decoder.default_initial_state(batch_dims=batch_dims)
            )
            sys.settrace(None)

            _dec_layer_idx = 2 * dec_layer_idx + int(get_cross_attentions)
            att_weights = process_captured_tensors(
              layer_mapping={"att_weights": (rf.dot_attention, _dec_layer_idx, "att_weights", -1)}
            )
            assert isinstance(att_weights, rf.Tensor)
            energies = process_captured_tensors(
              layer_mapping={"energy": (rf.dot_attention, _dec_layer_idx, "energy", -1)}
            )
            assert isinstance(energies, rf.Tensor)
            logits = process_captured_tensors(
              layer_mapping={"logits": (TransformerDecoder.__call__, 0, "logits", -1)}
            )
            log_probs = rf.log_softmax(logits, axis=model.target_dim)

            # shapes for trafo decoder
            # print("att_weights", att_weights)  # [B, H, S, T] -> [B, T, S, 1]
            # print("energies", energies)  # [B, H, S, T]
            # print("logits", logits)  # [B, S, V]

            # print("x_linear", x_linear)  # [B, T, F]
            # print("logits", logits)  # [B, S, V]

            json_vocab_path = Path(
              "/work/asr4/zeyer/setups-data/combined/2021-05-31/work/i6_core/text/label/sentencepiece/vocab/ExtractSentencePieceVocabJob.mSRAzk018au3/output/spm.vocab")

            def _plot_attention_weights():
              if get_cross_attentions:
                att_head_dim = att_weights.remaining_dims(batch_dims + [enc_spatial_dim, targets_spatial_dim])
                assert len(att_head_dim) == 1
                att_head_dim = att_head_dim[0]
              else:
                # this is either the att head dim or the kv-dim
                att_head_and_kv_dim = att_weights.remaining_dims(batch_dims + [targets_spatial_dim])
                # att head dim is static, kv-dim is dynamic
                att_head_dim = [dim for dim in att_head_and_kv_dim if dim.is_static()][0]
                kv_dim = [dim for dim in att_head_and_kv_dim if dim.is_dynamic()][0]
                # kv_dim has seq lens (1, 2, ..., S) but needs single len (S) for my hdf_dump script
                kv_dim.dyn_size_ext.raw_tensor = targets_spatial_dim.dyn_size_ext.raw_tensor.clone()

              for tensor, tensor_name in (
                      # (att_weights, "att_weights"),
                      (energies, "energies"),
              ):
                for h in range(att_head_dim.dimension):
                  tensor_single_head = rf.gather(
                    tensor,
                    indices=rf.constant(h, dims=batch_dims),
                    axis=att_head_dim,
                  )
                  dummy_att_head_dim = rf.Dim(1, name="att_head")
                  tensor_single_head = rf.expand_dim(tensor_single_head, dim=dummy_att_head_dim)

                  if get_cross_attentions:
                    # [B, S, T, 1]
                    tensor_single_head = tensor_single_head.copy_transpose(
                      batch_dims + [targets_spatial_dim, enc_spatial_dim, dummy_att_head_dim])
                  else:
                    # [B, S, S, 1]
                    tensor_single_head = tensor_single_head.copy_transpose(
                      batch_dims + [targets_spatial_dim, kv_dim, dummy_att_head_dim])
                    # optionally copy lower triangular part to upper triangular part
                    # att_weights_single_head.raw_tensor[0, :, :, 0] = copy_lower_triangular_matrix_to_upper(
                    #   att_weights_single_head.raw_tensor[0, :, :, 0]
                    # )

                  if get_cross_attentions:
                    dirname = f"cross-att/enc-layer-{enc_layer_idx + 1}_dec-layer-{dec_layer_idx + 1}"
                  else:
                    dirname = f"self-att/dec-layer-{dec_layer_idx + 1}"
                  dirname = os.path.join(dirname, f"{tensor_name}_head-{h}")

                  dump_hdfs(
                    att_weights=tensor_single_head,
                    batch_dims=batch_dims,
                    dirname=dirname,
                    seq_tags=seq_tags,
                  )

                  json_vocab_path = Path("/work/asr4/zeyer/setups-data/combined/2021-05-31/work/i6_core/text/label/sentencepiece/vocab/ExtractSentencePieceVocabJob.mSRAzk018au3/output/spm.vocab")
                  targets_hdf = Path("targets.hdf")
                  if get_cross_attentions:
                    ref_alignment_hdf = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2021_22/i6_core/returnn/forward/ReturnnForwardJob.1fohfY7LLczN/output/alignments.hdf")
                    ref_alignment_json_vocab_path = Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab")
                    ref_alignment_blank_idx = 10025
                  else:
                    ref_alignment_hdf = targets_hdf
                    ref_alignment_json_vocab_path = json_vocab_path
                    ref_alignment_blank_idx = -1  # dummy index

                  plot_att_weights(
                    att_weight_hdf=Path(os.path.join(dirname, "att_weights.hdf")),
                    # targets_hdf=Path(os.path.join(hdf_dirname, "targets.hdf")),
                    targets_hdf=targets_hdf,
                    seg_starts_hdf=None,
                    seg_lens_hdf=None,
                    center_positions_hdf=None,
                    target_blank_idx=None,
                    ref_alignment_blank_idx=ref_alignment_blank_idx,
                    ref_alignment_hdf=ref_alignment_hdf,
                    ref_alignment_json_vocab_path=ref_alignment_json_vocab_path,
                    json_vocab_path=json_vocab_path,
                    segment_whitelist=list(seq_tags.raw_tensor),
                    plot_name=dirname
                  )

            _plot_attention_weights()

      _plot_log_prob_gradient_wrt_to_input(
        input=enc_args["enc"],
        log_probs=log_probs,
        targets=non_blank_targets,
        batch_dims=batch_dims,
        seq_tags=seq_tags,
        enc_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=non_blank_targets_spatial_dim,
        input_name="enc-12",
        json_vocab_path=json_vocab_path
      )



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
  analyze_gradients_def = config.typed_value("_analyze_gradients_def")

  analyze_gradients_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    targets=align_targets,
    targets_spatial_dim=align_targets_spatial_dim,
    seq_tags=extern_data["seq_tag"],
  )

  # analyze_att_energy_gradients(
  #   model=model,
  #   data=data,
  #   data_spatial_dim=data_spatial_dim,
  #   align_targets=align_targets,
  #   align_targets_spatial_dim=align_targets_spatial_dim,
  #   seq_tags=extern_data["seq_tag"],
  # )


def _returnn_v2_get_forward_callback():
  from returnn.tensor import TensorDict
  from returnn.forward_iface import ForwardCallbackIface
  from returnn.config import get_global_config

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      pass

    def init(self, *, model):
      pass

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
      pass

    def finish(self):
      pass

  return _ReturnnRecogV2ForwardCallbackIface()
