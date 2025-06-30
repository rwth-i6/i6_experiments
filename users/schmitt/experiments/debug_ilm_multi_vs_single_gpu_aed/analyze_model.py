from typing import Optional, Dict, List, Callable
import sys
import ast
import os
import numpy as np
import torch

from returnn.tensor import TensorDict
import returnn.frontend as rf
from returnn.config import get_global_config
from returnn.frontend.attention import dot_attention

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.train import (
  forward_sequence as forward_sequence_global,
  get_s_and_att as get_s_and_att_global,
)
from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import GlobalAttentionModel
from i6_experiments.users.schmitt.visualization.visualization import plot_att_weights
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.dump_att_weights import dump_hdfs

from sisyphus import Path

fontsize_axis = 16
fontsize_ticks = 14


code_obj_to_func = None
captured_tensors = None  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor


def _plot_log_prob_gradient_wrt_to_input_batched(
        input_: rf.Tensor,
        log_probs: rf.Tensor,
        targets: rf.Tensor,
        batch_dims: List[rf.Dim],
        seq_tags: rf.Tensor,
        enc_spatial_dim: rf.Dim,
        targets_spatial_dim: rf.Dim,
        dirname: str,
        json_vocab_path: Optional[Path],
        ref_alignment_hdf: Optional[Path],
        ref_alignment_blank_idx: Optional[int],
        ref_alignment_json_vocab_path: Optional[Path],
        return_gradients: bool = False,
        print_progress: bool = False,
        dummy_singleton_dim: rf.Dim = rf.Dim(1, name="dummy"),
        input_batch_first: bool = True,
        ref_alignment_is_positional_alignment: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
):
  """

  Args:
    input_: if input_batch_first is True, shape [B, spatial, D], else [spatial, B, D]
    log_probs:
    targets:
    batch_dims:
    seq_tags:
    enc_spatial_dim:
    targets_spatial_dim:
    dirname:
    json_vocab_path:
    ref_alignment_hdf:
    ref_alignment_blank_idx:
    ref_alignment_json_vocab_path:
    return_gradients:
    print_progress:
    dummy_singleton_dim:
    input_batch_first:

  Returns:

  """
  print(f"Plotting log prob gradient w.r.t. input for {dirname}")

  from returnn.config import get_global_config
  config = get_global_config()

  if os.path.exists(dirname) and not return_gradients:
    return

  input_raw = input_.raw_tensor
  log_probs_raw = log_probs.raw_tensor
  B = log_probs_raw.size(0)  # noqa
  S = log_probs_raw.size(1)  # noqa

  log_probs_sum = rf.gather(
    log_probs,
    indices=targets,
  )
  log_probs_sum = rf.reduce_sum(log_probs_sum, axis=batch_dims)
  log_probs_sum_raw = log_probs_sum.raw_tensor

  s_gradients = []
  for s in range(S):
    if print_progress:
      print(f"Step {s}/{S}")
    input_raw.retain_grad()
    log_probs_sum_raw.retain_grad()
    log_probs_sum_raw[s].backward(retain_graph=True)

    x_linear_grad_l2 = torch.linalg.vector_norm(input_raw.grad, dim=-1)
    s_gradients.append(x_linear_grad_l2)

    # zero grad before next step
    input_raw.grad.zero_()
    log_probs_sum_raw.grad.zero_()

  x_linear_grad_l2_raw = torch.stack(s_gradients, dim=1)[:, :, :, None]

  if not input_batch_first:
    x_linear_grad_l2_raw = x_linear_grad_l2_raw.permute(2, 0, 1, 3)  # [B, S, spatial, 1]

  input_spatial_dim = input_.remaining_dims(batch_dims + [input_.feature_dim])[0]
  x_linear_grad_l2 = rf.convert_to_tensor(
    x_linear_grad_l2_raw,
    dims=batch_dims + [targets_spatial_dim, input_spatial_dim, dummy_singleton_dim],
  )

  if return_gradients:
    return x_linear_grad_l2
  else:
    log_x_linear_grad_l2 = rf.log(x_linear_grad_l2)
    dump_hdfs(
      att_weights=log_x_linear_grad_l2,  # use log for better visualization
      batch_dims=batch_dims,
      dirname=dirname,
      seq_tags=seq_tags,
    )

    plot_att_weights(
      att_weight_hdf=Path(os.path.join(dirname, "att_weights.hdf")),
      targets_hdf=Path("targets.hdf"),
      seg_starts_hdf=None,
      seg_lens_hdf=None,
      center_positions_hdf=None,
      target_blank_idx=None,
      ref_alignment_blank_idx=ref_alignment_blank_idx,
      ref_alignment_hdf=ref_alignment_hdf,
      ref_alignment_json_vocab_path=ref_alignment_json_vocab_path,
      json_vocab_path=json_vocab_path,
      segment_whitelist=list(seq_tags.raw_tensor),
      plot_name=dirname,
      plot_w_color_gradient=config.bool("debug", False),
      vmin=vmin,
      vmax=vmax,
      ref_alignment_is_positional_alignment=ref_alignment_is_positional_alignment,
    )


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
          # print(f"{func.__qualname__} tensor var changed: {k} = {v}")
          captured_tensors[func][-1].setdefault(k, []).append(v)
    return _trace_func


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


def pretty_print_log_probs(
        log_probs: rf.Tensor,
        targets: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        alias: str,
        json_vocab: Dict,
):
  """

  Args:
    log_probs: tensor of shape [B, S]
    targets: tensor of shape [B, S]

  Returns:

  """

  log_probs = log_probs.copy_transpose([batch_dim, spatial_dim]).raw_tensor
  targets = targets.copy_transpose([batch_dim, spatial_dim]).raw_tensor

  tuples = list(zip(targets[0].tolist(), log_probs[0].tolist()))
  tuples = map(lambda x: f"{json_vocab[x[0]]}: \t {x[1]}", tuples)
  print(f"\n{alias} log_probs")
  print("\n".join(tuples))
  print("\n")


def analyze_model(
        model: GlobalAttentionModel,
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

  center_positions = None
  non_blank_targets = targets
  non_blank_targets_spatial_dim = targets_spatial_dim

  max_num_labels = rf.reduce_max(non_blank_targets_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

  config = get_global_config()
  ref_alignment_hdf = Path(config.typed_value("ref_alignment_hdf", str))
  ref_alignment_blank_idx = config.typed_value("ref_alignment_blank_idx", int)
  ref_alignment_vocab_path = Path(config.typed_value("ref_alignment_vocab_path", str))
  json_vocab_path = Path(config.typed_value("json_vocab_path", str))
  with open(json_vocab_path, "r") as f:
    json_vocab = ast.literal_eval(f.read())
    json_vocab = {v: k for k, v in json_vocab.items()}

  torch.set_printoptions(threshold=1000)

  with torch.enable_grad():
    # ------------------- run encoder and capture encoder input -----------------
    enc_layer_cls = type(model.encoder.layers[0])
    self_att_cls = type(model.encoder.layers[0].self_att)

    set_trace_variables(
      funcs_to_trace_list=[
        model.encoder.encode,
        type(model.encoder).__call__,
        enc_layer_cls.__call__,
        self_att_cls.__call__,
        self_att_cls.attention,
        dot_attention,
        rf.Sequential.__call__,
      ]
    )

    # ----------------------------------- run encoder ---------------------------------------
    sys.settrace(_trace_func)

    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encoder.encode(
      data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)

    sys.settrace(None)

    # ----------------------------------------------------------------------------------

    assert type(model.label_decoder) is GlobalAttDecoder

    # dump targets into "targets.hdf"
    dump_hdfs(
      batch_dims=batch_dims,
      seq_tags=seq_tags,
      align_targets=non_blank_targets,
      align_target_dim=model.target_dim.dimension
    )

    def _body_am(xs, state: rf.State):
      new_state = rf.State()
      loop_out_, new_state.decoder = model.label_decoder.loop_step(
        **enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=xs["input_embed"],
        state=state.decoder,
        use_zero_att=False,
        # detach_att=True,
      )
      return loop_out_, new_state

    def _body_ilm(xs, state: rf.State):
      new_state = rf.State()
      loop_out_, new_state.decoder = model.label_decoder.loop_step(
        **enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=xs["input_embed"],
        state=state.decoder,
        use_zero_att=True,
      )
      return loop_out_, new_state

    set_trace_variables(
      funcs_to_trace_list=[
        forward_sequence_global,
        get_s_and_att_global,
        GlobalAttDecoder.loop_step,
        GlobalAttDecoder.decode_logits,
      ]
    )

    # sys.settrace(_trace_func)
    input_embeddings = model.label_decoder.target_embed(targets)
    input_embeddings = rf.shift_right(input_embeddings, axis=targets_spatial_dim, pad_value=0.0)

    for alias, use_zero_att, _body in [
      ("AM", False, _body_am),
      ("ILM", True, _body_ilm),
    ]:
      xs = {"input_embed": input_embeddings}
      loop_out, final_state, _ = rf.scan(
        spatial_dim=targets_spatial_dim,
        xs=xs,
        ys=model.label_decoder.loop_step_output_templates(batch_dims=batch_dims),
        initial=rf.State(
          decoder=model.label_decoder.decoder_default_initial_state(
            batch_dims=batch_dims,
            enc_spatial_dim=enc_spatial_dim,
            use_zero_att=use_zero_att,
          ),
        ),
        body=_body,
      )
      logits, _ = model.label_decoder.decode_logits(
        input_embed=input_embeddings,
        s=loop_out["s"],
        att=loop_out["att"],
        detach_att=True,
      )
      log_probs = rf.log_softmax(logits, axis=model.target_dim)
      log_probs_gathered = rf.gather(
        log_probs,
        indices=non_blank_targets,
        axis=model.target_dim,
      )

      pretty_print_log_probs(
        log_probs_gathered,
        non_blank_targets,
        batch_dim=batch_dims[0],
        spatial_dim=non_blank_targets_spatial_dim,
        alias=alias,
        json_vocab=json_vocab,
      )
      print(f"{alias} log_probs_sum", rf.reduce_sum(log_probs_gathered, axis=non_blank_targets_spatial_dim).raw_tensor)

      input_name = "enc11"
      log_probs = log_probs.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])
      _plot_log_prob_gradient_wrt_to_input_batched(
        input_=enc_args["enc"],
        log_probs=log_probs,
        targets=non_blank_targets,
        batch_dims=batch_dims,
        seq_tags=seq_tags,
        enc_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=non_blank_targets_spatial_dim,
        json_vocab_path=json_vocab_path,
        ref_alignment_hdf=ref_alignment_hdf,
        ref_alignment_blank_idx=ref_alignment_blank_idx,
        ref_alignment_json_vocab_path=ref_alignment_vocab_path,
        dirname=f"{input_name}/log-prob-grads_wrt_{input_name}_log-space",
        # return_gradients=True,
        dummy_singleton_dim=rf.Dim(1, name="dummy"),
        vmin=-30,
        vmax=0,
      )

      # input_name = "s"
      # _plot_log_prob_gradient_wrt_to_input_batched(
      #   input_=loop_out["s"],
      #   input_batch_first=False,
      #   log_probs=log_probs,
      #   targets=non_blank_targets,
      #   batch_dims=batch_dims,
      #   seq_tags=seq_tags,
      #   enc_spatial_dim=enc_spatial_dim,
      #   targets_spatial_dim=non_blank_targets_spatial_dim,
      #   json_vocab_path=json_vocab_path,
      #   ref_alignment_hdf=Path("targets.hdf"),
      #   ref_alignment_blank_idx=-1,
      #   ref_alignment_json_vocab_path=json_vocab_path,
      #   dirname=f"{input_name}/log-prob-grads_wrt_{input_name}_log-space",
      #   # return_gradients=True,
      #   dummy_singleton_dim=rf.Dim(1, name="dummy"),
      #   ref_alignment_is_positional_alignment=False,
      # )
    exit()
    # sys.settrace(None)

    # att_weights = process_captured_tensors(
    #   layer_mapping={
    #     f"att_weight_step{i}": (GlobalAttDecoder.loop_step, i, "att_weights", -1) for i in range(max_num_labels)},
    # )
    # att_weights, _ = rf.stack(att_weights, out_dim=non_blank_targets_spatial_dim)


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
  analyze_model_def = config.typed_value("_analyze_model_def")

  analyze_model_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    targets=align_targets,
    targets_spatial_dim=align_targets_spatial_dim,
    seq_tags=extern_data["seq_tag"],
  )


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
