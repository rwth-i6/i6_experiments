from sisyphus import tk, Path
import copy
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from i6_core.returnn.training import Checkpoint, ReturnnTrainingJob

from .config_builder import AEDConfigBuilder
from .tools_paths import RETURNN_EXE, RETURNN_ROOT

from .model.aed import AEDModel
from .model.decoder import GlobalAttDecoder

from returnn.tensor import Tensor, Dim, single_step_dim, TensorDict
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder


def _returnn_v2_train_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  targets = extern_data[default_target_key]
  targets_spatial_dim = targets.get_time_dim_tag()
  train_def = config.typed_value("_train_def")
  train_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    targets=targets,
    targets_spatial_dim=targets_spatial_dim,
  )


def get_s_and_att(
        *,
        model: GlobalAttDecoder,
        enc_args: Dict[str, Tensor],
        input_embeddings: Tensor,
        enc_spatial_dim: Dim,
        targets_spatial_dim: Dim,
        batch_dims: List[Dim],
) -> Tuple[Tensor, Tensor, rf.State]:

  from returnn.config import get_global_config
  config = get_global_config()

  def _body(xs, state: rf.State):
    new_state = rf.State()
    loop_out_, new_state.decoder = model.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=xs["input_embed"],
      state=state.decoder,
      use_mini_att=model.use_mini_att,
    )
    return loop_out_, new_state

  xs = {"input_embed": input_embeddings}
  loop_out, final_state, _ = rf.scan(
    spatial_dim=targets_spatial_dim,
    xs=xs,
    ys=model.loop_step_output_templates(
      batch_dims=batch_dims),
    initial=rf.State(
      decoder=model.decoder_default_initial_state(
        batch_dims=batch_dims,
        enc_spatial_dim=enc_spatial_dim,
        use_mini_att=model.use_mini_att,
      ),
    ),
    body=_body,
  )

  return loop_out["s"], loop_out["att"], final_state


def forward_sequence(
        model: Union[GlobalAttDecoder, TransformerDecoder],
        targets: rf.Tensor,
        targets_spatial_dim: Dim,
        enc_args: Dict[str, rf.Tensor],
        enc_spatial_dim: Dim,
        batch_dims: List[Dim],
) -> rf.Tensor:
  input_embeddings = model.target_embed(targets)
  input_embeddings = rf.shift_right(input_embeddings, axis=targets_spatial_dim, pad_value=0.0)
  input_embeddings = rf.dropout(input_embeddings, drop_prob=model.target_embed_dropout, axis=None)

  s, att, final_state = get_s_and_att(
    model=model,
    enc_args=enc_args,
    input_embeddings=input_embeddings,
    enc_spatial_dim=enc_spatial_dim,
    targets_spatial_dim=targets_spatial_dim,
    batch_dims=batch_dims,
  )

  logits = model.decode_logits(
    input_embed=input_embeddings,
    s=s,
    att=att,
  )

  return logits


def from_scratch_training(
        *,
        model: AEDModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        targets: rf.Tensor,
        targets_spatial_dim: Dim
):
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  config = get_global_config()  # noqa
  aux_loss_layers = config.typed_value("aux_loss_layers")
  aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
  aed_loss_scale = config.float("aed_loss_scale", 1.0)
  use_normalized_loss = config.bool("use_normalized_loss", True)

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

  collected_outputs = {}
  enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
  if aux_loss_layers:
    for i, layer_idx in enumerate(aux_loss_layers):
      if layer_idx > len(model.encoder.layers):
        continue
      linear = getattr(model.encoder, f"enc_aux_logits_{layer_idx}")
      aux_logits = linear(collected_outputs[str(layer_idx - 1)])
      aux_loss = rf.ctc_loss(
        logits=aux_logits,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
      )
      aux_loss.mark_as_loss(
        f"ctc_{layer_idx}",
        scale=aux_loss_scales[i],
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
      )

  batch_dims = data.remaining_dims(data_spatial_dim)

  logits = forward_sequence(
    model=model.decoder,
    targets=targets,
    targets_spatial_dim=targets_spatial_dim,
    enc_args=enc_args,
    enc_spatial_dim=enc_spatial_dim,
    batch_dims=batch_dims
  )

  logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False)
  targets_packed, _ = rf.pack_padded(
    targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
  )

  log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
  log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
  loss = rf.cross_entropy(
    target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
  )
  loss.mark_as_loss("ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

  best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
  frame_error = best != targets_packed
  frame_error.mark_as_loss(name="fer", as_error=True)


class TrainExperiment:
  def __init__(
          self,
          config_builder: AEDConfigBuilder,
          alias: str,
          train_opts: Dict,
          train_rqmt: Optional[Dict] = None
  ):
    self.alias = alias
    self.config_builder = copy.deepcopy(config_builder)
    self.num_epochs = train_opts["num_epochs"]
    self.train_opts = copy.deepcopy(train_opts) if train_opts is not None else {}

    self.train_rqmt = train_rqmt if train_rqmt is not None else {}
    self.alias = self.alias + "/train"

    if train_opts.get("use_mini_att", False):
      self.alias += "_mini_lstm"

  def run_train(self) -> Tuple[Dict[int, Checkpoint], Path, Path]:
    train_config = self.config_builder.get_train_config(opts=self.train_opts)

    train_job = ReturnnTrainingJob(
      train_config,
      num_epochs=self.num_epochs,
      log_verbosity=5,
      returnn_python_exe=RETURNN_EXE,
      returnn_root=RETURNN_ROOT,
      mem_rqmt=self.train_rqmt.get("mem", 15),
      time_rqmt=self.train_rqmt.get("time", 30),
      cpu_rqmt=self.train_rqmt.get("cpu", 4),
      horovod_num_processes=self.train_rqmt.get("horovod_num_processes", None),
      distributed_launch_cmd=self.train_rqmt.get("distributed_launch_cmd", "mpirun"),
    )
    if self.train_rqmt.get("gpu_mem", 11) > 11:
      train_job.rqmt["gpu_mem"] = self.train_rqmt["gpu_mem"]
    if "sbatch_args" in self.train_rqmt:
      train_job.rqmt["sbatch_args"] = self.train_rqmt["sbatch_args"]

    train_job.add_alias(self.alias)
    tk.register_output(train_job.get_one_alias() + "/models", train_job.out_model_dir)
    tk.register_output(train_job.get_one_alias() + "/lr_file", train_job.out_learning_rates)

    return train_job.out_checkpoints, train_job.out_model_dir, train_job.out_learning_rates
