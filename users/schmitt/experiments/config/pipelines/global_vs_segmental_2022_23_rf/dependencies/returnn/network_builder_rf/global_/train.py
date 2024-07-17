from typing import Dict, List, Tuple

from returnn.tensor import TensorDict
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import GlobalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder,
  GlobalAttEfficientDecoder
)


def _returnn_v2_train_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  targets = extern_data[default_target_key]
  targets_spatial_dim = targets.get_time_dim_tag()
  train_def: TrainDef = config.typed_value("_train_def")
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
        batch_dims: List[Dim]
) -> Tuple[Tensor, Tensor]:
  def _body(input_embed: Tensor, state: rf.State):
    new_state = rf.State()
    loop_out_, new_state.decoder = model.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed,
      state=state.decoder,
      use_mini_att=model.use_mini_att,
    )
    return loop_out_, new_state

  loop_out, _, _ = rf.scan(
    spatial_dim=targets_spatial_dim,
    xs=input_embeddings,
    ys=model.loop_step_output_templates(batch_dims=batch_dims),
    initial=rf.State(
      decoder=model.decoder_default_initial_state(
        batch_dims=batch_dims,
        enc_spatial_dim=enc_spatial_dim,
        use_mini_att=model.use_mini_att,
      ),
    ),
    body=_body,
  )

  return loop_out["s"], loop_out["att"]


def get_s_and_att_efficient(
        *,
        model: GlobalAttEfficientDecoder,
        enc_args: Dict[str, Tensor],
        input_embeddings: Tensor,
        enc_spatial_dim: Dim,
        targets_spatial_dim: Dim,
        batch_dims: List[Dim]
) -> Tuple[Tensor, Tensor]:
  if "lstm" in model.decoder_state:
    s, _ = model.s_wo_att(
      input_embeddings,
      state=model.s_wo_att.default_initial_state(batch_dims=batch_dims),
      spatial_dim=targets_spatial_dim,
    )
  else:
    s = model.s_wo_att_linear(input_embeddings)

  att = model(
    enc=enc_args["enc"],
    enc_ctx=enc_args["enc_ctx"],
    enc_spatial_dim=enc_spatial_dim,
    s=s,
    input_embed=input_embeddings,
    input_embed_spatial_dim=targets_spatial_dim,
    use_mini_att=model.use_mini_att,
  )

  return s, att


def from_scratch_training(
        *,
        model: GlobalAttentionModel,
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

  force_inefficient_loop = config.bool("force_inefficient_loop", False)

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

  collected_outputs = {}
  enc_args, enc_spatial_dim = model.encoder.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
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
  input_embeddings = model.label_decoder.target_embed(targets)
  input_embeddings = rf.shift_right(input_embeddings, axis=targets_spatial_dim, pad_value=0.0)

  if type(model.label_decoder) is GlobalAttDecoder or force_inefficient_loop:
    s, att = get_s_and_att(
      model=model.label_decoder,
      enc_args=enc_args,
      input_embeddings=input_embeddings,
      enc_spatial_dim=enc_spatial_dim,
      targets_spatial_dim=targets_spatial_dim,
      batch_dims=batch_dims
    )
  else:
    assert type(model.label_decoder) is GlobalAttEfficientDecoder
    s, att = get_s_and_att_efficient(
      model=model.label_decoder,
      enc_args=enc_args,
      input_embeddings=input_embeddings,
      enc_spatial_dim=enc_spatial_dim,
      targets_spatial_dim=targets_spatial_dim,
      batch_dims=batch_dims
    )

  logits = model.label_decoder.decode_logits(input_embed=input_embeddings, s=s, att=att)
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


from_scratch_training: TrainDef[GlobalAttentionModel]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"
