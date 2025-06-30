from typing import Dict, List, Tuple, Optional, Union

from returnn.datasets.lm import convert_to_ascii
from returnn.tensor import TensorDict
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

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
        batch_dims: List[Dim],
        detach_att: bool = False,
        mask_att_opts: Optional[Dict] = None,
) -> Tuple[Tensor, Tensor, rf.State]:

  from returnn.config import get_global_config
  config = get_global_config()

  hard_att_opts = config.typed_value("hard_att_opts", None)

  def _body(xs, state: rf.State):
    new_state = rf.State()
    loop_out_, new_state.decoder = model.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=xs["input_embed"],
      state=state.decoder,
      use_mini_att=model.use_mini_att,
      hard_att_opts=hard_att_opts,
      mask_att_opts={"frame_idx": xs.get("h_t")} if xs.get("h_t") is not None else None,
      detach_att=detach_att,
    )
    return loop_out_, new_state

  xs = {"input_embed": input_embeddings}
  if mask_att_opts:
    xs["h_t"] = mask_att_opts["frame_idx"]
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


def get_s_and_att_efficient(
        *,
        model: GlobalAttEfficientDecoder,
        enc_args: Dict[str, Tensor],
        input_embeddings: Tensor,
        enc_spatial_dim: Dim,
        targets_spatial_dim: Dim,
        batch_dims: List[Dim],
        h_s: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, rf.State]:
  if "lstm" in model.decoder_state:
    s, final_state = model.s_wo_att(
      input_embeddings,
      state=model.s_wo_att.default_initial_state(batch_dims=batch_dims),
      spatial_dim=targets_spatial_dim,
    )
  else:
    s = model.s_wo_att_linear(input_embeddings)
    final_state = None

  if h_s is not None:
    att = h_s
  else:
    att = model(
      enc=enc_args["enc"],
      enc_ctx=enc_args.get("enc_ctx"),
      enc_spatial_dim=enc_spatial_dim,
      s=s,
      input_embed=input_embeddings,
      input_embed_spatial_dim=targets_spatial_dim,
      use_mini_att=model.use_mini_att,
    )

  return s, att, final_state


def forward_sequence(
        model: Union[GlobalAttDecoder, GlobalAttEfficientDecoder, TransformerDecoder],
        targets: rf.Tensor,
        targets_spatial_dim: Dim,
        enc_args: Dict[str, rf.Tensor],
        att_enc_args: Dict[str, rf.Tensor],
        enc_spatial_dim: Dim,
        batch_dims: List[Dim],
        return_label_model_states: bool = False,
        center_positions: Optional[rf.Tensor] = None,
        detach_att_before_readout: bool = False,
        detach_h_t_before_readout: bool = False,
) -> Tuple[rf.Tensor, Optional[Tuple[rf.Tensor, Dim]], Optional[Tensor]]:
  from returnn.config import get_global_config
  config = get_global_config()

  if type(model) is TransformerDecoder:
    logits, _ = model(
      rf.shift_right(targets, axis=targets_spatial_dim, pad_value=0),
      spatial_dim=targets_spatial_dim,
      encoder=model.transform_encoder(att_enc_args["enc"], axis=enc_spatial_dim),
      state=model.default_initial_state(batch_dims=batch_dims)
    )
    h_t_logits = None
  else:
    input_embeddings = model.target_embed(targets)
    input_embeddings = rf.shift_right(input_embeddings, axis=targets_spatial_dim, pad_value=0.0)
    input_embeddings = rf.dropout(input_embeddings, drop_prob=model.target_embed_dropout, axis=None)

    if type(model) is GlobalAttDecoder:
      mask_att_opts = None
      if config.bool("mask_att_around_h_t", False):
        mask_att_opts = {"frame_idx": center_positions}
      s, att, final_state = get_s_and_att(
        model=model,
        enc_args=att_enc_args,
        input_embeddings=input_embeddings,
        enc_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        batch_dims=batch_dims,
        detach_att=detach_att_before_readout,
        mask_att_opts=mask_att_opts,
      )
    else:
      assert type(model) is GlobalAttEfficientDecoder

      if model.replace_att_by_h_s:
        s_range = rf.range_over_dim(targets_spatial_dim)
        h_s = rf.gather(att_enc_args["enc"], axis=enc_spatial_dim, indices=s_range)
      else:
        h_s = None

      s, att, final_state = get_s_and_att_efficient(
        model=model,
        enc_args=att_enc_args,
        input_embeddings=input_embeddings,
        enc_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        batch_dims=batch_dims,
        h_s=h_s
      )

    if (
            model.use_current_frame_in_readout or
            model.use_current_frame_in_readout_w_gate or
            model.use_current_frame_in_readout_random or
            model.use_current_frame_in_readout_w_double_gate or
            model.use_sep_h_t_readout
    ):
      h_t = rf.gather(enc_args["enc"], axis=enc_spatial_dim, indices=center_positions)
    else:
      h_t = None

    logits, h_t_logits = model.decode_logits(
      input_embed=input_embeddings,
      s=s,
      att=att,
      h_t=att if model.use_mini_att and h_t is not None else h_t,
      detach_att=detach_att_before_readout,
      detach_h_t=detach_h_t_before_readout
    )

  if return_label_model_states:
    assert model is not TransformerDecoder, "not implemented yet"
    # need to run the loop one more time to get the last output (which is not needed for the loss computation)
    last_embedding = rf.gather(
        input_embeddings,
        axis=targets_spatial_dim,
        indices=rf.copy_to_device(targets_spatial_dim.get_size_tensor() - 1)
    )
    if type(model) is GlobalAttDecoder:
      last_loop_out, _ = model.loop_step(
        **att_enc_args,
        enc_spatial_dim=enc_spatial_dim,
        input_embed=last_embedding,
        state=final_state.decoder,
      )
      last_s_out = last_loop_out["s"]
    else:
      if "lstm" in model.decoder_state:
        last_s_out, _ = model.s_wo_att(
          last_embedding,
          state=final_state,
          spatial_dim=single_step_dim,
        )
      else:
        last_s_out = model.s_wo_att_linear(last_embedding)

    singleton_dim = Dim(name="singleton", dimension=1)

    s_concat = rf.concat(
      (s, targets_spatial_dim),
      (rf.expand_dim(last_s_out, singleton_dim), singleton_dim),
    )

    return logits, s_concat, h_t_logits

  return logits, None, h_t_logits


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

  logits, _, _ = forward_sequence(
    model=model.label_decoder,
    targets=targets,
    targets_spatial_dim=targets_spatial_dim,
    enc_args=enc_args,
    att_enc_args=enc_args,
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


from_scratch_training: TrainDef[GlobalAttentionModel]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"
