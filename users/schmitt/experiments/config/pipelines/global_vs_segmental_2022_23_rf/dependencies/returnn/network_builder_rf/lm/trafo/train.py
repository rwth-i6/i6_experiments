from typing import Dict, List, Tuple, Optional, Union

from returnn.tensor import TensorDict
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.lm.trafo.model import TransformerDecoder


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


def from_scratch_training(
        *,
        model: TransformerDecoder,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        targets: rf.Tensor,
        targets_spatial_dim: Dim
):
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  config = get_global_config()  # noqa
  use_normalized_loss = config.bool("use_normalized_loss", True)

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio


  batch_dims = data.remaining_dims(data_spatial_dim)

  logits, _, _ = model(
    rf.shift_right(targets, axis=targets_spatial_dim, pad_value=0),
    spatial_dim=targets_spatial_dim,
    state=model.default_initial_state(batch_dims=batch_dims)
  )

  logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False)
  targets_packed, _ = rf.pack_padded(
    targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
  )

  log_prob = rf.log_softmax(logits_packed, axis=model.vocab_dim)
  # log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=targets.sparse_dim)
  loss = rf.cross_entropy(
    target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.vocab_dim
  )
  loss.mark_as_loss("ce", use_normalized_loss=use_normalized_loss)

  # pp = p(a_1^S)^{-1/S} = exp(-1/S * log(p(a_1^S)))
  target_log_probs = rf.gather(
    log_prob,
    indices=targets_packed,
    axis=model.vocab_dim,
  )
  target_log_prob_sum = rf.reduce_sum(target_log_probs, axis=target_log_probs.dims)
  target_log_prob_avg = target_log_prob_sum / rf.reduce_sum(targets_spatial_dim.dyn_size_ext, axis=batch_dims)
  perplexity = rf.exp(-target_log_prob_avg)
  perplexity.mark_as_loss(name="ppl", as_error=True)
