from i6_experiments.users.schmitt import hdf

from typing import Optional, Dict, Any, Tuple, Sequence


import returnn.frontend as rf

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import GlobalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_ import recog

from returnn.tensor import Dim, single_step_dim, TensorDict


def model_forward(
        *,
        model: GlobalAttentionModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: Dim,
):
  # use recognition with cheating targets to get the forward scores
  _, forward_scores, _, _, _, beam_dim = recog.model_recog(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    beam_size=1,
    cheating_targets=non_blank_targets,
    cheating_targets_spatial_dim=non_blank_targets_spatial_dim,
  )

  # reduce to best score (remove beam dim)
  forward_scores = rf.reduce_max(forward_scores, axis=beam_dim)

  return forward_scores


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
  forward_def = config.typed_value("_forward_def")

  default_target_key = config.typed_value("target")
  targets = extern_data[default_target_key]
  targets_spatial_dim = targets.get_time_dim_tag()

  scores = forward_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    non_blank_targets=targets,
    non_blank_targets_spatial_dim=targets_spatial_dim,
  )

  rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim,])


_v2_forward_out_scores_filename = "scores.py.gz"


def _returnn_v2_get_forward_callback():
  from typing import TextIO
  from returnn.tensor import Tensor, TensorDict
  from returnn.forward_iface import ForwardCallbackIface

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      self.score_file: Optional[TextIO] = None

    def init(self, *, model):
      import gzip

      self.score_file = gzip.open(_v2_forward_out_scores_filename, "wt")
      self.score_file.write("{\n")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
      scores: Tensor = outputs["scores"]  # []
      self.score_file.write(f"{seq_tag!r}: ")
      score = float(scores.raw_tensor)
      self.score_file.write(f"{score!r},\n")

    def finish(self):
      self.score_file.write("}\n")
      self.score_file.close()

  return _ReturnnRecogV2ForwardCallbackIface()
