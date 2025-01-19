from typing import Dict, List, Tuple, Optional, Union

from returnn.tensor import TensorDict
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.lm.trafo.model import TransformerDecoder


def model_forward(
        *,
        model: TransformerDecoder,
        targets: rf.Tensor,
        targets_spatial_dim: Dim
):
  batch_dims = targets.remaining_dims(targets_spatial_dim)

  logits, _ = model(
    rf.shift_right(targets, axis=targets_spatial_dim, pad_value=0),
    spatial_dim=targets_spatial_dim,
    state=model.default_initial_state(batch_dims=batch_dims)
  )

  log_prob = rf.log_softmax(logits, axis=model.vocab_dim)
  ce = rf.cross_entropy(
    target=targets, estimated=log_prob, estimated_type="log-probs", axis=model.vocab_dim
  )

  return ce


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

  ce = forward_def(
    model=model,
    targets=data,
    targets_spatial_dim=data_spatial_dim,
  )

  rf.get_run_ctx().mark_as_output(ce, "ce", dims=[batch_dim, data_spatial_dim])
  rf.get_run_ctx().mark_as_output(data, "targets", dims=[batch_dim, data_spatial_dim])


_v2_forward_out_scores_filename = "scores.py.gz"


def _returnn_v2_get_forward_callback():
  from typing import TextIO
  from returnn.tensor import Tensor, TensorDict
  from returnn.forward_iface import ForwardCallbackIface
  import math

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      self.score_file: Optional[TextIO] = None
      self.ce_sum = 0
      self.num_labels = 0
      self.num_words = 0

    def init(self, *, model):
      import gzip

      self.score_file = gzip.open(_v2_forward_out_scores_filename, "wt")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
      ce: Tensor = outputs["ce"]  # [S]
      targets: Tensor = outputs["targets"]  # [S]

      # calculate CE sum
      spatial_dim = ce.dims[0]
      spatial_len = spatial_dim.dyn_size_ext.raw_tensor.item()
      # set padding frames to zero because rf.reduce_sum() does not ignore them here for some reason
      ce.raw_tensor[spatial_len:] = 0
      self.ce_sum += rf.reduce_sum(ce, axis=ce.dims).raw_tensor.item()
      self.num_labels += spatial_len

      # calculate number of words by counting BPEs not ending with @@
      target_ids = targets.raw_tensor[:spatial_len]
      targets_serialized = targets.sparse_dim.vocab.get_seq_labels(target_ids).split()
      word_ends = [bpe for bpe in targets_serialized if "@@" not in bpe]
      self.num_words += len(word_ends)

    def finish(self):
      perplexity = math.exp(self.ce_sum / self.num_labels)
      label_word_ratio = self.num_labels / self.num_words
      word_perplexity = perplexity ** label_word_ratio
      self.score_file.write(f"Label perplexity: {perplexity}\n")
      self.score_file.write(f"Word perplexity: {word_perplexity}\n")
      self.score_file.write(f"Label/word ratio: {label_word_ratio}\n")
      self.score_file.close()

  return _ReturnnRecogV2ForwardCallbackIface()
