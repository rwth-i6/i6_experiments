"""
Transformer LM

First: import our existing TF model

checkpoint: /work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/transfo_24_d00.4096_1024.sgd.lr1.8_heads/bk-net-model/network.023
config example: /work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.i6YlJ7HAXfGs/output/returnn.config
"""

from __future__ import annotations
from typing import Union, Any, Tuple, Optional
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.label_scorer import LabelScorerIntf


class MakeModel:
    """for import"""

    def __init__(self, vocab_dim: Union[int, Dim], model_dim: Union[int, Dim], *, num_layers: int, **extra):
        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.extra = extra

    def __call__(self) -> TransformerDecoder:
        if isinstance(self.vocab_dim, int):
            vocab_dim = Dim(self.vocab_dim, name="vocab")
        elif isinstance(self.vocab_dim, Dim):
            vocab_dim = self.vocab_dim
        else:
            raise TypeError(f"vocab dim type {type(self.vocab_dim).__name__}")

        if isinstance(self.model_dim, int):
            model_dim = Dim(self.model_dim, name="model")
        elif isinstance(self.model_dim, Dim):
            model_dim = self.model_dim
        else:
            raise TypeError(f"model dim type {type(self.model_dim).__name__}")

        if vocab_dim.vocab is None:
            from returnn.datasets.util.vocabulary import Vocabulary

            vocab_dim.vocab = Vocabulary.create_vocab_from_labels(
                [str(i) for i in range(vocab_dim.dimension)],
            )

        opts = self.extra.copy()
        for k, v in list(opts.items()):
            if k.endswith("_dim") and isinstance(v, int):
                opts[k] = Dim(v, name=k[: -len("_dim")])

        return self.make_model(vocab_dim=vocab_dim, model_dim=model_dim, num_layers=self.num_layers, **opts)

    @classmethod
    def make_model(cls, vocab_dim: Dim, model_dim: Dim, *, num_layers: int, **extra) -> TransformerDecoder:
        """make"""
        return TransformerDecoder(
            encoder_dim=None,
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            **extra,
        )


def make_label_sync_label_scorer_torch(
        model: TransformerDecoder,
) -> LabelScorerIntf:
  """
  Make label scorer
  """
  import torch
  import tree
  import functools
  from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.label_scorer import (
    StateObjTensorExt,
    StateObjIgnored,
  )

  class LabelScorer(LabelScorerIntf):
    """TransformerDecoder label scorer"""

    def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
      """Initial state."""
      batch_dim = Dim(batch_size, name="batch")
      beam_dim = Dim(1, name="initial-beam")
      batch_dims_ = [batch_dim, beam_dim]
      decoder_state = model.default_initial_state(batch_dims=batch_dims_)
      return tree.map_structure(
        functools.partial(self._map_tensor_to_raw, batch_dim=batch_dim, beam_dim=beam_dim), decoder_state
      )

    def max_remaining_seq_score(
            self, *, state: Any, max_remaining_steps: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
      """max remaining"""
      return torch.zeros((1, 1), device=device)

    def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
            prev_align_label: Optional[torch.Tensor] = None,  # not used
            t: Optional[int] = None,  # not used
    ) -> Tuple[torch.Tensor, Any]:
      """update state"""
      batch_dim = Dim(prev_label.shape[0], name="batch")
      beam_dim = Dim(prev_label.shape[1], name="beam")

      def _map_raw_to_tensor(v):
        if isinstance(v, StateObjTensorExt):
          tensor: Tensor = v.extra
          tensor = tensor.copy_template_new_dim_tags(
            (batch_dim, beam_dim) + tensor.dims[2:], keep_special_axes=True
          )
          tensor.raw_tensor = v.tensor
          return tensor
        elif isinstance(v, StateObjIgnored):
          return v.content
        else:
          raise TypeError(f"_map_raw_to_tensor: unexpected {v} ({type(v).__name__})")

      logits, decoder_state = model(
        rf.convert_to_tensor(prev_label, dims=[batch_dim, beam_dim], sparse_dim=model.vocab_dim),
        spatial_dim=single_step_dim,
        state=tree.map_structure(_map_raw_to_tensor, prev_state),
      )
      label_log_prob = rf.log_softmax(logits, axis=model.vocab_dim)
      assert set(label_log_prob.dims) == {batch_dim, beam_dim, model.vocab_dim}

      return (
        self._map_tensor_to_raw(label_log_prob, batch_dim=batch_dim, beam_dim=beam_dim).tensor,
        tree.map_structure(
          functools.partial(self._map_tensor_to_raw, batch_dim=batch_dim, beam_dim=beam_dim), decoder_state
        ),
      )

    @staticmethod
    def _map_tensor_to_raw(v, *, batch_dim: Dim, beam_dim: Dim):
      if isinstance(v, Tensor):
        if beam_dim not in v.dims:
          return StateObjIgnored(v)
        batch_dims_ = [batch_dim, beam_dim]
        v = v.copy_transpose(batch_dims_ + [dim for dim in v.dims if dim not in batch_dims_])
        raw = v.raw_tensor
        return StateObjTensorExt(raw, v.copy_template())
      elif isinstance(v, Dim):
        return StateObjIgnored(v)
      else:
        raise TypeError(f"_map_tensor_to_raw: unexpected {v} ({type(v).__name__})")

  return LabelScorer()


def make_time_sync_label_scorer_torch(
        model: TransformerDecoder,
        align_target_dim: Dim,
) -> LabelScorerIntf:
  """
  Make label scorer
  """
  import torch
  import tree
  import functools
  from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.label_scorer import (
    StateObjTensorExt,
    StateObjIgnored,
  )

  class LabelScorer(LabelScorerIntf):
    """TransformerDecoder label scorer"""

    def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
      """Initial state."""
      batch_dim = Dim(batch_size, name="batch")
      beam_dim = Dim(1, name="initial-beam")
      batch_dims_ = [batch_dim, beam_dim]
      decoder_state = model.default_initial_state(batch_dims=batch_dims_)
      return tree.map_structure(
        functools.partial(self._map_tensor_to_raw, batch_dim=batch_dim, beam_dim=beam_dim), decoder_state
      )

    def max_remaining_seq_score(
            self, *, state: Any, max_remaining_steps: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
      """max remaining"""
      return torch.zeros((1, 1), device=device)

    def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
            prev_align_label: Optional[torch.Tensor] = None,
            t: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any]:
      """update state"""
      batch_dim = Dim(prev_label.shape[0], name="batch")
      beam_dim = Dim(prev_label.shape[1], name="beam")

      def _map_raw_to_tensor(v):
        if isinstance(v, StateObjTensorExt):
          tensor: Tensor = v.extra
          tensor = tensor.copy_template_new_dim_tags(
            (batch_dim, beam_dim) + tensor.dims[2:], keep_special_axes=True
          )
          tensor.raw_tensor = v.tensor
          return tensor
        elif isinstance(v, StateObjIgnored):
          return v.content
        else:
          raise TypeError(f"_map_raw_to_tensor: unexpected {v} ({type(v).__name__})")

      initial_output_mask = rf.convert_to_tensor(prev_label == -1, dims=[batch_dim, beam_dim])
      prev_label = rf.convert_to_tensor(prev_label, dims=[batch_dim, beam_dim], sparse_dim=model.vocab_dim)
      # replace -1 by 0 (assuming 0 is the BOS symbol)
      prev_label = rf.where(
        initial_output_mask,
        rf.zeros_like(prev_label),
        prev_label
      )
      logits, decoder_state = model(
        prev_label,
        spatial_dim=single_step_dim,
        state=tree.map_structure(_map_raw_to_tensor, prev_state),
      )
      label_log_prob = rf.log_softmax(logits, axis=model.vocab_dim)
      blank_log_prob = rf.zeros(
        [Dim(1, name="blank_log_prob_label_scorer")],
        dtype="float32"
      )
      output_log_prob, _ = rf.concat(
        (label_log_prob, model.vocab_dim), (blank_log_prob, blank_log_prob.dims[0]),
        out_dim=align_target_dim,
        allow_broadcast=True
      )
      assert set(output_log_prob.dims) == {batch_dim, beam_dim, align_target_dim}

      return (
        self._map_tensor_to_raw(output_log_prob, batch_dim=batch_dim, beam_dim=beam_dim).tensor,
        tree.map_structure(
          functools.partial(self._map_tensor_to_raw, batch_dim=batch_dim, beam_dim=beam_dim), decoder_state
        ),
      )

    @staticmethod
    def _map_tensor_to_raw(v, *, batch_dim: Dim, beam_dim: Dim):
      if isinstance(v, Tensor):
        if beam_dim not in v.dims:
          return StateObjIgnored(v)
        batch_dims_ = [batch_dim, beam_dim]
        v = v.copy_transpose(batch_dims_ + [dim for dim in v.dims if dim not in batch_dims_])
        raw = v.raw_tensor
        return StateObjTensorExt(raw, v.copy_template())
      elif isinstance(v, Dim):
        return StateObjIgnored(v)
      else:
        raise TypeError(f"_map_tensor_to_raw: unexpected {v} ({type(v).__name__})")

  return LabelScorer()


def from_scratch_model_def(*, epoch: int, vocab_dim: Dim) -> TransformerDecoder:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  config = get_global_config()  # noqa

  model_dim = config.typed_value("model_dim", Dim(1024, name="transformer-dec-model-dim"))
  num_layers = config.typed_value("num_layers", 24)
  embed_dim = config.typed_value("embed_dim", Dim(128, name="transformer-dec-embed-dim"))
  decoder_layer_opts = {"self_att_opts": {"with_bias": False, "att_dropout_broadcast": False}}
  input_embedding_scale = config.typed_value("input_embedding_scale", 1.0)
  share_embedding = config.typed_value("share_embedding", False)
  logits_with_bias = config.typed_value("logits_with_bias", True)
  input_dropout = config.typed_value("input_dropout", 0.1)

  return MakeModel.make_model(
    vocab_dim=vocab_dim,
    model_dim=model_dim,
    num_layers=num_layers,
    embed_dim=embed_dim,
    decoder_layer_opts=decoder_layer_opts,
    input_embedding_scale=input_embedding_scale,
    share_embedding=share_embedding,
    logits_with_bias=logits_with_bias,
    input_dropout=input_dropout,
  )


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
  from returnn.tensor import Tensor
  from returnn.config import get_global_config

  config = get_global_config()
  default_target_key = config.typed_value("target")
  extern_data_dict = config.typed_value("extern_data")

  if default_target_key in extern_data_dict:
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
  else:
    non_blank_target_dimension = config.typed_value("non_blank_target_dimension", None)
    vocab = config.typed_value("vocab", None)
    assert non_blank_target_dimension and vocab
    target_dim = Dim(description="non_blank_target_dim", dimension=non_blank_target_dimension, kind=Dim.Types.Spatial)
    targets = Tensor(name=default_target_key, sparse_dim=target_dim, vocab=vocab)

  model_def = config.typed_value("_model_def")
  model = model_def(epoch=epoch, vocab_dim=targets.sparse_dim)
  return model
