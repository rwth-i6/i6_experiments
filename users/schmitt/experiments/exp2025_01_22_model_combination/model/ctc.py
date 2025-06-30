from typing import Dict, Sequence, Optional, Tuple
import numpy as np

import returnn.frontend as rf
from returnn.frontend import Dim, Tensor
from returnn.frontend.tensor_array import TensorArray

from .aed import get_conformer, Model

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils


def load_missing_params(name, shape, preload_model_state):
  if name.startswith("output_layer"):
    return preload_model_state[f"enc_aux_logits_12{name[len('output_layer'):]}"]

  return None


class CTCModel(Model):
  def __init__(
          self,
          *,
          encoder_opts: Dict,
          in_dim: Dim,
          target_dim: Dim,
          feature_extraction: Optional[str] = "log-mel-on-the-fly",
  ):
    super(CTCModel, self).__init__(
      encoder_opts=encoder_opts,
      in_dim=in_dim,
      feature_extraction=feature_extraction,
    )

    from returnn.config import get_global_config
    config = get_global_config(return_empty_if_none=True)

    self.in_dim = in_dim
    # for CTC aux loss
    self.blank_idx = target_dim.dimension  # type: int
    self.target_dim = target_dim
    self.feature_extraction = feature_extraction

    # encoder
    encoder_opts.update({"in_dim": in_dim})
    self.encoder = get_conformer(encoder_opts)

    self.wb_target_dim = target_dim + 1
    self.output_layer = rf.Linear(self.encoder.out_dim, self.wb_target_dim)

    self.ctc_am_scale = config.float("ctc_am_scale", 1.0)
    self.ctc_prior_scale = config.float("ctc_prior_scale", 0.0)
    self.ctc_prior_type = config.value("ctc_prior_type", "batch")

    static_prior = config.typed_value("static_prior")
    self.static_prior = None  # in log prob, if set
    if static_prior:
      assert isinstance(static_prior, dict)
      assert set(static_prior.keys()) == {"file", "type"}
      v = np.loadtxt(static_prior["file"])
      # The `type` is about what is stored in the file.
      # We always store it in log prob here, so we potentially need to convert it.
      if static_prior["type"] == "log_prob":
        pass  # already log prob
      elif static_prior["type"] == "prob":
        v = np.log(v)
      else:
        raise ValueError(f"invalid static_prior type {static_prior['type']!r}")
      self.static_prior = rf.Parameter(
        rf.convert_to_tensor(v, dims=[self.wb_target_dim], dtype=rf.get_default_float_dtype()),
        auxiliary=True,
        non_critical_for_restore=True,
      )

  def __call__(
          self,
          source: Tensor,
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Tensor, Tensor, Dim]:
    """
    Encode, get CTC logits.
    Use :func:`log_probs_wb_from_logits` to get log probs
    (might be just log_softmax, but there are some other cases).

    :return: logits, enc, enc_spatial_dim
    """
    enc, enc_spatial_dim = super(CTCModel, self).encode(
      source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
    )
    logits = self.output_layer(enc)

    return logits, enc, enc_spatial_dim


  def log_probs_wb_from_logits(self, logits: Tensor, *, aux_layer: Optional[int] = None) -> Tensor:
    """
    :param logits: incl blank
    :param aux_layer: whether the logits come from some intermediate aux layer.
        That might influence the prior.
    :return: log probs with blank from logits (wb_target_dim)
        If out_blank_separated, we use a separate sigmoid for the blank.
        Also, potentially adds label smoothing on the gradients.
    """
    log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)

    if self.ctc_am_scale == 1 and self.ctc_prior_scale == 0:  # fast path
      return log_probs
    log_probs_am = log_probs
    log_probs = log_probs_am * self.ctc_am_scale
    if self.ctc_prior_scale:
      if self.ctc_prior_type == "batch":
        # Warning: this is sum, but we want mean!
        log_prob_prior = rf.reduce_logsumexp(
          log_probs_am, axis=[dim for dim in log_probs_am.dims if dim != self.wb_target_dim]
        )
        assert log_prob_prior.dims == (self.wb_target_dim,)
      elif self.ctc_prior_type == "static":
        log_prob_prior = self.static_prior
        assert log_prob_prior.dims == (self.wb_target_dim,)
      else:
        raise ValueError(f"invalid ctc_prior_type {self.ctc_prior_type!r}")

      log_probs -= log_prob_prior * self.ctc_prior_scale
    return log_probs


def ctc_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> CTCModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  in_dim, epoch  # noqa
  config = get_global_config()  # noqa
  encoder_opts = config.typed_value("encoder_opts")

  feature_extraction = config.typed_value("feature_extraction", "log-mel-on-the-fly")
  feature_dimension = config.int("feature_dim", 80)
  if feature_extraction is None:
    feature_dim = config.typed_value("features_dim")
    in_dim = feature_dim
  else:
    assert feature_extraction == "log-mel-on-the-fly"
    in_dim = Dim(name="logmel", dimension=feature_dimension, kind=Dim.Types.Feature)

  return CTCModel(
    encoder_opts=encoder_opts,
    target_dim=target_dim,
    feature_extraction=feature_extraction,
    in_dim=in_dim,
  )


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
  from returnn.tensor import Tensor
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  extern_data_dict = config.typed_value("extern_data")
  data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])

  if default_target_key in extern_data_dict:
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
  elif "data_flat" in extern_data_dict:
    targets = Tensor(name=default_target_key, **extern_data_dict["data_flat"])
  else:
    non_blank_target_dimension = config.typed_value("non_blank_target_dimension", None)
    vocab = config.typed_value("vocab", None)
    assert non_blank_target_dimension and vocab
    target_dim = Dim(description="non_blank_target_dim", dimension=non_blank_target_dimension, kind=Dim.Types.Spatial)
    targets = Tensor(name=default_target_key, sparse_dim=target_dim, vocab=vocab)

  model_def = config.typed_value("_model_def")
  model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
  return model


def model_recog(
        *,
        model: CTCModel,
        data: Tensor,
        data_spatial_dim: Dim,
        beam_size: int,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
  """
  Function is run within RETURNN.

  Earlier we used the generic beam_search function,
  but now we just directly perform the search here,
  as this is overall simpler and shorter.

  :return:
      recog results including beam {batch, beam, out_spatial},
      log probs {batch, beam},
      out_spatial_dim,
      final beam_dim
  """
  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs={})

  # Eager-mode implementation of beam search.
  # Initial state.
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

  label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, Vocab
  label_log_prob = rf.where(
    enc_spatial_dim.get_mask(),
    label_log_prob,
    rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
  )
  label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
    label_log_prob, k_dim=Dim(beam_size, name=f"pre-filter-beam"), axis=[model.wb_target_dim]
  )  # seq_log_prob, backrefs_global: Batch, Spatial, PreFilterBeam. backrefs_pre_filter -> Vocab
  label_log_prob_pre_filter_ta = TensorArray.unstack(
    label_log_prob_pre_filter, axis=enc_spatial_dim
  )  # t -> Batch, PreFilterBeam
  backrefs_pre_filter_ta = TensorArray.unstack(backrefs_pre_filter, axis=enc_spatial_dim)  # t -> Batch, PreFilterBeam

  max_seq_len = int(enc_spatial_dim.get_dim_value())
  seq_targets = []
  seq_backrefs = []
  for t in range(max_seq_len):
    # Filter out finished beams
    seq_log_prob = seq_log_prob + label_log_prob_pre_filter_ta[t]  # Batch, InBeam, PreFilterBeam
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, pre_filter_beam_dim]
    )  # seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> PreFilterBeam.
    target = rf.gather(backrefs_pre_filter_ta[t], indices=target)  # Batch, Beam -> Vocab
    seq_targets.append(target)
    seq_backrefs.append(backrefs)

  # Backtrack via backrefs, resolve beams.
  seq_targets_ = []
  indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
  for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
    # indices: FinalBeam -> Beam
    # backrefs: Beam -> PrevBeam
    seq_targets_.insert(0, rf.gather(target, indices=indices))
    indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

  seq_targets__ = TensorArray(seq_targets_[0])
  for target in seq_targets_:
    seq_targets__ = seq_targets__.push_back(target)
  out_spatial_dim = enc_spatial_dim
  seq_targets = seq_targets__.stack(axis=out_spatial_dim)

  non_blank_targets, non_blank_targets_spatial_dim = utils.get_masked(
    seq_targets,
    utils.get_non_blank_mask(seq_targets, model.blank_idx),
    enc_spatial_dim,
    [beam_dim] + batch_dims,
  )
  non_blank_targets.sparse_dim = model.target_dim

  return (
    seq_targets,
    seq_log_prob,
    out_spatial_dim,
    beam_dim,
  )


def ctc_model_rescore(
        *,
        model: CTCModel,
        data: Tensor,
        data_spatial_dim: Dim,
        targets: Tensor,
        targets_beam_dim: Dim,
        targets_spatial_dim: Dim,
) -> Tensor:
  """RescoreDef API"""
  import returnn.frontend as rf
  from returnn.tensor import Tensor, Dim

  logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
  assert isinstance(logits, Tensor) and isinstance(enc_spatial_dim, Dim)
  assert logits.feature_dim  # we expect a feature dim
  assert enc_spatial_dim in logits.dims
  log_probs = model.log_probs_wb_from_logits(logits)
  assert isinstance(log_probs, Tensor)

  batch_dims = targets.remaining_dims(targets_spatial_dim)

  # Note: Using ctc_loss directly requires quite a lot of memory,
  # as we would broadcast the log_probs over the beam dim.
  # Instead, we do a loop over the beam dim to avoid this.
  neg_log_prob_ = []
  for beam_idx in range(targets_beam_dim.get_dim_value()):
    targets_b = rf.gather(targets, axis=targets_beam_dim, indices=beam_idx)
    targets_b_seq_lens = rf.gather(targets_spatial_dim.dyn_size_ext, axis=targets_beam_dim, indices=beam_idx)
    targets_b_spatial_dim = Dim(targets_b_seq_lens, name=f"{targets_spatial_dim.name}_beam{beam_idx}")
    targets_b, _ = rf.replace_dim(targets_b, in_dim=targets_spatial_dim, out_dim=targets_b_spatial_dim)
    targets_b, _ = rf.slice(targets_b, axis=targets_b_spatial_dim, size=targets_b_spatial_dim)
    # Note: gradient does not matter (not used), thus no need use our ctc_loss_fixed_grad.
    neg_log_prob = rf.ctc_loss(
      logits=log_probs,
      logits_normalized=True,
      targets=targets_b,
      input_spatial_dim=enc_spatial_dim,
      targets_spatial_dim=targets_b_spatial_dim,
      blank_index=model.blank_idx,
    )
    neg_log_prob_.append(neg_log_prob)
  neg_log_prob, _ = rf.stack(neg_log_prob_, out_dim=targets_beam_dim)
  log_prob_targets_seq = -neg_log_prob
  assert log_prob_targets_seq.dims_set == set(batch_dims)
  return log_prob_targets_seq
