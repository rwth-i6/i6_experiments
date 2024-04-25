from typing import Optional, Dict, Any, Sequence, Tuple, List, TYPE_CHECKING, Union
import functools
import tree

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import BaseModel

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

# if TYPE_CHECKING:
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.training import TrainDef
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.recog import RecogDef
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor, _log_mel_feature_dim


class GlobalAttentionModel(BaseModel):
  def __init__(self, eos_idx: int, **kwargs):
    super(GlobalAttentionModel, self).__init__(**kwargs)
    self.eos_idx = eos_idx
    self.bos_idx = eos_idx

  def decoder_default_initial_state(self, *, batch_dims: Sequence[Dim], enc_spatial_dim: Dim) -> rf.State:
    """Default initial state"""
    state = rf.State(
      s=self.s.default_initial_state(batch_dims=batch_dims),
      att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]),
      accum_att_weights=rf.zeros(
        list(batch_dims) + [enc_spatial_dim, self.att_num_heads], feature_dim=self.att_num_heads
      ),
    )
    state.att.feature_dim_axis = len(state.att.dims) - 1
    return state

  def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
    """loop step out"""
    return {
      "s": Tensor(
        "s", dims=batch_dims + [self.s.out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
      ),
      "att": Tensor(
        "att",
        dims=batch_dims + [self.att_num_heads * self.encoder.out_dim],
        dtype=rf.get_default_float_dtype(),
        feature_dim_axis=-1,
      ),
    }

  def loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_ctx: rf.Tensor,
          inv_fertility: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          state: Optional[rf.State] = None,
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    """step of the inner loop"""
    if state is None:
      batch_dims = enc.remaining_dims(
        remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
      )
      state = self.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
    state_ = rf.State()

    prev_att = state.att

    s, state_.s = self.s(rf.concat_features(input_embed, prev_att), state=state.s, spatial_dim=single_step_dim)

    weight_feedback = self.weight_feedback(state.accum_att_weights)
    s_transformed = self.s_transformed(s)
    energy_in = enc_ctx + weight_feedback + s_transformed
    energy = self.energy(rf.tanh(energy_in))
    att_weights = rf.softmax(energy, axis=enc_spatial_dim)

    state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5
    att0 = rf.dot(att_weights, enc, reduce=enc_spatial_dim, use_mask=False)
    att0.feature_dim = self.encoder.out_dim
    att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
    state_.att = att

    return {"s": s, "att": att}, state_

  def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
    """logits for the decoder"""
    readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
    readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
    readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
    logits = self.output_prob(readout)
    return logits


class MakeModel:
  """for import"""

  def __init__(self, in_dim: int, target_dim: int, *, eos_label: int = 0, num_enc_layers: int = 12):
    self.in_dim = in_dim
    self.target_dim = target_dim
    self.eos_label = eos_label
    self.num_enc_layers = num_enc_layers

  def __call__(self) -> GlobalAttentionModel:
    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
    target_dim = Dim(name="target", dimension=self.target_dim, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(
      [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
    )

    return self.make_model(in_dim, target_dim, num_enc_layers=self.num_enc_layers)

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          target_dim: Dim,
          *,
          num_enc_layers: int = 12,
          pos_emb_dropout: float = 0.0,
          language_model: Optional[Dict[str, Any]] = None,
          **extra,
  ) -> GlobalAttentionModel:
    """make"""
    lm = None
    if language_model:
      assert isinstance(language_model, dict)
      language_model = language_model.copy()
      cls_name = language_model.pop("class")
      assert cls_name == "TransformerDecoder"
      language_model.pop("vocab_dim", None)  # will just overwrite

      from . import trafo_lm

      lm = trafo_lm.MakeModel(vocab_dim=target_dim, **language_model)()
      lm = (lm, functools.partial(trafo_lm.make_label_scorer_torch, model=lm))

    return GlobalAttentionModel(
      in_dim=in_dim,
      num_enc_layers=num_enc_layers,
      enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
      enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
      enc_att_num_heads=8,
      enc_conformer_layer_opts=dict(
        conv_norm_opts=dict(use_mask=True),
        self_att_opts=dict(
          # Shawn et al 2018 style, old RETURNN way.
          with_bias=False,
          with_linear_pos=False,
          with_pos_bias=False,
          learnable_pos_emb=True,
          separate_pos_emb_per_head=False,
          pos_emb_dropout=pos_emb_dropout,
        ),
        ff_activation=lambda x: rf.relu(x) ** 2.0,
      ),
      eos_idx=_get_eos_idx(target_dim),
      target_dim=target_dim,
      blank_idx=target_dim.dimension,
      language_model=lm,
      **extra,
    )


def _get_bos_idx(target_dim: Dim) -> int:
  """for non-blank labels"""
  assert target_dim.vocab
  if target_dim.vocab.bos_label_id is not None:
    bos_idx = target_dim.vocab.bos_label_id
  elif target_dim.vocab.eos_label_id is not None:
    bos_idx = target_dim.vocab.eos_label_id
  elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
    bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
  else:
    raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
  return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
  """for non-blank labels"""
  assert target_dim.vocab
  if target_dim.vocab.eos_label_id is not None:
    eos_idx = target_dim.vocab.eos_label_id
  else:
    raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
  return eos_idx


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> GlobalAttentionModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  in_dim, epoch  # noqa
  config = get_global_config()  # noqa
  enc_aux_logits = config.typed_value("aux_loss_layers")
  pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
  # real input is raw audio, internally it does logmel
  in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
  lm_opts = config.typed_value("external_language_model")
  return MakeModel.make_model(
    in_dim, target_dim, enc_aux_logits=enc_aux_logits or (), pos_emb_dropout=pos_emb_dropout, language_model=lm_opts
  )


from_scratch_model_def: ModelDef[GlobalAttentionModel]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


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
  enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
  if aux_loss_layers:
    for i, layer_idx in enumerate(aux_loss_layers):
      if layer_idx > len(model.encoder.layers):
        continue
      linear = getattr(model, f"enc_aux_logits_{layer_idx}")
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
  input_embeddings = model.target_embed(targets)
  input_embeddings = rf.shift_right(input_embeddings, axis=targets_spatial_dim, pad_value=0.0)

  def _body(input_embed: Tensor, state: rf.State):
    new_state = rf.State()
    loop_out_, new_state.decoder = model.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed,
      state=state.decoder,
    )
    return loop_out_, new_state

  loop_out, _, _ = rf.scan(
    spatial_dim=targets_spatial_dim,
    xs=input_embeddings,
    ys=model.loop_step_output_templates(batch_dims=batch_dims),
    initial=rf.State(
      decoder=model.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim),
    ),
    body=_body,
  )

  logits = model.decode_logits(input_embed=input_embeddings, **loop_out)
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


def model_recog(
        *,
        model: GlobalAttentionModel,
        data: Tensor,
        data_spatial_dim: Dim,
        max_seq_len: Optional[int] = None,
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
  assert not model.language_model  # not implemented here. use the pure PyTorch search instead

  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
  beam_size = 12
  length_normalization_exponent = 1.0
  if max_seq_len is None:
    max_seq_len = enc_spatial_dim.get_size_tensor()
  else:
    max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
  print("** max seq len:", max_seq_len.raw_tensor)

  # Eager-mode implementation of beam search.
  # Initial state.
  beam_dim = Dim(1, name="initial-beam")
  batch_dims_ = [beam_dim] + batch_dims
  decoder_state = model.decoder_default_initial_state(batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
  target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
  ended = rf.constant(False, dims=batch_dims_)
  out_seq_len = rf.constant(0, dims=batch_dims_)
  seq_log_prob = rf.constant(0.0, dims=batch_dims_)

  i = 0
  seq_targets = []
  seq_backrefs = []
  while True:
    if i == 0:
      input_embed = rf.zeros(batch_dims_ + [model.target_embed.out_dim], feature_dim=model.target_embed.out_dim)
    else:
      input_embed = model.target_embed(target)
    step_out, decoder_state = model.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=input_embed,
      state=decoder_state,
    )
    logits = model.decode_logits(input_embed=input_embed, **step_out)
    label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
    # Filter out finished beams
    label_log_prob = rf.where(
      ended,
      rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=-1.0e30),
      label_log_prob,
    )
    seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
    seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
      seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.target_dim]
    )  # seq_log_prob, backrefs, target: Batch, Beam
    seq_targets.append(target)
    seq_backrefs.append(backrefs)
    decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), decoder_state)
    ended = rf.gather(ended, indices=backrefs)
    out_seq_len = rf.gather(out_seq_len, indices=backrefs)
    i += 1

    ended = rf.logical_or(ended, target == model.eos_idx)
    ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
    if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
      break
    out_seq_len = out_seq_len + rf.where(ended, 0, 1)

    if i > 1 and length_normalization_exponent != 0:
      # Length-normalized scores, so we evaluate score_t/len.
      # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
      # Because we count with EOS symbol, shifted by one.
      seq_log_prob *= rf.where(
        ended,
        (i / (i - 1)) ** length_normalization_exponent,
        1.0,
      )

  if i > 0 and length_normalization_exponent != 0:
    seq_log_prob *= (1 / i) ** length_normalization_exponent

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
  out_spatial_dim = Dim(out_seq_len, name="out-spatial")
  seq_targets = seq_targets__.stack(axis=out_spatial_dim)

  return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[GlobalAttentionModel]
model_recog.output_with_beam = True
# output_blank_label=blank is actually wrong for AED, but now we don't change it anymore
# because it would change all recog hashes.
# Also, it does not matter too much -- it will just cause an extra SearchRemoveLabelJob,
# which will not have any effect here.
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False
