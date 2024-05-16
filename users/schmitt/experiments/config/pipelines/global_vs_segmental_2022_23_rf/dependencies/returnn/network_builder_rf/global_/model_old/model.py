from typing import Optional, Dict, Any, Sequence, Tuple, List
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base_old import BaseModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base_old import _batch_size_factor, _log_mel_feature_dim


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

      from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.lm.trafo import model as trafo_lm

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
  lm_opts = config.typed_value("external_lm")
  return MakeModel.make_model(
    in_dim, target_dim, enc_aux_logits=enc_aux_logits or (), pos_emb_dropout=pos_emb_dropout, language_model=lm_opts
  )


from_scratch_model_def: ModelDef[GlobalAttentionModel]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
  from returnn.tensor import Tensor
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  extern_data_dict = config.typed_value("extern_data")
  data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
  targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])

  model_def = config.typed_value("_model_def")
  model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
  return model
