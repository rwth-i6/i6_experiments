from typing import Optional, Dict, Any, Sequence, Tuple, List, Union
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import (
  _batch_size_factor,
  get_common_config_params,
  apply_weight_dropout,
)

from .base import BaseLabelDecoder


class GlobalAttDecoder(BaseLabelDecoder):
  def __init__(self, eos_idx: int, **kwargs):
    from returnn.config import get_global_config
    config = get_global_config()
    self.replace_att_by_h_s = config.bool("replace_att_by_h_s", False)
    if not kwargs.get("use_hard_attention") and self.replace_att_by_h_s:
      kwargs["use_hard_attention"] = True

    super(GlobalAttDecoder, self).__init__(**kwargs)

    self.eos_idx = eos_idx
    self.bos_idx = eos_idx

  def decoder_default_initial_state(
          self,
          *,
          batch_dims: Sequence[Dim],
          enc_spatial_dim: Dim,
          use_mini_att: bool = False,
          use_zero_att: bool = False,
  ) -> rf.State:
    """Default initial state"""
    state = rf.State()

    if self.trafo_att and not use_mini_att and not use_zero_att:
      state.trafo_att = self.trafo_att.default_initial_state(batch_dims=batch_dims)
      # att_dim = self.trafo_att.model_dim
    # else:
      # att_dim = self.att_num_heads * self.enc_out_dim

    state.att = rf.zeros(list(batch_dims) + [self.att_dim])
    state.att.feature_dim_axis = len(state.att.dims) - 1

    if "lstm" in self.decoder_state:
      state.s = self.get_lstm().default_initial_state(batch_dims=batch_dims)
      if use_mini_att:
        state.mini_att_lstm = self.mini_att_lstm.default_initial_state(batch_dims=batch_dims)

    if self.use_weight_feedback and not use_mini_att and not use_zero_att:
      state.accum_att_weights = rf.zeros(
        list(batch_dims) + [enc_spatial_dim, self.att_num_heads], feature_dim=self.att_num_heads
      )

    return state

  def loop_step_output_templates(
          self,
          batch_dims: List[Dim],
  ) -> Dict[str, Tensor]:
    """loop step out"""
    output_templates = {
      "s": Tensor(
        "s", dims=batch_dims + [self.get_lstm().out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
      ),
      "att": Tensor(
        "att",
        dims=batch_dims + [self.att_num_heads * self.enc_out_dim],
        dtype=rf.get_default_float_dtype(),
        feature_dim_axis=-1,
      ),
    }
    return output_templates

  def loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_ctx: rf.Tensor,
          inv_fertility: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          state: Optional[rf.State] = None,
          use_mini_att: bool = False,
          use_zero_att: bool = False,
          hard_att_opts: Optional[Dict] = None,
          mask_att_opts: Optional[Dict] = None,
          detach_att: bool = False,
          h_s: Optional[rf.Tensor] = None,
          set_prev_att_to_zero: bool = False,
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    """step of the inner loop"""
    if state is None:
      batch_dims = enc.remaining_dims(
        remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
      )
      state = self.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
    state_ = rf.State()

    output_dict = {}

    prev_att = state.att
    if detach_att:
      prev_att = rf.stop_gradient(prev_att)
    if set_prev_att_to_zero:
      prev_att = rf.zeros_like(prev_att)
    prev_s_state = state.s if "lstm" in self.decoder_state else None

    input_embed = rf.dropout(input_embed, drop_prob=self.target_embed_dropout, axis=None)
    s, s_state = self._update_state(input_embed, prev_att, prev_s_state)
    output_dict["s"] = s
    if "lstm" in self.decoder_state:
      state_.s = s_state

    if h_s is not None:
      att = h_s
    elif use_mini_att:
      if "lstm" in self.decoder_state:
        att_lstm, state_.mini_att_lstm = self.mini_att_lstm(
          input_embed, state=state.mini_att_lstm, spatial_dim=single_step_dim)
        pre_mini_att = att_lstm
      else:
        att_linear = self.mini_att_linear(input_embed)
        pre_mini_att = att_linear
      att = self.mini_att(pre_mini_att)
    elif use_zero_att:
      att = rf.zeros_like(prev_att)
    else:
      if self.trafo_att:
        att, state_.trafo_att = self.trafo_att(
          input_embed,
          state=state.trafo_att,
          spatial_dim=single_step_dim,
          encoder=None if self.use_trafo_att_wo_cross_att else self.trafo_att.transform_encoder(enc, axis=enc_spatial_dim)
        )
      else:
        if self.use_weight_feedback:
          weight_feedback = self.weight_feedback(state.accum_att_weights)
        else:
          weight_feedback = rf.zeros((self.enc_key_total_dim,))

        if hard_att_opts is None or rf.get_run_ctx().epoch > hard_att_opts["until_epoch"]:
          s_transformed = self.s_transformed(s)
          energy_in = enc_ctx + weight_feedback + s_transformed

          energy = self.energy(rf.tanh(energy_in))

          # set energies to -inf for the given frame indices
          # e.g. we use this to mask the frames around h_t when we have h_t in the readout in order to force
          # the attention to focus on the other frames
          if mask_att_opts is not None:
            enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext)
            mask_frame_idx = mask_att_opts["frame_idx"]  # [B]
            # just some heuristic
            # i.e. we mask:
            # 1 frame, if there are less than 6 frames
            # 3 frames, if there are less than 9 frames
            # 5 frames, otherwise
            mask_dimension = (enc_spatial_sizes // 3) * 2 - 1
            mask_dimension = rf.clip_by_value(mask_dimension, 1, 5)
            mask_dim = Dim(dimension=mask_dimension, name="att_mask")  # [5]
            # example mask: [-inf, -inf, -inf, 0, 0]
            mask_range = rf.range_over_dim(mask_dim)
            mask = rf.where(
              mask_range < mask_dimension,
              float("-inf"),
              0.0
            )
            # move mask to correct position along the time axis
            mask_indices = mask_range + mask_frame_idx - (mask_dimension // 2)  # [B, 5]
            mask_indices.sparse_dim = enc_spatial_dim
            mask_indices = rf.clip_by_value(
              mask_indices,
              0,
              enc_spatial_sizes - 1
            )
            mask = rf.scatter(
              mask,
              indices=mask_indices,
              indices_dim=mask_dim,
            )  # [B, T]
            energy = energy + mask

          att_weights = rf.softmax(energy, axis=enc_spatial_dim)
          att_weights = rf.dropout(att_weights, drop_prob=self.att_weight_dropout, axis=None)
        if hard_att_opts is not None and rf.get_run_ctx().epoch <= hard_att_opts["until_epoch"] + hard_att_opts["num_interpolation_epochs"]:
          if hard_att_opts["frame"] == "middle":
            frame_idx = rf.copy_to_device(enc_spatial_dim.dyn_size_ext) // 2
          else:
            frame_idx = rf.constant(hard_att_opts["frame"], dims=enc_spatial_dim.dyn_size_ext.dims)
          frame_idx.sparse_dim = enc_spatial_dim
          one_hot = rf.one_hot(frame_idx)
          one_hot = rf.expand_dim(one_hot, dim=self.att_num_heads)

          if rf.get_run_ctx().epoch <= hard_att_opts["until_epoch"] and hard_att_opts["frame"] == "middle":
            att_weights = one_hot
          else:
            interpolation_factor = (rf.get_run_ctx().epoch - hard_att_opts["until_epoch"]) / hard_att_opts["num_interpolation_epochs"]
            att_weights = (1 - interpolation_factor) * one_hot + interpolation_factor * att_weights

        if self.use_weight_feedback:
          state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5

        att = self.get_att(att_weights, enc, enc_spatial_dim)

    state_.att = att
    output_dict["att"] = att

    return output_dict, state_


class GlobalAttentionModel(rf.Module):
  def __init__(
          self,
          *,
          target_dim: Dim,
          blank_idx: int,
          enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
          att_dropout: float = 0.1,
          l2: float = 0.0001,
          language_model: Optional[rf.Module] = None,
          enc_in_dim: Dim,
          enc_out_dim: Dim = Dim(name="enc", dimension=512),
          enc_num_layers: int = 12,
          enc_aux_logits: Sequence[int] = (),  # layers
          enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
          enc_num_heads: int = 4,
          encoder_layer_opts: Optional[Dict[str, Any]] = None,
          encoder_layer: Optional[Union[ConformerEncoderLayer, rf.Module, type, Dict[str, Any], Any]] = None,
          encoder_cls: rf.Module = GlobalConformerEncoder,
          enc_input_layer_cls: rf.Module = ConformerConvSubsample,
          dec_att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
          enc_dropout: float = 0.1,
          eos_idx: int,
          bos_idx: int,
          use_weight_feedback: bool = True,
          use_att_ctx_in_state: bool = True,
          use_mini_att: bool = False,
          label_decoder_state: str = "nb-lstm",
          num_dec_layers: int = 1,
          target_embed_dim: int = 640,
          readout_dimension: int = 1024,
          ilm_dimension: int = 1024,
          feature_extraction_opts: Optional[Dict[str, Any]] = None,
          target_embed_dropout: float = 0.0,
          att_weight_dropout: float = 0.0,
          use_readout: bool = True,
          enc_ctx_layer: Optional[str] = None,
          external_aed_kwargs: Optional[Dict[str, Any]] = None,
  ):
    super(GlobalAttentionModel, self).__init__()

    self.bos_idx = bos_idx
    self.eos_idx = eos_idx

    if encoder_cls == LinearEncoder:
      self.encoder = encoder_cls(
        enc_in_dim,
        enc_out_dim,
        num_layers=enc_num_layers,
        enc_key_total_dim=enc_key_total_dim,
        dec_att_num_heads=dec_att_num_heads,
        decoder_type="trafo" if label_decoder_state == "trafo" else "lstm",
        feature_extraction_opts=feature_extraction_opts,
        enc_ctx_layer=enc_ctx_layer,
      )
    else:
      self.encoder = encoder_cls(
        enc_in_dim,
        enc_out_dim,
        num_layers=enc_num_layers,
        target_dim=target_dim,
        wb_target_dim=None,
        aux_logits=enc_aux_logits,
        ff_dim=enc_ff_dim,
        num_heads=enc_num_heads,
        encoder_layer_opts=encoder_layer_opts,
        enc_key_total_dim=enc_key_total_dim,
        dec_att_num_heads=dec_att_num_heads,
        dropout=enc_dropout,
        att_dropout=att_dropout,
        l2=l2,
        decoder_type="trafo" if label_decoder_state == "trafo" else "lstm",
        feature_extraction_opts=feature_extraction_opts,
        encoder_layer=encoder_layer,
        input_layer_cls=enc_input_layer_cls,
        enc_ctx_layer=enc_ctx_layer,
      )

    self.decoder_state = label_decoder_state
    if label_decoder_state != "trafo":
      assert num_dec_layers == 1

      if not use_weight_feedback and not use_att_ctx_in_state:
        decoder_cls = GlobalAttEfficientDecoder
      else:
        decoder_cls = GlobalAttDecoder

      self.label_decoder = decoder_cls(
        enc_out_dim=self.encoder.out_dim,
        target_dim=target_dim,
        att_num_heads=dec_att_num_heads,
        att_dropout=att_dropout,
        blank_idx=blank_idx,
        enc_key_total_dim=enc_key_total_dim,
        l2=l2,
        eos_idx=eos_idx,
        use_weight_feedback=use_weight_feedback,
        use_att_ctx_in_state=use_att_ctx_in_state,
        use_mini_att=use_mini_att,
        decoder_state=label_decoder_state,
        target_embed_dim=Dim(name="target_embed", dimension=target_embed_dim),
        target_embed_dropout=target_embed_dropout,
        att_weight_dropout=att_weight_dropout,
        readout_dimension=readout_dimension,
        ilm_dimension=ilm_dimension,
        use_readout=use_readout,
      )
    else:
      # hard code for now
      self.eos_idx = eos_idx
      self.bos_idx = bos_idx

      model_dim = Dim(name="dec", dimension=512)
      self.label_decoder = TransformerDecoder(
        num_layers=num_dec_layers,
        encoder_dim=self.encoder.out_dim,
        vocab_dim=target_dim,
        model_dim=model_dim,
        sequential=rf.Sequential,
        share_embedding=True,
        input_embedding_scale=model_dim.dimension**0.5
      )

    if language_model:
      self.language_model = language_model
    else:
      self.language_model = None

    self.blank_idx = blank_idx
    self.target_dim = target_dim

    if use_mini_att:
      for name, param in self.named_parameters():
        if "mini_att" not in name:
          param.trainable = False

    if external_aed_kwargs:
      self.aed_model = GlobalAttentionModel(
        enc_in_dim=enc_in_dim,
        enc_ff_dim=enc_ff_dim,
        enc_key_total_dim=enc_key_total_dim,
        enc_num_heads=8,
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        feature_extraction_opts=feature_extraction_opts,
        eos_idx=0,
        bos_idx=0,
        enc_aux_logits=(),
        encoder_layer_opts=encoder_layer_opts,
      )
    else:
      self.aed_model = None

    apply_weight_dropout(self)


class MakeModel:
  """for import"""

  def __init__(
          self,
          in_dim: int,
          target_dim: int,
          *,
          eos_label: int = 0,
          num_enc_layers: int = 12,
          enc_aux_logits: Sequence[int] = (),
          target_embed_dim: int = 640,
  ):
    self.in_dim = in_dim
    self.target_dim = target_dim
    self.eos_label = eos_label
    self.num_enc_layers = num_enc_layers

    # do not set attribute otherwise to keep job hashes
    if enc_aux_logits != ():
      self.enc_aux_logits = enc_aux_logits

    if target_embed_dim != 640:
      self.target_embed_dim = target_embed_dim

  def __call__(self) -> GlobalAttentionModel:
    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
    target_dim = Dim(name="target", dimension=self.target_dim, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(
      [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
    )

    extra = {}
    if hasattr(self, "enc_aux_logits"):
      extra["enc_aux_logits"] = self.enc_aux_logits

    if hasattr(self, "target_embed_dim"):
      extra["target_embed_dim"] = self.target_embed_dim

    return self.make_model(in_dim, target_dim, num_enc_layers=self.num_enc_layers, **extra)

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          target_dim: Dim,
          *,
          language_model: Optional[Dict[str, Any]] = None,
          use_weight_feedback: bool = True,
          use_att_ctx_in_state: bool = True,
          use_mini_att: bool = False,
          label_decoder_state: str = "nb-lstm",
          num_dec_layers: int = 1,
          target_embed_dim: int = 640,
          feature_extraction_opts: Optional[Dict[str, Any]] = None,
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

    return GlobalAttentionModel(
      enc_in_dim=in_dim,
      enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
      enc_num_heads=8,
      eos_idx=_get_eos_idx(target_dim),
      bos_idx=_get_bos_idx(target_dim),
      target_dim=target_dim,
      blank_idx=target_dim.dimension,
      language_model=lm,
      use_weight_feedback=use_weight_feedback,
      use_att_ctx_in_state=use_att_ctx_in_state,
      use_mini_att=use_mini_att,
      label_decoder_state=label_decoder_state,
      num_dec_layers=num_dec_layers,
      target_embed_dim=target_embed_dim,
      feature_extraction_opts=feature_extraction_opts,
      **extra,
    )


def _get_bos_idx(target_dim: Dim) -> int:
  """for non-blank labels"""
  assert target_dim.vocab
  if target_dim.vocab.bos_label_id is not None:
    bos_idx = target_dim.vocab.bos_label_id
  elif target_dim.vocab.eos_label_id is not None:
    bos_idx = target_dim.vocab.eos_label_id
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
  num_dec_layers = config.int("num_label_decoder_layers", 1)
  enc_ctx_layer = config.typed_value("enc_ctx_layer", None)

  common_config_params = get_common_config_params()
  in_dim = common_config_params.pop("in_dim")

  return MakeModel.make_model(
    in_dim,
    target_dim,
    num_dec_layers=num_dec_layers,
    enc_ctx_layer=enc_ctx_layer,
    **common_config_params,
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

  if default_target_key in extern_data_dict:
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
  else:
    non_blank_target_dimension = config.typed_value("non_blank_target_dimension", None)
    vocab = config.typed_value("vocab", None)
    assert non_blank_target_dimension and vocab
    target_dim = Dim(description="non_blank_target_dim", dimension=non_blank_target_dimension, kind=Dim.Types.Spatial)
    targets = Tensor(name=default_target_key, sparse_dim=target_dim, vocab=vocab)

  model_def = config.typed_value("_model_def")
  model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
  return model
