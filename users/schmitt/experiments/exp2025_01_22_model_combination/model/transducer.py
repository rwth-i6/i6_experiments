from typing import Optional, Dict, Any, Sequence, Tuple, List, Union
import functools
import copy

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from .base import _batch_size_factor, get_common_config_params, apply_weight_dropout, BaseLabelDecoder


class SegmentalAttentionModel(rf.Module):
  def __init__(
          self,
          *,
          length_model_state_dim: Dim,
          length_model_embed_dim: Dim,
          center_window_size: int,
          window_step_size: int,
          use_vertical_transitions: bool,
          align_target_dim: Dim,
          target_dim: Dim,
          blank_idx: int,
          enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
          att_dropout: float = 0.1,
          l2: float = 0.0001,
          external_aed_model_kwargs: Optional[Dict] = None,
          external_transducer_kwargs: Optional[Dict] = None,
          enc_in_dim: Dim,
          enc_out_dim: Dim = Dim(name="enc", dimension=512),
          enc_num_layers: int = 12,
          enc_aux_logits: Sequence[int] = (),  # layers
          enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
          enc_num_heads: int = 4,
          encoder_layer_opts: Optional[Dict[str, Any]] = None,
          dec_att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
          enc_dropout: float = 0.1,
          use_att_ctx_in_state: bool = True,
          blank_decoder_version: int = 1,
          use_joint_model: bool = False,
          use_weight_feedback: bool = True,
          label_decoder_state: str = "nb-lstm",
          use_mini_att: bool = False,
          gaussian_att_weight_opts: Optional[Dict[str, Any]] = None,
          separate_blank_from_softmax: bool = False,
          reset_eos_params: bool = False,
          blank_decoder_opts: Optional[Dict[str, Any]] = None,
          use_current_frame_in_readout: bool = False,
          use_current_frame_in_readout_w_gate: bool = False,
          use_current_frame_in_readout_random: bool = False,
          target_embed_dim: int = 640,
          feature_extraction_opts: Optional[Dict[str, Any]] = None,
          trafo_decoder_opts: Optional[Dict[str, Any]] = None,
          target_embed_dropout: float = 0.0,
          att_weight_dropout: float = 0.0,
          use_trafo_att: bool = False,
          readout_dimension: int = 1024,
          ilm_dimension: int = 1024,
          use_readout: bool = True,
  ):
    super(SegmentalAttentionModel, self).__init__()

    from returnn.config import get_global_config
    config = get_global_config()

    self.encoder = encoder_cls(
      enc_in_dim,
      enc_out_dim,
      num_layers=enc_num_layers,
      target_dim=target_dim,
      wb_target_dim=align_target_dim,
      aux_logits=enc_aux_logits,
      ff_dim=enc_ff_dim,
      num_heads=enc_num_heads,
      encoder_layer_opts=encoder_layer_opts,
      encoder_layer=encoder_layer,
      enc_key_total_dim=enc_key_total_dim,
      dec_att_num_heads=dec_att_num_heads,
      dropout=enc_dropout,
      att_dropout=att_dropout,
      l2=l2,
      use_weight_feedback=use_weight_feedback,
      feature_extraction_opts=feature_extraction_opts,
      decoder_type="trafo" if label_decoder_state == "trafo" else "lstm",
      need_enc_ctx=center_window_size != 1 and not use_trafo_att,  # win size 1 means hard att, so no enc ctx needed
      input_layer_cls=enc_input_layer_cls,
    )

    if config.bool("use_sep_att_encoder", False):
      self.att_encoder = copy.deepcopy(self.encoder)
    else:
      self.att_encoder = None

    assert blank_decoder_version in {1, 3, 4, 5, 6, 7, 8, 9, 10, 11} or (
            blank_decoder_version is None and use_joint_model)
    assert label_decoder_state in {"nb-lstm", "joint-lstm", "nb-2linear-ctx1", "trafo"}
    if not use_joint_model:
      assert label_decoder_state in ("nb-lstm", "nb-2linear-ctx1", "trafo")
      assert not separate_blank_from_softmax, "only makes sense when jointly modeling blank and non-blanks"

    # label decoder
    # if center_window_size is None, use global attention decoder
    if center_window_size is None:
      if label_decoder_state == "trafo":
        self.bos_idx = 1
        assert not language_model, "need to change bos_idx in recog implementation"

        model_dim = Dim(name="dec", dimension=512)
        if trafo_decoder_opts is None:
          decoder_layer_opts = None
          trafo_opts = {
            "share_embedding": True,
            "input_embedding_scale": model_dim.dimension ** 0.5,
          }
        else:
          assert trafo_decoder_opts["self_att_type"] == "rel_pos_causal_self_attention"
          decoder_layer_opts = {
            "self_att": RelPosCausalSelfAttention,
            "att_lin_with_bias": False,
            "self_att_opts": {
              "with_pos_bias": False,
              "with_linear_pos": False,
              "learnable_pos_emb": True,
              "separate_pos_emb_per_head": False,
            }
          }
          trafo_opts = {
            "logits_with_bias": True,
            "share_embedding": False,
            "input_embedding_scale": 1.0,
            "use_sin_pos_enc": False,
          }

        self.label_decoder = TransformerDecoder(
          num_layers=6,
          encoder_dim=self.encoder.out_dim,
          vocab_dim=target_dim,
          model_dim=model_dim,
          sequential=rf.Sequential,
          decoder_layer_opts=decoder_layer_opts,
          **trafo_opts,
        )
      else:
        self.bos_idx = 0
        if not use_weight_feedback and not use_att_ctx_in_state:
          label_decoder_cls = GlobalAttEfficientDecoder
        else:
          assert not use_trafo_att, "Trafo attention needs GlobalAttEfficientDecoder"
          label_decoder_cls = GlobalAttDecoder

        self.label_decoder = label_decoder_cls(
          enc_out_dim=self.encoder.out_dim,
          target_dim=target_dim,
          att_num_heads=dec_att_num_heads,
          att_dropout=att_dropout,
          blank_idx=blank_idx,
          enc_key_total_dim=enc_key_total_dim,
          l2=l2,
          use_weight_feedback=use_weight_feedback,
          use_att_ctx_in_state=use_att_ctx_in_state,
          decoder_state=label_decoder_state,
          use_mini_att=use_mini_att,
          separate_blank_from_softmax=separate_blank_from_softmax,
          reset_eos_params=reset_eos_params,
          use_current_frame_in_readout=use_current_frame_in_readout,
          use_current_frame_in_readout_w_gate=use_current_frame_in_readout_w_gate,
          use_current_frame_in_readout_random=use_current_frame_in_readout_random,
          target_embed_dim=Dim(name="target_embed", dimension=target_embed_dim),
          eos_idx=blank_idx,
          target_embed_dropout=target_embed_dropout,
          att_weight_dropout=att_weight_dropout,
          use_trafo_attention=use_trafo_att,
          readout_dimension=readout_dimension,
          ilm_dimension=ilm_dimension,
          use_readout=use_readout,
        )
    else:
      if label_decoder_state == "trafo" or use_trafo_att:
        raise NotImplementedError("Trafo decoder and attention not implemented for segmental model")
      else:
        self.bos_idx = 0

        if not use_weight_feedback and not use_att_ctx_in_state:
          label_decoder_cls = SegmentalAttEfficientLabelDecoder
        else:
          label_decoder_cls = SegmentalAttLabelDecoder

        self.label_decoder = label_decoder_cls(
          enc_out_dim=self.encoder.out_dim,
          target_dim=target_dim,
          att_num_heads=dec_att_num_heads,
          att_dropout=att_dropout,
          blank_idx=blank_idx,
          enc_key_total_dim=enc_key_total_dim,
          l2=l2,
          center_window_size=center_window_size,
          use_weight_feedback=use_weight_feedback,
          use_att_ctx_in_state=use_att_ctx_in_state,
          decoder_state=label_decoder_state,
          use_mini_att=use_mini_att,
          gaussian_att_weight_opts=gaussian_att_weight_opts,
          separate_blank_from_softmax=separate_blank_from_softmax,
          reset_eos_params=reset_eos_params,
          use_current_frame_in_readout=use_current_frame_in_readout,
          use_current_frame_in_readout_w_gate=use_current_frame_in_readout_w_gate,
          use_current_frame_in_readout_random=use_current_frame_in_readout_random,
          target_embed_dim=Dim(name="target_embed", dimension=target_embed_dim),
          target_embed_dropout=target_embed_dropout,
          att_weight_dropout=att_weight_dropout,
          use_hard_attention=center_window_size == 1,
          readout_dimension=readout_dimension,
          ilm_dimension=ilm_dimension,
          use_readout=use_readout,
        )

    if not use_joint_model:
      if label_decoder_state == "trafo":
        label_state_dim = model_dim
      else:
        label_state_dim = self.label_decoder.get_lstm().out_dim

      # this is just to keep the hash of one exp the same
      blank_decoder_opts.pop("version", None)
      if blank_decoder_version == 1:
        self.blank_decoder = BlankDecoderV1(
          length_model_state_dim=length_model_state_dim,
          length_model_embed_dim=length_model_embed_dim,
          align_target_dim=align_target_dim,
          encoder_out_dim=self.encoder.out_dim,
        )
      elif blank_decoder_version == 3:
        self.blank_decoder = BlankDecoderV3(
          length_model_state_dim=length_model_state_dim,
          label_state_dim=label_state_dim,
          encoder_out_dim=self.encoder.out_dim,
        )
      elif blank_decoder_version == 4:
        self.blank_decoder = BlankDecoderV4(
          length_model_state_dim=length_model_state_dim,
          label_state_dim=label_state_dim,
          encoder_out_dim=self.encoder.out_dim,
          **blank_decoder_opts,
        )
      elif blank_decoder_version == 5:
        self.blank_decoder = BlankDecoderV5(
          label_state_dim=label_state_dim,
          encoder_out_dim=self.encoder.out_dim,
        )
      elif blank_decoder_version == 6:
        self.blank_decoder = BlankDecoderV6(
          length_model_state_dim=length_model_state_dim,
          label_state_dim=label_state_dim,
          encoder_out_dim=self.encoder.out_dim,
        )
      elif blank_decoder_version == 7:
        self.blank_decoder = BlankDecoderV7(
          length_model_state_dim=length_model_state_dim,
          label_state_dim=label_state_dim,
          encoder_out_dim=self.encoder.out_dim,
          **blank_decoder_opts,
        )
      elif blank_decoder_version == 8:
        self.blank_decoder = BlankDecoderV8(
          length_model_state_dim=length_model_state_dim,
          encoder_out_dim=self.encoder.out_dim,
        )
      elif blank_decoder_version == 9:
        self.blank_decoder = BlankDecoderV9(
          energy_in_dim=self.label_decoder.enc_key_total_dim,
        )
      elif blank_decoder_version == 10:
        self.blank_decoder = BlankDecoderV10(
          energy_in_dim=self.label_decoder.enc_key_total_dim,
        )
      else:
        self.blank_decoder = BlankDecoderV11(
          label_state_dim=label_state_dim,
          encoder_out_dim=self.encoder.out_dim,
        )
    else:
      self.blank_decoder = None

    if language_model:
      self.language_model, self.language_model_make_label_scorer = language_model
    else:
      self.language_model = None
      self.language_model_make_label_scorer = None

    self.blank_idx = blank_idx
    self.center_window_size = center_window_size
    self.window_step_size = window_step_size
    self.use_vertical_transitions = use_vertical_transitions
    if label_decoder_state == "trafo":
      self.target_dim = self.label_decoder.vocab_dim
    else:
      self.target_dim = self.label_decoder.target_dim
    self.align_target_dim = align_target_dim
    self.use_joint_model = use_joint_model
    self.blank_decoder_version = blank_decoder_version
    self.label_decoder_state = label_decoder_state

    if external_aed_model_kwargs:
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
        use_mini_att=use_mini_att,
      )
    else:
      self.aed_model = None

    if external_transducer_kwargs:
      self.external_transducer_model = SegmentalAttentionModel(
        enc_in_dim=enc_in_dim,
        enc_ff_dim=enc_ff_dim,
        enc_key_total_dim=enc_key_total_dim,
        enc_num_heads=8,
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        feature_extraction_opts=feature_extraction_opts,
        enc_aux_logits=(),
        encoder_layer_opts=encoder_layer_opts,
        center_window_size=center_window_size,
        blank_decoder_version=blank_decoder_version,
        length_model_state_dim=length_model_state_dim,
        length_model_embed_dim=length_model_embed_dim,
        use_att_ctx_in_state=use_att_ctx_in_state,
        use_weight_feedback=use_weight_feedback,
        label_decoder_state=label_decoder_state,
        window_step_size=window_step_size,
        use_vertical_transitions=use_vertical_transitions,
        align_target_dim=align_target_dim,
        blank_decoder_opts=blank_decoder_opts,
        use_mini_att=use_mini_att,
      )
    else:
      self.external_transducer_model = None

    apply_weight_dropout(self)


class SegmentalAttLabelDecoder(BaseLabelDecoder):
  def __init__(self, center_window_size: int, gaussian_att_weight_opts: Optional[Dict] = None, **kwargs):
    super(SegmentalAttLabelDecoder, self).__init__(**kwargs)

    if center_window_size == 1:
      assert not self.use_weight_feedback, "Weight feedback has no effect with window size 1!"

    self.center_window_size = center_window_size
    self.accum_att_weights_dim = Dim(name="accum_att_weights", dimension=center_window_size)
    self.gaussian_att_weight_opts = gaussian_att_weight_opts

  def default_initial_state(
          self,
          *,
          batch_dims: Sequence[Dim],
          segment_starts_sparse_dim: Optional[Dim] = None,
          segment_lens_sparse_dim: Optional[Dim] = None,
          use_mini_att: bool = False,
          use_zero_att: bool = False,
  ) -> rf.State:
    """Default initial state"""
    if self.center_window_size == 1:
      att_dim = self.enc_out_dim
    else:
      att_dim = self.att_num_heads * self.enc_out_dim
    state = rf.State(
      att=rf.zeros(list(batch_dims) + [att_dim]),
      segment_starts=rf.zeros(batch_dims, sparse_dim=segment_starts_sparse_dim, dtype="int32"),
      segment_lens=rf.zeros(batch_dims, sparse_dim=segment_lens_sparse_dim, dtype="int32"),
    )
    state.att.feature_dim_axis = len(state.att.dims) - 1

    if "lstm" in self.decoder_state:
      state.s = self.get_lstm().default_initial_state(batch_dims=batch_dims)
      if use_mini_att:
        state.mini_att_lstm = self.mini_att_lstm.default_initial_state(batch_dims=batch_dims)

    if self.use_weight_feedback and not use_mini_att and not use_zero_att:
      state.accum_att_weights = rf.zeros(
        list(batch_dims) + [self.accum_att_weights_dim, self.att_num_heads], feature_dim=self.att_num_heads
      )

    return state

  def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
    """loop step out"""
    if self.center_window_size == 1:
      att_dim = self.enc_out_dim
    else:
      att_dim = self.att_num_heads * self.enc_out_dim

    return {
      "s": Tensor(
        "s", dims=batch_dims + [self.get_lstm().out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
      ),
      "att": Tensor(
        "att",
        dims=batch_dims + [att_dim],
        dtype=rf.get_default_float_dtype(),
        feature_dim_axis=-1,
      ),
    }

  def _get_prev_accum_att_weights_scattered(
          self,
          prev_accum_att_weights: Tensor,
          segment_starts: Tensor,
          prev_segment_starts: Tensor,
          prev_segment_lens: Tensor,
  ) -> Tensor:

    overlap_len = rf.cast(prev_segment_starts + prev_segment_lens - segment_starts, "int32")
    overlap_len = rf.where(
      rf.logical_or(overlap_len < 0, overlap_len > prev_segment_lens),
      rf.convert_to_tensor(0),
      overlap_len
    )
    overlap_start = prev_segment_lens - overlap_len

    slice_dim = Dim(name="slice", dimension=overlap_len)
    gather_positions = rf.range_over_dim(slice_dim)
    gather_positions += overlap_start
    prev_accum_att_weights_overlap = rf.gather(
      prev_accum_att_weights, axis=self.accum_att_weights_dim, indices=gather_positions, clip_to_valid=True
    )
    overlap_range = rf.range_over_dim(slice_dim)

    prev_accum_att_weights_scattered = rf.scatter(
      prev_accum_att_weights_overlap,
      out_dim=self.accum_att_weights_dim,
      indices=overlap_range,
      indices_dim=slice_dim,
    )

    return prev_accum_att_weights_scattered

  def _get_accum_att_weights(
          self,
          att_t_dim: Dim,
          enc_spatial_dim: Dim,
          inv_fertility: Tensor,
          att_weights: Tensor,
          prev_accum_att_weights_scattered: Tensor,
          gather_positions: Tensor,
  ) -> Tensor:
    att_weights_range = rf.range_over_dim(att_t_dim)
    att_weights_scattered = rf.scatter(
      att_weights,
      out_dim=self.accum_att_weights_dim,
      indices=att_weights_range,
      indices_dim=att_t_dim,
    )

    inv_fertility_sliced = rf.gather(inv_fertility, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)
    inv_fertility_scattered = rf.scatter(
      inv_fertility_sliced,
      out_dim=self.accum_att_weights_dim,
      indices=att_weights_range,
      indices_dim=att_t_dim,
    )

    accum_att_weights = prev_accum_att_weights_scattered + att_weights_scattered * inv_fertility_scattered * 0.5

    return accum_att_weights

  def _get_weight_feedback(
          self,
          prev_accum_att_weights_scattered: Tensor,
          att_t_dim: Dim,
  ) -> Tensor:
    gather_positions = rf.range_over_dim(att_t_dim)
    prev_accum_att_weights_sliced = rf.gather(
      prev_accum_att_weights_scattered,
      axis=self.accum_att_weights_dim,
      indices=gather_positions,
      clip_to_valid=True
    )

    return self.weight_feedback(prev_accum_att_weights_sliced)

  def loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_ctx: rf.Tensor,
          inv_fertility: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          segment_starts: rf.Tensor,
          segment_lens: rf.Tensor,
          center_positions: rf.Tensor,
          state: rf.State,
          use_mini_att: bool = False,
          use_zero_att: bool = False,
          detach_prev_att: bool = False,
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    state_ = rf.State()

    prev_att = state.att
    if detach_prev_att:
      prev_att = rf.stop_gradient(prev_att)

    prev_s_state = state.s if "lstm" in self.decoder_state else None

    input_embed = rf.dropout(input_embed, drop_prob=self.target_embed_dropout, axis=None)
    s, s_state = self._update_state(input_embed, prev_att, prev_s_state)
    if "lstm" in self.decoder_state:
      state_.s = s_state

    if use_mini_att:
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
      slice_dim = Dim(name="slice", dimension=segment_lens)
      gather_positions = rf.range_over_dim(slice_dim)
      gather_positions += segment_starts

      prev_segment_starts = state.segment_starts
      prev_segment_lens = state.segment_lens

      enc_sliced = rf.gather(enc, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)

      if self.use_weight_feedback:
        prev_accum_att_weights = state.accum_att_weights
        prev_accum_att_weights_scattered = self._get_prev_accum_att_weights_scattered(
          prev_accum_att_weights=prev_accum_att_weights,
          segment_starts=segment_starts,
          prev_segment_starts=prev_segment_starts,
          prev_segment_lens=prev_segment_lens,
        )
        weight_feedback = self._get_weight_feedback(
          prev_accum_att_weights_scattered=prev_accum_att_weights_scattered,
          att_t_dim=slice_dim,
        )
      else:
        weight_feedback = rf.zeros((self.enc_key_total_dim,))

      if self.center_window_size == 1:
        att = rf.gather(enc, axis=enc_spatial_dim, indices=center_positions, clip_to_valid=True)
      else:
        if self.gaussian_att_weight_opts:
          att_weights = -0.5 * ((gather_positions - center_positions) / self.gaussian_att_weight_opts["std"]) ** 2
          att_weights = rf.softmax(att_weights, axis=slice_dim, use_mask=True)
          att_weights = rf.expand_dim(att_weights, dim=self.att_num_heads)
        else:
          enc_ctx_sliced = rf.gather(enc_ctx, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)
          s_transformed = self.s_transformed(s)
          energy_in = enc_ctx_sliced + weight_feedback + s_transformed
          energy = self.energy(rf.tanh(energy_in))
          att_weights = rf.softmax(energy, axis=slice_dim)
          # axis = None is equivalent to settings dropout_noise_shape in old RETURNN
          att_weights = rf.dropout(att_weights, drop_prob=self.att_weight_dropout, axis=None)

        # we do not need use_mask because the softmax output is already padded with zeros
        att = self.get_att(att_weights, enc_sliced, reduce_dim=slice_dim)

        if self.use_weight_feedback:
          accum_att_weights = self._get_accum_att_weights(
            att_t_dim=slice_dim,
            enc_spatial_dim=enc_spatial_dim,
            inv_fertility=inv_fertility,
            att_weights=att_weights,
            prev_accum_att_weights_scattered=prev_accum_att_weights_scattered,
            gather_positions=gather_positions,
          )
          accum_att_weights.feature_dim = self.att_num_heads
          state_.accum_att_weights = accum_att_weights

    state_.att = att
    state_.segment_starts = segment_starts
    state_.segment_lens = segment_lens

    return {"s": s, "att": att}, state_


class BlankDecoderBase(rf.Module, ABC):
  def __init__(self):
    super(BlankDecoderBase, self).__init__()
    self.emit_prob_dim = Dim(name="emit_prob", dimension=1)

  @property
  @abstractmethod
  def _s(self) -> rf.LSTM:
    pass

  def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
    """Default initial state"""
    state = rf.State(
      s_blank=self._s.default_initial_state(batch_dims=batch_dims),
      i=rf.zeros(batch_dims, dtype="int32"),
    )
    return state

  def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
    """loop step out"""
    return {
      "s_blank": Tensor(
        "s_blank",
        dims=batch_dims + [self._s.out_dim],
        dtype=rf.get_default_float_dtype(),
        feature_dim_axis=-1
      ),
    }

  @staticmethod
  def _get_am(enc: rf.Tensor, enc_spatial_dim: Dim, state: rf.State, spatial_dim) -> rf.Tensor:
    if spatial_dim == single_step_dim:
      i = state.i
      clip_to_valid = True
    else:
      i = rf.range_over_dim(spatial_dim)
      # do clipping here, because rf.gather complains that i.dims_set is not a superset of enc_spatial_dim.dims_set
      seq_lens = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, i.device)
      i = rf.where(i < seq_lens, i, seq_lens - 1)
      clip_to_valid = False

    return rf.gather(enc, axis=enc_spatial_dim, indices=i, clip_to_valid=clip_to_valid)

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return None


class BlankDecoderV3(BlankDecoderBase):
  def __init__(
          self,
          length_model_state_dim: Dim,
          label_state_dim: Dim,
          encoder_out_dim: Dim,
  ):
    super(BlankDecoderV3, self).__init__()
    self.length_model_state_dim = length_model_state_dim

    self.s = rf.LSTM(
      encoder_out_dim + label_state_dim,
      self.length_model_state_dim,
    )
    self.emit_prob = rf.Linear(self.length_model_state_dim, self.emit_prob_dim)

  @property
  def _s(self) -> rf.LSTM:
    return self.s

  def loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_spatial_dim: Dim,
          label_model_state: rf.Tensor,
          state: rf.State,
          spatial_dim=single_step_dim
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    state_ = rf.State()

    am = self._get_am(enc, enc_spatial_dim, state, spatial_dim)
    s_blank, state_.s_blank = self.s(
      rf.concat_features(am, label_model_state),
      state=state.s_blank,
      spatial_dim=spatial_dim
    )

    state_.i = state.i + 1

    return {"s_blank": s_blank}, state_

  def decode_logits(self, *, s_blank: Tensor) -> Tensor:
    """logits for the decoder"""
    logits = self.emit_prob(s_blank)
    return logits

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return ["s"]


class BlankDecoderV4(BlankDecoderBase):
  def __init__(
          self,
          length_model_state_dim: Dim,
          label_state_dim: Dim,
          encoder_out_dim: Dim,
          dropout: float = 0.0,
  ):
    super(BlankDecoderV4, self).__init__()
    self.length_model_state_dim = length_model_state_dim
    self.dropout = dropout

    self.s = rf.Linear(
      encoder_out_dim + label_state_dim,
      self.length_model_state_dim,
    )
    self.emit_prob = rf.Linear(self.length_model_state_dim // 2, self.emit_prob_dim)

  def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
    """Default initial state"""
    state = rf.State()
    return state

  @property
  def _s(self) -> rf.LSTM:
    raise NotImplementedError

  def decode_logits(
          self,
          *,
          enc: Tensor,
          label_model_states_unmasked: rf.Tensor,
          allow_broadcast: bool = False,
  ) -> Tensor:
    """logits for the decoder"""

    s_input = rf.concat_features(enc, label_model_states_unmasked, allow_broadcast=allow_broadcast)
    s_input = rf.dropout(s_input, drop_prob=self.dropout, axis=s_input.feature_dim)

    s_blank = self.s(s_input)
    s_blank = rf.reduce_out(s_blank, mode="max", num_pieces=2, out_dim=self.emit_prob.in_dim)
    s_blank = rf.dropout(s_blank, drop_prob=0.3, axis=rf.dropout_broadcast_default() and s_blank.feature_dim)
    logits = self.emit_prob(s_blank)
    return logits

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return ["s"]


class MakeModel:
  """for import"""

  def __init__(self, in_dim: int, align_target_dim: int, target_dim: int, *, center_window_size: int, eos_label: int = 0, num_enc_layers: int = 12):
    self.in_dim = in_dim
    self.align_target_dim = align_target_dim
    self.target_dim = target_dim
    self.center_window_size = center_window_size
    self.eos_label = eos_label
    self.num_enc_layers = num_enc_layers

  def __call__(self) -> SegmentalAttentionModel:
    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
    align_target_dim = Dim(name="align_target", dimension=self.align_target_dim, kind=Dim.Types.Feature)
    target_dim = Dim(name="non_blank_target", dimension=self.target_dim, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(
      [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
    )

    return self.make_model(in_dim, align_target_dim, target_dim, center_window_size=self.center_window_size)

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          align_target_dim: Dim,
          target_dim: Dim,
          *,
          center_window_size: int,
          language_model: Optional[Dict[str, Any]] = None,
          external_aed_kwargs: Optional[Dict[str, Any]] = None,
          external_transducer_kwargs: Optional[Dict[str, Any]] = None,
          use_att_ctx_in_state: bool,
          blank_decoder_version: int,
          blank_decoder_opts: Optional[Dict[str, Any]] = None,
          use_joint_model: bool,
          use_weight_feedback: bool,
          label_decoder_state: str,
          enc_key_total_dim: int,
          enc_ff_dim: int,
          use_mini_att: bool = False,
          gaussian_att_weight_opts: Optional[Dict[str, Any]] = None,
          separate_blank_from_softmax: bool = False,
          reset_eos_params: bool = False,
          use_current_frame_in_readout: bool = False,
          target_embed_dim: int = 640,
          feature_extraction_opts: Optional[Dict[str, Any]] = None,
          **extra,
  ) -> SegmentalAttentionModel:
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
      lm = (lm, functools.partial(trafo_lm.make_time_sync_label_scorer_torch, model=lm, align_target_dim=align_target_dim))

    return SegmentalAttentionModel(
      enc_in_dim=in_dim,
      enc_ff_dim=Dim(name="enc-ff", dimension=enc_ff_dim, kind=Dim.Types.Feature),
      enc_key_total_dim=Dim(name="enc_key_total_dim", dimension=enc_key_total_dim),
      enc_num_heads=8,
      target_dim=target_dim,
      align_target_dim=align_target_dim,
      blank_idx=0 if use_joint_model else target_dim.dimension,
      language_model=lm,
      external_aed_model_kwargs=external_aed_kwargs,
      length_model_state_dim=Dim(name="length_model_state", dimension=128, kind=Dim.Types.Feature),
      length_model_embed_dim=Dim(name="length_model_embed", dimension=128, kind=Dim.Types.Feature),
      center_window_size=center_window_size,
      use_att_ctx_in_state=use_att_ctx_in_state,
      blank_decoder_version=blank_decoder_version,
      blank_decoder_opts=blank_decoder_opts,
      use_joint_model=use_joint_model,
      use_weight_feedback=use_weight_feedback,
      label_decoder_state=label_decoder_state,
      use_mini_att=use_mini_att,
      gaussian_att_weight_opts=gaussian_att_weight_opts,
      separate_blank_from_softmax=separate_blank_from_softmax,
      reset_eos_params=reset_eos_params,
      use_current_frame_in_readout=use_current_frame_in_readout,
      target_embed_dim=target_embed_dim,
      feature_extraction_opts=feature_extraction_opts,
      external_transducer_kwargs=external_transducer_kwargs,
      **extra,
    )


def from_scratch_model_def(
        *, epoch: int, in_dim: Dim, align_target_dim: Dim, target_dim: Dim) -> SegmentalAttentionModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  in_dim, epoch  # noqa
  config = get_global_config()  # noqa

  center_window_size = config.typed_value("center_window_size")
  window_step_size = config.int("window_step_size", 1)
  use_vertical_transitions = config.bool("use_vertical_transitions", False)

  blank_decoder_version = config.int("blank_decoder_version", 1)
  blank_decoder_opts = config.typed_value("blank_decoder_opts", {})
  use_joint_model = config.bool("use_joint_model", False)
  gaussian_att_weight_opts = config.typed_value("gaussian_att_weight_opts", None)
  separate_blank_from_softmax = config.bool("separate_blank_from_softmax", False)
  reset_eos_params = config.bool("reset_eos_params", False)
  use_current_frame_in_readout = config.bool("use_current_frame_in_readout", False)
  use_current_frame_in_readout_w_gate = config.bool("use_current_frame_in_readout_w_gate", False)
  use_current_frame_in_readout_random = config.bool("use_current_frame_in_readout_random", False)

  enc_key_total_dim = config.int("enc_key_total_dim", 1024)
  enc_ff_dim = config.int("enc_ff_dim", 2048)

  trafo_decoder_opts = config.typed_value("trafo_decoder_opts", None)
  use_trafo_att = config.bool("use_trafo_att", False)

  external_transducer_kwargs = config.typed_value("external_transducer_kwargs", None)

  common_config_params = get_common_config_params()
  in_dim = common_config_params.pop("in_dim")

  return MakeModel.make_model(
    in_dim,
    align_target_dim,
    target_dim,
    center_window_size=center_window_size,
    window_step_size=window_step_size,
    use_vertical_transitions=use_vertical_transitions,
    blank_decoder_version=blank_decoder_version,
    blank_decoder_opts=blank_decoder_opts,
    use_joint_model=use_joint_model,
    enc_key_total_dim=enc_key_total_dim,
    enc_ff_dim=enc_ff_dim,
    gaussian_att_weight_opts=gaussian_att_weight_opts,
    separate_blank_from_softmax=separate_blank_from_softmax,
    reset_eos_params=reset_eos_params,
    use_current_frame_in_readout=use_current_frame_in_readout,
    use_current_frame_in_readout_w_gate=use_current_frame_in_readout_w_gate,
    use_current_frame_in_readout_random=use_current_frame_in_readout_random,
    trafo_decoder_opts=trafo_decoder_opts,
    use_trafo_att=use_trafo_att,
    external_transducer_kwargs=external_transducer_kwargs,
    **common_config_params,
  )


from_scratch_model_def: ModelDef[SegmentalAttentionModel]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
  """
  Here, we use a separate blank model and define the blank_index=len(target_vocab). In this case, the target_dim
  is one smaller than the align_target_dim and the EOS label is unused.
  """
  from returnn.tensor import Tensor, Dim
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  extern_data_dict = config.typed_value("extern_data")
  non_blank_vocab = config.typed_value("non_blank_vocab")
  data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])

  align_target_dim = config.typed_value("align_target_dim", None)
  non_blank_target_dim = config.typed_value("non_blank_target_dim", None)

  # if dim tags are not given, maybe dimensions (integers) are given. then we can create the dim tags here.
  if align_target_dim is None:
    align_target_dimension = config.int("align_target_dimension", None)
    if align_target_dimension:
      align_target_dim = Dim(
        description="align_target_dim", dimension=align_target_dimension, kind=Dim.Types.Spatial)
  if non_blank_target_dim is None:
    non_blank_target_dimension = config.int("non_blank_target_dimension", None)
    if non_blank_target_dimension:
      non_blank_target_dim = Dim(
        description="non_blank_target_dim", dimension=non_blank_target_dimension, kind=Dim.Types.Spatial)

  if default_target_key in extern_data_dict:
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
  else:
    assert align_target_dim is not None or non_blank_target_dim is not None

  if align_target_dim is None:
    if non_blank_target_dim is None:
      # if both align_target_dim and non_blank_target_dim are not set, we assume it is the sparse dim of the targets tensor
      align_target_dim = targets.sparse_dim
    else:
      # else, we assume align_target_dim is non_blank_target_dim + 1
      align_target_dim = Dim(
        description="align_target_dim", dimension=non_blank_target_dim.dimension + 1, kind=Dim.Types.Spatial)

  if non_blank_target_dim is None:
    # non_blank_target_dim = Dim(
    #   description="non_blank_target_dim",
    #   dimension=align_target_dim.dimension - 1,
    #   kind=Dim.Types.Spatial,
    # )
    non_blank_targets = Tensor(
      name="non_blank_targets",
      sparse_dim=Dim(description="non_blank_target_dim", dimension=align_target_dim.dimension - 1, kind=Dim.Types.Spatial),
      vocab=non_blank_vocab,
    )
    non_blank_target_dim = non_blank_targets.sparse_dim
    # else:
    #   # # else, we assume non_blank_target_dim is align_target_dim - 1
    #   # non_blank_target_dim = Dim(
    #   #   description="non_blank_target_dim",
    #   #   dimension=align_target_dim.dimension - 1,
    #   #   kind=Dim.Types.Spatial,
    #   #   vocab=non_blank_vocab,
    #   # )

  model_def = config.typed_value("_model_def")
  model = model_def(
    epoch=epoch, in_dim=data.feature_dim, align_target_dim=align_target_dim, target_dim=non_blank_target_dim)
  return model


def _returnn_v2_get_joint_model(*, epoch: int, **_kwargs_unused):
  """
  Here, we reinterpret the EOS label as a blank label and use a single softmax for both blank and non-blank labels.
  Therefore, we assume align_target_dim and target_dim to be the same.
  """
  from returnn.tensor import Tensor
  from returnn.config import get_global_config
  from returnn.datasets.util.vocabulary import BytePairEncoding

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  extern_data_dict = config.typed_value("extern_data")
  data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
  non_blank_vocab = config.typed_value("non_blank_vocab")

  if default_target_key in extern_data_dict:
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
  else:
    align_target_dimension = config.int("align_target_dimension", None)
    assert align_target_dimension
    align_target_dim = Dim(
      description="align_target_dim", dimension=align_target_dimension, kind=Dim.Types.Spatial)
    targets = Tensor(name=default_target_key, sparse_dim=align_target_dim)

  if non_blank_vocab is not None:
    targets.sparse_dim.vocab = BytePairEncoding(**non_blank_vocab)

  model_def = config.typed_value("_model_def")
  model = model_def(
    epoch=epoch, in_dim=data.feature_dim, align_target_dim=targets.sparse_dim, target_dim=targets.sparse_dim)
  return model
