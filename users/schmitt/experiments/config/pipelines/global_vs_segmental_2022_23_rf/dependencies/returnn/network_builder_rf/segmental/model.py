from typing import Optional, Dict, Any, Sequence, Tuple, List
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.attention import RelPosCausalSelfAttention

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
  BlankDecoderV1,
  BlankDecoderV3,
  BlankDecoderV4,
  BlankDecoderV5,
  BlankDecoderV6,
  BlankDecoderV7,
  BlankDecoderV8,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder, SegmentalAttEfficientLabelDecoder
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder, GlobalAttEfficientDecoder
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.encoder.global_ import GlobalConformerEncoder
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.supports_label_scorer_torch import RFModelWithMakeLabelScorer


class SegmentalAttentionModel(rf.Module):
  def __init__(
          self,
          *,
          length_model_state_dim: Dim,
          length_model_embed_dim: Dim,
          center_window_size: int,
          align_target_dim: Dim,
          target_dim: Dim,
          blank_idx: int,
          enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
          att_dropout: float = 0.1,
          l2: float = 0.0001,
          language_model: Optional[RFModelWithMakeLabelScorer] = None,
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
  ):
    super(SegmentalAttentionModel, self).__init__()

    self.encoder = GlobalConformerEncoder(
      enc_in_dim,
      enc_out_dim,
      num_layers=enc_num_layers,
      target_dim=target_dim,
      wb_target_dim=align_target_dim,
      aux_logits=enc_aux_logits,
      ff_dim=enc_ff_dim,
      num_heads=enc_num_heads,
      encoder_layer_opts=encoder_layer_opts,
      enc_key_total_dim=enc_key_total_dim,
      dec_att_num_heads=dec_att_num_heads,
      dropout=enc_dropout,
      att_dropout=att_dropout,
      l2=l2,
      use_weight_feedback=use_weight_feedback,
      feature_extraction_opts=feature_extraction_opts,
      decoder_type="trafo" if label_decoder_state == "trafo" else "lstm",
    )

    assert blank_decoder_version in {1, 3, 4, 5, 6, 7, 8}
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
        )
    else:
      if label_decoder_state == "trafo":
        raise NotImplementedError("Trafo decoder not implemented for segmental model")
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
        )

    if not use_joint_model:
      if label_decoder_state == "trafo":
        label_state_dim = model_dim
      else:
        label_state_dim = self.label_decoder.get_lstm().out_dim

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
      else:
        self.blank_decoder = BlankDecoderV8(
          length_model_state_dim=length_model_state_dim,
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
    if label_decoder_state == "trafo":
      self.target_dim = self.label_decoder.vocab_dim
    else:
      self.target_dim = self.label_decoder.target_dim
    self.align_target_dim = align_target_dim
    self.use_joint_model = use_joint_model
    self.blank_decoder_version = blank_decoder_version
    self.label_decoder_state = label_decoder_state


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
          num_enc_layers: int = 12,
          pos_emb_dropout: float = 0.0,
          language_model: Optional[Dict[str, Any]] = None,
          use_att_ctx_in_state: bool,
          blank_decoder_version: int,
          blank_decoder_opts: Optional[Dict[str, Any]] = None,
          use_joint_model: bool,
          use_weight_feedback: bool,
          label_decoder_state: str,
          enc_out_dim: int,
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
      enc_num_layers=num_enc_layers,
      enc_out_dim=Dim(name="enc", dimension=enc_out_dim, kind=Dim.Types.Feature),
      enc_ff_dim=Dim(name="enc-ff", dimension=enc_ff_dim, kind=Dim.Types.Feature),
      enc_key_total_dim=Dim(name="enc_key_total_dim", dimension=enc_key_total_dim),
      enc_num_heads=8,
      encoder_layer_opts=dict(
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
      target_dim=target_dim,
      align_target_dim=align_target_dim,
      blank_idx=0 if use_joint_model else target_dim.dimension,
      language_model=lm,
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
      **extra,
    )


def from_scratch_model_def(
        *, epoch: int, in_dim: Dim, align_target_dim: Dim, target_dim: Dim) -> SegmentalAttentionModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  in_dim, epoch  # noqa
  config = get_global_config()  # noqa
  enc_aux_logits = config.typed_value("aux_loss_layers")
  pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
  log_mel_feature_dim = config.int("log_mel_feature_dim", 80)
  # real input is raw audio, internally it does logmel
  in_dim = Dim(name="logmel", dimension=log_mel_feature_dim, kind=Dim.Types.Feature)
  lm_opts = config.typed_value("external_lm")
  center_window_size = config.typed_value("center_window_size")

  use_att_ctx_in_state = config.bool("use_att_ctx_in_state", True)
  label_decoder_state = config.typed_value("label_decoder_state", "nb-lstm")
  blank_decoder_version = config.int("blank_decoder_version", 1)
  blank_decoder_opts = config.typed_value("blank_decoder_opts", {})
  use_joint_model = config.bool("use_joint_model", False)
  use_weight_feedback = config.bool("use_weight_feedback", True)
  use_mini_att = config.bool("use_mini_att", False)
  gaussian_att_weight_opts = config.typed_value("gaussian_att_weight_opts", None)
  separate_blank_from_softmax = config.bool("separate_blank_from_softmax", False)
  reset_eos_params = config.bool("reset_eos_params", False)
  use_current_frame_in_readout = config.bool("use_current_frame_in_readout", False)
  use_current_frame_in_readout_w_gate = config.bool("use_current_frame_in_readout_w_gate", False)
  use_current_frame_in_readout_random = config.bool("use_current_frame_in_readout_random", False)

  enc_out_dim = config.int("enc_out_dim", 512)
  enc_key_total_dim = config.int("enc_key_total_dim", 1024)
  enc_ff_dim = config.int("enc_ff_dim", 2048)

  target_embed_dim = config.int("target_embed_dim", 640)

  feature_extraction_opts = config.typed_value("feature_extraction_opts", None)

  trafo_decoder_opts = config.typed_value("trafo_decoder_opts", None)

  return MakeModel.make_model(
    in_dim,
    align_target_dim,
    target_dim,
    center_window_size=center_window_size,
    enc_aux_logits=enc_aux_logits or (),
    pos_emb_dropout=pos_emb_dropout,
    language_model=lm_opts,
    use_att_ctx_in_state=use_att_ctx_in_state,
    blank_decoder_version=blank_decoder_version,
    blank_decoder_opts=blank_decoder_opts,
    use_joint_model=use_joint_model,
    use_weight_feedback=use_weight_feedback,
    label_decoder_state=label_decoder_state,
    enc_out_dim=enc_out_dim,
    enc_key_total_dim=enc_key_total_dim,
    enc_ff_dim=enc_ff_dim,
    use_mini_att=use_mini_att,
    gaussian_att_weight_opts=gaussian_att_weight_opts,
    separate_blank_from_softmax=separate_blank_from_softmax,
    reset_eos_params=reset_eos_params,
    use_current_frame_in_readout=use_current_frame_in_readout,
    use_current_frame_in_readout_w_gate=use_current_frame_in_readout_w_gate,
    use_current_frame_in_readout_random=use_current_frame_in_readout_random,
    target_embed_dim=target_embed_dim,
    feature_extraction_opts=feature_extraction_opts,
    trafo_decoder_opts=trafo_decoder_opts,
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
