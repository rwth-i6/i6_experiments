from typing import Optional, Dict, Any, Sequence, Tuple, List
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor, _log_mel_feature_dim
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
  BlankDecoderV1,
  BlankDecoderV3,
  BlankDecoderV5,
  BlankDecoderV6,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder, SegmentalAttEfficientLabelDecoder
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
    )

    assert blank_decoder_version in {1, 3, 4, 5, 6}
    assert label_decoder_state in {"nb-lstm", "joint-lstm"}
    if not use_joint_model:
      assert label_decoder_state == "nb-lstm"

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
    )

    if not use_joint_model:
      if blank_decoder_version == 1:
        self.blank_decoder = BlankDecoderV1(
          length_model_state_dim=length_model_state_dim,
          length_model_embed_dim=length_model_embed_dim,
          align_target_dim=align_target_dim,
          encoder_out_dim=self.encoder.out_dim,
        )
      elif blank_decoder_version in {3, 4}:
        # the logic for blank_decoder_version == 4 is in the train/recog code
        self.blank_decoder = BlankDecoderV3(
          length_model_state_dim=length_model_state_dim,
          label_state_dim=self.label_decoder.get_lstm().out_dim,
          encoder_out_dim=self.encoder.out_dim,
        )
      elif blank_decoder_version == 5:
        self.blank_decoder = BlankDecoderV5(
          label_state_dim=self.label_decoder.get_lstm().out_dim,
          encoder_out_dim=self.encoder.out_dim,
        )
      else:
        self.blank_decoder = BlankDecoderV6(
          length_model_state_dim=length_model_state_dim,
          label_state_dim=self.label_decoder.get_lstm().out_dim,
          encoder_out_dim=self.encoder.out_dim,
        )
    else:
      self.blank_decoder = None

    if language_model:
      self.language_model, self.language_model_make_label_scorer = language_model
    else:
      self.language_model = None
      self.language_model_make_label_scorer = None

    self.blank_idx = self.label_decoder.blank_idx
    self.center_window_size = center_window_size
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
          use_joint_model: bool,
          use_weight_feedback: bool,
          label_decoder_state: str,
          enc_out_dim: int,
          enc_key_total_dim: int,
          enc_ff_dim: int,
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
      use_joint_model=use_joint_model,
      use_weight_feedback=use_weight_feedback,
      label_decoder_state=label_decoder_state,
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
  # real input is raw audio, internally it does logmel
  in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
  lm_opts = config.typed_value("external_lm")
  center_window_size = config.typed_value("center_window_size")
  if center_window_size is None:
    raise ValueError("center_window_size is not set!")

  use_att_ctx_in_state = config.bool("use_att_ctx_in_state", True)
  label_decoder_state = config.typed_value("label_decoder_state", "nb-lstm")
  blank_decoder_version = config.int("blank_decoder_version", 1)
  use_joint_model = config.bool("use_joint_model", False)
  use_weight_feedback = config.bool("use_weight_feedback", True)

  enc_out_dim = config.int("enc_out_dim", 512)
  enc_key_total_dim = config.int("enc_key_total_dim", 1024)
  enc_ff_dim = config.int("enc_ff_dim", 2048)

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
    use_joint_model=use_joint_model,
    use_weight_feedback=use_weight_feedback,
    label_decoder_state=label_decoder_state,
    enc_out_dim=enc_out_dim,
    enc_key_total_dim=enc_key_total_dim,
    enc_ff_dim=enc_ff_dim,
  )


from_scratch_model_def: ModelDef[SegmentalAttentionModel]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
  from returnn.tensor import Tensor, Dim
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  extern_data_dict = config.typed_value("extern_data")
  non_blank_vocab = config.typed_value("non_blank_vocab")
  data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
  targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
  non_blank_targets = Tensor(
    name="non_blank_targets",
    sparse_dim=Dim(description="non_blank_vocab", dimension=targets.sparse_dim.dimension - 1, kind=Dim.Types.Spatial),
    vocab=non_blank_vocab,
  )

  model_def = config.typed_value("_model_def")
  model = model_def(
    epoch=epoch, in_dim=data.feature_dim, align_target_dim=targets.sparse_dim, target_dim=non_blank_targets.sparse_dim)
  return model


def _returnn_v2_get_model_for_full_sum_training(*, epoch: int, **_kwargs_unused):
  from returnn.tensor import Tensor, Dim
  from returnn.config import get_global_config

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  default_target_key = config.typed_value("target")
  extern_data_dict = config.typed_value("extern_data")
  data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
  targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])

  model_def = config.typed_value("_model_def")
  model = model_def(
    epoch=epoch, in_dim=data.feature_dim, align_target_dim=targets.sparse_dim, target_dim=targets.sparse_dim)
  return model
