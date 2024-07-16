from typing import Optional, Dict, Any, Sequence, Tuple, List
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder,
  GlobalAttEfficientDecoder
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.encoder.global_ import GlobalConformerEncoder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor, _log_mel_feature_dim
from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.supports_label_scorer_torch import RFModelWithMakeLabelScorer


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
          dec_att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
          enc_dropout: float = 0.1,
          eos_idx: int,
          use_weight_feedback: bool = True,
          use_att_ctx_in_state: bool = True,
          use_mini_att: bool = False,
          decoder_state: str = "nb-lstm",
          decoder_type: str = "lstm",
          num_dec_layers: int = 1,
          target_embed_dim: Dim = Dim(name="target_embed", dimension=640),
  ):
    super(GlobalAttentionModel, self).__init__()

    self.encoder = GlobalConformerEncoder(
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
    )

    if decoder_type == "lstm":
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
        decoder_state=decoder_state,
        target_embed_dim=target_embed_dim,
      )
    else:
      assert decoder_type == "trafo"

      self.label_decoder = TransformerDecoder(
        num_layers=num_dec_layers,
        encoder_dim=self.encoder.out_dim,
        vocab_dim=target_dim,
        model_dim=Dim(name="dec", dimension=512),
        sequential=rf.Sequential,
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
          target_embed_dim: int = 640
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
          num_enc_layers: int = 12,
          pos_emb_dropout: float = 0.0,
          language_model: Optional[Dict[str, Any]] = None,
          use_weight_feedback: bool = True,
          use_att_ctx_in_state: bool = True,
          use_mini_att: bool = False,
          decoder_state: str = "nb-lstm",
          decoder_type: str = "lstm",
          num_dec_layers: int = 1,
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
      enc_num_layers=num_enc_layers,
      enc_out_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
      enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
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
      eos_idx=_get_eos_idx(target_dim),
      target_dim=target_dim,
      blank_idx=target_dim.dimension,
      language_model=lm,
      use_weight_feedback=use_weight_feedback,
      use_att_ctx_in_state=use_att_ctx_in_state,
      use_mini_att=use_mini_att,
      decoder_state=decoder_state,
      decoder_type=decoder_type,
      num_dec_layers=num_dec_layers,
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
  use_weight_feedback = config.bool("use_weight_feedback", True)
  use_att_ctx_in_state = config.bool("use_att_ctx_in_state", True)
  use_mini_att = config.bool("use_mini_att", False)
  decoder_state = config.typed_value("label_decoder_state", "nb-lstm")
  decoder_type = config.typed_value("label_decoder_type", "lstm")
  num_dec_layers = config.int("num_dec_layers", 1)

  return MakeModel.make_model(
    in_dim,
    target_dim,
    enc_aux_logits=enc_aux_logits or (),
    pos_emb_dropout=pos_emb_dropout,
    language_model=lm_opts,
    use_weight_feedback=use_weight_feedback,
    use_att_ctx_in_state=use_att_ctx_in_state,
    use_mini_att=use_mini_att,
    decoder_state=decoder_state,
    decoder_type=decoder_type,
    num_dec_layers=num_dec_layers,
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
