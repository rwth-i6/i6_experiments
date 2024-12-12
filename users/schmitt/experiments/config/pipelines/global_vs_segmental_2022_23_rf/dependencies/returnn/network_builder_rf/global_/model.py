from typing import Optional, Dict, Any, Sequence, Tuple, List, Union
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import ConformerEncoderLayer, ConformerConvSubsample

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder,
  GlobalAttEfficientDecoder
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.encoder.global_ import (
  GlobalConformerEncoder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.encoder.ff import LinearEncoder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import (
  _batch_size_factor,
  get_common_config_params,
  apply_weight_dropout,
)


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
