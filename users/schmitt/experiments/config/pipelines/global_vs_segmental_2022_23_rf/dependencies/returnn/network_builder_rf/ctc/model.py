from typing import Optional, Dict, Any, Sequence, Tuple, List, Union
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import ConformerEncoderLayer, ConformerConvSubsample

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.encoder.global_ import (
  GlobalConformerEncoder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import (
  _returnn_v2_get_model as _returnn_v2_get_segmental_model,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import (
  _batch_size_factor,
  get_common_config_params,
  apply_weight_dropout,
)


class CtcModel(rf.Module):
  def __init__(
          self,
          *,
          target_dim: Dim,
          align_target_dim: Dim,
          enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
          att_dropout: float = 0.1,
          l2: float = 0.0001,
          enc_in_dim: Dim,
          enc_out_dim: Dim = Dim(name="enc", dimension=512),
          enc_num_layers: int = 12,
          enc_aux_logits: Sequence[int] = (),  # layers
          enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
          enc_num_heads: int = 4,
          encoder_layer_opts: Optional[Dict[str, Any]] = None,
          encoder_layer: Optional[Union[ConformerEncoderLayer, rf.Module, type, Dict[str, Any], Any]] = None,
          enc_input_layer_cls: rf.Module = ConformerConvSubsample,
          encoder_cls: rf.Module = GlobalConformerEncoder,
          dec_att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
          enc_dropout: float = 0.1,
          feature_extraction_opts: Optional[Dict[str, Any]] = None,
  ):
    super(CtcModel, self).__init__()

    self.target_dim = target_dim
    self.align_target_dim = align_target_dim
    self.blank_idx = target_dim.dimension

    if enc_num_layers not in enc_aux_logits:
      enc_aux_logits = list(enc_aux_logits) + [enc_num_layers]

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
      enc_key_total_dim=enc_key_total_dim,
      dec_att_num_heads=dec_att_num_heads,
      dropout=enc_dropout,
      att_dropout=att_dropout,
      l2=l2,
      decoder_type="trafo",
      feature_extraction_opts=feature_extraction_opts,
      encoder_layer=encoder_layer,
      input_layer_cls=enc_input_layer_cls,
    )


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

  def __call__(self) -> CtcModel:
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
          align_target_dim: Dim,
          *,
          feature_extraction_opts: Optional[Dict[str, Any]] = None,
          **extra,
  ) -> CtcModel:
    """make"""

    return CtcModel(
      target_dim=target_dim,
      align_target_dim=align_target_dim,
      enc_in_dim=in_dim,
      enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
      enc_num_heads=8,
      feature_extraction_opts=feature_extraction_opts,
      **extra,
    )


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim, align_target_dim: Dim) -> CtcModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  in_dim, epoch  # noqa
  config = get_global_config()  # noqa

  common_config_params = get_common_config_params()
  in_dim = common_config_params.pop("in_dim")

  common_config_params = {
    k: v for k, v in common_config_params.items() if k not in [
      "language_model",
      "use_weight_feedback",
      "use_att_ctx_in_state",
      "label_decoder_state",
      "target_embed_dim",
      "use_mini_att",
      "att_dropout",
      "target_embed_dropout",
      "att_weight_dropout",
      "ilm_dimension",
      "readout_dimension",
      "use_readout",
    ]
  }

  return MakeModel.make_model(
    in_dim,
    target_dim,
    align_target_dim,
    **common_config_params,
  )


from_scratch_model_def: ModelDef[CtcModel]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
  return _returnn_v2_get_segmental_model(epoch=epoch, **_kwargs_unused)
