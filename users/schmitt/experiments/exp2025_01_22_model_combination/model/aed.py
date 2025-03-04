from typing import Optional, Dict, Any, Sequence, Tuple, List, Union
import math
import copy

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoderLayer, ConformerConvSubsample, ConformerEncoder
import returnn.frontend

import i6_experiments
import i6_experiments.users.schmitt.experiments.exp2025_01_22_model_combination.model.conformer_tina

from .decoder import GlobalAttDecoder, GlobalAttEfficientDecoder

_batch_size_factor = 160


def get_conformer(opts: Dict) -> ConformerEncoder:
  opts = copy.deepcopy(opts)
  enc_cls = eval(opts.pop("class"))
  enc_input_layer_cls = eval(opts.pop("input_layer_cls")["class"])
  enc_out_dimension = opts.pop("out_dimension")
  in_dim = opts.pop("in_dim")
  enc_input_layer_opts = dict(
    out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
    filter_sizes=[(3, 3), (3, 3), (3, 3)],
    pool_sizes=[(1, 2)],
    strides=[(1, 1), (3, 1), (2, 1)],
  )
  enc_input_layer_opts.update(opts.pop("input_layer_opts", {}))
  encoder = enc_cls(
    in_dim,
    out_dim=Dim(name="enc", dimension=enc_out_dimension),
    ff_dim=Dim(name="enc-ff", dimension=2048),
    input_layer=enc_input_layer_cls(
      in_dim,
      **enc_input_layer_opts
    ),
    **opts
  )

  return encoder


class Model(rf.Module):
  def __init__(
          self,
          *,
          encoder_opts: Dict,
          in_dim: Dim,
          feature_extraction: Optional[str] = "log-mel-on-the-fly",
  ):
    super(Model, self).__init__()

    from returnn.config import get_global_config
    config = get_global_config(return_empty_if_none=True)

    self.in_dim = in_dim
    self.feature_extraction = feature_extraction

    # encoder
    encoder_opts.update({"in_dim": in_dim})
    self.encoder = get_conformer(encoder_opts)

    # specaugment
    self._specaugment_opts = {
      "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
      "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
      "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
                                      or (in_dim.dimension // 5),
      "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
    }

    apply_weight_dropout(self)

  def encode(
          self,
          source: Tensor,
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Tensor, Dim]:
    """encode, and extend the encoder output for things we need in the decoder"""
    # log mel filterbank features
    if self.feature_extraction == "log-mel-on-the-fly":
      source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
        raw_audio=source,
        in_spatial_dim=in_spatial_dim,
        out_dim=self.in_dim,
        log_base=math.exp(
          2.3026  # almost 10.0 but not exactly...
        )
      )
    else:
      assert self.feature_extraction is None

    # SpecAugment
    source = rf.audio.specaugment(
      source,
      spatial_dim=in_spatial_dim,
      feature_dim=self.in_dim,
      **self._specaugment_opts,
    )

    if collected_outputs is None:
      collected_outputs = {}

    enc, enc_spatial_dim = self.encoder(
      source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
    )

    return enc, enc_spatial_dim


class AEDModel(Model):
  def __init__(
          self,
          *,
          encoder_opts: Dict,
          decoder_opts: Dict,
          eos_idx: int,
          bos_idx: int,
          in_dim: Dim,
          target_dim: Dim,
          enc_aux_logits: Sequence[int] = (),
          feature_extraction: Optional[str] = "log-mel-on-the-fly",
  ):
    super(AEDModel, self).__init__(
      encoder_opts=encoder_opts,
      in_dim=in_dim,
      feature_extraction=feature_extraction,
    )

    self.bos_idx = bos_idx
    self.eos_idx = eos_idx
    # for CTC aux loss
    self.blank_idx = target_dim.dimension  # type: int
    self.target_dim = target_dim

    # auxiliary logits
    if enc_aux_logits:
      wb_target_dim = target_dim + 1
    for i in enc_aux_logits:
      setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))

    # decoder
    decoder_cls = eval(decoder_opts.pop("class"))
    self.decoder = decoder_cls(
      enc_out_dim=self.encoder.out_dim,
      target_dim=target_dim,
      **decoder_opts
    )

    apply_weight_dropout(self)

  def encode(
          self,
          source: Tensor,
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Dict[str, Tensor], Dim]:
    """encode, and extend the encoder output for things we need in the decoder"""

    enc, enc_spatial_dim = super(AEDModel, self).encode(
      source=source,
      in_spatial_dim=in_spatial_dim,
      collected_outputs=collected_outputs,
    )

    if self.decoder.enc_ctx_layer is None:
      enc_ctx = self.decoder.enc_ctx(enc)
    else:
      enc_ctx = self.decoder.enc_ctx(collected_outputs[self.decoder.enc_ctx_layer])
    if self.decoder.use_weight_feedback:
      inv_fertility = rf.sigmoid(self.decoder.inv_fertility(enc))
    else:
      inv_fertility = None

    return dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility), enc_spatial_dim


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


class MakeModel:
  """for import"""

  def __init__(
          self,
          in_dim: int,
          target_dim: int,
          eos_label: int = 0,
          **extra,
  ):
    self.in_dim = in_dim
    self.target_dim = target_dim
    self.eos_label = eos_label
    self.extra = extra

  def __call__(self) -> AEDModel:
    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
    target_dim = Dim(name="target", dimension=self.target_dim, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(
      [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
    )
    return self.make_model(in_dim, target_dim, **self.extra)

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          target_dim: Dim,
          *,
          encoder_opts: Dict = None,
          decoder_opts: Dict = None,
          enc_aux_logits: Sequence[int] = (),
          feature_extraction: Optional[str] = "log-mel-on-the-fly",

  ) -> AEDModel:
    """make"""

    return AEDModel(
      encoder_opts=encoder_opts,
      decoder_opts=decoder_opts,
      bos_idx=_get_bos_idx(target_dim),
      eos_idx=_get_eos_idx(target_dim),
      target_dim=target_dim,
      enc_aux_logits=enc_aux_logits,
      feature_extraction=feature_extraction,
      in_dim=in_dim,
    )


def aed_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> AEDModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  in_dim, epoch  # noqa
  config = get_global_config()  # noqa
  enc_aux_logits = config.typed_value("aux_loss_layers", ())
  encoder_opts = config.typed_value("encoder_opts")
  decoder_opts = config.typed_value("decoder_opts")

  feature_extraction = config.typed_value("feature_extraction", "log-mel-on-the-fly")
  feature_dimension = config.int("feature_dim", 80)
  if feature_extraction is None:
    feature_dim = config.typed_value("features_dim")
    in_dim = feature_dim
  else:
    assert feature_extraction == "log-mel-on-the-fly"
    in_dim = Dim(name="logmel", dimension=feature_dimension, kind=Dim.Types.Feature)

  return AEDModel(
    encoder_opts=encoder_opts,
    decoder_opts=decoder_opts,
    bos_idx=_get_bos_idx(target_dim),
    eos_idx=_get_eos_idx(target_dim),
    target_dim=target_dim,
    enc_aux_logits=enc_aux_logits,
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
  else:
    non_blank_target_dimension = config.typed_value("non_blank_target_dimension", None)
    vocab = config.typed_value("vocab", None)
    assert non_blank_target_dimension and vocab
    target_dim = Dim(description="non_blank_target_dim", dimension=non_blank_target_dimension, kind=Dim.Types.Spatial)
    targets = Tensor(name=default_target_key, sparse_dim=target_dim, vocab=vocab)

  model_def = config.typed_value("_model_def")
  model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
  return model


def apply_weight_dropout(model: rf.Module):
  from returnn.config import get_global_config
  config = get_global_config(return_empty_if_none=True)  # noqa

  weight_dropout = config.float("weight_dropout", None)
  if weight_dropout:
    # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
    # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
    blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
    blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
    for mod in model.modules():
      if isinstance(mod, blacklist):
        continue
      for param_name, param in mod.named_parameters(recurse=False):
        if param_name.endswith("bias"):  # no bias
          continue
        if param.auxiliary:
          continue
        rf.weight_dropout(mod, param_name, drop_prob=weight_dropout)
