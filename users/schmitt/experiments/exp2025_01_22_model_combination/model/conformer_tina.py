from typing import Optional, Dict, Any, Sequence, Tuple, List, Union
import functools
import math

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoderLayer, ConformerConvSubsample, ConformerEncoder


class ConformerEncoderLayerWithSwitchedOrder(ConformerEncoderLayer):
  """
  Represents a conformer block but with switched order of conv and mhsa.
  """

  def __call__(self, inp: Tensor, *, spatial_dim: Dim) -> Tensor:
    """forward"""
    # FFN
    x_ffn1_ln = self.ffn1_layer_norm(inp)
    x_ffn1 = self.ffn1(x_ffn1_ln)
    x_ffn1_out = 0.5 * rf.dropout(x_ffn1, self.dropout, axis=self.dropout_broadcast and self.out_dim) + inp

    # Conv
    x_conv_ln = self.conv_layer_norm(x_ffn1_out)
    x_conv = self.conv_block(x_conv_ln, spatial_dim=spatial_dim)
    x_conv_out = rf.dropout(x_conv, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_ffn1_out

    # MHSA
    x_mhsa_ln = self.self_att_layer_norm(x_conv_out)
    x_mhsa = self.self_att(x_mhsa_ln, axis=spatial_dim)
    x_mhsa = rf.dropout(x_mhsa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
    x_mhsa_out = x_mhsa + x_conv_out

    # FFN
    x_ffn2_ln = self.ffn2_layer_norm(x_mhsa_out)
    x_ffn2 = self.ffn2(x_ffn2_ln)
    x_ffn2_out = 0.5 * rf.dropout(x_ffn2, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_mhsa_out

    # last LN layer
    return self.final_layer_norm(x_ffn2_out)


class TinaConformerModel(rf.Module):
  def __init__(
          self,
          *,
          target_dim: Dim,
          in_dim: Dim,
  ):
    super(TinaConformerModel, self).__init__()

    self.in_dim = in_dim
    self.target_dim = target_dim

    enc_out_dim = Dim(name="enc", dimension=512)
    enc_ff_dim = Dim(name="enc-ff", dimension=2048)
    self.encoder = ConformerEncoder(
      in_dim,
      out_dim=enc_out_dim,
      ff_dim=enc_ff_dim,
      num_layers=12,
      input_layer=rf.encoder.conformer.ConformerConvSubsample(
        in_dim=in_dim,
        out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
        filter_sizes=[(3, 3), (3, 3), (3, 3)],
        pool_sizes=[(1, 2)],
        strides=[(1, 1), (2, 1), (2, 1)],
        activation=rf.swish,  # rf.relu
      ),
      encoder_layer=ConformerEncoderLayerWithSwitchedOrder(
        out_dim=enc_out_dim,
        ff_dim=enc_ff_dim,
        num_heads=8,
        conv_norm_opts=dict(
          use_mask=False,  # True,
          eps=1e-5,  # 1e-3
        ),
        self_att_opts=dict(
          # Shawn et al 2018 style, old RETURNN way.
          with_bias=False,
          with_linear_pos=False,
          with_pos_bias=False,
          learnable_pos_emb=True,
          separate_pos_emb_per_head=False,
          pos_emb_dropout=0.0,
          learnable_pos_emb_clipping=32,  # 16
        ),
        self_att=rf.RelPosSelfAttention,
        ff_activation=rf.swish,  # rf.relu_square,
      )
    )

  def encode(
          self,
          source: Tensor,
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Dict[str, Tensor], Dim]:
    """encode, and extend the encoder output for things we need in the decoder"""
    if collected_outputs is None:
      collected_outputs = {}

    enc, enc_spatial_dim = self.encoder(
      source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
    )

    return dict(enc=enc), enc_spatial_dim


class FramewiseProbModel(TinaConformerModel):
  def __init__(self, **kwargs):
    super(FramewiseProbModel, self).__init__(**kwargs)

    self.projection_matrix = rf.Linear(self.encoder.out_dim, self.target_dim)

  def __call__(self, x, *args, **kwargs):
    x = self.projection_matrix(x)
    x = rf.softmax(x, axis=self.target_dim)
    return x


class DiphoneFHModel(TinaConformerModel):
  def __init__(self, left_target_dim, **kwargs):
    super(DiphoneFHModel, self).__init__(**kwargs)

    self.left_target_dim = left_target_dim

    # left context (p(phi_l | x))
    self.linear1_left_context = rf.Linear(self.encoder.out_dim, self.encoder.out_dim)
    self.linear2_left_context = rf.Linear(self.encoder.out_dim, self.encoder.out_dim)
    self.left_output = rf.Linear(self.encoder.out_dim, left_target_dim)

    # center label (p(phi_c | phi_l, x))
    past_embed_dim = Dim(name="past-embed", dimension=32, kind=Dim.Types.Feature)
    self.past_embed = rf.Embedding(left_target_dim, past_embed_dim)
    linear_diphone_dim = Dim(name="linear1-diphone", dimension=544, kind=Dim.Types.Feature)
    self.linear1_diphone = rf.Linear(self.encoder.out_dim + past_embed_dim, linear_diphone_dim)
    self.linear2_diphone = rf.Linear(linear_diphone_dim, linear_diphone_dim)
    self.center_output = rf.Linear(linear_diphone_dim, self.target_dim)

  def _get_left_enc_rep(self, x):
    """

    :param x: output of encoder
    :return:
    """
    x = self.linear1_left_context(x)
    x = rf.relu(x)
    x = self.linear2_left_context(x)
    x = rf.relu(x)
    return x

  def get_left_probs(self, x):
    """

    :param x: output of encoder
    :return:
    """
    x = self._get_left_enc_rep(x)
    x = self.left_output(x)
    x = rf.softmax(x, axis=self.left_output.out_dim)
    return x

  def get_center_probs(self, enc: rf.Tensor, left_context: rf.Tensor):
    """

    :param enc: encoder output [B, T, F]
    :param left_context: left context label [B, ...] (sparse)
    :return:
    """
    emb = self.past_embed(left_context)
    res = self.linear1_diphone(rf.concat_features(enc, emb, allow_broadcast=True))
    res = rf.relu(res)
    res = self.linear2_diphone(res)
    res = rf.relu(res)
    res = self.center_output(res)
    return rf.softmax(res, axis=self.center_output.out_dim)


class MonophoneFHModel(TinaConformerModel):
  def __init__(self, **kwargs):
    super(MonophoneFHModel, self).__init__(**kwargs)

    # center label (p(phi_c | x))
    linear_diphone_dim = Dim(name="linear1-diphone", dimension=544, kind=Dim.Types.Feature)
    self.linear1_diphone = rf.Linear(self.encoder.out_dim, linear_diphone_dim)
    self.linear2_diphone = rf.Linear(linear_diphone_dim, linear_diphone_dim)
    self.center_output = rf.Linear(linear_diphone_dim, self.target_dim)

  def get_center_probs(self, enc: rf.Tensor):
    """

    :param enc: encoder output [B, T, F]
    :return:
    """
    res = self.linear1_diphone(enc)
    res = rf.relu(res)
    res = self.linear2_diphone(res)
    res = rf.relu(res)
    res = self.center_output(res)
    return rf.softmax(res, axis=self.center_output.out_dim)


class PhonTransducer(TinaConformerModel):
  def __init__(self, **kwargs):
    super(PhonTransducer, self).__init__(**kwargs)

    raise NotImplementedError


class MakeModel:
  """for import"""

  def __init__(
          self,
          in_dim: int,
          target_dim: int,
          **extra,
  ):
    self.in_dim = in_dim
    self.target_dim = target_dim
    self.extra = extra

  def __call__(self) -> TinaConformerModel:
    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
    target_dim = Dim(name="target", dimension=self.target_dim, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(
      [str(i) for i in range(target_dim.dimension)]
    )

    # map variables in extra ending with "_dim" to corresponding Dim objects
    extra_ = {}
    for k, v in self.extra.items():
      if k.endswith("_dim"):
        extra_[k] = Dim(name=k[:len("_dim")], dimension=v, kind=Dim.Types.Feature)

    return self.make_model(in_dim, target_dim, **extra_)

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          target_dim: Dim,
          **extra,
  ) -> TinaConformerModel:
    """make"""

    return TinaConformerModel(
      target_dim=target_dim,
      in_dim=in_dim,
    )


class MakeFramewiseProbModel(MakeModel):
  """for import"""

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          target_dim: Dim,
          **extra,
  ) -> FramewiseProbModel:
    """make"""

    return FramewiseProbModel(
      target_dim=target_dim,
      in_dim=in_dim,
    )


class MakeDiphoneFHModel(MakeModel):
  """for import"""

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          target_dim: Dim,
          **extra,
  ) -> DiphoneFHModel:
    """make"""

    left_target_dim = extra.pop("left_target_dim")
    assert not extra

    return DiphoneFHModel(
      target_dim=target_dim,
      in_dim=in_dim,
      left_target_dim=left_target_dim,
    )


class MakeMonophoneFHModel(MakeModel):
  """for import"""

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          target_dim: Dim,
          **extra,
  ) -> MonophoneFHModel:
    """make"""

    return MonophoneFHModel(
      target_dim=target_dim,
      in_dim=in_dim,
    )


class MakePhonTransducerModel(MakeModel):
  """for import"""

  @classmethod
  def make_model(
          cls,
          in_dim: Dim,
          target_dim: Dim,
          **extra,
  ) -> PhonTransducer:
    """make"""

    return PhonTransducer(
      target_dim=target_dim,
      in_dim=in_dim,
    )


def get_common_params():
  from returnn.config import get_global_config
  config = get_global_config()  # noqa

  feature_extraction = config.typed_value("feature_extraction", "log-mel-on-the-fly")
  feature_dimension = config.int("feature_dim", 80)
  if feature_extraction is None:
    feature_dim = config.typed_value("features_dim")
    in_dim = feature_dim
  else:
    assert feature_extraction == "log-mel-on-the-fly"
    in_dim = Dim(name="logmel", dimension=feature_dimension, kind=Dim.Types.Feature)

  target_dimension = config.int("target_dimension", None)
  target_dim = Dim(description="target_dim", dimension=target_dimension, kind=Dim.Types.Spatial)

  return dict(
    in_dim=in_dim,
    target_dim=target_dim
  )


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> TinaConformerModel:
  """Function is run within RETURNN."""
  in_dim, epoch  # noqa

  common_params = get_common_params()

  return MakeModel.make_model(
    **common_params,
  )


def from_scratch_framewise_prob_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> TinaConformerModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config
  config = get_global_config()  # noqa

  in_dim, epoch  # noqa

  common_params = get_common_params()

  return MakeFramewiseProbModel.make_model(
    **common_params,
  )


def from_scratch_phon_transducer_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> TinaConformerModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config
  config = get_global_config()  # noqa

  in_dim, epoch  # noqa

  common_params = get_common_params()

  return MakePhonTransducerModel.make_model(
    **common_params,
  )


def from_scratch_monophone_fh_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> TinaConformerModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config
  config = get_global_config()  # noqa

  in_dim, epoch  # noqa

  common_params = get_common_params()

  return MakeMonophoneFHModel.make_model(
    **common_params,
  )


def from_scratch_diphone_fh_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> TinaConformerModel:
  """Function is run within RETURNN."""
  from returnn.config import get_global_config
  config = get_global_config()  # noqa

  in_dim, epoch  # noqa

  common_params = get_common_params()

  left_target_dimension = config.int("left_target_dimension", None)
  left_target_dim = Dim(description="left_target_dim", dimension=left_target_dimension, kind=Dim.Types.Spatial)

  return MakeDiphoneFHModel.make_model(
    left_target_dim=left_target_dim,
    **common_params,
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
