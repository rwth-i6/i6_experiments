from typing import Optional, Dict, Any, Sequence, Tuple, List, Union
import math
import copy

from returnn.tensor import Tensor, Dim, single_step_dim, TensorDict
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoderLayer, ConformerConvSubsample, ConformerEncoder
import returnn.frontend

import i6_experiments
import i6_experiments.users.schmitt.experiments.exp2025_01_22_model_combination.model.conformer_tina

from .decoder import GlobalAttDecoder, GlobalAttEfficientDecoder
from .. import tensor_utils

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


# def _returnn_v2_rescore_step(*, model, extern_data: TensorDict, **_kwargs_unused):
#   from returnn.config import get_global_config
#
#   config = get_global_config()
#   default_input_key = config.typed_value("default_input")
#   default_target_key = config.typed_value("target")
#   data = extern_data[default_input_key]
#   data_spatial_dim = data.get_time_dim_tag()
#   targets = extern_data[default_target_key]
#   targets_spatial_dim = targets.get_time_dim_tag()
#   train_def = config.typed_value("_train_def")
#   train_def(
#     model=model,
#     data=data,
#     data_spatial_dim=data_spatial_dim,
#     targets=targets,
#     targets_spatial_dim=targets_spatial_dim,
#   )

def _returnn_v2_rescore_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  # Similar to i6_experiments.users.zeyer.recog._returnn_v2_forward_step,
  # but using score_def instead of recog_def.
  import returnn.frontend as rf
  from returnn.tensor import Tensor, Dim, batch_dim
  from returnn.config import get_global_config

  if rf.is_executing_eagerly():
    batch_size = int(batch_dim.get_dim_value())
    for batch_idx in range(batch_size):
      seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
      print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  if default_input_key:
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
  else:
    data, data_spatial_dim = None, None

  targets_beam_dim = config.typed_value("beam_dim")
  targets_flat = extern_data["data_flat"]
  targets_flat_time_dim = config.typed_value("data_flat_spatial_dim")
  targets_seq_lens = extern_data["data_seq_lens"]  # [B, beam]
  # TODO stupid that targets_seq_lens first is copied CPU->GPU and now back to CPU...
  targets_spatial_dim = Dim(rf.copy_to_device(targets_seq_lens, "cpu"), name="targets_spatial")
  targets = rf.pad_packed(targets_flat, in_dim=targets_flat_time_dim, dims=[targets_beam_dim, targets_spatial_dim])

  rescore_def = config.typed_value("_rescore_def")
  scores = rescore_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    targets=targets,
    targets_spatial_dim=targets_spatial_dim,
    targets_beam_dim=targets_beam_dim,
  )

  assert isinstance(scores, Tensor)
  rf.get_run_ctx().mark_as_output(targets, "hyps", dims=[batch_dim, targets_beam_dim, targets_spatial_dim])
  rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, targets_beam_dim])


def get_s_and_att(
        *,
        model: GlobalAttDecoder,
        enc_args: Dict[str, Tensor],
        input_embeddings: Tensor,
        enc_spatial_dim: Dim,
        targets_spatial_dim: Dim,
        batch_dims: List[Dim],
) -> Tuple[Tensor, Tensor, rf.State]:

  from returnn.config import get_global_config
  config = get_global_config()

  hard_att_opts = config.typed_value("hard_att_opts", None)

  def _body(xs, state: rf.State):
    new_state = rf.State()
    loop_out_, new_state.decoder = model.loop_step(
      **enc_args,
      enc_spatial_dim=enc_spatial_dim,
      input_embed=xs["input_embed"],
      state=state.decoder,
      hard_att_opts=hard_att_opts,
    )
    return loop_out_, new_state

  xs = {"input_embed": input_embeddings}
  loop_out, final_state, _ = rf.scan(
    spatial_dim=targets_spatial_dim,
    xs=xs,
    ys=model.loop_step_output_templates(
      batch_dims=batch_dims),
    initial=rf.State(
      decoder=model.decoder_default_initial_state(
        batch_dims=batch_dims,
        enc_spatial_dim=enc_spatial_dim,
      ),
    ),
    body=_body,
  )

  return loop_out["s"], loop_out["att"], final_state


def forward_sequence(
        model: Union[GlobalAttDecoder],
        targets: rf.Tensor,
        targets_spatial_dim: Dim,
        enc_args: Dict[str, rf.Tensor],
        enc_spatial_dim: Dim,
        batch_dims: List[Dim],
) -> rf.Tensor:
  input_embeddings = model.target_embed(targets)
  input_embeddings = rf.shift_right(input_embeddings, axis=targets_spatial_dim, pad_value=0.0)
  input_embeddings = rf.dropout(input_embeddings, drop_prob=model.target_embed_dropout, axis=None)

  s, att, final_state = get_s_and_att(
    model=model,
    enc_args=enc_args,
    input_embeddings=input_embeddings,
    enc_spatial_dim=enc_spatial_dim,
    targets_spatial_dim=targets_spatial_dim,
    batch_dims=batch_dims,
  )

  logits = model.decode_logits(
    input_embed=input_embeddings,
    s=s,
    att=att,
  )

  return logits


import torch
def rescore(
        *,
        model: AEDModel,
        data: Tensor,
        data_spatial_dim: Dim,
        targets: Tensor,
        targets_beam_dim: Dim,
        targets_spatial_dim: Dim,
        max_approx: bool = False,
):
  """Function is run within RETURNN."""
  from returnn.config import get_global_config

  config = get_global_config()  # noqa

  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    batch_dims = data.remaining_dims(data_spatial_dim)
  else:
    batch_dims = data.remaining_dims([data_spatial_dim, data.feature_dim])

  collected_outputs = {}
  enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)

  batch_dimension = batch_dims[0].dyn_size_ext.raw_tensor.item()
  spatial_dimension = targets_spatial_dim.dyn_size_ext.raw_tensor.max().item()
  beam_dimension = targets_beam_dim.get_dim_value().item()

  log_probs_tensor = rf.Tensor(
    "log_probs",
    dims=batch_dims + [targets_beam_dim],
    dtype="float32",
    raw_tensor=torch.zeros((batch_dimension, beam_dimension), dtype=torch.float32)
  )
  for beam_idx in range(targets_beam_dim.get_dim_value()):
    targets_b = rf.gather(targets, axis=targets_beam_dim, indices=beam_idx)
    targets_b_seq_lens = rf.gather(targets_spatial_dim.dyn_size_ext, axis=targets_beam_dim, indices=beam_idx)
    targets_b_spatial_dim = Dim(targets_b_seq_lens, name=f"{targets_spatial_dim.name}_beam{beam_idx}")
    targets_b, _ = rf.replace_dim(targets_b, in_dim=targets_spatial_dim, out_dim=targets_b_spatial_dim)
    targets_b, _ = rf.slice(targets_b, axis=targets_b_spatial_dim, size=targets_b_spatial_dim)
    logits = forward_sequence(
      model=model.decoder,
      targets=targets_b,
      targets_spatial_dim=targets_b_spatial_dim,
      enc_args=enc_args,
      enc_spatial_dim=enc_spatial_dim,
      batch_dims=batch_dims
    )
    log_prob = rf.log_softmax(logits, axis=model.target_dim)

    # get negative log probs
    neg_log_prob = rf.gather(log_prob, axis=model.target_dim, indices=targets_b)
    print("targets_b: ", targets_b.raw_tensor)
    neg_log_prob = rf.reduce_sum(neg_log_prob, axis=targets_b_spatial_dim)
    print("neg_log_prob: ", neg_log_prob.raw_tensor)
    neg_log_prob /= rf.copy_to_device(targets_b_spatial_dim.dyn_size_ext, neg_log_prob.device)
    print("neg_log_prob: ", neg_log_prob.raw_tensor)
    exit()

    # negate to get log probs
    log_probs_tensor.raw_tensor[:, beam_idx] = -neg_log_prob.raw_tensor

    # print("loss: ", loss)
    # print("loss.shape: ", loss.raw_tensor.shape)
    # exit()
    #
    # for batch_idx in range(batch_dimension):
    #   loss_batch = loss.raw_tensor[batch_idx]
    #   log_probs_tensor.raw_tensor[batch_idx, beam_idx, :targets_b_spatial_dim.dyn_size_ext.raw_tensor[batch_idx]] = loss_batch

  return log_probs_tensor
