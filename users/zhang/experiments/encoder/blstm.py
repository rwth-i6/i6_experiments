from __future__ import annotations

import copy

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import torch
from torch import nn
from torchaudio.functional import mask_along_axis

from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf
import returnn.torch.frontend as rtf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder
from returnn.frontend.encoder.conformer import ConformerConvSubsample

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zhang.experiments.ctc import model_recog
from i6_experiments.users.zhang.experiments.lm.ffnn import FFNN_LM_flashlight, FeedForwardLm
from returnn.frontend.decoder.transformer import TransformerDecoder

if TYPE_CHECKING:
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
    from i6_experiments.users.zeyer.datasets.task import Task
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput

from ..configs import *
from ..configs import _get_cfg_lrlin_oclr_by_bs_nep, _batch_size_factor


_log_mel_feature_dim = 80


def py():
    pass


_train_experiments: Dict[str, ModelWithCheckpoints] = {}

def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.sis_setup import get_prefix_for_config

        prefix_name = get_prefix_for_config(__file__)
    global _sis_prefix
    _sis_prefix = prefix_name


def _remove_eos_label_v2(res: RecogOutput) -> RecogOutput:
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
    from i6_core.returnn.search import SearchRemoveLabelJob

    return RecogOutput(SearchRemoveLabelJob(res.output, remove_label="</s>", output_gzip=True).out_search_results)


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

def flip(source: Tensor, *, axis: Dim) -> Tensor:
    """flip, ignoring masking"""
    axis_int = source.get_axis_from_description(axis, allow_int=False)
    out = source.copy_template("flip")
    out.raw_tensor = torch.flip(source.raw_tensor, [axis_int])
    return out

rf.flip = flip

class BlstmEncoder(ISeqDownsamplingEncoder):
    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        *,
        num_layers: int,
        input_dropout: float = 0.1,
        input_layer: Optional[Union[ConformerConvSubsample, ISeqDownsamplingEncoder, rf.Module, Any]],
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_dropout = input_dropout
        # TODO not sure if this is correct
        self.dropout_broadcast = rf.dropout_broadcast_default()

        # TODO once we figured out good defaults, we would create ConformerConvSubsample here when not given
        if callable(input_layer) or input_layer is None:
            pass  # leave it as is
        elif isinstance(input_layer, dict):
            input_layer = rf.build_from_dict(input_layer, in_dim)
            input_layer: ConformerConvSubsample  # maybe not true, but assume for some attribs
        else:
            raise TypeError(f"unexpected input_layer {input_layer!r}")
        self.input_layer = input_layer
        self.input_projection = (
            rf.Linear(self.input_layer.out_dim if self.input_layer else self.in_dim, self.out_dim, with_bias=False)
            if input_layer
            else None
        )

        self.input_forward_lstm = rf.LSTM(out_dim, out_dim)
        self.input_backward_lstm = rf.LSTM(out_dim, out_dim)

        self.forward_lstm_layers = rf.Sequential(rf.LSTM(2 * out_dim, out_dim) for _ in range(num_layers - 1))
        self.backward_lstm_layers = rf.Sequential(rf.LSTM(2 * out_dim, out_dim) for _ in range(num_layers - 1))

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ):
        if self.input_layer:
            x_subsample, out_spatial_dim = self.input_layer(source, in_spatial_dim=in_spatial_dim)
        else:
            x_subsample, out_spatial_dim = source, in_spatial_dim

        x = self.input_projection(x_subsample) if self.input_projection else x_subsample
        x = rf.dropout(x, self.input_dropout, axis=self.dropout_broadcast and self.out_dim)

        x_flipped = rf.flip(x, axis=out_spatial_dim)  # reverse input

        #import pdb; pdb.set_trace()
        #batch_dims = [d for d in source.dims if d != out_spatial_dim and d != self.in_dim]
        batch_dims = [source.dims[0]]
        x_fwd_lstm, _ = self.input_forward_lstm(x, state=self.input_forward_lstm.default_initial_state(batch_dims=batch_dims),
                                                spatial_dim=out_spatial_dim)
        x_bwd_lstm, _ = self.input_backward_lstm(x_flipped, state=self.input_backward_lstm.default_initial_state(batch_dims=batch_dims),
                                                 spatial_dim=out_spatial_dim)

        for i, (fwd_lstm_layer, bwd_lstm_layer) in enumerate(zip(self.forward_lstm_layers, self.backward_lstm_layers)):
            lstm_inp, _ = rf.concat((x_fwd_lstm,self.out_dim), (x_bwd_lstm,self.out_dim))
            lstm_inp_flipped = rf.flip(lstm_inp, axis=out_spatial_dim)  # reverse input
            x_fwd_lstm, _ = fwd_lstm_layer(lstm_inp, state=fwd_lstm_layer.default_initial_state(batch_dims=batch_dims),
                                           spatial_dim=out_spatial_dim)
            x_bwd_lstm, _ = bwd_lstm_layer(lstm_inp_flipped, state=bwd_lstm_layer.default_initial_state(batch_dims=batch_dims),
                                           spatial_dim=out_spatial_dim)
            if collected_outputs is not None:
                collected_outputs[str(i)] = x_fwd_lstm
                collected_outputs[str(i+5)] = x_bwd_lstm
        return rf.concat((x_fwd_lstm,self.out_dim), (x_bwd_lstm,self.out_dim))[0], out_spatial_dim


def ctc_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    num_enc_layers = config.int("num_enc_layers", 6)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)

    return Model(
        in_dim,
        num_enc_layers=num_enc_layers,
        enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
        enc_other_opts=None,
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        enc_aux_logits=enc_aux_logits or (),
    )


ctc_model_def: ModelDef[Model]
ctc_model_def.behavior_version = 21
ctc_model_def.backend = "torch"
ctc_model_def.batch_size_factor = _batch_size_factor


# def ctc_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim):
#     """Function is run within RETURNN."""
#     from returnn.config import get_global_config
#
#     config = get_global_config()  # noqa
#     aux_loss_layers = config.typed_value("aux_loss_layers")
#     aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
#     use_normalized_loss = config.bool("use_normalized_loss", True)
#     use_fixed_ctc_grad = config.typed_value("use_fixed_ctc_grad", False)
#
#     ctc_loss = rf.ctc_loss
#     if use_fixed_ctc_grad:
#         from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad
#
#         assert use_fixed_ctc_grad == "v2"  # v2 has the fix for scaled/normalized CTC loss
#         ctc_loss = ctc_loss_fixed_grad
#
#     if data.feature_dim and data.feature_dim.dimension == 1:
#         data = rf.squeeze(data, axis=data.feature_dim)
#     assert not data.feature_dim  # raw audio
#
#     if config.bool("use_eos_postfix", False):
#         targets, (targets_spatial_dim,) = rf.pad(
#             targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
#         )
#
#     collected_outputs = {} if aux_loss_layers else None
#     logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
#     if aux_loss_layers:
#         for i, layer_idx in enumerate(aux_loss_layers):
#             if layer_idx > len(model.encoder.layers):
#                 continue
#             aux_logits = model.aux_logits_from_collected_outputs(layer_idx, collected_outputs)
#             aux_log_probs = model.log_probs_wb_from_logits(aux_logits, aux_layer=layer_idx)
#             aux_loss = ctc_loss(
#                 logits=aux_log_probs,
#                 logits_normalized=True,
#                 targets=targets,
#                 input_spatial_dim=enc_spatial_dim,
#                 targets_spatial_dim=targets_spatial_dim,
#                 blank_index=model.blank_idx,
#             )
#             aux_loss.mark_as_loss(
#                 f"ctc_{layer_idx}",
#                 scale=aux_loss_scales[i],
#                 custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
#                 use_normalized_loss=use_normalized_loss,
#             )
#
#     log_probs = model.log_probs_wb_from_logits(logits)
#     loss = ctc_loss(
#         logits=log_probs,
#         logits_normalized=True,
#         targets=targets,
#         input_spatial_dim=enc_spatial_dim,
#         targets_spatial_dim=targets_spatial_dim,
#         blank_index=model.blank_idx,
#     )
#     loss.mark_as_loss(
#         "ctc",
#         custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
#         use_normalized_loss=use_normalized_loss,
#     )
#
#
# ctc_training: TrainDef[Model]
# ctc_training.learning_rate_control_error_measure = "ctc"



class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_build_dict: Optional[Dict[str, Any]] = None,
        enc_aux_logits: Sequence[int] = (),  # layers, 1-indexed
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_input_layer: Optional[Dict[str, Any]] = None,
        enc_other_opts: Optional[Dict[str, Any]] = None,
        recog_language_model: Optional[FeedForwardLm | TransformerDecoder] = None,
    ):
        super(Model, self).__init__()

        self.in_dim = in_dim
        self.num_enc_layers = num_enc_layers

        import numpy
        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        if enc_build_dict:
            # Warning: We ignore the other args (num_enc_layers, enc_model_dim, enc_other_opts, etc).
            self.encoder = rf.build_from_dict(enc_build_dict, in_dim)
            self.encoder: BlstmEncoder  # might not be true, but assume similar/same interface

        else:
            if not enc_input_layer:
                enc_input_layer = ConformerConvSubsample(
                    in_dim,
                    out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],
                )

            enc_opts = {"input_layer": enc_input_layer, "num_layers": num_enc_layers}

            if enc_other_opts:
                for k, v in enc_other_opts.items():
                    assert k not in enc_opts, f"enc_other_opts key {k!r} already in enc_opts {enc_opts}"
                    enc_opts[k] = v

            self.encoder = BlstmEncoder(in_dim, enc_model_dim, **enc_opts)

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        if not wb_target_dim:
            wb_target_dim = target_dim + 1
        self.enc_aux_selected_layers = enc_aux_logits
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))
        # TODO Make sure it is right to do this
        self.enc_logits = rf.Linear(2*self.encoder.out_dim, wb_target_dim)
        #----------------------------------
        self.wb_target_dim = wb_target_dim
        self.out_blank_separated = config.bool("out_blank_separated", False)
        self.blank_logit_shift = config.float("blank_logit_shift", 0.0)

        self.ctc_am_scale = config.float("ctc_am_scale", 1.0)
        self.ctc_prior_scale = config.float("ctc_prior_scale", 0.0)
        self.ctc_prior_type = config.value("ctc_prior_type", "batch")

        static_prior = config.typed_value("static_prior")
        self.static_prior = None  # in log prob, if set
        if static_prior:
            assert isinstance(static_prior, dict)
            assert set(static_prior.keys()) == {"file", "type"}
            v = numpy.loadtxt(static_prior["file"])
            # The `type` is about what is stored in the file.
            # We always store it in log prob here, so we potentially need to convert it.
            if static_prior["type"] == "log_prob":
                pass  # already log prob
            elif static_prior["type"] == "prob":
                v = numpy.log(v)
            else:
                raise ValueError(f"invalid static_prior type {static_prior['type']!r}")
            self.static_prior = rf.Parameter(
                rf.convert_to_tensor(v, dims=[self.wb_target_dim], dtype=rf.get_default_float_dtype()),
                auxiliary=True,
                non_critical_for_restore=True,
            )
        self.prior_running_mean_momentum = config.typed_value("prior_running_mean_momentum", None)
        self.prior_running_mean_per_layer = config.bool("prior_running_mean_per_layer", False)
        self.prior_running_mean = None  # in std prob, if set
        if self.prior_running_mean_momentum is not None:
            self.prior_running_mean = rf.Parameter(
                [self.wb_target_dim], auxiliary=True, initial=1.0 / self.wb_target_dim.dimension
            )
            if self.prior_running_mean_per_layer:
                for i in enc_aux_logits:
                    setattr(
                        self,
                        f"prior_running_mean_{i}",
                        rf.Parameter([self.wb_target_dim], auxiliary=True, initial=1.0 / self.wb_target_dim.dimension),
                    )

        if target_dim.vocab and not wb_target_dim.vocab:
            from returnn.datasets.util.vocabulary import Vocabulary

            # Just assumption for code now, might extend this later.
            assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
            vocab_labels = list(target_dim.vocab.labels) + [model_recog.output_blank_label]
            wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                vocab_labels, user_defined_symbols={model_recog.output_blank_label: blank_idx}
            )

        ctc_label_smoothing = config.float("ctc_label_smoothing", 0.0)
        ctc_label_smoothing_exclude_blank = config.bool("ctc_label_smoothing_exclude_blank", self.out_blank_separated)
        self.ctc_label_smoothing_exclude_blank = ctc_label_smoothing_exclude_blank
        if not self.out_blank_separated:
            self.ctc_label_smoothing_opts = {
                "smoothing": ctc_label_smoothing,
                "axis": self.wb_target_dim,
                "exclude_labels": [self.blank_idx] if ctc_label_smoothing_exclude_blank else None,
            }
        else:  # separate blank
            self.ctc_label_smoothing_opts = {
                "smoothing": ctc_label_smoothing,
                "axis": self.target_dim if ctc_label_smoothing_exclude_blank else self.wb_target_dim,
            }
        self.log_prob_normed_grad_opts = config.typed_value("log_prob_normed_grad", None)
        self.log_prob_normed_grad_exclude_blank = config.bool(
            "log_prob_normed_grad_exclude_blank", self.out_blank_separated
        )
        self.grad_prior = None  # in std prob, if set
        if (
            self.log_prob_normed_grad_opts
            and self.log_prob_normed_grad_opts.get("prior_running_mean_momentum") is not None
        ):
            # Note: We might want to use the static_prior for this purpose here.
            # However, there are some differences, and it would probably just cause confusion and potential bugs.
            # - static_prior is in log space, but here we want std prob space.
            # - static_prior is supposed to be static, also with non_critical_for_restore=True.
            _grad_prior_dim = self.target_dim if self.log_prob_normed_grad_exclude_blank else self.wb_target_dim
            self.grad_prior = rf.Parameter([_grad_prior_dim], auxiliary=True, initial=1.0 / _grad_prior_dim.dimension)
            if self.prior_running_mean_per_layer:
                for i in enc_aux_logits:
                    setattr(
                        self,
                        f"grad_prior_{i}",
                        rf.Parameter([_grad_prior_dim], auxiliary=True, initial=1.0 / _grad_prior_dim.dimension),
                    )

        self.feature_batch_norm = None
        if config.bool("feature_batch_norm", False):
            self.feature_batch_norm = rf.BatchNorm(self.in_dim, affine=False, use_mask=True)
        self.feature_norm = config.bool("feature_norm", False)
        self.feature_stats = None
        feature_stats = config.typed_value("feature_stats")
        if feature_stats:
            assert isinstance(feature_stats, dict)
            self.feature_stats = rf.ParameterList(
                {
                    k: rf.Parameter(
                        rf.convert_to_tensor(numpy.loadtxt(v), dims=[self.in_dim], dtype=rf.get_default_float_dtype()),
                        auxiliary=True,
                        non_critical_for_restore=True,
                    )
                    for k, v in feature_stats.items()
                }
            )

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

        self.decoder = None
        aux_attention_decoder = config.typed_value("aux_attention_decoder", None)
        if aux_attention_decoder:
            # Auxiliary attention decoder for regularization.
            # "Keep Decoding Parallel With Effective Knowledge Distillation
            #  From Language Models To End-To-End Speech Recognisers", 2024
            # https://ieeexplore.ieee.org/document/10447305
            assert isinstance(aux_attention_decoder, dict)
            aux_attention_decoder = aux_attention_decoder.copy()
            aux_attention_decoder.setdefault("class", "returnn.frontend.decoder.transformer.TransformerDecoder")
            if isinstance(aux_attention_decoder.get("model_dim", None), int):
                aux_attention_decoder["model_dim"] = Dim(aux_attention_decoder["model_dim"], name="dec_model")
            self.decoder = rf.build_from_dict(
                aux_attention_decoder, encoder_dim=self.encoder.out_dim, vocab_dim=target_dim
            )

        vn = config.typed_value("variational_noise", None)
        if vn:
            # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
            # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
            blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
            blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
            for mod in self.modules():
                if isinstance(mod, blacklist):
                    continue
                for param_name, param in mod.named_parameters(recurse=False):
                    if param_name.endswith("bias"):  # no bias
                        continue
                    if param.auxiliary:
                        continue
                    rf.weight_noise(mod, param_name, std=vn)

        weight_dropout = config.typed_value("weight_dropout", None)
        if weight_dropout:
            # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
            # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
            blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
            blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
            for mod in self.modules():
                if isinstance(mod, blacklist):
                    continue
                for param_name, param in mod.named_parameters(recurse=False):
                    if param_name.endswith("bias"):  # no bias
                        continue
                    if param.auxiliary:
                        continue
                    rf.weight_dropout(mod, param_name, drop_prob=weight_dropout)
        self.recog_language_model = recog_language_model

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Dim]:
        """
        Encode, get CTC logits.
        Use :func:`log_probs_wb_from_logits` to get log probs
        (might be just log_softmax, but there are some other cases).

        :return: logits, enc, enc_spatial_dim
        """
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
        )
        if self.feature_batch_norm:
            source = self.feature_batch_norm(source)
        if self.feature_norm:
            source = rf.normalize(source, axis=in_spatial_dim)
        if self.feature_stats:
            source = (source - self.feature_stats.mean) / self.feature_stats.std_dev
        if self._mixup:
            source = self._mixup(source, spatial_dim=in_spatial_dim)
        # SpecAugment
        source = rf.audio.specaugment(
            source,
            spatial_dim=in_spatial_dim,
            feature_dim=self.in_dim,
            **self._specaugment_opts,
        )
        # Encoder including convolutional frontend
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        logits = self.enc_logits(enc)
        return logits, enc, enc_spatial_dim

    def aux_logits_from_collected_outputs(self, aux_layer: int, collected_outputs: Dict[str, Tensor]) -> Tensor:
        """
        :param aux_layer:
        :param collected_outputs: from __call__
        :return: logits
        """
        linear: rf.Linear = getattr(self, f"enc_aux_logits_{aux_layer}")
        aux_logits = linear(collected_outputs[str(aux_layer - 1)])
        return aux_logits

    def log_probs_wb_from_logits(self, logits: Tensor, *, aux_layer: Optional[int] = None) -> Tensor:
        """
        :param logits: incl blank
        :param aux_layer: whether the logits come from some intermediate aux layer.
            That might influence the prior.
        :return: log probs with blank from logits (wb_target_dim)
            If out_blank_separated, we use a separate sigmoid for the blank.
            Also, potentially adds label smoothing on the gradients.
        """
        if not self.out_blank_separated:  # standard case, joint distrib incl blank
            if self.blank_logit_shift:
                logits += rf.sparse_to_dense(
                    self.blank_idx, label_value=self.blank_logit_shift, other_value=0, axis=self.wb_target_dim
                )
            log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)
        else:  # separate blank
            assert self.blank_idx == self.target_dim.dimension  # not implemented otherwise
            dummy_blank_feat_dim = Dim(1, name="blank_feat")
            logits_wo_blank, logits_blank = rf.split(
                logits, axis=self.wb_target_dim, out_dims=[self.target_dim, dummy_blank_feat_dim]
            )
            log_probs_wo_blank = rf.log_softmax(logits_wo_blank, axis=self.target_dim)
            log_probs_wo_blank = self._maybe_apply_on_log_probs(log_probs_wo_blank, aux_layer=aux_layer)
            if self.blank_logit_shift:
                logits_blank += self.blank_logit_shift
            log_probs_blank = rf.log_sigmoid(logits_blank)
            log_probs_emit = rf.squeeze(rf.log_sigmoid(-logits_blank), axis=dummy_blank_feat_dim)
            log_probs, _ = rf.concat(
                (log_probs_wo_blank + log_probs_emit, self.target_dim),
                (log_probs_blank, dummy_blank_feat_dim),
                out_dim=self.wb_target_dim,
            )
            log_probs.feature_dim = self.wb_target_dim

        prior_running_mean = None
        if self.prior_running_mean_momentum is not None:
            prior_running_mean = self.prior_running_mean
            if self.prior_running_mean_per_layer and aux_layer is not None:
                prior_running_mean = getattr(self, f"prior_running_mean_{aux_layer}")

            def _update_running_stats():
                batch_prior = rf.reduce_mean(
                    rf.exp(log_probs), axis=[d for d in log_probs.dims if d != self.wb_target_dim]
                )
                assert batch_prior.dims == (self.wb_target_dim,)
                prior_running_mean.assign_add(self.prior_running_mean_momentum * (batch_prior - prior_running_mean))

            rf.cond(rf.get_run_ctx().train_flag, _update_running_stats, lambda: None)

        log_probs = self._maybe_apply_on_log_probs(log_probs, aux_layer=aux_layer)
        if self.ctc_am_scale == 1 and self.ctc_prior_scale == 0:  # fast path
            return log_probs
        log_probs_am = log_probs
        log_probs = log_probs_am * self.ctc_am_scale
        if self.ctc_prior_scale:
            if self.ctc_prior_type == "batch":
                # Warning: this is sum, but we want mean!
                log_prob_prior = rf.reduce_logsumexp(
                    log_probs_am, axis=[dim for dim in log_probs_am.dims if dim != self.wb_target_dim]
                )
                assert log_prob_prior.dims == (self.wb_target_dim,)
            elif self.ctc_prior_type == "batch_fixed":
                log_prob_prior = rf.reduce_logmeanexp(
                    log_probs_am, axis=[dim for dim in log_probs_am.dims if dim != self.wb_target_dim]
                )
                assert log_prob_prior.dims == (self.wb_target_dim,)
            elif self.ctc_prior_type == "batch_stop_grad":
                log_prob_prior = rf.stop_gradient(
                    rf.reduce_logmeanexp(
                        log_probs_am, axis=[dim for dim in log_probs_am.dims if dim != self.wb_target_dim]
                    )
                )
                assert log_prob_prior.dims == (self.wb_target_dim,)
            elif self.ctc_prior_type == "seq":
                log_prob_prior = rf.reduce_logmeanexp(
                    log_probs_am, axis=[dim for dim in log_probs_am.dims if dim not in (batch_dim, self.wb_target_dim)]
                )
                assert log_prob_prior.dims_set == {batch_dim, self.wb_target_dim}
            elif self.ctc_prior_type == "seq_stop_grad":
                log_prob_prior = rf.stop_gradient(
                    rf.reduce_logmeanexp(
                        log_probs_am,
                        axis=[dim for dim in log_probs_am.dims if dim not in (batch_dim, self.wb_target_dim)],
                    )
                )
                assert log_prob_prior.dims_set == {batch_dim, self.wb_target_dim}
            elif self.ctc_prior_type == "static":
                log_prob_prior = self.static_prior
                assert log_prob_prior.dims == (self.wb_target_dim,)
            elif self.ctc_prior_type == "running_mean":
                assert prior_running_mean is not None
                log_prob_prior = rf.safe_log(prior_running_mean)
                assert log_prob_prior.dims == (self.wb_target_dim,)
            else:
                raise ValueError(f"invalid ctc_prior_type {self.ctc_prior_type!r}")
            log_probs -= log_prob_prior * self.ctc_prior_scale
        return log_probs

    def _maybe_apply_on_log_probs(self, log_probs: Tensor, *, aux_layer: Optional[int] = None) -> Tensor:
        """
        :param log_probs: either with blank or without blank
        :param aux_layer:
        :return: log probs, maybe some smoothing applied (all on gradients so far, not on log probs itself)
        """
        assert log_probs.feature_dim in (self.wb_target_dim, self.target_dim)
        if not self.out_blank_separated:
            assert log_probs.feature_dim == self.wb_target_dim

        log_probs = self._maybe_apply_log_probs_normed_grad(log_probs, aux_layer=aux_layer)

        if self.ctc_label_smoothing_exclude_blank:
            if self.out_blank_separated:
                if log_probs.feature_dim == self.target_dim:
                    log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)
            else:
                assert log_probs.feature_dim == self.wb_target_dim
                assert self.ctc_label_smoothing_opts["exclude_labels"] == [self.blank_idx]
                log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)
        else:
            if log_probs.feature_dim == self.wb_target_dim:
                log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)

        return log_probs

    def _maybe_apply_log_probs_normed_grad(self, log_probs: Tensor, *, aux_layer: Optional[int] = None) -> Tensor:
        if not self.log_prob_normed_grad_opts:
            return log_probs

        assert log_probs.feature_dim in (self.wb_target_dim, self.target_dim)
        if not self.out_blank_separated:
            assert log_probs.feature_dim == self.wb_target_dim
        if self.log_prob_normed_grad_exclude_blank:
            assert self.out_blank_separated
            if log_probs.feature_dim == self.wb_target_dim:
                return log_probs
        else:  # not excluded blank
            if log_probs.feature_dim == self.target_dim:
                return log_probs

        from alignments.util import normed_gradient, NormedGradientFuncInvPrior

        opts: Dict[str, Any] = self.log_prob_normed_grad_opts.copy()
        func_opts = opts.pop("func")
        assert isinstance(func_opts, dict)
        func_opts = func_opts.copy()
        assert func_opts.get("class", "inv_prior") == "inv_prior"  # only case for now: NormedGradientFuncInvPrior
        func_opts.pop("class", None)
        func = NormedGradientFuncInvPrior(**func_opts)

        assert "prior_running_mean" not in opts  # will be set by us here, and only when needed
        if opts.get("prior_running_mean_momentum") is not None:
            assert self.grad_prior is not None
            grad_prior = self.grad_prior
            if self.prior_running_mean_per_layer and aux_layer is not None:
                grad_prior = getattr(self, f"grad_prior_{aux_layer}")
            opts["prior_running_mean"] = grad_prior.raw_tensor

        assert log_probs.batch_dim_axis is not None and log_probs.feature_dim_axis is not None
        log_probs_ = log_probs.copy_template()
        log_probs_.raw_tensor = normed_gradient(
            log_probs.raw_tensor,
            batch_axis=log_probs.batch_dim_axis,
            feat_axis=log_probs.feature_dim_axis,
            **opts,
            func=func,
        )
        return log_probs_
