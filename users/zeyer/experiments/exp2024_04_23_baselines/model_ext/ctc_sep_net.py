"""
Conformer with separate FF (or any other) net.
Like:
https://github.com/rwth-i6/returnn-experiments/blob/master/2016-ctc-paper/switchboard-extended2020/ctcfbw.p2ff500c.ce05s01.prior05am.cpea001.am01b03.nopretrain.nbf.config
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Sequence, Tuple, Dict
import functools

if TYPE_CHECKING:
    import torch

from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf
from returnn.frontend.encoder.base import ISeqFramewiseEncoder
from returnn.frontend.encoder.conformer_v2 import (
    ConformerEncoderV2,
    ConformerFrontend,
    ConformerEncoderLayer,
    ConformerConvSubsample,
)
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import TrainDef
from i6_experiments.users.zeyer.nn_rf.layerdrop import SequentialLayerDrop
from ..ctc import model_recog

# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


class ModelSepNet(rf.Module):
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
        enc_conformer_layer: Optional[Dict[str, Any]] = None,
        enc_other_opts: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.in_dim = in_dim

        import numpy
        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        assert not enc_build_dict

        if not enc_input_layer:
            enc_input_layer = ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            )

        enc_input_opts = {"input_layer": enc_input_layer}
        if enc_other_opts:
            enc_other_opts = enc_other_opts.copy()
            for k in ["input_embedding_scale", "input_dropout"]:
                if k in enc_other_opts:
                    enc_input_opts[k] = enc_other_opts.pop(k)
        self.encoder_frontend = ConformerFrontend(in_dim, enc_model_dim, **enc_input_opts)

        enc_opts = {"num_layers": num_enc_layers}
        if enc_conformer_layer:
            enc_opts["encoder_layer"] = enc_conformer_layer
        enc_layer_drop = config.float("enc_layer_drop", 0.0)
        if enc_layer_drop:
            assert "sequential" not in enc_opts
            enc_opts["sequential"] = functools.partial(SequentialLayerDrop, layer_drop=enc_layer_drop)
        if enc_other_opts:
            for k, v in enc_other_opts.items():
                assert k not in enc_opts, f"enc_other_opts key {k!r} already in enc_opts {enc_opts}"
                enc_opts[k] = v
        self.encoder = ConformerEncoderV2(enc_model_dim, **enc_opts)

        # Experiments without final layer norm. (We might clean this up when this is not successful.)
        # Just patch the encoder here.
        enc_conformer_final_layer_norm = config.typed_value("enc_conformer_final_layer_norm", None)
        if enc_conformer_final_layer_norm is None:
            pass
        elif enc_conformer_final_layer_norm == "last":  # only in the last, i.e. remove everywhere else
            for layer in self.encoder.layers[:-1]:
                layer: ConformerEncoderLayer
                layer.final_layer_norm = rf.identity
        else:
            raise ValueError(f"invalid enc_conformer_final_layer_norm {enc_conformer_final_layer_norm!r}")

        disable_encoder_self_attention = config.typed_value("disable_encoder_self_attention", None)
        if disable_encoder_self_attention is not None:
            # Disable self-attention in encoder.
            from ..model_ext.disable_self_att import apply_disable_self_attention_

            apply_disable_self_attention_(self.encoder, disable_encoder_self_attention)

        separate_enc_net_dict = config.typed_value("separate_enc_net", None)
        assert isinstance(separate_enc_net_dict, dict)
        self.separate_enc_net: FeedForwardNet = rf.build_from_dict(separate_enc_net_dict, enc_model_dim)

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        if not wb_target_dim:
            wb_target_dim = target_dim + 1
        self.enc_aux_selected_layers = enc_aux_logits
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))
        self.enc_logits = rf.Linear(self.encoder.out_dim, wb_target_dim)
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
        feat, enc_spatial_dim = self.encoder_frontend(source, in_spatial_dim=in_spatial_dim)
        if collected_outputs is not None:
            collected_outputs["feat"] = feat
        enc = self.encoder(feat, spatial_dim=enc_spatial_dim, collected_outputs=collected_outputs)
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
            log_probs = _concat_vec_with_blank(
                log_probs_wo_blank + log_probs_emit,
                log_probs_blank,
                blank_idx=self.blank_idx,
                target_dim=self.target_dim,
                wb_target_dim=self.wb_target_dim,
                blank_dim=dummy_blank_feat_dim,
            )

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


class FeedForwardNet(ISeqFramewiseEncoder):
    def __init__(self, model_dim: Dim, *, num_layers: int = 8, activation=rf.relu):
        super().__init__()
        self.model_dim = model_dim
        self.out_dim = model_dim
        self.num_layers = num_layers
        self.activation = activation
        self.layers = rf.ModuleList(*[rf.Linear(model_dim, model_dim) for _ in range(num_layers)])

    def __call__(self, source: Tensor, *, spatial_dim: Dim) -> Tensor:
        x = source
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < self.num_layers:
                x = self.activation(x)
        return x


def _concat_vec_with_blank(
    vec: Tensor, blank: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_dim: Dim, blank_idx: int
) -> Tensor:
    """
    :param vec: (B, T, D)
    :param blank: (B, T, blank_dim)
    :param target_dim: D
    :param wb_target_dim: D+1
    :param blank_dim: 1
    :param blank_idx: assumed to be last index currently...
    :return: (B, T, D+1)
    """
    assert target_dim in vec.dims and blank_dim in blank.dims
    assert blank_idx == target_dim.dimension == wb_target_dim.dimension - 1  # not implemented otherwise
    res, _ = rf.concat(
        (vec, target_dim),
        (blank, blank_dim),
        out_dim=wb_target_dim,
    )
    res.feature_dim = wb_target_dim
    return res


def ctc_training_with_sep_net(
    *, model: ModelSepNet, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    use_fixed_ctc_grad = config.typed_value("use_fixed_ctc_grad", False)
    sep_net_grad_interpolate_alpha = config.float("sep_net_grad_interpolate_alpha", 0.0)

    from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad

    # only makes sense here
    assert use_fixed_ctc_grad == "v2"  # v2 has the fix for scaled/normalized CTC loss
    ctc_loss = ctc_loss_fixed_grad

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    if config.bool("use_eos_postfix", False):
        targets, (targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )

    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            aux_logits = model.aux_logits_from_collected_outputs(layer_idx, collected_outputs)
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits, aux_layer=layer_idx)
            aux_loss = ctc_loss(
                logits=aux_log_probs,
                logits_normalized=True,
                targets=targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            aux_loss.mark_as_loss(
                f"ctc_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
            # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
            # error = rf.edit_distance(
            #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
            # )
            # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())

    log_probs = model.log_probs_wb_from_logits(logits)
    sep_logits = model.separate_enc_net(collected_outputs["feat"], spatial_dim=enc_spatial_dim)
    sep_log_probs = model.log_probs_wb_from_logits(sep_logits)
    log_probs, sep_log_probs = _interpolate_grad_probs(
        log_probs,
        sep_log_probs,
        spatial_dim=enc_spatial_dim,
        wb_target_dim=model.wb_target_dim,
        alpha=sep_net_grad_interpolate_alpha,
    )

    loss = ctc_loss(
        logits=log_probs,
        logits_normalized=True,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    loss.mark_as_loss(
        "ctc",
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
    )

    if model.decoder:
        # potentially also other types but just assume
        # noinspection PyTypeChecker
        decoder: TransformerDecoder = model.decoder

        input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
        )
        targets_w_eos, _ = rf.pad(
            targets,
            axes=[targets_spatial_dim],
            padding=[(0, 1)],
            value=model.eos_idx,
            out_dims=[targets_w_eos_spatial_dim],
        )

        batch_dims = data.remaining_dims(data_spatial_dim)
        logits, _ = model.decoder(
            input_labels,
            spatial_dim=targets_w_eos_spatial_dim,
            encoder=decoder.transform_encoder(enc, axis=enc_spatial_dim),
            state=model.decoder.default_initial_state(batch_dims=batch_dims),
        )

        logits_packed, pack_dim = rf.pack_padded(
            logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
        )
        targets_packed, _ = rf.pack_padded(
            targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )

        log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        loss.mark_as_loss("aed_ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name="aed_fer", as_error=True)


ctc_training_with_sep_net: TrainDef[ModelSepNet]
ctc_training_with_sep_net.learning_rate_control_error_measure = "ctc"


def _interpolate_grad_probs(
    log_probs_main: Tensor,
    log_probs_sep: Tensor,
    *,
    spatial_dim: Dim,
    wb_target_dim: Dim,
    alpha: float,
) -> Tuple[Tensor, Tensor]:
    """
    Interpolate grads of log_probs an sep_log_probs

    Note: careful about scaling, one might be diff scaled than the other...

    grad loss_scale * ctc_loss w.r.t. log prob is -y * loss_scale.
    (Note: You need to used :func:`ctc_loss_fixed_grad`/:func:`torch_ctc_fixed_grad` for that...)

    Based on the grad, we can determine loss_scale (just loss_scale = -sum(grad)).
    We assume such gradient for log_prob_main and log_prob_sep,
    and then we interpolate (linearly) y_main with y_sep.

    :param log_probs_main: [B,T,D]. from the main model (e.g. Conformer)
    :param log_probs_sep: [B,T,D]. from the separate model (e.g. FF)
    :param spatial_dim: T
    :param wb_target_dim: D
    :param alpha: how much to linearly mixin the y_sep into y_main. 0: only y_main, 1: only y_sep
    :return: log_prob_main, log_prob_sep. the log probs are not modified.
        the gradient of log_prob_main is interpolated with log_prob_sep.
    """
    log_probs_main = log_probs_main.copy_transpose((spatial_dim, batch_dim, wb_target_dim))
    log_probs_sep = log_probs_sep.copy_transpose((spatial_dim, batch_dim, wb_target_dim))
    log_probs_main_raw, log_probs_sep_raw = _torch_interpolate_grad_probs(
        log_probs_main.raw_tensor, log_probs_sep.raw_tensor, alpha=alpha
    )
    log_probs_main.raw_tensor = log_probs_main_raw
    log_probs_sep.raw_tensor = log_probs_sep_raw
    return log_probs_main, log_probs_sep


def _torch_interpolate_grad_probs(
    log_probs_main: torch.Tensor, log_probs_sep: torch.Tensor, *, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    See :func:`_interpolate_grad_probs` for full doc.

    :param log_probs_main: [T,B,D]
    :param log_probs_sep: [T,B,D]
    :param alpha: how much to linearly mixin the y_sep into y_main. 0: only y_main, 1: only y_sep
    :return: log_prob_main, log_prob_sep
    """
    import torch

    # We avoid the global torch import in this module, thus we lazily define these classes here.
    global _InterpolateGradFunc
    if not _InterpolateGradFunc:

        class _InterpolateGradFunc(torch.autograd.Function):
            # noinspection PyShadowingNames
            @staticmethod
            def forward(ctx, log_probs_main, log_probs_sep):
                return log_probs_main, log_probs_sep

            @staticmethod
            def backward(ctx, grad_log_probs_main, grad_log_probs_sep):
                y_main_scaled = -grad_log_probs_main  # [T,B,D]
                y_sep_scaled = -grad_log_probs_sep  # [T,B,D]
                scale_main = y_main_scaled.sum(dim=-1, keepdim=True)  # [T,B,1]
                scale_sep = y_sep_scaled.sum(dim=-1, keepdim=True)  # [T,B,1]
                # y_main_scaled / scale_main = y_main, and sum(y_main) = 1. Likewise for y_sep.
                # We want y_interpolated = ((1-alpha) * y_main + alpha * y_sep),
                # and y_interpolated_scaled = y_interpolated * scale_main.
                # I.e.:
                # y_interpolated_scaled = (1-alpha) * y_main_scaled + alpha * y_sep_scaled * scale_main / scale_sep
                # To make this nan-safe, use torch.where(scale_sep != 0, scale_main / scale_sep, 0).
                scale_ratio = torch.where(scale_sep != 0, scale_main / scale_sep, 0.0)
                y_interpolated_scaled = y_main_scaled * (1 - alpha) + y_sep_scaled * (alpha * scale_ratio)
                return -y_interpolated_scaled, grad_log_probs_sep

    log_probs_main, log_probs_sep = _InterpolateGradFunc.apply(log_probs_main, log_probs_sep)
    return log_probs_main, log_probs_sep


_InterpolateGradFunc = None
