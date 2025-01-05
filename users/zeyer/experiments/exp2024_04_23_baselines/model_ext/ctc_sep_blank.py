"""
Conformer with separate blank.
"""


from __future__ import annotations
from typing import Optional, Any, Sequence, Tuple, Dict
import functools

from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf
from returnn.frontend.encoder.base import ISeqFramewiseEncoder
from returnn.frontend.encoder.conformer_v2 import (
    ConformerEncoderV2,
    ConformerFrontend,
    ConformerEncoderLayer,
    ConformerConvSubsample,
)

from i6_experiments.users.zeyer.nn_rf.layerdrop import SequentialLayerDrop
from ..ctc import model_recog

# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


class ModelSepBlank(rf.Module):
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

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        if not wb_target_dim:
            wb_target_dim = target_dim + 1
        self.enc_aux_selected_layers = enc_aux_logits
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, target_dim))
        self.enc_logits = rf.Linear(self.encoder.out_dim, target_dim)
        self.wb_target_dim = wb_target_dim
        self.out_blank_separated = config.bool("out_blank_separated", False)
        self.blank_logit_shift = config.float("blank_logit_shift", 0.0)

        separate_blank_model_dict = config.typed_value("separate_blank_model", None)
        assert isinstance(separate_blank_model_dict, dict)
        self.separate_blank_model = rf.build_from_dict(separate_blank_model_dict, enc_model_dim)

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
        enc = self.encoder(feat, spatial_dim=enc_spatial_dim, collected_outputs=collected_outputs)
        logits = self.enc_logits(enc)
        blank_logit = self.separate_blank_model(feat)
        if collected_outputs is not None:
            collected_outputs["blank_logit"] = blank_logit
        logits = _concat_vec_with_blank(
            logits,
            blank_logit,
            target_dim=self.target_dim,
            wb_target_dim=self.wb_target_dim,
            blank_dim=self.separate_blank_model.out_dim,
            blank_idx=self.blank_idx,
        )
        return logits, enc, enc_spatial_dim

    def aux_logits_from_collected_outputs(self, aux_layer: int, collected_outputs: Dict[str, Tensor]) -> Tensor:
        """
        :param aux_layer:
        :param collected_outputs: from __call__
        :return: logits
        """
        blank_logit = collected_outputs["blank_logit"]
        linear: rf.Linear = getattr(self, f"enc_aux_logits_{aux_layer}")
        aux_logits = linear(collected_outputs[str(aux_layer - 1)])
        aux_logits = _concat_vec_with_blank(
            aux_logits,
            blank_logit,
            target_dim=self.target_dim,
            wb_target_dim=self.wb_target_dim,
            blank_dim=self.separate_blank_model.out_dim,
            blank_idx=self.blank_idx,
        )
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


class SeparateBlankModel(ISeqFramewiseEncoder):
    """
    Model for blank logits
    """

    def __init__(self, in_dim: Dim):
        super().__init__()
        self.hidden_dim = in_dim * 2
        self.out_dim = Dim(1, name="blank")
        self.layer1 = rf.Linear(in_dim, self.hidden_dim)
        self.layer2 = rf.Linear(self.hidden_dim, self.out_dim)

    def __call__(self, source: Tensor, *, spatial_dim: Dim) -> Tensor:
        x = self.layer1(source)
        x = rf.relu(x)
        x = self.layer2(x)
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
