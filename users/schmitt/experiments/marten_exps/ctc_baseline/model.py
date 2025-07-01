import functools
from typing import Optional, Tuple, Sequence, Dict, Any

import numpy as np
import torch

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer, ConformerConvSubsample
from returnn.datasets.util.vocabulary import Vocabulary

from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.schmitt.experiments.marten_exps.language_models.ffnn import FeedForwardLm

OUT_BLANK_LABEL = "<blank>"
# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

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
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_conformer_layer: Optional[Dict[str, Any]] = None,
        enc_other_opts: Optional[Dict[str, Any]] = None,
        train_language_model: Optional[FeedForwardLm] = None,
        recog_language_model: Optional[FeedForwardLm] = None,
        output_bias_init: Optional[str] = None,
    ):
        super(Model, self).__init__()

        import numpy
        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        enc_layer_drop = config.float("enc_layer_drop", 0.0)
        if enc_layer_drop:
            enc_sequential = functools.partial(SequentialLayerDrop, layer_drop=enc_layer_drop)
        else:
            enc_sequential = rf.Sequential

        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer=enc_conformer_layer,
            num_layers=num_enc_layers,
            sequential=enc_sequential,
            **(enc_other_opts or {}),
        )

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
            from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.model_ext.disable_self_att import apply_disable_self_attention_

            apply_disable_self_attention_(self.encoder, disable_encoder_self_attention)

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        if not wb_target_dim:
            wb_target_dim = target_dim + 1
        bias_init = None
        if output_bias_init is not None:
            bias_init = np.loadtxt(output_bias_init, dtype="float32")
            bias_init = rf.convert_to_tensor(torch.tensor(bias_init), dims=[wb_target_dim], dtype="float32", name="bias_init")
        for i in enc_aux_logits:
            enc_aux_logits_tmp = rf.Linear(self.encoder.out_dim, wb_target_dim)
            if bias_init is not None:
                enc_aux_logits_tmp.bias.initial = bias_init
            setattr(self, f"enc_aux_logits_{i}", enc_aux_logits_tmp)
        self.enc_logits = rf.Linear(self.encoder.out_dim, wb_target_dim)
        if bias_init is not None:
            self.enc_logits.bias.initial = bias_init
        self.wb_target_dim = wb_target_dim
        self.out_blank_separated = config.bool("out_blank_separated", False)

        if target_dim.vocab and not wb_target_dim.vocab:

            # Just assumption for code now, might extend this later.
            assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
            vocab_labels = list(target_dim.vocab.labels) + [OUT_BLANK_LABEL]
            wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                vocab_labels, user_defined_symbols={OUT_BLANK_LABEL: blank_idx}
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
            self.decoder = rf.build_from_dict(aux_attention_decoder, encoder_dim=enc_model_dim, vocab_dim=target_dim)

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
        
        self.train_language_model = train_language_model
        self.recog_language_model = recog_language_model

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Dim]:
        """encode, get logits"""
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

    def log_probs_wb_from_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: incl blank
        :return: log probs with blank from logits (wb_target_dim)
            If out_blank_separated, we use a separate sigmoid for the blank.
        """
        if not self.out_blank_separated:  # standard case, joint distrib incl blank
            log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)
        else:  # separate blank
            assert self.blank_idx == self.target_dim.dimension  # not implemented otherwise
            dummy_blank_feat_dim = Dim(1, name="blank_feat")
            logits_wo_blank, logits_blank = rf.split(
                logits, axis=self.wb_target_dim, out_dims=[self.target_dim, dummy_blank_feat_dim]
            )
            log_probs_wo_blank = rf.log_softmax(logits_wo_blank, axis=self.target_dim)
            log_probs_wo_blank = self._maybe_apply_on_log_probs(log_probs_wo_blank)
            log_probs_blank = rf.log_sigmoid(logits_blank)
            log_probs_emit = rf.squeeze(rf.log_sigmoid(-logits_blank), axis=dummy_blank_feat_dim)
            log_probs, _ = rf.concat(
                (log_probs_wo_blank + log_probs_emit, self.target_dim),
                (log_probs_blank, dummy_blank_feat_dim),
                out_dim=self.wb_target_dim,
            )
            log_probs.feature_dim = self.wb_target_dim
        log_probs = self._maybe_apply_on_log_probs(log_probs)
        return log_probs

    def _maybe_apply_on_log_probs(self, log_probs: Tensor) -> Tensor:
        """
        :param log_probs: either with blank or without blank
        :return: log probs, maybe some smoothing applied (all on gradients so far, not on log probs itself)
        """
        assert log_probs.feature_dim in (self.wb_target_dim, self.target_dim)
        if not self.out_blank_separated:
            assert log_probs.feature_dim == self.wb_target_dim

        log_probs = self._maybe_apply_log_probs_normed_grad(log_probs)

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

    def _maybe_apply_log_probs_normed_grad(self, log_probs: Tensor) -> Tensor:
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
        assert func_opts.get("class", "inv_prior") == "inv_prior"  # only case for now
        func_opts.pop("class", None)
        func = NormedGradientFuncInvPrior(**func_opts)

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


class Wav2VecModel(rf.Module):
  """Model definition"""

  def __init__(
          self,
          *,
          w2v_opts: Dict[str, Any],
          target_dim: Dim,
          wb_target_dim: Optional[Dim] = None,
          blank_idx: int,
          eos_idx: int,
          bos_idx: int,
          train_language_model: Optional[FeedForwardLm] = None,
          recog_language_model: Optional[FeedForwardLm] = None,
          rescore_language_model: Optional[FeedForwardLm] = None,
  ):
    super(Wav2VecModel, self).__init__()

    import transformers
    from returnn.config import get_global_config
    config = get_global_config(return_empty_if_none=True)

    w2v_config_file = w2v_opts["config_file"]
    wav2vec_config = transformers.Wav2Vec2Config.from_pretrained(w2v_config_file)

    # preprocessor_config = transformers.Wav2Vec2FeatureExtractor.from_pretrained(
    #   "/work/asr4/schmitt/sisyphus_work_dirs/2025_03_10_ctc_usr/i6_core/returnn/training/ReturnnTrainingJob.L6t5ebVFPeDZ/work/wav2vec_config/preprocessor_config.json")
    # tokenizer_config = transformers.Wav2Vec2Tokenizer.from_pretrained(
    #   "/work/asr4/schmitt/sisyphus_work_dirs/2025_03_10_ctc_usr/i6_core/returnn/training/ReturnnTrainingJob.L6t5ebVFPeDZ/work/wav2vec_config")
    # self.processor =transformers. Wav2Vec2Processor(preprocessor_config, tokenizer_config)

    self._current_extracted_features = None

    self.wav2vec2 = transformers.Wav2Vec2Model(wav2vec_config)
    self.wav2vec2.freeze_feature_encoder()

    if not w2v_opts.get("use_spec_augment", True):
      self.wav2vec2.config.apply_spec_augment = False

    if w2v_opts["freeze_encoder_first_n_steps"] > 0:
      self.set_wav2vec_encoder_trainable(False, except_layers=w2v_opts.get("unfrozen_encoder_layers", None))

    num_enc_layers = w2v_opts.get("num_enc_layers", len(self.wav2vec2.encoder.layers))
    if num_enc_layers != len(self.wav2vec2.encoder.layers):
      assert num_enc_layers < len(self.wav2vec2.encoder.layers)
      n_layers_to_remove = len(self.wav2vec2.encoder.layers) - num_enc_layers
      for i in range(n_layers_to_remove):
        del self.wav2vec2.encoder.layers[-1]

    self.target_dim = target_dim
    self.blank_idx = blank_idx
    self.eos_idx = eos_idx
    self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

    if not wb_target_dim:
      wb_target_dim = target_dim + 1

    if target_dim.vocab and not wb_target_dim.vocab:
      # Just assumption for code now, might extend this later.
      assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
      vocab_labels = list(target_dim.vocab.labels) + [OUT_BLANK_LABEL]
      wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
        vocab_labels, user_defined_symbols={OUT_BLANK_LABEL: blank_idx}
      )
    self.wb_target_dim = wb_target_dim

    w2v_hidden_size = self.wav2vec2.encoder.layers[0].feed_forward.output_dense.out_features
    self.enc_out_dim = Dim(name="enc", dimension=w2v_hidden_size, kind=Dim.Types.Feature)

    if config.bool("use_subsampled_enc_logits", False):
      # Subsampled encoder logits.
      self.enc_logits = EncLogitsSubsample(
        in_dim=self.enc_out_dim,
        out_dim=wb_target_dim,
      )
    else:
      enc_logits = []
      enc_logits_n_layers = w2v_opts.get("enc_logits_n_layers", 1)
      for i in range(enc_logits_n_layers):
        if i == enc_logits_n_layers - 1:
          out_dim = wb_target_dim
        else:
          out_dim = self.enc_out_dim
        enc_logits.append(rf.Linear(self.enc_out_dim, out_dim))

        if i != enc_logits_n_layers - 1:
          enc_logits.append(rf.relu)

      if len(enc_logits) > 1:
        self.enc_logits = rf.Sequential(enc_logits)
      else:
        self.enc_logits = rf.Linear(self.enc_out_dim, wb_target_dim)

    self.train_language_model = train_language_model
    self.recog_language_model = recog_language_model
    self.rescore_language_model = rescore_language_model
    self.decoder = None

  def set_wav2vec_encoder_trainable(self, trainable: bool, except_layers: Optional[Sequence[int]] = None):
    for param in self.wav2vec2.encoder.parameters():
      param.requires_grad = trainable
    for param in self.wav2vec2.feature_projection.parameters():
      param.requires_grad = trainable

    if except_layers is not None:
      for layer_idx in except_layers:
        print(f"Setting layer {layer_idx} trainable: {not trainable}")
        for param in self.wav2vec2.encoder.layers[layer_idx].parameters():
          param.requires_grad = not trainable

  def set_param_grads_to_zero(self):
    for param in self.parameters(recurse=True):
      param.raw_tensor.grad = None

  def __call__(
          self,
          source: Tensor,  # [B, T] or [B, T, 1]
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Tensor, Tensor, Dim]:
    from returnn.config import get_global_config
    config = get_global_config()

    # remove feature_dim if it is 1
    if source.feature_dim and source.feature_dim.dimension == 1:
        source = rf.squeeze(source, axis=source.feature_dim)
        assert not source.feature_dim  # raw audio

    source_dyn_lengths = source.get_sequence_lengths()

    # preprocessing using returnn -> more efficient
    mask = source.get_sequence_mask_broadcast(in_spatial_dim)
    source_mean = rf.reduce_mean(source, axis=[in_spatial_dim]).raw_tensor.unsqueeze(1)  # [B, 1]
    # replace padded samples by mean in order to set var-contributions to 0 (see source_var)
    source_raw = torch.where(mask, source.raw_tensor, source_mean)  # [B, T]

    # normalization denominator only over non-padded frames
    # padded frames in the sum evaluate to 0 because samples are set to the mean
    source_var = 1 / (source_dyn_lengths.to(source.device) - 1) * torch.sum((source_raw - source_mean) ** 2, dim=1)  # [B]
    source_var = source_var.unsqueeze(1)  # [B, 1]
    # normalize to 0 mean and unit variance
    source_raw = (source_raw - source_mean) / torch.sqrt(source_var + 1e-7)
    # set padded samples to 0
    source_raw = torch.where(mask, source_raw, 0.0)

    # # fairseq preprocessing (same as above but using list of np arrays)
    # # create list of numpy arrays (one array per audio seq)
    # source_raw = source.copy_transpose([batch_dim, in_spatial_dim]).raw_tensor
    # source_raw_np = source_raw.detach().cpu().numpy()
    # source_raw_np_list = [seq[:length] for seq, length in zip(source_raw_np, source_dyn_lengths)]
    # source_raw = self.processor(
    #   source_raw_np_list,
    #   sampling_rate=16_000,
    #   return_tensors="pt",
    #   padding=True,
    #   # in the downloaded config, this is set to false. explanation on huggingface:
    #   # "For all models whose processor has config.return_attention_mask == False, such as wav2vec2-base,
    #   # attention_mask should not be passed to avoid degraded performance when doing batched inference"
    #   # however, when not passing it, the samples are not properly normalized because of padding
    #   return_attention_mask=True,
    # ).input_values  # [B, T]

    # # check if the mean and std are correct
    # source_raw_np = source_raw.detach().cpu().numpy()
    # source_raw_np_list = [seq[:length] for seq, length in zip(source_raw_np, source_dyn_lengths)]
    # for array_ in source_raw_np_list:
    #   print("mean: ", np.mean(array_))
    #   print("std: ", np.std(array_))

    w2v_output = self.wav2vec2(source_raw)
    enc_raw = w2v_output.last_hidden_state
    self._current_extracted_features = w2v_output.extract_features
    # gradient_penalty_opts = config.typed_value("gradient_penalty_opts", {})
    # if gradient_penalty_opts and rf.get_run_ctx().train_flag:
    #   self._current_extracted_features.requires_grad = True

    # get dyn seq lengths of wav2vec encoder output
    enc_dyn_lengths_raw = source_dyn_lengths
    for conv_layer in self.wav2vec2.feature_extractor.conv_layers:
      enc_dyn_lengths_raw = torch.floor((enc_dyn_lengths_raw - (conv_layer.conv.kernel_size[0] - 1) - 1) / conv_layer.conv.stride[0] + 1)
    enc_dyn_lengths_raw = enc_dyn_lengths_raw.to(torch.int32)
    enc_dyn_lengths = rf.Tensor(
      name="wav2vec_dyn_lengths",
      dims=[batch_dim],
      dtype="int32",
      raw_tensor=enc_dyn_lengths_raw,
    )

    enc_spatial_dim = Dim(name="wav2vec_seq", dimension=enc_dyn_lengths, kind=Dim.Types.Spatial)
    enc = rf.Tensor(
      "wav2vec_states",
      dims=[batch_dim, enc_spatial_dim, self.enc_out_dim],
      dtype=rf.get_default_float_dtype(),
      raw_tensor=enc_raw,
    )

    if isinstance(self.enc_logits, EncLogitsSubsample):
      logits, enc_spatial_dim = self.enc_logits(enc, in_spatial_dim=enc_spatial_dim)
    else:
      logits = self.enc_logits(enc)

    if config.bool("collapse_logits_segments", False):
      logits, enc_spatial_dim = collapse_logits_segment(logits, self.wb_target_dim, enc_spatial_dim)

    return logits, enc, enc_spatial_dim

  def log_probs_wb_from_logits(self, logits: Tensor) -> Tensor:
    """
    :param logits: incl blank
    :return: log probs with blank from logits (wb_target_dim)
        If out_blank_separated, we use a separate sigmoid for the blank.
    """
    log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)
    return log_probs


class EncLogitsSubsample(rf.Module):
  def __init__(
          self,
          *,
          in_dim: Dim,
          out_dim: Dim,
  ):
    super(EncLogitsSubsample, self).__init__()

    self.batch_norm = rf.BatchNorm(in_dim)
    self.batch_norm.gamma.initial = 30.0  # as in https://arxiv.org/pdf/2204.02492
    self.linear = rf.Linear(in_dim, in_dim)
    self.conv = rf.Conv1d(
      in_dim=in_dim,
      out_dim=out_dim,
      filter_size=9,
      strides=3,
      with_bias=False,
      padding="valid",  # no padding
    )

  def __call__(
          self,
          x: Tensor,  # [B, T, F]
          *,
          in_spatial_dim: Dim,
  ) -> Tuple[Tensor, Dim]:
    x = self.batch_norm(x)
    inter_x = self.linear(rf.dropout(x, 0.1))
    x = x + inter_x  # residual connection
    x = rf.dropout(x, 0.1)
    x, spatial_dim = self.conv(x, in_spatial_dim=in_spatial_dim)
    x = x.copy_transpose([batch_dim, spatial_dim, self.conv.out_dim])

    return x, spatial_dim


def collapse_logits_segment(logits: Tensor, vocab_dim: Dim, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
  """
  From https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/models/wav2vec_u.py#L146
  """
  logits = logits.copy_transpose([batch_dim, in_spatial_dim, vocab_dim])
  logits_raw = logits.raw_tensor
  padding_mask = ~in_spatial_dim.get_mask(dim_order=[batch_dim, in_spatial_dim]).raw_tensor

  preds = logits_raw.argmax(dim=-1)

  if padding_mask.any():
    preds[padding_mask] = -1  # mark pad
  uniques = []

  bsz, tsz, csz = logits_raw.shape

  for b, p in enumerate(preds):
    uniques.append(p.cpu().unique_consecutive(return_inverse=True, return_counts=True))

  new_tsz = max(u[0].numel() for u in uniques)
  new_logits_raw = logits_raw.new_zeros(bsz, new_tsz, csz)
  new_enc_sizes = rf.Tensor("enc_collapsed_sizes", dims=[batch_dim], dtype="int32", raw_tensor=torch.zeros(bsz, dtype=torch.int32))

  for b in range(bsz):
    u, idx, c = uniques[b]
    keep = u != -1

    if rf.get_run_ctx().train_flag:
      # randomly select index from segment to keep
      u[0] = 0
      u[1:] = c.cumsum(0)[:-1]
      m = c > 1
      r = torch.rand(m.sum())
      o = (c[m] * r).long()
      u[m] += o
      new_logits_raw[b, : u.numel()] = logits_raw[b, u]
    else:
      # mean pool logits over segment
      new_logits_raw[b].index_add_(
        dim=0, index=idx.to(new_logits_raw.device), source=logits_raw[b]
      )
      new_logits_raw[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits_raw.device)

    new_sz = keep.sum()
    if not keep.all():
      kept_logits = new_logits_raw[b, : c.numel()][keep]
      new_logits_raw[b, :new_sz] = kept_logits

    if new_sz < new_tsz:
      pad = new_tsz - new_sz
      new_logits_raw[b, -pad:] = 0

    new_enc_sizes.raw_tensor[b] = new_sz

  new_enc_spatial_dim = Dim(new_enc_sizes)

  new_logits = rf.Tensor("collapsed_logits", dims=[batch_dim, new_enc_spatial_dim, vocab_dim], raw_tensor=new_logits_raw, dtype=logits.dtype)
  new_logits.feature_dim = vocab_dim

  return new_logits, new_enc_spatial_dim
