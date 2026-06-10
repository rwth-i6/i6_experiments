import functools
from typing import Optional, Tuple, Sequence, Dict, Any

import numpy as np
import torch

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim
from i6_experiments.users.mueller.experiments.language_models.ffnn import FeedForwardLm

# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

class Wav2VecModel(rf.Module):
  """Model definition"""

  def __init__(
        self,
        *,
        w2v_opts: Dict[str, Any],
        target_dim: Optional[Dim] = None,
        wb_target_dim: Optional[Dim | int] = None,
        blank_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
        bos_idx: Optional[int] = None,
  ):
    super(Wav2VecModel, self).__init__()

    import transformers
    from returnn.config import get_global_config
    config = get_global_config(return_empty_if_none=True)
    assert config

    w2v_config_file = w2v_opts["config_file"]
    wav2vec_config = transformers.Wav2Vec2Config.from_pretrained(w2v_config_file)

    # preprocessor_config = transformers.Wav2Vec2FeatureExtractor.from_pretrained(
    #   "/work/asr4/schmitt/sisyphus_work_dirs/2025_03_10_ctc_usr/i6_core/returnn/training/ReturnnTrainingJob.L6t5ebVFPeDZ/work/wav2vec_config/preprocessor_config.json")
    # tokenizer_config = transformers.Wav2Vec2Tokenizer.from_pretrained(
    #   "/work/asr4/schmitt/sisyphus_work_dirs/2025_03_10_ctc_usr/i6_core/returnn/training/ReturnnTrainingJob.L6t5ebVFPeDZ/work/wav2vec_config")
    # self.processor =transformers. Wav2Vec2Processor(preprocessor_config, tokenizer_config)

    self.wav2vec2 = transformers.Wav2Vec2Model(wav2vec_config)
    self.wav2vec2.freeze_feature_encoder()

    if w2v_opts["freeze_encoder_first_n_steps"] > 0:
      self.set_wav2vec_encoder_trainable(False)

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
    self.wb_target_dim = wb_target_dim

    w2v_hidden_size = self.wav2vec2.encoder.layers[0].feed_forward.output_dense.out_features
    self.enc_out_dim = Dim(name="enc", dimension=w2v_hidden_size, kind=Dim.Types.Feature)

    enc_spatial_dim = Dim(name="wav2vec_seq", dimension=None, kind=Dim.Types.Spatial)
    config.set("model_outputs", {"output": {"dims": [enc_spatial_dim, self.enc_out_dim]}})

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
    elif enc_logits_n_layers == 1:
      self.enc_logits = rf.Linear(self.enc_out_dim, wb_target_dim)

    ctc_label_smoothing = config.float("ctc_label_smoothing", 0.0)
    ctc_label_smoothing_exclude_blank = config.bool("ctc_label_smoothing_exclude_blank", False)
    self.ctc_label_smoothing_exclude_blank = ctc_label_smoothing_exclude_blank

    self.ctc_label_smoothing_opts = {
      "smoothing": ctc_label_smoothing,
      "axis": self.wb_target_dim,
      "exclude_labels": [self.blank_idx] if ctc_label_smoothing_exclude_blank else None,
    }

    self.log_prob_normed_grad_opts = config.typed_value("log_prob_normed_grad", None)
    self.log_prob_normed_grad_exclude_blank = config.bool(
      "log_prob_normed_grad_exclude_blank", False
    )

    self.feature_batch_norm = None
    self.feature_norm = config.bool("feature_norm", False)
    self.feature_stats = None

    self._specaugment_opts = {
      "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
      "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
      "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
                                      or (_log_mel_feature_dim // 5),
      "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
    }

    self._mixup = None

    self.decoder = None
    aux_attention_decoder = config.typed_value("aux_attention_decoder", None)
    if aux_attention_decoder:
      assert isinstance(aux_attention_decoder, dict)
      aux_attention_decoder = aux_attention_decoder.copy()
      aux_attention_decoder.setdefault("class", "returnn.frontend.decoder.transformer.TransformerDecoder")
      if isinstance(aux_attention_decoder.get("model_dim", None), int):
        aux_attention_decoder["model_dim"] = Dim(aux_attention_decoder["model_dim"], name="dec_model")
      self.decoder = rf.build_from_dict(aux_attention_decoder, encoder_dim=self.enc_out_dim, vocab_dim=target_dim)

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

  def set_wav2vec_encoder_trainable(self, trainable: bool):
    for param in self.wav2vec2.encoder.parameters():
      param.requires_grad = trainable
    for param in self.wav2vec2.feature_projection.parameters():
      param.requires_grad = trainable

  def __call__(
          self,
          source: Tensor,  # [B, T] or [B, T, 1]
          *,
          in_spatial_dim: Dim,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Tensor, Tensor, Dim]:

    # remove feature_dim if it is 1
    if source.feature_dim and source.feature_dim.dimension == 1:
        source = rf.squeeze(source, axis=source.feature_dim)
        assert not source.feature_dim  # raw audio

    source_dyn_lengths = source.get_sequence_lengths()

    batch_dim = source.dims[0]

    # preprocessing using returnn -> more efficient
    mask = source.get_sequence_mask_broadcast(in_spatial_dim)

    # dtype to float32
    source_template = source.copy_template(dtype="float32")
    source_new = Tensor(
      name=source_template.name,
      dims=source_template.dims,
      dtype=source_template.dtype,
      raw_tensor=source._raw_tensor.to(torch.float32),
    )

    source_mean = rf.reduce_mean(source_new, axis=[in_spatial_dim]).raw_tensor.unsqueeze(1)  # [B, 1]
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

    enc_raw = self.wav2vec2(source_raw).last_hidden_state

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

    if hasattr(self, "enc_logits"):
        logits = self.enc_logits(enc)
    else:
        logits = enc
    return logits, enc, enc_spatial_dim

  def log_probs_wb_from_logits(self, logits: Tensor) -> Tensor:
    """
    :param logits: incl blank
    :return: log probs with blank from logits (wb_target_dim)
        If out_blank_separated, we use a separate sigmoid for the blank.
    """
    log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)
    return log_probs
