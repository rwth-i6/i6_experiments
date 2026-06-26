from typing import Any, Dict, Optional

from torch import nn
import transformers


class Wav2VecModel(nn.Module):
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
    wav2vec_config.hidden_dropout = w2v_opts.get("hidden_dropout", 0.1)
    wav2vec_config.attention_dropout = w2v_opts.get("att_dropout", 0.1)
    wav2vec_config.layerdrop = w2v_opts.get("layer_dropout", 0.0)
    wav2vec_config.mask_feature_prob = w2v_opts.get("mask_feature_prob", 0.0)
    wav2vec_config.mask_feature_length = w2v_opts.get("mask_feature_length", 10)
    wav2vec_config.mask_time_prob = w2v_opts.get("mask_time_prob", 0.05)

    self.wav2vec2 = transformers.Wav2Vec2Model(wav2vec_config)
