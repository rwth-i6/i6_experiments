from dataclasses import dataclass
import torch
from torch import nn
import math
import os
from typing import Any, Dict


from . import modules
from . import commons
from . import attentions
try:
    from .monotonic_align.path import maximum_path
except:
    import subprocess
    import sys
    subprocess.call(
        [sys.executable, 'setup.py', 'build_ext', '--inplace'],
        cwd=os.path.join(os.path.realpath(os.path.dirname(__file__)), "monotonic_align")
    )
    from .monotonic_align.path import maximum_path

from ..tts_shared.tts_base_model.base_model_v1 import BaseTTSModelV1
from ..tts_shared import DbMelFeatureExtractionConfig
from ..tts_shared.encoder.transformer import TTSTransformerTextEncoderV1Config
from ..tts_shared.encoder.duration_predictor import SimpleConvDurationPredictorV1Config
from .unet import Unet


@dataclass
class DiffusionDecoderConfig():
    """
        Args:
            target_channels: Number of feature- and latent space channels
            hidden_channels: Number of hidden channels
            kernel_size: Kernel Size for convolutions in coupling blocks
            dilation_rate: Dilation Rate to define dilation in convolutions of coupling block
            num_blocks: Number of coupling blocks
            num_layers_per_block: Number of layers in CNN of the coupling blocks
            dropout: Dropout probability for CNN in coupling blocks.
            num_splits: Number of splits for the 1x1 convolution for flows in the decoder.
            num_squeeze: Squeeze the feature and latent space by this factor.
            use_sigmoid_scale: Boolean to define if log probs in coupling layers should be rescaled using sigmoid. Defaults to False.
    """
    unet_channels: int
    unet_in_channels: int
    unet_out_channels: int
    dim_mults: list[int]
    groups: int
    with_time_emb: bool
    beta_0: float
    beta_1: float
    iteration_steps: int
    time_scale: float




class DiffusionDecoder(nn.Module):
  def __init__(self, 
      cfg: DiffusionDecoderConfig
    ):

    super().__init__()
    self.cfg = cfg
    self.delta_t = cfg.time_scale*1.0 / cfg.iteration_steps
    self.discrete_betas = torch.linspace(cfg.beta_0, cfg.beta_1, cfg.iteration_steps)
    self.unet = Unet(dim=cfg.unet_channels, out_dim=cfg.unet_out_channels, dim_mults=cfg.dim_mults, groups=cfg.groups, channels=cfg.unet_in_channels, with_time_emb=cfg.with_time_emb)

  def marginal_prob(self, mu, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None]) * x + (1-torch.exp(log_mean_coeff[:, None, None]) ) * mu
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def cal_loss(self, x, mu, t, z, std, g=None):
    time_steps = t * (self.N - 1)
    if g:
        x = torch.stack([x, mu, g], 1)
    else:
        x = torch.stack([x, mu], 1)
    grad = self.unet(x, time_steps)
    loss = torch.square(grad + z / std[:, None, None]) * torch.square(std[:, None, None])
    return loss

  def forward(self, mu, y=None, g=None, gen=False):
    if not gen:
      t = torch.FloatTensor(y.shape[0]).uniform_(0, self.cfg.time_scale-self.delta_t).to(y.device)+self.delta_t  # sample a random t
      mean, std = self.marginal_prob(mu, y, t)
      z = torch.randn_like(y)
      x = mean + std[:, None, None] * z
      loss = self.cal_loss(x, mu, t, z, std, g)
      return loss
    else:
      with torch.no_grad():
        y_T = torch.randn_like(mu) + mu
        y_t_plus_one = y_T
        y_t = None
        for n in range(self.N - 1, 0, -1):
          t = torch.FloatTensor(1).fill_(n).to(mu.device)
          if g:
              x = torch.stack([y_t_plus_one, mu, g], 1)
          else:
              x = torch.stack([y_t_plus_one, mu], 1)
          grad = self.unet(x, t)
          y_t = y_t_plus_one-0.5*self.delta_t*self.discrete_betas[n]*(mu-y_t_plus_one-grad)
          y_t_plus_one = y_t

      return y_t


@dataclass
class Config:
    """
        Args:
            num_speakers: input size of the speaker embedding matrix
            speaker_embedding_size: output size of the speaker embedding matrix
            mean_only: if true, standard deviation for latent z estimation is fixed to 1
    """
    feature_extraction_config: DbMelFeatureExtractionConfig
    encoder_config: TTSTransformerTextEncoderV1Config
    duration_predictor_config: SimpleConvDurationPredictorV1Config
    diffusion_decoder_config: DiffusionDecoderConfig

    num_speakers: int
    speaker_embedding_size: int
    mean_only: bool = False

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = DbMelFeatureExtractionConfig(**d["feature_extraction_config"])
        d["encoder_config"] = TTSTransformerTextEncoderV1Config.from_dict(d["encoder_config"])
        d["duration_predictor_config"] = SimpleConvDurationPredictorV1Config(**d["duration_predictor_config"])
        d["flow_decoder_config"] = FlowDecoderConfig(**d["flow_decoder_config"])
        return Config(**d)


class Model(BaseTTSModelV1):
    """
    Flow-based TTS model based on GlowTTS Structure
    Following the definition from https://arxiv.org/abs/2005.11129
    and code from https://github.com/jaywalnut310/glow-tts
    """

    def __init__(
        self,
        config: Dict[str, Any],
        **kwargs,
    ):
        # build config from nested dict
        config = Config.from_dict(config)
        self.config = config
        self.n_sqz = config.flow_decoder_config.num_squeeze

        super().__init__(
            feature_extraction_config=config.feature_extraction_config,
            encoder_config=config.encoder_config,
            duration_predictor_config=config.duration_predictor_config,
            num_speakers=config.num_speakers,
            speaker_embedding_size=config.speaker_embedding_size,
        )


        self.proj_m = nn.Conv1d(config.encoder_config.basic_dim, config.flow_decoder_config.target_channels, 1)
        if not config.mean_only:
            self.proj_s = nn.Conv1d(config.encoder_config.basic_dim, config.flow_decoder_config.target_channels, 1)

        self.decoder = DiffusionDecoder(
            cfg=config.diffusion_decoder_config,
        )

    def forward(
        self, x, x_lengths, raw_audio=None, raw_audio_lengths=None, g=None, gen=False, noise_scale=1.0, length_scale=1.0
    ):
        if not gen:
            y, y_lengths = self.extract_features(
                raw_audio=raw_audio,
                raw_audio_lengths=raw_audio_lengths
            )
            feature_lengths = y_lengths
        else:
            y, y_lengths = (None, None)

        h, h_mask, spk, log_durations = self.forward_encoder(
            phon_labels=x,
            labels_lengths=x_lengths,
            speaker_label=g,
        )
        
        h_m = self.proj_m(h) * h_mask
        if not self.config.mean_only:
            h_logs = self.proj_s(h) * h_mask
        else:
            h_logs = torch.zeros_like(h_m)

        if gen:  # durations from dp only used during generation
            w = torch.exp(log_durations) * h_mask * length_scale  # durations
            w_ceil = torch.ceil(w)  # durations ceiled
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = None
        else:
            y_max_length = y.size(2)

        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(h_mask.dtype)
        attn_mask = torch.unsqueeze(h_mask, -1) * torch.unsqueeze(z_mask, 2)

        if gen:
            attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
            z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), h_m.transpose(1, 2)).transpose(1, 2)
            z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), h_logs.transpose(1, 2)).transpose(1, 2)
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * h_mask

            z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
            y, logdet = self.decoder(z, z_mask, g=spk, reverse=True)
            return (y, z_m, z_logs, logdet, z_mask, y_lengths), (h_m, h_logs, h_mask), (attn, log_durations, logw_, w_ceil)
        else:
            z, logdet = self.decoder(y, z_mask, g=spk, reverse=False)
            with torch.no_grad():
                x_s_sq_r = torch.exp(-2 * h_logs)
                logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - h_logs, [1]).unsqueeze(-1)  # [b, t, 1]
                logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z**2))  # [b, t, d] x [b, d, t'] = [b, t, t']
                logp3 = torch.matmul((h_m * x_s_sq_r).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
                logp4 = torch.sum(-0.5 * (h_m**2) * x_s_sq_r, [1]).unsqueeze(-1)  # [b, t, 1]
                logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

                attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
                # embed()

            z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), h_m.transpose(1, 2)).transpose(
                1, 2
            )  # [b, t', t], [b, t, d] -> [b, d, t']
            z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), h_logs.transpose(1, 2)).transpose(
                1, 2
            )  # [b, t', t], [b, t, d] -> [b, d, t']

            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * h_mask
            return (z, z_m, z_logs, logdet, z_mask), (h_m, h_logs, h_mask), (y_lengths, feature_lengths), (attn, log_durations, logw_)

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_samples = data["raw_audio"]  # [B, T, 1]
    audio_features_len = data["raw_audio:size1"]  # [B]
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B, N]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)

    (z, z_m, z_logs, logdet, z_mask), _, _, (attn, logw, logw_) = model(
        phonemes, phonemes_len, raw_samples, audio_features_len, speaker_labels
    )

    l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
    l_dp = commons.duration_loss(logw, logw_, phonemes_len)

    run_ctx.mark_as_loss(name="mle", loss=l_mle)
    run_ctx.mark_as_loss(name="dp", loss=l_dp)

