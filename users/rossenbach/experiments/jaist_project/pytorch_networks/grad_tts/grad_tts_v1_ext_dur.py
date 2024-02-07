from dataclasses import dataclass
import torch
from torch import nn
import math
import os
import random
from typing import Any, Dict

try:
    from ..glow_tts.monotonic_align.path import maximum_path
except:
    import subprocess
    import sys
    subprocess.call(
        [sys.executable, 'setup.py', 'build_ext', '--inplace'],
        cwd=os.path.join(os.path.realpath(os.path.dirname(__file__)), "monotonic_align")
    )
    from ..glow_tts.monotonic_align.path import maximum_path

from ..tts_shared.tts_base_model.base_model_v1 import BaseTTSModelV1
from ..tts_shared import DbMelFeatureExtractionConfig
from ..tts_shared.encoder.transformer import TTSTransformerTextEncoderV1Config
from ..tts_shared.encoder.duration_predictor import SimpleConvDurationPredictorV1Config
from ..tts_shared.util import sequence_mask, generate_path

from .diffusion import get_noise, GradLogPEstimator2d


@dataclass
class DiffusionDecoderConfig():
    """

    :param n_feats: target feature size
    :param dim: internal dimension of the diffusion process
    :param n_spks: number of speakers
    :param spk_emb_dim:
    :param beta_min:
    :param beta_max:
    :param pe_scale:
    """
    n_feats: int
    dim: int
    beta_min: float
    beta_max: float
    pe_scale: int


class DiffusionDecoder(torch.nn.Module):
    def __init__(
            self,
            cfg: DiffusionDecoderConfig,
            num_speakers: int,
            speaker_embedding_size: int
    ):

        super().__init__()
        self.cfg = cfg
        self.estimator = GradLogPEstimator2d(cfg.dim, n_spks=num_speakers,
                                             spk_emb_dim=speaker_embedding_size,
                                             pe_scale=cfg.pe_scale)

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.cfg.beta_min, self.cfg.beta_max, cumulative=True)
        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (1.0 - torch.exp(-0.5 * cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device,
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                   device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.cfg.beta_min, self.cfg.beta_max,
                                cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, x0, mask, mu, t, spk=None):
        """

        :param x0: target features (diffusion step zero) [B, F, T]?
        :param mask: [B, 1, T]?
        :param mu: [B, F, T]?
        :param t: just noise? unclear?
        :param spk: [B, emb_size]
        :return:
        """
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.cfg.beta_min, self.cfg.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, mask, mu, t, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.cfg.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, spk)


@dataclass
class Config:
    """
        Args:
            num_speakers: input size of the speaker embedding matrix
            speaker_embedding_size: output size of the speaker embedding matrix
    """
    feature_extraction_config: DbMelFeatureExtractionConfig
    encoder_config: TTSTransformerTextEncoderV1Config
    duration_predictor_config: SimpleConvDurationPredictorV1Config
    diffusion_decoder_config: DiffusionDecoderConfig

    num_speakers: int
    speaker_embedding_size: int
    decoder_segment_num_frames: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = DbMelFeatureExtractionConfig(**d["feature_extraction_config"])
        d["encoder_config"] = TTSTransformerTextEncoderV1Config.from_dict(d["encoder_config"])
        d["duration_predictor_config"] = SimpleConvDurationPredictorV1Config(**d["duration_predictor_config"])
        d["diffusion_decoder_config"] = DiffusionDecoderConfig(**d["diffusion_decoder_config"])
        return Config(**d)



def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


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

        super().__init__(
            feature_extraction_config=config.feature_extraction_config,
            encoder_config=config.encoder_config,
            duration_predictor_config=config.duration_predictor_config,
            num_speakers=config.num_speakers,
            speaker_embedding_size=config.speaker_embedding_size,
        )

        self.proj_m = nn.Conv1d(config.encoder_config.basic_dim, config.diffusion_decoder_config.n_feats, 1)

        self.decoder = DiffusionDecoder(
            cfg=config.diffusion_decoder_config,
            num_speakers=config.num_speakers,
            speaker_embedding_size=config.speaker_embedding_size
        )

    def forward(
        self, x, x_lengths, raw_audio=None, raw_audio_lengths=None, durations=None, g=None, gen=False, noise_scale=1.0, length_scale=1.0, num_timesteps=10,
    ):
        if not gen:
            y, y_lengths = self.extract_features(
                raw_audio=raw_audio,
                raw_audio_lengths=raw_audio_lengths
            )
        else:
            y, y_lengths = (None, None)

        h, h_mask, spk, log_durations = self.forward_encoder(
            phon_labels=x,
            labels_lengths=x_lengths,
            speaker_label=g,
        )
        # speaker embedding is returned as [B, spk_emb, 1], but GlowTTS does not need that last axis
        spk = spk.squeeze(-1)
        # same for log durations, remove remaining axis here
        log_durations = log_durations.squeeze(1)

        h_mu = self.proj_m(h) * h_mask

        if gen:  # durations from dp only used during generation
            w = torch.exp(log_durations) * h_mask * length_scale  # durations
            w_ceil = torch.ceil(w)  # durations ceiled
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()

            # Extra code for GradTTS
            y_max_length = int(y_lengths.max())
            y_max_length = fix_len_compatibility(y_max_length)
        else:
            y_max_length = y.size(2)
            # y_max_length = fix_len_compatibility(y_max_length)

        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(h_mask.dtype)
        attn_mask = torch.unsqueeze(h_mask, -1) * torch.unsqueeze(y_mask, 2)

        if gen:
            attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
            z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), h_mu.transpose(1, 2)).transpose(1, 2)
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * h_mask

            # Like GlowTTS, but we do not support only unit variance
            z = (z_m + torch.randn_like(z_m) * noise_scale) * y_mask

            y, logdet = self.decoder(z, y_mask, z_m, num_timesteps=num_timesteps, spk=spk)
            return (y, z_m, logdet, y_mask, y_lengths), (h_mu, h_mask), (attn, log_durations, logw_, w_ceil)
        else:
            with torch.no_grad():
                attn = self.generate_path(durations, attn_mask.squeeze(1))  # [B, N, T]
                attn = attn.detach()

            out_size = self.config.decoder_segment_num_frames

            # compute duration loss before segmenting decoder
            # [B, N, T] -> [B, N] masked with [B, N]
            target_log_durations = torch.log(1e-8 + torch.sum(attn, -1)) * h_mask.squeeze(1)
            duration_loss = torch.sum((log_durations - target_log_durations) ** 2) / torch.sum(x_lengths)

            # Not using out_size crashed....
            if not isinstance(out_size, type(None)):
                max_offset = (y_lengths - out_size).clamp(0)
                offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
                out_offset = torch.LongTensor([
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]).to(y_lengths)

                attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
                y_cut = torch.zeros(y.shape[0], self.config.diffusion_decoder_config.n_feats, out_size, dtype=y.dtype, device=y.device)
                y_cut_lengths = []
                for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                    y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                    y_cut_lengths.append(y_cut_length)
                    cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                    y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                    attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                y_cut_lengths = torch.LongTensor(y_cut_lengths)
                y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

                attn = attn_cut
                y = y_cut
                y_mask = y_cut_mask

            # Align encoded text with mel-spectrogram and get mu_y segment
            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), h_mu.transpose(1, 2))
            mu_y = mu_y.transpose(1, 2)

            # Compute loss of score-based decoder
            diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)

            # Compute loss between aligned encoder outputs and mel-spectrogram
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.config.diffusion_decoder_config.n_feats)

            return duration_loss, prior_loss, diff_loss


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_samples = data["raw_audio"]  # [B, T, 1]
    audio_features_len = data["raw_audio:size1"]  # [B]
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B, N]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)
    durations = data["durations"]  # [B, N]

    duration_loss, prior_loss, diff_loss = model(
        phonemes, phonemes_len, raw_samples, audio_features_len, durations, speaker_labels
    )

    run_ctx.mark_as_loss(name="duration_loss", loss=duration_loss)
    run_ctx.mark_as_loss(name="prior_loss", loss=prior_loss)
    run_ctx.mark_as_loss(name="diff_loss", loss=diff_loss)

