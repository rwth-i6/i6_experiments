"""
Decoder module originally taken and modified from ESPNet:

https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/tacotron2/decoder.py
"""
from dataclasses import dataclass
import torch
from typing import Any, Dict

from ...tts_shared.tts_base_model.base_model_v1 import BaseTTSModelV1
from ...tts_shared import DbMelFeatureExtractionConfig
from ...tts_shared.encoder.transformer import TTSTransformerTextEncoderV1Config, GlowTTSMultiHeadAttentionV1Config
from ...tts_shared.encoder.duration_predictor import SimpleConvDurationPredictorV1Config
from ...tts_shared.util import sequence_mask

from ...tts_shared.espnet_tacotron import decoder_init, Prenet, Postnet, ZoneOutCell


@dataclass
class Tacotron2DecoderConfig():
    """
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            dlayers (int, optional): The number of decoder lstm layers.
            dunits (int, optional): The number of decoder lstm units.
            prenet_layers (int, optional): The number of prenet layers.
            prenet_units (int, optional): The number of prenet units.
            postnet_layers (int, optional): The number of postnet layers.
            postnet_filts (int, optional): The number of postnet filter size.
            postnet_chans (int, optional): The number of postnet filter channels.
            output_activation_fn (torch.nn.Module, optional):
                Activation function for outputs.
            use_concate (bool, optional): Whether to concatenate encoder embedding
                with decoder lstm outputs.
            dropout_rate (float, optional): Dropout rate.
            zoneout_rate (float, optional): Zoneout rate.
            reduction_factor (int, optional): Reduction factor.
    """
    dlayers: int
    dunits: int
    prenet_layers: int
    prenet_units: int
    postnet_layers: int
    postnet_chans: int
    postnet_filts: int
    use_batch_norm: bool
    use_concate: bool
    dropout_rate: float
    zoneout_rate: float
    reduction_factor: int


class Tacotron2Decoder(torch.nn.Module):
    """Decoder module of Spectrogram prediction network.

    This is a module of decoder of Spectrogram prediction network in Tacotron2,
    which described in `Natural TTS
    Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The decoder generates the sequence of
    features from the sequence of the hidden states.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    The attention part was removed, as the att_c is given as the upsampled encoder output
    """

    def __init__(
        self,
        cfg: Tacotron2DecoderConfig,
        encoder_output_size: int,
        speaker_embedding_size,
        num_features: int,
    ):
        """Initialize Tacotron2 decoder module.



        """
        super(Tacotron2Decoder, self).__init__()

        # store the hyperparameters
        self.idim = encoder_output_size
        self.odim = num_features
        self.use_concate = cfg.use_concate
        self.reduction_factor = cfg.reduction_factor

        # define lstm network
        prenet_units = cfg.prenet_units if cfg.prenet_layers != 0 else num_features
        self.lstm = torch.nn.ModuleList()
        for layer in range(cfg.dlayers):
            iunits = encoder_output_size + speaker_embedding_size + prenet_units if layer == 0 else cfg.dunits
            lstm = torch.nn.LSTMCell(iunits, cfg.dunits)
            if cfg.zoneout_rate > 0.0:
                lstm = ZoneOutCell(lstm, cfg.zoneout_rate)
            self.lstm += [lstm]

        # define prenet
        if cfg.prenet_layers > 0:
            self.prenet = Prenet(
                idim=num_features,
                n_layers=cfg.prenet_layers,
                n_units=prenet_units,
                dropout_rate=cfg.dropout_rate,
            )
        else:
            self.prenet = None

        # define postnet
        if cfg.postnet_layers > 0:
            self.postnet = Postnet(
                idim=encoder_output_size + speaker_embedding_size,
                odim=num_features,
                n_layers=cfg.postnet_layers,
                n_chans=cfg.postnet_chans,
                n_filts=cfg.postnet_filts,
                use_batch_norm=cfg.use_batch_norm,
                dropout_rate=cfg.dropout_rate,
            )
        else:
            self.postnet = None

        # define projection layers
        iunits = encoder_output_size + speaker_embedding_size + cfg.dunits if cfg.use_concate else cfg.dunits
        self.feat_out = torch.nn.Linear(iunits, num_features * cfg.reduction_factor, bias=False)

        # initialize
        self.apply(decoder_init)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def forward(self, h_t, h_t_lengths, target_audio_features, speaker_embedding):
        """Calculate forward propagation.

        Args:
            hs (Tensor): Batch of the sequences of padded hidden states (B, Tmax, idim).
            hlens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor):
                Batch of the sequences of padded target features (B, Lmax, odim).

        Returns:
            Tensor: Batch of output tensors after postnet (B, Lmax, odim).
            Tensor: Batch of output tensors before postnet (B, Lmax, odim).

        Note:
            This computation is performed in teacher-forcing manner.
        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            target_audio_features = target_audio_features[:, self.reduction_factor - 1 :: self.reduction_factor]

        # initialize hidden states of decoder
        c_list = [self._zero_state(h_t)]
        z_list = [self._zero_state(h_t)]
        for _ in range(1, len(self.lstm)):
            c_list += [self._zero_state(h_t)]
            z_list += [self._zero_state(h_t)]
        prev_out = h_t.new_zeros(h_t.size(0), self.odim)



        # loop for an output sequence
        outs = []
        for y, h in zip(target_audio_features.transpose(0, 1), h_t.transpose(0, 1)):
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            xs = torch.cat([h, prenet_out, speaker_embedding], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            zcs = (
                torch.cat([z_list[-1], h, speaker_embedding], dim=1)
                if self.use_concate
                else z_list[-1]
            )
            outs += [self.feat_out(zcs).view(h_t.size(0), self.odim, -1)]
            prev_out = y  # teacher forcing

        before_outs = torch.cat(outs, dim=2)  # (B, odim, Lmax)

        if self.reduction_factor > 1:
            before_outs = before_outs.view(
                before_outs.size(0), self.odim, -1
            )  # (B, odim, Lmax)

        if self.postnet is not None:
            after_outs = before_outs + self.postnet(before_outs)  # (B, odim, Lmax)
        else:
            after_outs = before_outs

        mask = sequence_mask((h_t_lengths // self.reduction_factor) * self.reduction_factor).unsqueeze(-1)

        before_outs = before_outs.transpose(2, 1) * mask  # (B, Lmax, odim)
        after_outs = after_outs.transpose(2, 1) * mask  # (B, Lmax, odim)

        return after_outs, before_outs

    def inference(
        self,
        h_t,
        t_mask,
        speaker_embedding,
    ):
        """Generate the sequence of features given the sequences of characters.

        :param h_t: encoder outputs in (B, T, F)
        :param h_t_mask: [B, 1, T]
        Args:
            h (Tensor): Input sequence of encoder hidden states (T, C).

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).

        Note:
            This computation is performed in auto-regressive manner.

        .. _`Deep Voice 3`: https://arxiv.org/abs/1710.07654

        """
        # setup

        # initialize hidden states of decoder
        c_list = [self._zero_state(h_t)]
        z_list = [self._zero_state(h_t)]
        for _ in range(1, len(self.lstm)):
            c_list += [self._zero_state(h_t)]
            z_list += [self._zero_state(h_t)]
        prev_out = h_t.new_zeros(h_t.size(0), self.odim)

        # loop for an output sequence
        idx = 0
        outs, att_ws, probs = [], [], []
        for h in h_t.transpose(0, 1):  # [T, B, F]
            # updated index
            idx += self.reduction_factor

            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            print(h.shape)
            print(prenet_out.shape)
            print(speaker_embedding.shape)
            xs = torch.cat([h, prenet_out, speaker_embedding], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            zcs = (
                torch.cat([z_list[-1], h, speaker_embedding], dim=1)
                if self.use_concate
                else z_list[-1]
            )
            outs += [self.feat_out(zcs).view(-1, self.odim, self.reduction_factor)]  # [(B, odim, r), ...]
            prev_out = outs[-1][:, :, -1]  # (B, odim)
        outs = torch.cat(outs, dim=2)
        print(outs.shape)
        print(t_mask.shape)
        outs = outs * t_mask  # (B, odim, L)
        if self.postnet is not None:
            outs = outs + self.postnet(outs)  # (1, odim, L)

        return outs * t_mask


@dataclass
class Config():
    """
        Args:
            num_speakers: input size of the speaker embedding matrix
            speaker_embedding_size: output size of the speaker embedding matrix
            mean_only: if true, standard deviation for latent z estimation is fixed to 1
    """
    feature_extraction_config: DbMelFeatureExtractionConfig
    encoder_config: TTSTransformerTextEncoderV1Config
    duration_predictor_config: SimpleConvDurationPredictorV1Config
    decoder_config: Tacotron2DecoderConfig

    num_speakers: int
    speaker_embedding_size: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = DbMelFeatureExtractionConfig(**d["feature_extraction_config"])
        d["encoder_config"] = TTSTransformerTextEncoderV1Config.from_dict(d["encoder_config"])
        d["duration_predictor_config"] = SimpleConvDurationPredictorV1Config(**d["duration_predictor_config"])
        d["decoder_config"] = Tacotron2DecoderConfig(**d["decoder_config"])
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

        super().__init__(
            feature_extraction_config=config.feature_extraction_config,
            encoder_config=config.encoder_config,
            duration_predictor_config=config.duration_predictor_config,
            num_speakers=config.num_speakers,
            speaker_embedding_size=config.speaker_embedding_size,
        )

        self.decoder = Tacotron2Decoder(
            cfg=config.decoder_config,
            encoder_output_size=config.encoder_config.basic_dim,
            speaker_embedding_size=config.speaker_embedding_size,
            num_features=config.feature_extraction_config.num_filters,
        )

    def forward(
        self, phon_labels, labels_lengths, speaker_label, raw_audio=None, raw_audio_lengths=None, target_durations=None
    ):
        """

        :param phon_labels:
        :param labels_lengths:
        :param speaker_label:
        :param raw_audio:
        :param raw_audio_lengths:
        :param target_durations:
        :return:
          - output_features as [B, F, T]
          - target features as [B, F, T]
          - feature lengths as [B] (of T)
          - log_durations as [B, N]
        """
        if raw_audio is not None:
            target_features, y_lengths = self.extract_features(
                raw_audio=raw_audio.squeeze(-1),
                raw_audio_lengths=raw_audio_lengths,
                time_last=False,
            )
        else:
            target_features, y_lengths = (None, None)

        h, h_mask, spk, log_durations = self.forward_encoder(
            phon_labels=phon_labels,
            labels_lengths=labels_lengths,
            speaker_label=speaker_label,
        )


        if target_durations is not None:
            # target durations can be considered masked (= padding area is zero)
            upsampler_durations = target_durations
        else:
            # create masked absolute durations in the desired shape
            masked_durations = torch.ceil(torch.exp(log_durations)) * h_mask  # [B, 1, N]
            masked_durations = masked_durations.squeeze(1)  # [B, N]
            if self.training:
                raise ValueError("target_durations need to be provided for training")
            upsampler_durations = masked_durations

        if raw_audio is not None:
            feature_lengths = y_lengths
        else:
            feature_lengths = torch.sum(upsampler_durations, dim=1)

        t_mask = sequence_mask(feature_lengths).unsqueeze(1)  # B, 1, T
        attn_mask = h_mask.unsqueeze(-1) * t_mask.unsqueeze(2)  # B, 1, N, T
        path = self.generate_path(upsampler_durations, attn_mask.squeeze(1))

        # path as [B, T, N]  x h as [B, N, F] -> [B, T, F]
        upsampled_h = torch.matmul(path.transpose(1, 2), h.transpose(1, 2))

        if raw_audio is None:
            # inference
            output_features = self.decoder.inference(h_t=upsampled_h, t_mask=t_mask, speaker_embedding=spk.squeeze(-1))
            return output_features, target_features, feature_lengths, log_durations.squeeze(1)
        else:
            # Training
            output_features_after, output_features_before = self.decoder(h_t=upsampled_h, h_t_lengths=feature_lengths, target_audio_features=target_features, speaker_embedding=spk.squeeze(-1))

        return output_features_after, output_features_before, target_features, feature_lengths, log_durations.squeeze(1)
        

def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_samples = data["raw_audio"]  # [B, T, 1]
    audio_features_len = data["raw_audio:size1"]  # [B]
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B, N]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)
    durations = data["durations"]  # [B, N]
    durations_len = data["durations:size1"]  # [B]

    output_features_after, output_features_before, target_features, features_lengths, log_durations = model(
        phonemes, phonemes_len, speaker_labels, raw_samples, audio_features_len, durations,
    )

    log_duration_targets = torch.log(torch.clamp_min(durations, 1))

    # frames might be drop depending on reduction factor, so adapt
    target_features = target_features[:, :output_features_after.size(1), :]

    l_l1_after = torch.sum(torch.abs(output_features_after - target_features)) / model.config.feature_extraction_config.num_filters
    l_l1_before = torch.sum(torch.abs(output_features_before - target_features)) / model.config.feature_extraction_config.num_filters
    l_dp = torch.sum((log_durations - log_duration_targets) ** 2)

    run_ctx.mark_as_loss(name="l1_after", loss=l_l1_after, inv_norm_factor=torch.sum(features_lengths))
    run_ctx.mark_as_loss(name="l1_before", loss=l_l1_before, inv_norm_factor=torch.sum(features_lengths))
    run_ctx.mark_as_loss(name="dp", loss=l_dp, inv_norm_factor=torch.sum(phonemes_len))

