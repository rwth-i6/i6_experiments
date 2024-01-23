"""
Like v1, but with LayerNorm in the duration prediction, similar to Timurs/Benedikts implementation

Adds
"""

from dataclasses import dataclass
import torch
import numpy
from torch import nn
import multiprocessing
from librosa import filters
import sys
import time
import os
import soundfile
from typing import Any, Dict, Optional, Tuple, Union

from returnn.torch.context import get_run_ctx

from .nar_taco_v2_config import (
    NarEncoderConfig,
    NarTacoDecoderConfig,
    ConvDurationSigmaPredictorConfig,
    ModelConfig,
    DbMelFeatureExtractionConfig,
)


def _lengths_to_op_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to operation compatible key mask

    :param lengths: [B]
    :return: B x T, where 1 means within sequence and 0 means outside sequence
    """
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)
    return padding_mask


class DbMelFeatureExtraction(nn.Module):

    def __init__(
            self,
            config: DbMelFeatureExtractionConfig
    ):
        super().__init__()
        self.register_buffer("n_fft", torch.tensor(int(config.win_size * config.sample_rate)))
        self.register_buffer("hop_length", torch.tensor(int(config.hop_size * config.sample_rate)))
        self.register_buffer("min_amp", torch.tensor(config.min_amp))
        self.center = config.center
        if config.norm is not None:
            self.apply_norm = True
            self.register_buffer("norm_mean", torch.tensor(config.norm[0]))
            self.register_buffer("norm_std_dev", torch.tensor(config.norm[1]))
        else:
            self.apply_norm = False

        self.register_buffer("mel_basis", torch.tensor(filters.mel(
            sr=config.sample_rate,
            n_fft=int(config.sample_rate * config.win_size),
            n_mels=config.num_filters,
            fmin=config.f_min,
            fmax=config.f_max)))
        self.register_buffer("window", torch.hann_window(int(config.win_size * config.sample_rate)))
        self.output_size = config.num_filters

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length in samples: [B]
        :return features as [B,T,F] and length in frames [B]
        """

        S = torch.abs(torch.stft(
            raw_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            pad_mode="constant",
            return_complex=True,
        )) ** 2
        if len(S.size()) == 2:
            # For some reason torch.stft "eats" batch sizes of 1, so we need to add it again if needed
            S = torch.unsqueeze(S, 0)
        melspec = torch.einsum("...ft,mf->...mt", S, self.mel_basis)
        melspec = 20 * torch.log10(torch.max(self.min_amp, melspec))
        feature_data = torch.transpose(melspec, 1, 2)

        if self.apply_norm:
            feature_data = (feature_data - self.norm_mean) / self.norm_std_dev

        if self.center:
            length = (length // self.hop_length) + 1
        else:
            length = ((length - self.n_fft) // self.hop_length) + 1

        return feature_data, length.int()


class GaussianUpsampling(nn.Module):
    """
    Performs gaussian upsampling for given input, duration and (learned) variances
    
    See Section 3.1 in https://arxiv.org/pdf/2010.04301.pdf
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        inp: torch.Tensor,
        masked_durations: torch.Tensor,
        variances: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param inp: H, [B, N, F]
        :param masked_durations: d, [B, N], padding area has to be zero
        :param variances: [B, N]
        :return: upsampled input
        """

        # given a duration sequence computes each center, as e.g.
        # durations = [5, 3, 6, 4]
        # center = [2.5, 6.5, 11, 16]
        center = torch.cumsum(masked_durations, dim=1) - (0.5 * masked_durations)  # [B, N]

        # the duration predictor can predict non-zero durations outside of the actual sequence,
        # thus a mask has to be applied to put all other values to zero
        l = torch.sum(masked_durations, dim=1)  # [B]

        # Non-graph op to do the range, maybe there will be tensor-based solution in the future
        t = torch.arange(1, torch.max(l).item() + 1, device=get_run_ctx().device).to(torch.float32)  # [T]
        t = torch.unsqueeze(torch.unsqueeze(t, 0), 0)  # [1, 1, T]

        normal = torch.distributions.Normal(loc=center.unsqueeze(2), scale=variances.unsqueeze(2))
        # normal distribution over [B, N, 1]

        w_t = torch.exp(normal.log_prob(t))  #  [B, N, T]
        w_t = w_t / (torch.sum(w_t, dim=1, keepdim=True) + 1e-16)  # normalize w_t over N

        output = torch.bmm(w_t.transpose(1, 2), inp)  # [B, T, N] * [B, N, F] = [B, T, F]

        return output


class Conv1DBlock(torch.nn.Module):
    """
    A 1D-Convolution with ReLU, batch-norm and non-broadcasted dropout
    Will pad to the same output length

    Extended with xavier_init
    """

    def __init__(self, in_size, out_size, filter_size, dropout):
        """
        :param in_size: input feature size
        :param out_size: output feature size
        :param filter_size: filter size
        :param dropout: dropout probability
        """
        super().__init__()
        assert filter_size % 2 == 1, "Only odd filter sizes allowed"
        self.conv = nn.Conv1d(in_size, out_size, filter_size, padding=filter_size // 2)
        self.bn = nn.BatchNorm1d(num_features=out_size)
        self.dropout = dropout

        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        """
        :param x: [B, F_in, T]
        :return: [B, F_out, T]
        """
        x = self.conv(x)
        x = nn.functional.relu(x)
        # TODO: does not consider masking!
        x = self.bn(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


class DPConvBlock(torch.nn.Module):
    """
    A 1D-Convolution with ReLU, batch-norm and non-broadcasted dropout
    Will pad to the same output length

    Extended with xavier_init
    """

    def __init__(self, in_size, out_size, filter_size, dropout):
        """
        :param in_size: input feature size
        :param out_size: output feature size
        :param filter_size: filter size
        :param dropout: dropout probability
        """
        super().__init__()
        assert filter_size % 2 == 1, "Only odd filter sizes allowed"
        self.conv = nn.Conv1d(in_size, out_size, filter_size, padding=filter_size // 2)
        self.ln = nn.LayerNorm(normalized_shape=out_size)
        self.dropout = dropout

        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        """
        :param x: [B, F_in, T]
        :return: [B, F_out, T]
        """
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


class ConvDurationSigmaPredictor(torch.nn.Module):
    """
    Convolution based duration predictor
    """


    def __init__(self, config: ConvDurationSigmaPredictorConfig, in_size: int):
        """
        :param in_size: e.g. from encoder blstm + speaker embedding
        """
        super().__init__()
        self.duration_convs = nn.Sequential(
            DPConvBlock(in_size, config.hidden_size,
                        filter_size=config.filter_size, dropout=config.dropout),
            DPConvBlock(config.hidden_size, config.hidden_size, filter_size=config.filter_size, dropout=config.dropout),
        )
        self.sigma_convs = nn.Sequential(
            DPConvBlock(in_size, config.hidden_size,
                        filter_size=config.filter_size, dropout=config.dropout),
            DPConvBlock(config.hidden_size, config.hidden_size, filter_size=config.filter_size, dropout=config.dropout),
        )
        self.duration_linear = nn.Linear(config.hidden_size, 1)
        self.sigma_linear = nn.Linear(config.hidden_size, 1)

    def forward(self, x: torch.Tensor):
        """

        :param x: [B, N, in_size]
        :return: [B, N, 1], duration as float
        """
        conv_in = torch.transpose(x, 1, 2)  # [B, in_size, N]
        dur_conv_out = self.duration_convs(conv_in)  # [B, hidden_size, N]
        sigma_conv_out = self.sigma_convs(conv_in)  # [B, hidden_size, N]
        dur_linear_in = torch.transpose(dur_conv_out, 1, 2)  # [B, N, hidden_size]
        sigma_linear_in = torch.transpose(sigma_conv_out, 1, 2)  # [B, N, hidden_size]
        dur_linear = self.duration_linear(dur_linear_in)  # [B, N, 1]
        sigma_linear = self.sigma_linear(sigma_linear_in)  # [B, N, 1]
        dur = nn.functional.softplus(dur_linear)  # [B, N, 1]
        sigma = nn.functional.softplus(sigma_linear)  # [B, N, 1]

        return dur.squeeze(dim=2), sigma.squeeze(dim=2)


class NarTacoEncoder(torch.nn.Module):
    """

    """

    def __init__(self, config: NarEncoderConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.label_in_dim, config.embedding_size)
        self.encoder_convs = nn.Sequential(
            Conv1DBlock(config.embedding_size, config.conv_hidden_size,
                        filter_size=config.filter_size, dropout=config.dropout),
            Conv1DBlock(config.conv_hidden_size, config.conv_hidden_size, filter_size=config.filter_size, dropout=config.dropout),
            Conv1DBlock(config.conv_hidden_size, config.conv_hidden_size, filter_size=config.filter_size, dropout=config.dropout),
        )
        self.blstm = nn.LSTM(input_size=config.conv_hidden_size, hidden_size=config.lstm_size, bidirectional=True, batch_first=True)

        self.output_size = 2*config.lstm_size

    def forward(self, label_in, label_in_len):
        """

        :param label_in: [B, N]
        :param label_in_len: [B]
        :return [B, N, lstm_size * 2]
        """
        transformed_labels = self.embedding(label_in)  # [B, N, embedding_size]
        transformed_labels_transposed = torch.transpose(transformed_labels, 1, 2)  # [B, embedding_size, N]
        conv_out = self.encoder_convs(transformed_labels_transposed)  # [B, conv_hidden_size, N]
        blstm_in = torch.transpose(conv_out, 1, 2) # [B, N, conv_hidden_size]

        # Sequences are sorted by decoder length, so here we do no sorting
        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(blstm_in, label_in_len.to("cpu"), batch_first=True, enforce_sorted=False)
        blstm_packed_out, _ = self.blstm(blstm_packed_in)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            blstm_packed_out, padding_value=0.0, batch_first=True
        )  # [B, N, lstm_size*2]

        return blstm_out


class NarTacoDecoder(torch.nn.Module):

    def __init__(self, config: NarTacoDecoderConfig, input_size: int, mel_target_size: int, linear_target_size: int):
        super().__init__()
        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.lstm_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout,
        )

        self.final_dropout = nn.Dropout(p=config.dropout, inplace=True)
        self.log_mel_linear = nn.Linear(2*config.lstm_size, mel_target_size)

    def forward(self, upsampling_output: torch.Tensor, masked_durations: torch.Tensor, speaker_embeddings: torch.Tensor):
        """

        :param upsampling_output: [B, T, encoder_out]
        :param speaker_embeddings: [B, speaker_embedding_size]
        :return:
        """

        length = torch.sum(masked_durations, dim=1).int()

        speaker_embeddings_broadcasted = torch.repeat_interleave(
            torch.unsqueeze(speaker_embeddings, 1), upsampling_output.size()[1], dim=1
        )  # [B, T, #SPK_EMB_SIZE]

        blstm_in = torch.concat([upsampling_output, speaker_embeddings_broadcasted], dim=2)
        # [B, T, encoder_out + speaker_embedding_size]

        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(
            blstm_in, length.to("cpu"), batch_first=True, enforce_sorted=self.training)
        blstm_packed_out, _ = self.blstm(blstm_packed_in)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            blstm_packed_out, padding_value=0.0, batch_first=True
        )  # [B, T, lstm_size*2]

        blstm_out = self.final_dropout(blstm_out)
        norm_log_mel_out = self.log_mel_linear(blstm_out)
        return norm_log_mel_out


class Model(torch.nn.Module):
    """
    Non-Attentive Tacotron with simple BLSTM decoder and gaussian upsampling.

    Uses fixed speaker embeddings
    """

    def __init__(
        self,
        config: Union[ModelConfig, Dict[str, Any]],
        **kwargs,
    ):
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        super().__init__()
        self.feature_extracton = DbMelFeatureExtraction(config=config.feature_extraction_config)
        self.speaker_embedding = nn.Embedding(251, config.speaker_embedding_size)

        self.encoder = NarTacoEncoder(config=config.encoder_config)
        self.duration_predictor = ConvDurationSigmaPredictor(
            config=config.duration_predictor_config,
            in_size=self.encoder.output_size + config.speaker_embedding_size
        )
        self.upsampler = GaussianUpsampling()
        self.decoder = NarTacoDecoder(
            config=config.decoder_config,
            input_size=config.speaker_embedding_size + self.encoder.output_size,
            mel_target_size=self.feature_extracton.output_size,
            linear_target_size=0
        )

    def forward(
        self,
        phonemes: torch.Tensor,
        phonemes_len: torch.Tensor,
        speaker_labels: torch.Tensor,
        target_durations: Optional[torch.Tensor] = None,
    ):
        """
        :param phonemes: [B, N]
        :param phonemes_len: [B]
        :param speaker_labels: [B, 1, 1?]
        :param target_durations: [B, N] only used for training
        :return:
        """
        speaker_embeddings: torch.Tensor = self.speaker_embedding(torch.squeeze(speaker_labels, dim=1))
        b_n_mask = _lengths_to_op_mask(phonemes_len).float()  # [B, N]

        encoder_out = self.encoder(label_in=phonemes, label_in_len=phonemes_len)
        speaker_embeddings_broadcast_n = torch.repeat_interleave(
            torch.unsqueeze(speaker_embeddings, 1), phonemes.size()[1], dim=1
        )  # [B, N, #SPK_EMB_SIZE]

        predictor_in = torch.concat([encoder_out, speaker_embeddings_broadcast_n], dim=2)
        
        durations, sigmas = self.duration_predictor(predictor_in)
        masked_durations = durations * b_n_mask

        if target_durations is not None:
            # target durations can be considered masked (= padding area is zero)
            upsampler_durations = target_durations
        else:
            if self.training:
                raise ValueError("target_durations need to be provided for training")
            upsampler_durations = masked_durations

        upsampled_encoder = self.upsampler(encoder_out, upsampler_durations, sigmas)  # [B, T, F]

        log_mel = self.decoder(
            upsampling_output=upsampled_encoder,
            masked_durations=upsampler_durations,
            speaker_embeddings=speaker_embeddings
        )

        return log_mel, masked_durations


def train_step(*, model: Model, data, run_ctx, **kwargs):

    audio_features = data["audio_features"]  # [B, T, F]
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    phonemes = data["phonemes"][indices, :]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"][indices]  # [B]
    speaker_labels = data["speaker_labels"][indices, :]  # [B, 1] (sparse)
    durations = data["durations"][indices, :]  # [B, N]
    durations_len = data["durations:size1"][indices]  # [B]

    tags = [data["seq_tag"][i] for i in list(indices.cpu().numpy())]

    assert torch.equal(phonemes_len, durations_len)

    squeezed_features = torch.squeeze(audio_features)
    with torch.no_grad():
        audio_features, audio_features_len = model.feature_extracton(squeezed_features, audio_features_len)
        
    num_frames = torch.sum(audio_features_len)
    num_phones = torch.sum(phonemes_len)

    log_mels, masked_durations = model(
        phonemes=phonemes,
        phonemes_len=phonemes_len,
        speaker_labels=speaker_labels,
        target_durations=durations,
    )

    b_t_mask = _lengths_to_op_mask(audio_features_len).float().unsqueeze(2)  # [B, T, 1]

    masked_log_mels = log_mels * b_t_mask
    masked_audio_features = audio_features * b_t_mask
    mel_loss = nn.functional.l1_loss(masked_log_mels, masked_audio_features, reduction="sum")
    mel_loss = mel_loss / model.feature_extracton.output_size

    run_ctx.mark_as_loss(name="log_mel_l1", loss=mel_loss, inv_norm_factor=num_frames)

    #from returnn.datasets.util.vocabulary import Vocabulary
    #vocab: Vocabulary = run_ctx.engine.train_dataset.datasets["audio"].targets
    #print(phonemes)
    #print(vocab.labels)

    #for i, phoneme in enumerate(phonemes[-1].cpu().detach().numpy()):
    #    vocab: Vocabulary = run_ctx.engine.train_dataset.datasets["audio"].targets
    #    print(f"{vocab.labels[phoneme]}, {masked_durations[-1][i]}, {durations[-1][i]}")
    duration_loss = nn.functional.l1_loss(masked_durations, durations, reduction="sum")

    run_ctx.mark_as_loss(name="duration_l1", loss=duration_loss, inv_norm_factor=num_phones)


def forward_init_hook(run_ctx, **kwargs):
    import json
    import sys
    sys.path.insert(0, "/u/rossenbach/src/vocoder_collection/univnet")
    from utils import AttrDict
    from inference import load_checkpoint
    from generator import UnivNet as Generator
    import numpy as np

    with open(
            "/work/asr3/rossenbach/schuemann/vocoder/vocoder_resources/vocoder_test/vocoder_collection/config_univ.json") as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = Generator(h).to(run_ctx.device)

    state_dict_g = load_checkpoint("/work/asr3/rossenbach/schuemann/vocoder/cp_libri_full/g_02310000", run_ctx.device)
    generator.load_state_dict(state_dict_g['generator'])

    run_ctx.generator = generator


def forward_finish_hook(run_ctx, **kwargs):
    pass


MAX_WAV_VALUE = 32768.0


def forward_step(*, model: Model, data, run_ctx, **kwargs):
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)

    tags = data["seq_tag"]

    log_mels, masked_durations = model(
        phonemes=phonemes,
        phonemes_len=phonemes_len,
        speaker_labels=speaker_labels,
        target_durations=None,
    )

    mel = torch.transpose(log_mels, 1, 2)
    noise = torch.randn([1, 64, mel.shape[-1]])
    audios = run_ctx.generator.forward(noise, mel)
    audios = audios * MAX_WAV_VALUE
    audios = audios.cpu().numpy().astype('int16')

    os.makedirs("audio_out/", exist_ok=True)
    for audio, tag in zip(audios, tags):
        soundfile.write(f"audio_out/" + tag.replace("/", "_") + ".wav", audio[0], 16000)

