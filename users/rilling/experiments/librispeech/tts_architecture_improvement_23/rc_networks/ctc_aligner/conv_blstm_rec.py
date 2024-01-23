"""
Implementation of the CTC Aligner, updated for the new non-lazy init
"""
from typing import Tuple, Union
from returnn_common import nn
from ..shared.convolution import Conv1DStack
from .parameters import ConvBlstmRecParams


class TTSDecoder(nn.Module):
    """
    Decoder for audio reconstruction
    """

    def __init__(self, in_dim: nn.Dim, lstm_dim: int = 512):
        """
        :param lstm_dim: LSTM dimension size
        """
        super(TTSDecoder, self).__init__()
        self.lstm_dim = nn.FeatureDim("dec_lstm_dim", lstm_dim)
        self.lstm_1_fw = nn.LSTM(in_dim=in_dim, out_dim=self.lstm_dim)
        self.lstm_1_bw = nn.LSTM(in_dim=in_dim, out_dim=self.lstm_dim)
        self.lstm_2_fw = nn.LSTM(in_dim=2*self.lstm_dim, out_dim=self.lstm_dim)
        self.lstm_2_bw = nn.LSTM(in_dim=2*self.lstm_dim, out_dim=self.lstm_dim)

    def __call__(
        self, phoneme_probs: nn.Tensor, speaker_embedding: nn.Tensor, audio_time: nn.Dim
    ):
        """
        :param phoneme_probs:
        :param speaker_embedding:
        :param audio_time:
        :return:
        """
        cat, _ = nn.concat(
            (phoneme_probs, phoneme_probs.feature_dim),
            (speaker_embedding, speaker_embedding.feature_dim),
            allow_broadcast=True,
        )
        lstm_fw, _ = self.lstm_1_fw(cat, spatial_dim=audio_time, direction=1)
        lstm_bw, _ = self.lstm_1_bw(cat, spatial_dim=audio_time, direction=-1)
        # TODO maybe dropout?
        cat, _  = nn.concat((lstm_fw, lstm_fw.feature_dim), (lstm_bw, lstm_bw.feature_dim))
        lstm_fw, _ = self.lstm_2_fw(cat, spatial_dim=audio_time, direction=1)
        lstm_bw, _ = self.lstm_2_bw(cat, spatial_dim=audio_time, direction=-1)
        cat, _ = nn.concat((lstm_fw, lstm_fw.feature_dim), (lstm_bw, lstm_bw.feature_dim))
        return cat


class CTCAligner(nn.Module):
    """
    CTC Aligner from Timur Schümann implemented in returnn common
    """

    def __init__(
        self,
        audio_feature_dim: nn.Dim,
        speaker_label_dim: nn.Dim,
        phoneme_dim: nn.Dim,
        parameters: ConvBlstmRecParams,
    ):
        """

        :param audio_feature_dim:
        :param speaker_label_dim:
        :param phoneme_dim:
        :param parameters:
        """
        super(CTCAligner, self).__init__()

        self.audio_hidden_dim = nn.FeatureDim("audio_hidden_dim", parameters.audio_emb_size)
        self.speaker_embedding_dim = nn.FeatureDim("speaker_embedding_dim", parameters.speaker_emb_size)
        self.hidden_dim = nn.FeatureDim("hidden_size", parameters.conv_hidden_size)
        self.enc_lstm_dim = nn.FeatureDim("enc_lstm_dim", parameters.enc_lstm_size)

        self.audio_embedding = nn.Linear(in_dim=audio_feature_dim, out_dim=self.audio_hidden_dim)
        self.speaker_embedding = nn.Embedding(in_dim=speaker_label_dim, out_dim=self.speaker_embedding_dim)
        self.enc_conv_stack = Conv1DStack(in_dim=self.speaker_embedding_dim + self.audio_hidden_dim, dropout=[parameters.dropout]*5)
        self.enc_lstm_fw = nn.LSTM(in_dim=self.enc_conv_stack.out_dim, out_dim=self.enc_lstm_dim)
        self.enc_lstm_bw = nn.LSTM(in_dim=self.enc_conv_stack.out_dim, out_dim=self.enc_lstm_dim)

        self.softmax_dim = nn.FeatureDim("softmax_linear", phoneme_dim.dimension)

        self.softmax_lin = nn.Linear(
            in_dim=2*self.enc_lstm_dim,
            out_dim=self.softmax_dim,
        )
        self.tts_decoder = TTSDecoder(in_dim=self.softmax_dim + self.speaker_embedding_dim, lstm_dim=parameters.rec_lstm_size)
        self.reconstruction_lin = nn.Linear(
            in_dim=2*self.tts_decoder.lstm_dim,
            out_dim=nn.FeatureDim("reconstruction_dim", 80)
        )

        self.spectrogram_drop = parameters.dropout
        self.reconstruction_scale = parameters.reconstruction_scale
        self.training = parameters.training

    def __call__(
        self,
        audio_features: nn.Tensor,
        speaker_labels: nn.Tensor,
        phonemes: nn.Tensor,
        audio_time: nn.Dim,
        speaker_label_time: nn.Dim,
        phoneme_time: nn.Dim,
    ):
        """
        :param audio_features:
        :param speaker_labels:
        :param phonemes:
        :param audio_time:
        :param speaker_label_time:
        :return:
        """
        speaker_label_notime = nn.squeeze(speaker_labels, axis=speaker_label_time)

        # embedding
        speaker_embedding = self.speaker_embedding(speaker_label_notime)
        audio_embedding = self.audio_embedding(audio_features)
        # encoder
        cat, _ = nn.concat(
            (speaker_embedding, speaker_embedding.feature_dim),
            (audio_embedding, audio_embedding.feature_dim),
            allow_broadcast=True,
        )
        enc_conv = self.enc_conv_stack(cat, time_dim=audio_time)
        enc_fw, _ = self.enc_lstm_fw(enc_conv, spatial_dim=audio_time, direction=1)
        enc_bw, _ = self.enc_lstm_bw(enc_conv, spatial_dim=audio_time, direction=-1)
        cat, _ = nn.concat((enc_fw, enc_fw.feature_dim), (enc_bw, enc_bw.feature_dim))

        # spectogram loss
        spectogram_encoder = nn.dropout(
            cat, dropout=self.spectrogram_drop, axis=cat.feature_dim
        )
        spectogram_encoder = self.softmax_lin(spectogram_encoder)
        softmax = nn.softmax(spectogram_encoder, axis=spectogram_encoder.feature_dim)
        ctc = nn.ctc_loss(logits=spectogram_encoder, targets=phonemes)
        ctc.mark_as_loss(name="ctc", custom_inv_norm_factor=nn.length(dim=phoneme_time))

        if self.training:
            # TTS decoder
            tts_decoder = self.tts_decoder(
                phoneme_probs=softmax,
                speaker_embedding=speaker_embedding,
                audio_time=audio_time,
            )
            reconstruction_lin = self.reconstruction_lin(tts_decoder)
            audio_features, _ = nn.replace_dim(
                audio_features,
                in_dim=audio_features.feature_dim,
                out_dim=reconstruction_lin.feature_dim,
            )
            reconstruction_loss = nn.mean_squared_difference(
                reconstruction_lin, audio_features
            )
            reconstruction_loss.mark_as_loss(name="mse", scale=self.reconstruction_scale)
            return reconstruction_lin
        else:
            # replace the CTC blank label probability manually with zero
            # within the RETURNN backend this will be replaced via safe_log with 1e-20
            slice_out, slice_dim = nn.slice(
                softmax, axis=softmax.feature_dim, slice_start=0, slice_end=self.softmax_dim.dimension - 1,
            )
            padding = nn.pad(
                slice_out,
                axes=slice_out.feature_dim,
                mode="constant",
                padding=[(0, 1)],
                value=0,
            )
            extract_alignment = nn.forced_alignment(
                padding, align_target=phonemes, topology="ctc", input_type="prob", blank_included=True
            )
            dur_dump = nn.hdf_dump(extract_alignment, filename="durations.hdf")
            return dur_dump


def construct_network(
    epoch: int,
    audio_features: nn.Data,
    phonemes: nn.Data,
    speaker_labels: nn.Data,
    **kwargs
):
    """

    :param epoch:
    :param audio_features
    :param phonemes
    :param speaker_labels
    :param kwargs:
    :return:
    """
    params = ConvBlstmRecParams(**kwargs)
    net = CTCAligner(
        audio_feature_dim=audio_features.feature_dim_or_sparse_dim,
        speaker_label_dim=speaker_labels.feature_dim_or_sparse_dim,
        phoneme_dim=phonemes.feature_dim_or_sparse_dim,
        parameters=params,
    )
    out = net(
        audio_features=nn.get_extern_data(audio_features),
        speaker_labels=nn.get_extern_data(speaker_labels),
        phonemes=nn.get_extern_data(phonemes),
        audio_time=audio_features.dim_tags[audio_features.time_dim_axis],
        speaker_label_time=speaker_labels.dim_tags[speaker_labels.time_dim_axis],
        phoneme_time=phonemes.dim_tags[phonemes.time_dim_axis],
    )
    out.mark_as_default_output()

    return net
