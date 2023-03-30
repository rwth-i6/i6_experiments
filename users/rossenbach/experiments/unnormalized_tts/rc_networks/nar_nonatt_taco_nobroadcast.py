"""
Implementation of the CTC NAR Network
"""
import numpy as np
from returnn_common import nn
from typing import Tuple, Union, Optional, Sequence

from .conv_modules import Conv1DBlock, ConvStack
from .gaussian_upsampling import GaussianUpsampling, LstmVarianceNetwork
from .features import FeatureExtractor

class LConvBlock(nn.Module):
    """
    LConvBlock from Parallel tacotron 1 with grouped/depthwise convolution instead of full lightweight conv
    """

    def __init__(
        self,
        in_dim: nn.Dim,
        linear_size: Union[int, nn.Dim] = 512,
        filter_size: int = 17,
    ):
        super(LConvBlock, self).__init__()
        if isinstance(linear_size, int):
            self.groups = linear_size
            self.lin_dim = nn.FeatureDim("LConv_linear", linear_size)
        else:
            self.lin_dim = linear_size
            self.groups = linear_size.dimension

        self.glu = nn.Linear(in_dim, 2*self.lin_dim)
        self.linear_up = nn.Linear(self.lin_dim, 4*self.lin_dim)
        self.linear_down = nn.Linear(4*self.lin_dim, self.lin_dim)
        self.conv = nn.Conv1d(
            in_dim=self.lin_dim,
            out_dim=self.lin_dim,
            filter_size=filter_size,
            groups=self.groups,
            with_bias=False,
            padding="same",
        )

    def __call__(self, spectrogramm: nn.Tensor, time_dim: nn.Dim) -> nn.Tensor:

        residual = spectrogramm
        x = self.glu(spectrogramm)
        x = nn.gating(x)  # GLU
        x, _ = self.conv(x, in_spatial_dim=time_dim)  # grouped conv / lightweight conv
        x = x + residual

        residual = x
        x = self.linear_up(x)
        x = nn.relu(x)
        x = self.linear_down(x)
        x = x + residual

        return x


class Decoder(nn.Module):
    """
      Decoder Block of the CTC Model
      """

    def __init__(
            self,
            in_dim: nn.Dim,
            dec_lstm_size_1: int = 1024,
            dec_lstm_size_2: int = 1024,
            linear_size: int = 80,
            dropout: float = 0.5,
    ):
        """

            :param dec_lstm_size_1: output dimension of the first lstm layer
            :param dec_lstm_size_2: output dimension of the second lstm layer
            :param linear_size: output dimension of the linear layer
            :param dropout: dropout values
            """
        super(Decoder, self).__init__()

        self.dropout = dropout

        self.dec_lstm_dim_1 = nn.FeatureDim("dec_lstm_dim", dec_lstm_size_1)
        self.dec_lstm_dim_2 = nn.FeatureDim("dec_lstm_dim", dec_lstm_size_2)
        self.linear_dim = nn.FeatureDim("dec_linear", linear_size)

        self.dec_lstm_fw_1 = nn.LSTM(in_dim=in_dim, out_dim=self.dec_lstm_dim_1)
        self.dec_lstm_bw_1 = nn.LSTM(in_dim=in_dim, out_dim=self.dec_lstm_dim_1)
        self.dec_lstm_fw_2 = nn.LSTM(in_dim=2*self.dec_lstm_dim_1, out_dim=self.dec_lstm_dim_2)
        self.dec_lstm_bw_2 = nn.LSTM(in_dim=2*self.dec_lstm_dim_1, out_dim=self.dec_lstm_dim_2)
        self.linear_out = nn.Linear(in_dim=2*self.dec_lstm_dim_2, out_dim=self.linear_dim)

    def __call__(
            self, rep: nn.Tensor, speaker_embedding: nn.Tensor, time_dim: nn.Dim
    ) -> nn.Tensor:
        """

            :param rep: upsampled / repeated input
            :param speaker_embedding: speaker label embedding
            :param time_dim: time dimension tag
            :return:
            """

        cat, _ = nn.concat(
            (rep, rep.feature_dim),
            (speaker_embedding, speaker_embedding.feature_dim),
            allow_broadcast=True,
        )

        dec_lstm_fw, _ = self.dec_lstm_fw_1(cat, spatial_dim=time_dim, direction=1)
        dec_lstm_bw, _ = self.dec_lstm_bw_1(cat, spatial_dim=time_dim, direction=-1)

        cat = nn.concat_features(dec_lstm_fw, dec_lstm_bw)
        dec_drop = nn.dropout(cat, axis=cat.feature_dim, dropout=self.dropout)

        dec_lstm_fw, _ = self.dec_lstm_fw_2(dec_drop, spatial_dim=time_dim, direction=1)
        dec_lstm_bw, _ = self.dec_lstm_bw_2(dec_drop, spatial_dim=time_dim, direction=-1)

        cat = nn.concat_features(dec_lstm_fw, dec_lstm_bw)
        dec_drop = nn.dropout(cat, axis=cat.feature_dim, dropout=self.dropout)

        lin_out = self.linear_out(dec_drop)

        return lin_out


class DurationPredictor(nn.Module):
    """
      Module for duration prediction
      During training not connected, only loss, during forward then actually predicts
      """

    def __init__(
            self,
            in_dim: nn.Dim,
            num_layers: int = 2,
            conv_sizes: Sequence[int] = (256, 256),
            filter_sizes: Sequence[int] = (3, 3),
            dropout: Sequence[float] = (0.5, 0.5),
            l2: float = 1e-07,
    ):
        """

            :param num_layers: number of convolutional layers
            :param conv_sizes: dimensions for the convolutions in the block
            :param filter_sizes: sizes for the filters in the block
            :param dropout: dropout values
            :param l2: weight decay value
            """
        super(DurationPredictor, self).__init__()

        assert len(conv_sizes) == num_layers
        assert len(filter_sizes) == num_layers
        assert len(dropout) == num_layers

        self.dropout = dropout
        self.l2 = l2

        self.lin_dim = nn.FeatureDim(
            "pred_lin_dim", 1
        )  # fixed to one to predict one duration per time/batch
        # simplify tags a bit if possible
        if len(set(conv_sizes)) == 1:  # all sizes equal
            out_dims = [nn.FeatureDim("dur_conv_dim", conv_sizes[0])] * num_layers
        else:
            out_dims = [
                nn.FeatureDim("dur_conv_dim_%s" % str(x), conv_sizes[x])
                for x in range(num_layers)
            ]

        self.modules = nn.ModuleList()
        self.norms = nn.ModuleList()

        temp_in_dim = in_dim
        for x in range(num_layers):
            self.modules.append(
                nn.Conv1d(
                    in_dim=temp_in_dim,
                    out_dim=out_dims[x],
                    filter_size=filter_sizes[x],
                    padding="same",
                    with_bias=False,
                )
            )
            temp_in_dim = out_dims[x]
            self.norms.append(nn.Normalize(param_shape=out_dims[x]))

        self.linear = nn.Linear(in_dim=temp_in_dim, out_dim=self.lin_dim)

    def __call__(self, inp: nn.Tensor, time_dim: nn.Dim) -> nn.Tensor:
        """
            Applies numlayers convolutional layers with a linear transformation at the end

            :param inp: should be concat of lstm fw bw and speaker embedding
            :param time_dim: time dimension tag
            :return:
            """
        x = inp
        for (i, module) in enumerate(self.modules):
            x, _ = module(x, in_spatial_dim=time_dim)
            for param in module.parameters():
                param.weight_decay = self.l2

            x = nn.relu(x)
            x = self.norms[i](x, axis=x.feature_dim)
            x = nn.dropout(x, axis=x.feature_dim, dropout=self.dropout[i])

        linear = self.linear(x)
        softplus = nn.log(nn.exp(linear) + 1.0)

        return softplus


class VariancePredictor(nn.Module):
    """
      Pitch predictor from FastSpeech 2
      """

    def __init__(
            self, in_dim: nn.Dim, conv_dim: Union[int, nn.Dim] = 256, filter_size: int = 3, dropout: float = 0.1
    ):
        """

        :param conv_dim:
        :param filter_size:
        :param dropout: default from NVIDIA implementation, differs from other defaults in this network
        """
        super(VariancePredictor, self).__init__()
        self.dropout = dropout

        self.conv_1 = nn.Conv1d(
            in_dim=in_dim,
            out_dim=conv_dim
            if isinstance(conv_dim, nn.Dim)
            else nn.FeatureDim("pitch_conv_dim", conv_dim),
            filter_size=filter_size,
            padding="same",
            with_bias=False,
        )
        self.norm_1 = nn.LayerNorm(in_dim=conv_dim)
        self.conv_2 = nn.Conv1d(
            in_dim=conv_dim,
            out_dim=conv_dim
            if isinstance(conv_dim, nn.Dim)
            else nn.FeatureDim("pitch_conv_dim", conv_dim),
            filter_size=filter_size,
            padding="same",
            with_bias=False,
        )
        self.norm_2 = nn.LayerNorm(in_dim=conv_dim)
        self.linear = nn.Linear(in_dim=conv_dim, out_dim=nn.FeatureDim("pitch_pred", 1))

    def __call__(self, inp: nn.Tensor, time_dim: nn.SpatialDim) -> nn.Tensor:
        x, _ = self.conv_1(inp, in_spatial_dim=time_dim)
        x = nn.relu(x)
        x = self.norm_1(x)
        x = nn.dropout(
            x, dropout=self.dropout, axis=[x.feature_dim]
        )
        x, _ = self.conv_2(x, in_spatial_dim=time_dim)
        x = nn.relu(x)
        x = self.norm_2(x)
        x = nn.dropout(x, dropout=self.dropout, axis=[x.feature_dim])
        x = self.linear(x)
        return x


class NARTTSModel(nn.Module):
    """
      NAR TTS Model from Timur Schümann implemented in returnn common
      """

    def __init__(
            self,
            label_in_dim: nn.Dim,
            speaker_in_dim: nn.Dim,
            embedding_size: int = 256,
            speaker_embedding_size: int = 256,
            enc_lstm_size: int = 256,
            dec_lstm_size: int = 1024,
            dropout: float = 0.5,
            training: bool = True,
            use_true_durations: bool = False,
            dump_speaker_embeddings: bool = False,
            calc_speaker_embedding: bool = False,
            duration_scale: float = 1.0,
            skip_speaker_embeddings: bool = False,
            test_vae: bool = False,
    ):
        super(NARTTSModel, self).__init__()

        # params
        self.dropout = dropout
        self.training = training
        self.use_true_durations = use_true_durations
        self.dump_speaker_embeddings = dump_speaker_embeddings
        self.calc_speaker_embedding = calc_speaker_embedding
        self.duration_scale = duration_scale
        self.skip_speaker_embeddings = skip_speaker_embeddings
        self.test_vae = test_vae

        # dims
        self.embedding_dim = nn.FeatureDim("embedding_dim", embedding_size)
        self.speaker_embedding_dim = nn.FeatureDim(
            "speaker_embedding_dim", speaker_embedding_size
        )
        self.enc_lstm_dim = nn.FeatureDim("enc_lstm_dim", enc_lstm_size)
        self.dec_lstm_size = dec_lstm_size

        # layers
        self.feature_extractor = FeatureExtractor()
        self.embedding = nn.Linear(in_dim=label_in_dim, out_dim=self.embedding_dim)
        self.speaker_embedding = nn.Linear(in_dim=speaker_in_dim, out_dim=self.speaker_embedding_dim)
        self.conv_stack = ConvStack(in_dim=self.embedding_dim, num_layers=3, filter_sizes=[3,3,3], dropout=[dropout, dropout, dropout])
        self.enc_lstm_fw = nn.LSTM(in_dim=self.conv_stack.out_dim, out_dim=self.enc_lstm_dim)
        self.enc_lstm_bw = nn.LSTM(in_dim=self.conv_stack.out_dim, out_dim=self.enc_lstm_dim)
        self.decoder = Decoder(in_dim=2*self.enc_lstm_dim + self.speaker_embedding_dim, dec_lstm_size_1=dec_lstm_size, dropout=dropout)

        self.variance_net = LstmVarianceNetwork(in_dim=2*self.enc_lstm_dim)
        self.upsamling = GaussianUpsampling()

        self.duration = DurationPredictor(in_dim=2*self.enc_lstm_dim + self.speaker_embedding_dim, dropout=(dropout, dropout))


    def __call__(
            self,
            text: nn.Tensor,
            durations: Union[nn.Tensor, None],
            speaker_labels: nn.Tensor,
            target_audio_samples: nn.Tensor,
            phon_time_dim: nn.Dim,
            speaker_label_time: nn.Dim,
            duration_time: Union[nn.Dim, None],
    ) -> nn.Tensor:
        """

            :param text:
            :param durations:
            :param speaker_labels:
            :param target_speech:
            :param phon_time_dim: data label time
            :param speaker_label_time: speaker label time
            :param speech_time: audio time
            :return:
            """
        duration_int = None
        duration_float = None
        # input data prep
        if self.training or self.use_true_durations:
            assert durations is not None
            durations, _ = nn.reinterpret_new_dim(
                durations, in_dim=duration_time, out_dim=phon_time_dim
            )  # we know that the durations match the phonemes, so adapt to encoder
            duration_float = nn.cast(durations, dtype="float32")  # [B, Label-time, 1]
            durations = nn.squeeze(durations, axis=durations.feature_dim)
            duration_int = nn.cast(durations, dtype="int32")  # [B, Label-time]
        speaker_notime = nn.squeeze(speaker_labels, axis=speaker_label_time)
        if self.training or self.calc_speaker_embedding:
            speaker_embedding = self.speaker_embedding(speaker_notime)
        else:
            speaker_embedding = speaker_notime
        if self.dump_speaker_embeddings:
            return speaker_embedding

        # target data prep
        audio_linear_features, audio_logmel_features, audio_time = self.feature_extractor(target_audio_samples)

        # embedding
        emb = self.embedding(text)

        # conv block
        conv = self.conv_stack(emb, time_dim=phon_time_dim)

        # lstm encoder
        enc_lstm_fw, _ = self.enc_lstm_fw(conv, spatial_dim=phon_time_dim, direction=1)
        enc_lstm_bw, _ = self.enc_lstm_bw(conv, spatial_dim=phon_time_dim, direction=-1)
        cat, _  = nn.concat(
            (enc_lstm_fw, enc_lstm_fw.feature_dim),
            (enc_lstm_bw, enc_lstm_bw.feature_dim),
        )
        encoder = nn.dropout(cat, dropout=self.dropout, axis=cat.feature_dim)

        # duration predictor
        duration_in, _  = nn.concat(
            (cat, cat.feature_dim),
            (speaker_embedding, speaker_embedding.feature_dim),
            allow_broadcast=True,
        )
        duration_in = nn.dropout(
            duration_in, dropout=self.dropout, axis=duration_in.feature_dim
        )

        if self.use_true_durations:
            duration_prediction = duration_float
        else:
            duration_prediction = self.duration(
                inp=duration_in, time_dim=phon_time_dim
            )  # [B, Label-time, 1]

        if self.training:
            duration_prediction, _ = nn.reinterpret_new_dim(duration_prediction, in_dim=duration_prediction.feature_dim, out_dim=duration_float.feature_dim)
            duration_prediction_loss = nn.mean_absolute_difference(
                duration_prediction, duration_float
            )
            duration_prediction_loss.mark_as_loss(name="duration_loss")
        else:
            rint = nn.rint(duration_prediction)
            duration_int = nn.cast(rint, dtype="int32")
            duration_int = nn.squeeze(duration_int, axis=duration_int.feature_dim)


        var = self.variance_net(inp=encoder, durations=duration_int, time_dim=phon_time_dim)
        rep, rep_dim = self.upsamling(
            inp=encoder,
            durations=duration_int,
            variances=var,
            time_dim=phon_time_dim,
            out_dim=audio_time,
        )

        # decoder
        dec_mel = self.decoder(
            rep=rep, speaker_embedding=speaker_embedding, time_dim=rep_dim
        )

        # prepare target speech for loss
        if self.training:
            rep_dim.declare_same_as(audio_time)
            audio_logmel_features, _ = nn.reinterpret_new_dim(
                audio_logmel_features,
                in_dim=audio_logmel_features.feature_dim,
                out_dim=dec_mel.feature_dim,
                name="reinterpred_logmel"
            )

            dec_logmel_loss = nn.mean_absolute_difference(dec_mel, audio_logmel_features)
            dec_logmel_loss.mark_as_loss("decoder_logmel_mea_loss")

        return dec_mel


def construct_network(
        epoch: int,
        phoneme_data: nn.Data,  # phoneme labels
        speaker_label_data: nn.Data,  # speaker labels
        phoneme_duration_data: Optional[nn.Data] = None,  # durations
        audio_data: Optional[nn.Data] = None,  # target speech
        **kwargs
):
    net = NARTTSModel(
        label_in_dim = phoneme_data.feature_dim_or_sparse_dim,
        speaker_in_dim = speaker_label_data.feature_dim_or_sparse_dim,
        **kwargs
    )

    phonemes = nn.get_extern_data(phoneme_data) if phoneme_data is not None else None

    out = net(
        text=phonemes,
        durations=nn.get_extern_data(phoneme_duration_data) if phoneme_duration_data is not None else None,
        speaker_labels=nn.get_extern_data(speaker_label_data) if speaker_label_data is not None else None,
        target_audio_samples=nn.get_extern_data(audio_data) if audio_data is not None else None,
        phon_time_dim=phoneme_data.dim_tags[phoneme_data.time_dim_axis],
        speaker_label_time=speaker_label_data.dim_tags[speaker_label_data.time_dim_axis],
        duration_time=phoneme_duration_data.dim_tags[phoneme_duration_data.time_dim_axis],
    )
    out.mark_as_default_output()

    return net
