"""
Implementation of the CTC NAR Network
"""
from returnn_common import nn
from typing import Tuple, Union, Optional


class Conv1DBlock(nn.Module):
    """
    1D Convolutional Block
    """

    def __init__(
        self,
        dim: Union[int, nn.Dim] = 256,
        filter_size: int = 5,
        bn_epsilon: float = 1e-5,
        dropout: float = 0.5,
        l2: float = 1e-07,
    ):
        """
        :param dim: feature dimension of the convolution
        :param filter_size: filter size of the conv, int because we are doing 1D here
        :param bn_epsilon: batch_normalization epsilon value
        :param dropout: dropout value
        :param l2: weight decay value
        """
        super(Conv1DBlock, self).__init__()
        if isinstance(dim, int):
            self.conv_dim = nn.FeatureDim("conv_dim_%d" % dim, dim)
        elif isinstance(dim, nn.Dim):
            self.conv_dim = dim
        else:
            raise Exception("Wrong Dim given!")
        self.conv = nn.Conv1d(
            out_dim=self.conv_dim,
            filter_size=filter_size,
            padding="same",
            with_bias=False,
        )
        self.bn = nn.BatchNorm(
            epsilon=bn_epsilon, use_mask=False
        )
        self.dropout = dropout
        self.l2 = l2

    def __call__(self, inp: nn.Tensor, time_dim: nn.SpatialDim) -> nn.Tensor:
        """
        First convolution, relu, then L2 regularization, batchnorm and dropout

        :param inp: input tensor
        :return:
        """
        conv, _ = self.conv(inp, in_spatial_dim=time_dim)
        # set weight decay
        for param in self.conv.parameters():
            param.weight_decay = self.l2

        conv = nn.relu(conv)
        bn = self.bn(conv)
        drop = nn.dropout(
            bn, dropout=self.dropout, axis=[nn.batch_dim, time_dim, bn.feature_dim]
        )

        return drop


class ConvStack(nn.Module):
    """
    Stacks :class:`Conv1DBlock` modules
    """

    def __init__(
        self,
        num_layers: int = 3,
        dim_sizes: Tuple[int] = (256, 256, 256),
        filter_sizes: Tuple[int] = (5, 5, 5),
        bn_epsilon: float = 1e-5,
        dropout: Tuple[float] = (0.5, 0.5, 0.5),
        l2: float = 1e-07,
    ):
        """
        :param num_layers: number of conv block layers
        :param dim_sizes: dimensions for the convolutions in the block
        :param filter_sizes: sizes for the filters in the block
        :param bn_epsilon: batch_normalization epsilon value
        :param dropout: dropout values
        :param l2: weight decay value
        """
        super(ConvStack, self).__init__()
        assert len(dim_sizes) == num_layers  # mismatch in dim_sizes
        assert len(filter_sizes) == num_layers  # mismatch in filter_sizes
        assert len(dropout) == num_layers  # mismatch in dropout

        self.num_layers = num_layers
        # simplify tags a bit if possible
        if len(set(dim_sizes)) == 1:  # all sizes equal
            out_dims = [nn.FeatureDim("conv_dim", dim_sizes[0])] * num_layers
        else:
            out_dims = [
                nn.FeatureDim("conv_dim_%s" % str(x), dim_sizes[x])
                for x in range(num_layers)
            ]

        self.stack = nn.Sequential(
            [
                Conv1DBlock(
                    dim=out_dims[x],
                    filter_size=filter_sizes[x],
                    bn_epsilon=bn_epsilon,
                    dropout=dropout[x],
                    l2=l2,
                )
                for x in range(num_layers)
            ]
        )

    def __call__(self, inp: nn.Tensor, time_dim: nn.Dim) -> nn.Tensor:
        """
        Applies all conv blocks in sequence

        :param inp: input tensor
        :return:
        """
        out = self.stack(inp, time_dim=time_dim)
        return out


class LConvBlock(nn.Module):
    """
    LConvBlock from Parallel tacotron 1 with grouped/depthwise convolution instead of full lightweight conv
    """
    def __init__(self, linear_size: Union[int, nn.Dim] = 512, num_heads: int = 8, filter_size: int = 17):
        super(LConvBlock, self).__init__()
        if isinstance(linear_size, int):
            self.groups = linear_size
            self.lin_dim = nn.FeatureDim("LConv_linear", linear_size)
        else:
            self.lin_dim = linear_size
            self.groups = linear_size.dimension

        self.glu = nn.Linear(self.lin_dim * 2)
        self.linear_up = nn.Linear(self.lin_dim * 4)
        self.linear_down = nn.Linear(self.lin_dim)
        self.conv = nn.Conv1d(
            out_dim=self.lin_dim,
            filter_size=filter_size,
            groups=self.groups,
            with_bias=False,
            padding='same'
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
        dec_lstm_size_1: int = 1024,
        dec_lstm_size_2: int = 1024,
        linear_size: int = 80,
        dropout: int = 0.5,
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

        self.dec_lstm_fw_1 = nn.LSTM(out_dim=self.dec_lstm_dim_1)
        self.dec_lstm_bw_1 = nn.LSTM(out_dim=self.dec_lstm_dim_1)
        self.dec_lstm_fw_2 = nn.LSTM(out_dim=self.dec_lstm_dim_2)
        self.dec_lstm_bw_2 = nn.LSTM(out_dim=self.dec_lstm_dim_2)
        self.linear_out = nn.Linear(out_dim=self.linear_dim)
        self.linear_ref = nn.Linear(out_dim=self.linear_dim)

    def __call__(self, rep: nn.Tensor, speaker_embedding: nn.Tensor, time_dim: nn.Dim) -> nn.Tensor:
        """

        :param rep: upsampled / repeated input
        :param speaker_embedding: speaker label embedding
        :param time_dim: time dimension tag
        :return:
        """

        cat = nn.concat(
            (rep, rep.feature_dim),
            (speaker_embedding, speaker_embedding.feature_dim),
            allow_broadcast=True,
        )

        dec_lstm_fw, _ = self.dec_lstm_fw_1(cat, axis=time_dim, direction=1)
        dec_lstm_bw, _ = self.dec_lstm_bw_1(cat, axis=time_dim, direction=-1)

        cat = nn.concat(
            (dec_lstm_fw, dec_lstm_fw.feature_dim),
            (dec_lstm_bw, dec_lstm_bw.feature_dim),
        )
        dec_drop = nn.dropout(cat, axis=cat.feature_dim, dropout=self.dropout)

        dec_lstm_fw, _ = self.dec_lstm_fw_2(dec_drop, axis=time_dim, direction=1)
        dec_lstm_bw, _ = self.dec_lstm_bw_2(dec_drop, axis=time_dim, direction=-1)

        cat = nn.concat(
            (dec_lstm_fw, dec_lstm_fw.feature_dim),
            (dec_lstm_bw, dec_lstm_bw.feature_dim),
        )
        dec_drop = nn.dropout(cat, axis=cat.feature_dim, dropout=self.dropout)

        lin_out = self.linear_out(dec_drop)
        # lin_ref = self.linear_ref(dec_drop)

        # return lin_out, lin_ref
        return lin_out


class DurationPredictor(nn.Module):
    """
    Module for duration prediction
    During training not connected, only loss, during forward then actually predicts
    """

    def __init__(
        self,
        num_layers: int = 2,
        conv_sizes: Tuple[int] = (256, 256),
        filter_sizes: Tuple[int] = (3, 3),
        dropout: Tuple[float] = (0.5, 0.5),
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

        for x in range(num_layers):
            self.modules.append(
                nn.Conv1d(
                    out_dim=out_dims[x],
                    filter_size=filter_sizes[x],
                    padding="same",
                    with_bias=False,
                )
            )
            self.norms.append(nn.Normalize(param_shape=out_dims[x]))

        self.linear = nn.Linear(out_dim=self.lin_dim)

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


class VariationalAutoEncoder(nn.Module):
  """
  VAE from Tacotron 1
  """

  def __init__(self, num_lconv_blocks: int = 3 , num_mixed_blocks: int = 6, linear_size: Union[int, nn.Dim] = 512,
               num_heads: int = 8,
               lconv_filter_size: int = 17, dropout: float = 0.5, conv_filter_size: int = 5, out_size=32):
    super(VariationalAutoEncoder, self).__init__()
    self.lconv_stack = nn.Sequential(
        [LConvBlock(
            linear_size=linear_size, num_heads=num_heads, filter_size=lconv_filter_size) for _ in range(num_lconv_blocks)])
    self.mixed_stack = nn.Sequential()
    for _ in range(num_mixed_blocks):
        self.mixed_stack.append(LConvBlock(linear_size=linear_size, num_heads=num_heads, filter_size=lconv_filter_size))
        self.mixed_stack.append(Conv1DBlock(dim=linear_size, filter_size=conv_filter_size, dropout=dropout))
    self.lin_out = nn.Linear(nn.FeatureDim("VAE_out_dim", out_size))

  def __call__(self, spectrogramm: nn.Tensor, spectrogramm_time: nn.SpatialDim) -> nn.Tensor:
    x = self.lconv_stack(spectrogramm, time_dim=spectrogramm_time)
    x = self.mixed_stack(x, time_dim=spectrogramm_time)
    x = nn.reduce(x, mode="mean", axis=spectrogramm_time, use_time_mask=True)  # [B, F]
    x = self.lin_out(x)  # [B, F]
    return x


class NARTTSModel(nn.Module):
    """
    NAR TTS Model from Timur Schümann implemented in returnn common
    """

    def __init__(
        self,
        embedding_size: int = 256,
        speaker_embedding_size: int = 256,
        enc_lstm_size: int = 256,
        dec_lstm_size: int = 1024,
        dropout: int = 0.5,
        training: bool = True,
        gauss_up: bool = False,
        use_true_durations: bool = False,
        dump_speaker_embeddings: bool = False,
        dump_vae: bool = False,
        use_vae: bool = False,
        calc_speaker_embedding: bool = False,
    ):
        super(NARTTSModel, self).__init__()

        # params
        self.dropout = dropout
        self.training = training
        self.gauss_up = gauss_up
        self.use_true_durations = use_true_durations
        self.dump_speaker_embeddings = dump_speaker_embeddings
        self.calc_speaker_embedding = calc_speaker_embedding
        if dump_vae:
          assert use_vae, "Needs to use VAE in order to dump it!"
        self.dump_vae = dump_vae

        # dims
        self.embedding_dim = nn.FeatureDim("embedding_dim", embedding_size)
        self.speaker_embedding_dim = nn.FeatureDim(
            "speaker_embedding_dim", speaker_embedding_size
        )
        self.enc_lstm_dim = nn.FeatureDim("enc_lstm_dim", enc_lstm_size)
        self.dec_lstm_size = dec_lstm_size

        # layers
        self.embedding = nn.Linear(out_dim=self.embedding_dim)
        self.speaker_embedding = nn.Linear(out_dim=self.speaker_embedding_dim)
        self.conv_stack = ConvStack()
        self.enc_lstm_fw = nn.LSTM(out_dim=self.enc_lstm_dim)
        self.enc_lstm_bw = nn.LSTM(out_dim=self.enc_lstm_dim)
        self.decoder = Decoder(dec_lstm_size_1=dec_lstm_size, dropout=dropout)
        if self.gauss_up:
            # only import here to reduce serializer imports and not import for every model that doesn't use it
            from i6_experiments.users.hilmes.modules.gaussian_upsampling import (
                GaussianUpsampling,
                VarianceNetwork,
            )

            self.variance_net = VarianceNetwork()
            self.upsamling = GaussianUpsampling()
        else:
            self.variance_net = None
            self.upsamling = None
        self.duration = DurationPredictor()
        self.use_vae = use_vae
        if self.use_vae:
            vae_lin_dim = nn.FeatureDim("vae_embedding", 512)
            self.vae_embedding = nn.Linear(vae_lin_dim)
            self.vae = VariationalAutoEncoder(linear_size=vae_lin_dim)
        else:
            self.vae = None

    def __call__(
        self,
        text: nn.Tensor,
        durations: Union[nn.Tensor, None],
        speaker_labels: nn.Tensor,
        target_speech: nn.Tensor,
        time_dim: nn.Dim,
        label_time: nn.Dim,
        speech_time: Union[nn.Dim, None],
        duration_time: Union[nn.Dim, None],
        speaker_prior: Union[nn.Tensor, None],
        prior_time: Union[nn.Dim, None]
    ) -> nn.Tensor:
        """

        :param text:
        :param durations:
        :param speaker_labels:
        :param target_speech:
        :param time_dim: data label time
        :param label_time: speaker label time
        :param speech_time: audio time
        :return:
        """
        duration_int = None
        duration_float = None
        latents = None
        # input data prep
        if self.training or self.use_true_durations:
            durations, _ = nn.reinterpret_new_dim(
                durations, in_dim=duration_time, out_dim=time_dim
            )
            durations = nn.squeeze(durations, axis=durations.feature_dim)
            duration_int = nn.cast(durations, dtype="int32")
            duration_float = nn.cast(durations, dtype="float32")  # [B, Label-time]
        label_notime = nn.squeeze(speaker_labels, axis=label_time)
        if self.training or self.calc_speaker_embedding:
            speaker_embedding = self.speaker_embedding(label_notime)
        else:
            speaker_embedding = label_notime
        if self.use_vae:
            if speaker_prior is None:
                vae_speaker_embedding = self.vae_embedding(target_speech)  # TODO: this is not in the paper I think but doesnt make sense otherwise
                latents = self.vae(vae_speaker_embedding, speech_time)
            else:
                latents = nn.squeeze(speaker_prior, axis=prior_time)
            speaker_embedding = nn.concat(
                (speaker_embedding, speaker_embedding.feature_dim),
                (latents, latents.feature_dim)
            )

        if self.dump_vae and self.dump_speaker_embeddings:
          speaker_embedding = nn.concat(
            (speaker_embedding, speaker_embedding.feature_dim),
            (latents, latents.feature_dim)
          )
          return speaker_embedding
        elif self.dump_speaker_embeddings:
            return speaker_embedding
        elif self.dump_vae:
            return latents

        # embedding
        emb = self.embedding(text)

        # conv block
        conv = self.conv_stack(emb, time_dim=time_dim)

        # lstm encoder
        enc_lstm_fw, _ = self.enc_lstm_fw(conv, axis=time_dim, direction=1)
        enc_lstm_bw, _ = self.enc_lstm_bw(conv, axis=time_dim, direction=-1)
        cat = nn.concat(
            (enc_lstm_fw, enc_lstm_fw.feature_dim),
            (enc_lstm_bw, enc_lstm_bw.feature_dim),
        )
        encoder = nn.dropout(cat, dropout=self.dropout, axis=cat.feature_dim)

        # duration predictor
        duration_in = nn.concat(
            (enc_lstm_fw, enc_lstm_fw.feature_dim),
            (enc_lstm_bw, enc_lstm_bw.feature_dim),
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
                inp=duration_in, time_dim=time_dim
            )  # [B, Label-time, 1]

        if self.training:
            duration_prediction_loss = nn.mean_absolute_difference(
                duration_prediction, duration_float
            )
            duration_prediction_loss.mark_as_loss()
        else:
            rint = nn.rint(duration_prediction)
            duration_int = nn.cast(rint, dtype="int32")
            duration_int = nn.squeeze(duration_int, axis=duration_int.feature_dim)

        # upsampling
        if self.gauss_up:
            var = self.variance_net(
                inp=encoder, durations=duration_int, time_dim=time_dim
            )
            rep, rep_dim = self.upsamling(
                inp=encoder,
                durations=duration_int,
                variances=var,
                time_dim=time_dim,
                out_dim=speech_time,
            )
        else:
            rep, rep_dim = nn.repeat(
                encoder, axis=time_dim, repetitions=duration_int, out_dim=speech_time
            )

        # decoder
        dec_lin = self.decoder(
            rep=rep, speaker_embedding=speaker_embedding, time_dim=rep_dim
        )

        # prepare target speech for loss
        if self.training:
            target_speech, _ = nn.reinterpret_new_dim(
                target_speech,
                in_dim=target_speech.feature_dim,
                out_dim=dec_lin.feature_dim,
            )

            dec_lin_loss = nn.mean_absolute_difference(
                dec_lin, target_speech
            )
            dec_lin_loss.mark_as_loss()

        # dec_lin.mark_as_default_output()

        return dec_lin


def construct_network(
    epoch: int,
    net_module: nn.Module,
    phoneme_data: nn.Data,  # phoneme labels
    duration_data: nn.Data,  # durations
    label_data: nn.Data,  # speaker labels
    audio_data: nn.Data,  # target speech
    time_dim: nn.Dim,  # phoneme time dim
    label_time_dim: nn.Dim,  # speaker_label time
    speech_time_dim: nn.Dim,  # audio features time
    duration_time_dim: nn.Dim,  # durations time
    speaker_prior: Optional[nn.Data] = None,  # VAE speaker prior
    prior_time: Optional[nn.Dim] = None,  # VAE speaker prior time
    **kwargs
):
    net = net_module(**kwargs)
    out = net(
        text=nn.get_extern_data(phoneme_data) if phoneme_data is not None else None,
        durations=nn.get_extern_data(duration_data) if duration_data is not None else None,
        speaker_labels=nn.get_extern_data(label_data) if label_data is not None else None,
        target_speech=nn.get_extern_data(audio_data) if audio_data is not None else None,
        speaker_prior=nn.get_extern_data(speaker_prior) if speaker_prior is not None else None,
        time_dim=time_dim or None,
        label_time=label_time_dim or None,
        speech_time=speech_time_dim or None,
        duration_time=duration_time_dim or None,
        prior_time=prior_time or None
    )
    out.mark_as_default_output()

    return net
