"""
Implementation of the CTC NAR Network
"""
from returnn_common import nn
from typing import Tuple, Union, Optional, Sequence

from .conv_modules import Conv1DBlock, ConvStack

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


class VariationalAutoEncoder(nn.Module):
    """
      VAE from Tacotron 1
      """

    def __init__(
            self,
            in_dim: nn.Dim,
            num_lconv_blocks: int = 3,
            num_mixed_blocks: int = 6,
            linear_size: Union[int, nn.Dim] = 512,
            lconv_filter_size: int = 17,
            dropout: float = 0.5,
            conv_filter_size: int = 5,
            out_size: int = 32,
            latent_size: int = 8,
    ):
        super(VariationalAutoEncoder, self).__init__()

        self.lconv_stack = nn.Sequential()
        temp_in = in_dim
        for _ in range(num_lconv_blocks):
            lconv_block = LConvBlock(
                in_dim=temp_in, linear_size=linear_size, filter_size=lconv_filter_size
            )
            self.lconv_stack.append(lconv_block)
            temp_in = lconv_block.lin_dim

        self.mixed_stack = nn.Sequential()
        for _ in range(num_mixed_blocks):
            lconv_block = LConvBlock(
                in_dim=temp_in, linear_size=linear_size, filter_size=lconv_filter_size
            )
            self.mixed_stack.append(lconv_block)
            temp_in = lconv_block.lin_dim
            conv1d_block = Conv1DBlock(in_dim=temp_in, dim=linear_size, filter_size=conv_filter_size, dropout=dropout)
            self.mixed_stack.append(conv1d_block)
            temp_in = conv1d_block.conv_dim

        self.latent_dim = nn.FeatureDim("VAE_lat_dim", latent_size)
        self.lin_mu = nn.Linear(in_dim=temp_in, out_dim=self.latent_dim)
        self.lin_log_var = nn.Linear(in_dim=temp_in, out_dim=self.latent_dim)
        self.lin_out = nn.Linear(in_dim=self.latent_dim, out_dim=nn.FeatureDim("VAE_out_dim", out_size))

    def __call__(
            self, spectrogramm: nn.Tensor, spectrogramm_time: nn.SpatialDim
    ) -> [nn.Tensor, nn.Tensor, nn.Tensor]:
        x = self.lconv_stack(spectrogramm, time_dim=spectrogramm_time)
        x = self.mixed_stack(x, time_dim=spectrogramm_time)
        x = nn.reduce(x, mode="mean", axis=spectrogramm_time, use_time_mask=True)  # [B, F]

        # draw latent
        mu = self.lin_mu(x)  # [B, F]
        log_var = self.lin_log_var(x)
        std = nn.exp(0.5 * log_var)
        eps = nn.random_normal(std.shape_ordered)
        eps = eps * std
        eps = eps + mu

        out = self.lin_out(eps)
        return out, mu, log_var


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
            gauss_up: bool = False,
            use_true_durations: bool = False,
            dump_speaker_embeddings: bool = False,
            calc_speaker_embeddings: bool = False,
            dump_vae: bool = False,
            use_vae: bool = False,
            kl_beta: float = 1.0,
            use_pitch_pred: bool = False,
            use_energy_pred: bool = False,
            duration_add: float = 0.0,
            skip_speaker_embeddings: bool = False,
            test_vae: bool = False,
            log_energy: bool = False,
            scale_kl_loss: bool = False,
            vae_usage: str = "speak_emb_cat",
            use_true_pitch: bool = False,
            use_true_energy: bool = False,
            test: bool = False,
    ):
        super(NARTTSModel, self).__init__()

        # params
        self.dropout = dropout
        self.training = training
        self.gauss_up = gauss_up
        self.use_true_durations = use_true_durations
        self.dump_speaker_embeddings = dump_speaker_embeddings
        self.calc_speaker_embeddings = calc_speaker_embeddings
        assert use_vae or not dump_vae, "Needs to use VAE in order to dump it!"
        self.dump_vae = dump_vae
        self.use_pitch_pred = use_pitch_pred
        self.use_energy_pred = use_energy_pred
        self.duration_add = duration_add
        self.skip_speaker_embeddings = skip_speaker_embeddings
        self.test_vae = test_vae
        assert use_energy_pred or not log_energy, "Ja, ich mach auch immer log auf ne Energy die ich gar nicht predicte..."
        self.log_energy = log_energy
        assert use_vae or not scale_kl_loss, "Wat soll ich scalen wenns das nicht gibt!"
        self.scale_kl_loss = scale_kl_loss
        assert vae_usage in ["speak_emb_cat", "energy_add", "energy_mul"], "Not supported usage of the VAE"
        if vae_usage == "energy_add":
            assert use_energy_pred, "Need to pred energy to add VAE"
        self.vae_usage = vae_usage
        assert use_pitch_pred or not use_true_pitch
        self.use_true_pitch = use_true_pitch
        assert use_energy_pred or not use_true_energy
        self.use_true_energy = use_true_energy
        self.test = test

        # dims
        self.embedding_dim = nn.FeatureDim("embedding_dim", embedding_size)
        self.speaker_embedding_dim = nn.FeatureDim(
            "speaker_embedding_dim", speaker_embedding_size
        )
        self.enc_lstm_dim = nn.FeatureDim("enc_lstm_dim", enc_lstm_size)
        self.dec_lstm_size = dec_lstm_size

        # layers
        self.embedding = nn.Linear(in_dim=label_in_dim, out_dim=self.embedding_dim)
        self.speaker_embedding = nn.Linear(in_dim=speaker_in_dim, out_dim=self.speaker_embedding_dim)
        self.conv_stack = ConvStack(in_dim=self.embedding_dim, dim_sizes=[embedding_size] * 3, num_layers=3, filter_sizes=[3,3,3], dropout=[dropout, dropout, dropout])
        self.enc_lstm_fw = nn.LSTM(in_dim=self.conv_stack.out_dim, out_dim=self.enc_lstm_dim)
        self.enc_lstm_bw = nn.LSTM(in_dim=self.conv_stack.out_dim, out_dim=self.enc_lstm_dim)
        self.decoder = Decoder(in_dim=2*self.enc_lstm_dim + self.speaker_embedding_dim, dec_lstm_size_1=dec_lstm_size, dropout=dropout)
        if self.gauss_up:
            # only import here to reduce serializer imports and not import for every model that doesn't use it
            from .gaussian_upsampling import (
                GaussianUpsampling,
                LstmVarianceNetwork,
            )
            self.variance_net = LstmVarianceNetwork(in_dim=2*self.enc_lstm_dim)
            self.upsamling = GaussianUpsampling()
        else:
            self.variance_net = None
            self.upsamling = None
        self.duration = DurationPredictor(in_dim=2*self.enc_lstm_dim + self.speaker_embedding_dim, dropout=(dropout, dropout))
        self.use_vae = use_vae
        self.kl_scale = kl_beta
        # if self.use_vae:
        #     vae_lin_dim = nn.FeatureDim("vae_embedding", 512)
        #     self.vae_embedding = nn.Linear(vae_lin_dim)
        #     if self.test_vae:
        #         self.test_lconv_stack = nn.Sequential(
        #             [
        #                 LConvBlock(
        #                     linear_size=vae_lin_dim, filter_size=17
        #                 )
        #                 for _ in range(3)
        #             ]
        #         )
        #         self.test_mixed_stack = nn.Sequential()
        #         for _ in range(6):
        #             self.test_mixed_stack.append(
        #                 LConvBlock(
        #                     linear_size=vae_lin_dim, filter_size=17
        #                 )
        #             )
        #             self.test_mixed_stack.append(
        #                 Conv1DBlock(dim=vae_lin_dim, filter_size=5, dropout=dropout)
        #             )
        #         self.test_latent_dim = nn.FeatureDim("VAE_lat_dim", 8)
        #         self.test_lin_mu = nn.Linear(self.test_latent_dim)
        #         self.test_lin_log_var = nn.Linear(self.test_latent_dim)
        #         self.test_lin_out = nn.Linear(nn.FeatureDim("VAE_out_dim", 512))
        #     elif self.skip_speaker_embeddings:
        #         self.vae = VariationalAutoEncoder(linear_size=vae_lin_dim, out_size=speaker_embedding_size, dropout=dropout)
        #     else:
        #         self.vae = VariationalAutoEncoder(linear_size=vae_lin_dim, dropout=dropout)
        # else:
        #     self.vae = None
        self.vae = None
        #if self.use_pitch_pred:
        #    self.pitch_pred = VariancePredictor()
        #    self.pitch_emb = nn.Conv1d(out_dim=2 * self.enc_lstm_dim,
        #                               filter_size=3,
        #                               padding="same",
        #                               with_bias=False)
        #else:
        #    self.pitch_pred = None
        #    self.pitch_emb = None
        self.pitch_pred = None
        self.pitch_emb = None
        # if self.use_energy_pred:
        #     self.energy_pred = VariancePredictor()
        #     self.energy_emb = nn.Linear(2 * self.enc_lstm_dim)
        # else:
        #     self.energy_pred = None
        #     self.energy_emb = None
        self.energy_pred = None
        self.energy_emb = None

    def __call__(
            self,
            text: nn.Tensor,
            durations: Union[nn.Tensor, None],
            speaker_labels: nn.Tensor,
            target_speech: nn.Tensor,
            pitch: Union[nn.Tensor, None],
            energy: Union[nn.Tensor, None],
            phon_time_dim: nn.Dim,
            speaker_label_time: nn.Dim,
            speech_time: Union[nn.Dim, None],
            duration_time: Union[nn.Dim, None],
            speaker_prior: Union[nn.Tensor, None],
            prior_time: Union[nn.Dim, None],
            pitch_time: Union[nn.Dim, None],
            energy_time: Union[nn.Dim, None],
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
        latents = None
        # input data prep
        if self.training or self.use_true_durations:
            durations, _ = nn.reinterpret_new_dim(
                durations, in_dim=duration_time, out_dim=phon_time_dim
            )  # we know that the durations match the phonemes, so adapt to encoder
            duration_float = nn.cast(durations, dtype="float32")  # [B, Label-time, 1]
            durations = nn.squeeze(durations, axis=durations.feature_dim)
            duration_int = nn.cast(durations, dtype="int32")  # [B, Label-time]
        label_notime = nn.squeeze(speaker_labels, axis=speaker_label_time)
        if self.training or self.calc_speaker_embeddings:
            speaker_embedding = self.speaker_embedding(label_notime)
        else:
            speaker_embedding, _ = nn.reinterpret_new_dim(label_notime, in_dim=speaker_labels.feature_dim, out_dim=self.speaker_embedding_dim)
        if self.use_vae and not self.test_vae:
            if speaker_prior is None:
                vae_speaker_embedding = self.vae_embedding(target_speech)
                latents, mu, log_var = self.vae(vae_speaker_embedding, speech_time)
                if self.training:
                    comb = log_var - (mu ** 2) - nn.exp(log_var)
                    kl_loss = -0.5 * (1 + comb)
                    if self.scale_kl_loss:
                        norm = nn.minimum(nn.cast(nn.global_train_step(), dtype="float32") / 100000.0, 1.0)
                        kl_loss = kl_loss * norm
                        kl_loss.mark_as_loss(scale=self.kl_scale)
                    else:
                        kl_loss.mark_as_loss(scale=self.kl_scale)
                if self.skip_speaker_embeddings:
                    speaker_embedding = latents
                elif self.vae_usage == "speak_emb_cat":
                    speaker_embedding = nn.concat_features(
                        speaker_embedding, latents,
                    )
            else:
                latents = nn.squeeze(speaker_prior, axis=prior_time)
                if self.skip_speaker_embeddings:
                    speaker_embedding = latents
                elif self.vae_usage == "speak_emb_cat":
                    speaker_embedding = nn.concat(
                        (speaker_embedding, speaker_embedding.feature_dim),
                        (latents, latents.feature_dim),
                    )

        if self.dump_vae and self.dump_speaker_embeddings:
            speaker_embedding = nn.concat(
                (speaker_embedding, speaker_embedding.feature_dim),
                (latents, latents.feature_dim),
            )
            return speaker_embedding
        elif self.dump_speaker_embeddings:
            return speaker_embedding
        elif self.dump_vae:
            return latents

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
        if self.use_pitch_pred:
            if self.use_true_pitch:
                pitch_time.declare_same_as(phon_time_dim)
                pitch_pred = pitch
            else:
                pitch_pred = self.pitch_pred(duration_in, phon_time_dim)
            if self.training:
                pitch, _ = nn.reinterpret_new_dim(pitch, in_dim=pitch_time, out_dim=phon_time_dim)
                pitch, _ = nn.reinterpret_new_dim(pitch, in_dim=pitch.feature_dim, out_dim=pitch_pred.feature_dim)
                pitch_loss = nn.squared_difference(pitch_pred, pitch, name="pitch_loss")
                pitch_loss.mark_as_loss()
                if self.test:
                    pitch_pred = pitch  # this is new, maybe we got a problem here
            pitch_embedding, _ = self.pitch_emb(pitch_pred, in_spatial_dim=phon_time_dim)
            encoder = encoder + pitch_embedding

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
            if self.duration_add != 0:
                duration_prediction = duration_prediction + self.duration_add
            rint = nn.rint(duration_prediction)
            duration_int = nn.cast(rint, dtype="int32")
            duration_int = nn.squeeze(duration_int, axis=duration_int.feature_dim)

        # upsampling
        if self.gauss_up:
            var = self.variance_net(inp=encoder, durations=duration_int, time_dim=phon_time_dim)
            rep, rep_dim = self.upsamling(
                inp=encoder,
                durations=duration_int,
                variances=var,
                time_dim=phon_time_dim,
                out_dim=speech_time,
            )
        else:
            if self.test_vae:
                vae_speaker_embedding = self.vae_embedding(encoder)
                x = self.test_lconv_stack(vae_speaker_embedding, time_dim=phon_time_dim)
                x = self.test_mixed_stack(x, time_dim=phon_time_dim)

                # draw latent
                mu = self.test_lin_mu(x)  # [B, F]
                log_var = self.test_lin_log_var(x)
                std = nn.exp(0.5 * log_var)
                eps = nn.random_normal(std.shape_ordered)
                eps = eps * std
                eps = eps + mu

                out = self.test_lin_out(eps)
                if self.training:
                    comb = log_var - (mu ** 2) - nn.exp(log_var)
                    kl_loss = -0.5 * (1 + comb)
                    kl_loss.mark_as_loss(name="kl_loss", scale=self.kl_scale)
                encoder = out

            rep, rep_dim = nn.repeat(encoder, axis=phon_time_dim, repetitions=duration_int)

        if self.use_energy_pred:
            if self.use_true_energy:
                energy_time.declare_same_as(rep_dim)
                energy_pred = energy
            else:
                energy_pred = self.energy_pred(rep, rep_dim)
            if self.vae_usage == "energy_add":
                latents = nn.reduce(latents, axis=latents.feature_dim, mode="mean")
                energy_pred += latents
            elif self.vae_usage == "energy_mul":
                latents = nn.reduce(latents, axis=latents.feature_dim, mode="mean")
                energy_pred *= latents
            if self.log_energy:
                energy_pred = nn.safe_log(energy_pred)
            if self.training:
                energy.feature_dim.declare_same_as(energy_pred.feature_dim)
                energy_time.declare_same_as(rep_dim)
                if self.log_energy:
                    energy = nn.safe_log(energy)
                energy_loss = nn.squared_difference(energy_pred, energy)
                #if self.log_energy:
                #  energy_loss = nn.exp(energy_loss)
                energy_loss.mark_as_loss(name="energy_loss")
                if self.test:
                    energy_pred = energy
            energy_embedding = self.energy_emb(energy_pred)
            rep += energy_embedding


        # decoder
        dec_lin = self.decoder(
            rep=rep, speaker_embedding=speaker_embedding, time_dim=rep_dim
        )

        # prepare target speech for loss
        if self.training:
            target_speech.feature_dim.declare_same_as(dec_lin.feature_dim)
            rep_dim.declare_same_as(speech_time)
            dec_lin_loss = nn.mean_absolute_difference(dec_lin, target_speech)
            dec_lin_loss.mark_as_loss("decoder_mea_loss")

        # dec_lin.mark_as_default_output()

        return dec_lin


def construct_network(
        epoch: int,
        phoneme_data: nn.Data,  # phoneme labels
        speaker_label_data: nn.Data,  # speaker labels
        phoneme_duration_data: Optional[nn.Data] = None,  # durations
        audio_data: Optional[nn.Data] = None,  # target speech
        speaker_prior_data: Optional[nn.Data] = None,  # VAE speaker prior
        pitch_data: Optional[nn.Data] = None,  # Pitch information
        energy_data: Optional[nn.Data] = None,
        **kwargs
):
    net = NARTTSModel(
        label_in_dim = phoneme_data.feature_dim_or_sparse_dim,
        speaker_in_dim = speaker_label_data.feature_dim_or_sparse_dim,
        **kwargs
    )

    phonemes = nn.get_extern_data(phoneme_data) if phoneme_data is not None else None
    speaker_prior = nn.get_extern_data(speaker_prior_data) if speaker_prior_data is not None else None
    pitch = nn.get_extern_data(pitch_data) if pitch_data is not None else None
    energy = nn.get_extern_data(energy_data) if energy_data is not None else None

    prior_time = speaker_prior_data.dim_tags[speaker_prior_data.time_dim_axis] if speaker_prior_data else None
    pitch_time = pitch_data.dim_tags[pitch_data.time_dim_axis] if pitch_data else None
    energy_time = pitch_data.dim_tags[energy_data.time_dim_axis] if pitch_data else None

    out = net(
        text=phonemes,
        durations=nn.get_extern_data(phoneme_duration_data) if phoneme_duration_data is not None else None,
        speaker_labels=nn.get_extern_data(speaker_label_data) if speaker_label_data is not None else None,
        target_speech=nn.get_extern_data(audio_data) if audio_data is not None else None,
        speaker_prior=speaker_prior,
        pitch=pitch,
        energy=energy,
        phon_time_dim=phoneme_data.dim_tags[phoneme_data.time_dim_axis],
        speaker_label_time=speaker_label_data.dim_tags[speaker_label_data.time_dim_axis],
        speech_time=audio_data.dim_tags[audio_data.time_dim_axis] if audio_data is not None else None,
        duration_time=phoneme_duration_data.dim_tags[phoneme_duration_data.time_dim_axis] if phoneme_duration_data is not None else None,
        prior_time=prior_time,
        pitch_time=pitch_time,
        energy_time=energy_time
    )
    out.mark_as_default_output()

    return net
