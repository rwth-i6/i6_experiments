"""
Implementation of the CTC Aligner, updated for the new non-lazy init
"""
from typing import Tuple, Union
from returnn_common import nn


class Conv1DBlock(nn.Module):
    """
    1D Convolutional Block
    """

    def __init__(
        self,
        in_dim: nn.Dim,
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
            in_dim=in_dim,
            out_dim=self.conv_dim,
            filter_size=filter_size,
            padding="same",
            with_bias=False,
        )
        self.bn = nn.BatchNorm(
            in_dim=self.conv_dim,
            epsilon=bn_epsilon, use_mask=False
        )
        self.dropout = dropout
        self.l2 = l2

    def __call__(self, inp: nn.Tensor, time_dim: nn.SpatialDim):
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
        in_dim: nn.Dim,
        num_layers: int = 5,
        dim_sizes: Tuple[int] = (256,),
        filter_sizes: Tuple[int] = (5, 5, 5, 5, 5),
        bn_epsilon: float = 1e-5,
        dropout: Tuple[float] = (0.35, 0.35, 0.35, 0.35, 0.35),
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
        assert (
            len(dim_sizes) == num_layers or len(dim_sizes) == 1
        )  # mismatch in dim_sizes
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

        sequential_list = []
        temp_in_dim = in_dim
        for x in range(num_layers):
            sequential_list.append(
                Conv1DBlock(
                    in_dim=temp_in_dim,
                    dim=out_dims[x],
                    filter_size=filter_sizes[x],
                    bn_epsilon=bn_epsilon,
                    dropout=dropout[x],
                    l2=l2,
                )
            )
            temp_in_dim = out_dims[x]

        self.stack = nn.Sequential(sequential_list)
        self.out_dim = out_dims[-1]

    def __call__(self, inp: nn.Tensor, time_dim: nn.Dim):
        """
        Applies all conv blocks in sequence

        :param inp: input tensor
        :return:
        """
        out = self.stack(inp, time_dim=time_dim)
        return out


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
        in_feature_dim: nn.Dim,
        in_speaker_dim: nn.Dim,
        out_label_dim: nn.Dim,
        audio_emb_size: int = 256,
        speaker_emb_size: int = 256,
        hidden_size: int = 256,
        enc_lstm_size: int = 512,
        spectrogram_drop: float = 0.35,
        reconstruction_scale: int = 0.5,
        training=True,
    ):
        """
        :param audio_emb_size: Embedding size for audio features
        :param speaker_emb_size: Embedding size for speaker embedding
        :param hidden_size: Feature dimension inside the network
        :param enc_lstm_size: Encoder LSTM dimension size
        :param spectrogram_drop: Dropout for spectrogram encoder
        :param reconstruction_scale: Loss scale for reconstruction loss
        """
        super(CTCAligner, self).__init__()

        self.audio_hidden_dim = nn.FeatureDim("audio_hidden_dim", audio_emb_size)
        self.speaker_hidden_dim = nn.FeatureDim("speaker_hidden_dim", speaker_emb_size)
        self.hidden_dim = nn.FeatureDim("hidden_size", hidden_size)
        self.enc_lstm_dim = nn.FeatureDim("enc_lstm_dim", enc_lstm_size)

        self.audio_embedding = nn.Linear(in_dim=in_feature_dim, out_dim=self.audio_hidden_dim)
        self.speaker_embedding = nn.Linear(in_dim=in_speaker_dim, out_dim=self.speaker_hidden_dim)
        self.enc_conv_stack = ConvStack(in_dim=self.speaker_hidden_dim + self.audio_hidden_dim)
        self.enc_lstm_fw = nn.LSTM(in_dim=self.enc_conv_stack.out_dim, out_dim=self.enc_lstm_dim)
        self.enc_lstm_bw = nn.LSTM(in_dim=self.enc_conv_stack.out_dim, out_dim=self.enc_lstm_dim)

        self.softmax_dim = nn.FeatureDim("softmax_linear", out_label_dim.dimension)

        self.softmax_lin = nn.Linear(
            in_dim=2*self.enc_lstm_dim,
            out_dim=self.softmax_dim,
        )
        self.tts_decoder = TTSDecoder(in_dim=self.softmax_dim + self.speaker_hidden_dim)
        self.reconstruction_lin = nn.Linear(
            in_dim=2*self.tts_decoder.lstm_dim,
            out_dim=nn.FeatureDim("reconstruction_dim", 80)
        )

        self.spectrogram_drop = spectrogram_drop
        self.reconstruction_scale = reconstruction_scale
        self.training = training

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
            audio_features, _ = nn.reinterpret_new_dim(
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
    net_module: nn.Module,
    audio_data: nn.Data,
    label_data: nn.Data,
    phoneme_data: nn.Data,
    audio_time_dim: nn.Dim,
    label_time_dim: nn.Dim,
    phoneme_time_dim: nn.Dim,
    **kwargs
):
    """

    :param epoch:
    :param net_module:
    :param audio_data:
    :param label_data:
    :param phoneme_data:
    :param audio_time_dim:
    :param label_time_dim:
    :param kwargs:
    :return:
    """
    net = net_module(
        in_feature_dim=audio_data.feature_dim_or_sparse_dim,
        in_speaker_dim=label_data.feature_dim_or_sparse_dim,
        out_label_dim=phoneme_data.feature_dim_or_sparse_dim,
        **kwargs
    )
    out = net(
        audio_features=nn.get_extern_data(audio_data),
        speaker_labels=nn.get_extern_data(label_data),
        phonemes=nn.get_extern_data(phoneme_data),
        audio_time=audio_time_dim,
        speaker_label_time=label_time_dim,
        phoneme_time=phoneme_time_dim,
    )
    out.mark_as_default_output()

    return net
