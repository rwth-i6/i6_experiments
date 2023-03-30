"""
RC re-implementation of the on-the-fly feature extraction
"""
from typing import Tuple, Union
from returnn_common import nn
from returnn.tf.util.data import Dim


class LogMelFeatureExtractor(nn.Module):
    """
    Returns both the absolute and the log-mel spectrogram.

    The default values are for 80-dimensional 10ms/25ms windowed features over 16 kHz [-1,1] ranged audio samples
    """

    def __init__(self, frame_shift=160, frame_size=400, fft_size=512, log_mel_size=80):
        """

        :param frame_shift:
        :param frame_size:
        :param fft_size:
        :param log_mel_size:
        """
        super().__init__()
        self.frame_shift = frame_shift
        self.frame_size = frame_size
        self.fft_size = fft_size
        self.log_mel_size = log_mel_size

        assert frame_size <= fft_size, "fft_size has to be at least equal to the frame size"

        self.out_feature_dim = nn.FeatureDim("extractor_mel_feature", log_mel_size)
        self.out_linear_feature_dim = nn.FeatureDim("extractor_linear_feature", fft_size // 2 + 1)

    def __call__(self, inp: nn.Tensor, audio_time: nn.Dim) -> Tuple[nn.Tensor, nn.Tensor, Dim]:
        stft, time_dims = nn.stft(inp, frame_shift=self.frame_shift, fft_size=self.fft_size, frame_size=self.frame_size,
                          in_spatial_dims=[audio_time], out_spatial_dims=None)
        time_dim = time_dims[0]
        # does not work because of Dim.same_as, see "short_repr"
        time_dim.name = "stft_time"
        abs = nn.abs(stft)
        power = abs ** 2
        mel_filterbank = nn.make_layer(
            layer_dict={
                "class": "mel_filterbank",
                "from": power,
                "fft_size": self.fft_size,
                "nr_of_filters": self.log_mel_size,
                "n_out": self.log_mel_size,
            },
            name="mel_filterbank"
        )
        mel_filterbank, feature_dim = nn.replace_dim(mel_filterbank, in_dim=mel_filterbank.feature_dim,
                                                     out_dim=self.out_feature_dim)

        log = nn.safe_log(mel_filterbank, eps=1e-10)
        log10 = log / 2.3026
        return abs, log10, time_dim
