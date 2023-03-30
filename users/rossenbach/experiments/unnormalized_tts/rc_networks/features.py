"""
RC re-implementation of the on-the-fly feature extraction
"""
from typing import Tuple, Union
from returnn_common import nn

class FeatureExtractor(nn.Module):
    """
    Returns both the absolute and the log-mel spectrogram
    """
    def __init__(self, fft_size=512, log_mel_size=80):
        super().__init__()
        self.fft_size = fft_size
        self.log_mel_size = log_mel_size
        self.out_feature_dim = nn.FeatureDim("extractor_mel_feature", log_mel_size)
        self.out_linear_feature_dim = nn.FeatureDim("extractor_linear_feature", fft_size//2 + 1)

    def __call__(self, inp: nn.Tensor):
        stft = nn.make_layer(
            layer_dict={
                "class": "stft",
                "frame_shift": 160,
                "frame_size": 400,
                "fft_size": self.fft_size,
                "from": inp,
            },
            name="stft"
        )
        abs = nn.abs(stft)
        power = abs ** 2
        time_dim = None
        for dim in stft.shape:
            print(dim)
            # workaround because of the custom layer
            if dim != stft.feature_dim and dim != stft.batch_dim:
                time_dim = dim
        mel_filterbank = nn.make_layer(
            layer_dict={
                "class": "mel_filterbank",
                "from": power,
                "fft_size": 512,
                "nr_of_filters": self.log_mel_size,
                "n_out": self.log_mel_size,
            },
            name="mel_filterbank"
        )
        mel_filterbank, feature_dim = nn.reinterpret_new_dim(mel_filterbank, in_dim=mel_filterbank.feature_dim, out_dim=self.out_feature_dim)

        log = nn.safe_log(mel_filterbank, eps=1e-10)
        log10 = log / 2.3026
        return abs, log10, time_dim
