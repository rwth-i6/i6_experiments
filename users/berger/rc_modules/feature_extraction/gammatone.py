from typing import Tuple
from returnn_common.asr.gt import GammatoneV2
from returnn_common import nn


def make_gammatone_module(*args, sample_rate: int = 8_000, **kwargs) -> Tuple[GammatoneV2, nn.Dim]:
    assert sample_rate in {8_000, 16_000}
    default_kwargs = {
        8000: {
            "gt_filterbank_size": 320,
            "temporal_integration_size": 200,
            "temporal_integration_strides": 80,
            "freq_max": 3_800,
        },
        16000: {
            "gt_filterbank_size": 640,
            "temporal_integration_size": 400,
            "temporal_integration_strides": 160,
            "freq_max": 7_500,
        },
    }[sample_rate]

    default_kwargs.update(kwargs)

    # if sample_rate == 8_000:
    #     out_dim = nn.FeatureDim("channels", 40)
    # else:
    #     out_dim = nn.FeatureDim("channels", 50)

    module = GammatoneV2(*args, sample_rate=sample_rate, **default_kwargs)
    for p in module.parameters():
        p.trainable = False

    return module, module.out_dim
