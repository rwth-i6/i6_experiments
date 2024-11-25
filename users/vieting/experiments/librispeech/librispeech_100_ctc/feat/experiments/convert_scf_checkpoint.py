import numpy as np

from sisyphus import tk
from i6_core.returnn.training import Checkpoint
from i6_core.returnn.conversion import ConvertTfCheckpointToPtJob


def make_model():
    from ..pytorch_networks.ctc.conformer_1023.feature_extraction import (
        SupervisedConvolutionalFeatureExtractionV1,
        SupervisedConvolutionalFeatureExtractionV1Config,
    )

    scf_config = SupervisedConvolutionalFeatureExtractionV1Config(
        wave_norm=True,
        num_tf=150,
        size_tf=160,
        stride_tf=10,
        num_env=5,
        size_env=40,
        stride_env=16
    )
    return SupervisedConvolutionalFeatureExtractionV1(scf_config)


def map_func(reader, name: str, var) -> np.ndarray:
    name_map = {
        "conv_tf.weight": "features/conv_h/W",
        "conv_env.weight": "features/conv_l/W",
        "normalization_env.weight": "features/conv_l_act/scale",
        "normalization_env.bias": "features/conv_l_act/bias",
        "wave_norm.weight": "features/wave_norm/scale",
        "wave_norm.bias": "features/wave_norm/bias",
    }
    if name not in name_map:
        print("Warning: {} not found in {}".format(name, name_map))
        return np.zeros(var.shape)
    value = reader.get_tensor(name_map[name])
    if "conv" in name:
        value = value.transpose(list(range(var.ndim))[::-1])
    return value


def get_scf_checkpoint():
    # From CTC training on LibriSpeech 960, WER 7.0% on dev-other with 4gram LM (result in ITG paper)
    ckpt = Checkpoint(
        tk.Path(
            "/u/vieting/setups/librispeech/20220824_features/alias/00_mono-eow_ctc_features_v3_paper/"
            "train_ns-monophone-eow_ctc_conf_scf750_specaug-f15_win1sz-160/output/models/epoch.490.index",
            hash_overwrite="scf_ctc_conf_itg_e490",
        )
    )
    job = ConvertTfCheckpointToPtJob(
        checkpoint=ckpt,
        make_model_func=make_model,
        map_func=map_func,
    )
    return job.out_checkpoint
