"""
run:
export PYTHONPATH=recipe:ext/returnn
python -m i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2_test

Real-world examples:
i6_experiments/users/zeyer/experiments/exp2025_10_21_chunked_ctc.py
"""

from typing import Dict, Any, Tuple

from returnn.util import BehaviorVersion, better_exchook
import returnn.frontend as rf
from returnn.tensor import Dim, Tensor, batch_dim
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)
from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v1 import (
    ChunkedConformerEncoder,
    ChunkedConformerEncoderLayer,
)
from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
    ChunkedConformerEncoderV2,
    ChunkedConformerEncoderLayerV2,
)
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep

_log_mel_feature_dim = 80
feat_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)


def _setup_test():
    BehaviorVersion.set_min_behavior_version(25)
    rf.select_backend_torch()
    if batch_dim.dyn_size_ext is None:
        batch_dim.dyn_size_ext = rf.convert_to_tensor(3, dims=[])


def tests():
    better_exchook.install()
    _setup_test()
    test_conformer_v2()


def test_conformer_v2():
    downsampling = 6
    left_n, center_size, right_size = (16, 5, 4)

    build_dict = rf.build_dict(
        ChunkedConformerEncoder,
        encoder_layer=rf.build_dict(ChunkedConformerEncoderLayer),
        chunk_stride=center_size * downsampling,
        chunk_history=left_n,
        input_chunk_size_dim=(center_size + right_size) * downsampling,
        end_chunk_size_dim=center_size,
    )
    model = _build_model(build_dict)

    # f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2"
    # model.enc_build_dict
    build_dict_v2 = rf.build_dict(
        ChunkedConformerEncoderV2,
        encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
        chunk_stride=center_size * downsampling,
        chunk_history=left_n,
        input_chunk_size_dim=(center_size + right_size) * downsampling,
        end_chunk_size_dim=center_size,
    )
    model_v2 = _build_model(build_dict_v2)

    num_params = sum(p.num_elements() for p in model.parameters())
    num_params_v2 = sum(p.num_elements() for p in model_v2.parameters())
    print(f"num_params: {num_params:.2e} vs {num_params_v2:.2e}")
    assert num_params_v2 == num_params

    input_data, time_dim = _make_input_data()
    res, out_spatial_dim = model(input_data, in_spatial_dim=time_dim)
    res_v2, out_spatial_dim_v2 = model_v2(input_data, in_spatial_dim=time_dim)


def _build_model(build_dict: Dict[str, Any]):
    base_build_dict = rf.build_dict(
        ConformerEncoder,
        input_layer=rf.build_dict(
            ConformerConvSubsample,
            out_dims=[32, 64, 64],
            filter_sizes=[(3, 3), (3, 3), (3, 3)],
            pool_sizes=[(1, 2)],
            strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
        ),
        # original:
        # num_layers=16,
        # out_dim=1024,
        num_layers=2,
        out_dim=64,
        encoder_layer=rf.build_dict(
            ConformerEncoderLayer,
            ff=rf.build_dict(
                ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
            ),
            num_heads=8,
        ),
    )
    build_dict = dict_update_deep(base_build_dict, build_dict)

    # rf.audio.log_mel_filterbank_from_raw(..., sampling_rate=16_000, out_dim=feat_dim) but we skip that here

    encoder = rf.build_from_dict(build_dict, feat_dim)
    encoder: ConformerEncoder  # might not be true, but assume similar/same interface
    return encoder


def _make_input_data() -> Tuple[Tensor, Dim]:
    time_dim = Dim(rf.convert_to_tensor([16_000 - i for i in range(batch_dim.get_dim_value())], dims=[batch_dim]))
    return rf.random_normal([batch_dim, time_dim, feat_dim]), time_dim


if __name__ == "__main__":
    tests()
