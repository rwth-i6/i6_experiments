"""
run:
export PYTHONPATH=recipe:ext/returnn
python -m i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2_test

Real-world examples:
i6_experiments/users/zeyer/experiments/exp2025_10_21_chunked_ctc.py

Similar tests, see: RETURNN tests.test_rf_encoder_conformer.test_e_branchformer,
or other RF tests in RETURNN.
"""

from typing import Dict, Any, Tuple

import torch

from returnn.util import BehaviorVersion, better_exchook
from returnn.util.debug import PyTracer, check_py_traces_rf_to_pt_equal
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
    print(f"num_params: {num_params} vs {num_params_v2}")
    assert num_params_v2 == num_params
    params_by_name = {name: p for name, p in model.named_parameters()}
    params_by_name_v2 = {name: p for name, p in model_v2.named_parameters()}
    assert set(params_by_name_v2.keys()) == set(params_by_name.keys())
    for name, p in params_by_name_v2.items():
        p0 = params_by_name[name]
        assert p.shape == p0.shape, f"{name} {p.shape} vs {p0.shape}"
        assert p.dtype == p0.dtype, f"{name} {p.dtype} vs {p0.dtype}"
        with torch.no_grad():
            p.raw_tensor.copy_(p0.raw_tensor)

    input_data, time_dim = _make_input_data()

    with (
        PyTracer(
            [
                ConformerEncoder.__call__,
                ChunkedConformerEncoderLayer.__call__,
                ChunkedConformerEncoderLayerV2.__call__,
            ],
            Tensor,
        ) as trace_v1,
        torch.no_grad(),
    ):
        res, out_spatial_dim = model(input_data, in_spatial_dim=time_dim)

    with (
        PyTracer(
            [
                ConformerEncoder.__call__,
                ChunkedConformerEncoderLayer.__call__,
                ChunkedConformerEncoderLayerV2.__call__,
            ],
            Tensor,
        ) as trace_v2,
        torch.no_grad(),
    ):
        res_v2, out_spatial_dim_v2 = model_v2(input_data, in_spatial_dim=time_dim)

    print(f"out: {res} vs {res_v2}")
    # Final check.
    # Actually the check_py_traces_rf_to_pt_equal should already have covered also this final output,
    # but anyway do it again now to be sure.
    assert res.dims == (batch_dim, out_spatial_dim, model.out_dim)
    assert res_v2.dims == (batch_dim, out_spatial_dim_v2, model_v2.out_dim)
    assert res.raw_tensor.shape == res_v2.raw_tensor.shape  # [B,T,D]
    assert (
        res.raw_tensor.shape[:1] == out_spatial_dim.dyn_size_ext.raw_tensor.shape
    )  # [B]
    for b in range(res.raw_tensor.shape[0]):
        seq_len = out_spatial_dim.dyn_size_ext.raw_tensor[b]
        torch.testing.assert_allclose(
            res.raw_tensor[b, :seq_len],
            res_v2.raw_tensor[b, :seq_len],
            rtol=1e-5,
            atol=1e-5,
        )
    # Check that there is sth non-zero.
    assert out_spatial_dim.dyn_size_ext.raw_tensor.max() > 0
    assert torch.mean(res.raw_tensor**2) > 0.1
    print("All matching!")


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
                ConformerPositionwiseFeedForward,
                activation=rf.build_dict(rf.relu_square),
                with_bias=False,
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
    time_dim = Dim(
        rf.convert_to_tensor(
            [201 - i * 11 for i in range(batch_dim.get_dim_value())],
            dims=[batch_dim],
            name="time",
        )
    )
    return rf.random_normal([batch_dim, time_dim, feat_dim]), time_dim


if __name__ == "__main__":
    tests()
