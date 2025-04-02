"""
Soft collapse repeated
"""

from typing import Tuple
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim


def soft_collapse_repeated(
    log_probs: Tensor,
    *,
    spatial_dim: Dim,
    classes_dim: Dim,
    threshold: float,
    reduce_type: str = "logmeanexp",
) -> Tuple[Tensor, Dim]:
    """
    :param log_probs: shape {OtherDims..., Spatial, Classes}
    :param spatial_dim:
    :param classes_dim:
    :param threshold:
    :param reduce_type: "logmeanexp" or "max_renorm"
    :return: shape {OtherDims..., OutSpatial, Classes}, out_spatial_dim
    """
    argmax_classes = rf.reduce_argmax(log_probs, axis=classes_dim)  # {OtherDims..., Spatial} -> Classes
    log_probs_classes = rf.gather(log_probs, indices=argmax_classes)  # {OtherDims..., Spatial}
    probs_classes = rf.exp(log_probs_classes)
    mask_threshold = probs_classes >= threshold  # {OtherDims..., Spatial}
    argmax_classes_shifted = rf.shift_right(argmax_classes, axis=spatial_dim, pad_value=-1)  # {OtherDims..., Spatial}
    mask_threshold_shifted = rf.shift_right(
        mask_threshold, axis=spatial_dim, pad_value=False
    )  # {OtherDims..., Spatial}
    # Always take the first one in mask_repeated (when going left to right).
    mask_repeated = argmax_classes_shifted == argmax_classes  # {OtherDims..., Spatial}
    # We could also mask the first frame just to be sure, but the cases where this would go wrong are very rare.
    mask = mask_repeated & mask_threshold & mask_threshold_shifted  # {OtherDims..., Spatial}
    keep_mask = rf.logical_not(mask)  # {OtherDims..., Spatial}
    # To be sure.
    keep_mask = keep_mask.copy_masked(mask_value=False)  # {OtherDims..., Spatial}
    # Very similar to the internal masked_select code.
    idxs = rf.cumsum(rf.cast(keep_mask, "int32"), spatial_dim=spatial_dim)  # {OtherDims..., Spatial} -> 1+OutSpatial
    new_size = rf.gather(idxs, indices=spatial_dim.get_dim_value_tensor() - 1, axis=spatial_dim)  # {OtherDims...}
    out_spatial_dim = Dim(new_size, name="soft_collapse_repeated")
    idxs = idxs - 1  # {OtherDims..., Spatial} -> OutSpatial
    idxs.sparse_dim = out_spatial_dim
    if reduce_type == "logmeanexp":
        res = rf.scatter_logmeanexp(log_probs, indices=idxs, indices_dim=spatial_dim, out_dim=out_spatial_dim)
    elif reduce_type == "max_renorm":
        res = rf.scatter(log_probs, mode="max", indices=idxs, indices_dim=spatial_dim, out_dim=out_spatial_dim)
        res = rf.log_softmax(res, axis=classes_dim)
    else:
        raise ValueError(f"invalid reduce_type {reduce_type!r}")
    return res, out_spatial_dim


def test_soft_collapse_repeated():
    import numpy as np

    rf.select_backend_torch()

    time_dim = Dim(7, name="time")
    vocab_dim = Dim(4, name="vocab")
    demo_probs1 = rf.convert_to_tensor(
        np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.9, 0.05, 0.01, 0.04],
                [0.9, 0.01, 0.03, 0.06],
                [0.01, 0.03, 0.06, 0.9],
                [0.05, 0.03, 0.02, 0.9],
                [0.03, 0.03, 0.04, 0.9],
            ],
            dtype="float32",
        ),
        dims=[time_dim, vocab_dim],
    )
    log_probs_out1, out_dim = soft_collapse_repeated(
        rf.log(demo_probs1), spatial_dim=time_dim, classes_dim=vocab_dim, threshold=0.8
    )
    probs_out1 = rf.exp(log_probs_out1)
    print(f"probs_out1: {probs_out1} {probs_out1.raw_tensor.numpy()}")
    assert out_dim.dyn_size_ext.raw_tensor == 4
    assert np.allclose(
        probs_out1.raw_tensor.numpy(),
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.03, 0.02, 0.05],
            [0.03, 0.03, 0.04, 0.9],
        ],
    )
