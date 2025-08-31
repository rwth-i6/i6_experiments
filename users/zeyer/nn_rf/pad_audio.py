"""
Pad audio (raw samples, maybe also features) left/right, for augmentation or modeling,
by random lengths or fixed lengths, with zeros or random noise.
"""

from typing import Any, Dict, Tuple, Union, Optional, Sequence, List
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


PadSideT = Union[int, Tuple[int, int]]


def pad_audio(
    audio: Tensor,
    *,
    in_spatial_dim: Dim,
    feature_dim: Union[None, Dim, Sequence[Dim]] = None,
    opts: Union[None, int, Tuple[PadSideT, PadSideT], Dict[str, Any]],
) -> Tuple[Tensor, Dim]:
    if isinstance(opts, dict) and "train" in opts:
        assert set(opts.keys()).issubset({"train", "eval"})
        return rf.cond(
            rf.get_run_ctx().is_train_flag_enabled(func=pad_audio),
            lambda: pad_audio(audio, in_spatial_dim=in_spatial_dim, feature_dim=feature_dim, opts=opts["train"]),
            lambda: pad_audio(audio, in_spatial_dim=in_spatial_dim, feature_dim=feature_dim, opts=opts.get("eval")),
        )
    if opts is None:
        return audio, in_spatial_dim
    elif isinstance(opts, int):
        opts = {"padding": (opts, opts)}
    elif isinstance(opts, tuple):
        opts = {"padding": opts}
    elif isinstance(opts, dict):
        assert "padding" in opts
    else:
        raise TypeError(f"invalid padding: {opts} type {type(opts)}")
    return _pad_audio(audio, in_spatial_dim=in_spatial_dim, feature_dim=feature_dim, **opts)


def _pad_audio(
    audio: Tensor,
    *,
    in_spatial_dim: Dim,
    feature_dim: Dim,
    padding: Tuple[PadSideT, PadSideT],
    mode: str = "constant",
    value: Optional[float] = None,
) -> Tuple[Tensor, Dim]:
    assert isinstance(padding, tuple) and len(padding) == 2
    if mode == "constant" and value is None:
        value = 0.0
    if all(isinstance(p, int) for p in padding):
        audio, (in_spatial_dim,) = rf.pad(
            audio, axes=[in_spatial_dim], padding=[padding], handle_dynamic_dims=True, mode=mode, value=value
        )
        return audio, in_spatial_dim
    if feature_dim is None:
        feature_dim = audio.feature_dim or ()
    batch_dims = audio.remaining_dims(
        [in_spatial_dim] + (list(feature_dim) if isinstance(feature_dim, (list, tuple)) else [feature_dim])
    )
    pad_amount_lr = [
        Dim(_pad_amount(amount, batch_dims=batch_dims, device=audio.device), name=f"pad_{name}")
        for amount, name in zip(padding, ["left", "right"])
    ]
    assert mode == "constant"  # not really implemented otherwise yet
    lr = [
        rf.full(dims=batch_dims + [pad_amount], fill_value=value, dtype=audio.dtype, device=audio.device)
        for pad_amount in pad_amount_lr
    ]
    return rf.concat(
        (lr[0], pad_amount_lr[0]),
        (audio, in_spatial_dim),
        (lr[1], pad_amount_lr[1]),
        allow_broadcast=True,
        handle_dynamic_dims=True,
    )


def _pad_amount(n: Union[int, Tuple[int, int]], *, batch_dims: List[Dim], device: Optional[str]) -> Union[int, Tensor]:
    if isinstance(n, int):
        return n
    if isinstance(n, tuple) and len(n) == 2 and all(isinstance(i, int) for i in n):
        return rf.random_uniform(batch_dims, minval=n[0], maxval=n[1] + 1, dtype="int32", device=device)
    raise TypeError(f"invalid pad amount {n} of type {type(n)}")
