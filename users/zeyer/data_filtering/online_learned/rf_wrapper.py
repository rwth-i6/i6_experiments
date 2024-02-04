"""
Wrapper for RETURNN frontend, to easily plug it into a rf.Sequential.
"""

from typing import Optional, Tuple, Dict
import torch
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.torch.frontend.bridge import PTModuleAsRFModule
from .filter_base import LearnedDataFilterBase as _LearnedDataFilterBasePT


class LearnedDataFilter(PTModuleAsRFModule):
    """
    Data filter
    """

    pt_module: _LearnedDataFilterBasePT

    def __init__(self, pt_module: _LearnedDataFilterBasePT):
        super().__init__(pt_module)
        self._recent_batch_dim: Optional[Tuple[Dim, Dim]] = None
        self._recent_spatial_dim: Optional[Tuple[Dim, Dim]] = None

    def __call__(self, x: Tensor, *, spatial_dim: Dim) -> Tuple[Tensor, Dim, Dim]:
        """
        :param x: [batch_dim, spatial_dim, feature_dim]
        :return: filtered x, new spatial dim, new batch dim
        """
        assert x.feature_dim.dimension == self.pt_module.in_features
        assert spatial_dim in x.dims
        batch_dims = list(set(x.dims).difference({x.feature_dim, spatial_dim}))
        assert len(batch_dims) == 1  # not implemented otherwise...
        (batch_dim,) = batch_dims
        batch_dim: Dim
        assert spatial_dim.dyn_size_ext.dims == (batch_dim,)  # not implemented otherwise
        btd_axes = (x.dims.index(batch_dims[0]), x.dims.index(spatial_dim), x.dims.index(x.feature_dim))
        seq_lens_raw = spatial_dim.dyn_size_ext.raw_tensor
        new_x_raw, new_seq_lens_raw = self.pt_module(x.raw_tensor, seq_lens=seq_lens_raw, btd_axes=btd_axes)
        new_batch_dim = Dim(
            Tensor(
                batch_dim.name + "_filtered",
                (),
                dtype="int32",
                raw_tensor=torch.tensor(new_x_raw.shape[btd_axes[0]], dtype=torch.int32),
            )
        )
        new_spatial_dim = Dim(
            Tensor(
                spatial_dim.name + "_filtered",
                [new_batch_dim],
                dtype="int32",
                raw_tensor=new_seq_lens_raw.to(torch.int32),
            )
        )
        new_x_dims = list(x.dims)
        new_x_dims[btd_axes[0]] = new_batch_dim
        new_x_dims[btd_axes[1]] = new_spatial_dim
        new_x = Tensor(
            x.name + "_filtered",
            new_x_dims,
            dtype=x.dtype,
            feature_dim=x.feature_dim,
            sparse_dim=x.sparse_dim,
            raw_tensor=new_x_raw,
        )
        self._recent_spatial_dim = (spatial_dim, new_spatial_dim)
        self._recent_batch_dim = (batch_dim, new_batch_dim)
        return new_x, new_spatial_dim, new_batch_dim

    def filter_batch(self, x: Tensor, *, dim_map: Optional[Dict[Dim, Dim]] = None) -> Tuple[Tensor, Dict[Dim, Dim]]:
        """filter"""
        dim_map: Dict[Dim, Dim] = dict(dim_map) if dim_map else {}
        batch_dim = self._recent_batch_dim[0]
        dim_map[batch_dim] = self._recent_batch_dim[1]
        batch_axis = x.dims.index(self._recent_batch_dim[0])
        time_axis = None
        if self._recent_spatial_dim[0] in x.dims:
            time_axis = x.dims.index(self._recent_spatial_dim[0])
            dim_map[self._recent_spatial_dim[0]] = self._recent_spatial_dim[1]
        new_x_dims = [dim_map.get(dim, dim) for dim in x.dims]
        new_x_raw = self.pt_module.filter_batch(x.raw_tensor, batch_axis=batch_axis, time_axis=time_axis)
        for axis, dim in enumerate(x.dims):
            dim: Dim
            if (
                axis not in (batch_axis, time_axis)
                and dim.dyn_size_ext is not None
                and batch_dim in dim.dyn_size_ext.dims
            ):
                if dim in dim_map:
                    new_dim = dim_map[dim]
                else:
                    new_seq_lens, _ = self.filter_batch(dim.dyn_size_ext)
                    new_dim = Dim(new_seq_lens)
                    dim_map[dim] = new_dim
                new_x_dims[axis] = new_dim
                new_x_raw = new_x_raw[(slice(None),) * axis + (slice(None, new_dim.get_dim_value()),)]
        new_x = Tensor(
            x.name + "_filtered",
            new_x_dims,
            dtype=x.dtype,
            feature_dim=x.feature_dim,
            sparse_dim=x.sparse_dim,
            raw_tensor=new_x_raw,
        )
        return new_x, dim_map

    def score_estimator_loss(self, model_loss: Optional[Tensor] = None) -> Tensor:
        """
        score estimator loss

        :param model_loss: [B'] or [B',T_] or None
        :return: [B'] or []
        """
        model_loss_raw = None
        if model_loss is not None:
            if model_loss.dims_set == {self._recent_batch_dim[1]}:
                model_loss_raw = model_loss.raw_tensor
            elif model_loss.dims_set == {self._recent_batch_dim[1], self._recent_spatial_dim[1]}:
                model_loss_raw = model_loss.copy_compatible_to_dims_raw(
                    (self._recent_batch_dim[1], self._recent_spatial_dim[1])
                )
            else:
                raise ValueError(f"unexpected model loss {model_loss} shape")
        if model_loss_raw is not None:
            loss_raw = self.pt_module.score_estimator_loss(model_loss_raw)
            return rf.convert_to_tensor(loss_raw, dims=[self._recent_batch_dim[1]], name="score_estimator_loss")
        else:
            loss_raw = self.pt_module.score_estimator_loss()
            return rf.convert_to_tensor(loss_raw, dims=[], name="score_estimator_loss")
