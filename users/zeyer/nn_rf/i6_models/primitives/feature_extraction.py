"""
Wraps i6_models feature extraction
"""

from __future__ import annotations
from typing import Optional, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


class RasrCompatibleLogMelFeatureExtractionV1(rf.Module):
    def __init__(self, out_dim: Optional[Dim] = None, **config):
        super().__init__()

        from i6_models.primitives import feature_extraction as i6_mod

        self._config = i6_mod.RasrCompatibleLogMelFeatureExtractionV1Config(**config)
        self._torch_mod = i6_mod.RasrCompatibleLogMelFeatureExtractionV1(self._config)

        if not out_dim:
            out_dim = Dim(self._config.num_filters, name="feature")
        else:
            assert out_dim.dimension == self._config.num_filters
        self.out_dim = out_dim

    def __call__(self, raw_audio: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        if raw_audio.feature_dim and raw_audio.feature_dim.dimension == 1:
            raw_audio = rf.squeeze(raw_audio, axis=raw_audio.feature_dim)

        batch_dims = raw_audio.remaining_dims(in_spatial_dim)
        assert len(batch_dims) == 1  # not implemented yet otherwise...
        raw_audio = raw_audio.copy_transpose(batch_dims + [in_spatial_dim])
        out_raw, out_length_raw = self._torch_mod(
            raw_audio.raw_tensor, in_spatial_dim.get_size_tensor(device="cpu").copy_compatible_to_dims_raw(batch_dims)
        )
        out_spatial_dim = Dim(rf.convert_to_tensor(out_length_raw, dims=batch_dims), name="time")
        out = rf.convert_to_tensor(out_raw, dims=batch_dims + [out_spatial_dim, self.out_dim])
        return out, out_spatial_dim
