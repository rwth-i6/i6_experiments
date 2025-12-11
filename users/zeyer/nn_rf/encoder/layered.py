"""
Some layered structure
"""

from typing import Any, Optional, Sequence, Tuple, Dict
from returnn.config import get_global_config
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder, ISeqFramewiseEncoder, IEncoder


class Layered(ISeqDownsamplingEncoder):
    """
    Any layered structure composed of encoders.
    """

    def __init__(self, layers: Sequence[Dict[str, Any]]):
        super().__init__()
        self.layers = rf.ModuleList(rf.build_from_dict(layer) for layer in layers)

        self.downsample_factor = 1
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ISeqDownsamplingEncoder):
                self.downsample_factor *= layer.downsample_factor
            assert isinstance(layer, (ISeqDownsamplingEncoder, ISeqFramewiseEncoder, IEncoder)), (
                f"{self}: Layer {i} has unsupported type {type(layer)}."
            )
        self.out_dim = self.layers[-1].out_dim

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
        targets: Optional[Tensor] = None,
        targets_spatial_dim: Optional[Dim] = None,
    ) -> Tuple[Tensor, Dim]:
        tensor = source
        spatial_dim = in_spatial_dim
        for name, layer in self.layers.items():
            if isinstance(ILayerUsingTargets, layer):
                tensor = layer(
                    tensor, spatial_dim=spatial_dim, targets=targets, targets_spatial_dim=targets_spatial_dim
                )
            elif isinstance(layer, ISeqDownsamplingEncoder):
                tensor, spatial_dim = layer(tensor, in_spatial_dim=spatial_dim)
            elif isinstance(layer, ISeqFramewiseEncoder):
                tensor = layer(tensor, spatial_dim=spatial_dim)
            elif isinstance(layer, IEncoder):
                tensor = layer(tensor)
            else:
                raise TypeError(f"Unsupported layer type: {type(layer)}")
            if collected_outputs is not None:
                collected_outputs[name] = tensor
        return tensor, spatial_dim


class ILayerUsingTargets(ISeqFramewiseEncoder):
    """
    E.g. can be used to implement a seq-level loss,
    or also downsampling which could utilize the targets if available,
    or so.
    """

    def __call__(
        self,
        source: Tensor,
        *,
        spatial_dim: Dim,
        targets: Optional[Tensor] = None,
        targets_spatial_dim: Optional[Dim] = None,
    ) -> Tensor:
        raise NotImplementedError


class CtcLoss(ILayerUsingTargets):
    """CTC"""

    def __init__(
        self,
        *,
        loss_name: str,
        loss_scale: float = 1.0,
        use_normalized_loss: bool = None,
        in_dim: Dim,
        wb_target_dim: Dim,
        blank_index: int,
        with_bias: bool = False,
    ):
        super().__init__()

        config = get_global_config(return_empty_if_none=True)

        if use_normalized_loss is None:
            # Note: The default here is different from :func:`mark_as_loss`.
            # However, this is the default as in most of my setups for this config option.
            use_normalized_loss = config.typed_value("use_normalized_loss", True)

        self.loss_name = loss_name
        self.loss_scale = loss_scale
        self.use_normalized_loss = use_normalized_loss
        self.in_dim = in_dim
        self.wb_target_dim = wb_target_dim
        self.blank_index = blank_index
        self.linear = rf.Linear(in_dim, wb_target_dim, with_bias=with_bias)

    def __call__(
        self,
        source: Tensor,
        *,
        spatial_dim: Dim,
        targets: Optional[Tensor] = None,
        targets_spatial_dim: Optional[Dim] = None,
    ) -> Tensor:
        if targets is None:
            return source

        logits = self.linear(source)

        loss = rf.ctc_loss(
            logits=logits,
            logits_normalized=False,
            input_spatial_dim=spatial_dim,
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
            blank_index=self.blank_index,
        )
        loss.mark_as_loss(
            self.loss_name,
            scale=self.loss_scale,
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
            use_normalized_loss=self.use_normalized_loss,
        )

        decoded, decoded_spatial_dim = rf.ctc_greedy_decode(
            logits,
            in_spatial_dim=spatial_dim,
            blank_index=self.blank_index,
            wb_target_dim=self.wb_target_dim,
            target_dim=targets.sparse_dim,
        )
        error = rf.edit_distance(
            a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
        )
        error.mark_as_loss(
            f"{self.loss_name}_err",
            as_error=True,
            custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
        )

        return source
