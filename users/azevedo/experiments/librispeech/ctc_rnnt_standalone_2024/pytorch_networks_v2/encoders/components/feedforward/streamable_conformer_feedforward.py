import torch
from typing import Callable, Optional, Literal
from dataclasses import dataclass

from i6_models.config import ModelConfiguration
from i6_models.parts.conformer import ConformerPositionwiseFeedForwardV2

from ....streamable_module import StreamableModule
from ....base_config import BaseConfig



@dataclass(kw_only=True)
class StreamableConformerPositionwiseFeedForwardConfig(BaseConfig):
    """
    New attribute:
        dropout_broadcast_axes: string of axes to which dropout is broadcast, e.g. "T" for broadcasting to the time axis
                                setting to None to disable broadcasting
    Default value for `activation` removed
    """

    input_dim: int
    hidden_dim: int
    dropout: float
    activation: Callable[[torch.Tensor], torch.Tensor]
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]

    def check_valid(self):
        assert self.dropout_broadcast_axes in [
            None,
            "B",
            "T",
            "BT",
        ], "invalid value, supported are None, 'B', 'T' and 'BT'"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        # TODO: change for more flexibility
        assert d["activation"].lower() == "silu"
        d["activation"] = torch.nn.functional.silu
        return StreamableConformerPositionwiseFeedForwardConfig(**d)


class StreamableConformerPositionwiseFeedForward(StreamableModule):
    def __init__(self, cfg: StreamableConformerPositionwiseFeedForwardConfig):
        super().__init__()

        self.ff = ConformerPositionwiseFeedForwardV2(cfg)

    def forward_offline(self, x):
        return self.ff(x)
    
    def forward_streaming(self, x):
        return self.ff(x)
    
    def infer(self, x):
        return self.ff(x)