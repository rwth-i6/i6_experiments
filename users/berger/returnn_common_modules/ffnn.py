from returnn_common import nn
from typing import Optional, Tuple


class LinearLayer(nn.Module):
    def __init__(
        self, layer_size: int, dropout: Optional[float] = None, with_bias: bool = True
    ):
        super().__init__()
        self.out_dim = nn.FeatureDim("linear_out_dim", dimension=layer_size)
        self.ff = nn.Linear(self.out_dim, with_bias=with_bias)
        self.dropout = dropout

    def __call__(self, input_data: nn.Tensor) -> Tuple[nn.Tensor, nn.Dim]:
        out, out_dim = self.ff(input_data)
        out = nn.dropout(out, self.dropout)
        return out, out_dim


class FFNetwork(nn.Module):
    def __init__(self, num_layers: int, layer_size: int, **kwargs):
        super().__init__()
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(LinearLayer(layer_size, **kwargs))

    def __call__(self, input_data: nn.Tensor) -> Tuple[nn.Tensor, nn.Dim]:
        out = input_data
        for layer in self.layers:
            out, out_dim = layer(out)
        return out, out_dim
