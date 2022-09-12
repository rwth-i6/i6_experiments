from returnn_common import nn
from typing import Optional


class SoftmaxOutput(nn.Module):
    def __init__(self, out_dim: nn.Dim):
        super().__init__()
        self.out_dim = out_dim
        self.out_projection = nn.Linear(out_dim)

    def __call__(self, input_data: nn.Tensor) -> nn.Tensor:
        out_embed = self.out_projection(input_data)
        out = nn.softmax(out_embed, axis=self.out_dim)


class SoftmaxCEOutput(nn.Module):
    def __init__(self, targets: Optional[nn.Tensor], out_dim: Optional[nn.Dim] = None):
        super().__init__()
        self.targets = targets
        self.out_dim = out_dim
        if out_dim is not None:
            assert out_dim == targets.feature_dim
        self.out_projection = nn.Linear(out_dim)

    def __call__(self, input_data: nn.Tensor, train: bool = True) -> nn.Tensor:
        out_embed = self.out_projection(input_data)
        if train:
            ce_loss = nn.sparse_softmax_cross_entropy_with_logits(
                logits=out_embed, targets=self.targets, axis=self.out_dim
            )
            ce_loss.mark_as_loss()
        return nn.softmax(out_embed, axis=self.out_dim)
