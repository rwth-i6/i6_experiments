from __future__ import annotations

from returnn.tensor import Dim
import returnn.frontend as rf

class DummyModel(rf.Module):
    def __init__(self,
                 in_dim: Dim,
                 target_dim: Dim):
        self.in_dim = in_dim,
        self.target_dim = target_dim

def dummy_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim):
    dummy_model = DummyModel(in_dim=in_dim, target_dim=target_dim)
    return dummy_model
