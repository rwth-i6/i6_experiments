from typing import Optional

from returnn.tensor import Dim

from .model import Wav2VecModel

def w2v_model_def(*, epoch: int, in_dim: Dim, target_dim: Optional[Dim] = None) -> Wav2VecModel:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config
    config = get_global_config()
    assert config
    
    w2v_opts = config.typed_value("w2v_opts", {})
    return Wav2VecModel(
        w2v_opts=w2v_opts,
        wb_target_dim=-1,
    )
