from typing import Optional, Tuple

import returnn.frontend as rf
from returnn.tensor import Dim as ReturnnDim
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import single_step_dim

from ..encoder.attention_zoneout import _last_updated_frame


class CausalAttentionTimeZoneoutMixin:
    """
    Attention time zoneout for a causal self-attention class whose call gives the attention output and its state: in
    training, a frame takes the attention output of the frame before it with the keep probability, so a run of kept
    frames repeats the output of the last frame before the run.

    In evaluation the attention output is unchanged. The keeps are drawn over the dims of the source besides the
    features, so the first frame of a call never keeps, and neither does a single step, the only frame of its call.
    Mixed in before the attention class, which the calls pass their further arguments to.
    """

    def __init__(self, *args, zoneout_prob: float, **kwargs):
        """
        :param zoneout_prob: probability to keep a frame's previous output
        """
        super().__init__(*args, **kwargs)
        assert 0.0 <= zoneout_prob < 1.0, f"invalid zoneout_prob {zoneout_prob}"
        self.zoneout_prob = zoneout_prob

    def draw_keep(self, dims, *, device: Optional[str]) -> ReturnnTensor:
        """
        :param dims: the dims to draw over
        :return: bool, True with the keep probability
        """
        return rf.random_uniform(dims, device=device) < self.zoneout_prob

    def __call__(self, source: ReturnnTensor, axis: ReturnnDim, **kwargs) -> Tuple[ReturnnTensor, rf.State]:
        """
        :param source: the layer input
        :param axis: the spatial dim, ``single_step_dim`` for one step
        :return: the attention output and the state of the attention class
        """
        out, state = super().__call__(source, axis, **kwargs)
        if not self.zoneout_prob or axis == single_step_dim:
            return out, state
        train = rf.get_run_ctx().is_train_flag_enabled(func=rf.dropout)
        assert isinstance(train, bool), f"{self}: attention zoneout needs a static train flag, got {train!r}"
        if not train:
            return out, state
        keep = self.draw_keep(source.remaining_dims((axis, self.in_dim)) + [axis], device=source.device)
        return rf.gather(out, indices=_last_updated_frame(keep, axis=axis), axis=axis), state
