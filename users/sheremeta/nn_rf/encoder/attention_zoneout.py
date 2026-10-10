from typing import Optional, Tuple

import returnn.frontend as rf
from returnn.tensor import Dim as ReturnnDim
from returnn.tensor import Tensor as ReturnnTensor

ZONEOUT_MODES = ("head", "row", "time")


class AttentionZoneoutRelPosSelfAttention(rf.RelPosSelfAttention):
    """
    Relative-position self-attention with attention zoneout: in training, parts of the attention keep their state
    of the layer below or of the frame before instead of being updated.

    - head: a head of a sequence attends with the queries and keys of the layer below
    - row: a frame takes the attention output computed with the queries and keys of the layer below
    - time: a frame takes the attention output of the frame before it

    In evaluation, head and row use the expected output, the mix of both attention outputs by the keep probability,
    and time uses the attention output unchanged.
    """

    def __init__(self, *args, zoneout_mode: str, zoneout_prob: float, **kwargs):
        """
        :param zoneout_mode: head, row or time
        :param zoneout_prob: probability to keep a head, a frame or a frame's previous output
        """
        super().__init__(*args, **kwargs)
        assert zoneout_mode in ZONEOUT_MODES, f"unknown zoneout_mode {zoneout_mode!r}"
        assert 0.0 <= zoneout_prob < 1.0, f"invalid zoneout_prob {zoneout_prob}"
        self.zoneout_mode = zoneout_mode
        self.zoneout_prob = zoneout_prob
        # set by AttentionZoneoutSequential around each call: the queries and keys of the layer below, and own ones
        self.qk_below: Optional[Tuple[ReturnnTensor, ReturnnTensor]] = None
        self.qk_out: Optional[Tuple[ReturnnTensor, ReturnnTensor]] = None
        self._qkv_given: Optional[Tuple[ReturnnTensor, ReturnnTensor, ReturnnTensor]] = None

    def draw_keep(self, dims, *, device: Optional[str]) -> ReturnnTensor:
        """
        :param dims: the dims to draw over
        :return: bool, True with the keep probability
        """
        return rf.random_uniform(dims, device=device) < self.zoneout_prob

    def forward_qkv(self, source: ReturnnTensor) -> Tuple[ReturnnTensor, ReturnnTensor, ReturnnTensor]:
        """
        :return: q, k, v, the given ones while :func:`_attend` runs
        """
        if self._qkv_given is not None:
            return self._qkv_given
        return super().forward_qkv(source)

    def _attend(self, source: ReturnnTensor, q: ReturnnTensor, k: ReturnnTensor, v: ReturnnTensor, *, axis: ReturnnDim):
        """
        :return: the attention output for the given queries, keys and values
        """
        self._qkv_given = (q, k, v)
        try:
            return super().__call__(source, axis=axis)
        finally:
            self._qkv_given = None

    def __call__(self, source: ReturnnTensor, *, axis: ReturnnDim, **_kwargs) -> ReturnnTensor:
        """forward"""
        q, k, v = super().forward_qkv(source)
        self.qk_out = (q, k)
        p = self.zoneout_prob
        if not p:
            return self._attend(source, q, k, v, axis=axis)
        train = rf.get_run_ctx().is_train_flag_enabled(func=rf.dropout)
        assert isinstance(train, bool), f"{self}: attention zoneout needs a static train flag, got {train!r}"
        batch_dims = source.remaining_dims((axis, self.in_dim))

        if self.zoneout_mode == "time":
            out = self._attend(source, q, k, v, axis=axis)
            if not train:
                return out
            keep = self.draw_keep(batch_dims + [axis], device=source.device)
            return rf.gather(out, indices=_last_updated_frame(keep, axis=axis), axis=axis)

        if self.qk_below is None:
            return self._attend(source, q, k, v, axis=axis)
        q_below, k_below = self.qk_below

        if self.zoneout_mode == "head" and train:
            keep = self.draw_keep(batch_dims + [self.num_heads], device=source.device)
            q = rf.where(keep, q_below, q)
            k = rf.where(keep, k_below, k)
            self.qk_out = (q, k)
            return self._attend(source, q, k, v, axis=axis)

        out = self._attend(source, q, k, v, axis=axis)
        out_below = self._attend(source, q_below, k_below, v, axis=axis)
        if not train:
            return p * out_below + (1.0 - p) * out
        keep = self.draw_keep(batch_dims + [axis], device=source.device)
        return rf.where(keep, out_below, out)


def _last_updated_frame(keep: ReturnnTensor, *, axis: ReturnnDim) -> ReturnnTensor:
    """
    :param keep: bool {..., axis}, the frames which keep the output of the frame before
    :return: int {..., axis}, for every frame the last frame up to it which is not kept, the first frame never is
    """
    frames = rf.range_over_dim(axis, device=keep.device)
    src = rf.where(rf.logical_and(keep, frames > 0), frames - 1, frames)
    # pointer jumping: each step doubles the run of kept frames which is followed
    if rf.is_static_traceable():
        max_len = axis.capacity
        assert max_len is not None, f"attention zoneout over {axis} needs its capacity under static tracing"
    else:
        max_len = int(axis.get_dim_value())
    for _ in range(max(max_len - 1, 1).bit_length()):
        src = rf.gather(src, indices=src, axis=axis)
    return src


class AttentionZoneoutSequential(rf.Sequential):
    """
    Runs the encoder layers in order and hands each layer's attention the queries and keys of the layer below.
    """

    def __call__(self, inp, *, collected_outputs=None, **kwargs) -> ReturnnTensor:
        """forward"""
        qk_below = None
        for name, module in self.items():
            att = module.self_att
            assert isinstance(att, AttentionZoneoutRelPosSelfAttention), f"{self}: unexpected attention {att!r}"
            att.qk_below = qk_below
            try:
                inp = module(inp, **kwargs)
                qk_below = att.qk_out
            finally:
                att.qk_below = None
                att.qk_out = None
            if collected_outputs is not None:
                collected_outputs[name] = inp
        return inp
