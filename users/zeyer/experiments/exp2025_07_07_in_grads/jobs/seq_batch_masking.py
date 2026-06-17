"""Sequence-length masking for batched (B>1) forward.

Reused by every job that runs a batched model forward
-- grad extract, forced align, attention extract/align --
so the masking lives here, in the shared model layer, not in any single job.

Why it is needed:
many encoder modules do not mask padded time frames themselves
(conv, pooling / time-reduction, norm-over-time),
so a seq's output inside a B>1 batch differs from its single (B=1) output.
This zeroes padded positions before each such module,
threading the per-sample valid length through the module's stride / kernel geometry.

Assumes the time axis is the LAST dim of the module input
(true for nn.Conv1d / AvgPool1d / MaxPool1d, which is what the encoders here use);
Conv2d spectrogram frontends would need an explicit time-axis -- not handled yet (assert).
GroupNorm / BatchNorm over time need masked statistics, not input-zeroing --
not handled here; the models we batch use per-position LayerNorm (no leak).
"""

import torch


def _last(x):
    return x[-1] if isinstance(x, (tuple, list)) else x


class SeqMaskHooks:
    """Forward hooks that mask padded time positions before each time-mixing module.

    Use as a context manager around a batched forward::

        with SeqMaskHooks(encoder, feat_lengths):
            enc = encoder(feats_padded, ...)

    ``input_lengths`` are the per-sample valid lengths
    at the resolution of the FIRST hooked module's input.
    """

    MASKED_TYPES = (
        torch.nn.Conv1d,
        torch.nn.AvgPool1d,
        torch.nn.MaxPool1d,
    )

    def __init__(self, model, input_lengths):
        self.valid = input_lengths.long().clone()
        self.handles = []
        self.n_pre = 0
        self.n_post = 0
        for _nm, m in model.named_modules():
            if isinstance(m, self.MASKED_TYPES):
                assert m.kernel_size, m
                self.handles.append(m.register_forward_pre_hook(self._pre))
                self.handles.append(m.register_forward_hook(self._post))

    def _pre(self, module, inputs):
        self.n_pre += 1
        x = inputs[0]
        t = x.shape[-1]
        mask = torch.arange(t, device=x.device)[None, :] < self.valid.to(x.device)[:, None]  # [B, T]
        view = [x.shape[0]] + [1] * (x.ndim - 2) + [t]
        return (x * mask.view(view).to(x.dtype),) + tuple(inputs[1:])

    def _post(self, module, _inp, _out):
        self.n_post += 1
        k, s = _last(module.kernel_size), _last(module.stride)
        p, d = _last(getattr(module, "padding", 0)), _last(getattr(module, "dilation", 1))
        self.valid = (self.valid + 2 * p - d * (k - 1) - 1) // s + 1

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        self.remove()
