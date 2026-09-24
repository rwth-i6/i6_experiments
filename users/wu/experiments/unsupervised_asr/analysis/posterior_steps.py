"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/eval_jobs.py (posterior_forward_step :229,
PosteriorHdfCallback :268, FORWARD_DATA_KEY).

RETURNN-side code of the posterior dump: the ``forward_step`` and the ``forward_callback`` that write
``posteriors.hdf`` ([T, n_out] float32 per utterance) and ``posteriors.frames.json``.  The sisyphus
side (config builder and the ``ReturnnForwardJobV2``) is :mod:`.posterior`.  Nothing here is imported
at sisyphus graph time, and ``returnn`` / ``torch`` are imported inside the functions only.
"""

from __future__ import annotations

__all__ = ["FORWARD_DATA_KEY", "posterior_forward_step", "PosteriorHdfCallback"]

# The key a plain HDFDataset serves its single stream under.  RETURNN's HDFDataset ALWAYS calls it
# "data" (returnn/datasets/hdf.py), and there is no config option that renames it: the forward
# template must declare ``extern_data`` under this name or ``raw_dict_to_extern_data`` raises
# KeyError before the first batch.
FORWARD_DATA_KEY = "data"


def posterior_forward_step(*, model, extern_data, features_key: str = FORWARD_DATA_KEY, **_kwargs):
    """RETURNN ``forward_step``: mark the recognizer's per-frame log-probs as the output."""
    import torch

    import returnn.frontend as rf
    from returnn.tensor import Dim, Tensor as ReturnnTensor

    feats_ = extern_data[features_key]
    feats = feats_.raw_tensor
    b_dim = feats_.dims[0]
    feat_lens = feats_.dims[1].dyn_size_ext.raw_tensor.to(feats.device)

    with torch.no_grad():
        log_probs = model(feats, feat_lens)  # [B, T_out, n_out]
    out_lens = model.output_lengths(feat_lens)

    if int(model.stride) == 1:
        # stride 1 keeps T_out == T exactly (recognizer.conv_output_lengths), so the input spatial
        # dim is reused -- no second dyn-size tensor that could drift from the features.
        t_dim = feats_.dims[1]
    else:
        t_dim = Dim(
            ReturnnTensor(
                "out_lens",
                dims=[b_dim],
                dtype="int32",
                raw_tensor=out_lens.to(device="cpu", dtype=torch.int32),
            ),
            name="out_time",
        )
    out_dim = Dim(int(log_probs.size(-1)), name="out_labels")
    # left on the device: RETURNN slices each sequence out of the batch, cuts the padding by the
    # spatial dim and converts to numpy itself (torch/engine.py:1389-1414)
    lp = log_probs.detach().float()
    rf.get_run_ctx().mark_as_output(
        rf.convert_to_tensor(lp, dims=[b_dim, t_dim, out_dim]), "log_probs", dims=[b_dim, t_dim, out_dim]
    )


def PosteriorHdfCallback(
    *,
    out_hdf_file: str = "posteriors.hdf",
    out_frames_file: str = "posteriors.frames.json",
    n_out: int = 41,
    **_kwargs,
):
    """Build the forward callback writing ``posteriors.hdf`` ([T, n_out] float32 per utterance).

    A FACTORY, not a class: RETURNN asserts the callback is a ``ForwardCallbackIface`` instance
    (torch/engine.py:1344) and calls ``forward_callback()`` when it is callable (__main__.py:581),
    so returning an instance from a function is enough -- and it keeps ``import returnn`` out of
    module import time.

    float32 and not fp16: the dump feeds an argmax (and, in the source, a beam search whose pruning
    threshold is an absolute score difference); a dev split is a few hundred MB.
    """
    import json

    import numpy as np

    from returnn.datasets.hdf import SimpleHDFWriter
    from returnn.forward_iface import ForwardCallbackIface

    class _PosteriorHdfCallback(ForwardCallbackIface):
        def __init__(self):
            self._writer = None
            self._frames = {}

        def init(self, *args, **kwargs):
            self._writer = SimpleHDFWriter(filename=out_hdf_file, dim=n_out, ndim=2)
            self._frames = {}

        def process_seq(self, *, seq_tag, outputs, **kwargs):
            x = np.asarray(outputs["log_probs"].raw_tensor, dtype="float32")
            assert x.ndim == 2 and x.shape[1] == n_out, (seq_tag, x.shape)
            assert seq_tag not in self._frames, f"duplicate seq tag in forward: {seq_tag}"
            self._writer.insert_batch(x[None, :, :], [x.shape[0]], [seq_tag])
            self._frames[seq_tag] = int(x.shape[0])

        def finish(self, **kwargs):
            self._writer.close()
            with open(out_frames_file, "w") as fh:
                json.dump(self._frames, fh)
            n, f = len(self._frames), sum(self._frames.values())
            print(f"posterior dump: {n} utts, {f} frames, {n_out} outputs", flush=True)

    return _PosteriorHdfCallback()
