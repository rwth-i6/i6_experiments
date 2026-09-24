"""New in the port: the RETURNN-side half of the reverse-model scoring that speech-llm c49559ce
src/speech_llm/sae/emc/blankfree_eval_jobs.py (BlankfreeDerangementGapJob.run :159-177) and
blankfree_decode_gap_jobs.py (BlankfreeDecodeGapJob.run) did in-process.

The source jobs loaded the ``reverse.`` checkpoint slice with ``torch.load`` and called
``reverse.evaluate(model, items, per_utterance=True)`` once per condition.  The port runs that model
forward as a ``ReturnnForwardJobV2`` (brief: every model forward outside RETURNN becomes an RFJv2 plus a
CPU reader).  To keep the numbers the source's, the forward step does NOT score RETURNN's batches
directly: it regroups the batch into the source's per-condition item lists (in the source's item
order) and calls the same ``evaluate`` on each list, so ``evaluate``'s own length bucketing (batches of
8 sorted by (S, U)) and ``pack_batch`` padding are unchanged.  That needs every item of a condition in
ONE RETURNN batch; the ``index`` stream carries (condition id, position, condition size) and the step
asserts completeness instead of silently scoring a partial list.

Items dataset (written by :class:`.gaps.ReverseGapItemsJob`), one sequence per scored item, seq tag
``"<condition>/<utterance tag>"``:

    data    int32 [S]  unit stream z
    phones  int32 [U]  phone-id string y (``phones.PHONES`` ids)
    eta     float32 [E] speaker vector, stored as a length-E sequence
    index   int32 [3]  (condition id, position in the condition's item list, condition size)

Output ``scores.json``: ``{seq tag: log p_phi(z | y, eta)}`` (the python float ``evaluate`` returns).
The model runs on CPU as in the source (``reverse.evaluate`` builds CPU tensors).
"""

from __future__ import annotations

__all__ = ["ITEM_KEYS", "get_reverse_model", "reverse_score_forward_step", "ReverseScoreCallback"]

#: the dataset keys of one scored item (see the module docstring)
ITEM_KEYS = ("data", "phones", "eta", "index")


def get_reverse_model(*, epoch: int = 0, step: int = 0, reverse_kwargs=None, **_kwargs):
    """RETURNN ``get_model``: ``SegmentalReverseModel(ReverseConfig(**reverse_kwargs))``.

    ``reverse_kwargs=None`` is the bed's ``ReverseConfig()`` (``gaps.gap_reverse_config``).  RETURNN
    passes ``epoch``, ``step`` and a forward-compatibility sentinel kwarg; none is a model argument.
    """
    from ..model.reverse import ReverseConfig, SegmentalReverseModel

    return SegmentalReverseModel(ReverseConfig(**dict(reverse_kwargs or {})))


def reverse_score_forward_step(*, model, extern_data, **_kwargs):
    """RETURNN ``forward_step``: ``log p_phi`` per item through ``reverse.evaluate``, per condition."""
    import torch

    import returnn.frontend as rf

    from ..model.reverse import evaluate

    z_ = extern_data["data"]
    b_dim = z_.dims[0]

    def _seqs(key):
        t = extern_data[key]
        raw = t.raw_tensor.detach().cpu().numpy()
        lens = t.dims[1].dyn_size_ext.raw_tensor.detach().cpu().numpy()
        return [raw[b, : int(lens[b])] for b in range(raw.shape[0])]

    z, y, eta, index = (_seqs(k) for k in ITEM_KEYS)
    n = len(z)
    groups = {}
    for b in range(n):
        assert index[b].shape == (3,), index[b].shape
        cond, pos, total = (int(v) for v in index[b])
        groups.setdefault(cond, []).append((pos, b, total))
    log_p = [None] * n
    for cond, members in sorted(groups.items()):
        members.sort()
        total = members[0][2]
        assert [m[0] for m in members] == list(range(total)), (
            f"condition {cond}: {len(members)} of {total} items in this batch; the scoring needs a "
            "whole condition in one batch (raise batch_size / max_seqs)")
        # the source's items: (z as python ints, y as python ints, eta row as float32 array)
        items = [(z[b].tolist(), [int(k) for k in y[b].tolist()], eta[b]) for _, b, _ in members]
        _total, _frames, rows = evaluate(model, items, per_utterance=True)
        for r in rows:
            log_p[members[r["index"]][1]] = r["log_p"]
    assert all(v is not None for v in log_p)
    out = torch.tensor(log_p, dtype=torch.float64)  # exact carrier of the python floats
    rf.get_run_ctx().mark_as_output(rf.convert_to_tensor(out, dims=[b_dim]), "log_p", dims=[b_dim])


def ReverseScoreCallback(*, out_file: str = "scores.json", **_kwargs):
    """Forward callback writing ``{seq tag: log_p}`` (a factory, see ``PosteriorHdfCallback``)."""
    import json

    import numpy as np

    from returnn.forward_iface import ForwardCallbackIface

    class _ReverseScoreCallback(ForwardCallbackIface):
        def __init__(self):
            self._scores = {}

        def init(self, *args, **kwargs):
            self._scores = {}

        def process_seq(self, *, seq_tag, outputs, **kwargs):
            assert seq_tag not in self._scores, f"duplicate seq tag in forward: {seq_tag}"
            self._scores[seq_tag] = float(np.asarray(outputs["log_p"].raw_tensor).reshape(()))

        def finish(self, **kwargs):
            with open(out_file, "w") as fh:
                json.dump(self._scores, fh)
            print(f"reverse scoring: {len(self._scores)} items", flush=True)

    return _ReverseScoreCallback()
