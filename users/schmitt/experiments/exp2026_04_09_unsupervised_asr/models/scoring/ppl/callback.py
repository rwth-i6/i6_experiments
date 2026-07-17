__all__ = ["PplScoresCallback"]

import math

import numpy as np

from i6_core.util import uopen
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict


class PplScoresCallback(ForwardCallbackIface):
    """
    Writes the per-sequence cross entropy / perplexity plus a corpus-level summary.

    Consumes the ``ce`` (summed token CE) and ``num_tokens`` (scored tokens incl. eos) outputs marked
    by the PPL ``forward_step``. Per seq it writes ``{'ce', 'num_tokens', 'ppl'}`` to a python dict
    file; in ``finish`` it writes a summary with the corpus perplexity ``exp(sum ce / sum tokens)``.
    """

    def __init__(
        self,
        *,
        out_scores_file: str = "ppl_scores.py.gz",
        out_summary_file: str = "ppl_summary.txt",
        vocab=None,  # injected by serialize_forward; unused here (we score ids, not labels)
    ):
        self.out_scores_file = out_scores_file
        self.out_summary_file = out_summary_file

        self._out_file = None
        self._total_ce = 0.0
        self._total_tokens = 0
        self._num_seqs = 0

    def init(self, *args, **kwargs):
        self._out_file = uopen(self.out_scores_file, "wt")
        self._out_file.write("{\n")
        self._total_ce = 0.0
        self._total_tokens = 0
        self._num_seqs = 0

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, **kwargs):
        ce = float(np.asarray(outputs["ce"].raw_tensor))
        num_tokens = int(np.asarray(outputs["num_tokens"].raw_tensor))
        ppl = math.exp(ce / num_tokens) if num_tokens > 0 else float("nan")

        self._out_file.write(f"  {seq_tag!r}: {{'ce': {ce!r}, 'num_tokens': {num_tokens}, 'ppl': {ppl!r}}},\n")
        self._out_file.flush()

        self._total_ce += ce
        self._total_tokens += num_tokens
        self._num_seqs += 1

    def finish(self, **kwargs):
        self._out_file.write("}\n")
        self._out_file.close()
        self._out_file = None

        mean_ce = self._total_ce / self._total_tokens if self._total_tokens > 0 else float("nan")
        corpus_ppl = math.exp(mean_ce) if self._total_tokens > 0 else float("nan")
        with open(self.out_summary_file, "w") as f:
            f.write(f"num_seqs={self._num_seqs}\n")
            f.write(f"total_tokens={self._total_tokens}\n")
            f.write(f"total_ce={self._total_ce!r}\n")
            f.write(f"mean_ce_per_token={mean_ce!r}\n")
            f.write(f"perplexity={corpus_ppl!r}\n")
