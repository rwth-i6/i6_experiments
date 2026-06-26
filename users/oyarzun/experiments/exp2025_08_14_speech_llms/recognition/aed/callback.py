__all__ = ["RecognitionToTextDictCallback"]


import numpy as np

from i6_core.util import uopen
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict


class RecognitionToTextDictCallback(ForwardCallbackIface):
    def __init__(self, *, vocab: str, out_hyps_file: str = "search_out.py.gz"):
        self.vocab_file = vocab
        self.out_hyps_file = out_hyps_file

        self.vocab: Vocabulary

    def init(self, *args, **kwargs):
        self.vocab = Vocabulary.create_vocab(vocab_file=self.vocab_file, unknown_label=None)

        self._out_file = uopen(self.out_hyps_file, "wt")
        self._out_file.write("{\n")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, **kwargs):
        tokens: np.ndarray = outputs["tokens"].raw_tensor  # Beam, Time
        assert tokens.ndim == 2
        dyn_dims = [d for d in outputs["tokens"].dims if d.dyn_size_ext is not None]
        assert len(dyn_dims) == 1, f"found more than one dynamic dim: {dyn_dims}"
        tokens_lens: np.ndarray = dyn_dims[0].dyn_size_ext.raw_tensor  # Beam
        assert tokens_lens.ndim == 1
        scores: np.ndarray = outputs["scores"].raw_tensor  # Beam
        assert scores.ndim == 1

        if not tokens.size or not scores.size:
            self._out_file.write(f"  {repr(seq_tag)}: '',\n")
            return

        best_hyp_idx = np.argmax(scores)
        best_hyp = tokens[best_hyp_idx]
        best_hyp_len = tokens_lens[best_hyp_idx]
        text = self.vocab.get_seq_labels(best_hyp[:best_hyp_len])

        self._out_file.write(f"  {repr(seq_tag)}: {repr(text)},\n")
        self._out_file.flush()

    def finish(self, **kwargs):
        self._out_file.write("}\n")
        self._out_file.close()
        self._out_file = None
