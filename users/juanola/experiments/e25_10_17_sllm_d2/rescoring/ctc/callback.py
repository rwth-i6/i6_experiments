__all__ = ["RecognitionToTextDictCallbackV2"]

from typing import Optional, Union, Dict, List

import numpy as np

from i6_core.util import uopen
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict
from returnn.tensor import Tensor
from returnn.util.basic import Stats


class RecognitionToTextDictCallbackV2(ForwardCallbackIface):
    def __init__(
        self,
        *,
        vocab: Union[str, Dict],
        out_hyps_file: str = "search_out.py.gz",
        remove_labels: Optional[List[int]] = None,
        include_beam: bool = False,
        merge_labels: bool = True,
    ):
        self.vocab_opts = vocab
        self.out_hyps_file = out_hyps_file
        self.remove_labels = remove_labels
        self.include_beam = include_beam
        self.merge_labels = merge_labels

        self.vocab: Vocabulary

    def init(self, *args, **kwargs):
        if isinstance(self.vocab_opts, str):
            self.vocab = Vocabulary.create_vocab(vocab_file=self.vocab_opts, unknown_label=None)
        else:
            assert isinstance(self.vocab_opts, dict)
            self.vocab = Vocabulary.create_vocab(**self.vocab_opts)

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

        if self.include_beam:
            num_beam = scores.shape[0]
            self._out_file.write(f"{seq_tag!r}: [\n")
            for i in range(num_beam):
                score = float(scores[i])
                hyp_ids = tokens[i, : tokens_lens[i] if tokens_lens.shape else tokens_lens]
                if self.merge_labels:
                    hyp_serialized = self.vocab.get_seq_labels(hyp_ids)
                else:
                    labels = [self.vocab.id_to_label(int(l)) for l in hyp_ids]
                    hyp_serialized = " ".join(labels)
                self._out_file.write(f"  ({score!r}, {hyp_serialized!r}),\n")
            self._out_file.write("],\n")
        else:
            if not tokens.size or not scores.size:
                self._out_file.write(f"  {repr(seq_tag)}: '',\n")
                return

            best_hyp_idx = np.argmax(scores)
            best_hyp = tokens[best_hyp_idx]
            best_hyp_len = tokens_lens[best_hyp_idx]
            best_hyp = best_hyp[:best_hyp_len]
            if self.remove_labels is not None:
                best_hyp = np.array([t for t in best_hyp if t not in self.remove_labels], dtype=best_hyp.dtype)
            if self.merge_labels:
                text = self.vocab.get_seq_labels(best_hyp)
            else:
                labels = [self.vocab.id_to_label(int(l)) for l in best_hyp]
                text = " ".join(labels)

            self._out_file.write(f"  {repr(seq_tag)}: {repr(text)},\n")
            self._out_file.flush()

    def finish(self, **kwargs):
        self._out_file.write("}\n")
        self._out_file.close()
        self._out_file = None
