__all__ = ["RecognitionToTextDictCallback"]

from typing import Union, Dict, Optional, List

import numpy as np

from i6_core.util import uopen
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict


class RecognitionToTextDictCallback(ForwardCallbackIface):
    def __init__(self, *, out_hyps_file: str = "search_out.py.gz", include_beam: bool = False, **unused_kwargs):
        self.out_hyps_file = out_hyps_file
        self.include_beam = include_beam

        self.vocab: Vocabulary

    def init(self, *args, **kwargs):
        self._out_file = uopen(self.out_hyps_file, "wt")
        self._out_file.write("{\n")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, **kwargs):
        tokens: np.ndarray = outputs["tokens"].raw_tensor  # Beam, Time
        assert tokens.ndim == 1  # (T,)
        scores: np.ndarray = outputs["scores"].raw_tensor  # Beam

        if not tokens.size or not scores.size:
            self._out_file.write(f"  {repr(seq_tag)}: '',\n")
            return

        text = bytes(tokens).decode("utf-8")

        self._out_file.write(f"  {repr(seq_tag)}: {repr(text)},\n")
        self._out_file.flush()

    def finish(self, **kwargs):
        self._out_file.write("}\n")
        self._out_file.close()
        self._out_file = None
