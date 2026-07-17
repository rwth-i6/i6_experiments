from typing import Union, Iterable, Iterator, Optional, Dict, Any, Callable

from sisyphus import tk

import torch
import numpy as np

from returnn.tensor import Dim, Tensor, TensorDict
from returnn.datasets.util.vocabulary import Vocabulary


class PhonemizeAndInsertSilence:
    """ """

    preserves_num_seqs = True

    def __init__(
        self,
        target_key: str,
        new_target_key: str,
        lexicon_file: str,
        sil_prob: float,
        surround_w_sil: bool,
        vocab_opts: Dict,
        min_num_sil: int = 1,
        max_num_sil: int = 1,
        **unused_kwargs,
    ):
        """ """

        self.target_key = target_key
        self.new_target_key = new_target_key
        self.lexicon_file = lexicon_file
        self.sil_prob = sil_prob
        self.surround_w_sil = surround_w_sil

        assert min_num_sil >= 1 and max_num_sil >= min_num_sil
        self.min_num_sil = min_num_sil
        self.max_num_sil = max_num_sil

        self.wrd_to_phn = {}

        with open(self.lexicon_file, "r") as lf:
            for line in lf:
                items = line.rstrip().split()
                assert len(items) > 1, line
                assert items[0] not in self.wrd_to_phn, items
                self.wrd_to_phn[items[0]] = items[1:]

        from returnn.datasets.util.vocabulary import Vocabulary

        self.vocab = Vocabulary.create_vocab(**vocab_opts)

    def __call__(self, seq_or_iterator: Union[TensorDict, Iterator[TensorDict]], *args, **kwargs):
        """
        Expects `data[self.target_key]` to be a 1D array of utf-8 encoded bytes representing the text sequence.
        Converts it to a string, lowercases it, applies the vocab to get token ids, and stores it back in
        `data[self.target_key]` as a 1D array of token ids.
        """

        if isinstance(seq_or_iterator, Iterator):
            return (self.phonemize_with_sil(seq) for seq in seq_or_iterator)
        else:
            assert isinstance(seq_or_iterator, TensorDict)
            return self.phonemize_with_sil(seq_or_iterator)

    def phonemize_with_sil(
        self,
        data: TensorDict,
    ):
        targets: np.ndarray = data.data[self.target_key].raw_tensor
        raw_bytes = targets.tobytes()
        # decode and strip null bytes (\x00) which are common in fixed-length numpy byte strings
        text = raw_bytes.decode("utf-8").replace("\x00", "").lower()

        sil = "<SIL>"

        words = text.strip().split()

        if not all(w in self.wrd_to_phn for w in words):
            raise ValueError(
                "Not all words in lexicon: {}".format(", ".join(w for w in words if w not in self.wrd_to_phn))
            )

        phones_w_sil = []
        if self.surround_w_sil:
            phones_w_sil.append(sil)

        sample_sil_probs = None
        if self.sil_prob > 0 and len(words) > 1:
            sample_sil_probs = np.random.random(len(words) - 1)

        for j, w in enumerate(words):
            phones_w_sil.extend(self.wrd_to_phn[w])
            if sample_sil_probs is not None and j < len(sample_sil_probs) and sample_sil_probs[j] < self.sil_prob:
                num_sil = np.random.randint(self.min_num_sil, self.max_num_sil + 1)
                phones_w_sil += [sil] * num_sil

        if self.surround_w_sil:
            phones_w_sil.append(sil)

        phones_wo_sil = [p for p in phones_w_sil if p != sil]

        tensor_w_sil = data.data[self.target_key].copy_template(dtype="int32")
        tensor_w_sil.raw_tensor = np.array(self.vocab.get_seq(" ".join(phones_w_sil)), dtype=np.int32)
        tensor_wo_sil = data.data[self.target_key].copy_template(dtype="int32")
        tensor_wo_sil.raw_tensor = np.array(self.vocab.get_seq(" ".join(phones_wo_sil)), dtype=np.int32)
        data.data[self.new_target_key] = tensor_w_sil
        data.data[self.target_key] = tensor_wo_sil

        return data
