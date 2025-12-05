from __future__ import annotations

import returnn.frontend as rf
from returnn.datasets.basic import Dataset, DatasetSeq, TensorDict, init_dataset, convert_data_dims
from returnn.datasets.cached2 import CachedDataset2
import returnn.util.basic as util
from returnn.util.basic import NumbersDict, load_json, OptionalNotImplementedError
import random
import numpy
import sys
import math
from returnn.log import log
from decimal import Decimal
from itertools import islice
from typing import Callable
from collections.abc import Iterator

class ConcatDataPostproc(Callable[[Iterator[TensorDict]], Iterator[TensorDict]]):
    """
    Takes two seqs, and concatenates them (sometimes)
    """

    preserves_num_seqs = False

    def __init__(self, concat_dims: dict[str, rf.Dim], *, concat_prob: float = 0.1, length_key: str = "data"):
        """
        :param concat_dims: dict of dimensions along which to concatenate. If not given, that tensor will not be concatenated and will be taken from the second sequence.
        :param concat_prob: probability of concatenating two sequences.
        :param length_key: data key to determine the segment length from for ordering.
        """
        self.length_key = length_key
        self.concat_prob = concat_prob
        self.concat_dims = concat_dims

    def __call__(self, iterator: Iterator[TensorDict], **kwargs) -> Iterator[TensorDict]:
        """:return: generator"""
        iterator = iter(iterator)
        

        is_first = True
        while True:

            # get one
            n = next(iterator)
            if is_first:
                # print every entry once
                is_first = False
                print("ConcatDataPostproc: first data entry (debug):")
                for name, tensor in n.data.items():
                    print(f"ConcatDataPostproc: tensor '{name}': {tensor}")

            if random.random() > self.concat_prob:
                yield n
                continue

            # get another one
            m = next(iterator, None)
            if m is None: # done
                yield n
                break
 
            # we copy everything from m, because that tensors' complete_frac and other info is more relevant
            x = {}
            for name, tensor in m.data.items():
                if name in self.concat_dims:
                    dim = self.concat_dims[name]
                    assert dim in n.data[name].dims, f"concat dim {dim} not in tensor {name} dims {n.data[name].dims}"
                    # TODO maybe a seperator token?
                    concatenated, _ = rf.concat((n.data[name], dim), (tensor, dim))
                    x[name] = concatenated
                else:
                    x[name] = tensor

            yield TensorDict(x)

    def _get_seq_len(self, tdict: TensorDict) -> int:
        """
        :return: segment length of the segment in `tdict` as measured by `self.length_key` for comparison.
        """
        return tdict.data[self.length_key].raw_tensor.shape[0]

