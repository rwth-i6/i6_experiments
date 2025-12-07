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
from collections.abc import Iterator, Sequence
import torch
from returnn.torch.frontend._backend import TorchBackend


class ConcatDataPostproc(Callable[[Iterator[TensorDict]], Iterator[TensorDict]]):
    """
    Takes two seqs, and concatenates them (sometimes)
    """

    preserves_num_seqs = False

    def __init__(self, concat_keys: Sequence[str], *, concat_prob: float = 0.1, length_key: str = "data"):
        """
        :param concat_keys: data keys which should be concatenated
        :param concat_prob: probability of concatenating two sequences.
        :param length_key: data key to determine the segment length from for ordering.
        """
        self.length_key = length_key
        self.concat_prob = concat_prob
        self.concat_keys = concat_keys

    def __call__(self, iterator: Iterator[TensorDict], **kwargs) -> Iterator[TensorDict]:
        """:return: generator"""
        iterator = iter(iterator)

        is_first = True
        try:
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
                if m is None:  # done
                    yield n
                    break

                # we copy everything from m, because that tensors' complete_frac and other info is more relevant
                x = {}
                for name, tensor in m.data.items():
                    if name in self.concat_keys:
                        assert len(n.data[name].dims_set) == 1
                        assert len(tensor.dims_set) == 1
                        # TODO maybe a seperator token?

                        # rf concat doesnt work because the dims are whacky
                        # just use torch code
                        # concatenated, _ = rf.concat((n.data[name], n.data[name].dims[0]), (tensor, tensor.dims[0]))
                        a: numpy.ndarray = n.data[name].raw_tensor
                        b: numpy.ndarray = tensor.raw_tensor
                        assert isinstance(a, numpy.ndarray)
                        assert isinstance(b, numpy.ndarray)
                        assert a.ndim == 1
                        assert b.ndim == 1

                        # apparently these are numpy arrays???
                        concat = numpy.concatenate([a, b], axis=0)
                        # concat = torch.cat([a, b], dim=0)

                        # rf.convert_to_tensor crashes for some reason...
                        x[name] = rf.Tensor(
                            f"{tensor.name or 'b'}_concat",
                            dims=tensor.dims,
                            dtype=tensor.dtype,
                            sparse_dim=tensor.sparse_dim,
                            feature_dim=tensor.feature_dim,
                            raw_tensor=concat,
                        )
                    else:
                        x[name] = tensor

                yield TensorDict(x)
        except StopIteration:
            return  # postprocessing dataset doesnt like StopIteration being raised?
