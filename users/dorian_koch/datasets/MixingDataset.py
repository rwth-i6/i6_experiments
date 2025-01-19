from __future__ import annotations

from typing import Literal, Optional, Union, Any, Callable, Sequence, List, Dict, Tuple
from returnn.datasets.basic import Dataset, DatasetSeq, init_dataset, convert_data_dims
from returnn.datasets.cached2 import CachedDataset2
import returnn.util.basic as util
from returnn.util.basic import NumbersDict, load_json, OptionalNotImplementedError
from i6_experiments.users.zeyer.utils.basic import make_hashable
from random import Random
import numpy
import sys
import typing
from i6_experiments.users.zeyer.utils.lru_cache import lru_cache
import math
from returnn.log import log


class Bitarray:
    def __init__(self, size: int):
        self.size = size
        self.data = numpy.zeros((size + 31) // 32, dtype=numpy.uint32)

    def set(self, i: int, value: bool):
        if value:
            self.data[i // 32] |= 1 << (i % 32)
        else:
            self.data[i // 32] &= ~(1 << (i % 32))

    def get(self, i: int) -> bool:
        return bool(self.data[i // 32] & (1 << (i % 32)))

    def __len__(self):
        return self.size


class MixingDataset(CachedDataset2):
    """
    This mixes two datasets. They are expected to provide the same data-keys and data-dimensions.
    They must not have the same sequence tags.

    Sequences are chosen based on a mixing factor.
    With a mixing factor of 0.75, a minibatch should contain on average 25% of data from the left dataset and 75% from the right dataset.
    The % is based on the size of the data, not the amount of sequences.
    So even with a 50/50 split, it may be possible to have a minibatch with a single long sequence from the left dataset and many short sequences from the right dataset.

    Both datasets work in steplock, meaning that they are at the same epoch at all times.
    This means that, under some configurations, an epoch of one dataset may be seen many times.
        If this is problematic maybe wrap it in a MultiEpochDataset? (does it support num_seqs? idk)

    # TODO i overcomplicated some things in the design of this, 
    1. I hyper optimized for memory usage, which makes the code very messy
    2. Because of 1, this doesnt scale well at all inside a MultiProcDataset
    3. This supports random access, but I had to hack some stuff together because apparently other Datasets don't support that?
    """

    def __init__(
        self,
        left_dataset: Dict[str, Any],
        right_dataset: Dict[str, Any],
        mixing_ratio: float = 0.5,
        how_to_handle_end_of_data_from_one_dataset: Union[Literal["exception"], Literal["wrap_around"], Literal["early_exit"]] = "wrap_around",
        *,
        data_key: str = "data",
        control_dataset: str = "left",
        **kwargs,
    ):
        """
        :param left_dataset:
        :param right_dataset:
        :param mixing_ratio: probability to choose the right dataset
        :param data_key: key for the mixing process, mixing considers the size of the data
        :param how_to_handle_end_of_data_from_one_dataset: what to do when one dataset is exhausted
            exception: raise an exception, this should practically never be used in training
            wrap_around: wrap around to the beginning of the dataset that is exhausted. Terminate when both datasets have terminated at least once.
            early_exit: end epoch when one dataset has been exhausted
        :param control_dataset: which dataset is used for i.e. `get_data_dtype`
        """
        super().__init__(**kwargs)
        assert 0.0 <= mixing_ratio <= 1.0
        self.mixing_ratio = mixing_ratio
        self.how_to_handle_end_of_data_from_one_dataset = how_to_handle_end_of_data_from_one_dataset
        self.left_dataset = init_dataset(left_dataset, parent_dataset=self)
        self.right_dataset = init_dataset(right_dataset, parent_dataset=self)
        self.control_dataset = self.left_dataset if control_dataset == "left" else self.right_dataset
        self.num_inputs = make_hashable(self.left_dataset.num_inputs) # make_hashable normalizes lists/tuples to just tuples
        self.num_outputs = make_hashable(self.left_dataset.num_outputs)
        self.labels = self.left_dataset.labels
        self.data_key = data_key

        assert make_hashable(self.right_dataset.num_inputs) == self.num_inputs
        assert make_hashable(self.right_dataset.num_outputs) == self.num_outputs
        self._reset_params()

    def _reset_params(self):
        assert not (0 < self.right_dataset.num_seqs < 10) and not (0 < self.left_dataset.num_seqs < 10), "mixing can go wrong when one dataset has very few seqs"
        # left finishes first
        lff = self.right_dataset.num_seqs * (1 + (1-self.mixing_ratio)/(self.mixing_ratio))
        # right finishes first
        rff = self.left_dataset.num_seqs * (1 + (self.mixing_ratio)/(1-self.mixing_ratio))
        if self.how_to_handle_end_of_data_from_one_dataset in ["exception", "early_exit"]:
            assert 0.0 < self.mixing_ratio < 1.0, "not implemented"
            self.total_num_seqs_upper_bound = math.ceil(min(lff, rff)) # only one needs to finish
        elif self.how_to_handle_end_of_data_from_one_dataset == "wrap_around":
            assert 0.0 < self.mixing_ratio < 1.0, "not implemented"
            self.total_num_seqs_upper_bound = math.ceil(max(lff, rff)) # both need to finish
        else:
            assert False

        assert not math.isnan(self.total_num_seqs_upper_bound) and not math.isinf(self.total_num_seqs_upper_bound)
        # for good measure
        self.total_num_seqs_upper_bound += 10
        self.total_num_seqs_upper_bound *= 2

        if self.total_num_seqs_upper_bound > 0:
            print(f"MixingDataset init: {self.left_dataset.num_seqs} + {self.right_dataset.num_seqs}, upperbound={self.total_num_seqs_upper_bound}, mixingratio={self.mixing_ratio}", file=log.v4)
        else:
            print("MixingDataset init: both datasets are empty", file=log.v4)
        self._estimated_num_seqs = self.total_num_seqs_upper_bound
        assert self.total_num_seqs_upper_bound < 2**31, "sequences do not fit into int32"
        # 0 means left, 1 means right
        self.bitset_chooser = Bitarray(self.total_num_seqs_upper_bound)
        # cache indices to both datasets at a particular sequence index
        self.index_cache = numpy.zeros(
            ((self.total_num_seqs_upper_bound + 1023) // 1024, 2), dtype=numpy.int32
        )
        # up until which point we have chosen
        self.chooser_index = 0
        self.is_chooser_done = False
        self.chooser_childindices = [0, 0]
        self.datasets_exhausted = [False, False]
        self.datasets_loaded_until = [0, 0] # we need to _load_seqs the datasets
        # we will get out of balance while choosing, we will correct this by biasing the next choice
        self.bias = 0.0
        self.datalens = [0, 0]
        self._get_childindices_at_seq_idx.cache_clear()

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :type epoch: int|None
        :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
        :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
        """
        need_reinit = self.epoch is None or self.epoch != epoch
        super().init_seq_order(
            epoch=epoch, seq_list=seq_list, seq_order=seq_order
        )
        if not need_reinit:
            return False

        if seq_order is not None:
            raise NotImplementedError(
                "Predefined order via sequence indices for MixingDataset"
            )
        if seq_list is not None:
            raise NotImplementedError(
                "Predefined order via sequence tags for MixingDataset"
            )
        elif self.seq_ordering != "default":
            raise NotImplementedError("seq_ordering %s" % self.seq_ordering)

        self.left_dataset.init_seq_order(epoch=epoch)
        self.right_dataset.init_seq_order(epoch=epoch)
        # TODO test whether both datasets have exactly the same metadata, like dtype
        self._reset_params()
        return True

    @staticmethod
    def _data_metric(v: numpy.ndarray):
        return v.shape[0] if v.ndim >= 1 else 1

    def _make_sure_idx_is_loaded_in_child_ds(self, dataset_index, seq_idx):
        chosen_dataset = self.left_dataset if dataset_index == 0 else self.right_dataset
        child_len = chosen_dataset.num_seqs

        # TODO fix this stupid hack
        if hasattr(chosen_dataset, "expected_load_seq_start") and seq_idx < chosen_dataset.expected_load_seq_start:
            chosen_dataset.init_seq_order(epoch=self.epoch)
            self.datasets_loaded_until[dataset_index] = 0

        if self.datasets_loaded_until[dataset_index] <= seq_idx:
            # 512 is just some arbitrary number TODO maybe decrease this for more intensive workloads?
            start = seq_idx
            end = (seq_idx + min(child_len - 1, 512)) % child_len

            if end < start:
                # print(f"({dataset_index}) end < start: loading segs from {start} to {child_len}", file=log.v4)
                chosen_dataset.load_seqs(start, child_len)
                self.datasets_loaded_until[dataset_index] = child_len
                assert self.datasets_loaded_until[dataset_index] >= seq_idx
                # not sure if we should also load from 0 to end here, it may erase the data from start to child_lens? idk
            else:
                # print(f"({dataset_index}) just loading segs from {start} to {end}", file=log.v4)
                chosen_dataset.load_seqs(start, end)
                self.datasets_loaded_until[dataset_index] = end

    def _run_seq_idx(self, seq_idx):
        if seq_idx < self.chooser_index:
            raise Exception("seq_idx < chooser_index")
        assert seq_idx < self.total_num_seqs_upper_bound, "This assert fails only when the two datasets are very unbalanced, in the sense that one dataset has many long sequences while the other mostly has shorter once. Keep them on equal lengths on average please! Otherwise you need to somehow increase this upper bound (which will not cause issues, just eat more ram)"
        if self.is_chooser_done:
            raise Exception("chooser is done. change attribute 'how_to_handle_end_of_data_from_one_dataset' to 'exception' if you want to know why (probably because early_exit)")

        child_lens = [
            self.left_dataset.num_seqs,
            self.right_dataset.num_seqs
        ]
        while seq_idx >= self.chooser_index:
            # we need to choose more
            chooseRight = self.bias >= 0 and self.mixing_ratio > 0
            self.bitset_chooser.set(self.chooser_index, chooseRight)
            if self.chooser_index % 1024 == 0:
                # this works, because index_cache is a numpy array, otherwise we would need to explictly copy
                self.index_cache[self.chooser_index // 1024] = self.chooser_childindices
            dataset_index = 1 if chooseRight else 0
            chosen_dataset = self.right_dataset if chooseRight else self.left_dataset

            if self.chooser_childindices[dataset_index] % child_lens[dataset_index] == 0 and self.chooser_childindices[dataset_index] > 0:
                self.datasets_exhausted[dataset_index] = True
                print(f"MixingDataset: ({dataset_index}) exhausted", file=log.v4)
                self._print_progress()
                if self.how_to_handle_end_of_data_from_one_dataset == "exception":
                    self.is_chooser_done = True
                    raise Exception(
                        "MixingDataset: end of dataset %d %r" % (dataset_index, chosen_dataset)
                    )
                elif self.how_to_handle_end_of_data_from_one_dataset == "early_exit":
                    # the last decision is invalid (beyond the end of the dataset), 
                    # but hopefully the other functions notice that we exited early and dont use the decision ...
                    self.is_chooser_done = True
                    break
                elif self.how_to_handle_end_of_data_from_one_dataset == "wrap_around":
                    # im not sure of the logic inside the datasets and whether it keeps data that has been loaded before indefinitely,
                    # so just start loading them at the beginning again
                    if all(self.datasets_exhausted):
                        self.is_chooser_done = True
                        c0 = self.chooser_childindices[0] / max(1, child_lens[0])
                        c1 = self.chooser_childindices[1] / max(1, child_lens[1])
                        print(f"MixingDataset: optimal mixing ratio = {(self.datalens[1] / c1) / max(1, self.datalens[0]/c0 + self.datalens[1]/c1)}", file=log.v4)
                        break
                    # the modulo operator below will wrap around
                else:
                    assert False, f"{self.how_to_handle_end_of_data_from_one_dataset} not implemented"

            self._make_sure_idx_is_loaded_in_child_ds(dataset_index, self.chooser_childindices[dataset_index] % child_lens[dataset_index])
            datalen = MixingDataset._data_metric(chosen_dataset.get_data(
                self.chooser_childindices[dataset_index] % child_lens[dataset_index], self.data_key
            ))
            #print(f"({dataset_index}) datalen={datalen} shape={data.shape}")
            self.bias -= (
                (1 - self.mixing_ratio) if chooseRight else -self.mixing_ratio
            ) * max(datalen, 1)
            self.datalens[dataset_index] += datalen
            self.chooser_childindices[dataset_index] += 1
            self.chooser_index += 1

        assert not math.isnan(self.bias) and not math.isinf(
            self.bias
        )  # this should never ever happen

        if self.is_chooser_done:
            return None
        return (self.chooser_childindices[0], self.chooser_childindices[1])

    @lru_cache(maxsize=500)
    def _get_childindices_at_seq_idx(self, seq_idx):
        """
        May return None if we could not progress to the desired seq_idx.
        """
        if seq_idx < 0:
            raise Exception("seq_idx < 0")
        if seq_idx >= self.chooser_index:
            ran_ids = self._run_seq_idx(seq_idx)
            if seq_idx >= self.chooser_index:
                return None # we could not progress to the desired seq_idx, maybe early exit or exhaustion?
            return (ran_ids[0] % self.left_dataset.num_seqs, ran_ids[1] % self.right_dataset.num_seqs)
        # maybe in cache? this should happen often when we go over the dataset sequentially
        restore_from_idx = seq_idx - (seq_idx % 1024)
        restore_indices = self.index_cache[restore_from_idx // 1024]
        for try_seq in range(seq_idx, max(seq_idx - 20, restore_from_idx), -1):
            result = self._get_childindices_at_seq_idx.cache_peek(try_seq)
            if result is not None:
                restore_from_idx = try_seq
                restore_indices = result
                break
        # convert to list to avoid changin the index cache elements
        restore_indices = list(restore_indices)

        # replay the steps
        while restore_from_idx < seq_idx:
            if self.bitset_chooser.get(restore_from_idx):
                restore_indices[1] += 1 # right
            else:
                restore_indices[0] += 1 # left
            restore_from_idx += 1

        return (restore_indices[0] % self.left_dataset.num_seqs, restore_indices[1] % self.right_dataset.num_seqs)

    def _get_dataset_and_childindex_at_seq_idx(self, seq_idx):
        indices = self._get_childindices_at_seq_idx(seq_idx)
        assert indices is not None
        choose_right = self.bitset_chooser.get(seq_idx)
        dataset_index = 1 if choose_right else 0
        return dataset_index, indices[1 if choose_right else 0]

    def _collect_single_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        dataset_idx, dataset_seq_idx = self._get_dataset_and_childindex_at_seq_idx(seq_idx)
        dataset = self.left_dataset if dataset_idx == 0 else self.right_dataset
        self._make_sure_idx_is_loaded_in_child_ds(dataset_idx, dataset_seq_idx)
        seq_tag = dataset.get_tag(dataset_seq_idx)
        features = {
            k: dataset.get_data(dataset_seq_idx, k) for k in dataset.get_data_keys()
        }
        return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features)

    def is_less_than_num_seqs(self, seq_idx: int):
        if seq_idx < self.chooser_index:
            return True
        if self.is_chooser_done:
            return False
        ids = self._get_childindices_at_seq_idx(seq_idx)
        return ids is not None

    @property
    def num_seqs(self):
        """
        :rtype: int
        """
        if self.is_chooser_done:
            return self.chooser_index
        # we can calculate this, but its very expensive! TODO what do?
        raise Exception("num_seqs not known yet")

    def get_target_list(self):
        """
        :rtype: list[str]
        """
        return self.control_dataset.get_target_list()
    
    def get_data_keys(self) -> List[str]:
        """data keys"""
        return self.control_dataset.get_data_keys()

    def _print_progress(self):
        if self.left_dataset.num_seqs > 0:
            print(f"MixingDataset: Left dataset: {self.chooser_childindices[0]}/{self.left_dataset.num_seqs} ({self.chooser_childindices[0] / self.left_dataset.num_seqs * 100}%) exhausted={self.datasets_exhausted[0]}, avg_datalen={self.datalens[0]/max(1, self.chooser_childindices[0])}", file=log.v4)
        else:
            print("MixingDataset: Left dataset: empty", file=log.v4)
        if self.right_dataset.num_seqs > 0:
            print(f"MixingDataset: Right dataset: {self.chooser_childindices[1]}/{self.right_dataset.num_seqs} ({self.chooser_childindices[1] / self.right_dataset.num_seqs * 100}%) exhausted={self.datasets_exhausted[1]}, avg_datalen={self.datalens[1]/max(1, self.chooser_childindices[1])}", file=log.v4)
        else:
            print("MixingDataset: Right dataset: empty", file=log.v4)

    def finish_epoch(self, *, free_resources: bool = False):
        """finish epoch"""
        super().finish_epoch(free_resources=free_resources)
        print("MixingDataset: finishing epoch! Datasets:", file=log.v4)
        self._print_progress()

        self.left_dataset.finish_epoch(free_resources=free_resources)
        self.right_dataset.finish_epoch(free_resources=free_resources)

    def get_data_dim(self, key: str) -> int:
        """data dim"""
        return self.control_dataset.get_data_dim(key)

    def get_data_shape(self, data_key: str) -> List[int]:
        """data shape"""
        return self.control_dataset.get_data_shape(data_key)

    def get_data_dtype(self, key: str) -> str:
        """data dtype"""
        return self.control_dataset.get_data_dtype(key)

    def is_data_sparse(self, key: str) -> bool:
        """is data sparse"""
        return self.control_dataset.is_data_sparse(key)
    
    def get_epoch_continuous(self, sorted_seq_idx: int) -> float:
        assert self.left_dataset.num_seqs > 0 and self.right_dataset.num_seqs > 0
        indices = self._get_childindices_at_seq_idx(sorted_seq_idx)
        if indices is None:
            return 1.0 # we are done
        frac_left = indices[0] / self.left_dataset.num_seqs
        frac_right = indices[1] / self.right_dataset.num_seqs
        if self.how_to_handle_end_of_data_from_one_dataset == "wrap_around":
            return min(frac_left, frac_right)
        # "early_exit" or "exception"
        return max(frac_left, frac_right)
    
    # TODO implement is_less_than_num_seqs
