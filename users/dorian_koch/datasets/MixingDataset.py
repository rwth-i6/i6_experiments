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
from decimal import Decimal


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


THandleEndOfData = Literal["exception", "wrap_around", "early_exit"]

'''
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
        how_to_handle_end_of_data: Optional[List[THandleEndOfData]] = None,
        how_to_handle_end_of_data_from_one_dataset: Optional[THandleEndOfData] = "wrap_around",  # deprecated
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
        :param how_to_handle_end_of_data_from_one_dataset: (deprecated) what to do when one dataset is exhausted
            exception: raise an exception, this should practically never be used in training
            wrap_around: wrap around to the beginning of the dataset that is exhausted. Terminate when both datasets have terminated at least once.
            early_exit: end epoch when one dataset has been exhausted
        :param control_dataset: which dataset is used for i.e. `get_data_dtype`
        """
        super().__init__(**kwargs)
        assert 0.0 <= mixing_ratio <= 1.0
        self.mixing_ratio = mixing_ratio
        if how_to_handle_end_of_data_from_one_dataset is not None:
            assert how_to_handle_end_of_data is None
            how_to_handle_end_of_data = [how_to_handle_end_of_data_from_one_dataset] * 2
        else:
            assert how_to_handle_end_of_data is not None
        assert len(how_to_handle_end_of_data) == 2

        self.how_to_handle_end_of_data = how_to_handle_end_of_data
        self.left_dataset = init_dataset(left_dataset, parent_dataset=self)
        self.right_dataset = init_dataset(right_dataset, parent_dataset=self)
        self.control_dataset = self.left_dataset if control_dataset == "left" else self.right_dataset
        self.num_inputs = make_hashable(
            self.left_dataset.num_inputs
        )  # make_hashable normalizes lists/tuples to just tuples
        self.num_outputs = make_hashable(self.left_dataset.num_outputs)
        self.labels = self.left_dataset.labels
        self.data_key = data_key

        assert make_hashable(self.right_dataset.num_inputs) == self.num_inputs
        assert make_hashable(self.right_dataset.num_outputs) == self.num_outputs
        self._reset_params()

    def _reset_params(self):
        assert not (0 < self.right_dataset.num_seqs < 10) and not (
            0 < self.left_dataset.num_seqs < 10
        ), "mixing can go wrong when one dataset has very few seqs"
        # left terminates epoch
        lff = self.left_dataset.num_seqs * (1 + (self.mixing_ratio) / (1 - self.mixing_ratio))
        # right terminates epoch
        rff = self.right_dataset.num_seqs * (1 + (1 - self.mixing_ratio) / (self.mixing_ratio))
        finish_seqs_arr = [lff, rff]

        assert len(self.how_to_handle_end_of_data) == 2, "this logic goes wrong with more than two datasets"

        if all(how in ["exception", "early_exit"] for how in self.how_to_handle_end_of_data):
            # epoch terminates if any dataset finishes
            self.total_num_seqs_upper_bound = math.ceil(min(finish_seqs_arr))
        elif all(how == "wrap_around" for how in self.how_to_handle_end_of_data):
            # epoch terminates if all datasets finish
            self.total_num_seqs_upper_bound = math.ceil(max(finish_seqs_arr))
        else:  # mix
            for i, how in enumerate(self.how_to_handle_end_of_data):
                if how == "wrap_around":
                    finish_seqs_arr[i] = float("inf")
            self.total_num_seqs_upper_bound = math.ceil(min(finish_seqs_arr))

        assert not math.isnan(self.total_num_seqs_upper_bound) and not math.isinf(self.total_num_seqs_upper_bound)
        # for good measure
        self.total_num_seqs_upper_bound += 10
        self.total_num_seqs_upper_bound *= 2

        if self.total_num_seqs_upper_bound > 0:
            print(
                f"MixingDataset init: {self.left_dataset.num_seqs} + {self.right_dataset.num_seqs}, upperbound={self.total_num_seqs_upper_bound}, mixingratio={self.mixing_ratio}",
                file=log.v4,
            )
        else:
            print("MixingDataset init: both datasets are empty", file=log.v4)
        self._estimated_num_seqs = self.total_num_seqs_upper_bound
        assert self.total_num_seqs_upper_bound < 2**31, "sequences do not fit into int32"
        # 0 means left, 1 means right
        self.bitset_chooser = Bitarray(self.total_num_seqs_upper_bound)
        # cache indices to both datasets at a particular sequence index
        self.index_cache = numpy.zeros(((self.total_num_seqs_upper_bound + 1023) // 1024, 2), dtype=numpy.int32)
        # up until which point we have chosen
        self.chooser_index = 0
        self.is_chooser_done = False
        self.chooser_childindices = [0, 0]
        self.datasets_exhausted = [False, False]
        self.datasets_loaded_until = [0, 0]  # we need to _load_seqs the datasets
        # we will get out of balance while choosing, we will correct this by biasing the next choice
        self.bias = 0.0
        self.datalens = [0, 0]
        self._get_raw_childindices_at_seq_idx.cache_clear()

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :type epoch: int|None
        :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
        :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
        """
        need_reinit = self.epoch is None or self.epoch != epoch
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        if not need_reinit:
            return False

        if seq_order is not None:
            raise NotImplementedError("Predefined order via sequence indices for MixingDataset")
        if seq_list is not None:
            raise NotImplementedError("Predefined order via sequence tags for MixingDataset")
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
        assert (
            seq_idx < self.total_num_seqs_upper_bound
        ), "This assert fails only when the two datasets are very unbalanced, in the sense that one dataset has many long sequences while the other mostly has shorter once. Keep them on equal lengths on average please! Otherwise you need to somehow increase this upper bound (which will not cause issues, just eat more ram)"
        if self.is_chooser_done:
            raise Exception(
                "chooser is done. change attribute 'how_to_handle_end_of_data' to 'exception' if you want to know why (probably because early_exit)"
            )

        child_lens = [self.left_dataset.num_seqs, self.right_dataset.num_seqs]
        while seq_idx >= self.chooser_index:
            # we need to choose more
            chooseRight = self.bias >= 0 and self.mixing_ratio > 0
            self.bitset_chooser.set(self.chooser_index, chooseRight)
            if self.chooser_index % 1024 == 0:
                # this works, because index_cache is a numpy array, otherwise we would need to explictly copy
                self.index_cache[self.chooser_index // 1024] = self.chooser_childindices
            dataset_index = 1 if chooseRight else 0
            chosen_dataset = self.right_dataset if chooseRight else self.left_dataset

            if (
                self.chooser_childindices[dataset_index] % child_lens[dataset_index] == 0
                and self.chooser_childindices[dataset_index] > 0
            ):
                self.datasets_exhausted[dataset_index] = True
                print(f"MixingDataset: ({dataset_index}) exhausted", file=log.v4)
                self._print_progress()
                c0 = self.chooser_childindices[0] / max(1, child_lens[0])
                c1 = self.chooser_childindices[1] / max(1, child_lens[1])
                print(
                    f"MixingDataset: optimal mixing ratio = {(self.datalens[1] / c1) / max(1, self.datalens[0]/c0 + self.datalens[1]/c1)} (assuming uniform random distribution)",
                    file=log.v4,
                )
                if self.how_to_handle_end_of_data[dataset_index] == "exception":
                    self.is_chooser_done = True
                    raise Exception("MixingDataset: end of dataset %d %r" % (dataset_index, chosen_dataset))
                elif self.how_to_handle_end_of_data[dataset_index] == "early_exit":
                    # the last decision is invalid (beyond the end of the dataset),
                    # but hopefully the other functions notice that we exited early and dont use the decision ...
                    self.is_chooser_done = True
                    break
                elif self.how_to_handle_end_of_data[dataset_index] == "wrap_around":
                    # im not sure of the logic inside the datasets and whether it keeps data that has been loaded before indefinitely,
                    # so just start loading them at the beginning again
                    if all(self.datasets_exhausted):
                        self.is_chooser_done = True
                        break
                    # the modulo operator below will wrap around
                else:
                    assert False, f"{self.how_to_handle_end_of_data[dataset_index]} not implemented"

            self._make_sure_idx_is_loaded_in_child_ds(
                dataset_index, self.chooser_childindices[dataset_index] % child_lens[dataset_index]
            )
            datalen = MixingDataset._data_metric(
                chosen_dataset.get_data(
                    self.chooser_childindices[dataset_index] % child_lens[dataset_index], self.data_key
                )
            )
            # print(f"({dataset_index}) datalen={datalen} shape={data.shape}")
            self.bias -= ((1 - self.mixing_ratio) if chooseRight else -self.mixing_ratio) * max(datalen, 1)
            self.datalens[dataset_index] += datalen
            self.chooser_childindices[dataset_index] += 1
            self.chooser_index += 1

        assert not math.isnan(self.bias) and not math.isinf(self.bias)  # this should never ever happen

        if self.is_chooser_done:
            return None
        return (self.chooser_childindices[0], self.chooser_childindices[1])

    @lru_cache(maxsize=500)
    def _get_raw_childindices_at_seq_idx(self, seq_idx):
        """
        May return None if we could not progress to the desired seq_idx.
        """
        if seq_idx < 0:
            raise Exception("seq_idx < 0")
        if seq_idx >= self.chooser_index:
            ran_ids = self._run_seq_idx(seq_idx)
            if seq_idx >= self.chooser_index or ran_ids is None:
                return None  # we could not progress to the desired seq_idx, maybe early exit or exhaustion?

            assert self.chooser_index == seq_idx + 1
            # reverse last decision to get actual indices
            if self.bitset_chooser.get(seq_idx):
                assert ran_ids[1] > 0
                return (ran_ids[0], ran_ids[1] - 1)
            else:
                assert ran_ids[0] > 0
                return (ran_ids[0] - 1, ran_ids[1])
        # maybe in cache? this should happen often when we go over the dataset sequentially
        restore_from_idx = seq_idx - (seq_idx % 1024)
        restore_indices = self.index_cache[restore_from_idx // 1024]
        for try_seq in range(seq_idx, max(seq_idx - 20, restore_from_idx), -1):
            result = self._get_raw_childindices_at_seq_idx.cache_peek(try_seq)
            if result is not None:
                restore_from_idx = try_seq
                restore_indices = result
                break
        # convert to list to avoid changing the index cache elements
        restore_indices = list(restore_indices)

        # replay the steps
        while restore_from_idx < seq_idx:
            if self.bitset_chooser.get(restore_from_idx):
                restore_indices[1] += 1  # right
            else:
                restore_indices[0] += 1  # left
            restore_from_idx += 1

        return (restore_indices[0], restore_indices[1])

    def _get_childindices_at_seq_idx(self, seq_idx):
        result = self._get_raw_childindices_at_seq_idx(seq_idx)
        if result is None:
            return result
        return (result[0] % self.left_dataset.num_seqs, result[1] % self.right_dataset.num_seqs)

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
        features = {k: dataset.get_data(dataset_seq_idx, k) for k in dataset.get_data_keys()}
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
        raise NotImplementedError("num_seqs not known yet")

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
            print(
                f"MixingDataset: Left dataset: {self.chooser_childindices[0]}/{self.left_dataset.num_seqs} ({self.chooser_childindices[0] / self.left_dataset.num_seqs * 100}%) exhausted={self.datasets_exhausted[0]}, avg_datalen={self.datalens[0]/max(1, self.chooser_childindices[0])}",
                file=log.v4,
            )
        else:
            print("MixingDataset: Left dataset: empty", file=log.v4)
        if self.right_dataset.num_seqs > 0:
            print(
                f"MixingDataset: Right dataset: {self.chooser_childindices[1]}/{self.right_dataset.num_seqs} ({self.chooser_childindices[1] / self.right_dataset.num_seqs * 100}%) exhausted={self.datasets_exhausted[1]}, avg_datalen={self.datalens[1]/max(1, self.chooser_childindices[1])}",
                file=log.v4,
            )
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

    def get_complete_frac(
        self, sorted_seq_idx: int, *, allow_only_lr_suitable: bool = False, **kwargs
    ) -> Optional[float]:
        assert self.left_dataset.num_seqs > 0 and self.right_dataset.num_seqs > 0
        indices = self._get_raw_childindices_at_seq_idx(sorted_seq_idx)
        if indices is None:
            return 1.0  # we are done
        frac_left = indices[0] / self.left_dataset.num_seqs
        frac_right = indices[1] / self.right_dataset.num_seqs
        fracs = [frac_left, frac_right]
        if all(how == "wrap_around" for how in self.how_to_handle_end_of_data):
            return min(fracs)
        if all(how in ["exception", "early_exit"] for how in self.how_to_handle_end_of_data):
            return min(1.0, max(fracs))
        # mix
        for i, how in enumerate(self.how_to_handle_end_of_data):
            if how == "wrap_around":
                fracs[i] = 0.0
        return min(1.0, max(fracs))
'''


class MixingDataset2(CachedDataset2):
    """
    This mixes multiple datasets. They are expected to provide the same data-keys and data-dimensions.
    They must not share the same sequence tags. # TODO why?

    Sequences are chosen based on mixing factors. The docstring for __init__ explains how to set them correctly.
    The % is based on the size of the data, not the amount of sequences. TODO it should be possible to change this behaviour
    So even with a 50/50 split, it may be possible to get a minibatch with a single long sequence from the first dataset and many short sequences from the second dataset.

    All datasets work in steplock, meaning that they are at the same epoch number at all times.
    A logical conclusion of this is that either some datasets will not be fully used (skipped to next epoch before all data could be read),
    or that the data from the other dataset is reused (index wraps around back to the beginning)
    This behaviour can be controlled via `how_to_handle_end_of_data`
    """

    def __init__(
        self,
        datasets: Dict[str, Dict[str, Any]],
        mixing_ratios: Dict[str, float],
        how_to_handle_end_of_data: Dict[str, THandleEndOfData],  # TODO maybe rename to how_to_handle_end_of_epoch
        *,
        data_key: str = "data",
        control_dataset: Optional[str] = None,
        **kwargs,
    ):
        """
        :param datasets
        :param mixing_ratios: how much to sample from a specific dataset
            Lets say we have mixing_ratios = {"a": 0.5, "b": 1, c: "3"}
            then for every one unit of data we sample from a, we would on average see 2 units from b and 6 units from c
            How much a unit is depends on what we want to consider as data size, see function `_data_metric` below
        :param data_key: key for the mixing process, mixing considers the size of this data key only
        :param how_to_handle_end_of_data: what to do when a child dataset is exhausted (reached the end of its epoch)
            exception: raise an exception (not very useful)
            wrap_around: wrap around to the beginning of the dataset that is exhausted.
                If all datasets have this property, then the MixinDataset epoch ends when all child datasets have wrapped around at least once
            early_exit: end MixingDataset epoch when this dataset has been exhausted
        :param control_dataset: which dataset is used for i.e. `get_data_dtype`
        """
        super().__init__(**kwargs)
        assert len(datasets) > 1
        assert datasets.keys() == mixing_ratios.keys()
        assert datasets.keys() == how_to_handle_end_of_data.keys()
        # TODO we could allow mixing_ratio = 0 with some additional logic, but I don't think this is necessary to implement
        assert all([0.0 < mixing_ratio for mixing_ratio in mixing_ratios.values()])
        # TODO a sum = 1 would have a nice probabilistic interpretation, but maybe this is too annoying for a user to calculate
        # imagine 5 different mixing ratios, and having to adjust all even when you only want to adjust the proportion of a single dataset
        assert sum(mixing_ratios.values()) > 0
        assert len(how_to_handle_end_of_data) == len(datasets)

        self.dataset_name_to_idx = {}
        self.datasets = []
        self.mixing_ratios = []
        self.how_to_handle_end_of_data = []
        for name, dataset in datasets.items():
            self.dataset_name_to_idx[name] = len(self.datasets)
            self.datasets.append(init_dataset(dataset, parent_dataset=self))
            self.mixing_ratios.append(Decimal(mixing_ratios[name]))
            self.how_to_handle_end_of_data.append(how_to_handle_end_of_data[name])

        if control_dataset is not None:
            self.control_dataset = self.datasets[control_dataset]
        else:
            self.control_dataset = self.datasets[0]
        self.num_inputs = make_hashable(
            self.control_dataset.num_inputs
        )  # make_hashable normalizes lists/tuples to just tuples
        self.num_outputs = make_hashable(self.control_dataset.num_outputs)
        self.labels = self.control_dataset.labels
        self.data_key = data_key
        self._last_decision = None

        for ds in self.datasets:  # TODO maybe we can relax these restrictions to <=
            assert self.num_inputs == make_hashable(ds.num_inputs)
            assert self.num_outputs == make_hashable(ds.num_outputs)
        self._reset_params()

    def _reset_params(self):
        assert all(
            [not (0 < ds.num_seqs < 10) for ds in self.datasets]
        ), "mixing can go wrong when one dataset has very few seqs"

        # here we estimate num_seqs of this MixingDataset, we don't really need this but it has been helpful for debugging
        # we consider each child dataset individually, and calculate how much total num_seqs we have seen when the child dataset has exhausted for the first time
        # this estimate assumes that the average `_data_metric` of all datasets is equal, which is not always true in practice
        try:
            finish_seqs_arr = []
            for i, ds in enumerate(self.datasets):
                completion_frac_of_other_datasets = 0
                for i2, ds2 in enumerate(self.datasets):
                    if i == i2:
                        continue
                    completion_frac_of_other_datasets += self.mixing_ratios[i2] / self.mixing_ratios[i]
                est = ds.num_seqs * (1 + completion_frac_of_other_datasets)
                finish_seqs_arr.append(est)

            num_seqs_estimate = 0
            if all(how in ["exception", "early_exit"] for how in self.how_to_handle_end_of_data):
                # epoch terminates if any dataset finishes
                num_seqs_estimate = math.ceil(min(finish_seqs_arr))
            elif all(how == "wrap_around" for how in self.how_to_handle_end_of_data):
                # epoch terminates if all datasets finish
                num_seqs_estimate = math.ceil(max(finish_seqs_arr))
            else:  # mix
                for i, how in enumerate(self.how_to_handle_end_of_data):
                    if how == "wrap_around":  # this dataset will never cause an epoch end
                        finish_seqs_arr[i] = float("inf")
                num_seqs_estimate = math.ceil(min(finish_seqs_arr))

            assert not math.isnan(num_seqs_estimate) and not math.isinf(num_seqs_estimate)
            self._estimated_num_seqs = num_seqs_estimate
        except NotImplementedError:
            pass  # num_seqs not implemented
        print(
            f"MixingDataset init: {" + ".join([ds.num_seqs for ds in self.datasets])}, _estimated_num_seqs={self._estimated_num_seqs}, mixingratios={self.mixing_ratios}",
            file=log.v4,
        )

        assert (
            self._estimated_num_seqs is None or self._estimated_num_seqs < 2**31
        ), "unreasonably large num_seqs estimate, adjust mixing ratios and/or `how_to_handle_end_of_data`"  # TODO do we still need this assert?

        # up until which point we have chosen
        self.chooser_index = 0
        self.is_chooser_done = False
        # the indices where the chooser loop will continue
        self.chooser_childindices = [0] * len(self.datasets)
        self.datasets_exhausted = [False] * len(self.datasets)
        # we need to _load_seqs the datasets TODO i think we can remove this if we only ever load one seq in advance
        self.datasets_loaded_until = [0] * len(self.datasets)

        # TODO the name `bias` can be misleading, find a better name
        self.bias = [Decimal(0.0)] * len(self.datasets)
        # total number of data units used from each dataset
        self.datalens = [0] * len(self.datasets)
        self._get_raw_childindices_and_decision_at_seq_idx.cache_clear()

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :type epoch: int|None
        :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
        :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
        """
        need_reinit = self.epoch is None or self.epoch != epoch
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        if not need_reinit:
            return False

        if seq_order is not None:
            raise NotImplementedError("Predefined order via sequence indices for MixingDataset")
        if seq_list is not None:
            raise NotImplementedError("Predefined order via sequence tags for MixingDataset")
        elif self.seq_ordering != "default":
            raise NotImplementedError("seq_ordering %s" % self.seq_ordering)

        for ds in self.datasets:
            ds.init_seq_order(epoch=epoch)
        self._reset_params()
        return True

    def _data_metric(self, v: numpy.ndarray):
        return Decimal(v.shape[0] if v.ndim >= 1 else 1)

    def _make_sure_idx_is_loaded_in_child_ds(self, dataset_index, seq_idx):
        """
        This makes sure that we can access the data at seq_idx in the child dataset
        """
        chosen_dataset = self.datasets[dataset_index]
        child_len = chosen_dataset.num_seqs

        # TODO this is needed because CachedDatasets wants to be accessed in a strictly increasing (iterator-like) order
        # therefore we need to reinitialize it before we start accessing from the beginning again
        if hasattr(chosen_dataset, "expected_load_seq_start") and seq_idx < chosen_dataset.expected_load_seq_start:
            prev_num_seqs = chosen_dataset.num_seqs
            chosen_dataset.init_seq_order(epoch=self.epoch)
            assert (
                chosen_dataset.num_seqs == prev_num_seqs
            ), "data in ds has changed even though we reinitialized the same epoch. this code is only supposed to reset the state back to index zero, not change the data"
            self.datasets_loaded_until[dataset_index] = 0

        if self.datasets_loaded_until[dataset_index] <= seq_idx:
            # loading up to 512 seqs because why not TODO figure out if this actually improves performance
            start = seq_idx
            end = (seq_idx + min(child_len - 1, 512)) % child_len

            if end < start:  # end exceeds num_seqs of the child dataset, so only load until the end
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
        assert not self.is_chooser_done

        child_lens = [ds.num_seqs for ds in self.datasets]
        while seq_idx >= self.chooser_index:
            # we need to choose
            dataset_index = max(range(len(self.bias)), key=self.bias.__getitem__)  # dataset idx with highest bias
            self._last_decision = dataset_index
            chosen_dataset = self.datasets[dataset_index]

            if (
                self.chooser_childindices[dataset_index] % child_lens[dataset_index] == 0
                and self.chooser_childindices[dataset_index] > 0
            ):
                self.datasets_exhausted[dataset_index] = True
                print(f"MixingDataset: ({dataset_index}) exhausted", file=log.v4)
                self._print_progress()
                """
                TODO: implement this optimal mixing ratio calc for more than two datasets
                c0 = self.chooser_childindices[0] / max(1, child_lens[0])
                c1 = self.chooser_childindices[1] / max(1, child_lens[1])
                print(
                    f"MixingDataset: optimal mixing ratio = {(self.datalens[1] / c1) / max(1, self.datalens[0]/c0 + self.datalens[1]/c1)} (assuming uniform random distribution)",
                    file=log.v4,
                )
                """
                if self.how_to_handle_end_of_data[dataset_index] == "exception":
                    self.is_chooser_done = True
                    self._last_decision = None
                    raise Exception("MixingDataset: end of dataset %d %r" % (dataset_index, chosen_dataset))
                elif self.how_to_handle_end_of_data[dataset_index] == "early_exit":
                    # the last decision can not be used to get more data (index is beyond the end of the dataset),
                    # so we break here
                    self.is_chooser_done = True
                    self._last_decision = None
                    break
                elif self.how_to_handle_end_of_data[dataset_index] == "wrap_around":
                    if all(self.datasets_exhausted):  # this only happens when all datasets are set to `wrap_around`
                        self.is_chooser_done = True
                        self._last_decision = None
                        break
                else:
                    assert False, f"{self.how_to_handle_end_of_data[dataset_index]} not implemented"

            self._make_sure_idx_is_loaded_in_child_ds(
                dataset_index, self.chooser_childindices[dataset_index] % child_lens[dataset_index]
            )
            datalen = self._data_metric(
                chosen_dataset.get_data(
                    self.chooser_childindices[dataset_index] % child_lens[dataset_index], self.data_key
                )
            )
            # print(f"({dataset_index}) datalen={datalen} shape={data.shape}")
            self.bias[dataset_index] -= max(datalen, 1) / self.mixing_ratios[dataset_index]
            assert not math.isnan(self.bias[dataset_index]) and not math.isinf(
                self.bias[dataset_index]
            )  # this should never ever happen
            if self.bias[dataset_index] < -100:  # -100 is arbitrary
                # if we don't shift back to 0 our bias will go towards negative infinity
                adjust = min(self.bias)
                self.bias = [b - adjust for b in bias]
            self.datalens[dataset_index] += datalen
            self.chooser_childindices[dataset_index] += 1
            self.chooser_index += 1

        if self.is_chooser_done:
            return None  # we exhausted before we could reach the seq_idx

        # note: these could exceed num_seqs of the child datasets (use modulo)
        return tuple(self.chooser_childindices)

    @lru_cache(maxsize=1)
    def _get_raw_childindices_and_decision_at_seq_idx(self, seq_idx):
        """
        May return None if we could not progress to the desired seq_idx.
        """
        assert seq_idx >= 0
        assert seq_idx >= self.chooser_index, "you may only access data iterator-like (no random access)"

        ran_ids = self._run_seq_idx(seq_idx)
        if seq_idx >= self.chooser_index or ran_ids is None:
            return None, None  # we could not progress to the desired seq_idx, epoch ends here

        assert self.chooser_index == seq_idx + 1
        # reverse last decision to get actual indices
        idxs = list(ran_ids)
        idxs[self._last_decision] -= 1
        return tuple(idxs), self._last_decision

    def _get_dataset_and_childindex_at_seq_idx(self, seq_idx):
        result, decision = self._get_raw_childindices_and_decision_at_seq_idx(seq_idx)
        assert result is not None and decision is not None

        return decision, result[decision] % self.datasets[decision].num_seqs

    def _collect_single_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        dataset_idx, dataset_seq_idx = self._get_dataset_and_childindex_at_seq_idx(seq_idx)
        self._make_sure_idx_is_loaded_in_child_ds(dataset_idx, dataset_seq_idx)
        dataset = self.datasets[dataset_idx]
        seq_tag = dataset.get_tag(dataset_seq_idx)
        features = {k: dataset.get_data(dataset_seq_idx, k) for k in dataset.get_data_keys()}
        return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features)

    def is_less_than_num_seqs(self, seq_idx: int):
        if seq_idx < self.chooser_index:
            return True
        if self.is_chooser_done:
            return False
        # TODO this could potentially lead to a bug because the following function can only go forward
        # then we fail when this function steps too far forward and we can't get the indices when we want to access the data
        ids, decision = self._get_raw_childindices_and_decision_at_seq_idx(seq_idx)
        return ids is not None

    @property
    def num_seqs(self):
        """
        :rtype: int
        """
        if self.is_chooser_done:
            return self.chooser_index
        raise NotImplementedError()

    def get_target_list(self):
        """
        :rtype: list[str]
        """
        return self.control_dataset.get_target_list()

    def get_data_keys(self) -> List[str]:
        """data keys"""
        return self.control_dataset.get_data_keys()

    def _print_progress(self):
        for ds_name, ds_idx in self.dataset_name_to_idx.items():
            ds = self.datasets[ds_idx]
            prefix = f"MixingDataset: [{ds_idx}]'{ds_name}'"
            if ds.num_seqs > 0:
                print(
                    f"{prefix}: {self.chooser_childindices[ds_idx]}/{ds.num_seqs} ({self.chooser_childindices[ds_idx] / ds.num_seqs * 100}%) exhausted={self.datasets_exhausted[ds_idx]}, avg_datalen={self.datalens[ds_idx]/max(1, self.chooser_childindices[ds_idx])}",
                    file=log.v4,
                )
            else:
                print(f"{prefix}: empty", file=log.v4)

    def finish_epoch(self, *, free_resources: bool = False):
        """finish epoch"""
        super().finish_epoch(free_resources=free_resources)
        print("MixingDataset: finishing epoch! Datasets:", file=log.v4)
        self._print_progress()

        for ds in self.datasets:
            ds.finish_epoch(free_resources=free_resources)

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

    def get_complete_frac(
        self, sorted_seq_idx: int, *, allow_only_lr_suitable: bool = False, **kwargs
    ) -> Optional[float]:
        # we need to get the raw ones (so not wrapped around) because `datasets_exhausted` is only set to true once we actually access that index
        # so we would read 0% completion when we are actually one decision away from ending the epoch, which also breaks monotonicity of this function
        indices, decision = self._get_raw_childindices_and_decision_at_seq_idx(sorted_seq_idx)
        if indices is None:
            return 1.0  # we are done

        fracs = []
        for i, ds in enumerate(self.datasets):
            if self.datasets_exhausted[i]:
                fracs.append(1.0)
            else:
                try:
                    frac = ds.get_complete_frac(
                        max(0, indices[i] - 1), allow_only_lr_suitable=allow_only_lr_suitable, **kwargs
                    )
                    if frac is None:
                        raise NotImplementedError()
                    fracs.append(frac)
                except NotImplementedError:
                    assert ds.num_seqs > 0
                    fracs.append(max(0, indices[i] - 1) / ds.num_seqs)
        if all(how == "wrap_around" for how in self.how_to_handle_end_of_data):
            return min(fracs)
        if all(how in ["exception", "early_exit"] for how in self.how_to_handle_end_of_data):
            return min(1.0, max(fracs))
        # mix
        for i, how in enumerate(self.how_to_handle_end_of_data):
            if how == "wrap_around":
                fracs[i] = 0.0
        return min(1.0, max(fracs))
