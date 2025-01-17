from __future__ import annotations

from typing import Literal, Optional, Union, Any, Callable, Sequence, List, Dict, Tuple
from returnn.datasets.basic import Dataset, DatasetSeq, init_dataset, convert_data_dims
from returnn.datasets.cached2 import CachedDataset2
import returnn.util.basic as util
from returnn.util.basic import NumbersDict, load_json, OptionalNotImplementedError
from returnn.log import log
from random import Random
import numpy
import sys
import typing
from i6_experiments.users.zeyer.utils.lru_cache import lru_cache
import math


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
    """

    def __init__(
        self,
        left_dataset: Dict[str, Any],
        right_dataset: Dict[str, Any],
        mixing_ratio: float = 0.5,
        how_to_handle_end_of_data_from_one_dataset: Union[Literal["exception"], Literal["wrap_around"], Literal["early_exit"]] = "wrap_around",
        *,
        data_key: str = "data",
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
        """
        super().__init__(**kwargs)
        assert 0.0 <= mixing_ratio <= 1.0
        self.mixing_ratio = mixing_ratio
        self.how_to_handle_end_of_data_from_one_dataset = how_to_handle_end_of_data_from_one_dataset
        self.left_dataset = init_dataset(left_dataset, parent_dataset=self)
        self.right_dataset = init_dataset(right_dataset, parent_dataset=self)
        self.num_inputs = self.left_dataset.num_inputs
        self.num_outputs = self.left_dataset.num_outputs
        self.labels = self.left_dataset.labels
        self.data_key = data_key

        assert self.right_dataset.num_inputs == self.num_inputs
        assert self.right_dataset.num_outputs == self.num_outputs
        self._reset_params()

    def _reset_params(self):
        # TODO fix this this upper bound
        self.total_num_seqs_upper_bound = self.left_dataset.num_seqs + self.right_dataset.num_seqs
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
        self._reset_params()
        return True

    def _load_seqs(self, start, end):
        """
        :param int start:
        :param int end:
        """
        end_indices = self._get_childindices_at_seq_idx(end)
        start_indices = self._get_childindices_at_seq_idx(start)

        assert end_indices is not None
        assert start_indices is not None

        if self.datasets_loaded_until[0] <= end_indices[0]:
            load_until = min(self.left_dataset.num_seqs, end_indices[0] + 1)
            self.left_dataset.load_seqs(start_indices[0], load_until)
            self.datasets_loaded_until[0] = load_until
        if self.datasets_loaded_until[1] <= end_indices[1]:
            load_until = min(self.right_dataset.num_seqs, end_indices[1] + 1)
            self.right_dataset.load_seqs(start_indices[1], load_until)
            self.datasets_loaded_until[1] = load_until

        super()._load_seqs(start=start, end=end)

    def _run_seq_idx(self, seq_idx):
        if seq_idx < self.chooser_index:
            raise Exception("seq_idx < chooser_index")
        assert seq_idx < self.total_num_seqs_upper_bound
        if self.is_chooser_done:
            raise Exception("chooser is done. change attribute 'how_to_handle_end_of_data_from_one_dataset' to 'exception' if you want to know why (probably because early_exit)")
        # get old childindices
        child_indices = self.chooser_childindices
        child_lens = [
            self.left_dataset.num_seqs,
            self.right_dataset.num_seqs
        ]
        while seq_idx >= self.chooser_index:
            # we need to choose more
            chooseRight = self.bias >= 0 and self.mixing_ratio > 0
            self.bitset_chooser.set(self.chooser_index, chooseRight)
            if self.chooser_index % 1024 == 0:
                self.index_cache[self.chooser_index // 1024] = child_indices
            dataset_index = 1 if chooseRight else 0
            chosen_dataset = self.right_dataset if chooseRight else self.left_dataset

            if child_indices[dataset_index] >= chosen_dataset.num_seqs:
                self.datasets_exhausted[dataset_index] = True
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
                    self.datasets_loaded_until[dataset_index] = 0
                    if all(self.datasets_exhausted):
                        self.is_chooser_done = True
                        break
                    # the modulo operator below will wrap around

            if self.datasets_loaded_until[dataset_index] <= child_indices[dataset_index] % child_lens[dataset_index]:
                # 512 is just some arbitrary number
                start = child_indices[dataset_index] % child_lens[dataset_index]
                end = (child_indices[dataset_index] + 512) % child_lens[dataset_index]
                if end < start:
                    chosen_dataset.load_seqs(start, child_lens[dataset_index])
                    self.datasets_loaded_until[dataset_index] = child_lens[dataset_index]
                    assert self.datasets_loaded_until[dataset_index] >= child_indices[dataset_index]
                    # not sure if we should also load from 0 to end here, it may erase the data from start to child_lens? idk
                else:
                    chosen_dataset.load_seqs(start, end)
                    self.datasets_loaded_until[dataset_index] = end
            datalen = chosen_dataset.get_data(
                child_indices[dataset_index] % child_lens[dataset_index], self.data_key
            ).shape[0]
            self.bias -= (
                (1 - self.mixing_ratio) if chooseRight else self.mixing_ratio
            ) * datalen
            child_indices[dataset_index] += 1
            self.chooser_index += 1

        assert not math.isnan(self.bias) and not math.isinf(
            self.bias
        )  # this should never ever happen

        self.chooser_childindices = child_indices
        return child_indices

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
            return ran_ids
        # maybe in cache? this should happen often when we go over the dataset sequentially
        restore_from_idx = seq_idx - (seq_idx % 1024)
        restore_indices = self.index_cache[restore_from_idx // 1024]
        for try_seq in range(seq_idx, max(seq_idx - 20, restore_from_idx), -1):
            result = self._get_childindices_at_seq_idx.cache_peek(try_seq)
            if result is not None:
                restore_from_idx = try_seq
                restore_indices = result
                break

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
        dataset = self.right_dataset if choose_right else self.left_dataset
        return dataset, indices[1 if choose_right else 0]

    def _collect_single_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        dataset, dataset_seq_idx = self._get_dataset_and_childindex_at_seq_idx(seq_idx)
        seq_tag = dataset.get_tag(dataset_seq_idx)
        features = {
            k: dataset.get_data(dataset_seq_idx, k) for k in dataset.get_data_keys()
        }
        return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features)

    @property
    def num_seqs(self):
        """
        :rtype: int
        """
        if self.is_chooser_done:
            return self.chooser_index
        # we can calculate this, but its very expensive! TODO what do?
        raise Exception("num_seqs not known yet")

    @property
    def _estimated_num_seqs(self):
        return self.total_num_seqs_upper_bound

    def get_target_list(self):
        """
        :rtype: list[str]
        """
        return self.left_dataset.get_target_list()

    def finish_epoch(self, *, free_resources: bool = False):
        """finish epoch"""
        super().finish_epoch(free_resources=free_resources)
        print("MixingDataset: finishing epoch! Datasets:")
        print(f"Left dataset: {self.chooser_childindices[0]}/{self.left_dataset.num_seqs} ({self.chooser_childindices[0] / self.left_dataset.num_seqs * 100}%) exhausted={self.datasets_exhausted[0]}")
        print(f"Right dataset: {self.chooser_childindices[1]}/{self.right_dataset.num_seqs} ({self.chooser_childindices[1] / self.right_dataset.num_seqs * 100}%) exhausted={self.datasets_exhausted[1]}")

        self.left_dataset.finish_epoch(free_resources=free_resources)
        self.right_dataset.finish_epoch(free_resources=free_resources)
