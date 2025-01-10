from __future__ import annotations

from typing import Optional, Union, Any, Callable, Sequence, List, Dict, Tuple
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
    It will try to interleave the sequences from both datasets.
    """

    def __init__(self, left_dataset: Dict[str, Any], right_dataset: Dict[str, Any], mixing_ratio: float = 0.5, *, data_key: str = "data", **kwargs):
        """
        :param left_dataset:
        :param right_dataset:
        :param mixing_ratio: probability to choose the right dataset
        :param data_key: key for the mixing process, mixing considers the size of the data 
        """
        super(MixingDataset, self).__init__(**kwargs)
        assert 0.0 <= mixing_ratio <= 1.0
        self.left_dataset = init_dataset(left_dataset, parent_dataset=self)
        self.right_dataset = init_dataset(right_dataset, parent_dataset=self)
        self.num_inputs = self.left_dataset.num_inputs
        self.num_outputs = self.left_dataset.num_outputs
        self.labels = self.left_dataset.labels
        self.data_key = data_key
        
        assert self.right_dataset.num_inputs == self.num_inputs
        assert self.right_dataset.num_outputs == self.num_outputs

        self.total_num_seqs = self.left_dataset.num_seqs + self.right_dataset.num_seqs
        assert self.total_num_seqs < 2**31, "sequences do not fit into int32"
        # 0 means left, 1 means right        
        self.bitset_chooser = Bitarray(self.total_num_seqs)
        # cache indices to both datasets at a particular sequence index
        self.index_cache = numpy.zeros(((self.total_num_seqs + 1023) // 1024, 2), dtype=numpy.int32)
        # up until which point we have chosen
        self.chooser_index = 0
        self.chooser_childindices = [0, 0]
        # we will get out of balance while choosing, we will correct this by biasing the choice
        self.bias = 0.0

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :type epoch: int|None
        :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
        :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
        """
        need_reinit = self.epoch is None or self.epoch != epoch
        super(MixingDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        self.dataset_seq_idx_offsets = [0]
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
        return True

    '''def _get_dataset_for_seq_idx(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: int
        """
        i = 0
        assert self.dataset_seq_idx_offsets is not None
        while i < len(self.dataset_seq_idx_offsets):
            if seq_idx + self.dataset_seq_idx_offsets[i] < 0:
                return i - 1
            i += 1
        return i - 1'''

    '''def _load_seqs(self, start, end):
        """
        :param int start:
        :param int end:
        """
        assert self.dataset_seq_idx_offsets is not None
        sub_start = start
        # We maybe need to call load_seqs on several of our datasets, thus we need this loop.
        while True:
            dataset_idx = self._get_dataset_for_seq_idx(sub_start)
            dataset = self.datasets[dataset_idx]
            dataset_seq_idx_start = sub_start + self.dataset_seq_idx_offsets[dataset_idx]
            dataset_seq_idx_end = end + self.dataset_seq_idx_offsets[dataset_idx]
            dataset.load_seqs(dataset_seq_idx_start, dataset_seq_idx_end)
            if dataset.is_less_than_num_seqs(dataset_seq_idx_end):
                # We are still inside this dataset and have loaded everything.
                # Thus we can stop now.
                break
            # We have reached the end of the dataset.
            if dataset_idx + 1 == len(self.datasets):
                # We are at the last dataset.
                break
            # Continue with the next one.
            self.dataset_seq_idx_offsets[dataset_idx + 1 : dataset_idx + 2] = [
                self.dataset_seq_idx_offsets[dataset_idx] - dataset.num_seqs
            ]
            sub_start = -self.dataset_seq_idx_offsets[dataset_idx + 1]
        super(MixingDataset, self)._load_seqs(start=start, end=end)'''

    def _run_seq_idx(self, seq_idx):
        if seq_idx < self.chooser_index:
            raise Exception("seq_idx < chooser_index")
        # get old childindices
        indices = self.chooser_childindices
        while seq_idx >= self.chooser_index:
            # we need to choose more
            if self.bias >= 0:
                # choose right
                self.bitset_chooser.set(self.chooser_index, True)
                # get right data
                datalen = self.right_dataset.get_data(indices[1], self.data_key).shape[0]
                self.bias -= datalen * 1.0# TODO fix bias with correct mixing factor

                
            self.chooser_index += 1
        self.chooser_childindices = indices

    @lru_cache(maxsize=500)
    def _get_childindices_at_seq_idx(self, seq_idx):
        if seq_idx < 0:
            raise Exception("seq_idx < 0")
        if seq_idx >= self.chooser_index:
            return self._run_seq_idx(seq_idx)
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
            if self.bitset_chooser.get(restore_from_idx): # right
                restore_indices[1] += 1
            else:
                restore_indices[0] += 1
            restore_from_idx += 1
        
        return restore_indices
    
    def _get_dataset_and_childindex_at_seq_idx(self, seq_idx):
        pass

        

    def _collect_single_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        assert self.dataset_seq_idx_offsets is not None
        dataset_idx = self._get_dataset_for_seq_idx(seq_idx)
        choose_right = self.bitset_chooser.get(seq_idx)
        if choose_right:
            dataset = self.right_dataset
        else:
            dataset = self.left_dataset
        dataset_seq_idx = seq_idx + self.dataset_seq_idx_offsets[dataset_idx]
        seq_tag = dataset.get_tag(dataset_seq_idx)
        features = {k: dataset.get_data(dataset_seq_idx, k) for k in dataset.get_data_keys()}
        return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features)

    @property
    def num_seqs(self):
        """
        :rtype: int
        """
        return sum([ds.num_seqs for ds in self.datasets])

    def get_target_list(self):
        """
        :rtype: list[str]
        """
        return self.datasets[0].get_target_list()