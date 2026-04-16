# taken from https://github.com/facebookresearch/fairseq/blob/aa79bb9c37b27e3f84e7a4e182175d3b50a79041/fairseq/tasks/denoising.py#L27

import math
import torch
class Masker:
    def __init__(self):
        import numpy as np

        _lambda = 3
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= k + 1
            if ps[-1] < 0.0000001:
                break

        self.mask_span_distribution = torch.distributions.Categorical(torch.FloatTensor(ps))
        self.mask_span_probs = ps
        self.num_mask_spans = len(ps)

        self.mask_idx = 100
        self.voc_valid_ids = np.arange(0, 100)
        self.voc_valid_size = self.voc_valid_ids.shape[0]
        self.random_ratio = 0.1
        self.replace_length = 1  # -1: replace with mask, 0: delete,

    def add_mask(self, source, p):
        source_ = source.clone()
        is_word_start = torch.ones(source_.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0

        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        print(f"num_to_mask: {num_to_mask}")
        if num_to_mask == 0:
            return source_

        lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))
        print(f"lengths: {lengths}")

        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)
        print(f"cum_length: {cum_length}")
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat(
                [
                    lengths,
                    mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                ],
                dim=0,
            )
            cum_length = torch.cumsum(lengths, 0)

        print(f"Extended lengths: {lengths}")

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts

        print(f"num_inserts: {num_inserts}, num_to_mask: {num_to_mask}, lengths: {lengths}")

        assert (lengths > 0).all()

        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio
        print(f"indices: {indices}")
        print(f"mask_random: {mask_random}")

        source_length = source_.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source_[indices] = self.mask_idx
            source_[indices[mask_random]] = self.voc_valid_ids[
                torch.randint(0, self.voc_valid_size - 1, size=(mask_random.sum(),))
            ]

        assert len(lengths.size()) == 1
        assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if self.replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source_[indices] = self.mask_idx
                source_[indices[mask_random]] = self.voc_valid_ids[
                    torch.randint(
                        0, self.voc_valid_size - 1, size=(mask_random.sum(),)
                    )
                ]

        source_ = source_[to_keep]

        return source_

    def add_mask_numpy(self, source_, p):
        import numpy as np

        is_word_start = np.ones(source_.shape)
        is_word_start[0] = 0
        is_word_start[-1] = 0

        num_to_mask = int(math.ceil(is_word_start.astype(np.float32).sum() * p))
        if num_to_mask == 0:
            return source_

        # lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))
        lengths = np.random.choice(self.num_mask_spans, size=num_to_mask, p=self.mask_span_probs)
        print(f"lengths: {lengths}")

        # Make sure we have enough to mask
        cum_length = np.cumsum(lengths, 0)
        print(f"cum_length: {cum_length}")
        while cum_length[-1] < num_to_mask:
            lengths = np.concatenate(
                [
                    lengths,
                    np.random.choice(self.num_mask_spans, size=num_to_mask, p=self.mask_span_probs),
                ],
                axis=0,
            )
            cum_length = np.cumsum(lengths, 0)

        print(f"Extended lengths: {lengths}")

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.shape[0]
        num_to_mask -= num_inserts

        print(f"num_inserts: {num_inserts}, num_to_mask: {num_to_mask}, lengths: {lengths}")

        assert (lengths > 0).all()

        assert is_word_start[-1] == 0
        word_starts = np.array(np.nonzero(is_word_start)).T

        indices = word_starts[
            np.random.permutation(word_starts.shape[0])[:num_to_mask]
        ].squeeze(1)
        mask_random = np.random.uniform(0.0, 1.0, size=num_to_mask) < self.random_ratio
        print(f"indices: {indices}")
        print(f"mask_random: {mask_random}")

        source_length = source_.shape[0]
        assert source_length - 1 not in indices
        to_keep = np.ones(source_length, dtype=bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source_[indices] = self.mask_idx
            source_[indices[mask_random]] = self.voc_valid_ids[
                np.random.randint(0, self.voc_valid_size - 1, size=(mask_random.sum(),))
            ]

        assert len(lengths.shape) == 1
        assert lengths.shape == indices.shape
        lengths -= 1
        while indices.shape[0] > 0:
            assert lengths.shape == indices.shape
            lengths -= is_word_start[indices + 1].astype(np.int64)
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if self.replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source_[indices] = self.mask_idx
                source_[indices[mask_random]] = self.voc_valid_ids[
                    np.random.randint(
                        0, self.voc_valid_size - 1, size=(mask_random.sum(),)
                    )
                ]

        source_ = source_[to_keep]

        return source_
