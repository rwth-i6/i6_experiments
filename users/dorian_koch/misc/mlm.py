# https://aclanthology.org/2020.acl-main.703/
# Select some indices, then mask consecutive tokens starting from there
# length is decided via poisson distribution

# SpeechT5: we randomly sample 30% of text spans to mask, where the span length of text spans
# draws from a Poisson distribution (λ = 3.5), and each span is replaced with a single mask token

"""
A lot of the code here is adapted from # https://github.com/microsoft/SpeechT5/blob/5d66cf5f37e97f4a1999ad519537decc16d852af/SpeechT5/speecht5/data/text_dataset.py

    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
"""

import math
from typing import Tuple
import torch
import returnn.frontend as rf
from functools import lru_cache


@lru_cache(maxsize=8)
def make_poisson_dist(_lambda: float):
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
    assert ps[-1] < 0.0000001, "increase k..."
    ps = torch.FloatTensor(ps)
    return torch.distributions.Categorical(ps)


def text_infill_masking(
    source: rf.Tensor, *, source_spatial_dim: rf.Dim, mask_prob: float, lambda_poisson: float, mask_token_id: int
) -> Tuple[rf.Tensor, rf.Dim]:
    """
    https://github.com/microsoft/SpeechT5/blob/5d66cf5f37e97f4a1999ad519537decc16d852af/SpeechT5/speecht5/data/text_dataset.py#L263

    :param source:
    :param source_spatial_dim:
    :param mask_prob:
    :param lambda_poisson:
    :param mask_token_id:
    :return: Tensor, Dim
    """
    assert source._raw_backend.name == "torch"

    # poisson dist has mean lambda_poisson
    # TODO: does this respect masks in source.dims? i think it doesnt matter
    mask_indices = rf.random_uniform(source.dims) < (mask_prob / lambda_poisson)
    num_indices = rf.reduce_sum(rf.cast(mask_indices, "int32"), axis=mask_indices.dims).raw_tensor.item()
    if num_indices == 0:
        return source, source_spatial_dim

    mask_span_distribution = make_poisson_dist(lambda_poisson)
    lengths = mask_span_distribution.sample(sample_shape=(num_indices,))

    max_spat_len = source_spatial_dim.get_dim_value_tensor()
    # now we dont want our spans to go beyond the max sequence length
    # i think this makes our code bias towards lower mask ratio, but hopefully doesnt matter?
    # TODO does this matter
    lengths = torch.minimum(lengths, max_spat_len - 1)

    num_to_mask_dim = rf.Dim(dimension=num_indices, name="num_mask_indices")
    lengths_tensor = rf.convert_to_tensor(lengths, dims=[num_to_mask_dim])

    # now we flatten mask_indices for easier indexing
    flat_mask_indices, all_merged = rf.merge_dims(mask_indices, dims=mask_indices.dims)
    # get all the indices where we want to start masking
    _, indices, _ = rf.top_k(
        rf.cast(flat_mask_indices, "int32"), k=num_indices, axis=all_merged, k_dim=num_to_mask_dim, sorted=False
    )
    all_indices = rf.range_over_dim(all_merged)
    # create mask over the spans
    maskmask = (all_indices >= indices) & (all_indices < (indices + lengths_tensor))

    # merge back to original shape
    mask_indices = rf.split_dims(maskmask, axis=all_merged, dims=mask_indices.dims, pad_to_multiples=False)
    source = rf.where(mask_indices, mask_token_id, source)

    # TODO: merge adjacent mask_token_id...
    return source, source_spatial_dim
