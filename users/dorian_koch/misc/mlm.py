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
from typing import Tuple, Union
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
    source: rf.Tensor,
    *,
    source_spatial_dim: rf.Dim,
    mask_prob: Union[float, Tuple[float, float]],
    lambda_poisson: float,
    mask_token_id: int,
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
    assert source._raw_backend.name == "torch"  # for poisson distribution sampling
    assert mask_token_id is not None
    assert lambda_poisson > 0.0
    if isinstance(mask_prob, tuple):
        assert 0.0 <= mask_prob[0] <= mask_prob[1] <= 1.0
        mask_prob = rf.random_uniform((), minval=mask_prob[0], maxval=mask_prob[1]).raw_tensor.item()
    else:
        assert 0.0 <= mask_prob <= 1.0

    # poisson dist has mean lambda_poisson, so each index will on average mask lambda_poisson tokens
    # TODO: does rf.random_uniform respect masks in source.dims? i think it doesnt matter
    mask_indices = rf.random_uniform(source.dims) < (mask_prob / lambda_poisson)
    num_indices = rf.reduce_sum(rf.cast(mask_indices, "int32"), axis=mask_indices.dims).raw_tensor.item()
    if num_indices == 0:
        return source, source_spatial_dim

    spat_lens = source_spatial_dim.get_size_tensor()
    total_tokens = rf.reduce_sum(spat_lens, axis=spat_lens.dims).raw_tensor.item()
    print(
        f"Masking {num_indices} spans, expect that to be {num_indices * lambda_poisson} masked tokens (of {total_tokens})",
        mask_indices.device,
    )

    mask_span_distribution = make_poisson_dist(lambda_poisson)
    lengths = mask_span_distribution.sample(sample_shape=(num_indices,))

    max_spat_len = source_spatial_dim.get_dim_value_tensor()
    # now we dont want our spans to go beyond the max sequence length
    # i think this makes our code bias towards lower mask ratio, but hopefully doesnt matter?
    # TODO does this matter
    # we don't actually need to do this...
    lengths = torch.minimum(lengths, max_spat_len.raw_tensor - 1)

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
    lengths_tensor = rf.copy_to_device(lengths_tensor, device=source.device)
    # print(indices.device, all_indices.device, lengths_tensor.device)
    maskmask = (all_indices >= indices) & (all_indices < (indices + lengths_tensor))  # [all_merged, num_to_mask_dim]
    assert maskmask.dims_set == set([all_merged, num_to_mask_dim])
    # merge all masks together
    maskmask = rf.reduce_any(maskmask, axis=num_to_mask_dim)  # [all_merged]

    actual_num_masked = rf.reduce_sum(rf.cast(maskmask, "int32"), axis=maskmask.dims).raw_tensor.item()
    print(f"Actually masking {actual_num_masked} tokens")

    # split back to original shape
    mask_indices = rf.split_dims(maskmask, axis=all_merged, dims=mask_indices.dims, pad_to_multiples=False)
    source = rf.where(mask_indices, mask_token_id, source)

    # now we eliminate consecutive duplicates of mask_token_id
    # i.e. replace spans of mask_token_id with single mask_token_id
    source_shifted = rf.shift_right(source, axis=source_spatial_dim, pad_value=mask_token_id)
    # if both the current and previous token are mask_token_id, we want to remove the current one
    mask_repeat = (source_shifted != mask_token_id) | (source != mask_token_id)
    # but always keep the first token
    mask_repeat |= rf.range_over_dim(source_spatial_dim) == 0
    # dont include any padded positions (not sure if this is necessary)
    mask_repeat &= source_spatial_dim.get_mask()
    assert mask_repeat.dims_set == source.dims_set

    old_source_dims = source.dims_set
    old_source_spatial = source_spatial_dim
    source, source_spatial_dim = rf.masked_select(
        source,
        mask=mask_repeat,
        dims=[source_spatial_dim],
    )
    assert (source.dims_set - {source_spatial_dim}) == (old_source_dims - {old_source_spatial})

    new_spat_lens = rf.copy_to_device(source_spatial_dim.get_size_tensor(), device=spat_lens.device)
    assert new_spat_lens.dims_set == spat_lens.dims_set
    smaller_test = new_spat_lens <= spat_lens
    assert rf.reduce_all(smaller_test, axis=smaller_test.dims).raw_tensor.item(), "spatial lengths should not increase"

    return source, source_spatial_dim
