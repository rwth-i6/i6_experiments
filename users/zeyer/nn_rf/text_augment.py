"""
Some simple augmentation on text.

Assuming you already have this:

.. python::

    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx, out_dims=[targets_w_eos_spatial_dim]
    )

"""

from typing import Optional, Collection, Sequence, Tuple
import functools
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def text_augment(
    *,
    input_labels: Tensor,
    targets_w_eos: Tensor,
    spatial_dim: Dim,
    eos_idx: int,
    exclude_labels: Collection[int] = (),
    ins_probs: Sequence[float],
    keep_del_sub_probs: Sequence[float],
) -> Tuple[Tensor, Tensor, Dim]:
    """

    :param input_labels: [Batch...,spatial_dim]. We assume this has BOS in the beginning.
    :param targets_w_eos: [Batch...,spatial_dim]. We assume this has EOS in the end,
        i.e. it is like input_labels but BOS removed and EOS added.
    :param spatial_dim: The spatial dimension of the labels.
    :param eos_idx: The index of the EOS token. Newly added frames after the end will use this for the targets.
        (For all other added frames, the targets will be the right target label.)
    :param exclude_labels: Labels to exclude from substitution on the input labels.
        E.g. you might want to exclude the BOS and EOS labels.
        (But this is maybe not necessary and still would have a good regularization effect.)
    :param ins_probs: A sequence of probabilities for each augmentation operation [insert0, insert1, insert2, ...].
        If no entries or only one entry, no insertions will be done.
    :param keep_del_sub_probs: A sequence of probabilities for each augmentation operation [keep, delete, substitute].
    :return: tuple (new input_labels, new targets_w_eos, new spatial_dim).
    """
    batch_dims = [d for d in input_labels.dims if d != spatial_dim]

    # Handle insertions first because for choosing the optimal target of inserted labels,
    # we still must know all the original targets.
    if len(ins_probs) >= 2:
        ins_dim = Dim(len(ins_probs), name="ins")
        ins_probs = rf.convert_to_tensor(ins_probs, dims=[ins_dim], dtype="float32", device=input_labels.device)
        ins_choices = rf.random_choice_with_replacement(
            batch_dims + [spatial_dim], probs=ins_probs, axis=ins_dim
        )  # e.g. [Batch,Spatial] -> ins_dim (how much to insert, 0, 1, 2, ...), each _after_ frame i in spatial
        new_seq_lens = rf.reduce_sum(ins_choices, axis=spatial_dim) + spatial_dim.dyn_size_ext
        new_spatial_dim = Dim(new_seq_lens, name="after_insert_spatial")
        # TODO correct...?
        new_indices = rf.cumsum(ins_choices + 1, spatial_dim=spatial_dim) - 1  # [Batch,Spatial] -> NewSpatial
        new_mask = rf.scatter(
            rf.sequence_mask(spatial_dim, device=input_labels.device),
            indices=new_indices,
            indices_dim=spatial_dim,
            out_dim=new_spatial_dim,
        )
        new_rnd_input_labels = _random_uniform_exclude(
            batch_dims + [new_spatial_dim],
            sparse_dim=input_labels.sparse_dim,
            exclude_labels=exclude_labels,
            device=input_labels.device,
        )
        input_labels = rf.masked_scatter(
            input_labels, backup=new_rnd_input_labels, mask=new_mask, dims=[new_spatial_dim], in_dim=spatial_dim
        )
        # TODO correct...?
        back_indices = rf.cumsum(
            rf.cast(new_mask, "int32"), spatial_dim=new_spatial_dim
        )  # [Batch,NewSpatial] -> Spatial
        targets_w_eos = rf.gather(targets_w_eos, indices=back_indices, axis=new_spatial_dim, clip_to_valid=True)
        spatial_dim = new_spatial_dim

    # Now handle the keep/delete/substitute operations.
    keep_del_sub_dim = Dim(len(keep_del_sub_probs), name="keep_del_sub")
    keep_del_sub_probs = rf.convert_to_tensor(
        keep_del_sub_probs, dims=[keep_del_sub_dim], dtype="float32", device=input_labels.device
    )
    keep_del_sub_choices = rf.random_choice_with_replacement(
        batch_dims + [spatial_dim], probs=keep_del_sub_probs, axis=keep_del_sub_dim
    )  # e.g. [Batch,Time] -> keep_del_sub_dim

    # Now first handle the deletions.
    non_del_mask = keep_del_sub_choices != 1  # 1 is delete, so we keep everything else
    input_labels, new_spatial_dim = rf.masked_select(input_labels, mask=non_del_mask, dims=[spatial_dim])
    targets_w_eos, _ = rf.masked_select(targets_w_eos, mask=non_del_mask, dims=[spatial_dim], out_dim=new_spatial_dim)
    keep_del_sub_choices, _ = rf.masked_select(
        keep_del_sub_choices, mask=non_del_mask, dims=[spatial_dim], out_dim=new_spatial_dim
    )  # [Batch,NewSpatial] -> keep_del_sub_dim (keep, substitute, no del)
    spatial_dim = new_spatial_dim

    # Handle the substitutions.
    rand_labels = _random_uniform_exclude(
        batch_dims + [spatial_dim],
        sparse_dim=input_labels.sparse_dim,
        exclude_labels=exclude_labels,
        device=input_labels.device,
    )
    input_labels = rf.where(keep_del_sub_choices == 0, input_labels, rand_labels)
    # TODO handle targets

    return input_labels, targets_w_eos, spatial_dim


def _random_uniform_exclude(
    dims: Sequence[Dim], *, sparse_dim: Dim, exclude_labels: Collection[int] = (), device: Optional[str] = None
) -> Tensor:
    exclude_labels = sorted(set(exclude_labels))
    out = rf.random_uniform(
        dims,
        minval=0,
        maxval=sparse_dim.dimension - len(exclude_labels),
        dtype="int32",
        sparse_dim=sparse_dim,
        device=device,
    )
    i = 0
    while i < len(exclude_labels):
        j = i + 1
        while j < len(exclude_labels) and exclude_labels[j] == exclude_labels[i] + (j - i):
            j += 1
        # exclude_labels[i:j] are consecutive, so we can just add the offset.
        out = rf.where(out >= exclude_labels[i], out + (j - i), out)
        i = j
    return out


def test_text_augment():
    from returnn.util.basic import BehaviorVersion

    rf.select_backend_torch()
    # Behavior version is important for rf.pad to handle dyn dims correctly.
    BehaviorVersion.set_min_behavior_version(24)

    # Data preparation
    examples = [
        "This is a test.",
        "Another example.",
        "Text augmentation is fun!",
        "Let's see how it works.",
    ]
    examples_bytes = [[int(b) for b in ex.encode("ascii")] for ex in examples]
    pad_idx, bos_idx, eos_idx = 0, 2, 3
    max_len = max(len(ex) for ex in examples_bytes)
    batch_dim = Dim(len(examples), name="batch")
    seq_lens = rf.convert_to_tensor([len(ex) for ex in examples_bytes], dims=[batch_dim])
    spatial_dim = Dim(seq_lens, name="spatial")
    labels = rf.convert_to_tensor(
        [ex + [pad_idx] * (max_len - len(ex)) for ex in examples_bytes], dims=[batch_dim, spatial_dim]
    )
    _dump_seq = functools.partial(_dump_seq_w_batch, batch_dim=batch_dim)

    _dump_seq("labels", labels)

    input_labels, (w_eos_spatial_dim,) = rf.pad(labels, axes=[spatial_dim], padding=[(1, 0)], value=bos_idx)
    targets_w_eos, _ = rf.pad(labels, axes=[spatial_dim], padding=[(0, 1)], value=eos_idx, out_dims=[w_eos_spatial_dim])
    _dump_seq("input_labels", input_labels)
    _dump_seq("targets_w_eos", targets_w_eos)


def _dump_seq_w_batch(prefix: str, tensor: Tensor, *, batch_dim: Dim):
    (dim_,) = tensor.remaining_dims(batch_dim)
    raw = tensor.copy_transpose([batch_dim, dim_]).raw_tensor
    ls = dim_.dyn_size
    bs = range(batch_dim.dimension)

    print(f"{prefix}: [{', '.join(''.join(map(_chr, raw[b, : ls[b]])) for b in bs)}]")


# Up to 31. 32 is space, so including and after that, there should be only printable ASCII characters.
# (Up to (including) 126. 127 is (non-printable) DEL, and after that, not ASCII anymore.)
_SpecialChars = "⓪①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛"  # ㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿


def _chr(c: int) -> str:
    assert c <= 126
    return _SpecialChars[c] if c < len(_SpecialChars) else chr(c)
