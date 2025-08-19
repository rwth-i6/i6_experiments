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
from returnn.util.basic import BehaviorVersion
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def text_augment(
    *,
    input_labels: Tensor,
    targets_w_eos: Tensor,
    spatial_dim: Dim,
    exclude_labels: Collection[int] = (),
    ins_probs: Sequence[float],
    ins_probs_last_frame: Optional[Sequence[float]] = None,
    keep_del_sub_probs: Sequence[float],
    keep_first_frame: bool = True,
    no_del_last_frame: bool = True,
) -> Tuple[Tensor, Tensor, Dim]:
    """

    :param input_labels: [Batch...,spatial_dim]. We assume this has BOS in the beginning.
    :param targets_w_eos: [Batch...,spatial_dim]. We assume this has EOS in the end,
        i.e. it is like input_labels but BOS removed and EOS added.
    :param spatial_dim: The spatial dimension of the labels.
    :param exclude_labels: Labels to exclude from substitution on the input labels.
        E.g. you might want to exclude the BOS and EOS labels.
        (But this is maybe not necessary and still would have a good regularization effect.)
    :param ins_probs: A sequence of probabilities for each augmentation operation [insert0, insert1, insert2, ...].
        If no entries or only one entry, no insertions will be done.
        Insertions are anyway only done after the first frame, so the first frame is always kept.
    :param ins_probs_last_frame: If given, uses these probabilities for the last frame.
    :param keep_del_sub_probs: A sequence of probabilities for each augmentation operation [keep, delete, substitute].
    :param keep_first_frame: Always keep (no substitute, no delete) first frame (e.g. BOS).
    :param no_del_last_frame: Do not delete (i.e. either keep or substitute) last frame.
        (This is done after insertions.)
    :return: tuple (new input_labels, new targets_w_eos, new spatial_dim).
    """
    assert BehaviorVersion.get() >= 23  # for correct masking, e.g. rf.pad, rf.scatter
    batch_dims = [d for d in input_labels.dims if d != spatial_dim]
    device = input_labels.device
    vocab_dim = input_labels.sparse_dim

    # Handle insertions first because for choosing the optimal target of inserted labels,
    # we still must know all the original targets.
    if len(ins_probs) >= 2:
        ins_dim = Dim(len(ins_probs), name="ins")
        ins_probs = rf.convert_to_tensor(ins_probs, dims=[ins_dim], dtype="float32", device=device)
        ins_choices = rf.random_choice_with_replacement(
            batch_dims + [spatial_dim], probs=ins_probs, axis=ins_dim
        )  # e.g. [Batch,Spatial] -> ins_dim (how much to insert, 0, 1, 2, ...), each _after_ frame i in spatial
        if ins_probs_last_frame is not None:
            ins_last_frame_dim = (
                ins_dim
                if len(ins_probs_last_frame) == ins_dim.dimension
                else Dim(len(ins_probs_last_frame), name="ins_last_frame")
            )
            ins_probs_last_frame = rf.convert_to_tensor(
                ins_probs_last_frame, dims=[ins_last_frame_dim], dtype="float32", device=device
            )
            ins_choices_last_frame = rf.random_choice_with_replacement(
                batch_dims, probs=ins_probs_last_frame, axis=ins_last_frame_dim
            )
            ins_choices = rf.where(
                rf.range_over_dim(spatial_dim, device=device) == spatial_dim.get_dyn_size_ext_for_device(device) - 1,
                ins_choices_last_frame,
                ins_choices,
            )
        new_seq_lens = rf.reduce_sum(ins_choices, axis=spatial_dim) + spatial_dim.dyn_size_ext
        new_spatial_dim = Dim(rf.copy_to_device(new_seq_lens, "cpu"), name="after_insert_spatial")
        new_indices = (
            rf.cumsum(ins_choices + 1, spatial_dim=spatial_dim) - 1 - ins_choices
        )  # [Batch,Spatial] -> NewSpatial
        new_mask = rf.scatter(
            rf.sequence_mask(spatial_dim, device=device),
            indices=new_indices.copy_masked(0),
            indices_dim=spatial_dim,
            out_dim=new_spatial_dim,
        )
        new_rnd_input_labels = _random_uniform_exclude(
            batch_dims + [new_spatial_dim],
            sparse_dim=vocab_dim,
            exclude_labels=exclude_labels,
            device=device,
        )
        input_labels = rf.masked_scatter(
            input_labels, backup=new_rnd_input_labels, mask=new_mask, dims=[new_spatial_dim], in_dim=spatial_dim
        )
        new_mask_i32 = rf.cast(new_mask, "int32")
        back_indices = (
            rf.cumsum(new_mask_i32, spatial_dim=new_spatial_dim) - new_mask_i32
        )  # [Batch,NewSpatial] -> Spatial
        targets_w_eos = rf.gather(targets_w_eos, indices=back_indices, axis=spatial_dim, clip_to_valid=True)
        spatial_dim = new_spatial_dim

    # Now handle the keep/delete/substitute operations.
    keep_del_sub_dim = Dim(len(keep_del_sub_probs), name="keep_del_sub")
    keep_del_sub_probs_ = rf.convert_to_tensor(
        keep_del_sub_probs, dims=[keep_del_sub_dim], dtype="float32", device=device
    )
    keep_del_sub_choices = rf.random_choice_with_replacement(
        batch_dims + [spatial_dim], probs=keep_del_sub_probs_, axis=keep_del_sub_dim
    )  # e.g. [Batch,Time] -> keep_del_sub_dim
    keep_del_sub_choices = rf.cast(keep_del_sub_choices, "int32")
    if keep_first_frame:
        keep_del_sub_choices = rf.where(rf.range_over_dim(spatial_dim) < 1, 0, keep_del_sub_choices)
    if no_del_last_frame:
        keep_del_sub_choices = rf.where(
            rf.range_over_dim(spatial_dim, device=device) == spatial_dim.get_dyn_size_ext_for_device(device) - 1,
            rf.where(rf.random_uniform(batch_dims, device=device) < keep_del_sub_probs[0], 0, 2),
            keep_del_sub_choices,
        )

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
        batch_dims + [spatial_dim], sparse_dim=vocab_dim, exclude_labels=exclude_labels, device=device
    )
    input_labels = rf.where(keep_del_sub_choices == 0, input_labels, rand_labels)
    # keep targets_w_eos unchanged for substitutions in the inputs.

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
    rf.select_backend_torch()
    rf.set_random_seed(42)
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
    vocab_dim = Dim(125, name="vocab")  # ASCII chars, first some specials, then all printable
    labels = rf.convert_to_tensor(
        [ex + [pad_idx] * (max_len - len(ex)) for ex in examples_bytes],
        dims=[batch_dim, spatial_dim],
        sparse_dim=vocab_dim,
    )
    _dump_seq = functools.partial(_dump_seq_w_batch, batch_dim=batch_dim)

    _dump_seq("labels", labels)

    input_labels, (w_eos_spatial_dim,) = rf.pad(labels, axes=[spatial_dim], padding=[(1, 0)], value=bos_idx)
    targets_w_eos, _ = rf.pad(labels, axes=[spatial_dim], padding=[(0, 1)], value=eos_idx, out_dims=[w_eos_spatial_dim])
    _dump_seq("input_labels", input_labels)
    _dump_seq("targets_w_eos", targets_w_eos)

    for _ in range(3):
        input_labels_, targets_w_eos_, spatial_dim_ = text_augment(
            input_labels=input_labels,
            targets_w_eos=targets_w_eos,
            spatial_dim=w_eos_spatial_dim,
            exclude_labels=list(range(0, 32)),  # exclude special chars and non-printable ASCII
            ins_probs=[0.9, 0.08, 0.02],
            keep_del_sub_probs=[0.8, 0.1, 0.1],
        )
        _dump_seq("augmented input_labels", input_labels_)
        _dump_seq("augmented targets_w_eos", targets_w_eos_)

    for _ in range(3):
        input_labels_, targets_w_eos_, spatial_dim_ = text_augment(
            input_labels=input_labels,
            targets_w_eos=targets_w_eos,
            spatial_dim=w_eos_spatial_dim,
            exclude_labels=list(range(0, 32)),  # exclude special chars and non-printable ASCII
            ins_probs=[0.7, 0.2, 0.1],
            ins_probs_last_frame=[0.1, 0.3, 0.3, 0.3],
            keep_del_sub_probs=[0.8, 0.1, 0.1],
        )
        _dump_seq("augmented input_labels", input_labels_)
        _dump_seq("augmented targets_w_eos", targets_w_eos_)


def test_text_augment_err_stats():
    rf.select_backend_torch()
    rf.set_random_seed(42)
    # Behavior version is important for rf.pad to handle dyn dims correctly.
    BehaviorVersion.set_min_behavior_version(24)

    examples = [
        "This is a test. A long test to see how it works. And maybe some more text.",
        "Another example: What happens if we have a longer text?",
        "Text augmentation is fun! Let's see how it works. And it can be very useful.",
        "Let there be some robot. And a cloud. And some rain. And some sunshine. And a rainbow.",
    ]
    examples_bytes = [[int(b) for b in ex.encode("ascii")] for ex in examples]
    pad_idx, bos_idx, eos_idx = 0, 2, 3
    max_len = max(len(ex) for ex in examples_bytes)
    batch_dim = Dim(len(examples), name="batch")
    seq_lens = rf.convert_to_tensor([len(ex) for ex in examples_bytes], dims=[batch_dim])
    spatial_dim = Dim(seq_lens, name="spatial")
    vocab_dim = Dim(125, name="vocab")  # ASCII chars, first some specials, then all printable
    labels = rf.convert_to_tensor(
        [ex + [pad_idx] * (max_len - len(ex)) for ex in examples_bytes],
        dims=[batch_dim, spatial_dim],
        sparse_dim=vocab_dim,
    )
    _dump_seq = functools.partial(_dump_seq_w_batch, batch_dim=batch_dim)
    _dump_seq("labels", labels)

    input_labels, (w_eos_spatial_dim,) = rf.pad(labels, axes=[spatial_dim], padding=[(1, 0)], value=bos_idx)
    targets_w_eos, _ = rf.pad(labels, axes=[spatial_dim], padding=[(0, 1)], value=eos_idx, out_dims=[w_eos_spatial_dim])

    num_errors = 0
    ref_len = 0
    ref_len_step = sum(len(ex) for ex in examples_bytes)
    for step in range(100):
        input_labels_, targets_w_eos_, w_eos_spatial_dim_ = text_augment(
            input_labels=input_labels,
            targets_w_eos=targets_w_eos,
            spatial_dim=w_eos_spatial_dim,
            exclude_labels=list(range(0, 32)),  # exclude special chars and non-printable ASCII
            ins_probs=[0.95, 0.04, 0.01],
            keep_del_sub_probs=[0.95, 0.03, 0.02],
        )
        labels_, spatial_dim_ = rf.slice(input_labels_, axis=w_eos_spatial_dim_, start=1)  # remove BOS
        _dump_seq("augmented input labels", labels_)
        errs = rf.edit_distance(labels, spatial_dim, labels_, spatial_dim_)  # [Batch]
        num_errors_step = rf.reduce_sum(errs, axis=errs.dims).raw_tensor.item()
        print(f"{step}: Errors: {num_errors_step} / {ref_len_step}, rate: {num_errors_step / ref_len_step:.1%}")
        num_errors += rf.reduce_sum(errs, axis=errs.dims).raw_tensor.item()
        ref_len += ref_len_step
    print(f"Total errors: {num_errors}, reference length: {ref_len}, error rate: {num_errors / ref_len:.1%}")


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
