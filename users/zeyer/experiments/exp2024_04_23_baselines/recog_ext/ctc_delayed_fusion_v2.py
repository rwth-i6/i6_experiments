"""
CTC decoding with neural LM.

See :func:`model_recog_with_recomb_delayed_fusion_v2`.
"""

from __future__ import annotations

from typing import Optional, Union, Callable, Sequence, Tuple
import os
import numpy as np
import itertools

from returnn.datasets.util.vocabulary import Vocabulary
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import RecogDef

from ..ctc import Model


def never_enable(**_kwargs) -> bool:
    """
    Never enable. This is a sanity experiment which should give the same results as standard rescoring.
    Note that the search will anyway convert labels and fuse LM scores at the very end for all the remaining labels.
    """
    return False


def enable_by_interval(*, t: int, interval: int, **_kwargs) -> bool:
    """
    Enable every `interval` steps, i.e. at t=interval-1, 2*interval-1, ...

    Bind `interval` via `functools.partial`.

    Use as `should_convert_labels_now_func`, `should_fuse_now_func` in config.
    """
    return t % interval == interval - 1


def convert_labels_func_no_op(
    *, new_am_labels: Tensor, new_am_labels_spatial_dim: Dim, lm_target_dim: Dim, **_kwargs
) -> Tuple[Tensor, Dim, Tensor]:
    """
    Convert AM labels to LM labels.

    Use as `convert_labels_func` in config.

    :param new_am_labels: Tensor of shape {batch..., new_am_labels_spatial_dim}
    :param new_am_labels_spatial_dim: Dim of new AM labels
    :param lm_target_dim: target dim of the LM
    :return: (new_lm_labels, new_lm_labels_spatial_dim, num_am_labels_converted)
        1. new_lm_labels: Tensor of shape {batch..., new_lm_labels_spatial_dim} -> lm_target_dim
        2. new_lm_labels_spatial_dim: Dim of new LM labels
        3. num_am_labels_converted: Tensor of shape {batch...} (int32)
    """
    assert lm_target_dim == new_am_labels.sparse_dim
    return new_am_labels, new_am_labels_spatial_dim, new_am_labels_spatial_dim.get_size_tensor()


def spm_space_first_is_word_start(label_idx: int, *, vocab: Vocabulary, **_kwargs) -> bool:
    """SPM with space-first, return whether the label is a word start (i.e. starts with "▁")"""
    label = vocab.id_to_label(label_idx)
    return label.startswith("▁")


def spm_label_merge_v2(labels: np.ndarray, *, vocab: Vocabulary, is_beginning: bool, is_end: bool, **_kwargs) -> str:
    """SPM label merge function, convert a list of SPM labels to a string"""
    res = "".join(vocab.id_to_label(label_idx) for label_idx in labels).replace("▁", " ")
    if is_beginning:
        res = res.lstrip()  # only strip left side, to keep the spaces in the middle and at the end
    if is_end:
        res = res.rstrip()
    return res


def seq_str_postprocess_lower_case(seq_str: str, **_kwargs) -> str:
    """convert to lower case"""
    return seq_str.lower()


# bind is_am_label_word_start or is_am_label_word_end via functools.partial
def convert_labels_func(
    *,
    new_am_labels: Tensor,
    new_am_labels_spatial_dim: Dim,
    first_am_labels: Union[bool, Tensor],
    last_am_labels: bool,
    lm_target_dim: Dim,
    is_am_label_word_start: Optional[Callable] = None,
    is_am_label_word_end: Optional[Callable] = None,
    custom_am_label_merge: Optional[Callable] = None,
    seq_str_postprocess_func: Optional[Callable] = None,
    **_kwargs,
) -> Tuple[Tensor, Dim, Tensor]:
    """
    Convert AM labels to LM labels.

    Use as `convert_labels_func` in config.

    :param new_am_labels: Tensor of shape {batch..., new_am_labels_spatial_dim}
    :param new_am_labels_spatial_dim: Dim of new AM labels
    :param first_am_labels: whether these are the first AM labels to convert.
        Note that some SPM tokenizers will tokenize "Hello world" as ["▁Hello", "▁world"],
        i.e. the first AM label has "▁" at the beginning.
        In the SPM merge function, stripping the spaces away after merging is only valid on the whole sequence,
        or at the beginning of the sequence, but not when merging a middle part of the sequence.
        So we need to know whether these are the first AM labels to convert.
    :param last_am_labels:
        It signals that we always should convert all the remaining AM labels,
        even if they do not end with a word end label.
    :param lm_target_dim: target dim of the LM
    :param is_am_label_word_start:
    :param is_am_label_word_end:
    :param custom_am_label_merge:
    :param seq_str_postprocess_func: optional. func (seq_str: str, vocab: Vocabulary, ...) -> str,
        postprocess the converted seq_str before converting to LM labels.
        For example :func:`seq_str_postprocess_lower_case`.
    :return: (new_lm_labels, new_lm_labels_spatial_dim, num_am_labels_converted)
        1. new_lm_labels: Tensor of shape {batch..., new_lm_labels_spatial_dim} -> lm_target_dim
        2. new_lm_labels_spatial_dim: Dim of new LM labels
        3. num_am_labels_converted: Tensor of shape {batch...} (int32)
    """
    assert is_am_label_word_start or is_am_label_word_end
    new_am_labels = rf.copy_to_device(new_am_labels, "cpu")
    assert new_am_labels.sparse_dim and new_am_labels.sparse_dim.vocab
    am_vocab = new_am_labels.sparse_dim.vocab
    assert lm_target_dim.vocab
    lm_vocab = lm_target_dim.vocab
    if isinstance(first_am_labels, bool):
        first_am_labels = rf.constant(first_am_labels, dims=(), dtype="bool", device="cpu")
    else:
        first_am_labels = rf.copy_to_device(first_am_labels, "cpu")

    batch_dims = new_am_labels.remaining_dims(new_am_labels_spatial_dim)
    lm_labels_list_by_bs = {}  # batch index -> list of new LM labels
    num_am_labels_converted_by_bs = {}  # batch index -> num AM labels converted
    for bs in itertools.product(*(range(d.get_dim_value()) for d in batch_dims)):  # e.g. (batch,beam) indices
        am_labels_ = new_am_labels
        am_lens = new_am_labels_spatial_dim.get_size_tensor()
        first_am_labels_ = first_am_labels
        for d, idx in zip(batch_dims, bs):
            am_labels_ = rf.gather(am_labels_, axis=d, indices=idx)
            if d in am_lens.dims:
                am_lens = rf.gather(am_lens, axis=d, indices=idx)
            if d in first_am_labels_.dims:
                first_am_labels_ = rf.gather(first_am_labels_, axis=d, indices=idx)
        assert am_labels_.dims == (new_am_labels_spatial_dim,) and am_lens.dims == first_am_labels_.dims == ()
        am_labels_, am_spatial_dim = rf.slice(am_labels_, axis=new_am_labels_spatial_dim, start=0, end=am_lens)
        assert am_labels_.dims == (am_spatial_dim,) and am_lens.dims == ()
        am_labels_raw = am_labels_.raw_tensor.tolist()
        am_lens_raw = am_lens.raw_tensor.item()
        if last_am_labels:
            am_full_words_len = am_lens_raw
        else:
            am_full_words_len = 0
            for i in reversed(range(am_lens_raw)):
                if is_am_label_word_end and is_am_label_word_end(am_labels_raw[i], vocab=am_vocab, **_kwargs):
                    am_full_words_len = i + 1
                    break
                if is_am_label_word_start and is_am_label_word_start(am_labels_raw[i], vocab=am_vocab, **_kwargs):
                    am_full_words_len = i
                    break
        num_am_labels_converted_by_bs[bs] = am_full_words_len
        if am_full_words_len > 0:
            am_labels_raw = am_labels_raw[:am_full_words_len]
            if custom_am_label_merge:
                seq_str = custom_am_label_merge(
                    am_labels_raw,
                    vocab=am_vocab,
                    is_beginning=first_am_labels_.raw_tensor.item(),
                    is_end=last_am_labels,
                    **_kwargs,
                )
                assert isinstance(seq_str, str)
            else:
                seq_str = am_vocab.get_seq_labels(am_labels_raw)
            if seq_str_postprocess_func is not None:
                seq_str = seq_str_postprocess_func(seq_str=seq_str, vocab=am_vocab, **_kwargs)
            lm_labels_list = lm_vocab.get_seq(seq_str)
        else:
            lm_labels_list = []
        lm_labels_list_by_bs[bs] = lm_labels_list

    new_lm_labels_lens = rf.zeros(batch_dims, dtype="int32", device="cpu")
    for bs in lm_labels_list_by_bs:
        new_lm_labels_lens.raw_tensor[bs] = len(lm_labels_list_by_bs[bs])
    new_lm_labels_spatial_dim = Dim(
        new_lm_labels_lens, name=f"new_lm_labels_spatial{int(new_am_labels_spatial_dim.get_dim_value())}"
    )
    new_lm_labels = rf.zeros(
        batch_dims + [new_lm_labels_spatial_dim], dtype="int32", sparse_dim=lm_target_dim, device="cpu"
    )
    for bs in lm_labels_list_by_bs:
        lm_labels_list = lm_labels_list_by_bs[bs]
        new_lm_labels.raw_tensor[bs][: len(lm_labels_list)] = rf.convert_to_tensor(lm_labels_list).raw_tensor
    num_am_labels_converted = rf.zeros(batch_dims, dtype="int32", device="cpu")
    for bs in num_am_labels_converted_by_bs:
        num_am_labels_converted.raw_tensor[bs] = num_am_labels_converted_by_bs[bs]

    return new_lm_labels, new_lm_labels_spatial_dim, num_am_labels_converted


def model_recog_with_recomb_delayed_fusion_v2(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Time-synchronous beam search with CTC and neural LM, with path recombination.
    With recombination of paths with the same label sequence.
    Delayed LM fusion:
    V1: LM score is only evaluated when a new label is added to the sequence.
    V2: Evaluating and adding the LM score can be done more delayed,
    e.g. after word end, or after fixed time intervals, https://arxiv.org/abs/2501.09258.

    Based on
    :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_v2.model_recog_with_recomb_v2`,
    :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_delayed_fusion.model_recog_with_recomb_delayed_fusion`.

    Note: It is recommended to test the LM API for correctness first, like in
    :func:`i6_experiments.users.zeyer.returnn.rf_lm_test_impl.test_qwen2_finetuned`.

    Also, there are several debug mechanisms in this code here,
    which can be enabled via environment variables, see the code below.

    Function is run within RETURNN.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import returnn
    from returnn.config import get_global_config
    from returnn.util.basic import get_fwd_compat_kwargs
    from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated
    from .ctc_debugging import _seq_label_print, _generic_seq_label_print, _generic_print

    config = get_global_config()
    debug = os.environ.get("DEBUG_CTC_RECOG") == "1"
    beam_size = config.int("beam_size", 12)
    version = config.int("recog_version", 1)
    assert version == 12
    recomb = config.typed_value("recog_recomb", "max")  # None, "max", "sum"
    ctc_soft_collapse_threshold = config.typed_value("ctc_soft_collapse_threshold", None)
    ctc_soft_collapse_reduce_type = config.typed_value("ctc_soft_collapse_reduce_type", "logmeanexp")
    delayed_prior = config.bool("delayed_prior", True)  # synced with LM or not

    if debug:
        if os.environ.get("DEBUG_CTC_RECOG_BEAM_SIZE"):
            beam_size = int(os.environ.get("DEBUG_CTC_RECOG_BEAM_SIZE"))
        print(f"*** Starting CTC + LM beam search recog with recomb delayed fusion v2 DEBUG MODE {beam_size=} ***")

    if debug and os.environ.get("DEBUG_CTC_RECOG_SHORTEN_INPUT"):  # maybe cut for faster debugging
        shorten_len = int(os.environ.get("DEBUG_CTC_RECOG_SHORTEN_INPUT"))
        print("size:", data_spatial_dim.get_size_tensor().raw_tensor.numpy(), "->", shorten_len)
        data, data_spatial_dim = rf.slice(data, axis=data_spatial_dim, size=shorten_len)

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    # 1.20260212.101626: RF PT pad, fix left pad for non-scalar value / RF fix causal self att with concat state
    # 1.20260215.233656: fix in gather_nested.
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20260215, 233656), returnn.__version__

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob, _, enc_spatial_dim = model.encode_and_get_ctc_log_probs(data, in_spatial_dim=data_spatial_dim)
    batch_dims = label_log_prob.remaining_dims((enc_spatial_dim, label_log_prob.feature_dim))

    if ctc_soft_collapse_threshold is not None:
        label_log_prob, enc_spatial_dim = soft_collapse_repeated(
            label_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=model.wb_target_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial_beam")
    batch_dims_ = [beam_dim] + batch_dims
    neg_inf = float("-inf")
    ctc_seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam
    lm_seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam
    prior_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=neg_inf),
    )
    label_log_prob_ta = TensorArray.unstack(label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB

    am_seq_label = _seq_label_history_init_state(vocab_dim=model.target_dim, batch_dims=batch_dims_)

    # We usually have TransformerDecoder, but any other type would also be ok when it has the same API.
    # noinspection PyUnresolvedReferences
    lm: TransformerDecoder = model.lm
    # noinspection PyUnresolvedReferences
    lm_scale: float = model.lm_scale

    lm_target_dim = lm.vocab_dim
    assert lm_target_dim.vocab is not None
    lm_vocab = lm_target_dim.vocab
    assert lm_vocab.bos_label_id is not None and lm_vocab.eos_label_id is not None
    lm_eos_idx = lm_vocab.eos_label_id

    lm_state = lm.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
    lm_logits, lm_state = lm(
        rf.constant(lm_vocab.bos_label_id, dims=batch_dims_, sparse_dim=lm_target_dim),
        spatial_dim=single_step_dim,
        state=lm_state,
    )  # Batch, InBeam, Vocab / ...
    lm_log_probs = rf.log_softmax(lm_logits, axis=lm_target_dim)  # Batch, InBeam, LmVocab

    if debug:
        print("initial LM log probs:", end="")
        _generic_print(lm_log_probs, dims_no_iter=batch_dims_, max_idx=5)

    am_seq_last_converted = rf.constant(0, dims=batch_dims_, dtype="int32", device="cpu")  # Batch, InBeam -> int32
    am_seq_num_consumed = rf.constant(0, dims=batch_dims_, dtype="int32", device="cpu")  # Batch, InBeam -> int32

    lm_seq_label = _seq_label_history_init_state(vocab_dim=lm_target_dim, batch_dims=batch_dims_)

    lm_seq_num_consumed = rf.constant(0, dims=batch_dims_, dtype="int32", device="cpu")  # Batch, InBeam -> int32

    def _debug_lm():
        batch_dims_debug = list(lm_seq_log_prob.dims)

        # Convert again, check same
        am_labels, am_spatial_dim = rf.slice(
            am_seq_label.history, axis=am_seq_label.hist_dim, size=am_seq_last_converted
        )
        lm_labels, lm_spatial_dim, num_am_labels_converted = convert_labels_func(
            new_am_labels=am_labels,
            new_am_labels_spatial_dim=am_spatial_dim,
            lm_target_dim=lm_target_dim,
            first_am_labels=True,
            last_am_labels=True,
        )
        lm_labels: Tensor
        lm_labels = lm_labels.copy_transpose(batch_dims_debug + [lm_spatial_dim]).copy_masked(0)
        lm_labels_actual = lm_seq_label.history
        lm_labels_actual = lm_labels_actual.copy_transpose(batch_dims_debug + [lm_seq_label.hist_dim]).copy_masked(0)
        if (lm_labels.raw_tensor.shape != lm_labels_actual.raw_tensor.shape) or (
            lm_labels.raw_tensor.cpu().numpy() != lm_labels_actual.raw_tensor.cpu().numpy()
        ).any():
            print("Mismatch:")
            print(f"{t=}")
            print(f"{am_seq_num_consumed=} {am_seq_num_consumed.raw_tensor.cpu().numpy()}")
            print(f"{am_seq_last_converted=} {am_seq_last_converted.raw_tensor.cpu().numpy()}")
            max_size = rf.maximum(lm_spatial_dim.get_size_tensor(), lm_seq_label.hist_dim.get_size_tensor())
            max_spatial_dim = Dim(max_size, name="max_spatial")
            lm_labels_ = rf.replace_dim_v2(lm_labels, in_dim=lm_spatial_dim, out_dim=max_spatial_dim)
            lm_labels_ = rf.copy_to_device(lm_labels_, "cpu")
            lm_labels_actual_ = rf.replace_dim_v2(
                lm_labels_actual, in_dim=lm_seq_label.hist_dim, out_dim=max_spatial_dim
            )
            lm_labels_actual_ = rf.copy_to_device(lm_labels_actual_, "cpu")
            print("Matching sizes:")
            _generic_print(lm_spatial_dim.get_size_tensor() == lm_seq_label.hist_dim.get_size_tensor())
            print("Matching seqs:")
            _generic_print(rf.reduce_all(lm_labels_ == lm_labels_actual_, axis=max_spatial_dim))
            # print("Matching tokens:")
            # _generic_print(lm_labels_ == lm_labels_actual_)
            print("debug LM labels ref:", end="")
            _generic_seq_label_print(lm_labels, spatial_dim=lm_spatial_dim)
            print("debug LM labels actual:", end="")
            _generic_seq_label_print(lm_labels_actual, spatial_dim=lm_seq_label.hist_dim)
            raise ValueError("debug LM labels mismatch")

        labels = lm_seq_label.history
        spatial_dim: Dim = lm_seq_label.hist_dim
        labels, spatial_dim = rf.slice(labels, axis=spatial_dim, size=lm_seq_num_consumed)
        if spatial_dim.get_dim_value_tensor().raw_tensor == 0:
            return None
        input_labels = rf.shift_right(labels, axis=spatial_dim, pad_value=lm_vocab.bos_label_id)
        if debug:
            print("debug LM input labels:", end="")
            _generic_seq_label_print(input_labels, spatial_dim=spatial_dim, dims_no_iter=batch_dims_debug)
        lm_state_debug = lm.default_initial_state(batch_dims=batch_dims_debug)  # Batch, InBeam, ...
        lm_logits_debug, lm_state_debug = lm(
            input_labels, spatial_dim=spatial_dim, state=lm_state_debug
        )  # Batch, InBeam, Vocab / ...
        lm_log_probs_debug = rf.log_softmax(lm_logits_debug, axis=lm_target_dim)  # Batch, InBeam, LmVocab
        if debug:
            print("debug LM log probs:", end="")
            _generic_print(lm_log_probs_debug, dims_no_iter=batch_dims_, max_idx=5)
        lm_seq_log_prob_debug = rf.reduce_sum(
            rf.gather(lm_log_probs_debug, axis=lm_target_dim, indices=labels), axis=spatial_dim
        )  # Batch, InBeam
        lm_seq_log_prob_debug *= lm_scale
        np.testing.assert_allclose(
            lm_seq_log_prob.raw_tensor.cpu().numpy(),
            lm_seq_log_prob_debug.copy_compatible_to_dims_raw(batch_dims_debug).cpu().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        return lm_state_debug, lm_log_probs_debug, lm_seq_log_prob_debug

    should_convert_labels_now_func = config.typed_dict["should_convert_labels_now_func"]
    # (...) -> bool
    should_fuse_now_func = config.typed_dict["should_fuse_now_func"]
    # (...) -> bool
    convert_labels_func = config.typed_dict["convert_labels_func"]
    # (...) -> Tensor

    if debug and os.environ.get("DEBUG_CTC_RECOG_NO_DELAYED_FUSION") == "1":
        should_convert_labels_now_func = None
        should_fuse_now_func = None

    def _convert_labels_now():
        nonlocal am_seq_last_converted
        new_am_labels, new_am_labels_spatial_dim = rf.slice(
            am_seq_label.history, axis=am_seq_label.hist_dim, start=am_seq_last_converted
        )
        if debug:
            print("new am labels to convert:", end="")
            _generic_seq_label_print(new_am_labels, new_am_labels_spatial_dim, dims_no_iter=batch_dims)
        new_lm_labels, new_lm_labels_spatial_dim, num_am_labels_converted = convert_labels_func(
            new_am_labels=new_am_labels,
            new_am_labels_spatial_dim=new_am_labels_spatial_dim,
            lm_target_dim=lm_target_dim,
            first_am_labels=am_seq_last_converted == 0,
            last_am_labels=is_last_frame,
            **get_fwd_compat_kwargs(),
        )
        if debug:
            print("converted new lm labels:", end="")
            _generic_seq_label_print(new_lm_labels, new_lm_labels_spatial_dim, dims_no_iter=batch_dims)
        assert isinstance(new_lm_labels, Tensor) and isinstance(new_lm_labels_spatial_dim, Dim)
        new_lm_labels = rf.copy_to_device(new_lm_labels, lm_seq_label.history.device)
        assert new_lm_labels_spatial_dim in new_lm_labels.dims
        am_seq_last_converted += num_am_labels_converted
        lm_seq_label.history, lm_seq_label.hist_dim = rf.concat(
            (lm_seq_label.history, lm_seq_label.hist_dim), (new_lm_labels, new_lm_labels_spatial_dim)
        )
        lm_seq_label.hist_dim.name = f"lm_hist{int(lm_seq_label.hist_dim.get_dim_value())}"

    # noinspection PyUnresolvedReferences
    labelwise_prior: Optional[rf.Parameter] = model.labelwise_prior
    if labelwise_prior is not None:
        assert len(labelwise_prior.dims) == 1
        assert labelwise_prior.dims[0] in {model.target_dim, lm_target_dim}

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        is_last_frame = t == max_seq_len - 1
        if debug:
            _seq_label_print("am", am_seq_label, dims_no_iter=batch_dims)
            _seq_label_print("lm", lm_seq_label, dims_no_iter=batch_dims)
            _debug_lm()

        prev_target = target
        prev_target_wb = target_wb

        seq_log_prob = (ctc_seq_log_prob + lm_seq_log_prob - prior_log_prob) + label_log_prob_ta[
            t
        ]  # Batch, InBeam, VocabWB

        _, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        ctc_seq_log_prob = rf.gather(ctc_seq_log_prob, indices=backrefs)  # Batch, Beam
        ctc_seq_log_prob += rf.gather(label_log_prob_ta[t], indices=target_wb, axis=model.wb_target_dim)  # Batch, Beam
        lm_seq_log_prob = rf.gather(lm_seq_log_prob, indices=backrefs)  # Batch, Beam
        prior_log_prob = rf.gather(prior_log_prob, indices=backrefs)  # Batch, Beam
        lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, LmVocab
        backref_dim_map = {backrefs.sparse_dim: beam_dim}
        lm_state = rf.nested.gather_nested(lm_state, indices=backrefs, dim_map=backref_dim_map)
        am_seq_label = rf.nested.gather_nested(am_seq_label, indices=backrefs, dim_map=backref_dim_map)
        lm_seq_label = rf.nested.gather_nested(lm_seq_label, indices=backrefs, dim_map=backref_dim_map)
        am_seq_num_consumed = rf.gather(am_seq_num_consumed, indices=backrefs)  # Batch, Beam
        lm_seq_num_consumed = rf.gather(lm_seq_num_consumed, indices=backrefs)  # Batch, Beam
        am_seq_last_converted = rf.gather(am_seq_last_converted, indices=backrefs)  # Batch, Beam

        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB

        got_new_label: Tensor = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _target_remove_blank(
                target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
            ),
            prev_target,
        )  # Batch, Beam -> Vocab

        got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")

        if labelwise_prior is not None and not delayed_prior and got_new_label_cpu.raw_tensor.sum().item() > 0:
            prior_log_prob += rf.where(
                got_new_label, rf.gather(labelwise_prior, axis=model.target_dim, indices=target), 0.0
            )

        if got_new_label_cpu.raw_tensor.sum().item() > 0:
            am_seq_label = rf.nested.mask_nested(
                _seq_label_append(am_seq_label, target),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                mask_value=am_seq_label,
            )

            # Recombine paths with the same label seq.
            if not recomb:
                pass
            elif recomb in ("max", "sum"):
                # Set seq_log_prob for batch entries to neg_inf if they have the same label seq.
                same_seq_labels, beam_dual_dim = _same_seq_labels(
                    am_seq_label.history, spatial_dim=am_seq_label.hist_dim, beam_dim=beam_dim
                )
                ctc_seq_log_prob_ext = rf.where(
                    same_seq_labels,
                    rf.replace_dim_v2(ctc_seq_log_prob, in_dim=beam_dim, out_dim=beam_dual_dim),
                    neg_inf,
                )  # Batch, Beam, BeamDual
                if recomb == "sum":
                    ctc_seq_log_prob = rf.reduce_logsumexp(ctc_seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam
                elif recomb == "max":
                    ctc_seq_log_prob = rf.reduce_max(ctc_seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam
                else:
                    raise ValueError(f"invalid recog recomb {recomb!r}")
                # V2: Do not select the argmax of the seq_log_prob_ext.
                # Instead, use the one where got_new_label=False if possible!
                got_new_label_ext = rf.where(
                    same_seq_labels,
                    rf.replace_dim_v2(rf.cast(got_new_label, dtype="int32"), in_dim=beam_dim, out_dim=beam_dual_dim),
                    100,
                )  # Batch, Beam, BeamDual
                idx = rf.reduce_argmin(got_new_label_ext, axis=beam_dual_dim)  # Batch, Beam -> BeamDual
                mask = idx == rf.range_over_dim(beam_dim)  # Batch, Beam -> 0|1
                ctc_seq_log_prob = rf.where(mask, ctc_seq_log_prob, neg_inf)
            else:
                raise ValueError(f"invalid recog_recomb {recomb!r}")

        if not is_last_frame and should_convert_labels_now_func:
            should_convert_labels_now = should_convert_labels_now_func(t=t, **get_fwd_compat_kwargs())
        else:
            should_convert_labels_now = True
        assert isinstance(should_convert_labels_now, bool)
        if should_convert_labels_now:
            _convert_labels_now()

        num_new_lm_labels = lm_seq_label.hist_dim.get_size_tensor() - lm_seq_num_consumed
        if not is_last_frame and should_fuse_now_func:
            should_fuse_now = should_fuse_now_func(num_new_lm_labels=num_new_lm_labels, t=t, **get_fwd_compat_kwargs())
        else:
            should_fuse_now = True
        should_fuse_now = (
            (num_new_lm_labels > 0)
            # No need to fuse if masked out due to CTC recombination.
            # However, anyway do it in last frame, to make the final check happy that all labels are consumed.
            & (True if is_last_frame else rf.copy_to_device(ctc_seq_log_prob > neg_inf, num_new_lm_labels.device))
            & should_fuse_now
        )

        if should_fuse_now.raw_tensor.sum().item() > 0:
            should_fuse_now_dev = rf.copy_to_device(should_fuse_now, lm_seq_log_prob.device)
            new_lm_labels, new_lm_labels_spatial_dim = rf.slice(
                lm_seq_label.history, axis=lm_seq_label.hist_dim, start=lm_seq_num_consumed
            )

            if debug:
                print("should fuse now:", end="")
                _generic_print(should_fuse_now, dims_no_iter=batch_dims, max_idx=5)

                print("new lm labels:", end="")
                _generic_seq_label_print(new_lm_labels, new_lm_labels_spatial_dim, dims_no_iter=batch_dims)

            (
                (
                    new_lm_labels_,
                    new_lm_labels_spatial_dim_,
                    lm_state_,
                    lm_seq_log_prob_,
                    prev_lm_log_probs_,
                    prior_log_prob_,
                ),
                packed_new_label_dim,
                packed_new_label_dim_map,
            ) = rf.nested.masked_select_nested(
                (
                    new_lm_labels,
                    new_lm_labels_spatial_dim,
                    lm_state,
                    lm_seq_log_prob,
                    lm_log_probs,
                    prior_log_prob
                    if labelwise_prior is not None and delayed_prior and labelwise_prior.dims[0] == lm_target_dim
                    else None,
                ),
                mask=should_fuse_now_dev,
                mask_cpu=should_fuse_now,
                dims=batch_dims + [beam_dim],
            )
            # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
            assert packed_new_label_dim.get_dim_value() > 0

            if debug:
                print("new lm feed:", end="")
                _generic_seq_label_print(
                    new_lm_labels_, new_lm_labels_spatial_dim_, dims_no_iter=[packed_new_label_dim]
                )
                print("lm state pos:", end="")
                _generic_print(lm_state_.pos, dims_no_iter=[packed_new_label_dim])

            lm_logits_, lm_state_ = lm(
                new_lm_labels_,
                spatial_dim=new_lm_labels_spatial_dim_,
                state=lm_state_,
            )  # FlatBatchBeam, [NewLmSpatial], Vocab / ...
            lm_log_probs_ = rf.log_softmax(lm_logits_, axis=lm_target_dim)  # FlatBatchBeam, NewLmSpatial, Vocab

            new_lm_log_probs_ = rf.gather(
                lm_log_probs_,
                axis=new_lm_labels_spatial_dim_,
                indices=rf.last_frame_position_of_dim(new_lm_labels_spatial_dim_),
            )
            lm_log_probs_ = rf.shift_right(lm_log_probs_, axis=new_lm_labels_spatial_dim_, pad_value=prev_lm_log_probs_)

            if debug:
                print("LM log probs:", end="")
                _generic_print(lm_log_probs_, dims_no_iter=[packed_new_label_dim], max_idx=5)

            # Now add LM score.
            lm_seq_log_prob_ += (
                rf.reduce_sum(
                    rf.gather(lm_log_probs_, axis=lm_target_dim, indices=new_lm_labels_),
                    axis=new_lm_labels_spatial_dim_,
                )
                * lm_scale
            )

            lm_seq_log_prob, lm_log_probs, lm_state = rf.nested.masked_scatter_nested(
                (lm_seq_log_prob_, new_lm_log_probs_, lm_state_),
                (lm_seq_log_prob, lm_log_probs, lm_state),
                mask=should_fuse_now_dev,
                mask_cpu=should_fuse_now,
                dims=batch_dims + [beam_dim],
                in_dim=packed_new_label_dim,
                masked_select_dim_map=packed_new_label_dim_map,
            )  # Batch, Beam, Vocab / ...

            if prior_log_prob_ is not None:
                prior_log_prob_ += rf.reduce_sum(
                    rf.gather(labelwise_prior, axis=lm_target_dim, indices=new_lm_labels_),
                    axis=new_lm_labels_spatial_dim_,
                )
                prior_log_prob = rf.nested.masked_scatter_nested(
                    prior_log_prob_,
                    prior_log_prob,
                    mask=should_fuse_now_dev,
                    mask_cpu=should_fuse_now,
                    dims=batch_dims + [beam_dim],
                    in_dim=packed_new_label_dim,
                    masked_select_dim_map=packed_new_label_dim_map,
                )  # Batch, Beam, Vocab / ...

            lm_seq_num_consumed = rf.where(
                should_fuse_now, lm_seq_label.hist_dim.get_size_tensor(), lm_seq_num_consumed
            )  # Batch, Beam -> int32

            new_am_seq_num_consumed = rf.where(
                should_fuse_now, am_seq_last_converted, am_seq_num_consumed
            )  # Batch, Beam -> int32
            new_am_labels, new_am_labels_spatial_dim = rf.slice(
                am_seq_label.history,
                axis=am_seq_label.hist_dim,
                start=am_seq_num_consumed,
                end=new_am_seq_num_consumed,
            )
            am_seq_num_consumed = new_am_seq_num_consumed

            if labelwise_prior is not None and delayed_prior and labelwise_prior.dims[0] == model.target_dim:
                prior_log_prob += rf.reduce_sum(
                    rf.gather(labelwise_prior, axis=model.target_dim, indices=new_am_labels),
                    axis=new_am_labels_spatial_dim,
                )

    assert (am_seq_last_converted == am_seq_label.hist_dim.get_size_tensor()).raw_tensor.all().item(), (
        f"seq len mismatch: {am_seq_last_converted=}\n"
        f" {am_seq_last_converted.raw_tensor.numpy()=}\n"
        f" vs {am_seq_label.hist_dim.get_size_tensor().copy_transpose(am_seq_last_converted.dims).raw_tensor.numpy()=},\n"
        f" {(am_seq_last_converted == am_seq_label.hist_dim.get_size_tensor()).raw_tensor.numpy()=}"
    )
    # TODO there is a rare edge case where this check is violated:
    #   the last AM token was just "▁".
    #   we should make a better check for this...
    # assert (am_seq_last_converted == am_seq_num_consumed).raw_tensor.all().item(), (
    #     f"seq len mismatch: {am_seq_last_converted=} {am_seq_num_consumed=}\n"
    #     f" {am_seq_last_converted.raw_tensor.numpy()=}\n"
    #     f" vs {am_seq_num_consumed.raw_tensor.numpy()=},\n"
    #     f" {(am_seq_last_converted == am_seq_num_consumed).raw_tensor.numpy()=}"
    # )
    assert (lm_seq_num_consumed == lm_seq_label.hist_dim.get_size_tensor()).raw_tensor.all().item(), (
        f"seq len mismatch: {lm_seq_num_consumed.raw_tensor.numpy()=}\n"
        f" vs {lm_seq_label.hist_dim.get_size_tensor().copy_transpose(lm_seq_num_consumed.dims).raw_tensor.numpy()=},\n"
        f" {am_seq_last_converted.raw_tensor.numpy()=}\n"
        f" {(lm_seq_num_consumed == lm_seq_label.hist_dim.get_size_tensor()).raw_tensor.numpy()=}"
    )

    # seq_log_prob, lm_log_probs: Batch, Beam
    # Add LM EOS score at the end.
    lm_eos_score = rf.gather(lm_log_probs, indices=lm_eos_idx, axis=lm_target_dim) * lm_scale  # Batch, Beam
    lm_seq_log_prob += lm_eos_score  # Batch, Beam -> VocabWB

    seq_log_prob = ctc_seq_log_prob + lm_seq_log_prob - prior_log_prob  # Batch, Beam

    # Backtrack via backrefs, resolve beams.
    seq_targets_wb_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target_wb in zip(seq_backrefs[::-1], seq_targets_wb[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_wb_.insert(0, rf.gather(target_wb, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets_wb__ = TensorArray(seq_targets_wb_[0])
    for target_wb in seq_targets_wb_:
        seq_targets_wb__ = seq_targets_wb__.push_back(target_wb)
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_wb__.stack(axis=out_spatial_dim)

    # Select valid.
    mask = rf.is_finite(seq_log_prob)  # Batch, Beam
    mask_cpu = rf.copy_to_device(mask, "cpu")
    (seq_targets_wb, seq_log_prob, out_spatial_dim), beam_dim, _ = rf.nested.masked_select_nested(
        (seq_targets_wb, seq_log_prob, out_spatial_dim), mask=mask, mask_cpu=mask_cpu, dims=[beam_dim]
    )

    if debug:
        raise Exception("success, but stop now (use hot reloading for easier debugging)")

    return seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_with_recomb_delayed_fusion_v2: RecogDef[Model]
model_recog_with_recomb_delayed_fusion_v2.output_with_beam = True
model_recog_with_recomb_delayed_fusion_v2.output_blank_label = "<blank>"
model_recog_with_recomb_delayed_fusion_v2.batch_size_dependent = True  # ...?


def _target_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == wb_target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, target_dim)


def _target_dense_extend_blank(
    target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int, value: float
) -> Tensor:
    assert target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
    return res


def _seq_label_history_init_state(*, vocab_dim: Dim, batch_dims: Sequence[Dim]) -> rf.State:
    """seq label state: history, hist_dim"""
    hist_dim = Dim(0, name="hist0")
    history = rf.zeros(list(batch_dims) + [hist_dim], dtype="int64", sparse_dim=vocab_dim)
    return rf.State(hist_dim=hist_dim, history=history)


def _seq_label_append(state: rf.State, new_label: Tensor) -> rf.State:
    hist_dim: Dim = state.hist_dim
    new_history, new_hist_dim = rf.cum_concat_step(new_label, prev_accum=state.history, axis=hist_dim)
    new_hist_dim.name = f"hist{int(new_hist_dim.get_dim_value())}"
    return rf.State(hist_dim=new_hist_dim, history=new_history)


def _same_seq_labels(seq: Tensor, *, spatial_dim: Dim, beam_dim: Dim) -> Tuple[Tensor, Dim]:
    beam_dual_dim = beam_dim.copy(same_as_self=False, description=beam_dim.description + "_dual")
    seq_label_dual, _ = rf.replace_dim(seq, in_dim=beam_dim, out_dim=beam_dual_dim)
    same_seq_labels = rf.compare_bc(seq, "==", seq_label_dual)  # Batch, Beam, BeamDual, Spatial
    same_seq_labels = rf.reduce_all(same_seq_labels, axis=spatial_dim)  # Batch, Beam, BeamDual
    if beam_dim in spatial_dim.get_size_tensor().dims:
        seq_labels_lens = spatial_dim.get_size_tensor(device=same_seq_labels.device)
        seq_labels_dual_lens = rf.replace_dim_v2(
            seq_labels_lens, in_dim=beam_dim, out_dim=beam_dual_dim
        )  # Batch, BeamDual
        same_seq_labels_lens = rf.compare_bc(seq_labels_lens, "==", seq_labels_dual_lens)  # Batch, Beam, BeamDual
        same_seq_labels = rf.logical_and(same_seq_labels, same_seq_labels_lens)
    return same_seq_labels, beam_dual_dim
