"""
TorchAudio (internally using Flashlight) for CTC with ngram LM
"""

from __future__ import annotations
from typing import Tuple, List

from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf

from i6_experiments.users.zeyer.model_interfaces import RecogDef

from ..ctc import Model


def model_recog_torchaudio(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    This function is run within RETURNN.

    Use together with
    :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext.get_ctc_with_ngram_lm_and_framewise_prior`
    and :func:`recog_model` or :func:`search_model` or so.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import torch
    from returnn.config import get_global_config
    from returnn.util import basic as util
    from torchaudio.models.decoder import CTCDecoder

    config = get_global_config()
    n_best = config.int("n_best", 1)

    # Eager-mode implementation of beam search using TorchAudio/Flashlight.

    dev_s = rf.get_default_device()
    dev = torch.device(dev_s)

    total_mem = None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        _, total_mem = torch.cuda.mem_get_info(dev if dev.index is not None else None)

    def _collect_mem_stats():
        if dev.type == "cuda":
            return [
                f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
            ]
        return ["(unknown)"]

    print(
        f"Memory usage {dev_s} before encoder forward:",
        " ".join(_collect_mem_stats()),
        "total:",
        util.human_bytes_size(total_mem) if total_mem else "(unknown)",
    )

    # https://github.com/pytorch/audio/blob/main/src/torchaudio/models/decoder/_ctc_decoder.py
    # https://docs.pytorch.org/audio/main/generated/torchaudio.models.decoder.ctc_decoder.html#torchaudio.models.decoder.ctc_decoder

    # noinspection PyTypeHints
    decoder: CTCDecoder = model.decoder

    assert data.dims_set == {batch_dim, data_spatial_dim, data.feature_dim}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert logits.dims_set == {batch_dim, enc_spatial_dim, model.wb_target_dim}

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob = label_log_prob.copy_transpose((batch_dim, enc_spatial_dim, model.wb_target_dim))
    batch_size, max_seq_len = label_log_prob.raw_tensor.shape[:2]
    assert enc_spatial_dim.dyn_size_ext.dims == (batch_dim,)

    label_log_prob = rf.cast(label_log_prob, "float32")
    label_log_prob = rf.copy_to_device(label_log_prob, "cpu")
    label_log_prob_raw = label_log_prob.raw_tensor.contiguous()
    float_bytes = 4

    print(f"Memory usage {dev_s} after encoder forward:", " ".join(_collect_mem_stats()))

    hyps = []
    scores = []
    for batch_idx in range(batch_size):
        emissions_ptr = label_log_prob_raw.data_ptr() + float_bytes * batch_idx * label_log_prob_raw.stride(0)
        seq_len = enc_spatial_dim.dyn_size[batch_idx]
        assert seq_len <= max_seq_len
        results = decoder(emissions_ptr, seq_len, model.wb_target_dim.dimension)
        # I get -1 (silence label?) at the beginning and end in the tokens? Filter those away.
        # These are also additional frames which don't correspond to the input frames?
        # When removing those two frames, the len of tokens (align labels) matches the emission frames
        # (as it should be).
        hyps_per_batch = [[label for label in result.tokens if label >= 0] for result in results]
        scores_per_batch = [result.score for result in results]
        print(
            f"batch {batch_idx + 1}/{batch_size}: {len(results)} hyps,"
            f" best score: {scores_per_batch[0]},"
            f" best seq {_format_align_label_seq(results[0].tokens, model.wb_target_dim)},"
            f" worst score: {scores_per_batch[-1]},"
            f" mem usage {dev_s}: {' '.join(_collect_mem_stats())}"
        )
        assert all(len(hyp) == seq_len for hyp in hyps_per_batch), (
            f"seq_len {seq_len}, hyps lens {[len(hyp) for hyp in hyps_per_batch]}"
        )
        if len(results) >= n_best:
            hyps_per_batch = hyps_per_batch[:n_best]
            scores_per_batch = scores_per_batch[:n_best]
        else:
            hyps_per_batch += [[]] * (n_best - len(results))
            scores_per_batch += [-1e30] * (n_best - len(results))
        assert len(hyps_per_batch) == len(scores_per_batch) == n_best
        hyps_per_batch = [hyp + [model.blank_idx] * (max_seq_len - len(hyp)) for hyp in hyps_per_batch]
        assert all(len(hyp) == max_seq_len for hyp in hyps_per_batch)
        hyps.append(hyps_per_batch)
        scores.append(scores_per_batch)
    hyps_pt = torch.tensor(hyps, dtype=torch.int32)
    assert hyps_pt.shape == (batch_size, n_best, max_seq_len)
    scores_pt = torch.tensor(scores, dtype=torch.float32)
    assert scores_pt.shape == (batch_size, n_best)

    beam_dim = Dim(n_best, name="beam")
    out_spatial_dim = enc_spatial_dim
    hyps_r = rf.convert_to_tensor(hyps_pt, dims=(batch_dim, beam_dim, out_spatial_dim), sparse_dim=model.wb_target_dim)
    scores_r = rf.convert_to_tensor(scores_pt, dims=(batch_dim, beam_dim))
    print(f"Memory usage ({dev_s}) after batch:", " ".join(_collect_mem_stats()))
    return hyps_r, scores_r, out_spatial_dim, beam_dim


# RecogDef API
model_recog_torchaudio: RecogDef[Model]
model_recog_torchaudio.output_with_beam = True
model_recog_torchaudio.output_blank_label = "<blank>"
model_recog_torchaudio.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def _format_align_label_seq(align_label_seq: List[int], wb_target_dim: Dim) -> str:
    seq_label: List[str] = []  # list of label
    seq_label_idx: List[int] = []  # list of label index
    seq_label_count: List[int] = []  # list of label count
    for align_label in align_label_seq:
        if seq_label_idx and seq_label_idx[-1] == align_label:
            seq_label_count[-1] += 1
        else:
            seq_label.append(wb_target_dim.vocab.id_to_label(align_label) if align_label >= 0 else str(align_label))
            seq_label_idx.append(align_label)
            seq_label_count.append(1)
    return " ".join(f"{label}*{count}" if count > 1 else label for label, count in zip(seq_label, seq_label_count))
