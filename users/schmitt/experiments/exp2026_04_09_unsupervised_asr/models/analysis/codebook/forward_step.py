__all__ = ["forward_step"]

import contextlib
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

from ....models.definitions.conformer_aed_discrete_shared_v1 import Model
from ....models.train_steps.util import get_random_mask, mask_sequence, expand_sequence


def _modality_present(data) -> bool:
    """A modality is present in the batch if its (padded) time dim is non-empty and at least one
    sequence has a non-zero length. With the ``CombinedDataset`` an audio-only batch carries
    length-0 text sequences (and vice versa), which we must not feed to the encoder."""
    if data is None:
        return False
    seq_lens = data.dims[1].dyn_size_ext.raw_tensor
    return data.raw_tensor.shape[1] > 0 and bool(seq_lens.max() > 0)


def _maybe_transform(indices, lens, mask_idx, masking_opts: Optional[Dict], expansion_opts: Optional[Dict]):
    """Optionally mask (collapse random spans to ``mask_idx``) and then upsample the encoder input
    exactly as in training. Returns ``(indices, lens, changed)`` where ``changed`` is True iff the
    sequence length was altered (in which case the caller must build a fresh time dim)."""
    changed = False
    if masking_opts is not None and masking_opts.get("mask_prob", 0.0) > 0.0:
        mask = get_random_mask(lens, **masking_opts)
        indices, lens = mask_sequence(indices, lens, mask, mask_value=mask_idx)
        changed = True
    if expansion_opts is not None:
        indices, lens = expand_sequence(indices, lens, **expansion_opts)
        changed = True
    return indices, lens, changed


def _dyn_time_dim(lens: Tensor, name: str) -> Dim:
    """Build a fresh dynamic time dim from per-seq lengths (RETURNN wants int32 sizes on CPU)."""
    lens_rf = rf.convert_to_tensor(lens.to(device="cpu", dtype=torch.int32), dims=[batch_dim])
    return Dim(lens_rf, name=name)


@contextlib.contextmanager
def _without_quantization(model: Model):
    """
    Temporarily detach the quantizer from the model, so ``forward_audio``/``forward_text`` return the
    *raw* (pre-quantization) encoder states.

    ``Model._maybe_quantize`` returns its input unchanged when ``model.quantizer is None``, so this
    both gives us the states that the quantizer's code selection is computed *from* (needed for the
    quantization-error metric) and avoids running the quantizer twice.

    Note this makes the analysis independent of ``codebook_prob``: which code a frame maps to is
    decided by the pre-quantization state, not by whether that frame ends up being replaced. So a
    ``recog_model_args`` override of ``codebook_prob`` does not change these results.
    """
    quantizer = model.quantizer
    assert quantizer is not None, (
        "codebook analysis requires a model with a quantizer (model_args.codebook_opts set)"
    )
    model.quantizer = None
    try:
        yield quantizer
    finally:
        model.quantizer = quantizer


def _codebook_stats(quantizer, states: Tensor, lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Per-frame codebook statistics for the (raw) encoder states ``[B, T, F]``.

    Reproduces the eval-time selection of :meth:`GumbelVectorQuantizer.forward` (argmax, no gumbel
    noise -- deterministic) and additionally returns what that method does not expose: the code
    *indices* themselves, the selection confidence, how far the selected code vector is from the
    state it would replace, and the summed selection *probabilities*.

    :return: ``(codes [B, T, G] int64, conf [B, T, G], cos [B, T], prob_sum [B, G, V])``.
        ``conf`` is the max softmax probability over the ``V`` entries of a group; ``cos`` the cosine
        similarity between the raw state and its fully quantized counterpart; ``prob_sum`` the
        softmax summed over the sequence's *valid* frames, from which the callback reconstructs the
        soft usage distribution that the diversity loss maximizes (the hard argmax usage can be
        collapsed while the soft one looks healthy -- that gap is a silent failure mode).
    """
    batch_size, max_len, feat_dim = states.shape
    num_groups = quantizer.groups
    num_vars = quantizer.num_vars

    flat = states.reshape(-1, feat_dim)
    # weight_proj output is laid out group-major (entry index g*V+v), cf. GumbelVectorQuantizer
    logits = quantizer.weight_proj(flat).view(-1, num_groups, num_vars).float()
    codes = logits.argmax(dim=-1)  # [N, G]
    probs = logits.softmax(dim=-1)  # [N, G, V]
    conf = probs.max(dim=-1).values  # [N, G]
    # sum the softmax over valid frames only (the per-frame outputs are sliced by RETURNN, but this
    # one has no time axis, so padding has to be masked out here)
    valid = (torch.arange(max_len, device=lens.device)[None, :] < lens[:, None]).to(probs.dtype)
    prob_sum = (probs.view(batch_size, max_len, num_groups, num_vars) * valid[:, :, None, None]).sum(dim=1)

    code_vars = quantizer.vars
    if quantizer.combine_groups:
        code_vars = code_vars.repeat(1, num_groups, 1)
    code_vars = code_vars.reshape(num_groups, num_vars, -1)  # [G, V, d]
    picked = code_vars[torch.arange(num_groups, device=codes.device)[None, :], codes]  # [N, G, d]
    quantized = picked.reshape(flat.shape[0], -1)  # [N, G*d = vq_dim]
    assert quantized.shape[-1] == feat_dim, (
        f"quantized dim {quantized.shape[-1]} != encoder dim {feat_dim}; the codebook analysis"
        " assumes codes replace encoder states in place (as Model asserts)"
    )
    cos = torch.nn.functional.cosine_similarity(flat.float(), quantized.float(), dim=-1)  # [N]

    return (
        codes.view(batch_size, max_len, num_groups),
        conf.view(batch_size, max_len, num_groups),
        cos.view(batch_size, max_len),
        prob_sum,
    )


def forward_step(
    *,
    model: Model,
    extern_data: TensorDict,
    audio_data_key: str = "data",
    text_data_key: str = "phon_indices",
    audio_masking_opts: Optional[Dict] = None,
    text_masking_opts: Optional[Dict] = None,
    text_expansion_opts: Optional[Dict] = None,
    **kwargs,
):
    """
    Forward step for the codebook (GumbelVectorQuantizer) analysis.

    For every modality present in the batch, the shared encoder is run (``forward_audio`` /
    ``forward_text``) with the quantizer detached, and the resulting raw encoder states are pushed
    through the quantizer's code selection by hand. The per-frame results are marked as outputs; all
    aggregation (usage histograms, joint-code distributions, code<->label contingency tables, ...)
    happens in the callback (:class:`...callback.CodebookUsageCallback`).

    Marked outputs, per modality ``m`` in ``{audio, text}``:

    - ``{m}_codes``:  ``[B, T, G]`` selected codebook entry per group (sparse over the ``V`` entries),
    - ``{m}_labels``: ``[B, T]`` the encoder *input* symbol that produced the frame (cluster id /
      phoneme id; the mask token where masking was applied), sparse over the modality's vocab,
    - ``{m}_conf``:   ``[B, T, G]`` max softmax probability of the selection,
    - ``{m}_cos``:    ``[B, T]`` cosine similarity between the raw state and its quantized version,
    - ``{m}_prob_sum``: ``[B, G, V]`` selection probabilities summed over the sequence's frames.

    Only small integer/scalar tensors are marked (no ``[B, T, F]`` states), so this is much cheaper
    than the encoder-PCA analysis and can run over a whole test set.

    Like the encoder-PCA analysis this handles both a paired ``MetaDataset`` (both modalities in
    every batch) and a ``CombinedDataset`` (audio-only / text-only batches): an absent modality is
    simply not marked for that batch.

    ``audio_masking_opts`` / ``text_masking_opts`` (``mask_prob``/``min_span``/``max_span``) and
    ``text_expansion_opts`` (``{"min_dup", "max_dup"}``) optionally mask / upsample the encoder input
    exactly as in training (masking first, then upsampling), so the analysis reflects the codes the
    encoder actually assigns during training. All default to None -> the raw input is fed unchanged.
    """
    ctx = rf.get_run_ctx()

    with _without_quantization(model) as quantizer:
        group_dim = Dim(quantizer.groups, name="codebook_group")
        entry_dim = Dim(quantizer.num_vars, name="codebook_entry")

        for modality, data_key, masking_opts, expansion_opts in (
            ("audio", audio_data_key, audio_masking_opts, None),
            ("text", text_data_key, text_masking_opts, text_expansion_opts),
        ):
            data = extern_data.data.get(data_key)
            if not _modality_present(data):
                continue

            if modality == "audio":
                forward_func, mask_idx, vocab_size = model.forward_audio, model.audio_mask_idx, model.audio_out_dim
            else:
                forward_func, mask_idx, vocab_size = model.forward_text, model.text_mask_idx, model.text_out_dim

            time_dim = data.dims[1]
            indices = data.raw_tensor.long()
            lens = time_dim.dyn_size_ext.raw_tensor.to(device=indices.device)
            indices, lens, changed = _maybe_transform(indices, lens, mask_idx, masking_opts, expansion_opts)
            # the encoder has no subsampling frontend, so its time dim is the (possibly masked /
            # upsampled) input time dim: reuse the input's dim tag when unchanged, else build a fresh one.
            if changed:
                time_dim = _dyn_time_dim(lens, f"{modality}_enc_time")

            states, _, _, _ = forward_func(indices, lens)
            codes, conf, cos, prob_sum = _codebook_stats(quantizer, states, lens)

            vocab_dim = Dim(vocab_size, name=f"{modality}_vocab")
            for name, raw, dims, sparse_dim in (
                (f"{modality}_codes", codes.int(), [batch_dim, time_dim, group_dim], entry_dim),
                (f"{modality}_labels", indices.int(), [batch_dim, time_dim], vocab_dim),
                (f"{modality}_conf", conf, [batch_dim, time_dim, group_dim], None),
                (f"{modality}_cos", cos, [batch_dim, time_dim], None),
                (f"{modality}_prob_sum", prob_sum, [batch_dim, group_dim, entry_dim], None),
            ):
                tensor = rf.convert_to_tensor(raw, dims=dims, sparse_dim=sparse_dim)
                ctx.mark_as_output(tensor, name, dims=dims)
