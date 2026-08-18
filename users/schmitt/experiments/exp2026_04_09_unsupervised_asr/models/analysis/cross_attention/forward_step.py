__all__ = ["forward_step"]

import contextlib
import functools
import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

from i6_models.parts.decoder.cross_att import CrossAttentionV1, CrossAttentionV1State

from ....models.definitions.conformer_aed_discrete_shared_v1 import Model
from ....models.train_steps.util import get_random_mask, mask_sequence, expand_sequence


def _cross_att_forward_capturing(
    self: CrossAttentionV1,
    captured: Dict[int, Tensor],
    layer_idx: int,
    x: Tensor,
    x_lens: Tensor,
    state: CrossAttentionV1State,
) -> Tuple[Tensor, CrossAttentionV1State]:
    """
    Drop-in replacement for :meth:`CrossAttentionV1.forward` that also stores the attention weights.

    ``torch.nn.functional.scaled_dot_product_attention`` (used by the original) fuses softmax and the
    value multiplication, so the attention weights are never materialized. Here the same computation
    is written out explicitly (equivalent up to the fused kernel's numerics) and the resulting
    ``[B, H, L, T]`` weights (query = decoder label position, key = encoder frame) are stashed in
    ``captured[layer_idx]``.

    Attention dropout is intentionally not applied: this only runs inside a forward (eval) job.
    """
    # Ev: attention key/value dim per head, L: query length (labels), T: key length (encoder frames)
    x = self.norm(x)
    q = self.q(x)  # B... L Ev
    q = torch.unflatten(q, -1, (self.num_heads, -1)).transpose(-3, -2)  # B... H L Ev

    k = state["k"]  # B... H T Ev
    v = state["v"]  # B... H T Ev
    logits = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))  # B... H L T
    mask = state["mask"]  # additive mask (0 / -inf), B... 1 1 T
    if mask is not None and mask.numel() > 0:
        logits = logits + mask
    att_weights = torch.softmax(logits, dim=-1)  # B... H L T
    captured[layer_idx] = att_weights.detach()

    att_out = att_weights @ v  # B... H L Ev
    out = att_out.transpose(-3, -2).flatten(-2)  # B... L E
    out = self.out_proj(out)  # B... L F
    out = self.dropout(out)

    return out, state


@contextlib.contextmanager
def _capture_cross_attention(decoder):
    """
    Patch every cross-attention module of ``decoder`` to also record its attention weights.

    Yields a dict ``{layer_idx: [B, H, L, T]}`` which is filled while the decoder runs. The patch is
    an instance attribute (``module.forward = ...``), which ``nn.Module.__call__`` picks up, and is
    removed again on exit so the model is left untouched.
    """
    captured: Dict[int, Tensor] = {}
    patched = []
    for layer_idx, block in enumerate(decoder.module_list):
        for module in block.module_list:
            if isinstance(module, CrossAttentionV1):
                module.forward = functools.partial(_cross_att_forward_capturing, module, captured, layer_idx)
                patched.append(module)
    assert patched, "decoder has no cross attention modules"
    try:
        yield captured
    finally:
        for module in patched:
            del module.forward  # remove the instance attribute -> the class method is used again


def _dyn_time_dim(lens: Tensor, name: str) -> Dim:
    """Build a fresh dynamic time dim from per-seq lengths (RETURNN wants int32 sizes on CPU)."""
    lens_rf = rf.convert_to_tensor(lens.to(device="cpu", dtype=torch.int32), dims=[batch_dim])
    return Dim(lens_rf, name=name)


# number of sequences whose attention weights were marked as output so far (process-global, the
# forward step runs sequentially in one process). Used by `max_num_seqs`.
_num_marked_seqs = 0


def forward_step(
    *,
    model: Model,
    extern_data: TensorDict,
    input_data_key: str = "data",
    target_data_key: str = "phon_indices",
    input_modality: str = "audio",
    output_modality: str = "text",
    masking_opts: Optional[Dict] = None,
    expansion_opts: Optional[Dict] = None,
    max_num_seqs: Optional[int] = None,
    **kwargs,
):
    """
    Forward step for the decoder cross-attention analysis.

    The (shared) encoder is run over ``input_data_key`` (``input_modality`` selects the encoder path
    and the mask token) and the ``output_modality`` decoder is then **teacher-forced** on the
    reference label sequence ``target_data_key`` (decoder input ``bos + labels``), while the cross
    attention weights of every decoder layer are recorded. Teacher forcing (rather than beam search
    as in ``recognition.discrete_audio_aed``) keeps the query axis aligned to the *reference*
    labels, which makes the plots comparable across models and checkpoints.

    The default (audio in, text out) is the standard ASR direction: query = phoneme position, key =
    audio cluster frame. Any other combination works as well (e.g. text->text for the same-modality
    reconstruction path), as long as the dataset provides both keys.

    ``masking_opts`` (``mask_prob``/``min_span``/``max_span``) and ``expansion_opts``
    (``{"min_dup", "max_dup"}``) optionally mask / upsample the encoder input exactly as in training
    (masking first, then upsampling), so the attention plots reflect what the encoder sees during
    training. The teacher-forced label sequence always stays the unmodified reference.

    Marked outputs (only for the first ``max_num_seqs`` sequences, if given):

    - ``att_weights``: ``[B, layer, head, query, key]`` cross-attention weights,
    - ``labels``: ``[B, query]`` the decoder input labels (``bos + reference``), for axis labelling.

    :param max_num_seqs: if given, stop computing/marking once this many sequences were processed.
        The plots only cover the first few sequences anyway, so this bounds the runtime and the size
        of the (large) attention tensors. Whole batches are processed, so slightly more sequences
        than this may be marked; the callback applies the exact limit.
    """
    global _num_marked_seqs

    assert input_modality in ("audio", "text"), input_modality
    assert output_modality in ("audio", "text"), output_modality

    if max_num_seqs is not None and _num_marked_seqs >= max_num_seqs:
        # nothing marked for this batch: RETURNN only enforces a fixed output set if `model_outputs`
        # is configured (it is not here), so the callback simply sees no outputs for these seqs.
        return

    ctx = rf.get_run_ctx()

    target = extern_data[target_data_key]
    labels = target.raw_tensor
    labels_lens = target.dims[1].dyn_size_ext.raw_tensor.to(device=labels.device)

    # input modality -> encoder path + mask token
    if input_modality == "audio":
        forward_func = model.forward_audio
        mask_idx = model.audio_mask_idx
    else:
        forward_func = model.forward_text
        mask_idx = model.text_mask_idx

    # output modality -> decoder (whose cross attention we record) + bos + teacher-forcing func
    if output_modality == "text":
        decoder = model.text_decoder
        bos_idx = model.text_bos_idx
        decode_seq = model.decode_text_seq
    else:
        decoder = model.audio_decoder
        bos_idx = model.audio_bos_idx
        decode_seq = model.decode_audio_seq

    data = extern_data[input_data_key]
    enc_indices = data.raw_tensor.long()
    enc_lens = data.dims[1].dyn_size_ext.raw_tensor.to(device=enc_indices.device)
    # the encoder has no subsampling frontend, so its time dim is the (possibly masked/upsampled)
    # input time dim: reuse the input's dim tag when unchanged, else build a fresh one.
    enc_time_dim = data.dims[1]
    if masking_opts is not None and masking_opts.get("mask_prob", 0.0) > 0.0:
        mask = get_random_mask(enc_lens, **masking_opts)
        enc_indices, enc_lens = mask_sequence(enc_indices, enc_lens, mask, mask_value=mask_idx)
        enc_time_dim = None
    if expansion_opts is not None:
        enc_indices, enc_lens = expand_sequence(enc_indices, enc_lens, **expansion_opts)
        enc_time_dim = None
    if enc_time_dim is None:
        enc_time_dim = _dyn_time_dim(enc_lens, "enc_time")

    encoder_output, _, encoder_lens, _ = forward_func(enc_indices, enc_lens)

    # teacher forcing: decoder input = bos + labels (the last position predicts eos)
    input_labels = F.pad(labels, (1, 0), "constant", value=bos_idx)
    input_labels_lens = labels_lens + 1

    with _capture_cross_attention(decoder) as captured:
        # pass a copy: decode_audio_seq shifts the labels into the shared vocab's audio range in-place
        decode_seq(input_labels.clone(), input_labels_lens, encoder_output, encoder_lens)

    # [B, layer, head, query, key]
    att_weights = torch.stack([captured[i] for i in sorted(captured)], dim=1).float()

    layer_dim = Dim(att_weights.shape[1], name="dec_layer")
    head_dim = Dim(att_weights.shape[2], name="att_head")
    query_dim = _dyn_time_dim(input_labels_lens, "query_time")

    att_weights_rf = rf.convert_to_tensor(
        att_weights, dims=[batch_dim, layer_dim, head_dim, query_dim, enc_time_dim]
    )
    ctx.mark_as_output(
        att_weights_rf, "att_weights", dims=[batch_dim, layer_dim, head_dim, query_dim, enc_time_dim]
    )

    vocab_dim = Dim(model.text_out_dim if output_modality == "text" else model.audio_out_dim, name="vocab")
    labels_rf = rf.convert_to_tensor(input_labels.int(), dims=[batch_dim, query_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(labels_rf, "labels", dims=[batch_dim, query_dim])

    _num_marked_seqs += int(labels.shape[0])
