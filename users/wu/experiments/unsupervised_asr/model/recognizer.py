"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/recognizer.py.

Port note: the optional content tap / head of the source (``content_k`` / ``content_layer``,
``CONTENT_TAPS``, ``forward_with_content``, ``has_content_head``) is CUT: no in-scope config states
it (the S3b-F content term is not ported).  The two constructor arguments are kept so a call that
names them still resolves, and anything but ``None`` raises ``ValueError``.  With them at ``None``
the source built no module, drew nothing from the RNG and ran the same statements in ``forward``
that run here, in the same order.

emc.recognizer -- theta, the §4a recognizer (SAE_4A.md:66-81, amended 2026-09-15).

Frozen wav2vec2-lv60 layer-15 features (1024-d @ 50 Hz, ``feature_dump.py``) in, per-frame CTC
log-probabilities over ``blank + 39 ARPAbet + SIL`` out.  The trainable part is the
wav2vec-U 2.0 *generator* shape, mirrored from the fairseq reference implementation
``fairseq/examples/wav2vec/unsupervised/models/wav2vec_u.py:288-367`` (``Generator``):

    BatchNorm1d(in_dim) over NON-PADDED frames only   (Generator.bn_padded_data, :362-367)
    residual  x <- x + in_proj(dropout(x))            (Generator.forward, :327-330)
    dropout                                           (:332)
    Conv1d(in_dim -> n_out, kernel, stride, padding=kernel//2, bias=False)   (:302-314)

with the hyper-parameters read off the §1c run's own resolved config
(``FairseqW2vu2TrainJob.HOb2GgtYT7Bc/output/train.log:2``, printed module at :20-29):
``generator_kernel 9, generator_dilation 1, generator_pad -1, generator_bias False,
generator_dropout 0.1, generator_batch_norm 30, generator_residual True, input_dim 1024``.

Two deliberate departures from that row, both registered in SAE_4A.md:66-81:

* **stride 1, not 3.**  The §1c generator strides 3 (16.7 Hz); §4a is CTC with blank and at 16.7 Hz
  the ~10 phones/s dev rate leaves 1.7 frames per phone, so fast utterances with repeated phones
  become CTC-infeasible.  ``stride=2`` (25 Hz) is the registered cost ablation and is supported.
  (The blank-free model builds this class at ``n_out = 40, stride = 3``.)
* **41 outputs, not 44.**  ``blank`` at index 0, ``1..39`` the ARPAbet types in
  ``phones.ARPABET_39`` order, ``40`` = SIL (``phones.PHONES`` with ``SIL_ID = 39``, so the emission
  index of a phone id is ``phone_id + 1``).

Output-length rule (dilation 1, padding = kernel // 2, kernel odd):

    T_out = floor((T + 2*(kernel//2) - kernel) / stride) + 1 = floor((T - 1) / stride) + 1
          = ceil(T / stride)

so ``stride=1`` gives T_out = T (the T = S the lattice asserts against the unit store) and
``stride=2`` gives T_out = ceil(T/2).  This is the same rule fairseq gets from slicing its padding
mask with ``[:, ::stride]`` (``wav2vec_u.py:335-336``).

This interface is FIXED; the train step and ``lattice.py`` code against it:
:meth:`ConvRecognizer.forward` returns the log-probabilities and nothing else.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from ..phones import ARPABET_39, N_TYPES, SIL_ID

__all__ = [
    "BLANK_ID",
    "N_OUT",
    "SIL_OUT_ID",
    "OUT_SYMBOLS",
    "emission_index",
    "ConvRecognizer",
    "conv_output_lengths",
]

# ---------------------------------------------------------------------------------------------
# Emission index space.  Index 0 is CTC blank; 1..40 are phones.PHONES in ITS order (39 ARPAbet
# then SIL), i.e. emission index = phone id + 1.  SAE_4A.md:66-81.
# ---------------------------------------------------------------------------------------------
BLANK_ID = 0
N_OUT = N_TYPES + 1  # 41 = blank + 39 ARPAbet + SIL
SIL_OUT_ID = SIL_ID + 1  # 40
OUT_SYMBOLS = ["<blank>"] + list(ARPABET_39) + ["SIL"]
assert len(OUT_SYMBOLS) == N_OUT and OUT_SYMBOLS[SIL_OUT_ID] == "SIL"


def emission_index(phone_id: int) -> int:
    """phones.PHONES id (0..39) -> recognizer output index (1..40)."""
    assert 0 <= phone_id < N_TYPES, phone_id
    return phone_id + 1


def conv_output_lengths(lens: torch.Tensor, stride: int) -> torch.Tensor:
    """ceil(lens / stride) -- the output-length rule documented in the module docstring."""
    return torch.div(lens + (stride - 1), stride, rounding_mode="floor")


class ConvRecognizer(nn.Module):
    """The w2v-U 2.0 generator at stride 1 over raw 1024-d L15 features, 41 CTC outputs.

    :param in_dim: feature dimension; 1024 = wav2vec2-Large-lv60 layer 15 (setup report §2).
    :param n_out: 41 = blank + 39 ARPAbet + SIL.
    :param kernel: 9, the §1c generator kernel (180 ms of context at 50 Hz).
    :param stride: 1 primary; 2 is the registered 25 Hz cost ablation (SAE_4A.md:76-79).
    :param hidden: width of the intermediate map when ``n_layers == 2``.
    :param n_layers: 1 (the fairseq generator, depth 1) or 2.  The second layer is POINTWISE
        (kernel 1) on purpose: it keeps both the 180 ms receptive field that the 2026-09-15
        amendment pins and the output-length rule of the single-conv generator, so the two depths
        are comparable at fixed context.  A second kernel-9 conv would widen the context to 17
        frames / 340 ms, which the amendment does not authorise.
    :param dropout: 0.1, the §1c ``generator_dropout``.
    :param batch_norm: the §1c ``generator_batch_norm`` = 30, which in fairseq is BOTH the on/off
        flag and the constant the BN weight is initialised to (``wav2vec_u.py:316-318``).
        ``0`` disables the BN, exactly as in fairseq.
    :param residual: the §1c ``generator_residual`` = True.
    :param bias: the §1c ``generator_bias`` = False (conv bias).
    :param content_k: removed in the port (the content head); must be ``None``.
    :param content_layer: removed in the port (the content head); must be ``None``.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        n_out: int = N_OUT,
        kernel: int = 9,
        stride: int = 1,
        hidden: int = 512,
        n_layers: int = 1,
        dropout: float = 0.1,
        batch_norm: float = 30.0,
        residual: bool = True,
        bias: bool = False,
        content_k: Optional[int] = None,
        content_layer: Optional[Union[int, str]] = None,
    ):
        super().__init__()
        if content_k is not None or content_layer is not None:
            raise ValueError(
                f"content_k = {content_k!r} / content_layer = {content_layer!r}: the recognizer's "
                "content tap and head (the S3b-F content term) were removed in the port; both must "
                "be None"
            )
        assert n_layers in (1, 2), f"n_layers must be 1 or 2 (SAE_4A.md:66-67), got {n_layers}"
        assert kernel % 2 == 1, f"even kernels break the padding=kernel//2 length rule: {kernel}"
        assert stride >= 1, stride
        self.in_dim = in_dim
        self.n_out = n_out
        self.kernel = kernel
        self.stride = stride
        self.n_layers = n_layers
        self.use_batch_norm = bool(batch_norm)
        self.use_residual = bool(residual)

        self.dropout = nn.Dropout(dropout)
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(in_dim)
            self.bn.weight.data.fill_(float(batch_norm))  # wav2vec_u.py:316-318
        if self.use_residual:
            self.in_proj = nn.Linear(in_dim, in_dim)  # wav2vec_u.py:319-320

        padding = kernel // 2  # generator_pad = -1 -> kernel // 2 (wav2vec_u.py:299-301)
        if n_layers == 1:
            self.conv = nn.Conv1d(in_dim, n_out, kernel, stride=stride, padding=padding, bias=bias)
            self.mid = None
        else:
            self.mid = nn.Conv1d(in_dim, hidden, kernel, stride=stride, padding=padding, bias=bias)
            self.act = nn.GELU()
            self.conv = nn.Conv1d(hidden, n_out, 1, stride=1, padding=0, bias=bias)

    # -- helpers ------------------------------------------------------------------------------
    def output_lengths(self, lens: torch.Tensor) -> torch.Tensor:
        """[B] input frame counts -> [B] output frame counts (ceil(T / stride))."""
        return conv_output_lengths(lens, self.stride)

    @property
    def out_logit_layer(self) -> nn.Conv1d:
        """The layer whose weights the flat init zeroes (``FlatRecognizerInitJob``)."""
        return self.conv

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def valid_mask(lens: torch.Tensor, max_len: int) -> torch.Tensor:
        """[B, T] bool, True on real frames."""
        ar = torch.arange(max_len, device=lens.device)
        return ar[None, :] < lens[:, None]

    # -- forward ------------------------------------------------------------------------------
    def forward(self, feats: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """[B, T, in_dim] float + [B] int -> [B, T_out, n_out] log-probabilities.

        Only the log-probs are returned (the interface the train step and ``lattice.py`` code
        against); the caller gets the output frame counts from :meth:`output_lengths`, which is a
        pure function of ``lens`` and the stride.

        ``feats`` may be float16 (the dump is fp16); it is cast to the module's parameter dtype,
        because a BatchNorm over fp16 activations is numerically pointless at this width.
        Padded frames are zeroed before the convolution so a batch's log-probs do not depend on
        what happened to sit in the padding.
        """
        assert feats.dim() == 3 and feats.shape[-1] == self.in_dim, feats.shape
        assert lens.dim() == 1 and lens.shape[0] == feats.shape[0], (lens.shape, feats.shape)
        param_dtype = self.conv.weight.dtype
        x = feats.to(param_dtype)
        mask = self.valid_mask(lens.to(feats.device), x.shape[1])  # [B, T]

        if self.use_batch_norm:
            # fairseq Generator.bn_padded_data (wav2vec_u.py:362-367): BN over the valid frames only
            normed = x.clone()
            valid = x[mask]
            if valid.numel():
                normed[mask] = self.bn(valid.unsqueeze(-1)).squeeze(-1)
            x = normed
        if self.use_residual:
            x = x + self.in_proj(self.dropout(x))  # wav2vec_u.py:327-330
        x = self.dropout(x)  # wav2vec_u.py:332
        x = x * mask.unsqueeze(-1).to(x.dtype)

        x = x.transpose(1, 2)  # [B, D, T]  (fairseq's TransposeLast pair)
        if self.mid is not None:
            x = self.act(self.mid(x))
            mid_out = self.dropout(x.transpose(1, 2))  # [B, T_out, hidden]
            x = mid_out.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, T_out, n_out]

        assert int(x.shape[1]) == int(self.output_lengths(torch.tensor([feats.shape[1]]))[0]), (
            x.shape[1],
            feats.shape[1],
            self.stride,
        )
        return torch.log_softmax(x.float(), dim=-1)


