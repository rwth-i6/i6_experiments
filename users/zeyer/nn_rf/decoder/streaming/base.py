"""
Shared infrastructure for the streaming chunked-decoder variants
(slow-fast-rna project).

The streaming decoders all share:
- a chunked Conformer encoder (``ChunkedConformerEncoderV2``) whose output frame
  ``t`` belongs to chunk ``t // chunk_size``;
- an EOC-extended target vocab (the spm vocab plus one end-of-chunk marker);
- the audio frontend (logmel + optional feature batch-norm + SpecAugment), reused
  from the offline AED baseline so the encoder sees identical features;
- per-chunk supervision from :mod:`.segmentation`.

The per-variant decoder (chunkwise / framewise / slow_fast / ...) lives in its own
module and is plugged into :class:`StreamingModel` via ``dec_build_dict``.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, Sequence, Tuple
import functools

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim


_log_mel_feature_dim = 80


class StreamingModel(rf.Module):
    """
    Chunked-encoder + pluggable streaming decoder, with an EOC-extended target vocab.

    Mirrors the relevant parts of :class:`...exp2024_04_23_baselines.aed.Model`
    (frontend, SpecAugment, aux CTC) but with a chunked encoder and a streaming
    decoder built from ``dec_build_dict``.
    """

    def __init__(
        self,
        *,
        target_dim: Dim,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        chunk_size: int,
        enc_build_dict: Dict[str, Any],
        dec_build_dict: Dict[str, Any],
        enc_aux_logits: Sequence[int] = (),
        feature_extraction: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        # Feature frontend (logmel).
        feat_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
        if not feature_extraction:
            feature_extraction = rf.build_dict(
                rf.Functional,
                func=functools.partial(rf.audio.log_mel_filterbank_from_raw, sampling_rate=16_000, out_dim=feat_dim),
                attribs={"out_dim": feat_dim},
            )
        self.feature_extraction = rf.build_from_dict(feature_extraction)
        self.in_dim = self.feature_extraction.out_dim

        self.feature_batch_norm = None
        if config.bool("feature_batch_norm", False):
            self.feature_batch_norm = rf.BatchNorm(self.in_dim, affine=False, use_mask=True)

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (self.in_dim.dimension // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        # Chunked encoder.
        self.encoder = rf.build_from_dict(enc_build_dict, self.in_dim)
        self.chunk_size = chunk_size

        self.target_dim = target_dim  # spm labels (no EOC)
        # EOC-extended vocab: labels [0..V-1] + EOC at index V.
        self.eoc_idx = target_dim.dimension
        self.target_dim_ext = target_dim + 1  # [labels..., EOC]
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx

        # Streaming decoder (chunkwise / framewise / ...).
        self.decoder = rf.build_from_dict(
            dec_build_dict,
            encoder_dim=self.encoder.out_dim,
            vocab_dim=self.target_dim_ext,
            chunk_size=chunk_size,
            eoc_idx=self.eoc_idx,
        )

        # Aux CTC heads (over with-blank vocab), same wiring idea as the AED model.
        wb_target_dim = target_dim + 1  # [labels..., blank]
        self.wb_target_dim = wb_target_dim
        for layer_idx in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{layer_idx}", rf.Linear(self.encoder.out_dim, wb_target_dim))
        self.enc_aux_logits = tuple(enc_aux_logits)

    def encode(
        self, source: Tensor, *, in_spatial_dim: Dim, collected_outputs: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Dim]:
        """raw audio -> chunked encoder output [B, enc_spatial, D], enc_spatial_dim."""
        if source.feature_dim and source.feature_dim.dimension == 1:
            source = rf.squeeze(source, axis=source.feature_dim)
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source, in_spatial_dim=in_spatial_dim, out_dim=self.in_dim, sampling_rate=16_000
        )
        if self.feature_batch_norm is not None:
            source = self.feature_batch_norm(source)
        source = rf.audio.specaugment(
            source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim, **self._specaugment_opts
        )
        enc, enc_spatial_dim = self.encoder(
            source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
        )
        return enc, enc_spatial_dim


def encoder_frame_chunk_idx(enc_spatial_dim: Dim, chunk_size: int) -> Tensor:
    """int [enc_spatial_dim] giving the chunk index ``frame // chunk_size`` of each encoder frame."""
    return rf.range_over_dim(enc_spatial_dim) // chunk_size
