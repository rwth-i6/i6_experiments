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
_batch_size_factor = 160  # raw audio (16 kHz) -> batch_size is in 10ms-frame-equivalents, as in the AED baseline


class StreamingModel(rf.Module):
    """
    Chunked-encoder + pluggable streaming decoder.

    The decoder vocab (``target_dim_ext``) is the spm vocab plus one end-of-chunk
    (EOC) marker at the last index -- it is passed in directly (it is the sparse
    dim of the ``aug_targets`` supervision stream, so cross-entropy dim-matches
    ``extern_data``). The raw spm dim (for the aux CTC head) is derived from it.

    Mirrors the relevant parts of :class:`...exp2024_04_23_baselines.aed.Model`
    (frontend, SpecAugment, aux CTC) but with a chunked encoder and a streaming
    decoder built from ``dec_build_dict``.
    """

    def __init__(
        self,
        *,
        target_dim_ext: Dim,
        chunk_size: int,
        enc_build_dict: Dict[str, Any],
        dec_build_dict: Dict[str, Any],
        enc_aux_logits: Sequence[int] = (),
        bos_idx: int = 0,
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

        # Decoder vocab incl. EOC (matches the aug_targets stream); raw spm vocab is one smaller.
        self.target_dim_ext = target_dim_ext
        self.eoc_idx = target_dim_ext.dimension - 1
        self.target_dim = Dim(target_dim_ext.dimension - 1, name="spm_vocab")  # raw labels, for aux CTC
        self.blank_idx = self.target_dim.dimension  # CTC blank (own label space; numerically == eoc_idx)
        self.bos_idx = bos_idx

        # Streaming decoder (chunkwise / framewise / ...).
        self.decoder = rf.build_from_dict(
            dec_build_dict,
            encoder_dim=self.encoder.out_dim,
            vocab_dim=self.target_dim_ext,
            chunk_size=chunk_size,
            eoc_idx=self.eoc_idx,
        )

        # Aux CTC heads over the with-blank spm vocab (own label space, internal only).
        self.wb_target_dim = self.target_dim + 1  # [labels..., blank]
        for layer_idx in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{layer_idx}", rf.Linear(self.encoder.out_dim, self.wb_target_dim))
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


def streaming_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> StreamingModel:
    """ModelDef: build a :class:`StreamingModel` from the global config.

    ``target_dim`` is the default-target (``aug_targets``) sparse dim, i.e. the
    EOC-extended decoder vocab. The variant is selected by ``dec_build_dict``.
    """
    from returnn.config import get_global_config

    in_dim, epoch  # noqa  (raw audio input; model builds its own logmel frontend)
    config = get_global_config()
    return StreamingModel(
        target_dim_ext=target_dim,
        chunk_size=config.int("chunk_size", 0),
        enc_build_dict=config.typed_value("enc_build_dict"),
        dec_build_dict=config.typed_value("dec_build_dict"),
        enc_aux_logits=config.typed_value("aux_loss_layers") or (),
        bos_idx=config.int("bos_idx", 0),
    )


streaming_model_def.behavior_version = 25
streaming_model_def.backend = "torch"
streaming_model_def.batch_size_factor = _batch_size_factor


def encoder_frame_chunk_idx(enc_spatial_dim: Dim, chunk_size: int) -> Tensor:
    """int [enc_spatial_dim] giving the chunk index ``frame // chunk_size`` of each encoder frame."""
    return rf.range_over_dim(enc_spatial_dim) // chunk_size
