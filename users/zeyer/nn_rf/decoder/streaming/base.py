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
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        return enc, enc_spatial_dim

    def aux_logits_from_collected_outputs(self, aux_layer: int, collected_outputs: Dict[str, Tensor]) -> Tensor:
        """Aux CTC logits for ``aux_layer`` (1-indexed) from the encoder's per-layer ``collected_outputs``."""
        linear: rf.Linear = getattr(self, f"enc_aux_logits_{aux_layer}")
        return linear(collected_outputs[str(aux_layer - 1)])

    def aux_ctc_losses(
        self, *, collected_outputs: Dict[str, Tensor], raw_targets: Tensor, raw_spatial_dim: Dim, enc_spatial_dim: Dim
    ) -> Dict[str, Tuple[Tensor, Dim]]:
        """CTC aux loss for every configured ``enc_aux_logits`` layer, each tapping its own encoder layer."""
        losses: Dict[str, Tuple[Tensor, Dim]] = {}
        for layer_idx in self.enc_aux_logits:
            aux_logits = self.aux_logits_from_collected_outputs(layer_idx, collected_outputs)
            aux_log_probs = rf.log_softmax(aux_logits, axis=self.wb_target_dim)
            ctc = rf.ctc_loss(
                logits=aux_log_probs,
                logits_normalized=True,
                targets=raw_targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=raw_spatial_dim,
                blank_index=self.blank_idx,
            )
            losses[f"ctc_{layer_idx}"] = (ctc, raw_spatial_dim)
        return losses


def streaming_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> StreamingModel:
    """ModelDef: build a :class:`StreamingModel` from the global config.

    ``target_dim`` is the default-target (``aug_targets``) sparse dim, i.e. the
    EOC-extended decoder vocab. The variant is selected by ``dec_build_dict``.

    At recog the eval dataset's default target is the raw spm vocab (one label
    smaller, no EOC), so the model would be built too small to load the checkpoint.
    Set ``target_dim_ext_int`` (+ optional ``aug_vocab`` opts, so the output dim
    carries a vocab for rendering) in the (search) config to build the model's own
    EOC-extended dim regardless of ``target_dim``. Training leaves it unset and so
    keeps using the dataset's aug-targets dim (hash-stable, no re-train).
    """
    from returnn.config import get_global_config

    in_dim, epoch  # noqa  (raw audio input; model builds its own logmel frontend)
    config = get_global_config()
    target_dim_ext_int = config.int("target_dim_ext_int", 0)
    if target_dim_ext_int:
        target_dim = Dim(target_dim_ext_int, name="aug_vocab")
        aug_vocab_opts = config.typed_value("aug_vocab")
        if aug_vocab_opts:
            from returnn.datasets.util.vocabulary import Vocabulary

            target_dim.vocab = Vocabulary.create_vocab(**aug_vocab_opts)
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


def model_recog_ctc(*, model: StreamingModel, data: Tensor, data_spatial_dim: Dim):
    """
    Greedy CTC recog using the encoder's aux CTC head (on the final encoder output).

    Encoder-quality probe: CTC marginalizes over alignments and ignores the streaming decoder,
    so it isolates the encoder from the alignment-trained decoder. RecogDef signature.
    """
    assert model.enc_aux_logits, "model_recog_ctc needs an aux CTC head (set aux_loss_layers)"
    collected_outputs = {}
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    layer_idx = model.enc_aux_logits[-1]
    aux_logits = model.aux_logits_from_collected_outputs(layer_idx, collected_outputs)
    labels, out_spatial_dim = rf.ctc_greedy_decode(
        aux_logits, in_spatial_dim=enc_spatial_dim, blank_index=model.blank_idx, wb_target_dim=model.wb_target_dim
    )
    # Render with the EOC-extended vocab; blank-removed ids (< blank) are exactly the raw spm pieces.
    labels.sparse_dim = model.target_dim_ext
    beam_dim = Dim(1, name="beam")
    labels = rf.expand_dim(labels, dim=beam_dim)
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    seq_log_prob = rf.zeros([beam_dim, *batch_dims], dtype="float32")
    return labels, seq_log_prob, out_spatial_dim, beam_dim


model_recog_ctc.output_with_beam = True
model_recog_ctc.output_blank_label = None
model_recog_ctc.batch_size_dependent = False


def label_smoothed_log_probs(log_probs: Tensor, *, axis: Dim) -> Tensor:
    """Config-gated label smoothing on the decoder CE log-probs (gradient-only,
    via :func:`rf.label_smoothed_log_prob_gradient`, as in the AED baseline).

    ``label_smoothing`` config float, default 0 = off -- opt-in,
    so existing finished trainings (which do not set it) are unaffected.
    """
    from returnn.config import get_global_config

    config = get_global_config(return_empty_if_none=True)
    smoothing = config.float("label_smoothing", 0.0)
    if smoothing:
        log_probs = rf.label_smoothed_log_prob_gradient(log_probs, smoothing, axis=axis)
    return log_probs


def rna_targets_on_enc_spatial(
    rna_targets: Tensor, *, in_spatial_dim: Dim, enc_spatial_dim: Dim, blank_idx: int
) -> Tensor:
    """Re-align the per-frame RNA target onto the encoder output length (per seq).

    The dataset pads the RNA target to a fixed chunk multiple of the alignment length,
    while the encoder pads its output to a multiple of ITS chunk size
    (which can differ, e.g. a dynamic chunk_size_train_pool, or no padding for chunk_size=None),
    so the two per-seq lengths can differ in both directions.
    Every frame beyond the alignment length is blank on the target side and padding on the encoder side,
    so pad with blank / truncate (only ever cutting blank padding) to exactly the encoder length.
    With equal lengths this is just the old dim re-tag, numerically identical.
    """
    idx = rf.range_over_dim(enc_spatial_dim, device=rna_targets.device)  # [enc_spatial]
    tgt_lens = rf.copy_to_device(in_spatial_dim.get_size_tensor(), rna_targets.device)  # [B]
    valid = idx < tgt_lens  # [B, enc_spatial]
    gathered = rf.gather(rna_targets, indices=rf.minimum(idx, tgt_lens - 1), axis=in_spatial_dim)
    blank = rf.constant(blank_idx, dims=(), sparse_dim=rna_targets.sparse_dim, dtype=rna_targets.dtype)
    return rf.where(valid, gathered, blank)


def encoder_frame_chunk_idx(enc_spatial_dim: Dim, chunk_size: int) -> Tensor:
    """int [enc_spatial_dim] giving the chunk index ``frame // chunk_size`` of each encoder frame."""
    return rf.range_over_dim(enc_spatial_dim) // chunk_size
