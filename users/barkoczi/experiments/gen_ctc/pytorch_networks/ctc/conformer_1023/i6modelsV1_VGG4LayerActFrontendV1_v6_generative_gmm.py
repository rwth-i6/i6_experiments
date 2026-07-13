import pickle
from typing import Dict

import torch
import torch.nn.functional as F

from i6_experiments.users.barkoczi.experiments.gen_ctc.loss.generative_loss import generative_nce

from .i6modelsV1_VGG4LayerActFrontendV1_v6_generative_conv_first import (
    Model as BaseModel,
    prior_finish_hook,
    prior_init_hook,
    prior_step,
)


def _path_str(path) -> str:
    return path.get_path() if hasattr(path, "get_path") else str(path)


def _read_returnn_vocab(path) -> Dict[str, int]:
    with open(_path_str(path), "rb") as f:
        vocab = pickle.load(f)
    if not isinstance(vocab, dict) or not all(
        isinstance(label, str) and isinstance(idx, int) for label, idx in vocab.items()
    ):
        raise ValueError(f"Expected a RETURNN vocabulary dict in {_path_str(path)!r}")
    return vocab


def _read_label_map(path) -> Dict[str, int]:
    label_map = {}
    with open(_path_str(path), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, idx = line.rsplit(maxsplit=1)
            label_map[label] = int(idx)
    if not label_map:
        raise ValueError(f"Alignment label map {_path_str(path)!r} is empty")
    return label_map


def _make_alignment_to_ctc_map(*, alignment_label_map, target_vocab, blank_index: int, silence_label: str):
    source_labels = _read_label_map(alignment_label_map)
    target_labels = _read_returnn_vocab(target_vocab)
    mapping = torch.full((max(source_labels.values()) + 1,), -1, dtype=torch.long)
    for label, source_idx in source_labels.items():
        if label == silence_label:
            target_idx = blank_index
        elif label in target_labels:
            target_idx = target_labels[label]
        else:
            raise ValueError(
                f"GMM alignment label {label!r} has no matching CTC label in {_path_str(target_vocab)!r}"
            )
        mapping[source_idx] = target_idx
    if torch.any(mapping < 0):
        missing_ids = torch.nonzero(mapping < 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"GMM alignment label map has unmapped indices: {missing_ids}")
    return mapping


class Model(BaseModel):
    def __init__(
        self,
        model_config_dict,
        *,
        alignment_label_map,
        target_vocab,
        alignment_subsampling_factor: int = 4,
        silence_label: str = "[SILENCE]",
        **kwargs,
    ):
        super().__init__(model_config_dict=model_config_dict, **kwargs)
        if alignment_subsampling_factor <= 0:
            raise ValueError(f"alignment_subsampling_factor must be positive, got {alignment_subsampling_factor}")
        self.alignment_subsampling_factor = alignment_subsampling_factor
        mapping = _make_alignment_to_ctc_map(
            alignment_label_map=alignment_label_map,
            target_vocab=target_vocab,
            blank_index=self.cfg.label_target_size,
            silence_label=silence_label,
        )
        self.register_buffer("alignment_to_ctc_map", mapping, persistent=False)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to("cpu")
    alignments = data["alignments"].to(torch.long)
    alignment_lengths = data["alignments:size1"].to(torch.long)

    log_probs, audio_features_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
    factor = model.alignment_subsampling_factor
    downsampled_alignments = alignments[:, ::factor]
    shared_time_dim = min(log_probs.shape[1], downsampled_alignments.shape[1])
    if shared_time_dim == 0:
        return

    downsampled_alignments = downsampled_alignments[:, :shared_time_dim]
    if torch.any(downsampled_alignments < 0) or torch.any(
        downsampled_alignments >= model.alignment_to_ctc_map.shape[0]
    ):
        raise ValueError("GMM alignment contains a label id outside the configured alignment label map")
    remapped_alignments = model.alignment_to_ctc_map[downsampled_alignments]
    hard_targets = F.one_hot(remapped_alignments, num_classes=log_probs.shape[-1]).to(log_probs.dtype)

    valid_seq_len = torch.div(alignment_lengths, factor, rounding_mode="floor").to(audio_features_len.device)
    valid_seq_len = torch.minimum(valid_seq_len, audio_features_len)
    valid_seq_len = torch.clamp(valid_seq_len, max=shared_time_dim)
    loss = generative_nce(
        log_probs[:, :shared_time_dim],
        hard_targets,
        sampling_type=model.cfg.sampling_type,
        seq_len=valid_seq_len,
        sampling_ratio=model.cfg.sampling_ratio,
        share_samples=model.cfg.share_samples,
        ratio_corrector=model.cfg.ratio_corrector,
    )
    inv_norm_factor = torch.clamp(valid_seq_len.sum().to(dtype=torch.float32), min=1.0)
    run_ctx.mark_as_loss(name="gmm_generative_nce", loss=loss.sum(), inv_norm_factor=inv_norm_factor)
