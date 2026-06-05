from dataclasses import dataclass
import json
from typing import Literal

import numpy as np
import torch

from .wav2vec2_pca_label_stats import _subsample_alignment


@dataclass
class ForwardConfig:
    phoneme_means_pt: str
    pooled_variance_pt: str
    output_filename: str = "frame_accuracy.txt"
    output_json_filename: str = "frame_accuracy.json"
    decode_mode: Literal["framewise", "viterbi"] = "framewise"
    target_layer: int = -1
    alignment_subsample_factor: int = 2
    variance_floor: float = 1.0e-5
    use_label_priors: bool = True
    self_loop_prob: float = 0.95
    log_interval_batches: int = 10


def _as_path(path) -> str:
    return path.get_path() if hasattr(path, "get_path") else str(path)


def _load_pt(path):
    return torch.load(_as_path(path), map_location="cpu", weights_only=False)


def _init_transition_log_probs(num_labels: int, self_loop_prob: float) -> torch.Tensor:
    if num_labels <= 1:
        return torch.zeros((num_labels, num_labels), dtype=torch.float32)
    self_loop_prob = min(max(float(self_loop_prob), 1.0e-6), 1.0 - 1.0e-6)
    other_prob = (1.0 - self_loop_prob) / (num_labels - 1)
    trans = torch.full((num_labels, num_labels), np.log(other_prob), dtype=torch.float32)
    trans.fill_diagonal_(np.log(self_loop_prob))
    return trans


def forward_init_hook(run_ctx, **kwargs):
    config = ForwardConfig(**kwargs.get("config", {}))
    means_payload = _load_pt(config.phoneme_means_pt)
    variance_payload = _load_pt(config.pooled_variance_pt)

    run_ctx.label_ids = torch.as_tensor(means_payload["label_ids"], dtype=torch.long)
    run_ctx.means = torch.as_tensor(means_payload["means"], dtype=torch.float32)
    variance = torch.as_tensor(variance_payload["variance"], dtype=torch.float32)
    run_ctx.inv_variance = torch.reciprocal(torch.clamp(variance, min=config.variance_floor))

    if run_ctx.means.ndim != 2:
        raise ValueError(f"Expected mean matrix [labels, dim], got {tuple(run_ctx.means.shape)}")
    if run_ctx.inv_variance.shape != (run_ctx.means.shape[1],):
        raise ValueError(
            f"Expected variance dim {run_ctx.means.shape[1]}, got {tuple(run_ctx.inv_variance.shape)}"
        )

    frame_counts = torch.as_tensor(means_payload.get("frame_counts", np.ones(len(run_ctx.label_ids))), dtype=torch.float32)
    priors = frame_counts / torch.clamp(frame_counts.sum(), min=1.0)
    run_ctx.log_priors = torch.log(torch.clamp(priors, min=1.0e-30))
    run_ctx.transition_log_probs = _init_transition_log_probs(len(run_ctx.label_ids), config.self_loop_prob)

    max_label_id = int(run_ctx.label_ids.max().item()) if len(run_ctx.label_ids) else -1
    run_ctx.label_to_index = torch.full((max_label_id + 1,), -1, dtype=torch.long)
    run_ctx.label_to_index[run_ctx.label_ids] = torch.arange(len(run_ctx.label_ids), dtype=torch.long)

    run_ctx.decode_mode = config.decode_mode
    run_ctx.target_layer = config.target_layer
    run_ctx.alignment_subsample_factor = config.alignment_subsample_factor
    run_ctx.use_label_priors = config.use_label_priors
    run_ctx.output_filename = config.output_filename
    run_ctx.output_json_filename = config.output_json_filename
    run_ctx.log_interval_batches = config.log_interval_batches
    run_ctx.self_loop_prob = config.self_loop_prob

    run_ctx.num_batches = 0
    run_ctx.num_sequences = 0
    run_ctx.num_truncated_sequences = 0
    run_ctx.correct_frames = 0
    run_ctx.total_frames = 0
    run_ctx.correct_by_label = torch.zeros((len(run_ctx.label_ids),), dtype=torch.long)
    run_ctx.total_by_label = torch.zeros((len(run_ctx.label_ids),), dtype=torch.long)


def _emission_scores(features: torch.Tensor, run_ctx) -> torch.Tensor:
    if features.shape[-1] < run_ctx.means.shape[-1]:
        raise ValueError(f"Feature dim {features.shape[-1]} is smaller than mean dim {run_ctx.means.shape[-1]}")
    features = features[..., : run_ctx.means.shape[-1]]
    means = run_ctx.means.to(features.device)
    inv_variance = run_ctx.inv_variance.to(features.device)
    diff = features[:, None, :] - means[None, :, :]
    scores = -0.5 * torch.sum(diff * diff * inv_variance[None, None, :], dim=-1)
    if run_ctx.use_label_priors:
        scores = scores + run_ctx.log_priors.to(features.device)[None, :]
    return scores


def _viterbi_decode(emissions: torch.Tensor, run_ctx) -> torch.Tensor:
    emissions = emissions.float()
    transitions = run_ctx.transition_log_probs.to(emissions.device)
    delta = emissions[0]
    backptrs = []
    for t in range(1, emissions.shape[0]):
        scores = delta[:, None] + transitions
        best_scores, best_prev = torch.max(scores, dim=0)
        delta = best_scores + emissions[t]
        backptrs.append(best_prev)

    state = int(torch.argmax(delta).item())
    path = [state]
    for backptr in reversed(backptrs):
        state = int(backptr[state].item())
        path.append(state)
    path.reverse()
    return torch.tensor(path, dtype=torch.long, device=emissions.device)


def _update_counts(pred_indices: torch.Tensor, label_indices: torch.Tensor, run_ctx):
    pred_indices = pred_indices.detach().cpu()
    label_indices = label_indices.detach().cpu()
    correct = pred_indices == label_indices
    run_ctx.correct_frames += int(correct.sum().item())
    run_ctx.total_frames += int(label_indices.numel())
    run_ctx.correct_by_label += torch.bincount(
        label_indices[correct], minlength=run_ctx.correct_by_label.shape[0]
    ).to(torch.long)
    run_ctx.total_by_label += torch.bincount(label_indices, minlength=run_ctx.total_by_label.shape[0]).to(torch.long)


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)
    alignments = data.get("alignments")
    if alignments is None:
        raise ValueError("Alignments not found in data stream. Check hdf_stream_name in the dataset builder.")
    alignments_len = data["alignments:size1"].to(torch.long)

    all_hidden_states, output_lengths = model.extract_hidden_states(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
    transformed_hidden_states = model.transform_hidden_states(all_hidden_states, output_lengths, update_pca=False)
    layer_features = transformed_hidden_states[model.return_layers.index(run_ctx.target_layer)]

    run_ctx.num_batches += 1
    for batch_idx in range(layer_features.shape[0]):
        feature_len = int(output_lengths[batch_idx].item())
        alignment_len = int(alignments_len[batch_idx].item())
        labels = alignments[batch_idx, :alignment_len].detach().cpu().to(torch.long)
        labels = _subsample_alignment(labels, run_ctx.alignment_subsample_factor)
        effective_len = min(feature_len, int(labels.shape[0]))

        run_ctx.num_sequences += 1
        if effective_len <= 0:
            continue
        if feature_len != int(labels.shape[0]):
            run_ctx.num_truncated_sequences += 1

        labels = labels[:effective_len]
        valid = (labels >= 0) & (labels < run_ctx.label_to_index.shape[0])
        label_indices = torch.full_like(labels, -1)
        label_indices[valid] = run_ctx.label_to_index[labels[valid]]
        valid = label_indices >= 0
        if not bool(valid.any()):
            continue

        features = layer_features[batch_idx, :effective_len][valid]
        label_indices = label_indices[valid]
        emissions = _emission_scores(features, run_ctx)
        if run_ctx.decode_mode == "framewise":
            pred_indices = torch.argmax(emissions, dim=-1)
        elif run_ctx.decode_mode == "viterbi":
            pred_indices = _viterbi_decode(emissions, run_ctx)
        else:
            raise ValueError(f"Unsupported decode_mode {run_ctx.decode_mode!r}")
        _update_counts(pred_indices, label_indices, run_ctx)

    if run_ctx.log_interval_batches and run_ctx.num_batches % run_ctx.log_interval_batches == 0:
        accuracy = run_ctx.correct_frames / max(run_ctx.total_frames, 1)
        print(
            f"[{run_ctx.decode_mode} frame accuracy] batches={run_ctx.num_batches}, "
            f"sequences={run_ctx.num_sequences}, frames={run_ctx.total_frames}, accuracy={accuracy:.6f}"
        )


def forward_finish_hook(run_ctx, **kwargs):
    accuracy = run_ctx.correct_frames / max(run_ctx.total_frames, 1)
    total_by_label = run_ctx.total_by_label.numpy()
    correct_by_label = run_ctx.correct_by_label.numpy()
    per_label_accuracy = correct_by_label / np.maximum(total_by_label, 1)
    label_ids = run_ctx.label_ids.numpy()

    payload = {
        "decode_mode": run_ctx.decode_mode,
        "accuracy": float(accuracy),
        "correct_frames": int(run_ctx.correct_frames),
        "total_frames": int(run_ctx.total_frames),
        "num_batches": int(run_ctx.num_batches),
        "num_sequences": int(run_ctx.num_sequences),
        "truncated_sequences": int(run_ctx.num_truncated_sequences),
        "use_label_priors": bool(run_ctx.use_label_priors),
        "self_loop_prob": float(run_ctx.self_loop_prob),
        "label_ids": label_ids.astype(int).tolist(),
        "total_by_label": total_by_label.astype(int).tolist(),
        "correct_by_label": correct_by_label.astype(int).tolist(),
        "per_label_accuracy": per_label_accuracy.tolist(),
    }

    with open(run_ctx.output_filename, "w", encoding="utf-8") as f:
        f.write(f"decode_mode {run_ctx.decode_mode}\n")
        f.write(f"accuracy {accuracy:.10f}\n")
        f.write(f"correct_frames {run_ctx.correct_frames}\n")
        f.write(f"total_frames {run_ctx.total_frames}\n")
        f.write(f"num_batches {run_ctx.num_batches}\n")
        f.write(f"num_sequences {run_ctx.num_sequences}\n")
        f.write(f"truncated_sequences {run_ctx.num_truncated_sequences}\n")
        f.write("label_id\ttotal\tcorrect\taccuracy\n")
        for label_id, total, correct, label_acc in zip(label_ids, total_by_label, correct_by_label, per_label_accuracy):
            f.write(f"{int(label_id)}\t{int(total)}\t{int(correct)}\t{float(label_acc):.10f}\n")

    with open(run_ctx.output_json_filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(
        f"[{run_ctx.decode_mode} frame accuracy] done: frames={run_ctx.total_frames}, "
        f"correct={run_ctx.correct_frames}, accuracy={accuracy:.6f}"
    )
