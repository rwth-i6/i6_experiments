from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ForwardConfig:
    pooled_variance_filename: str = "pooled_variance.txt"
    pooled_variance_npy_filename: str = "pooled_variance.npy"
    pooled_variance_pt_filename: str = "pooled_variance.pt"
    phoneme_means_filename: str = "phoneme_feature_means.txt"
    phoneme_means_npy_filename: str = "phoneme_feature_means.npy"
    phoneme_means_pt_filename: str = "phoneme_feature_means.pt"
    pooled_variance_filename_template: Optional[str] = None
    pooled_variance_npy_filename_template: Optional[str] = None
    pooled_variance_pt_filename_template: Optional[str] = None
    phoneme_means_filename_template: Optional[str] = None
    phoneme_means_npy_filename_template: Optional[str] = None
    phoneme_means_pt_filename_template: Optional[str] = None
    pca_dims: Optional[List[int]] = None
    target_layer: int = -1
    returnn_vocab: Optional[str] = None
    log_interval_batches: int = 10
    alignment_subsample_factor: int = 2


def _subsample_alignment(labels: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return labels
    return labels[::factor]


def _label_id_to_symbol(run_ctx, label_id: int) -> str:
    if run_ctx.vocab_labels is None:
        return f"id_{label_id}"
    if 0 <= label_id < len(run_ctx.vocab_labels):
        return str(run_ctx.vocab_labels[label_id])
    return f"id_{label_id}"


def _normalize_requested_dims(pca_dims: Optional[List[int]]) -> Optional[List[int]]:
    if pca_dims is None:
        return None
    dims = sorted({int(dim) for dim in pca_dims})
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"All requested pca_dims must be positive, got {dims}")
    return dims


def _format_filename(template: str, *, dim: int, target_layer: int) -> str:
    return template.format(pca_dim=dim, dim=dim, target_layer=target_layer, layer=target_layer)


def _get_output_filenames(run_ctx, dim: int) -> Tuple[str, str, str, str, str, str]:
    if run_ctx.pca_dims is None:
        return (
            run_ctx.pooled_variance_filename,
            run_ctx.pooled_variance_npy_filename,
            run_ctx.pooled_variance_pt_filename,
            run_ctx.phoneme_means_filename,
            run_ctx.phoneme_means_npy_filename,
            run_ctx.phoneme_means_pt_filename,
        )
    if run_ctx.pooled_variance_filename_template is None:
        raise ValueError("pooled_variance_filename_template must be provided when pca_dims is set.")
    if run_ctx.pooled_variance_npy_filename_template is None:
        raise ValueError("pooled_variance_npy_filename_template must be provided when pca_dims is set.")
    if run_ctx.pooled_variance_pt_filename_template is None:
        raise ValueError("pooled_variance_pt_filename_template must be provided when pca_dims is set.")
    if run_ctx.phoneme_means_filename_template is None:
        raise ValueError("phoneme_means_filename_template must be provided when pca_dims is set.")
    if run_ctx.phoneme_means_npy_filename_template is None:
        raise ValueError("phoneme_means_npy_filename_template must be provided when pca_dims is set.")
    if run_ctx.phoneme_means_pt_filename_template is None:
        raise ValueError("phoneme_means_pt_filename_template must be provided when pca_dims is set.")
    return (
        _format_filename(run_ctx.pooled_variance_filename_template, dim=dim, target_layer=run_ctx.target_layer),
        _format_filename(run_ctx.pooled_variance_npy_filename_template, dim=dim, target_layer=run_ctx.target_layer),
        _format_filename(run_ctx.pooled_variance_pt_filename_template, dim=dim, target_layer=run_ctx.target_layer),
        _format_filename(run_ctx.phoneme_means_filename_template, dim=dim, target_layer=run_ctx.target_layer),
        _format_filename(run_ctx.phoneme_means_npy_filename_template, dim=dim, target_layer=run_ctx.target_layer),
        _format_filename(run_ctx.phoneme_means_pt_filename_template, dim=dim, target_layer=run_ctx.target_layer),
    )


def _ensure_dim_state(run_ctx, dim: int) -> None:
    if dim not in run_ctx.feature_sums:
        run_ctx.feature_sums[dim] = None
        run_ctx.feature_sum_sqs[dim] = None
        run_ctx.label_sums_by_dim[dim] = {}
        run_ctx.active_pca_dims.append(dim)


def _get_active_dims(run_ctx, feature_dim: int) -> List[int]:
    if run_ctx.pca_dims is None:
        dims = [feature_dim]
    else:
        dims = run_ctx.pca_dims
        if any(dim > feature_dim for dim in dims):
            raise ValueError(f"Requested pca_dims {dims} exceed available feature dim {feature_dim}.")
    for dim in dims:
        _ensure_dim_state(run_ctx, dim)
    return dims


def forward_init_hook(run_ctx, **kwargs):
    config = ForwardConfig(**kwargs.get("config", {}))
    run_ctx.target_layer = config.target_layer
    run_ctx.pooled_variance_filename = config.pooled_variance_filename
    run_ctx.pooled_variance_npy_filename = config.pooled_variance_npy_filename
    run_ctx.pooled_variance_pt_filename = config.pooled_variance_pt_filename
    run_ctx.phoneme_means_filename = config.phoneme_means_filename
    run_ctx.phoneme_means_npy_filename = config.phoneme_means_npy_filename
    run_ctx.phoneme_means_pt_filename = config.phoneme_means_pt_filename
    run_ctx.pooled_variance_filename_template = config.pooled_variance_filename_template
    run_ctx.pooled_variance_npy_filename_template = config.pooled_variance_npy_filename_template
    run_ctx.pooled_variance_pt_filename_template = config.pooled_variance_pt_filename_template
    run_ctx.phoneme_means_filename_template = config.phoneme_means_filename_template
    run_ctx.phoneme_means_npy_filename_template = config.phoneme_means_npy_filename_template
    run_ctx.phoneme_means_pt_filename_template = config.phoneme_means_pt_filename_template
    run_ctx.pca_dims = _normalize_requested_dims(config.pca_dims)
    run_ctx.feature_sums: Dict[int, Optional[np.ndarray]] = {}
    run_ctx.feature_sum_sqs: Dict[int, Optional[np.ndarray]] = {}
    run_ctx.feature_count = 0
    run_ctx.label_sums_by_dim: Dict[int, Dict[int, np.ndarray]] = {}
    run_ctx.label_counts: Dict[int, int] = {}
    run_ctx.active_pca_dims: List[int] = []
    run_ctx.num_batches = 0
    run_ctx.num_sequences = 0
    run_ctx.num_truncated_sequences = 0
    run_ctx.vocab_labels = None
    run_ctx.log_interval_batches = config.log_interval_batches
    run_ctx.alignment_subsample_factor = config.alignment_subsample_factor

    if config.returnn_vocab is not None:
        from returnn.datasets.util.vocabulary import Vocabulary

        vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, unknown_label=None)
        run_ctx.vocab_labels = vocab.labels


def forward_finish_hook(run_ctx, **kwargs):
    if run_ctx.feature_count == 0:
        raise ValueError("No aligned frames were observed while computing PCA feature statistics.")

    print(
        f"[PCA label stats] done: batches={run_ctx.num_batches}, sequences={run_ctx.num_sequences}, "
        f"frames={run_ctx.feature_count}, truncated_sequences={run_ctx.num_truncated_sequences}"
    )

    if not run_ctx.active_pca_dims:
        raise ValueError("No PCA dimensions were active while computing PCA feature statistics.")

    label_ids = np.array(sorted(run_ctx.label_counts.keys()), dtype=np.int64)
    frame_counts = np.array([run_ctx.label_counts[label_id] for label_id in label_ids], dtype=np.int64)
    label_symbols = np.array([_label_id_to_symbol(run_ctx, int(label_id)) for label_id in label_ids], dtype=object)

    for dim in run_ctx.active_pca_dims:
        pooled_mean = run_ctx.feature_sums[dim] / run_ctx.feature_count
        pooled_variance = run_ctx.feature_sum_sqs[dim] / run_ctx.feature_count - np.square(pooled_mean)
        pooled_variance = np.maximum(pooled_variance, 0.0)
        pooled_variance_payload = {
            "pca_dim": int(dim),
            "target_layer": int(run_ctx.target_layer),
            "pooled_frame_count": int(run_ctx.feature_count),
            "num_sequences": int(run_ctx.num_sequences),
            "truncated_sequences": int(run_ctx.num_truncated_sequences),
            "variance": pooled_variance,
        }

        (
            pooled_variance_filename,
            pooled_variance_npy_filename,
            pooled_variance_pt_filename,
            phoneme_means_filename,
            phoneme_means_npy_filename,
            phoneme_means_pt_filename,
        ) = _get_output_filenames(run_ctx, dim)

        with open(pooled_variance_filename, "w", encoding="utf-8") as f:
            f.write(f"# pca_dim {dim}\n")
            f.write(f"# target_layer {run_ctx.target_layer}\n")
            f.write(f"# pooled_frame_count {run_ctx.feature_count}\n")
            f.write(f"# num_sequences {run_ctx.num_sequences}\n")
            f.write(f"# truncated_sequences {run_ctx.num_truncated_sequences}\n")
            np.savetxt(f, pooled_variance[None, :], fmt="%.10g")
        np.save(pooled_variance_npy_filename, pooled_variance_payload, allow_pickle=True)
        torch.save(pooled_variance_payload, pooled_variance_pt_filename)

        mean_matrix = np.stack(
            [run_ctx.label_sums_by_dim[dim][int(label_id)] / run_ctx.label_counts[int(label_id)] for label_id in label_ids]
        )
        phoneme_means_payload = {
            "pca_dim": int(dim),
            "target_layer": int(run_ctx.target_layer),
            "label_ids": label_ids,
            "labels": label_symbols,
            "frame_counts": frame_counts,
            "means": mean_matrix,
        }

        with open(phoneme_means_filename, "w", encoding="utf-8") as f:
            f.write("# label_id\tlabel\tframe_count\tmean_vector\n")
            for label_id, label_symbol, count, mean_vector in zip(label_ids, label_symbols, frame_counts, mean_matrix):
                mean_vector_str = " ".join(f"{value:.10g}" for value in mean_vector)
                f.write(f"{int(label_id)}\t{label_symbol}\t{int(count)}\t{mean_vector_str}\n")
        np.save(phoneme_means_npy_filename, phoneme_means_payload, allow_pickle=True)
        torch.save(phoneme_means_payload, phoneme_means_pt_filename)


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)
    alignments = data.get("alignments")
    if alignments is None:
        raise ValueError("Alignments not found in data stream. Check hdf_stream_name in the dataset builder.")
    alignments_len = data["alignments:size1"].to(torch.long)

    all_hidden_states, output_lengths = model.extract_hidden_states(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transformed_hidden_states = model.transform_hidden_states(
        all_hidden_states,
        output_lengths,
        update_pca=False,
    )

    layer_idx = model.return_layers.index(run_ctx.target_layer)
    layer_features = transformed_hidden_states[layer_idx]
    run_ctx.num_batches += 1

    for batch_idx in range(layer_features.shape[0]):
        feature_len = int(output_lengths[batch_idx].item())
        alignment_len = int(alignments_len[batch_idx].item())
        raw_labels = alignments[batch_idx, :alignment_len].detach().cpu().to(dtype=torch.long)
        labels = _subsample_alignment(raw_labels, run_ctx.alignment_subsample_factor)
        effective_len = min(feature_len, int(labels.shape[0]))

        run_ctx.num_sequences += 1
        if effective_len <= 0:
            continue
        if feature_len != int(labels.shape[0]):
            run_ctx.num_truncated_sequences += 1

        features_full = layer_features[batch_idx, :effective_len].detach().cpu().to(dtype=torch.float64)
        labels = labels[:effective_len]
        active_dims = _get_active_dims(run_ctx, int(features_full.shape[1]))

        run_ctx.feature_count += effective_len
        unique_labels = torch.unique(labels)
        for label_id_tensor in unique_labels:
            label_id = int(label_id_tensor.item())
            label_count = int((labels == label_id_tensor).sum().item())
            run_ctx.label_counts[label_id] = run_ctx.label_counts.get(label_id, 0) + label_count

        for dim in active_dims:
            features = features_full[:, :dim]
            batch_sum = torch.sum(features, dim=0).numpy()
            batch_sum_sq = torch.sum(features * features, dim=0).numpy()
            if run_ctx.feature_sums[dim] is None:
                run_ctx.feature_sums[dim] = batch_sum
                run_ctx.feature_sum_sqs[dim] = batch_sum_sq
            else:
                run_ctx.feature_sums[dim] += batch_sum
                run_ctx.feature_sum_sqs[dim] += batch_sum_sq

            for label_id_tensor in unique_labels:
                label_id = int(label_id_tensor.item())
                mask = labels == label_id_tensor
                label_features = features[mask]
                label_sum = torch.sum(label_features, dim=0).numpy()

                if label_id not in run_ctx.label_sums_by_dim[dim]:
                    run_ctx.label_sums_by_dim[dim][label_id] = label_sum
                else:
                    run_ctx.label_sums_by_dim[dim][label_id] += label_sum

    if run_ctx.log_interval_batches > 0 and run_ctx.num_batches % run_ctx.log_interval_batches == 0:
        dims_text = ",".join(str(dim) for dim in run_ctx.active_pca_dims) if run_ctx.active_pca_dims else "pending"
        print(
            f"[PCA label stats] progress: batches={run_ctx.num_batches}, "
            f"sequences={run_ctx.num_sequences}, frames={run_ctx.feature_count}, "
            f"labels_seen={len(run_ctx.label_counts)}, truncated_sequences={run_ctx.num_truncated_sequences}, "
            f"pca_dims={dims_text}"
        )
