from dataclasses import dataclass

import torch


@dataclass
class ForwardConfig:
    feature_output_filename: str = "features.hdf"
    pca_state_filename: str = "pca_state.pt"
    feature_dim: int = 1024
    pca_dim: int = 512
    covariance_chunk_size: int = 8192
    return_layer: int = 15


def forward_init_hook(run_ctx, **kwargs):
    from returnn.datasets.hdf import SimpleHDFWriter

    run_ctx.config = ForwardConfig(**kwargs.get("config", {}))
    run_ctx.hdf_writer = SimpleHDFWriter(run_ctx.config.feature_output_filename, dim=run_ctx.config.feature_dim, ndim=2)
    run_ctx.num_pca_frames = 0
    run_ctx.feature_sum = None
    run_ctx.feature_cross = None


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.hdf_writer.close()
    if run_ctx.num_pca_frames <= run_ctx.config.pca_dim:
        raise RuntimeError(
            f"Need more frames than pca_dim={run_ctx.config.pca_dim}, got {run_ctx.num_pca_frames}."
        )

    num_frames = int(run_ctx.num_pca_frames)
    feature_sum = run_ctx.feature_sum.cpu()
    feature_cross = run_ctx.feature_cross.cpu()
    mean = feature_sum / num_frames
    covariance = (feature_cross - num_frames * torch.outer(mean, mean)) / max(1, num_frames - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, : run_ctx.config.pca_dim].T.contiguous()
    explained_variance = torch.clamp(eigvals[: run_ctx.config.pca_dim], min=0.0)
    total_variance = torch.clamp(eigvals.sum(), min=1e-20)
    explained_variance_ratio = explained_variance / total_variance
    singular_values = torch.sqrt(explained_variance * max(1, num_frames - 1))
    noise_variance = (
        eigvals[run_ctx.config.pca_dim :].mean().item()
        if run_ctx.config.pca_dim < eigvals.shape[0]
        else 0.0
    )

    torch.save(
        {
            "return_layer": run_ctx.config.return_layer,
            "pca_dim": run_ctx.config.pca_dim,
            "n_samples_seen": num_frames,
            "mean": mean.float(),
            "var": torch.diag(covariance).float(),
            "components": components.float(),
            "singular_values": singular_values.float(),
            "explained_variance": explained_variance.float(),
            "explained_variance_ratio": explained_variance_ratio.float(),
            "noise_variance": float(noise_variance),
        },
        run_ctx.config.pca_state_filename,
    )


def _normalize_seq_tags(seq_tags):
    normalized = []
    for tag in seq_tags:
        if isinstance(tag, (bytes, bytearray)):
            normalized.append(tag.decode("utf8"))
        else:
            normalized.append(str(tag))
    return normalized


def _update_covariance(run_ctx, valid_features: torch.Tensor):
    if valid_features.numel() == 0:
        return
    if run_ctx.feature_sum is None:
        device = valid_features.device
        feature_dim = int(run_ctx.config.feature_dim)
        run_ctx.feature_sum = torch.zeros(feature_dim, dtype=torch.float64, device=device)
        run_ctx.feature_cross = torch.zeros(feature_dim, feature_dim, dtype=torch.float64, device=device)
    chunk_size = int(run_ctx.config.covariance_chunk_size)
    if chunk_size <= 0:
        raise ValueError("covariance_chunk_size must be positive.")
    for start in range(0, valid_features.shape[0], chunk_size):
        chunk = valid_features[start : start + chunk_size].to(dtype=torch.float64)
        run_ctx.feature_sum += chunk.sum(dim=0)
        run_ctx.feature_cross += chunk.T @ chunk
        run_ctx.num_pca_frames += int(chunk.shape[0])


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to(torch.long)

    with torch.no_grad():
        features, feature_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
        features = features.float()

    seq_tags = _normalize_seq_tags(data["seq_tag"])
    feature_len = feature_len.detach().to(dtype=torch.long)
    max_time = features.shape[1]
    valid_mask = torch.arange(max_time, device=features.device)[None, :] < feature_len[:, None].to(features.device)
    _update_covariance(run_ctx, features[valid_mask])

    run_ctx.hdf_writer.insert_batch(
        inputs=features.detach().cpu().numpy().astype("float32"),
        seq_len=feature_len.detach().cpu().tolist(),
        seq_tag=seq_tags,
    )
