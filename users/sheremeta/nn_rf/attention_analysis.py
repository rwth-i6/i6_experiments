"""Capture rf self-attention weights, reduce them to offset statistics and plot them."""

import contextlib
import dataclasses
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import returnn.frontend as rf
import returnn.frontend.attention as rf_attention
from returnn.tensor import Dim, Tensor


class CapturedAttention:
    def __init__(self, weights: torch.Tensor, rest_names: List[str]):
        self.weights = weights
        self.rest_names = rest_names

    def __repr__(self) -> str:
        return f"CapturedAttention(shape={tuple(self.weights.shape)}, rest={self.rest_names})"


def _weights(
    query: Tensor,
    keys: Tensor,
    *,
    key_dim: Dim,
    axis: Dim,
    causal_query_spatial_dim: Optional[Dim],
    scale: Optional[float],
) -> Tuple[Tensor, Dim]:
    kv_dim = Dim(None, name="capture_kv")
    keys_kv, _ = rf.replace_dim(keys, in_dim=axis, out_dim=kv_dim)

    energy = rf.matmul(query, keys_kv, reduce=key_dim)
    energy = energy * (key_dim.dimension**-0.5 if scale is None else scale)

    if causal_query_spatial_dim is not None:
        q_pos = rf.range_over_dim(
            causal_query_spatial_dim, device=energy.raw_tensor.device
        )
        k_pos = rf.range_over_dim(kv_dim, device=energy.raw_tensor.device)
        energy = rf.where(rf.compare(k_pos, "<=", q_pos), energy, float("-inf"))

    return rf.softmax(energy, axis=kv_dim), kv_dim


def _rel_pos_weights(
    query: Tensor,
    key: Tensor,
    pos_emb: Tensor,
    *,
    pos_bias_u: Optional[Tensor],
    pos_bias_v: Optional[Tensor],
    qk_feat_dim: Dim,
    kv_spatial_dim: Dim,
    query_spatial_dim: Dim,
    pos_emb_spatial_dim: Dim,
) -> Tensor:
    """Mirror of rf's generic rel_pos_self_attention scoring, stopping at the softmax."""
    from returnn.frontend.attention import _rel_pos_enc_shift

    q_u = (query + pos_bias_u) if pos_bias_u is not None else query
    q_v = (query + pos_bias_v) if pos_bias_v is not None else query

    matrix_ac = rf.matmul(q_u, key, reduce=qk_feat_dim)
    matrix_bd = rf.matmul(q_v, pos_emb, reduce=qk_feat_dim)
    matrix_bd = _rel_pos_enc_shift(matrix_bd, query_spatial_dim, pos_emb_spatial_dim, kv_spatial_dim)

    scores = (matrix_ac + matrix_bd) * (qk_feat_dim.dimension**-0.5)
    return rf.softmax(scores, axis=kv_spatial_dim)


def _check(
    weights: Tensor,
    values: Tensor,
    att: Tensor,
    *,
    axis: Dim,
    kv_dim: Dim,
    tolerance: float,
) -> None:
    values_kv, _ = rf.replace_dim(values, in_dim=axis, out_dim=kv_dim)
    recon = rf.matmul(weights, values_kv, reduce=kv_dim, use_mask=False)

    diff = float(rf.reduce_max(rf.abs(recon - att), axis=att.dims).raw_tensor)
    assert diff <= tolerance, (
        f"captured attention weights do not reproduce the layer output (max abs diff {diff:.3g} > {tolerance:.3g}). "
        f"capture only supports eval mode, attention dropout must be off"
    )


@contextlib.contextmanager
def capture_attention(*, tolerance: float = 1e-3) -> Iterator[List[CapturedAttention]]:
    collected: List[CapturedAttention] = []
    original = rf_attention.dot_attention

    def _wrapped(query: Tensor, keys: Tensor, values: Tensor, **kwargs: Any) -> Tensor:
        # forward the caller's kwargs verbatim, rf versions differ in which ones dot_attention takes
        att = original(query, keys, values, **kwargs)
        axis: Dim = kwargs["axis"]
        weights, kv_dim = _weights(
            query,
            keys,
            key_dim=kwargs["key_dim"],
            axis=axis,
            causal_query_spatial_dim=kwargs.get("causal_query_spatial_dim"),
            scale=kwargs.get("scale"),
        )
        _check(weights, values, att, axis=axis, kv_dim=kv_dim, tolerance=tolerance)

        rest = [d for d in weights.dims if d not in (axis, kv_dim)]
        raw = weights.copy_compatible_to_dims_raw(rest + [axis, kv_dim])
        collected.append(CapturedAttention(raw.detach(), [d.description for d in rest]))
        return att

    original_rel_pos = rf_attention._rel_pos_self_attention

    def _wrapped_rel_pos(
        query: Tensor, key: Tensor, value: Tensor, pos_emb: Tensor, **kwargs: Any
    ) -> Tensor:
        """The conformer uses RelPosSelfAttention, which never goes through dot_attention."""
        att = original_rel_pos(query, key, value, pos_emb, **kwargs)
        kv_dim: Dim = kwargs["kv_spatial_dim"]
        weights = _rel_pos_weights(
            query,
            key,
            pos_emb,
            pos_bias_u=kwargs.get("pos_bias_u"),
            pos_bias_v=kwargs.get("pos_bias_v"),
            qk_feat_dim=kwargs["qk_feat_dim"],
            kv_spatial_dim=kv_dim,
            query_spatial_dim=kwargs["query_spatial_dim"],
            pos_emb_spatial_dim=kwargs["pos_emb_spatial_dim"],
        )
        recon = rf.matmul(weights, value, reduce=kv_dim, use_mask=False)
        diff = float(rf.reduce_max(rf.abs(recon - att), axis=att.dims).raw_tensor)
        assert diff <= tolerance, (
            f"captured rel-pos attention weights do not reproduce the layer output "
            f"(max abs diff {diff:.3g} > {tolerance:.3g})"
        )

        rest = [d for d in weights.dims if d not in (kwargs["query_spatial_dim"], kv_dim)]
        raw = weights.copy_compatible_to_dims_raw(rest + [kwargs["query_spatial_dim"], kv_dim])
        collected.append(CapturedAttention(raw.detach(), [d.description for d in rest]))
        return att

    # SelfAttentionBase.attention and CausalSelfAttention resolve dot_attention from module globals
    rf_attention.dot_attention = _wrapped
    rf_attention._rel_pos_self_attention = _wrapped_rel_pos
    try:
        yield collected
    finally:
        rf_attention.dot_attention = original
        rf_attention._rel_pos_self_attention = original_rel_pos


@dataclasses.dataclass
class AttentionStats:
    """Attention mass binned by how far it reaches, summed over the queries that were counted.

    Sums rather than means, so stats from several batches add up before being normalized.
    Offsets are query minus key, so positive looks back and negative looks ahead. The decoder is
    causal and only ever fills the non-negative bins, the encoder is bidirectional and fills both.
    """

    offset_sum: torch.Tensor  # (layers, heads, num_bins)
    count: int
    delay_frames: int
    min_offset: int = 0

    def __add__(self, other: "AttentionStats") -> "AttentionStats":
        assert self.offset_sum.shape == other.offset_sum.shape, (self.offset_sum.shape, other.offset_sum.shape)
        assert self.delay_frames == other.delay_frames, (self.delay_frames, other.delay_frames)
        assert self.min_offset == other.min_offset, (self.min_offset, other.min_offset)
        return AttentionStats(
            offset_sum=self.offset_sum + other.offset_sum,
            count=self.count + other.count,
            delay_frames=self.delay_frames,
            min_offset=self.min_offset,
        )

    @property
    def max_offset(self) -> int:
        return self.min_offset + self.offset_sum.size(-1) - 1

    @property
    def offsets(self) -> torch.Tensor:
        """The offset each bin stands for."""
        return torch.arange(
            self.min_offset, self.max_offset + 1, device=self.offset_sum.device
        )

    @property
    def offset_hist(self) -> torch.Tensor:
        """(layers, heads, num_bins) mean attention mass at each offset."""
        return self.offset_sum / max(self.count, 1)

    def _bin(self, offset: int) -> int:
        return offset - self.min_offset

    def mass_at(self, offset: int, *, window: int = 0) -> torch.Tensor:
        """(layers, heads) mass within ``window`` frames of ``offset``."""
        lo = max(self._bin(offset - window), 0)
        hi = min(self._bin(offset + window), self.offset_sum.size(-1) - 1)
        return self.offset_hist[..., lo : hi + 1].sum(dim=-1)

    def onset_mass(self, *, window: int = 0) -> torch.Tensor:
        """(layers, heads) mass on the frame the emitted token's acoustics start at."""
        return self.mass_at(self.delay_frames, window=window)

    def peak_offset(self) -> torch.Tensor:
        """(layers, heads) offset carrying the most mass."""
        return self.offset_hist.argmax(dim=-1) + self.min_offset

    def centroid_offset(self) -> torch.Tensor:
        """(layers, heads) mass-weighted mean offset."""
        hist = self.offset_hist
        offsets = self.offsets.to(hist.dtype)
        total = hist.sum(dim=-1).clamp(min=1e-12)
        return (hist * offsets).sum(dim=-1) / total

    def lookahead_mass(self) -> torch.Tensor:
        """(layers, heads) mass on negative offsets, i.e. attention to frames not yet emitted."""
        split = self._bin(0)
        if split <= 0:
            return torch.zeros(self.offset_sum.shape[:2], device=self.offset_sum.device)
        return self.offset_hist[..., :split].sum(dim=-1)

    def ranked_heads(
        self, *, offset: Optional[int] = None, window: int = 2, top: Optional[int] = None
    ) -> List[Tuple[int, int, float]]:
        """(layer, head, mass) sorted by how much mass sits at ``offset``, the delay by default."""
        mass = self.mass_at(self.delay_frames if offset is None else offset, window=window)
        entries = [
            (int(layer), int(head), float(mass[layer, head]))
            for layer in range(mass.size(0))
            for head in range(mass.size(1))
        ]
        entries.sort(key=lambda e: -e[2])
        return entries if top is None else entries[:top]


def collect_offset_stats(
    weights_per_layer: Sequence[torch.Tensor],
    query_mask: torch.Tensor,
    *,
    delay_frames: int,
    max_offset: Optional[int] = None,
    signed: bool = False,
) -> AttentionStats:
    """Bin each layer/head's attention by query-minus-key offset over the masked query frames.

    ``weights_per_layer`` holds one (batch, heads, query, key) tensor per layer.
    ``signed`` also bins negative offsets, needed for the bidirectional encoder.
    """
    assert weights_per_layer, "no captured attention"
    batch, heads, num_queries, num_keys = weights_per_layer[0].shape
    assert num_queries == num_keys, (num_queries, num_keys)
    for w in weights_per_layer:
        assert w.shape == weights_per_layer[0].shape, (w.shape, weights_per_layer[0].shape)
    assert query_mask.shape == (batch, num_queries), (query_mask.shape, (batch, num_queries))

    device = weights_per_layer[0].device
    query_mask = query_mask.to(device)
    if max_offset is None:
        max_offset = num_queries - 1
    min_offset = -max_offset if signed else 0
    assert min_offset <= delay_frames <= max_offset, (min_offset, delay_frames, max_offset)

    # only offsets up to the sequence length are reachable, so gather over those and pad the rest
    reach = min(max_offset, num_queries - 1)
    positions = torch.arange(num_queries, device=device)
    offsets = torch.arange(-reach if signed else 0, reach + 1, device=device)
    key_index = positions.unsqueeze(1) - offsets.unsqueeze(0)  # (query, offset)
    in_range = (key_index >= 0) & (key_index < num_keys)
    gather_index = key_index.clamp(min=0, max=num_keys - 1).view(1, 1, num_queries, -1)

    num_bins = offsets.numel()
    weight = query_mask.to(weights_per_layer[0].dtype).view(batch, 1, num_queries, 1)
    layer_sums = []
    for w in weights_per_layer:
        picked = w.gather(3, gather_index.expand(batch, heads, num_queries, num_bins))
        picked = picked * in_range.view(1, 1, num_queries, num_bins)
        layer_sums.append((picked * weight).sum(dim=(0, 2)))

    reached = torch.stack(layer_sums, dim=0).float()
    full = reached.new_zeros(reached.shape[:2] + (max_offset - min_offset + 1,))
    lo = int(offsets[0].item()) - min_offset
    full[..., lo : lo + num_bins] = reached

    return AttentionStats(
        offset_sum=full,
        count=int(query_mask.sum().item()),
        delay_frames=delay_frames,
        min_offset=min_offset,
    )


def length_query_mask(num_queries: int, lens: torch.Tensor) -> torch.Tensor:
    """Every valid frame, for the encoder where there is no label notion."""
    positions = torch.arange(num_queries, device=lens.device)
    return positions.unsqueeze(0) < lens.unsqueeze(1)


def format_summary(stats: AttentionStats, *, window: int = 2, top: int = 5) -> str:
    """One readable block: peak, centroid and mass at the delay for the heads that reach furthest."""
    lines = [
        f"queries counted: {stats.count}",
        f"delay: {stats.delay_frames} frames, offsets {stats.min_offset}..{stats.max_offset}, window +/- {window}",
        "",
        f"{'layer':>5} {'head':>5} {'peak':>6} {'centroid':>9} {'mass@delay':>11} {'lookahead':>10}",
    ]
    peak = stats.peak_offset()
    centroid = stats.centroid_offset()
    ahead = stats.lookahead_mass()
    for layer, head, mass in stats.ranked_heads(window=window, top=top):
        lines.append(
            f"{layer:>5} {head:>5} {int(peak[layer, head]):>6} {float(centroid[layer, head]):>9.1f} "
            f"{mass:>11.4f} {float(ahead[layer, head]):>10.4f}"
        )
    return "\n".join(lines)


def stats_from_npz(path: str) -> Dict[str, AttentionStats]:
    """Rebuild the per-part stats that the forward callback saved to ``attention_stats.npz``."""
    arrays = np.load(path)
    delay = int(arrays["delay_frames"])
    parts = [k[: -len("_offset_sum")] for k in arrays.files if k.endswith("_offset_sum")]
    parts.sort(key=lambda p: (p not in ("encoder", "decoder"), p != "encoder", p))
    return {
        part: AttentionStats(
            offset_sum=torch.from_numpy(arrays[f"{part}_offset_sum"]),
            count=int(arrays[f"{part}_count"]),
            delay_frames=delay,
            min_offset=int(arrays[f"{part}_min_offset"]),
        )
        for part in parts
    }


def maps_from_npz(path: str) -> List[Dict[str, Any]]:
    """Rebuild the per-utterance map entries that the forward callback saved to ``attention_maps.npz``."""
    arrays = np.load(path)
    entries = []
    for i, tag in enumerate(arrays["seq_tags"]):
        entry: Dict[str, Any] = {"seq_tag": str(tag)}
        for field in ("encoder", "decoder", "text_codes", "emits", "encoder_frames", "decoder_frames"):
            if f"{i}_{field}" in arrays.files:
                entry[field] = arrays[f"{i}_{field}"]
        entries.append(entry)
    return entries


_COLS = 4


def _lazy_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _log_norm() -> Any:
    from matplotlib.colors import LogNorm

    return LogNorm()


def _parts(
    stats_by_part: Mapping[str, Optional[AttentionStats]]
) -> List[Tuple[str, AttentionStats]]:
    return [(k, v) for k, v in stats_by_part.items() if v is not None]


def _mean_hist(stats: AttentionStats) -> np.ndarray:
    """(bins,) mass averaged over every layer and head."""
    hist = stats.offset_hist.cpu().numpy()
    return hist.reshape(-1, hist.shape[-1]).mean(0)


def _layer_hist(stats: AttentionStats) -> np.ndarray:
    """(layers, bins) mass averaged over every head of each layer."""
    return stats.offset_hist.cpu().numpy().mean(axis=1)


def _offsets(stats: AttentionStats) -> np.ndarray:
    hist = stats.offset_hist.cpu().numpy()
    return np.arange(stats.min_offset, stats.min_offset + hist.shape[-1])


def _mark_reference_lines(ax: Any, delay_frames: int, legend: bool) -> None:
    ax.axvline(0, color="tab:cyan", lw=1.2, label="offset 0")
    if delay_frames:
        ax.axvline(delay_frames, color="tab:red", lw=1.2, label=f"offset {delay_frames}")
    if legend:
        ax.legend(fontsize=7)


def plot_offset_curves(
    stats_by_part: Mapping[str, Optional[AttentionStats]],
    *,
    delay_frames: int,
    path: str,
    span: int = 200,
) -> str:
    """Mean attention mass against offset, averaged over all layers and heads, one panel per part."""
    plt = _lazy_pyplot()
    parts = _parts(stats_by_part)
    fig, axes = plt.subplots(len(parts), 1, figsize=(11, 3.2 * len(parts)), squeeze=False)

    for ax, (part, stats) in zip(axes[:, 0], parts):
        offsets = _offsets(stats)
        lo = -span if stats.min_offset < 0 else 0
        sel = (offsets >= lo) & (offsets <= span)
        ax.semilogy(offsets[sel], _mean_hist(stats)[sel], color="black", lw=1.2)
        _mark_reference_lines(ax, delay_frames, legend=True)
        ax.set_title(part)
        ax.set_xlabel("offset (frames)")
        ax.set_ylabel("attention mass")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_layer_offsets(
    stats_by_part: Mapping[str, Optional[AttentionStats]],
    *,
    delay_frames: int,
    path: str,
    span: int = 200,
) -> str:
    """Per-layer offset mass, as overlaid curves and as a layer-by-offset map, all heads averaged."""
    plt = _lazy_pyplot()
    parts = _parts(stats_by_part)
    fig, axes = plt.subplots(
        len(parts), 2, figsize=(13, 3.4 * len(parts)), squeeze=False
    )

    for row, (part, stats) in enumerate(parts):
        per_layer = _layer_hist(stats)
        offsets = _offsets(stats)
        lo = -span if stats.min_offset < 0 else 0
        sel = (offsets >= lo) & (offsets <= span)
        num_layers = per_layer.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, num_layers))

        ax = axes[row, 0]
        for layer in range(num_layers):
            ax.semilogy(
                offsets[sel],
                per_layer[layer][sel],
                color=colors[layer],
                lw=1.0,
                label=f"L{layer + 1}",
            )
        _mark_reference_lines(ax, delay_frames, legend=False)
        ax.set_title(f"{part}, per layer")
        ax.set_xlabel("offset (frames)")
        ax.set_ylabel("attention mass")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5, ncol=2, loc="upper right")

        ax = axes[row, 1]
        image = ax.imshow(
            per_layer[:, sel],
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=(offsets[sel][0], offsets[sel][-1], 0.5, num_layers + 0.5),
            norm=_log_norm(),
        )
        ax.axvline(0, color="tab:cyan", lw=1.0)
        if delay_frames:
            ax.axvline(delay_frames, color="tab:red", lw=1.0)
        ax.set_title(f"{part}, layer by offset")
        ax.set_xlabel("offset (frames)")
        ax.set_ylabel("layer")
        fig.colorbar(image, ax=ax, fraction=0.03)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_heatmaps(
    maps: Sequence[Dict[str, Any]],
    *,
    delay_frames: int,
    path: str,
    parts: Sequence[str] = ("encoder", "decoder"),
) -> Optional[str]:
    """Query-by-key maps for one utterance, one panel per layer, averaged over all heads."""
    if not maps:
        return None
    plt = _lazy_pyplot()
    entry = maps[0]

    blocks = []
    for part in parts:
        if part not in entry:
            continue
        weights = entry[part].astype(np.float32)
        blocks.append((part, weights.mean(axis=1), int(entry.get(f"{part}_frames", weights.shape[-1]))))
    if not blocks:
        return None

    rows = sum(int(np.ceil(w.shape[0] / _COLS)) for _, w, _ in blocks)
    fig, axes = plt.subplots(rows, _COLS, figsize=(3.6 * _COLS, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    row_offset = 0
    for part, per_layer, frames in blocks:
        num_layers, n = per_layer.shape[0], per_layer.shape[-1]
        scaled_delay = delay_frames * n / max(frames, 1)
        for layer in range(num_layers):
            ax = axes[row_offset + layer // _COLS, layer % _COLS]
            ax.axis("on")
            ax.imshow(
                per_layer[layer],
                aspect="auto",
                origin="upper",
                cmap="viridis",
                interpolation="nearest",
                norm=_log_norm(),
            )
            ax.plot([0, n - 1], [0, n - 1], color="tab:cyan", lw=0.7, alpha=0.8)
            if scaled_delay >= 1:
                ax.plot(
                    [0, n - 1 - scaled_delay],
                    [scaled_delay, n - 1],
                    color="tab:red",
                    lw=0.8,
                    alpha=0.9,
                )
            ax.set_title(f"{part} L{layer + 1}", fontsize=9)
            ax.set_xlabel("key frame", fontsize=7)
            ax.set_ylabel("query frame", fontsize=7)
            ax.tick_params(labelsize=6)
        row_offset += int(np.ceil(num_layers / _COLS))

    fig.suptitle(entry["seq_tag"], fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
