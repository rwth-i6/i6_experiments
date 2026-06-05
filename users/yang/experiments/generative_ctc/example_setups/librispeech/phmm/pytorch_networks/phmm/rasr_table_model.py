from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from .rasr_table_model_cfg import ModelConfig


def _normalize_label(label: str) -> str:
    if label == "[SIL]":
        return "[SILENCE]"
    return label.replace("#", "")


class Model(nn.Module):
    """
    Table-backed dummy model for RASR decoding.

    The table is interpreted as a joint distribution p(phoneme, cluster). Given
    an input cluster-id sequence, the model emits one log-probability vector over
    phonemes for each cluster-id frame.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        if self.cfg.decode_model not in {"segment", "frame"}:
            raise ValueError(f"Unsupported decode_model {self.cfg.decode_model!r}; expected 'segment' or 'frame'")
        if self.cfg.segment_start_downsample_rate <= 0:
            raise ValueError(
                f"segment_start_downsample_rate must be positive, got {self.cfg.segment_start_downsample_rate}"
            )
        if self.cfg.max_segment_start_mismatch_ratio < 0.0:
            raise ValueError(
                "max_segment_start_mismatch_ratio must be non-negative, "
                f"got {self.cfg.max_segment_start_mismatch_ratio}"
            )
        if self.cfg.max_num_huge_segment_start_mismatches < 0:
            raise ValueError(
                "max_num_huge_segment_start_mismatches must be non-negative, "
                f"got {self.cfg.max_num_huge_segment_start_mismatches}"
            )

        try:
            from returnn.util.basic import cf

            table_file = cf(self.cfg.table_file)
        except Exception:
            table_file = str(self.cfg.table_file)

        npz = np.load(table_file, allow_pickle=True)
        if self.cfg.table_key not in npz:
            raise KeyError(f"Table key {self.cfg.table_key!r} not found in {table_file!r}; keys={npz.files}")
        if self.cfg.phoneme_tokens_key not in npz:
            raise KeyError(
                f"Phoneme token key {self.cfg.phoneme_tokens_key!r} not found in {table_file!r}; keys={npz.files}"
            )
        if self.cfg.cluster_tokens_key not in npz:
            raise KeyError(
                f"Cluster token key {self.cfg.cluster_tokens_key!r} not found in {table_file!r}; keys={npz.files}"
            )

        joint = np.asarray(npz[self.cfg.table_key], dtype="float64")
        if joint.ndim != 2:
            raise ValueError(f"Expected a 2-D table, got shape {joint.shape}")
        if np.any(joint < 0.0):
            raise ValueError("Joint table contains negative entries")
        total = float(joint.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"Joint table has invalid total mass {total}")
        joint = joint / total

        phoneme_tokens = [str(t) for t in np.asarray(npz[self.cfg.phoneme_tokens_key]).tolist()]
        cluster_tokens = [str(t) for t in np.asarray(npz[self.cfg.cluster_tokens_key]).tolist()]
        if joint.shape != (len(phoneme_tokens), len(cluster_tokens)):
            raise ValueError(
                f"Table shape {joint.shape} does not match "
                f"{len(phoneme_tokens)} phoneme tokens and {len(cluster_tokens)} cluster tokens"
            )

        phoneme_marginal = joint.sum(axis=1)
        cluster_marginal = joint.sum(axis=0)
        min_prob = float(self.cfg.min_prob)
        if min_prob <= 0.0:
            raise ValueError(f"min_prob must be positive, got {min_prob}")

        log_joint = np.log(np.maximum(joint, min_prob))
        log_phoneme_marginal = np.log(np.maximum(phoneme_marginal, min_prob))
        log_cluster_marginal = np.log(np.maximum(cluster_marginal, min_prob))

        generative_table = log_joint - log_phoneme_marginal[:, None]
        discriminative_table = log_joint - log_cluster_marginal[None, :]

        self.phoneme_tokens = phoneme_tokens
        self.cluster_tokens = cluster_tokens
        self._phoneme_to_row = {_normalize_label(token): i for i, token in enumerate(phoneme_tokens)}
        self._cluster_to_col = self._build_cluster_to_col(cluster_tokens)

        max_cluster_id = max(self._cluster_to_col) if self._cluster_to_col else -1
        cluster_id_to_col = np.full((max_cluster_id + 1,), -1, dtype="int64")
        for cluster_id, col in self._cluster_to_col.items():
            cluster_id_to_col[cluster_id] = col

        self.register_buffer("generative_logprob_table", torch.tensor(generative_table, dtype=torch.float32))
        self.register_buffer("discriminative_logprob_table", torch.tensor(discriminative_table, dtype=torch.float32))
        self.register_buffer("cluster_id_to_col", torch.tensor(cluster_id_to_col, dtype=torch.long))
        self.register_buffer("phoneme_marginal", torch.tensor(phoneme_marginal, dtype=torch.float32))
        self.register_buffer("cluster_marginal", torch.tensor(cluster_marginal, dtype=torch.float32))
        self.num_huge_segment_start_mismatches = 0

    @staticmethod
    def _downsample_starts(starts: torch.Tensor, *, downsample_rate: int, feature_len: int) -> torch.Tensor:
        starts = starts.to(dtype=torch.long)
        starts = starts[starts >= 0]
        if starts.numel() == 0:
            starts = starts.new_tensor([0])

        base = torch.div(starts, downsample_rate, rounding_mode="floor")
        remainder = starts.remainder(downsample_rate)
        random_offsets = torch.randint(
            low=0,
            high=downsample_rate,
            size=starts.shape,
            device=starts.device,
            dtype=torch.long,
        )
        downsampled = base + (random_offsets < remainder).to(dtype=torch.long)
        downsampled = downsampled[(downsampled >= 0) & (downsampled < feature_len)]
        if feature_len > 0:
            downsampled = torch.cat([downsampled.new_tensor([0]), downsampled])
        if downsampled.numel() == 0:
            return downsampled
        return torch.unique(downsampled, sorted=True)

    @staticmethod
    def _build_cluster_to_col(cluster_tokens: Sequence[str]) -> dict[int, int]:
        mapping = {}
        for col, token in enumerate(cluster_tokens):
            try:
                cluster_id = int(token)
            except ValueError as exc:
                raise ValueError(f"Cluster token {token!r} is not an integer id") from exc
            if cluster_id in mapping:
                raise ValueError(f"Duplicate cluster id {cluster_id}")
            mapping[cluster_id] = col
        return mapping

    def _row_indices_for_output_order(self, output_label_order: Optional[Sequence[str]], device) -> Optional[torch.Tensor]:
        if output_label_order is None:
            return None
        missing = []
        rows = []
        for label in output_label_order:
            normalized = _normalize_label(str(label))
            row = self._phoneme_to_row.get(normalized)
            if row is None:
                missing.append(str(label))
            else:
                rows.append(row)
        if missing:
            raise ValueError(
                "The table does not contain all labels from the RASR lexicon. "
                f"Missing after normalization: {missing[:20]}"
            )
        return torch.tensor(rows, dtype=torch.long, device=device)

    def _cluster_ids_to_logprobs(
        self,
        cluster_ids: torch.Tensor,
        *,
        table: torch.Tensor,
        output_label_order: Optional[Sequence[str]],
    ) -> torch.Tensor:
        if cluster_ids.numel() > 0:
            min_cluster_id = int(cluster_ids.min().detach().cpu())
            max_cluster_id = int(cluster_ids.max().detach().cpu())
            if min_cluster_id < 0 or max_cluster_id >= self.cluster_id_to_col.shape[0]:
                raise ValueError(
                    f"Cluster ids out of table range: min={min_cluster_id}, max={max_cluster_id}, "
                    f"known max={self.cluster_id_to_col.shape[0] - 1}"
                )
        cluster_cols = self.cluster_id_to_col.to(device=cluster_ids.device)[cluster_ids]
        if torch.any(cluster_cols < 0):
            bad_cluster_id = int(cluster_ids[cluster_cols < 0][0].detach().cpu())
            raise ValueError(f"Cluster id {bad_cluster_id} is not present in the table")

        logprobs = table.to(device=cluster_ids.device)[..., cluster_cols].permute(1, 2, 0).contiguous()
        row_indices = self._row_indices_for_output_order(output_label_order, device=cluster_ids.device)
        if row_indices is not None:
            logprobs = logprobs.index_select(dim=-1, index=row_indices)
        return logprobs

    def _expand_segment_cluster_ids(
        self,
        *,
        cluster_ids: torch.Tensor,
        cluster_ids_len: torch.Tensor,
        segment_starts: torch.Tensor,
        segment_starts_len: torch.Tensor,
        frame_lengths_10ms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded = []
        lengths = []
        downsample_rate = int(self.cfg.segment_start_downsample_rate)
        for batch_idx, (seq_cluster_ids, seq_len) in enumerate(zip(cluster_ids, cluster_ids_len)):
            seq_len = int(seq_len.detach().cpu())
            starts_len = int(segment_starts_len[batch_idx].detach().cpu())
            raw_frame_len = int(frame_lengths_10ms[batch_idx].reshape(-1)[0].detach().cpu())
            frame_len = (raw_frame_len + downsample_rate - 1) // downsample_rate
            all_starts = self._downsample_starts(
                segment_starts[batch_idx, :starts_len],
                downsample_rate=downsample_rate,
                feature_len=frame_len,
            )
            num_starts = int(all_starts.numel())
            if seq_len <= 0 or num_starts <= 0:
                expanded_ids = seq_cluster_ids.new_zeros((0,), dtype=torch.long)
                expanded.append(expanded_ids)
                lengths.append(0)
                continue
            mismatch_ratio = abs(num_starts - seq_len) / float(seq_len)
            if mismatch_ratio > self.cfg.max_segment_start_mismatch_ratio:
                self.num_huge_segment_start_mismatches += 1
                msg = (
                    f"Huge segment-count mismatch #{self.num_huge_segment_start_mismatches} for batch index {batch_idx}: "
                    f"cluster sequence length is {seq_len}, but downsampled segment starts has length {num_starts}; "
                    f"mismatch ratio {mismatch_ratio:.4f} > {self.cfg.max_segment_start_mismatch_ratio:.4f}. "
                    "Using the common valid length for now."
                )
                if self.num_huge_segment_start_mismatches > self.cfg.max_num_huge_segment_start_mismatches:
                    raise ValueError(msg)
                print(msg, flush=True)
            valid_len = min(seq_len, num_starts)
            starts = all_starts[:valid_len]
            seq_cluster_ids = seq_cluster_ids[:valid_len]
            if num_starts >= valid_len + 1:
                ends_tensor = all_starts[1 : valid_len + 1]
            else:
                ends_tensor = torch.cat(
                    [starts[1:], torch.tensor([frame_len], dtype=torch.long, device=cluster_ids.device)]
                )
            durations = ends_tensor - starts
            valid = durations > 0
            seq_cluster_ids = seq_cluster_ids[valid]
            durations = durations[valid]
            expanded_ids = torch.repeat_interleave(seq_cluster_ids.to(dtype=torch.long), durations)
            expanded.append(expanded_ids)
            lengths.append(int(expanded_ids.shape[0]))

        max_len = max(lengths) if lengths else 0
        padded = cluster_ids.new_zeros((len(expanded), max_len), dtype=torch.long)
        for batch_idx, expanded_ids in enumerate(expanded):
            padded[batch_idx, : expanded_ids.shape[0]] = expanded_ids
        return padded, torch.tensor(lengths, dtype=torch.long, device=cluster_ids.device)

    @staticmethod
    def _merge_consecutive_same_cluster_ids(
        cluster_ids: torch.Tensor,
        cluster_ids_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        merged = []
        lengths = []
        for seq_cluster_ids, seq_len in zip(cluster_ids, cluster_ids_len):
            seq_len = int(seq_len.detach().cpu())
            seq_cluster_ids = seq_cluster_ids[:seq_len]
            if seq_len <= 1:
                merged_ids = seq_cluster_ids.to(dtype=torch.long)
            else:
                keep = torch.ones((seq_len,), dtype=torch.bool, device=seq_cluster_ids.device)
                keep[1:] = seq_cluster_ids[1:] != seq_cluster_ids[:-1]
                merged_ids = seq_cluster_ids[keep].to(dtype=torch.long)
            merged.append(merged_ids)
            lengths.append(int(merged_ids.shape[0]))

        max_len = max(lengths) if lengths else 0
        padded = cluster_ids.new_zeros((len(merged), max_len), dtype=torch.long)
        for batch_idx, merged_ids in enumerate(merged):
            padded[batch_idx, : merged_ids.shape[0]] = merged_ids
        return padded, torch.tensor(lengths, dtype=torch.long, device=cluster_ids.device)

    def forward(
        self,
        *,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
        logprob_mode: str,
        output_label_order: Optional[Sequence[str]] = None,
        segment_starts: Optional[torch.Tensor] = None,
        segment_starts_len: Optional[torch.Tensor] = None,
        frame_lengths: Optional[torch.Tensor] = None,
    ):
        if logprob_mode == "generative":
            table = self.generative_logprob_table
        elif logprob_mode == "discriminative":
            table = self.discriminative_logprob_table
        else:
            raise ValueError(f"Unsupported logprob_mode {logprob_mode!r}; expected 'generative' or 'discriminative'")

        cluster_ids = raw_audio.to(dtype=torch.long)
        cluster_ids_len = raw_audio_len.to(dtype=torch.long)
        if self.cfg.decode_model == "frame":
            if segment_starts is None or segment_starts_len is None or frame_lengths is None:
                raise ValueError(
                    "segment_starts, segment_starts_len and frame_lengths must be passed to Model.forward "
                    "when decode_model='frame'"
                )
            cluster_ids, cluster_ids_len = self._expand_segment_cluster_ids(
                cluster_ids=cluster_ids,
                cluster_ids_len=cluster_ids_len,
                segment_starts=segment_starts,
                segment_starts_len=segment_starts_len,
                frame_lengths_10ms=frame_lengths,
            )
        elif self.cfg.merge_consecutive_same_cluster_ids_in_segment_mode:
            cluster_ids, cluster_ids_len = self._merge_consecutive_same_cluster_ids(cluster_ids, cluster_ids_len)

        logprobs = self._cluster_ids_to_logprobs(
            cluster_ids,
            table=table,
            output_label_order=output_label_order,
        )
        return logprobs, cluster_ids_len
