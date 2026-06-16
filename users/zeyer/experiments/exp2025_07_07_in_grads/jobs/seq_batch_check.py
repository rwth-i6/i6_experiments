"""Reusable single-vs-batch equivalence check for the model wrappers.

The only meaningful seq-batching correctness test:
forward each seq alone (B=1) and forward the whole batch (B>1) through the SAME model,
then compare each seq's final per-token log_probs (real tokens only).
A nonzero diff means the model's batched forward leaks padding into real frames
(or fp non-associativity); the seq-batch masking is what should drive it down.

Reused by BatchForwardEquivalenceProbeJob and by ad-hoc per-model checks,
and by any job that wants to assert its batched path matches the single path.
Works for any wrapper with the standard interface (``model(...) -> ForwardOutput`` + ``model.log_probs``).
"""

from typing import List, Tuple, Any
import torch


def model_log_probs(model: Any, batch: List[Tuple[torch.Tensor, int, List[str]]]) -> Tuple[torch.Tensor, List[int]]:
    """Final per-token log_probs ``[B, n_max, V]`` for a batch of any B.

    ``batch`` items are ``(audio_1d, sample_rate, words)``.
    Returns ``(log_probs, n_targets_per_seq)``.
    """
    nb = len(batch)
    ml = max(int(a.shape[0]) for a, _, _ in batch)
    raw = torch.zeros((nb, ml))
    for i, (a, _, _) in enumerate(batch):
        raw[i, : a.shape[0]] = a
    fwd = model(
        raw_inputs=raw,
        raw_inputs_sample_rate=batch[0][1],
        raw_input_seq_lens=torch.tensor([int(a.shape[0]) for a, _, _ in batch]),
        raw_targets=[w for _, _, w in batch],
        raw_target_seq_lens=torch.tensor([len(w) for _, _, w in batch]),
        omitted_prev_context=torch.zeros(nb, dtype=torch.int64),
    )
    tse = fwd.target_start_end.cpu()
    n_tgt = torch.tensor([int(tse[i, len(b[2]), 1]) for i, b in enumerate(batch)])
    lp = model.log_probs(forward_output=fwd, start=torch.zeros(nb, dtype=torch.int64), end=n_tgt)
    return lp.detach().float().cpu(), n_tgt.tolist()


def single_vs_batch_diff(model: Any, batch: List[Tuple[torch.Tensor, int, List[str]]]) -> List[float]:
    """Per-seq ``max|Δ|`` of final log_probs, each seq run single (B=1) vs inside the full batch.

    Returns one max-abs-diff per seq (real tokens only).
    """
    lp_b, n_b = model_log_probs(model, batch)
    out = []
    for i, b in enumerate(batch):
        lp_s, n_s = model_log_probs(model, [b])
        n = min(n_b[i], n_s[0])
        out.append(float((lp_b[i, :n] - lp_s[0, :n]).abs().max()))
    return out
