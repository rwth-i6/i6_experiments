"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree.py.

Expected run counts for the blank-free recognizer."""

import torch


def expected_run_counts(log_q: torch.Tensor, lens: torch.Tensor):
    """Return differentiable unigram and bigram counts of adjacent-run collapse."""
    q = log_q.exp()
    b, t_max, k = q.shape
    valid = torch.arange(t_max, device=q.device)[None, :] < lens.to(q.device)[:, None]
    q = q * valid.unsqueeze(-1)
    prev = torch.cat([q.new_zeros(b, 1, k), q[:, :-1]], dim=1)
    uni = (q * (1.0 - prev)).sum(dim=1)
    bi = torch.einsum("btj,btk->bjk", prev, q)
    bi = bi.masked_fill(torch.eye(k, dtype=torch.bool, device=q.device), 0.0)
    return uni, bi


def project_text_bigram(joint: torch.Tensor) -> torch.Tensor:
    """Project the inherited text joint onto adjacent-distinct support."""
    k = joint.shape[0]
    supported = joint.masked_fill(torch.eye(k, dtype=torch.bool, device=joint.device), 0.0)
    return supported / supported.sum()


