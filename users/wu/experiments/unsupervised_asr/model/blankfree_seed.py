"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_seed.py.

Port note: only ``collapse_adjacent`` and ``transcript_logprob`` (read by the p0 supervised
train step) are ported; ``theta_train_step`` and ``seed_support_rows`` are cut.

Exact blank-free supervised sequence likelihood and seed support census."""

import torch


def collapse_adjacent(sequence):
    return [phone for i, phone in enumerate(sequence) if i == 0 or sequence[i - 1] != phone]


def transcript_logprob(log_q, targets, target_lens, output_lens):
    """Sum paths whose adjacent-run collapse equals each target, without a blank."""
    b, t_max, _ = log_q.shape
    u_max = targets.shape[1]
    if bool((target_lens < 1).any()) or bool((output_lens < target_lens).any()):
        raise ValueError("blank-free target is empty or longer than its output")
    valid_pairs = torch.arange(1, u_max, device=targets.device)[None, :] < target_lens[:, None]
    if bool(((targets[:, 1:] == targets[:, :-1]) & valid_pairs).any()):
        raise ValueError("adjacent-identical target is outside blank-free support")
    neg = log_q.new_full((b, 1), -1.0e30)
    state = torch.cat([log_q[:, 0].gather(1, targets[:, :1].long()),
                       log_q.new_full((b, u_max - 1), -1.0e30)], dim=1)
    for t in range(1, t_max):
        advance = torch.cat([neg, state[:, :-1]], dim=1)
        emission = log_q[:, t].gather(1, targets.long())
        updated = torch.logaddexp(state, advance) + emission
        state = torch.where((t < output_lens)[:, None], updated, state)
    return state.gather(1, (target_lens - 1)[:, None]).squeeze(1)
