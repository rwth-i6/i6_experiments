"""
The lattice of the monotonic transducer from given log probabilities, its full sum and paths drawn from its posterior.

A position (frame t, labels emitted n) has two edges, blank keeps n and the next label raises it by one. The caller
gives the log probabilities of both edges, padded [B, T, U + 1], minus infinity where an edge does not exist.
"""

from typing import Optional, Tuple

import torch


def forward_scores(
    blank_lp: torch.Tensor, label_lp: torch.Tensor, frame_lens: torch.Tensor, label_lens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs the forward recursion over the frames.

    :param blank_lp: [B, T, U + 1] blank log probability of every position
    :param label_lp: [B, T, U + 1] next label log probability of every position
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :return: the log probability of all alignments [B] and of reaching every position [B, T + 1, U + 1], which
        stays as it was past the last frame of a sequence
    """
    batch_size, max_frames, max_prefix = blank_lp.shape
    frame_lens = frame_lens.to(blank_lp.device)
    label_lens = label_lens.to(blank_lp.device).long()
    alpha = blank_lp.new_full((batch_size, max_prefix), float("-inf"))
    alpha[:, 0] = 0.0
    pad = blank_lp.new_full((batch_size, 1), float("-inf"))
    alphas = [alpha]
    for t in range(max_frames):
        stay = alpha + blank_lp[:, t]
        move = torch.cat([pad, (alpha + label_lp[:, t])[:, :-1]], dim=1)
        # a position no edge reaches stays at minus infinity, where logaddexp has a nan gradient
        dead = torch.isneginf(stay) & torch.isneginf(move)
        updated = torch.logaddexp(stay.masked_fill(dead, 0.0), move.masked_fill(dead, 0.0))
        alpha = torch.where((t < frame_lens).unsqueeze(1), updated.masked_fill(dead, float("-inf")), alpha)
        alphas.append(alpha)
    return alpha.gather(1, label_lens.unsqueeze(1)).squeeze(1), torch.stack(alphas, dim=1)


def full_sum_loss(
    blank_lp: torch.Tensor, label_lp: torch.Tensor, frame_lens: torch.Tensor, label_lens: torch.Tensor
) -> torch.Tensor:
    """
    :param blank_lp: [B, T, U + 1] blank log probability of every position
    :param label_lp: [B, T, U + 1] next label log probability of every position
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :return: [B] minus the log probability of all alignments, zero (also in the gradient) for a sequence without
        frames or with more labels than frames, infinite where no alignment has any probability
    """
    total, _alpha = forward_scores(blank_lp, label_lp, frame_lens, label_lens)
    return torch.where(_alignable(frame_lens, label_lens, total.device), -total, torch.zeros_like(total))


def draw_paths(
    blank_lp: torch.Tensor,
    label_lp: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
    *,
    num_paths: int,
    uniform: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Draws alignments from the posterior, each backwards from the last frame of its sequence. At every frame a path
    came by blank from the same label count or by the label from one count lower, in proportion to the forward score
    of the position it came from times the edge, which makes every path an exact draw from the posterior.

    :param blank_lp: [B, T, U + 1] blank log probability of every position
    :param label_lp: [B, T, U + 1] next label log probability of every position
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param num_paths: paths per sequence
    :param uniform: [B, num_paths, T] the uniform random numbers of the draws, new ones where not given
    :return: [B, num_paths, T] the frames on which a path emits its next label, none in a sequence without frames,
        with more labels than frames or without any alignment of some probability
    """
    if uniform is None:
        uniform = torch.rand(blank_lp.shape[0], num_paths, blank_lp.shape[1], device=blank_lp.device)
    assert uniform.shape == (blank_lp.shape[0], num_paths, blank_lp.shape[1]), (uniform.shape, blank_lp.shape)
    if blank_lp.is_cuda:
        from .monotonic_lattice_kernel import draw_paths_kernel

        return draw_paths_kernel(
            blank_lp.float().contiguous(),
            label_lp.float().contiguous(),
            frame_lens.to(blank_lp.device).contiguous(),
            label_lens.to(blank_lp.device).contiguous(),
            uniform.float().contiguous(),
        ).bool()
    return _draw_paths_reference(blank_lp, label_lp, frame_lens, label_lens, uniform)


def _alignable(frame_lens: torch.Tensor, label_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    :return: [B] whether a sequence has frames and at most as many labels as frames
    """
    frame_lens, label_lens = frame_lens.to(device), label_lens.to(device)
    return (label_lens <= frame_lens) & (frame_lens > 0)


@torch.no_grad()
def _draw_paths_reference(
    blank_lp: torch.Tensor,
    label_lp: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
    uniform: torch.Tensor,
) -> torch.Tensor:
    """
    The draws of :func:`draw_paths` as torch ops, the reference of the kernel.
    """
    total, alpha = forward_scores(blank_lp, label_lp, frame_lens, label_lens)
    batch_size, num_paths, max_frames = uniform.shape
    device = blank_lp.device
    frame_lens = frame_lens.to(device)
    feasible = (_alignable(frame_lens, label_lens, device) & torch.isfinite(total)).unsqueeze(1)
    count = label_lens.to(device).long().unsqueeze(1).expand(-1, num_paths).clone()
    emits = torch.zeros(batch_size, num_paths, max_frames, dtype=torch.bool, device=device)
    for t in reversed(range(max_frames)):
        before = alpha[:, t]
        lower = (count - 1).clamp(min=0)
        stay = before.gather(1, count) + blank_lp[:, t].gather(1, count)
        move = (before.gather(1, lower) + label_lp[:, t].gather(1, lower)).masked_fill(count == 0, float("-inf"))
        label_prob = (move - torch.logaddexp(stay, move)).exp()
        emit = feasible & (t < frame_lens).unsqueeze(1) & (uniform[:, :, t].to(label_prob.dtype) < label_prob)
        emits[:, :, t] = emit
        count = count - emit.long()
    return emits
