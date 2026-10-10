import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

# the float32 floor, below which two impossible edges stay at minus infinity instead of giving nan
_FLOOR = tl.constexpr(-3.4028234663852886e38)


@triton.jit
def _log_add(a, b):
    top = tl.maximum(tl.maximum(a, b), _FLOOR)
    return top + tl.log(tl.exp(a - top) + tl.exp(b - top))


@triton.jit
def _draw_paths(
    blank_ptr,
    label_ptr,
    frame_lens_ptr,
    label_lens_ptr,
    uniform_ptr,
    alpha_ptr,
    emits_ptr,
    max_frames,
    max_prefix,
    num_paths,
    PREFIXES: tl.constexpr,
    PATHS: tl.constexpr,
):
    """
    One program per sequence. The forward recursion holds the label counts as lanes and writes the score of every
    position, then every path walks back from the last frame with the paths as lanes.
    """
    seq = tl.program_id(0)
    frame_len = tl.load(frame_lens_ptr + seq).to(tl.int32)
    label_len = tl.load(label_lens_ptr + seq).to(tl.int32)
    edges = seq.to(tl.int64) * max_frames * max_prefix
    rows = alpha_ptr + seq.to(tl.int64) * (max_frames + 1) * max_prefix

    counts = tl.arange(0, PREFIXES)
    in_block = counts < max_prefix
    valid = in_block & (counts <= label_len)
    tl.store(rows + counts, tl.where(counts == 0, 0.0, float("-inf")), mask=in_block)
    for t in range(0, max_frames):
        tl.debug_barrier()
        here = rows + t * max_prefix + counts
        alpha = tl.load(here, mask=in_block, other=float("-inf"))
        alpha_lower = tl.load(here - 1, mask=in_block & (counts >= 1), other=float("-inf"))
        blank = tl.load(blank_ptr + edges + t * max_prefix + counts, mask=valid, other=float("-inf"))
        label_lower = tl.load(
            label_ptr + edges + t * max_prefix + counts - 1, mask=valid & (counts >= 1), other=float("-inf")
        )
        updated = tl.where(valid, _log_add(alpha + blank, alpha_lower + label_lower), float("-inf"))
        tl.store(here + max_prefix, tl.where(t < frame_len, updated, alpha), mask=in_block)
    tl.debug_barrier()

    paths = tl.arange(0, PATHS)
    path_mask = paths < num_paths
    final = tl.load(rows + max_frames * max_prefix + label_len)
    feasible = (final > float("-inf")) & (label_len <= frame_len) & (frame_len > 0)
    count = tl.zeros((PATHS,), tl.int32) + label_len
    draws = (seq * num_paths + paths).to(tl.int64) * max_frames
    for back in range(0, max_frames):
        t = max_frames - 1 - back
        lower = tl.maximum(count - 1, 0)
        stay = tl.load(rows + t * max_prefix + count, mask=path_mask, other=float("-inf"))
        stay += tl.load(blank_ptr + edges + t * max_prefix + count, mask=path_mask, other=float("-inf"))
        move = tl.load(rows + t * max_prefix + lower, mask=path_mask, other=float("-inf"))
        move += tl.load(label_ptr + edges + t * max_prefix + lower, mask=path_mask, other=float("-inf"))
        move = tl.where(count > 0, move, float("-inf"))
        label_prob = tl.exp(move - _log_add(stay, move))
        uniform = tl.load(uniform_ptr + draws + t, mask=path_mask, other=1.0)
        emit = feasible & (t < frame_len) & (uniform < label_prob)
        tl.store(emits_ptr + draws + t, emit.to(tl.int8), mask=path_mask)
        count -= emit.to(tl.int32)


@triton_op("sheremeta::monotonic_lattice_draw_paths", mutates_args={})
def draw_paths_kernel(
    blank_lp: torch.Tensor,
    label_lp: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
    uniform: torch.Tensor,
) -> torch.Tensor:
    """
    The cuda version of :func:`.monotonic_lattice.draw_paths`.

    :param blank_lp: [B, T, U + 1] float32, contiguous
    :param label_lp: [B, T, U + 1] float32, contiguous
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param uniform: [B, num_paths, T] float32, contiguous
    :return: [B, num_paths, T] int8, one where a path emits its next label
    """
    batch_size, max_frames, max_prefix = blank_lp.shape
    num_paths = uniform.shape[1]
    alpha = blank_lp.new_empty((batch_size, max_frames + 1, max_prefix))
    emits = torch.empty((batch_size, num_paths, max_frames), dtype=torch.int8, device=blank_lp.device)
    wrap_triton(_draw_paths)[(batch_size,)](
        blank_lp,
        label_lp,
        frame_lens,
        label_lens,
        uniform,
        alpha,
        emits,
        max_frames,
        max_prefix,
        num_paths,
        PREFIXES=triton.next_power_of_2(max_prefix),
        PATHS=triton.next_power_of_2(num_paths),
    )
    return emits
