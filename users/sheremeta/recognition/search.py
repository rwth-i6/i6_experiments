"""Building blocks shared by the beam searches of the different models."""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def top_k_nd(
    source: torch.Tensor, *, k: int, dim: Sequence[int], sorted: bool = True
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Takes the top k over several dims at once, as if they were flattened into one.

    :param source: tensor to search
    :param k: number of entries to keep
    :param dim: dims to search jointly, negative values allowed
    :param sorted: whether the k results come in descending order
    :return: (values, indices) with values the top k and indices one index tensor per entry
        of ``dim``, in the same order
    """
    dim = [(d + source.ndim) % source.ndim for d in dim]
    source = source.permute([d for d in range(source.ndim) if d not in dim] + list(dim))
    source_flat = source.flatten(start_dim=source.ndim - len(dim))
    values, indices = torch.topk(source_flat, k=k, dim=-1, largest=True, sorted=sorted)
    indices_out: List[torch.Tensor] = []

    for i in reversed(list(range(len(dim)))):
        a_dim = source.shape[source.ndim - len(dim) + i]
        indices_out.insert(0, indices % a_dim)
        indices = indices // a_dim

    return values, indices_out


def write_at(buf: torch.Tensor, pos: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Writes one value per row into a buffer at a per-row position (only where a mask holds).

    :param buf: [..., L] buffer
    :param pos: [...] position along the last dim per row
    :param value: [...] value to write per row
    :param mask: [...] rows to write, the others keep their buffer entry
    :return: the updated buffer, ``buf`` itself is left unchanged
    """
    idx = pos.unsqueeze(-1)
    kept = buf.gather(-1, idx).squeeze(-1)
    return buf.scatter(-1, idx, torch.where(mask, value, kept).unsqueeze(-1))


def recombine_candidates(scores: torch.Tensor, keys: torch.Tensor, *, mode: str) -> torch.Tensor:
    """
    Merges candidates that share a key so only one of them survives, the others drop to -inf.

    Within a group the surviving member is the highest scoring one, ties go to the lowest
    index. Its score is the group's maximum or its log-sum-exp depending on ``mode``.

    :param scores: [B, N] candidate scores
    :param keys: [B, N, W] key per candidate, equal keys form a group
    :param mode: "max" keeps the best score, "sum" adds the group's probabilities
    :return: [B, N] scores with every non-surviving candidate at -inf
    """
    assert mode in ("max", "sum"), mode
    num_cand = scores.size(1)
    device = scores.device
    same = (keys.unsqueeze(2) == keys.unsqueeze(1)).all(dim=-1)
    neg_inf = torch.full((), float("-inf"), device=device, dtype=scores.dtype)
    grouped = torch.where(same, scores.unsqueeze(1), neg_inf)
    merged = grouped.max(dim=2).values if mode == "max" else grouped.logsumexp(dim=2)

    positions = torch.arange(num_cand, device=device)
    is_best = grouped == grouped.max(dim=2, keepdim=True).values
    rep = torch.where(
        is_best,
        positions.view(1, 1, -1),
        torch.full_like(same, num_cand, dtype=torch.int64),
    )
    keep = rep.min(dim=2).values == positions.view(1, -1)
    return torch.where(keep, merged, neg_inf)


def initial_scores(batch_size: int, beam_size: int, device: torch.device) -> torch.Tensor:
    """
    Beam scores before the first step, only beam 0 is alive so the first expansion yields K distinct continuations.

    :param batch_size: B
    :param beam_size: K
    :param device: device of the output
    :return: [B, K] with 0 at beam 0 and -inf elsewhere
    """
    scores = torch.full((batch_size, beam_size), float("-inf"), device=device)
    scores[:, 0] = 0.0
    return scores


def expand_beams(x: Optional[torch.Tensor], beam_size: int) -> Optional[torch.Tensor]:
    """
    Repeats every sequence once per beam, so row b * K + k of the result belongs to sequence b and beam k.

    :param x: [B, ...] per-sequence tensor, None passes through
    :param beam_size: K
    :return: [B * K, ...]
    """
    return None if x is None else x.repeat_interleave(beam_size, dim=0)


def flat_backrefs(beam_idx: torch.Tensor) -> torch.Tensor:
    """
    Turns parent beams given per sequence into row indices of the beam-major [B * K] layout.

    :param beam_idx: [B, K] parent beam of every entry within its own sequence
    :return: [B * K] parent row of every entry
    """
    batch_size, beam_size = beam_idx.shape
    offset = torch.arange(batch_size, device=beam_idx.device).unsqueeze(1) * beam_size
    return (offset + beam_idx).reshape(batch_size * beam_size)


def identity_beam(batch_size: int, beam_size: int, device: torch.device) -> torch.Tensor:
    """
    Parent beams that keep every entry where it is.

    :return: [B, K] with entry k holding k
    """
    return torch.arange(beam_size, device=device).unsqueeze(0).expand(batch_size, beam_size).contiguous()


def where_rows(mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    :func:`torch.where` with a mask over the leading dims only, broadcast over the trailing dims of the values.

    :param mask: [B] or [B, K] bool
    :param a: value where the mask holds
    :param b: value elsewhere
    :return: tensor shaped like ``b``
    """
    return torch.where(mask.reshape(*mask.shape, *([1] * (b.ndim - mask.ndim))), a, b)


def log_probs_per_beam(logits: torch.Tensor, beam_size: int) -> torch.Tensor:
    """
    Log-softmax of beam-major logits, unfolded into one row per sequence.

    :param logits: [B * K, V]
    :param beam_size: K
    :return: [B, K, V] float32
    """
    return F.log_softmax(logits.float(), dim=-1).view(-1, beam_size, logits.size(-1))


def one_hot_log_prob(vocab: int, symbol: int, *, device: torch.device) -> torch.Tensor:
    """
    Log-prob row that allows one symbol only.

    :return: [V] with 0 at ``symbol`` and -inf elsewhere
    """
    row = torch.full((vocab,), float("-inf"), device=device)
    row[symbol] = 0.0
    return row


def freeze_ended(log_prob: torch.Tensor, ended: torch.Tensor, symbol: int) -> torch.Tensor:
    """
    Gives the ended entries one continuation at log-prob 0, so they keep their score and emit ``symbol``.

    :param log_prob: [B, K, V]
    :param ended: [B, K] bool
    :param symbol: symbol the ended entries emit
    :return: [B, K, V]
    """
    row = one_hot_log_prob(log_prob.size(-1), symbol, device=log_prob.device)
    return torch.where(ended.unsqueeze(-1), row.view(1, 1, -1), log_prob)


def length_normalized(raw: torch.Tensor, length: torch.Tensor, exponent: float) -> torch.Tensor:
    """
    Divides summed log-probs by the length raised to the exponent, exponent 0 returns them unchanged.

    :param raw: [...] summed log-probs
    :param length: [...] number of emitted symbols, clamped to at least one
    :param exponent: length normalization exponent
    :return: [...] normalized scores
    """
    if exponent == 0.0:
        return raw
    return raw / length.clamp(min=1).to(raw.dtype).pow(exponent)


def pre_beam(scores: torch.Tensor, *, k: int, pre_beam_mult: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keeps the best K * pre_beam_mult candidates by a cheap score before an expensive term is added.

    At least K candidates survive, so a zero expensive term reproduces plain beam search.

    :param scores: [B, K, V] candidate scores
    :param k: K
    :param pre_beam_mult: multiple of K to keep
    :return: (scores [B, P], parent beam [B, P], symbol [B, P])
    """
    p = min(k * pre_beam_mult, k * scores.size(-1))
    values, (parent, symbol) = top_k_nd(scores, k=p, dim=[1, 2])
    return values, parent, symbol


def select_from_pre_beam(
    scores: torch.Tensor, parent: torch.Tensor, symbol: torch.Tensor, *, k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Final top K over the pre-beam candidates by their full score.

    :param scores: [B, P] full scores
    :param parent: [B, P] parent beam per candidate
    :param symbol: [B, P] symbol per candidate
    :param k: K
    :return: (scores [B, K], index into the pre-beam [B, K], parent beam [B, K], symbol [B, K])
    """
    values, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
    return values, idx, torch.gather(parent, 1, idx), torch.gather(symbol, 1, idx)


def order_beams(scores: torch.Tensor, *tensors: torch.Tensor, n: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
    """
    Sorts the beams of every sequence by score, best first, and reorders per-beam tensors alike.

    :param scores: [B, K]
    :param tensors: [B, K, ...] tensors indexed by beam
    :param n: number of beams to keep, all when None
    :return: (scores [B, N], the tensors reordered [B, N, ...])
    """
    order = scores.argsort(dim=1, descending=True)
    if n is not None:
        order = order[:, :n]
    reordered = [
        torch.gather(x, 1, order.view(*order.shape, *([1] * (x.ndim - 2))).expand(*order.shape, *x.shape[2:]))
        for x in tensors
    ]
    return (torch.gather(scores, 1, order), *reordered)


def best_beam(scores: torch.Tensor, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Picks the best scoring beam of every sequence out of per-beam tensors.

    :param scores: [B, K]
    :param tensors: [B, K, ...]
    :return: the tensors at the best beam, [B, ...] each
    """
    rows = torch.arange(scores.size(0), device=scores.device)
    best = scores.argmax(dim=1)
    return tuple(x[rows, best] for x in tensors)


@dataclass
class BeamHistory:
    """Symbols and parent beams recorded at every step, replayed backwards into full hypotheses."""

    symbols: List[torch.Tensor] = field(default_factory=list)
    parents: List[torch.Tensor] = field(default_factory=list)

    def record(self, symbol: torch.Tensor, parent: torch.Tensor) -> None:
        """
        :param symbol: [B, K] symbol every entry emitted at this step
        :param parent: [B, K] beam every entry continued
        """
        self.symbols.append(symbol)
        self.parents.append(parent)

    def hyps(self) -> torch.Tensor:
        """
        Follows every final entry back through the parents, so the result lists the symbols it actually emitted.

        :return: [B, K, T] symbols per surviving beam
        """
        batch_size, beam_size = self.symbols[0].shape
        device = self.symbols[0].device
        hyps = torch.zeros((batch_size, beam_size, len(self.symbols)), dtype=torch.long, device=device)
        cur = identity_beam(batch_size, beam_size, device)
        for t in range(len(self.symbols) - 1, -1, -1):
            hyps[:, :, t] = torch.gather(self.symbols[t], 1, cur)
            cur = torch.gather(self.parents[t], 1, cur)
        return hyps


class LabelSyncBeam:
    """
    Running state of a label-synchronous beam, every entry emits one symbol per step.

    Holds the summed log-prob, the number of emitted symbols for the length normalization, the ended flag
    and the history of every entry. Ended entries keep emitting ``end_symbol`` at log-prob 0. The caller
    decides after every :meth:`expand` which entries have ended and writes ``ended``.
    """

    def __init__(
        self, batch_size: int, beam_size: int, *, end_symbol: int, length_norm_exponent: float, device: torch.device
    ):
        self.beam_size = beam_size
        self.end_symbol = end_symbol
        self.exponent = float(length_norm_exponent)
        self.raw = initial_scores(batch_size, beam_size, device)
        self.length = torch.zeros((batch_size, beam_size), dtype=torch.long, device=device)
        self.ended = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=device)
        self.history = BeamHistory()

    def expand(self, log_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Keeps the best K continuations over all entries and symbols, ranked by the length-normalized score.

        :param log_prob: [B, K, V] log-probs of the next symbol per entry
        :return: (parent beam [B, K], symbol [B, K], parent had ended [B, K])
        """
        log_prob = freeze_ended(log_prob, self.ended, self.end_symbol)
        alive = (~self.ended).unsqueeze(-1)
        cand_raw = self.raw.unsqueeze(-1) + log_prob
        cand_length = (self.length.unsqueeze(-1) + alive.long()).expand_as(cand_raw)
        _, (parent, symbol) = top_k_nd(
            length_normalized(cand_raw, cand_length, self.exponent), k=self.beam_size, dim=[1, 2]
        )
        flat = parent * log_prob.size(-1) + symbol
        self.raw = cand_raw.flatten(1).gather(1, flat)
        self.length = cand_length.flatten(1).gather(1, flat)
        parent_ended = self.ended.gather(1, parent)
        self.history.record(symbol, parent)
        return parent, symbol, parent_ended

    def scores(self) -> torch.Tensor:
        """
        :return: [B, K] length-normalized score of every entry
        """
        return length_normalized(self.raw, self.length, self.exponent)
