"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/agg.py.

emc.agg -- the corpus-level term L_agg of SAE §4a, stage S0b (SAE_4A.md:78-80).

    L_agg = KL(c_text || c_hat) over phone unigrams and bigrams

``c_text`` are the m-gram distributions of the §1c text side (``prior.PhoneNgramPrior``); ``c_hat``
are the EXPECTED m-gram counts of the collapsed transcript ``B(path)`` under the recognizer's own
distribution ``q`` over CTC paths. It is a collapse guard, not an anchor: §1f showed low-order
matching cannot separate truth from a matched-length decoy (SAE_1f.md:682-687), so this term is
priced small and never reads as evidence.

Exactness
---------
``q(path | x) = prod_t q(path_t | x_t)`` is factorized per frame, so the expected counts of the
COLLAPSED string have a closed form -- no forward-backward and no sampling:

* a token of phone k STARTS at frame t iff ``path_t = k`` and ``path_{t-1} != k``, hence
  ``E[c_1(k)] = sum_t q_t(k) (1 - q_{t-1}(k))``;
* two tokens (h, k) are ADJACENT in the collapsed string iff some frame t emits k while the last
  non-blank frame before t carried h (and, when k = h, at least one blank lies between them).
  With ``A_t(h) = P(the last non-blank frame before t is h)``, which obeys the exact recursion
  ``A_{t+1}(h) = q_t(h) + q_t(blank) A_t(h)``, ``A_0 = 0``, this is
  ``E[c_2(h, k)] = sum_t A_t(h) q_t(k) - [h = k] sum_t q_{t-1}(h) q_t(h)``.

Both are exact for the CTC topology this phase trains under (the standard collapse: drop repeats,
then drop blanks). The SIL-run split the lattice allows (lattice.py docstring) is a property of the
LATTICE's latent, not of ``B``, and is deliberately not modelled here -- ``c_hat`` is the count of
the standard collapse. That is the one documented approximation of this module, and it can only
under-count adjacent SIL pairs.

Minibatch bias
--------------
The average sits inside the log (design review 2026-09-15, citing Yeh 2019 and Ni 2024): with ~16
utterances per step the expected-count estimate is noisy and biased, so the KL is taken against an
EXPONENTIAL MOVING AVERAGE of the expected counts across steps. The current step enters the EMA with
its gradient attached and the history enters detached, which is what makes the term steer at all.
Each step contributes its own count distribution (counts normalized per step BEFORE the EMA), so a
large batch does not dominate a small one.

``count_ema_decay`` and the two KL weights are REQUIRED constants with no default: they change the
objective, and the plan fixes neither (SAE_4A.md:78-80 states only "one small lambda_agg"). A config
must state them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "AggConfig",
    "expected_ngram_counts",
    "last_nonblank_table",
    "text_target_counts",
    "AggLoss",
]

# ``log q`` of a padded frame. Finite (so the cumulative log-blank stays finite) and far enough
# below any real log-prob that ``exp`` of it is 0 in every dtype this runs in.
_LOG_ZERO = -1e30


@dataclass
class AggConfig:
    """Shape and schedule of L_agg. ``count_ema_decay`` has NO default on purpose."""

    count_ema_decay: float  # required: the EMA of expected counts (design review 2026-09-15)
    n_phones: int = 40  # prior.PHONES (SAE_4A.md:40-41)
    blank_index: int = 0  # INTERFACES.md §1
    unigram_weight: float = 1.0  # L_agg = w1 * KL(unigram) + w2 * KL(bigram); equal by default
    bigram_weight: float = 1.0
    floor: float = 1e-8  # keeps log(c_hat) finite on a phone the recognizer never emits

    def __post_init__(self):
        assert 0.0 <= self.count_ema_decay < 1.0, "count_ema_decay is a decay in [0, 1)"
        assert self.blank_index == 0, "lattice.py and the recognizer both put blank at index 0"


def last_nonblank_table(log_probs: torch.Tensor, mask: torch.Tensor, n_phones: int) -> torch.Tensor:
    """``A[b, t, h] = P(the last non-blank frame before t carried h)``, ``[B, T, K]``, LOOP-FREE.

    The recursion ``A_{t+1} = q_t + q_blank_t * A_t``, ``A_0 = 0`` is linear in ``A`` with a scalar
    (per frame, shared over ``h``) multiplier, so it solves in closed form:

        ``A_t(h) = sum_{s<t} q_s(h) * exp(Lb_t - Lb_{s+1})``,  ``Lb_t = sum_{r<t} log q_blank_r``

    i.e. ONE cumulative sum plus ONE ``logcumsumexp`` -- the same hoisting the lattice DP does to its
    own frame loop (``lattice.py``: everything that does not depend on ``t`` leaves the loop, because
    the step is launch-bound).  The Python ``for t in range(T)`` this replaces cost three kernel
    launches per frame in the FORWARD and three more in the autograd BACKWARD, ~2 x 2 x 855 launches
    per step at the S1b shape, and it is the second per-frame loop of a step whose first one already
    dominates (reports/debug_s1b_step_time_2026-09-15.md).

    The cumulative log-blank arithmetic is in **float64** on purpose: ``|Lb_T|`` reaches ~1e3 on a
    real 855-frame utterance, ``log A_t = logcumsumexp(...)_{t-1} + Lb_t`` subtracts two numbers of
    that size, and in float32 that cancellation alone would cost ~1e-4 relative on every ``A`` (the
    sequential recursion never forms them).  The cost is a few [B, T, K] fp64 temporaries (35 MB at
    B = 128, T = 855, K = 40).

    ``log_probs`` must be FINITE (a ``log_softmax`` output is); a literal ``-inf`` blank makes
    ``Lb`` ``-inf`` and the closed form NaN, where the recursion would have given 0.

    Padded frames are given blank probability 1 here instead of 0, so ``A`` CARRIES its last value
    past the end of the utterance instead of dropping to 0.  Both are contracted against ``q_ph``,
    which IS zeroed past the end, so every consumer sees the same number.
    """
    b, t_max, _ = log_probs.shape
    m = mask.unsqueeze(-1)
    zero = log_probs.new_zeros((), dtype=torch.float64)
    # padding: blank prob 1 (A carries) and phone prob 0 (nothing is contributed or consumed)
    lb = torch.where(mask, log_probs[..., 0].double(), zero)  # [B, T]
    lq = torch.where(m, log_probs[..., 1 : 1 + n_phones].double(), zero + _LOG_ZERO)  # [B, T, K]
    lb_cum = torch.cat([lb.new_zeros(b, 1), lb.cumsum(dim=1)], dim=1)  # [B, T+1] = Lb_t
    u = lq - lb_cum[:, 1:].unsqueeze(-1)  # [B, T, K]: the s-th term, log q_s(h) - Lb_{s+1}
    cum = torch.logcumsumexp(u, dim=1)  # cum[t] = logsumexp_{s <= t} u_s
    log_a = cum[:, :-1] + lb_cum[:, 1:t_max].unsqueeze(-1)  # t = 1 .. T-1
    a = torch.cat([log_a.new_zeros(b, 1, n_phones), log_a.exp()], dim=1)  # A_0 = 0
    return a.to(log_probs.dtype)


def expected_ngram_counts(
    log_probs: torch.Tensor, lens: torch.Tensor, cfg: AggConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expected unigram ``[B, K]`` and bigram ``[B, K, K]`` counts of ``B(path)`` under ``q``.

    :param log_probs: ``[B, T, 1 + K]`` recognizer log-probs, blank at index 0.
    :param lens: ``[B]`` frame counts; padded frames contribute nothing (their phone probabilities
        are zeroed, so neither count can pick up mass past the end of an utterance).
    """
    b, t_max, n_sym = log_probs.shape
    k = cfg.n_phones
    assert n_sym == k + 1, (n_sym, k)
    probs = log_probs.exp()
    mask = (torch.arange(t_max, device=log_probs.device)[None, :] < lens.to(log_probs.device)[:, None])
    probs = probs * mask.unsqueeze(-1).to(probs.dtype)
    q_ph = probs[..., 1:]  # [B, T, K]
    prev = torch.cat([q_ph.new_zeros(b, 1, k), q_ph[:, :-1]], dim=1)  # q_{t-1}(k)

    uni = (q_ph * (1.0 - prev)).sum(dim=1)

    # A_t(h) = P(the last non-blank frame before t is h): the closed form of the T-step scan, so
    # this module adds NO per-frame loop to a step (see :func:`last_nonblank_table`).
    a_tab = last_nonblank_table(log_probs, mask, k)  # [B, T, K]
    bi = torch.einsum("bth,btk->bhk", a_tab, q_ph)
    # k = h is a NEW token only with a blank in between; remove the no-blank part q_{t-1}(h) q_t(h)
    same = (prev * q_ph).sum(dim=1)  # [B, K]
    bi = bi - torch.diag_embed(same)
    return uni, bi.clamp(min=0.0)


def text_target_counts(prior, *, n_phones: int = 40) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(c_text unigram [K], c_text bigram [K, K])`` from a fitted ``prior.PhoneNgramPrior``.

    The bigram target is the joint ``p(h) p(k | h)`` over the 40 phone types (the BOS context row of
    ``log_bi`` is a sentence-start context, not a token, and is dropped), renormalized. Both are the
    smoothed tables the prior itself publishes -- this module never re-counts the corpus.
    """
    import numpy as np

    uni = torch.as_tensor(np.asarray(prior.unigram_probs()), dtype=torch.float32)[:n_phones]
    uni = uni / uni.sum()
    cond = torch.as_tensor(np.exp(np.asarray(prior.log_bi)), dtype=torch.float32)[:n_phones, :n_phones]
    joint = uni.unsqueeze(-1) * cond
    return uni, joint / joint.sum()


class AggLoss(nn.Module):
    """KL(c_text || EMA of c_hat), unigram + bigram, with the EMA state in the module's buffers.

    The buffers ride in the ``state_dict``, so a resume continues the same average instead of
    restarting it (a restart would make the first steps after every resume a different objective).
    """

    def __init__(self, cfg: AggConfig, text_uni: torch.Tensor, text_bi: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        k = cfg.n_phones
        assert text_uni.shape == (k,) and text_bi.shape == (k, k)
        self.register_buffer("text_uni", text_uni.float() / text_uni.float().sum())
        self.register_buffer("text_bi", text_bi.float() / text_bi.float().sum())
        self.register_buffer("ema_uni", torch.zeros(k))
        self.register_buffer("ema_bi", torch.zeros(k, k))
        self.register_buffer("ema_steps", torch.zeros((), dtype=torch.long))

    def _blend(self, batch: torch.Tensor, ema: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
        """``decay * history (detached) + (1 - decay) * this step (with grad)``; step 1 is the step.

        ``first`` is a 0-dim BOOL TENSOR, not a Python bool: ``ema_steps`` lives on the device, and
        reading it with ``.item()`` synchronised the host on every training step.  Both branches are
        [K] / [K, K] tensors, so evaluating both and selecting is free, and the selected branch is
        bit-for-bit the one the ``if`` returned.
        """
        d = float(self.cfg.count_ema_decay)
        return torch.where(first, batch, d * ema.to(batch.dtype) + (1.0 - d) * batch)

    def forward(
        self, log_probs: torch.Tensor, lens: torch.Tensor, *, update_ema: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """``(L_agg scalar, diagnostics)``. The EMA advances only in ``train()`` mode."""
        cfg = self.cfg
        uni, bi = expected_ngram_counts(log_probs, lens, cfg)
        uni_b = uni.sum(dim=0)
        bi_b = bi.sum(dim=0)
        n_tok = uni_b.sum()
        uni_n = uni_b / n_tok.clamp(min=cfg.floor)
        bi_n = bi_b / bi_b.sum().clamp(min=cfg.floor)
        first = self.ema_steps == 0  # 0-dim bool TENSOR; a .item() here syncs the whole step
        uni_hat = self._blend(uni_n, self.ema_uni, first)
        bi_hat = self._blend(bi_n, self.ema_bi, first)
        text_uni = self.text_uni.to(uni_hat.dtype)
        text_bi = self.text_bi.to(bi_hat.dtype)
        kl_uni = (text_uni * (text_uni.clamp(min=cfg.floor).log() - uni_hat.clamp(min=cfg.floor).log())).sum()
        kl_bi = (text_bi * (text_bi.clamp(min=cfg.floor).log() - bi_hat.clamp(min=cfg.floor).log())).sum()
        loss = cfg.unigram_weight * kl_uni + cfg.bigram_weight * kl_bi
        if update_ema and self.training:
            with torch.no_grad():
                self.ema_uni.copy_(uni_hat.detach().float())
                self.ema_bi.copy_(bi_hat.detach().float())
                self.ema_steps += 1
        stats = {
            "agg/kl_unigram": float(kl_uni.detach()),
            "agg/kl_bigram": float(kl_bi.detach()),
            "agg/expected_tokens": float(n_tok.detach()),
        }
        return loss, stats
