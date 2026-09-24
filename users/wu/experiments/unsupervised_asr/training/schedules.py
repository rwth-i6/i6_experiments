"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_budget_jobs.py
(``budget_temperature_schedule``, ``budget_learning_rates``), with the phase-4a schedule helpers of
``prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/``:
``config_sae_4a_prepro_pack_v1._schedules`` (-> :func:`phase_schedules`),
``config_sae_4a_lexlat_k2_ext_v1._schedules`` (-> :func:`extend_schedule`) and
``config_sae_4a_lexlat_k2_prior_ablation_v1.prior_schedule`` (-> :func:`phone_trigram_weight_schedule`).

Every list here is per SUB-EPOCH, 1-based by position (entry ``e - 1`` is sub-epoch ``e``), and is
held at its last entry by its reader beyond its end; the builders therefore always return exactly
``n`` entries.
"""

import math
from typing import List, Sequence, Tuple

from .config import THETA_LEARNING_RATE

__all__ = [
    "LR_WARM_FRAC",
    "budget_temperature_schedule",
    "budget_learning_rates",
    "phase_schedules",
    "extend_schedule",
    "PHONE_TRIGRAM_MODES",
    "phone_trigram_weight_schedule",
]

#: the warmup fraction of the phase's N = 20 learning-rate schedule (pre-registered 2026-09-21):
#: ceil(0.10 * 20) = 2 warmup entries, the minimum ``budget_learning_rates`` accepts
LR_WARM_FRAC = 0.10


def budget_temperature_schedule(n: int, anneal_frac: float = 0.2, tau0: float = 8.0, tau1: float = 2.0) -> List[float]:
    """The pre-registered temperature list of the budget round: length ``n``, one value per sub-epoch.

    Geometric from ``tau0`` to ``tau1`` over the first ``ceil(anneal_frac * n)`` entries -- entry 1
    is exactly ``tau0`` and entry ``ceil(anneal_frac * n)`` is exactly ``tau1`` -- and ``tau1`` for
    every entry after it.

    :raises ValueError: if the anneal segment would hold fewer than two entries, or does not fit in
        ``n``, or the two endpoints are not a decreasing pair of positive temperatures.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"a schedule over {n} sub-epochs states nothing")
    if not (tau0 > tau1 > 0.0):
        raise ValueError(f"an anneal needs tau0 > tau1 > 0, got tau0={tau0!r} tau1={tau1!r}")
    m = math.ceil(float(anneal_frac) * n)
    if not 2 <= m <= n:
        raise ValueError(
            f"anneal_frac={anneal_frac!r} puts the end of the anneal at entry {m} of {n}; the "
            "pre-registered shape needs at least two entries (the first is tau0, the last tau1) "
            "and cannot run past the end of the run"
        )
    ratio = float(tau1) / float(tau0)
    out = []
    for i in range(1, n + 1):
        if i >= m:
            out.append(float(tau1))
        else:
            out.append(float(tau0) * ratio ** ((i - 1) / (m - 1)))
    assert len(out) == n and out[0] == float(tau0) and out[m - 1] == float(tau1)
    return out


def budget_learning_rates(
    n: int,
    peak: float = 1.0e-4,
    warm_frac: float = 0.05,
    hold_until: float = 0.6,
    floor: float = 0.1,
) -> List[float]:
    """The pre-registered base learning rate of the budget round: length ``n``, one per sub-epoch.

    Linear warmup from ``floor * peak`` at entry 1 to ``peak`` at entry ``ceil(warm_frac * n)``,
    ``peak`` until entry ``floor(hold_until * n)``, then a linear decay to ``floor * peak`` at entry
    ``n``.  ``floor`` is a FRACTION of the peak (0.1 = the design's "0.1x"), not a rate.

    :raises ValueError: if the warmup would hold fewer than two entries, if the hold ends before the
        warmup does, if there is no decay segment left, or if ``peak`` / ``floor`` are outside the
        shape the design states.
    """
    n = int(n)
    if n < 3:
        raise ValueError(f"a warmup / hold / decay schedule needs at least three sub-epochs, got {n}")
    if peak <= 0.0:
        raise ValueError(f"peak learning rate must be positive, got {peak!r}")
    if not 0.0 < floor < 1.0:
        raise ValueError(f"floor is a fraction of the peak and must lie in (0, 1), got {floor!r}")
    w = math.ceil(float(warm_frac) * n)
    h = math.floor(float(hold_until) * n)
    if not 2 <= w <= h < n:
        raise ValueError(
            f"warm_frac={warm_frac!r} / hold_until={hold_until!r} put the end of the warmup at "
            f"entry {w} and the end of the hold at entry {h} of {n}; the pre-registered shape needs "
            "at least two warmup entries, a hold that ends no earlier than the warmup, and at least "
            "one decay entry"
        )
    low = float(floor) * float(peak)
    out = []
    for i in range(1, n + 1):
        if i < w:
            out.append(low + (float(peak) - low) * (i - 1) / (w - 1))
        elif i <= h:
            out.append(float(peak))
        else:
            # written as an interpolation TOWARDS the floor so that entry n is exactly floor * peak
            out.append(low + (float(peak) - low) * (n - i) / (n - h))
    assert len(out) == n and out[0] == low and out[w - 1] == float(peak) and out[-1] == low
    return out


def phase_schedules(num_subepochs: int = 20) -> Tuple[List[float], List[float]]:
    """``(temperature_schedule, learning_rates)`` of the phase-4a arms (``prepro._schedules``).

    At the phase's N = 20: ``budget_temperature_schedule(20)`` (8 -> 2 over four sub-epochs, then
    2.0) and ``budget_learning_rates(20, peak=THETA_LEARNING_RATE, warm_frac=LR_WARM_FRAC)`` (two
    warmup entries, the peak to sub-epoch 12, linear back to the floor at 20).

    ``num_subepochs = 60`` is the E60 run: those 20 entries followed by :func:`extend_schedule`'s 40
    held ones.  It is one 60-sub-epoch job whose first 20 sub-epochs are the 20-sub-epoch run.
    """
    n = int(num_subepochs)
    tau = budget_temperature_schedule(20)
    assert len(tau) == 20, tau
    assert [round(t, 2) for t in tau[:4]] == [8.0, 5.04, 3.17, 2.0], (
        f"the N = 20 anneal is {[round(t, 2) for t in tau[:4]]}, not the bed's own four temperatures"
    )
    assert all(t == 2.0 for t in tau[3:]), tau

    peak = THETA_LEARNING_RATE
    lr = budget_learning_rates(20, peak=peak, warm_frac=LR_WARM_FRAC)
    assert len(lr) == 20, lr
    # the warmup is two sub-epochs: entry 0 is the floor, entry 1 is already the peak
    assert math.ceil(LR_WARM_FRAC * 20) == 2, LR_WARM_FRAC
    assert lr[0] < peak and lr[1] == peak, lr[:3]
    # ... held to sub-epoch 12 (floor(0.6 * 20)) and then linear back down to the floor
    assert lr[11] == peak and lr[12] < peak, lr[10:14]
    assert lr[-1] == lr[0], (lr[0], lr[-1])
    if n == 20:
        return tau, lr
    if n == 60:
        return extend_schedule(tau, lr, 60)
    raise ValueError(f"the phase's arms run 20 or 60 sub-epochs, not {n}")


def extend_schedule(
    temperature_schedule: Sequence[float], learning_rates: Sequence[float], num_subepochs: int
) -> Tuple[List[float], List[float]]:
    """``(tau, lr)`` of a continuation to ``num_subepochs`` (E60: ``config_sae_4a_lexlat_k2_ext_v1``).

    The given schedules followed by their HELD values: the anneal's final temperature (the list's own
    last entry, 2.0) and the peak learning rate :data:`THETA_LEARNING_RATE` (1e-4).  Neither hold is
    typed: both are read off their owner and asserted against the plan's 2.0 / 1e-4.  At the phase's
    N = 20 -> 60 this appends 40 x tau 2.0 and 40 x lr 1e-4, which is what the banked
    ``PackedBlankfreeContinueJob.xsWAIaMMsWOw`` configs state.
    """
    tau_in = [float(t) for t in temperature_schedule]
    lr_in = [float(v) for v in learning_rates]
    assert len(tau_in) == len(lr_in), (len(tau_in), len(lr_in))
    n_parent, n = len(tau_in), int(num_subepochs)
    assert n > n_parent, f"a continuation to {n} sub-epochs of a {n_parent}-sub-epoch run adds nothing"

    hold_lr = float(THETA_LEARNING_RATE)
    assert hold_lr == 1e-4, (
        f"the plan holds the learning rate at the peak 1e-4 after sub-epoch {n_parent}; the bed's peak is now {hold_lr}"
    )
    assert max(lr_in) == hold_lr, (max(lr_in), hold_lr)
    hold_tau = float(tau_in[-1])
    assert hold_tau == 2.0, (
        f"the plan holds the temperature at the anneal's final 2.0; the schedule ends at {hold_tau}"
    )
    n_extra = n - n_parent
    tau = tau_in + [hold_tau] * n_extra
    lr = lr_in + [hold_lr] * n_extra
    assert len(tau) == len(lr) == n, (len(tau), len(lr))
    return tau, lr


#: the three phone-trigram settings of a k2 arm
PHONE_TRIGRAM_MODES = ("full", "rampout", "off")


def phone_trigram_weight_schedule(mode: str, n_epochs: int, onset: int, ramp: int, full_lam: float) -> List[float]:
    """The per-sub-epoch phone-trigram weight beta of a k2 arm (``prior_weight_schedule``).

    beta scales the phone trigram inside the l_tau lattice DP and the rate term's passes (the
    model's ``prior_weight_at``).  ``onset`` / ``ramp`` / ``full_lam`` MUST be the arm's own
    ``lexlat_k2_onset`` / ``_ramp`` / ``_full_lam``: the word-graph weight is
    ``lam_lex(e) = lexlat_lambda(e, onset, ramp, full_lam)`` (0 before ``onset``, then linear to
    ``full_lam`` over ``ramp`` sub-epochs), the function the model's k2 runtime calls.

    * ``"full"``: 1.0 at every sub-epoch -- the phone trigram at full weight throughout, the banked
      ``k2lat_20_ma3000`` / ``off4_k2lat_20`` arms.  (Those configs carry no schedule at all, only
      the scalar ``prior_weight`` 1.0, which the model reads identically; a preset in this mode
      therefore writes no ``prior_weight_schedule`` key.)
    * ``"rampout"``: ``1 - lam_lex(e) / full_lam`` -- D15's replacement: the phone trigram gives way
      as the word graph ramps in (``[1]*7 + [2/3, 1/3] + [0]*11`` at N = 20, onset 8, ramp 3).
    * ``"off"``: 0.0 at every sub-epoch.  NOTE: the word graph only enters at ``onset``, so before the
      on-set this arm trains with NO language model term in the lattice at all (neither the phone
      trigram nor the word LM), not with a word LM instead of a phone LM.
    """
    if mode not in PHONE_TRIGRAM_MODES:
        raise ValueError(f"phone_trigram mode is one of {PHONE_TRIGRAM_MODES}, not {mode!r}")
    n = int(n_epochs)
    assert n >= 1, n
    if mode == "full":
        return [1.0] * n
    if mode == "off":
        return [0.0] * n
    from ..model.lexlat_train import lexlat_lambda

    onset, ramp, full = int(onset), int(ramp), float(full_lam)
    assert full > 0.0, full
    schedule = [1.0 - lexlat_lambda(e, onset=onset, ramp=ramp, full=full) / full for e in range(1, n + 1)]
    assert all(0.0 <= v <= 1.0 for v in schedule), schedule
    return schedule
