"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/lexlat_train.py.

Port note: only Design 3's curriculum is ported -- ``LEXLAT_FULL_LAM`` / ``LEXLAT_ONSET`` /
``LEXLAT_RAMP``, ``lexlat_lambda``, ``lambda_schedule`` and ``assert_ramp`` -- which the k2
runtime (``lexlat_k2_train``) imports and never re-declares.  The trie-DP runtime
(``LexlatRuntime``, ``LexlatSpec``, ``derive_max_candidates``, ``truncate_to_bigram``, the
context / escape budgets) is cut with the trie DP itself (see ``lexlat.py``).
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = [
    "LEXLAT_FULL_LAM",
    "LEXLAT_ONSET",
    "LEXLAT_RAMP",
    "assert_ramp",
    "lambda_schedule",
    "lexlat_lambda",
]


#: Design 3: ``lam_lex`` at full strength is 1 and is NOT swept (beta = 1 for the phone trigram and
#: the lexical increment is in the same nats-per-phone units).
LEXLAT_FULL_LAM = 1.0
#: Design 3: the on-set sub-epoch of the primary arm -- ``lam_lex = 0`` for sub-epochs 1-7.
LEXLAT_ONSET = 8
#: Design 3: the ramp is three sub-epochs (8, 9, 10 -> 1/3, 2/3, 1), the three strengths E0 is
#: measured at (design review amendment 4).
LEXLAT_RAMP = 3


def lexlat_lambda(epoch: int, *, onset: int, ramp: int, full: float = LEXLAT_FULL_LAM) -> float:
    """``lam_lex`` of sub-epoch ``epoch`` (1-based), Design 3's curriculum.

    ``0`` strictly before ``onset``; a LINEAR ramp ``full * i / ramp`` at the ``i``-th sub-epoch of
    the ramp (``i = 1`` at ``onset``); ``full`` from ``onset + ramp - 1`` on.  At the pre-registered
    ``onset = 8, ramp = 3`` that is 0 for 1-7, then 1/3, 2/3, 1 at 8, 9, 10 and 1 for 11-20; at the
    extra arm's ``onset = 5`` the same three values sit at 5, 6, 7 (Design 5).
    """
    onset, ramp = int(onset), int(ramp)
    assert onset >= 1, f"the on-set sub-epoch is 1-based, got {onset}"
    assert ramp >= 1, f"the ramp is at least one sub-epoch, got {ramp}"
    if int(epoch) < onset:
        return 0.0
    step = int(epoch) - onset + 1
    return float(full) * min(1.0, step / float(ramp))


def lambda_schedule(n_subepochs: int, *, onset: int, ramp: int,
                    full: float = LEXLAT_FULL_LAM) -> Sequence[float]:
    """The whole run's ``lam_lex`` per sub-epoch -- what a config asserts before it launches."""
    return [lexlat_lambda(e, onset=onset, ramp=ramp, full=full)
            for e in range(1, int(n_subepochs) + 1)]


def assert_ramp(n_subepochs: int, *, onset: int, ramp: int, full: float = LEXLAT_FULL_LAM) -> None:
    """Design 3's shape, checked where the arm is built: zero, then the ramp, then full strength."""
    sched = lambda_schedule(n_subepochs, onset=onset, ramp=ramp, full=full)
    assert all(v == 0.0 for v in sched[: onset - 1]), sched
    assert sched[onset - 1] == full / ramp, sched
    assert onset + ramp - 2 < len(sched), (
        f"the ramp (on-set {onset}, {ramp} sub-epochs) does not finish inside the run's "
        f"{n_subepochs} sub-epochs"
    )
    assert math.isclose(sched[onset + ramp - 2], full), sched
    assert all(math.isclose(v, full) for v in sched[onset + ramp - 2:]), sched
