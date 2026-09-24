"""Guard: the live server's adaptive jitter target (``moshi_family.personaplex.jitter.JitterTarget``).

The live-mic server steps the model on its own 80 ms clock from a queue whose size this controller picks
(2026-09-24; the user's tunnel delivered mic audio with gap p95 204 ms, max 719 ms). On synthetic arrival
traces (``simulate_arrivals`` + ``random_stalls``, pure Python, no torch):

  1. STEADY. A real-time stream with a constant network delay keeps the target at the floor (80 ms).
  2. P95 BURSTS. Stalls like the user's (gap p95 ~0.2 s, no long ones) settle the target between 200
     and 450 ms after 20 s (median), i.e. ~q95 lateness + a frame.
  3. SPIKE. One 700 ms stall grows the target at once (>= 600 ms = the ceiling) and it decays back to
     the floor within 30 s, one frame per hold period (never below the floor).
  4. BOUNDS. On 5 seeded traces with long stalls the target never leaves [floor, ceiling].
  5. CAN FAIL. A controller without the ceiling violates 4; one that never shrinks violates 3.

Run (any python3 with the recipe on PYTHONPATH; seconds)::

    PYTHONPATH=recipe/speech_llm/full_duplex:recipe python3 \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_jitter_buffer.py
"""

import statistics

from moshi_family.personaplex.jitter import JitterTarget, random_stalls, simulate_arrivals


def run(ctrl, arrivals):
    return [(a, ctrl.observe(a, d)) for a, d in arrivals]


def bounded(cls) -> bool:
    for seed in range(1, 6):
        c = cls()
        ts = run(c, simulate_arrivals(120, stalls=random_stalls(120, seed, p_long=0.01)))
        if any(not (c.floor_s - 1e-9 <= t <= c.ceil_s + 1e-9) for _, t in ts):
            return False
    return True


def spike_ok(cls) -> bool:
    c = cls()
    ts = run(c, simulate_arrivals(60, stalls=[(10.0, 0.7)]))
    peak = max(t for a, t in ts if 10.0 <= a <= 12.0)
    after = [t for a, t in ts if a >= 40.0]
    return peak >= 0.6 - 1e-9 and max(after) <= c.floor_s + 1e-9 and min(t for _, t in ts) >= c.floor_s - 1e-9


class _NoCeiling(JitterTarget):
    def _clamp(self, x):
        return max(self.floor_s, x)


class _NoShrink(JitterTarget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.shrink_hold_s = 1e9


def main():
    c = JitterTarget()
    ts = run(c, simulate_arrivals(60))
    assert max(t for _, t in ts) == c.floor_s, "a steady stream must stay at the floor"
    print("[ok] 1 steady stream stays at the floor (80 ms)")

    meds = []
    for seed in (1, 2, 3):
        c = JitterTarget()
        ts = run(c, simulate_arrivals(120, stalls=random_stalls(120, seed, p_long=0.0)))
        meds.append(statistics.median(t for a, t in ts if a > 20))
    assert all(0.2 <= m <= 0.45 for m in meds), meds
    print(f"[ok] 2 p95 ~0.2 s bursts settle at median target {[round(m * 1000) for m in meds]} ms")

    assert spike_ok(JitterTarget), "a single 700 ms stall must grow the target and decay back"
    print("[ok] 3 one 700 ms stall: grows to the ceiling, back at the floor within 30 s")

    assert bounded(JitterTarget), "target left [floor, ceiling]"
    print("[ok] 4 target stays within [80, 600] ms on 5 traces with long stalls")

    assert not bounded(_NoCeiling), "check 4 passes without the ceiling: it cannot fail"
    assert not spike_ok(_NoShrink), "check 3 passes without shrinking: it cannot fail"
    print("[ok] 5 no-ceiling and never-shrink controllers fail checks 4 and 3 (the guard can fail)")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
