"""Check fixed-hypothesis alignment and candidate accounting without a GPU."""

from functools import cache

import numpy as np
import pytest

from i6_experiments.users.berger.model_pipelines.common import bidirectional_rescoring as runtime


@pytest.mark.parametrize("tdt", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 3])
def test_transducer_cost_agrees_with_exhaustive_alignments(tdt, tokens):
    rng = np.random.default_rng(28)
    frames = 3
    blank = rng.random((frames, tokens + 1))
    label = rng.random((frames, tokens))
    duration = rng.random((frames, tokens + 1, 5)) if tdt else None

    @cache
    def best(t, u):
        if t >= frames:
            return 0.0 if u == tokens else np.inf
        if duration is None:
            return min(blank[t, u] + best(t + 1, u), label[t, u] + best(t, u + 1) if u < tokens else np.inf)
        return min(
            min(blank[t, u] + duration[t, u, d] + best(t + d, u) for d in range(1, 5)),
            min(label[t, u] + duration[t, u, d] + best(t + d, u + 1) for d in range(5)) if u < tokens else np.inf,
        )

    assert runtime.transducer_viterbi(blank, label, duration) == pytest.approx(best(0, 0))


def test_eos_padding_and_capped_generation():
    assert runtime.generated_ids([1, 2, 8, 9, 9], [8], 9) == ([1, 2, 8], [1, 2])
    assert runtime.generated_ids([1, 2], [8], 9) == ([1, 2], [1, 2])
    assert runtime.generated_ids([1, 2, 9], [9], 9) == ([1, 2, 9], [1, 2])


def test_bidirectional_selection_and_unscoreable_fallback():
    a = dict(text="a", first_pass_cost=4.0, auxiliary_cost=2.0)
    b = dict(text="b", first_pass_cost=5.0, auxiliary_cost=0.5)
    c = dict(text="c", first_pass_cost=3.0, auxiliary_cost=None)
    candidates = runtime.ordered_candidates([a, dict(a, first_pass_cost=5.0), c, b], n=3)
    assert candidates == [c, a, b]
    assert runtime.pick_candidate(candidates, scale=0, speech_policy=None) == c
    assert runtime.pick_candidate(candidates, scale=0.1, speech_policy=None) == a
    assert runtime.pick_candidate(candidates, scale=1, speech_policy=None) == b
    assert runtime.pick_candidate([c], scale=1, speech_policy=None) == c
