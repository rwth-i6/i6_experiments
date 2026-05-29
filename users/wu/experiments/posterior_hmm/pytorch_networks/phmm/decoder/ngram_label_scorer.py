"""
A pure-Python KenLM n-gram label scorer for LibRASR's `lexiconfree-timesync-beam-search`.

LibRASR has no built-in ARPA/n-gram *label* scorer (the classic `Lm::` ARPA machinery
only serves the word/lexicon-based search). But recent RASR exposes
`librasr.register_label_scorer_type(name, cls)`, letting us add a custom label scorer in
Python. This module builds such a class around a KenLM model so a count phoneme n-gram can
be dropped into the same `combine` label scorer that currently hosts the stateful-ONNX
neural LM -- as `scorer-2`, with `scorer-1` still the `no-op` AM.

Semantics mirror `StatefulOnnxLabelScorer`:
  * The scoring context is the emitted label history. Blank/loop transitions do NOT advance
    it (matching `blank_updates_history=False`, `loop_updates_history=False`).
  * `compute_scores_with_times` returns, per context, a vector of `-ln p_LM(token | history)`
    over ALL label indices plus a timeframe. The timeframe is the history length; the
    `combine` scorer takes the max over sub-scorers, and the AM frame index dominates, so it
    only needs to be monotonic.
  * `<s>` is never emitted -> its cost is +inf (same convention as the AM `-1e30`-padding of
    the bos/eos indices). KenLM scores `</s>` and all phonemes normally; tokens never seen in
    the LM text (e.g. silence) fall back to `<unk>`, exactly as the neural LM effectively does.

Scores are KenLM log10 probabilities; we convert to natural-log cost `-ln p = -log10 p * ln 10`.
RASR multiplies by the `scorer-2.scale` (= lm_scale) in the combine config, so we return the
unscaled cost here.
"""

import math
from functools import lru_cache
from typing import List, Sequence

INF_COST = 1.0e30
_LN10 = math.log(10.0)

# Guard against double-registration within one process (init() runs once per forward job,
# but be defensive against re-entry / multiple segments).
_REGISTERED = set()


def register_ngram_label_scorer(
    *,
    scorer_name: str,
    kenlm_binary_file: str,
    token_strings: Sequence[str],
    bos_index: int,
):
    """
    Build and register a KenLM-backed `librasr.LabelScorer` subclass under `scorer_name`.

    Must be called (in the forward process) BEFORE the `SearchAlgorithm` is constructed, so
    that RASR can instantiate `scorer_name` from its label-scorer factory.

    :param scorer_name: type name referenced by `scorer-2.type` in the RASR config
    :param kenlm_binary_file: path to the KenLM `.bin` (or ARPA) model
    :param token_strings: label-index -> phoneme symbol, in RASR label order (length = #labels)
    :param bos_index: label index of `<s>` (never a valid emission -> +inf cost)
    """
    import kenlm
    import librasr

    if scorer_name in _REGISTERED:
        return

    model = kenlm.Model(kenlm_binary_file)
    tokens: List[str] = list(token_strings)
    num_labels = len(tokens)
    order = model.order
    context_len = max(order - 1, 0)

    # Transitions on which the LM history does NOT advance (no new phoneme emitted).
    _NON_EMITTING = {
        librasr.TransitionType.LABEL_TO_BLANK,
        librasr.TransitionType.BLANK_LOOP,
        librasr.TransitionType.INITIAL_BLANK,
        librasr.TransitionType.LABEL_LOOP,
    }

    @lru_cache(maxsize=200000)
    def _scores_for_window(window):
        """
        :param window: tuple of the last <=context_len emitted label indices (the n-gram
            equivalence class). If it begins with bos_index we are still near sentence start
            and use KenLM's begin-sentence context; otherwise a null context.
        :return: list of -ln p costs over all label indices.
        """
        state = kenlm.State()
        if window and window[0] == bos_index:
            model.BeginSentenceWrite(state)
            feed = window[1:]
        else:
            model.NullContextWrite(state)
            feed = window
        for idx in feed:
            nxt = kenlm.State()
            model.BaseScore(state, tokens[idx], nxt)
            state = nxt

        costs = [INF_COST] * num_labels
        out = kenlm.State()
        for t in range(num_labels):
            if t == bos_index:
                continue  # <s> is never a valid emission
            log10p = model.BaseScore(state, tokens[t], out)
            costs[t] = -log10p * _LN10
        return costs

    class _NGramLabelScorer(librasr.LabelScorer):
        """History context = full tuple of emitted label indices starting with `<s>`."""

        def __init__(self, config):
            super().__init__(config)

        def reset(self):
            pass

        def signal_no_more_features(self):
            pass

        def add_inputs(self, inputs):
            # Pure LM: acoustic features are irrelevant.
            pass

        def get_initial_scoring_context(self):
            return (bos_index,)

        def extended_scoring_context(self, context, next_token, transition_type):
            if transition_type in _NON_EMITTING:
                return context
            return context + (int(next_token),)

        def compute_scores_with_times(self, contexts):
            results = []
            for ctx in contexts:
                window = ctx[-context_len:] if context_len > 0 else ()
                results.append((_scores_for_window(window), len(ctx)))
            return results

    librasr.register_label_scorer_type(scorer_name, _NGramLabelScorer)
    _REGISTERED.add(scorer_name)
