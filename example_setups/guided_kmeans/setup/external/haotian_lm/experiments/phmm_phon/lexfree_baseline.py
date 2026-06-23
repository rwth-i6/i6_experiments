"""
Lexicon-free pHMM recognition (phoneme Transformer LM + count KenLM n-gram), reusing the pHMM AM.

Thin topology-bound wrappers around the shared :mod:`...phon_lexfree` drivers. The pHMM lexicon
([SILENCE] at index 0, no label collapsing) is the only name-implied difference vs. the CTC lexfree
path; the phoneme LMs are shared with the CTC path (identical phoneme vocab) and reused by content
hash. The AM is fetched from the existing baseline (nothing in the AM pipeline changes).
"""
from .baseline import eow_phon_phmm_ls960_base
from .baseline_noeow import phon_phmm_ls960_base
from ...phon_lexfree import lexfree_neural_search, lexfree_count_search
from ...experiments.lm_phon.trafo import phon_noeow_trafo_12x512_baseline


def lexfree_eow_phon_phmm_ls960(**sweep_kwargs):
    """Neural Transformer phoneme LM (stateful-ONNX label scorer). See :func:`lexfree_neural_search`."""
    am_artifacts = eow_phon_phmm_ls960_base()
    phmm_lexicon = am_artifacts["phmm_lexicon"]
    return lexfree_neural_search(
        am_artifacts=am_artifacts,
        source_lexicon=phmm_lexicon,
        per_lexicon=phmm_lexicon,
        **sweep_kwargs,
    )


def lexfree_count_eow_phon_phmm_ls960(**sweep_kwargs):
    """Count phoneme n-gram (KenLM) LM. See :func:`lexfree_count_search`."""
    am_artifacts = eow_phon_phmm_ls960_base()
    phmm_lexicon = am_artifacts["phmm_lexicon"]
    # Focused count-LM recognition: prior=0.3 + LM-scale sweep on dev-other at the final checkpoint.
    # Caller may override.
    sweep_kwargs.setdefault("sweep_epoch", 125)
    # COLLAPSE repeated labels. The pHMM AM has no blank and emits a phoneme on EVERY frame; the
    # frame-synchronous lexicon-free search must collapse consecutive duplicates or it emits one token
    # per frame -> a ~29% INSERTION explosion (count-LM PER 33-52, far WORSE than greedy 9.84). With
    # collapse on, each phone occurrence emits once (genuine adjacent repeats are rare without a blank).
    sweep_kwargs.setdefault("collapse_repeated_labels", True)
    return lexfree_count_search(
        am_artifacts=am_artifacts,
        source_lexicon=phmm_lexicon,
        per_lexicon=phmm_lexicon,
        **sweep_kwargs,
    )


def lexfree_phon_phmm_ls960(**sweep_kwargs):
    """Neural Transformer phoneme LM for the non-EOW pHMM inventory."""
    am_artifacts = phon_phmm_ls960_base()
    phmm_lexicon = am_artifacts["phmm_lexicon"]
    sweep_kwargs.setdefault("lm_name", "phon_noeow_trafo12x512_3ep")
    sweep_kwargs.setdefault("ensure_lm_func", phon_noeow_trafo_12x512_baseline)
    return lexfree_neural_search(
        am_artifacts=am_artifacts,
        source_lexicon=phmm_lexicon,
        per_lexicon=phmm_lexicon,
        **sweep_kwargs,
    )


def lexfree_count_phon_phmm_ls960(**sweep_kwargs):
    """Count phoneme n-gram LM for the non-EOW pHMM inventory."""
    am_artifacts = phon_phmm_ls960_base()
    phmm_lexicon = am_artifacts["phmm_lexicon"]
    sweep_kwargs.setdefault("sweep_epoch", 125)
    sweep_kwargs.setdefault("collapse_repeated_labels", True)
    sweep_kwargs.setdefault("use_eow_phonemes", False)
    return lexfree_count_search(
        am_artifacts=am_artifacts,
        source_lexicon=phmm_lexicon,
        per_lexicon=phmm_lexicon,
        **sweep_kwargs,
    )
