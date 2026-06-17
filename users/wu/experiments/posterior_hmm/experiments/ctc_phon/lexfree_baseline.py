"""
Lexicon-free CTC recognition (phoneme Transformer LM + count KenLM n-gram), reusing the CTC AM.

Thin topology-bound wrappers around the shared :mod:`...phon_lexfree` drivers. Mirror of the pHMM
lexfree path; the only name-implied differences are CTC label semantics: the phoneme-only lexicon is
built from the CTC lexicon ([BLANK] at index 0, special="blank"), the search collapses repeated
labels around the blank, and the decoder's silence_label is [BLANK]. The phoneme LMs are shared with
the pHMM lexfree path (identical phoneme vocab) and reused by content hash.
"""
from .baseline import eow_phon_ctc_ls960_base
from ...phon_lexfree import lexfree_neural_search, lexfree_count_search

# The CTC blank lemma's phoneme; index 0 of the lexfree lexicon (must exist in the lexicon).
_BLANK_LABEL = "[BLANK]"


def lexfree_eow_phon_ctc_ls960(**sweep_kwargs):
    """Neural Transformer phoneme LM (stateful-ONNX label scorer). See :func:`lexfree_neural_search`."""
    am_artifacts = eow_phon_ctc_ls960_base()
    ctc_lexicon = am_artifacts["ctc_lexicon"]
    return lexfree_neural_search(
        am_artifacts=am_artifacts,
        source_lexicon=ctc_lexicon,
        per_lexicon=ctc_lexicon,
        lexfree_silence_phoneme=_BLANK_LABEL,
        decoder_silence_label=_BLANK_LABEL,
        collapse_repeated_labels=True,
        **sweep_kwargs,
    )


def lexfree_count_eow_phon_ctc_ls960(**sweep_kwargs):
    """Count phoneme n-gram (KenLM) LM. See :func:`lexfree_count_search`."""
    am_artifacts = eow_phon_ctc_ls960_base()
    ctc_lexicon = am_artifacts["ctc_lexicon"]
    # Focused count-LM recognition: prior=0.3 + LM-scale sweep on dev-other at the final checkpoint
    # (matches the pHMM sweep epoch). Caller may override.
    sweep_kwargs.setdefault("sweep_epoch", 125)
    return lexfree_count_search(
        am_artifacts=am_artifacts,
        source_lexicon=ctc_lexicon,
        per_lexicon=ctc_lexicon,
        lexfree_silence_phoneme=_BLANK_LABEL,
        decoder_silence_label=_BLANK_LABEL,
        collapse_repeated_labels=True,
        **sweep_kwargs,
    )
