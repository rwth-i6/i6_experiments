"""SAE Phase 0b — 𝒯_φ: boundary-free ARPAbet phonemization of the LibriSpeech LM corpus.

Reuses posterior_hmm's phonemization jobs (``data/phon_lm``: ``CollectOovWordsJob`` /
``TextToPhonemeJob`` / G2P training) — the canonical, heavily-used implementation — instead of
duplicating the lexicon / OOV / G2P / text→phoneme logic. SAE-specific here is only the wiring: the
*LM corpus only* (no audio transcripts), non-EOW plain phonemes, one whitespace-separated ARPAbet
stream per line (𝒯_φ) for the §1a KenLM and the §1.0 metric. One deterministic pronunciation per word
(SAE_PLAN §0b); residual-OOV lines (~0.06 %) are dropped by ``TextToPhonemeJob``. Downstream readers
(decipher / unsup_metric) keep only ``ARPABET_39`` tokens, so the drop-vs-tag choice is invisible.
"""

from typing import Tuple

from sisyphus import tk

from i6_core.g2p.apply import ApplyG2PModelJob
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
from i6_experiments.users.wu.experiments.posterior_hmm.data.phon_lm import (
    CollectOovWordsJob,
    TextToPhonemeJob,
    _train_g2p_model,
)
from i6_experiments.users.wu.experiments.unsupervised_asr.text import PREFIX, phon_lexicon


def lm_corpus_lexicon_and_g2p(g2p_concurrent: int = 16) -> Tuple[tk.Path, tk.Path, tk.Path]:
    """(LM text, bliss lexicon, G2P lexicon covering its OOVs) — the inputs any phonemization needs.

    Factored out of ``phonemize_lm_corpus`` so §1c's SIL-augmented variant reuses the *same* jobs
    (identical params -> identical hashes -> the already-computed G2P is reused, not recomputed).
    """
    text = get_librispeech_normalized_lm_data()
    lex = phon_lexicon()
    g2p_model = _train_g2p_model(prefix=PREFIX, bliss_lexicon=lex)

    oov = CollectOovWordsJob(text_files=[text], bliss_lexicon=lex)
    oov.add_alias(f"{PREFIX}/collect_lm_oov")
    g2p_oov = ApplyG2PModelJob(
        g2p_model=g2p_model, word_list_file=oov.out_word_list,
        filter_empty_words=True, concurrent=g2p_concurrent,
    ).out_g2p_lexicon
    return text, lex, g2p_oov


def phonemize_lm_corpus(g2p_concurrent: int = 16) -> tk.Path:
    """Wire 𝒯_φ over the LibriSpeech LM corpus. Returns the phonemized-text tk.Path."""
    text, lex, g2p_oov = lm_corpus_lexicon_and_g2p(g2p_concurrent=g2p_concurrent)

    phon = TextToPhonemeJob(text_file=text, bliss_lexicon=lex, g2p_lexicon=g2p_oov, gzip_output=True)
    phon.add_alias(f"{PREFIX}/phonemize_lm_corpus")
    tk.register_output(f"{PREFIX}/tphi.txt.gz", phon.out_text)
    tk.register_output(f"{PREFIX}/phonemize_unresolved.txt", phon.out_unresolved)
    return phon.out_text
