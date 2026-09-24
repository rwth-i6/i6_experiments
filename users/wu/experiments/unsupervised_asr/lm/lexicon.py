"""Ported from i6_experiments 5207c8adf users/wu/experiments/posterior_hmm/data/phon_lm.py
(``_load_lexicon_word_to_phon``, ``_load_g2p_lexicon``, ``CollectOovWordsJob``, ``_train_g2p_model``),
with the wiring of users/wu/experiments/unsupervised_asr/phonemize.py (``lm_corpus_lexicon_and_g2p``)
and users/wu/experiments/unsupervised_asr/text.py (``phon_lexicon``, ``PREFIX``).

The lexicon side of the text pipeline, all from public sources:

* the LibriSpeech LM text: ``i6_experiments.common`` ``get_librispeech_normalized_lm_data()``
  (openslr 11 ``librispeech-lm-norm.txt.gz``);
* the bliss lexicon: ``i6_experiments.common`` ``get_bliss_lexicon(use_stress_marker=False,
  add_silence=False)`` -- the stress-free 39-ARPAbet LibriSpeech lexicon (posterior_hmm's
  ``get_phon_lexicon(None, with_g2p=False)``);
* a Sequitur g2p model trained on it (i6_core ``BlissLexiconToG2PLexiconJob`` + ``TrainG2PModelJob``,
  i6_core defaults), applied to the LM corpus's OOV words (:class:`CollectOovWordsJob` +
  ``ApplyG2PModelJob(filter_empty_words=True, concurrent=16)``).

The g2p jobs read ``G2P_PATH`` / ``G2P_PYTHON`` from ``settings.py`` (i6_core's own convention).
"""

import itertools as it
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

from sisyphus import Job, Task, tk

from i6_core.g2p.apply import ApplyG2PModelJob
from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.g2p.train import TrainG2PModelJob
from i6_core.util import uopen

__all__ = [
    "PREFIX",
    "CollectOovWordsJob",
    "phon_lexicon",
    "lm_corpus_lexicon_and_g2p",
]

#: the alias prefix of the phonemization jobs (``unsupervised_asr/text.py``)
PREFIX = "sae/0b"


def _load_lexicon_word_to_phon(bliss_lexicon_path: str) -> Dict[str, str]:
    """
    Parse a Bliss lexicon XML and return word -> first-pronunciation-string mapping.
    Special lemmata are skipped.
    """
    word_to_phon: Dict[str, str] = {}
    with uopen(bliss_lexicon_path, "rt") as f:
        tree = ET.parse(f)
    for lemma in tree.findall(".//lemma"):
        if lemma.get("special") is not None:
            continue
        orth_el = lemma.find("orth")
        phon_el = lemma.find("phon")
        if orth_el is None or phon_el is None or orth_el.text is None or phon_el.text is None:
            continue
        word = orth_el.text.strip()
        if not word:
            continue
        if word in word_to_phon:
            continue
        word_to_phon[word] = phon_el.text.strip()
    return word_to_phon


def _load_g2p_lexicon(g2p_lexicon_path: str) -> Dict[str, str]:
    """
    Parse a Sequitur G2P output (`ApplyG2PModelJob.out_g2p_lexicon`) and return
    word -> first-pronunciation-string mapping.
    """
    word_to_phon: Dict[str, str] = {}
    with uopen(g2p_lexicon_path, "rt", encoding="utf-8") as f:
        for orth, data in it.groupby(
            (line.strip().split("\t") for line in f if line.strip()),
            lambda t: t[0],
        ):
            for entry in data:
                if len(entry) == 4 and orth not in word_to_phon:
                    word_to_phon[orth] = entry[3].strip()
                    break
    return word_to_phon


class CollectOovWordsJob(Job):
    """
    Walk a set of plain-text files and emit a sorted, unique list of words
    that are NOT covered by the given Bliss lexicon. The output is suitable
    as input to `ApplyG2PModelJob`.
    """

    def __init__(self, text_files: List[tk.Path], bliss_lexicon: tk.Path):
        super().__init__()
        self.text_files = list(text_files)
        self.bliss_lexicon = bliss_lexicon

        self.out_word_list = self.output_path("oov_words.txt")
        self.out_num_oov = self.output_var("num_oov")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 2})

    def run(self):
        word_to_phon = _load_lexicon_word_to_phon(self.bliss_lexicon.get_path())
        known = set(word_to_phon.keys())
        oov: set = set()
        for text_path in self.text_files:
            with uopen(text_path.get_path(), "rt") as f:
                for line in f:
                    for word in line.split():
                        if word not in known:
                            oov.add(word)
        with uopen(self.out_word_list.get_path(), "wt") as f:
            for word in sorted(oov):
                f.write(word + "\n")
        self.out_num_oov.set(len(oov))


def _train_g2p_model(prefix: str, bliss_lexicon: tk.Path) -> tk.Path:
    g2p_train_lex = BlissLexiconToG2PLexiconJob(
        bliss_lexicon=bliss_lexicon,
        include_pronunciation_variants=False,
        include_orthography_variants=False,
    ).out_g2p_lexicon
    g2p_train_job = TrainG2PModelJob(g2p_lexicon=g2p_train_lex)
    g2p_train_job.add_alias(os.path.join(prefix, "train_g2p"))
    return g2p_train_job.out_best_model


def phon_lexicon() -> tk.Path:
    """Stress-free plain-monophone (39 ARPAbet) LibriSpeech bliss lexicon, no silence lemma.

    The base LibriSpeech lexicon (no train-960 G2P augmentation); LM-corpus OOVs are covered by the
    per-corpus Sequitur G2P of :func:`lm_corpus_lexicon_and_g2p`.
    """
    from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon

    return get_bliss_lexicon(use_stress_marker=False, add_silence=False)


def lm_corpus_lexicon_and_g2p(g2p_concurrent: int = 16) -> Tuple[tk.Path, tk.Path, tk.Path]:
    """(LM text, bliss lexicon, G2P lexicon covering its OOVs) -- the inputs any phonemization needs."""
    from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data

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
