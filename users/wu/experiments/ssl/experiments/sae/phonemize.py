"""SAE Phase 0b — phonemize the LibriSpeech LM corpus to stress-free ARPAbet (𝒯_φ).

Pipeline: CorpusVocabJob (word types + OOV-vs-lexicon) -> ApplyG2PModelJob on the OOV list ->
PhonemizeCorpusJob (folded-lexicon first-pron, G2P for OOV) -> boundary-free (and optional <wb>)
phoneme stream. 𝒯_φ is the AR-input / decipherment-LM / §0b-CPT text; one deterministic pronunciation
per word (SAE_PLAN §0b).

Pure helpers (``load_bliss_prons``, ``parse_g2p_lexicon``, ``phonemize_line``) are numpy/sisyphus-free
so the mapping logic is unit-testable offline.
"""

from __future__ import annotations

import gzip
import os
from collections import Counter
from typing import Dict, List, Optional

from sisyphus import Job, Task, tk

from i6_core.g2p.apply import ApplyG2PModelJob

PREFIX = "sae/0b"
OOV_PHON = "[UNKNOWN]"
WB = "<wb>"


def _open(path, mode="rt"):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def load_bliss_prons(bliss_lexicon_path: str) -> Dict[str, str]:
    """{orth(UPPER) -> first pronunciation string} from a bliss lexicon, skipping special lemmas."""
    import xml.etree.ElementTree as ET

    with _open(bliss_lexicon_path, "rt") as f:
        tree = ET.parse(f)
    prons: Dict[str, str] = {}
    for lemma in tree.findall(".//lemma"):
        if lemma.get("special") is not None:
            continue
        phon = lemma.find("phon")
        if phon is None or not phon.text:
            continue
        for orth in lemma.findall("orth"):
            if orth.text is None:
                continue
            prons.setdefault(orth.text.strip().upper(), phon.text.strip())
    return prons


def parse_g2p_lexicon(g2p_lexicon_path: str) -> Dict[str, str]:
    """{orth(UPPER) -> phones} from ApplyG2PModelJob output (tab-separated: orth, idx, score, phones)."""
    prons: Dict[str, str] = {}
    with _open(g2p_lexicon_path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3].strip():
                prons.setdefault(parts[0].strip().upper(), parts[3].strip())
    return prons


def phonemize_line(line: str, prons: Dict[str, str], insert_wb: bool = False) -> Optional[List[str]]:
    """Map one whitespace-tokenized line to a flat phone list; OOV words -> OOV_PHON. None if empty."""
    words = line.split()
    if not words:
        return None
    out: List[str] = []
    for i, w in enumerate(words):
        if insert_wb and i > 0:
            out.append(WB)
        p = prons.get(w.upper())
        out.extend(p.split() if p is not None else [OOV_PHON])
    return out


class CorpusVocabJob(Job):
    """Word types + counts of a (gzipped) text corpus, split into in-lexicon vs OOV against a bliss lexicon."""

    def __init__(self, text_file: tk.Path, bliss_lexicon: tk.Path):
        self.text_file = text_file
        self.bliss_lexicon = bliss_lexicon
        self.out_word_counts = self.output_path("word_counts.txt.gz")
        self.out_oov_words = self.output_path("oov_words.txt")
        self.out_stats = self.output_path("vocab_stats.txt")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        counts: Counter = Counter()
        n_tokens = 0
        with _open(self.text_file.get_path(), "rt") as f:
            for line in f:
                w = line.split()
                n_tokens += len(w)
                counts.update(x.upper() for x in w)
        lex = set(load_bliss_prons(self.bliss_lexicon.get_path()).keys())
        oov = [w for w in counts if w not in lex]
        oov_tokens = sum(counts[w] for w in oov)
        with _open(self.out_word_counts.get_path(), "wt") as f:
            for w, c in counts.most_common():
                f.write(f"{w}\t{c}\n")
        with open(self.out_oov_words.get_path(), "wt") as f:
            for w in sorted(oov):
                f.write(w + "\n")
        with open(self.out_stats.get_path(), "wt") as f:
            f.write(f"tokens\t{n_tokens}\n")
            f.write(f"types\t{len(counts)}\n")
            f.write(f"oov_types\t{len(oov)}\n")
            f.write(f"oov_tokens\t{oov_tokens}\n")
            f.write(f"oov_type_rate\t{len(oov)/max(len(counts),1):.6f}\n")
            f.write(f"oov_token_rate\t{oov_tokens/max(n_tokens,1):.6f}\n")


class PhonemizeCorpusJob(Job):
    """Phonemize a (gzipped) text corpus with the folded lexicon (first pron) + G2P for OOV words."""

    def __init__(self, text_file: tk.Path, bliss_lexicon: tk.Path, g2p_oov_lexicon: tk.Path, insert_wb: bool = False):
        self.text_file = text_file
        self.bliss_lexicon = bliss_lexicon
        self.g2p_oov_lexicon = g2p_oov_lexicon
        self.insert_wb = insert_wb
        self.out_phonemes = self.output_path("phonemes.txt.gz")
        self.out_stats = self.output_path("phonemize_stats.txt")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        prons = load_bliss_prons(self.bliss_lexicon.get_path())
        for w, p in parse_g2p_lexicon(self.g2p_oov_lexicon.get_path()).items():
            prons.setdefault(w, p)  # lexicon takes precedence over G2P
        n_lines = n_words = n_oov = 0
        with _open(self.text_file.get_path(), "rt") as fin, _open(self.out_phonemes.get_path(), "wt") as fout:
            for line in fin:
                for w in line.split():
                    n_words += 1
                    if w.upper() not in prons:
                        n_oov += 1
                toks = phonemize_line(line, prons, insert_wb=self.insert_wb)
                if toks is None:
                    continue
                n_lines += 1
                fout.write(" ".join(toks) + "\n")
        with open(self.out_stats.get_path(), "wt") as f:
            f.write(f"lines\t{n_lines}\n")
            f.write(f"words\t{n_words}\n")
            f.write(f"residual_oov_words\t{n_oov}\n")
            f.write(f"residual_oov_rate\t{n_oov/max(n_words,1):.6f}\n")


def phonemize_lm_corpus(insert_wb: bool = False, g2p_concurrent: int = 16):
    """Wire the full 𝒯_φ pipeline. Returns the phonemized-corpus tk.Path."""
    from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
    from i6_experiments.users.wu.experiments.ssl.experiments.sae.text import folded_lexicon, librispeech_g2p_model

    text = get_librispeech_normalized_lm_data()
    lex = folded_lexicon()
    vocab = CorpusVocabJob(text, lex)
    vocab.add_alias(f"{PREFIX}/corpus_vocab")
    tk.register_output(f"{PREFIX}/vocab_stats.txt", vocab.out_stats)

    g2p_oov = ApplyG2PModelJob(
        librispeech_g2p_model(), vocab.out_oov_words, variants_number=1, concurrent=g2p_concurrent
    ).out_g2p_lexicon

    phon = PhonemizeCorpusJob(text, lex, g2p_oov, insert_wb=insert_wb)
    phon.add_alias(f"{PREFIX}/phonemize_corpus{'_wb' if insert_wb else ''}")
    tk.register_output(f"{PREFIX}/tphi{'_wb' if insert_wb else ''}.txt.gz", phon.out_phonemes)
    tk.register_output(f"{PREFIX}/phonemize_stats{'_wb' if insert_wb else ''}.txt", phon.out_stats)
    return phon.out_phonemes
