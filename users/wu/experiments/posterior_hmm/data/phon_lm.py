"""
Phoneme LM dataset pipeline for posterior HMM.

Mirrors data/bpe_lm.py but tokenizes the LibriSpeech LM corpus + LS-960 transcripts
into EOW phoneme sequences using the pHMM phoneme lexicon plus a Sequitur G2P
fallback for OOVs. The LM vocab equals the pHMM phoneme inventory extended with
`<s>` / `</s>` at the two indices > AM vocab size. `LmDataset` itself prepends
`<s>` and appends `</s>` at runtime, so the produced text files contain pure
whitespace-separated phoneme tokens with one sentence per line.
"""

import itertools as it
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedFormat

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.g2p.apply import ApplyG2PModelJob
from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.g2p.train import TrainG2PModelJob
from i6_core.returnn.config import CodeWrapper
from i6_core.text.processing import ConcatenateJob
from i6_core.util import uopen

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.datasets.librispeech.language_model import (
    get_librispeech_normalized_lm_data,
)
from i6_experiments.common.setups.returnn.datasets import ControlDataset, Dataset
from i6_experiments.common.setups.returnn.datastreams.base import Datastream

from .bpe_lm import TrainingDatasets
from .phon import get_phmm_eow_lexicon, get_phmm_eow_lm_vocab_datastream


SOURCE_DATASTREAM_KEY = "data"
TARGET_DATASTREAN_KEY = "delayed"


class LmDataset(ControlDataset):
    """
    RETURNN `LmDataset` wrapper for a phoneme-tokenized text file. Mirrors the
    BPE variant in data/bpe_lm.py but with a pickled phoneme vocab.
    """

    def __init__(
        self,
        *,
        corpus_file: tk.Path,
        vocab_file: tk.Path,
        partition_epoch: Optional[int] = None,
        segment_file: Optional[tk.Path] = None,
        seq_ordering: Optional[str] = None,
        random_subset: Optional[int] = None,
        additional_options: Optional[Dict] = None,
    ):
        super().__init__(
            partition_epoch=partition_epoch,
            segment_file=segment_file,
            seq_ordering=seq_ordering,
            random_subset=random_subset,
            additional_options=additional_options,
        )
        self.corpus_file = corpus_file
        self.vocab_file = vocab_file

    def as_returnn_opts(self) -> Dict[str, Any]:
        d = {
            "class": "LmDataset",
            "corpus_file": CodeWrapper(DelayedFormat('lambda: cf("{}")', self.corpus_file)),
            "orth_symbols_map_file": self.vocab_file,
            "orth_replace_map_file": "",
            "word_based": True,
            "seq_end_symbol": "</s>",
            "auto_replace_unknown_symbol": False,
            "unknown_symbol": None,
            "add_delayed_seq_data": True,
            "delayed_seq_data_start_symbol": "<s>",
        }
        sd = super().as_returnn_opts()
        assert all(k not in sd for k in d), (
            "conflicting keys in %s and %s" % (list(sd.keys()), list(d.keys()))
        )
        d.update(sd)
        return d


@dataclass()
class LMDatasetSettings:
    train_partition_epoch: int
    train_seq_ordering: str


def _load_lexicon_word_to_phon(bliss_lexicon_path: str) -> Dict[str, str]:
    """
    Parse a Bliss lexicon XML and return word → first-pronunciation-string mapping.
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
    word → first-pronunciation-string mapping.
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


class TextToPhonemeJob(Job):
    """
    Tokenize plain text into whitespace-separated phoneme sequences using a
    Bliss lexicon (first pronunciation per word) plus an optional G2P fallback
    for OOVs. Lines that still contain unresolvable words are dropped.
    """

    def __init__(
        self,
        text_file: tk.Path,
        bliss_lexicon: tk.Path,
        g2p_lexicon: Optional[tk.Path] = None,
        gzip_output: bool = True,
    ):
        super().__init__()
        self.text_file = text_file
        self.bliss_lexicon = bliss_lexicon
        self.g2p_lexicon = g2p_lexicon
        self.gzip_output = gzip_output

        out_name = "phon.txt.gz" if gzip_output else "phon.txt"
        self.out_text = self.output_path(out_name)
        self.out_unresolved = self.output_path("unresolved_words.txt")
        self.out_num_sentences_in = self.output_var("num_sentences_in")
        self.out_num_sentences_out = self.output_var("num_sentences_out")
        self.out_num_unresolved = self.output_var("num_unresolved")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        word_to_phon = _load_lexicon_word_to_phon(self.bliss_lexicon.get_path())
        if self.g2p_lexicon is not None:
            g2p_map = _load_g2p_lexicon(self.g2p_lexicon.get_path())
            for word, phon in g2p_map.items():
                word_to_phon.setdefault(word, phon)

        unresolved_words: set = set()
        n_in = 0
        n_out = 0
        with uopen(self.text_file.get_path(), "rt") as inf, uopen(self.out_text.get_path(), "wt") as outf:
            for line in inf:
                n_in += 1
                words = line.split()
                if not words:
                    continue
                phons: List[str] = []
                ok = True
                for word in words:
                    phon = word_to_phon.get(word)
                    if phon is None:
                        unresolved_words.add(word)
                        ok = False
                        break
                    phons.append(phon)
                if not ok:
                    continue
                outf.write(" ".join(phons) + "\n")
                n_out += 1

        with uopen(self.out_unresolved.get_path(), "wt") as f:
            for word in sorted(unresolved_words):
                f.write(word + "\n")
        self.out_num_sentences_in.set(n_in)
        self.out_num_sentences_out.set(n_out)
        self.out_num_unresolved.set(len(unresolved_words))


def _train_g2p_model(prefix: str, bliss_lexicon: tk.Path) -> tk.Path:
    g2p_train_lex = BlissLexiconToG2PLexiconJob(
        bliss_lexicon=bliss_lexicon,
        include_pronunciation_variants=False,
        include_orthography_variants=False,
    ).out_g2p_lexicon
    g2p_train_job = TrainG2PModelJob(g2p_lexicon=g2p_train_lex)
    g2p_train_job.add_alias(os.path.join(prefix, "train_g2p"))
    return g2p_train_job.out_best_model


def build_phon_lm_training_datasets(
    prefix: str,
    librispeech_key: str,
    settings: LMDatasetSettings,
) -> TrainingDatasets:
    """
    Build phoneme-tokenized LM training data using the same text corpora as the
    BPE variant: official LibriSpeech LM normalized text concatenated with
    LS-960 transcripts for training; dev-clean + dev-other for CV.
    """
    label_datastream = get_phmm_eow_lm_vocab_datastream(prefix=prefix, g2p_librispeech_key=librispeech_key)
    bliss_lexicon = get_phmm_eow_lexicon(g2p_librispeech_key=librispeech_key)

    # text sources
    bliss_corpus_dict = get_bliss_corpus_dict()
    lm_data_text = get_librispeech_normalized_lm_data()
    ls_train_text = CorpusToTxtJob(
        bliss_corpus=bliss_corpus_dict[librispeech_key],
        gzip=True,
    ).out_txt
    full_train_text = ConcatenateJob(
        text_files=[lm_data_text, ls_train_text],
        zip_out=True,
    ).out

    dev_clean_text = CorpusToTxtJob(bliss_corpus=bliss_corpus_dict["dev-clean"], gzip=True).out_txt
    dev_other_text = CorpusToTxtJob(bliss_corpus=bliss_corpus_dict["dev-other"], gzip=True).out_txt
    cv_text = ConcatenateJob(text_files=[dev_clean_text, dev_other_text], zip_out=True).out

    # G2P for OOVs across train+cv
    g2p_model = _train_g2p_model(prefix=prefix, bliss_lexicon=bliss_lexicon)
    oov_collect_job = CollectOovWordsJob(
        text_files=[full_train_text, cv_text],
        bliss_lexicon=bliss_lexicon,
    )
    oov_collect_job.add_alias(os.path.join(prefix, "collect_lm_oov_words"))
    g2p_apply_job = ApplyG2PModelJob(
        g2p_model=g2p_model,
        word_list_file=oov_collect_job.out_word_list,
        filter_empty_words=True,
        concurrent=8,
    )
    g2p_apply_job.add_alias(os.path.join(prefix, "apply_g2p_to_lm_oov"))
    g2p_oov_lexicon = g2p_apply_job.out_g2p_lexicon

    train_phon_job = TextToPhonemeJob(
        text_file=full_train_text,
        bliss_lexicon=bliss_lexicon,
        g2p_lexicon=g2p_oov_lexicon,
        gzip_output=True,
    )
    train_phon_job.add_alias(os.path.join(prefix, "phonemize_lm_train"))
    cv_phon_job = TextToPhonemeJob(
        text_file=cv_text,
        bliss_lexicon=bliss_lexicon,
        g2p_lexicon=g2p_oov_lexicon,
        gzip_output=True,
    )
    cv_phon_job.add_alias(os.path.join(prefix, "phonemize_lm_cv"))

    lm_train_dataset = LmDataset(
        corpus_file=train_phon_job.out_text,
        vocab_file=label_datastream.vocab,
        partition_epoch=settings.train_partition_epoch,
        segment_file=None,
        seq_ordering=settings.train_seq_ordering,
    )
    lm_cv_dataset = LmDataset(
        corpus_file=cv_phon_job.out_text,
        vocab_file=label_datastream.vocab,
        partition_epoch=1,
        segment_file=None,
        seq_ordering="sorted",
    )
    lm_devtrain_dataset = LmDataset(
        corpus_file=train_phon_job.out_text,
        vocab_file=label_datastream.vocab,
        partition_epoch=1,
        segment_file=None,
        seq_ordering="sorted",
        random_subset=3000,
    )

    return TrainingDatasets(
        train=lm_train_dataset,
        cv=lm_cv_dataset,
        devtrain=lm_cv_dataset,
        datastreams={
            SOURCE_DATASTREAM_KEY: label_datastream,
            TARGET_DATASTREAN_KEY: label_datastream,
        },
    )
