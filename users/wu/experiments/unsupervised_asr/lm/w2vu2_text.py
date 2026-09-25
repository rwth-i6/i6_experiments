"""Ported from i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/w2vu2/text.py
(``FairseqPreprocessTextJob``), w2vu2/pipeline.py (``_text_data``, the phone LM path) and
users/wu/experiments/posterior_hmm/data/phon_lm.py (``TextToPhonemeJob``, the text of the sae/1a
phone LM ``output/sae/1a/phoneme_lm_o4.bin``).

The text side of the wav2vec-U 2.0 GAN (section 1c):

* the SIL-augmented phone text: ``phone_text.get_phone_corpus()`` (``PhonemizeWithSilJob``,
  p_sil 0.5, surround, seed 0, all lines), reused as it is;
* :class:`FairseqTextDataJob` -- that text as fairseq's ``text_data`` dir (``dict.txt``,
  ``train.bin``, ``train.idx``), the output of the source's ``fairseq-preprocess --dataset-impl mmap
  --only-source --thresholdsrc 1000 --padding-factor 1``, written in-process with numpy;
* :class:`TextToPhonemeJob` -- the SIL-free phone text of the phone LM (one line per LM-corpus line,
  first pronunciation per word from the bliss lexicon, g2p for the rest, OOV lines dropped);
* :func:`get_w2vu2_phone_lm` -- the phone 4-gram on that text, i6_core's ``KenLMplzJob`` +
  ``CreateBinaryLMJob`` with the sae/1a arguments (``KenLMplzJob.0aJeN88X6EdW``: order 4,
  ``interpolate_unigrams=True``, no pruning, no vocabulary, ``discount_fallback=[0.5, 1.0, 1.5]``,
  ``mem=16``, ``time=2``; ``CreateBinaryLMJob.hvZoC014xnIe``: defaults).

``FairseqTextDataJob`` reproduces fairseq 0.12.2 (``fairseq_cli/preprocess.py`` with
``Dictionary.add_file_to_dictionary`` / ``finalize`` and ``MMapIndexedDatasetBuilder``):

* the dictionary: the four specials ``<s> <pad> </s> <unk>`` (count 1 each) at 0..3; every
  whitespace token counted, plus one ``</s>`` per line; the non-special symbols ordered by count,
  descending, ties alphabetical; symbols below ``threshold`` dropped; ``padding_factor`` 1 adds no
  ``madeupword``.  ``dict.txt`` holds ``"<symbol> <count>"`` for the non-specials in that order;
* ``train.bin``: per line the token ids (``<unk>`` = 3 for a dropped or unknown symbol) and ``</s>``
  = 2, as uint16 (the dtype fairseq picks for a vocabulary under 65,500; int32 above), little endian,
  concatenated;
* ``train.idx``: ``b"MMIDIDX\\x00\\x00"``, ``<Q`` version 1, ``<B`` dtype code (uint16 = 8, int32 = 4),
  ``<Q`` number of lines, then the int32 line sizes and the int64 byte offsets of the lines.

Only the train split exists: the GAN's valid split is label-free (``has_unpaired_text`` is per split),
which keeps the checkpoint selection unsupervised.  ``preprocess.log``, which fairseq-preprocess also
writes, is not written: the GAN never reads it.
"""

from __future__ import annotations

import gzip
import os
import struct
import sys
from array import array
from collections import Counter
from typing import Dict, List, Tuple

from sisyphus import Job, Task, tk

__all__ = [
    "FAIRSEQ_SPECIALS",
    "MMAP_INDEX_MAGIC",
    "fairseq_dictionary",
    "fairseq_best_dtype",
    "write_fairseq_text_data",
    "FairseqTextDataJob",
    "TextToPhonemeJob",
    "PHONE_LM_ORDER",
    "PHONE_LM_DISCOUNT_FALLBACK",
    "PHONE_LM_MEM_GB",
    "get_w2vu2_text_data",
    "get_w2vu2_phone_lm_text",
    "get_w2vu2_phone_lm",
]

#: fairseq ``Dictionary()``'s specials, in index order (bos, pad, eos, unk)
FAIRSEQ_SPECIALS = ("<s>", "<pad>", "</s>", "<unk>")
_EOS = 2
_UNK = 3
#: ``fairseq.data.indexed_dataset.MMapIndexedDataset.Index._HDR_MAGIC``
MMAP_INDEX_MAGIC = b"MMIDIDX\x00\x00"
#: ``indexed_dataset._code_to_dtype`` inverse, for the two dtypes ``best_fitting_int_dtype`` picks
_DTYPE_CODE = {"uint16": 8, "int32": 4}

#: the sae/1a phone LM (``KenLMplzJob.0aJeN88X6EdW``)
PHONE_LM_ORDER = 4
PHONE_LM_DISCOUNT_FALLBACK = (0.5, 1.0, 1.5)
PHONE_LM_MEM_GB = 16


def _open_text(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def fairseq_dictionary(counts: Counter, n_lines: int, threshold: int, padding_factor: int = 1
                       ) -> Tuple[List[str], List[int]]:
    """``(symbols, counts)`` of the finalized fairseq Dictionary, specials included.

    :param counts: the token counts of the text (``</s>`` excluded; :func:`_count` adds one per line).
    :param n_lines: the number of lines (the ``</s>`` count each line adds).
    """
    symbols = list(FAIRSEQ_SPECIALS)
    sym_counts = [1, 1, 1 + int(n_lines), 1]
    rest = {w: int(c) for w, c in counts.items() if w not in FAIRSEQ_SPECIALS}
    for w in FAIRSEQ_SPECIALS:
        if w in counts:  # a special spelled out in the text is counted onto the special
            sym_counts[FAIRSEQ_SPECIALS.index(w)] += int(counts[w])
    # Dictionary.finalize: Counter(dict(sorted(zip(symbols, counts)))).most_common(): count
    # descending; most_common's sort is stable, so ties keep the alphabetical order
    for w, c in sorted(sorted(rest.items()), key=lambda kv: kv[1], reverse=True):
        if c < threshold:
            break
        symbols.append(w)
        sym_counts.append(c)
    if padding_factor > 1:
        i = 0
        while len(symbols) % padding_factor != 0:
            symbols.append("madeupword{:04d}".format(i))
            sym_counts.append(0)
            i += 1
    return symbols, sym_counts


def fairseq_best_dtype(vocab_size: int) -> str:
    """``indexed_dataset.best_fitting_int_dtype``: uint16 below 65,500 symbols, else int32."""
    return "uint16" if vocab_size < 65500 else "int32"


def write_fairseq_text_data(path: str, *, dict_path: str, bin_path: str, idx_path: str,
                            threshold: int, padding_factor: int = 1) -> Dict[str, int]:
    """The body of :class:`FairseqTextDataJob` on plain paths; returns the counts it printed."""
    import numpy as np

    counts: Counter = Counter()
    n_lines = 0
    with _open_text(path) as fh:
        for line in fh:
            counts.update(line.split())
            n_lines += 1
    symbols, sym_counts = fairseq_dictionary(counts, n_lines, threshold, padding_factor)
    with open(dict_path, "wt") as fh:
        for w, c in zip(symbols[len(FAIRSEQ_SPECIALS):], sym_counts[len(FAIRSEQ_SPECIALS):]):
            fh.write(f"{w} {c}\n")

    dtype = fairseq_best_dtype(len(symbols))
    index: Dict[str, int] = {w: _UNK for w in counts}  # every token of the text: dropped -> <unk>
    index.update({w: i for i, w in enumerate(symbols)})
    lookup = index.__getitem__
    assert sys.byteorder == "little", "fairseq's mmap dataset is little endian"
    typecode = {"uint16": "H", "int32": "i"}[dtype]
    assert array(typecode).itemsize == np.dtype(dtype).itemsize
    sizes = array("i")
    n_tokens = 0
    chunk = array(typecode)
    with _open_text(path) as fh, open(bin_path, "wb") as out:
        for line in fh:
            toks = line.split()
            chunk.extend(map(lookup, toks))
            chunk.append(_EOS)
            sizes.append(len(toks) + 1)
            if len(chunk) >= (1 << 24):
                chunk.tofile(out)
                n_tokens += len(chunk)
                chunk = array(typecode)
        chunk.tofile(out)
        n_tokens += len(chunk)
    assert len(sizes) == n_lines, (len(sizes), n_lines)
    n_unk = sum(c for w, c in counts.items() if index[w] == _UNK)

    sizes_arr = np.frombuffer(sizes, dtype="<i4")
    pointers = np.zeros(len(sizes_arr), dtype="<i8")
    if len(sizes_arr) > 1:
        np.cumsum(sizes_arr[:-1].astype("<i8") * np.dtype(dtype).itemsize, out=pointers[1:])
    with open(idx_path, "wb") as fh:
        fh.write(MMAP_INDEX_MAGIC)
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<B", _DTYPE_CODE[dtype]))
        fh.write(struct.pack("<Q", len(sizes_arr)))
        fh.write(sizes_arr.tobytes(order="C"))
        fh.write(pointers.tobytes(order="C"))
    expected_bin = int(sizes_arr.sum(dtype=np.int64)) * np.dtype(dtype).itemsize
    assert os.path.getsize(bin_path) == expected_bin, "train.bin size != sum(sizes)"
    print(f"text_data: {n_lines} lines, {n_tokens} tokens (incl. </s>), {len(symbols)} symbols "
          f"({len(symbols) - len(FAIRSEQ_SPECIALS)} in dict.txt), {n_unk} <unk>, dtype {dtype}", flush=True)
    return {"lines": n_lines, "tokens": n_tokens, "symbols": len(symbols), "unk": n_unk}


class FairseqTextDataJob(Job):
    """Phone text -> fairseq ``text_data`` (``dict.txt``, ``train.bin``, ``train.idx``), in-process.

    The output of the source's ``fairseq-preprocess --dataset-impl mmap --trainpref <text>
    --only-source --thresholdsrc <threshold> --padding-factor 1`` (module docstring).  Two streaming
    passes over the (gzipped) text: count, then encode.
    """

    def __init__(self, *, text_file: tk.Path, threshold: int = 1000, padding_factor: int = 1):
        super().__init__()
        self.text_file = text_file
        self.threshold = int(threshold)
        self.padding_factor = int(padding_factor)

        self.out_dir = self.output_path("text_data", directory=True)
        # ``unpaired_audio_text`` loads exactly ``<text_data>/dict.txt`` -- do not rename it
        self.out_dict = self.output_path("text_data/dict.txt")
        self.out_bin = self.output_path("text_data/train.bin")
        self.out_idx = self.output_path("text_data/train.idx")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 6}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        write_fairseq_text_data(
            self.text_file.get_path(),
            dict_path=self.out_dict.get_path(),
            bin_path=self.out_bin.get_path(),
            idx_path=self.out_idx.get_path(),
            threshold=self.threshold,
            padding_factor=self.padding_factor,
        )


class TextToPhonemeJob(Job):
    """Plain text -> whitespace-separated phone lines, the sae/1a phone LM's text (SIL-free).

    First pronunciation per word from the bliss lexicon, the g2p lexicon for the rest; a line with a
    word neither resolves is dropped, and so is an empty line (``posterior_hmm/data/phon_lm.py``).
    """

    def __init__(self, text_file: tk.Path, bliss_lexicon: tk.Path, g2p_lexicon: tk.Path = None,
                 gzip_output: bool = True):
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
        from i6_core.util import uopen

        from .lexicon import _load_g2p_lexicon, _load_lexicon_word_to_phon

        word_to_phon = _load_lexicon_word_to_phon(self.bliss_lexicon.get_path())
        if self.g2p_lexicon is not None:
            for word, phon in _load_g2p_lexicon(self.g2p_lexicon.get_path()).items():
                word_to_phon.setdefault(word, phon)

        unresolved_words: set = set()
        n_in = n_out = 0
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


# ===================================================================================================
# the wiring
# ===================================================================================================
def get_w2vu2_text_data() -> FairseqTextDataJob:
    """The GAN's ``task.text_data``: :class:`FairseqTextDataJob` (threshold 1000, padding factor 1) on
    the p_sil 0.5 phone corpus (``FairseqPreprocessTextJob.bi2B89fES77z``'s arguments; its ``workers``
    only parallelised fairseq-preprocess)."""
    from .phone_text import get_phone_corpus

    job = FairseqTextDataJob(text_file=get_phone_corpus(), threshold=1000, padding_factor=1)
    job.add_alias("sae/1c/text_data_sil0.5")
    return job


def get_w2vu2_phone_lm_text() -> tk.Path:
    """The phone LM's text: :class:`TextToPhonemeJob` on the LM corpus, bliss lexicon and g2p lexicon
    of :func:`~.lexicon.lm_corpus_lexicon_and_g2p` (the inputs of ``get_phone_corpus``)."""
    from .lexicon import lm_corpus_lexicon_and_g2p

    text, lex, g2p = lm_corpus_lexicon_and_g2p()
    job = TextToPhonemeJob(text_file=text, bliss_lexicon=lex, g2p_lexicon=g2p, gzip_output=True)
    job.add_alias("sae/0b/phonemize_lm_corpus")
    return job.out_text


def get_w2vu2_phone_lm() -> tk.Path:
    """The GAN's ``task.kenlm_path``: the phone 4-gram's KenLM binary (module docstring)."""
    from i6_core.lm.kenlm import CreateBinaryLMJob, KenLMplzJob

    from ..default_tools import get_kenlm_binary_path

    kenlm = get_kenlm_binary_path()
    lmplz = KenLMplzJob(
        text=[get_w2vu2_phone_lm_text()],
        order=PHONE_LM_ORDER,
        interpolate_unigrams=True,
        pruning=None,
        vocabulary=None,
        discount_fallback=list(PHONE_LM_DISCOUNT_FALLBACK),
        kenlm_binary_folder=kenlm,
        mem=PHONE_LM_MEM_GB,
        time=2,
    )
    lmplz.add_alias("sae/1a/kenlm_plz_o4")
    binary = CreateBinaryLMJob(arpa_lm=lmplz.out_lm, kenlm_binary_folder=kenlm)
    binary.add_alias("sae/1a/phoneme_lm_o4")
    return binary.out_lm
