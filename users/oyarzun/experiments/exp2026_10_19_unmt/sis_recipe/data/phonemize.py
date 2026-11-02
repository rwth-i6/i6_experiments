"""
Phonemization and N-gram (Diphone / Triphone) Generation Jobs for UNMT.
Provides:
  - PhonemizeEnglishTextJob: Converts English text to space-separated ARPAbet phonemes
    using CMUDict with g2p_en fallback, preserving exact 1-to-1 line count.
  - GenerateTextNgramsJob: Converts space-separated phoneme sequences into sliding n-grams
    (diphones for n=2, triphones for n=3) with vocabulary and count extraction.
"""

from typing import Dict, Iterator, List, Optional
import os
import re
import gzip
import logging
from sisyphus import Job, Task, tk
from ..cluster_adapter import get_nltk_data_paths, setup_extra_python_paths


ENGLISH_ARPABET_PHONEMES: List[str] = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]

GERMAN_IPA_PHONEMES: List[str] = [
    "a", "aː", "aɪ", "aʊ", "b", "d", "dʒ", "eː", "f", "h",
    "iː", "j", "k", "l", "m", "n", "oː", "p", "pf", "r",
    "s", "t", "ts", "tʃ", "uː", "v", "x", "yː", "z", "øː",
    "ŋ", "œ", "ɔ", "ɔø", "ə", "ɛ", "ɛː", "ɜ", "ɡ", "ɪ",
    "ʃ", "ʊ", "ʊɐ", "ʏ", "ʒ", "ç",
]

_GERMAN_IPA_SET = set(GERMAN_IPA_PHONEMES)

_GERMAN_ALLOPHONE_MAP: Dict[str, List[str]] = {
    "ɑ": ["a"],
    "ɑː": ["aː"],
    "ɾ": ["r"],
    "ʀ": ["r"],
    "ʁ": ["r"],
    "ɹ": ["r"],
    "ɐ": ["ɜ"],
    "ɜː": ["ɜ"],
    "g": ["ɡ"],
    "i": ["iː"],
    "u": ["uː"],
    "y": ["yː"],
    "e": ["eː"],
    "o": ["oː"],
    "ø": ["øː"],
    "ɔɪ": ["ɔø"],
    "ɔʏ": ["ɔø"],
    "eɪ": ["eː"],
    "ɛɪ": ["eː"],
    "əʊ": ["oː"],
    "oʊ": ["oː"],
    "eə": ["ɛː"],
    "ɪə": ["iː", "ɜ"],
    "aɪə": ["aɪ", "ə"],
    "aʊə": ["aʊ", "ə"],
    "n̩": ["ə", "n"],
    "m̩": ["ə", "m"],
    "l̩": ["ə", "l"],
    "əl": ["ə", "l"],
    "ɒ": ["ɔ"],
    "æ": ["ɛ"],
    "ʌ": ["a"],
    "w": ["v"],
    "θ": ["s"],
    "ð": ["z"],
    "ɑ̃": ["a", "ŋ"],
    "ɛ̃": ["ɛ", "ŋ"],
    "ɔ̃": ["ɔ", "ŋ"],
    "œ̃": ["œ", "ŋ"],
}


def normalize_german_ipa_tokens(raw_tokens: List[str]) -> List[str]:
    """
    Normalizes raw espeak German IPA tokens into the canonical 46-symbol GERMAN_IPA_PHONEMES set.
    """
    out: List[str] = []
    for tok in raw_tokens:
        t = re.sub(r"[ˈˌː\u200d()]+", lambda m: "ː" if "ː" in m.group(0) else "", tok).strip()
        if not t:
            continue
        if t in _GERMAN_IPA_SET:
            out.append(t)
        elif t in _GERMAN_ALLOPHONE_MAP:
            out.extend(_GERMAN_ALLOPHONE_MAP[t])
        else:
            # Greedy longest-match fallback over canonical German IPA symbols + allophone map
            idx = 0
            while idx < len(t):
                matched = False
                for length in (3, 2, 1):
                    sub = t[idx : idx + length]
                    if sub in _GERMAN_IPA_SET:
                        out.append(sub)
                        idx += length
                        matched = True
                        break
                    if sub in _GERMAN_ALLOPHONE_MAP:
                        out.extend(_GERMAN_ALLOPHONE_MAP[sub])
                        idx += length
                        matched = True
                        break
                if not matched:
                    idx += 1
    return out


class PhonemizeEnglishTextJob(Job):
    """
    Phonemizes an English tokenized text file line-by-line using CMUDict + g2p_en fallback.
    Strips stress digits (e.g. AH0 -> AH, EY1 -> EY).
    Preserves exact line count so test sets (e.g. newstest2014) match the source line-for-line.
    """
    _version = 1

    def __init__(
        self,
        token_text_file: Optional[tk.Path] = None,
        text_file: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.token_text_file = token_text_file or text_file
        assert self.token_text_file is not None, "Must provide text_file or token_text_file"
        self.out_phoneme_text = self.output_path("text.phonemes.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 4})

    def run(self):
        setup_extra_python_paths()

        import nltk
        for nltk_data_path in get_nltk_data_paths():
            if os.path.exists(nltk_data_path) and nltk_data_path not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data_path)

        from nltk.corpus import cmudict
        cmu = cmudict.dict()
        from g2p_en import G2p
        g2p = G2p()

        cache: Dict[str, List[str]] = {}

        def word_to_phones(word: str) -> List[str]:
            if word in cache:
                return cache[word]
            w_clean = re.sub(r"[^a-zA-Z0-9\']", "", word).lower()
            if not w_clean:
                cache[word] = []
                return []
            if w_clean in cmu:
                phones = [re.sub(r"\d+", "", p) for p in cmu[w_clean][0]]
            else:
                try:
                    res = g2p(w_clean)
                    phones = [re.sub(r"\d+", "", p) for p in res if p.strip() and p.isalnum()]
                except Exception as e:
                    logging.warning(f"g2p failed for word {word!r}: {e}")
                    phones = []
            cache[word] = phones
            return phones

        in_path = self.token_text_file.get_path()
        open_fn = gzip.open if in_path.endswith(".gz") else open

        count = 0
        out_path = self.out_phoneme_text.get_path()
        tmp_out = out_path + ".tmp"
        with open_fn(in_path, "rt", encoding="utf-8", errors="replace") as f_in, open(
            tmp_out, "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                words = line.strip().split()
                line_phones: List[str] = []
                for w in words:
                    line_phones.extend(word_to_phones(w))
                f_out.write(" ".join(line_phones) + "\n")
                count += 1
                if count % 100000 == 0:
                    logging.info(f"Phonemized {count} sentences (cached {len(cache)} words)...")

        os.replace(tmp_out, out_path)
        logging.info(f"Finished phonemizing {count} sentences to {out_path}")


class GenerateTextNgramsJob(Job):
    """
    Takes a space-separated token file (e.g. phoneme sequences) and generates n-gram sequences
    (diphones for n=2, triphones for n=3).
    Exports:
      - out_ngram_text: sequence of space-separated n-gram tokens per utterance
      - out_vocab_txt: vocabulary file listing unique n-grams sorted by frequency descending
      - out_counts_txt: frequency counts for each n-gram
    """
    _version = 1

    def __init__(
        self,
        token_text_file: Optional[tk.Path] = None,
        phoneme_text_file: Optional[tk.Path] = None,
        text_file: Optional[tk.Path] = None,
        ngram_order: int = 2,
        delimiter: str = "_",
        min_count: int = 1,
        max_vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.token_text_file = token_text_file or phoneme_text_file or text_file
        assert self.token_text_file is not None, "Must provide token_text_file, phoneme_text_file, or text_file"
        self.ngram_order = ngram_order
        self.delimiter = delimiter
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size

        self.out_ngram_text = self.output_path(f"text.ngrams_{ngram_order}.txt")
        self.out_vocab_txt = self.output_path("vocab.txt")
        self.out_counts_txt = self.output_path("counts.txt")
        self.out_vocab_list = self.out_vocab_txt
        self.out_vocab_counts = self.out_counts_txt

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 2})

    def run(self):
        from collections import Counter
        counts = Counter()
        n = self.ngram_order
        delim = self.delimiter

        in_path = self.token_text_file.get_path()
        open_fn = gzip.open if in_path.endswith(".gz") else open

        count = 0
        with open_fn(in_path, "rt", encoding="utf-8", errors="replace") as f_in, open(
            self.out_ngram_text.get_path(), "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                tokens = line.strip().split()
                if not tokens:
                    f_out.write("\n")
                    continue
                if len(tokens) < n:
                    ngrams = [delim.join(tokens)]
                else:
                    ngrams = [delim.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

                counts.update(ngrams)
                f_out.write(" ".join(ngrams) + "\n")
                count += 1

        with open(self.out_vocab_txt.get_path(), "w", encoding="utf-8") as f_vocab, open(
            self.out_counts_txt.get_path(), "w", encoding="utf-8"
        ) as f_counts:
            num_written = 0
            for w, c in counts.most_common():
                if c < self.min_count:
                    break
                f_vocab.write(f"{w}\n")
                f_counts.write(f"{w} {c}\n")
                num_written += 1
                if self.max_vocab_size is not None and num_written >= self.max_vocab_size:
                    break

        logging.info(f"Generated {self.ngram_order}-grams for {count} lines; vocab size: {num_written}")


class PhonemizeGermanTextJob(Job):
    """
    Phonemizes a German tokenized text file line-by-line into space-separated German-specific
    IPA phonemes (46 canonical German IPA symbols in GERMAN_IPA_PHONEMES) using espeak + phonemizer.
    Builds a unique-word cache for fast corpus-scale processing while preserving exact 1-to-1 line count.
    """
    _version = 1

    def __init__(
        self,
        token_text_file: Optional[tk.Path] = None,
        text_file: Optional[tk.Path] = None,
        njobs: int = 4,
    ):
        super().__init__()
        self.token_text_file = token_text_file or text_file
        assert self.token_text_file is not None, "Must provide text_file or token_text_file"
        self.njobs = njobs
        self.out_phoneme_text = self.output_path("text.de.phonemes.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": max(1, self.njobs), "mem": 16, "time": 4})

    def run(self):
        import shutil
        import subprocess
        import tempfile
        from ..cluster_adapter import UNMTClusterPaths

        setup_extra_python_paths()

        in_path = self.token_text_file.get_path()
        open_fn = gzip.open if in_path.endswith(".gz") else open

        def clean_word(w: str) -> str:
            return re.sub(r"[^\w]", "", w, flags=re.UNICODE).replace("_", "").lower()

        unique_words_set = set()
        total_lines = 0
        with open_fn(in_path, "rt", encoding="utf-8", errors="replace") as f_in:
            for line in f_in:
                total_lines += 1
                for w in line.strip().split():
                    cw = clean_word(w)
                    if cw:
                        unique_words_set.add(cw)

        unique_words = sorted(unique_words_set)
        logging.info(f"Collected {len(unique_words)} unique German words across {total_lines} lines.")

        cache: Dict[str, List[str]] = {}
        has_local_espeak = (
            shutil.which("espeak") is not None
            or shutil.which("espeak-ng") is not None
            or os.path.exists("/usr/lib/x86_64-linux-gnu/libespeak.so.1")
            or os.path.exists("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")
        )

        if unique_words:
            if has_local_espeak:
                from phonemizer import phonemize
                from phonemizer.separator import Separator

                sep = Separator(phone=" ", word="|")
                chunk_size = 20000
                for start in range(0, len(unique_words), chunk_size):
                    chunk = unique_words[start : start + chunk_size]
                    phn_chunk = phonemize(
                        chunk,
                        language="de",
                        backend="espeak",
                        separator=sep,
                        strip=True,
                        with_stress=False,
                        language_switch="remove-flags",
                        njobs=self.njobs,
                    )
                    for w_str, p_str in zip(chunk, phn_chunk):
                        raw_toks = p_str.replace("|", " ").split()
                        cache[w_str] = normalize_german_ipa_tokens(raw_toks)
                    logging.info(f"Phonemized {min(start + chunk_size, len(unique_words))}/{len(unique_words)} unique words...")
            elif os.path.exists(UNMTClusterPaths.CURIUM_ESPEAK_APPTAINER):
                with tempfile.TemporaryDirectory() as tmpdir:
                    words_in = os.path.join(tmpdir, "words_in.txt")
                    words_out = os.path.join(tmpdir, "words_out.txt")
                    with open(words_in, "w", encoding="utf-8") as fw:
                        for w_str in unique_words:
                            fw.write(w_str + "\n")
                    helper_code = (
                        "import sys\n"
                        f"sys.path.append({UNMTClusterPaths.CURIUM_CONTAINER_LIBS!r})\n"
                        "from phonemizer import phonemize\n"
                        "from phonemizer.separator import Separator\n"
                        "with open(sys.argv[1], 'r', encoding='utf-8') as f:\n"
                        "    words = [l.strip() for l in f]\n"
                        "sep = Separator(phone=' ', word='|')\n"
                        "out = []\n"
                        "for i in range(0, len(words), 20000):\n"
                        "    out.extend(phonemize(words[i:i+20000], language='de', backend='espeak', "
                        f"separator=sep, strip=True, with_stress=False, language_switch='remove-flags', njobs={self.njobs}))\n"
                        "with open(sys.argv[2], 'w', encoding='utf-8') as f:\n"
                        "    for r in out:\n"
                        "        f.write(r.replace('|', ' ') + '\\n')\n"
                    )
                    subprocess.check_call(
                        [
                            "apptainer",
                            "exec",
                            "-B",
                            tmpdir,
                            "-B",
                            UNMTClusterPaths.CURIUM_SMT4_USER_DIR,
                            UNMTClusterPaths.CURIUM_ESPEAK_APPTAINER,
                            "python3",
                            "-c",
                            helper_code,
                            words_in,
                            words_out,
                        ]
                    )
                    with open(words_out, "r", encoding="utf-8") as fr:
                        for w_str, p_line in zip(unique_words, fr):
                            cache[w_str] = normalize_german_ipa_tokens(p_line.strip().split())
            else:
                raise RuntimeError("espeak / espeak-ng binary not found on host or via apptainer.")

        count = 0
        out_path = self.out_phoneme_text.get_path()
        tmp_out = out_path + ".tmp"
        with open_fn(in_path, "rt", encoding="utf-8", errors="replace") as f_in, open(
            tmp_out, "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                line_phones: List[str] = []
                for w in line.strip().split():
                    cw = clean_word(w)
                    if cw and cw in cache:
                        line_phones.extend(cache[cw])
                f_out.write(" ".join(line_phones) + "\n")
                count += 1

        os.replace(tmp_out, out_path)
        logging.info(f"Finished phonemizing {count} German sentences to {out_path}")


class WriteFixedVocabJob(Job):
    """
    Writes a deterministic newline-separated vocabulary list to vocab.txt for TrainWord2VecJob(fixed_vocab_file=...).
    """
    _version = 1

    def __init__(self, symbols: List[str]):
        super().__init__()
        self.symbols = list(symbols)
        self.out_vocab_txt = self.output_path("fixed_vocab.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.out_vocab_txt.get_path(), "w", encoding="utf-8") as f:
            for sym in self.symbols:
                f.write(f"{sym}\n")

