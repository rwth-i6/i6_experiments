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
        import sys
        extra_paths = [
            "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/container_libs",
        ]
        for p in extra_paths:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)

        import nltk
        nltk_data_path = "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/container_libs/nltk_data"
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
        with open_fn(in_path, "rt", encoding="utf-8", errors="replace") as f_in, open(
            self.out_phoneme_text.get_path(), "w", encoding="utf-8"
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

        logging.info(f"Finished phonemizing {count} sentences to {self.out_phoneme_text.get_path()}")


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
