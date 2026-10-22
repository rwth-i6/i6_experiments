"""
WMT14 German-English Data Preparation & RETURNN HDF Dumping Jobs for UNMT.
Handles:
  1. Extracting and tokenizing WMT14 newstest2014 (and newstest2013 dev) for German-English.
  2. Preparing WMT14 News Crawl monolingual corpora (de, en) from $HPCWORK.
  3. Dumping tokenized text corpora into RETURNN HDFDataset files using the UNdreaMT vocabulary indexing
     (<pad>=0, <unk>=1, </s>=2, <s>=3, regular words=4..V+3).
"""

from typing import Dict, List, Optional, Tuple, Iterator
import os
import re
import gzip
import html
import logging
import numpy as np

from sisyphus import Job, Task, tk


HPCWORK_WMT14_DIR = "/hpcwork/bpd03090/wmt14_de_en"


def simple_moses_tokenize(text: str, lang: str = "en", tokenizer=None) -> str:
    """
    Tokenizes a line of text using sacremoses MosesTokenizer if available,
    otherwise falls back to a fast Moses-compatible punctuation tokenizer.
    """
    text = text.strip()
    if not text:
        return ""
    if tokenizer is not None:
        return tokenizer.tokenize(text, return_str=True, escape=False)
    # Fallback regex punctuation separation
    text = re.sub(r"([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~„“”«»–—])", r" \1 ", text)
    return " ".join(text.split())


def simple_moses_detokenize(tokens: List[str], lang: str = "en", detokenizer=None) -> str:
    """
    Detokenizes a list of tokens using sacremoses MosesDetokenizer if available,
    otherwise uses standard punctuation detokenization rules.
    """
    if not tokens:
        return ""
    if detokenizer is not None:
        return detokenizer.detokenize(tokens, return_str=True, unescape=False)
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)\]}»”])", r"\1", text)
    text = re.sub(r"([(\[{«„“])\s+", r"\1", text)
    return text.strip()


def extract_sgm_file(sgm_path: str) -> List[str]:
    """
    Extracts plain text segments from WMT SGM files (<seg id="...">...</seg>).
    """
    seg_regex = re.compile(r"<seg[^>]*>(.*?)</seg>", re.IGNORECASE | re.DOTALL)
    sentences = []
    with open(sgm_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = seg_regex.search(line)
            if m:
                sent = html.unescape(m.group(1)).strip()
                sentences.append(" ".join(sent.split()))
    return sentences


class PrepareWmt14TestSetJob(Job):
    """
    Extracts and tokenizes WMT14 newstest2014 (or dev newstest2013) for German and English.
    Outputs both raw detokenized reference files (for sacrebleu) and tokenized files (for model input).
    """
    _version = 1

    def __init__(
        self,
        src_sgm_or_txt: tk.Path,
        trg_sgm_or_txt: tk.Path,
        src_lang: str = "de",
        trg_lang: str = "en",
        lowercase: bool = False,
    ):
        super().__init__()
        self.src_sgm_or_txt = src_sgm_or_txt
        self.trg_sgm_or_txt = trg_sgm_or_txt
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.lowercase = lowercase

        self.out_src_raw = self.output_path(f"test.raw.{src_lang}")
        self.out_trg_raw = self.output_path(f"test.raw.{trg_lang}")
        self.out_src_tok = self.output_path(f"test.tok.{src_lang}")
        self.out_trg_tok = self.output_path(f"test.tok.{trg_lang}")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import sys
        extra_paths = [
            "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/container_libs",
        ]
        for p in extra_paths:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)

        try:
            from sacremoses import MosesTokenizer
            src_tokenizer = MosesTokenizer(lang=self.src_lang)
            trg_tokenizer = MosesTokenizer(lang=self.trg_lang)
        except Exception:
            src_tokenizer = None
            trg_tokenizer = None

        def load_lines(path: str) -> List[str]:
            if path.endswith(".sgm"):
                return extract_sgm_file(path)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return [line.strip() for line in f if line.strip()]

        src_lines = load_lines(self.src_sgm_or_txt.get_path())
        trg_lines = load_lines(self.trg_sgm_or_txt.get_path())
        assert len(src_lines) == len(trg_lines), f"Line count mismatch: {len(src_lines)} vs {len(trg_lines)}"

        with (
            open(self.out_src_raw.get_path(), "w", encoding="utf-8") as f_s_raw,
            open(self.out_trg_raw.get_path(), "w", encoding="utf-8") as f_t_raw,
            open(self.out_src_tok.get_path(), "w", encoding="utf-8") as f_s_tok,
            open(self.out_trg_tok.get_path(), "w", encoding="utf-8") as f_t_tok,
        ):
            for s, t in zip(src_lines, trg_lines):
                f_s_raw.write(s + "\n")
                f_t_raw.write(t + "\n")
                s_tok = simple_moses_tokenize(s, lang=self.src_lang, tokenizer=src_tokenizer)
                t_tok = simple_moses_tokenize(t, lang=self.trg_lang, tokenizer=trg_tokenizer)
                if self.lowercase:
                    s_tok = s_tok.lower()
                    t_tok = t_tok.lower()
                f_s_tok.write(s_tok + "\n")
                f_t_tok.write(t_tok + "\n")


class PrepareWmt14MonolingualNewsCrawlJob(Job):
    """
    Prepares WMT14 News Crawl monolingual data for a given language ('de' or 'en'):
      - Reads one or more .shuffled.gz / .txt files from $HPCWORK.
      - Tokenizes using MosesTokenizer.
      - Filters sentences to length in [min_tokens, max_tokens] (default [3, 50] as in UNdreaMT).
      - Optionally caps at max_sentences for fast iteration / training.
    """
    _version = 1

    def __init__(
        self,
        input_files: List[tk.Path],
        lang: str = "de",
        min_tokens: int = 3,
        max_tokens: int = 50,
        max_sentences: Optional[int] = 2000000,
        lowercase: bool = False,
    ):
        super().__init__()
        self.input_files = input_files
        self.lang = lang
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.lowercase = lowercase

        self.out_tokenized_txt = self.output_path(f"news_crawl.{lang}.tok.txt")

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

        try:
            from sacremoses import MosesTokenizer
            tokenizer = MosesTokenizer(lang=self.lang)
        except Exception:
            tokenizer = None

        count = 0
        with open(self.out_tokenized_txt.get_path(), "w", encoding="utf-8") as f_out:
            for path_obj in self.input_files:
                filepath = path_obj.get_path()
                open_fn = gzip.open if filepath.endswith(".gz") else open
                with open_fn(filepath, "rt", encoding="utf-8", errors="replace") as f_in:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        tok_line = simple_moses_tokenize(line, lang=self.lang, tokenizer=tokenizer)
                        if self.lowercase:
                            tok_line = tok_line.lower()
                        tokens = tok_line.split()
                        if self.min_tokens <= len(tokens) <= self.max_tokens:
                            f_out.write(" ".join(tokens) + "\n")
                            count += 1
                            if self.max_sentences is not None and count >= self.max_sentences:
                                break
                if self.max_sentences is not None and count >= self.max_sentences:
                    break

        logging.info(f"Prepared {count} tokenized sentences for {self.lang} -> {self.out_tokenized_txt.get_path()}")


class DumpTextToReturnnHdfJob(Job):
    """
    Converts a tokenized text file (1 sentence per line) into a RETURNN SimpleHDFWriter / h5py HDF dataset
    using the provided RETURNN vocabulary file ({token: id}).
    Follows UNdreaMT's indexing convention:
      PAD = 0, OOV (<unk>) = 1, EOS (</s>) = 2, SOS (<s>) = 3, regular words >= 4.
    Note: Does NOT append EOS into the stored sequence if the model/train_step appends EOS, OR can optionally
    append EOS. By default append_eos=False so the stored sequence is exact token IDs and the model/train_step
    handles SOS/EOS explicitly (matching existing RETURNN AED pipelines).
    """
    _version = 1

    def __init__(
        self,
        text_file: tk.Path,
        returnn_vocab_file: tk.Path,
        unk_id: int = 1,
        append_eos: bool = False,
        eos_id: int = 2,
        max_sentences: Optional[int] = None,
        seq_tag_prefix: str = "line",
    ):
        super().__init__()
        self.text_file = text_file
        self.returnn_vocab_file = returnn_vocab_file
        self.unk_id = unk_id
        self.append_eos = append_eos
        self.eos_id = eos_id
        self.max_sentences = max_sentences
        self.seq_tag_prefix = seq_tag_prefix

        self.out_hdf = self.output_path("data.hdf")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "time": 2})

    def run(self):
        import ast
        from returnn.datasets.hdf import SimpleHDFWriter

        with open(self.returnn_vocab_file.get_path(), "r", encoding="utf-8") as f:
            vocab_dict: Dict[str, int] = ast.literal_eval(f.read().strip())

        num_classes = max(vocab_dict.values()) + 1 if vocab_dict else 4
        writer = SimpleHDFWriter(filename=self.out_hdf.get_path(), dim=num_classes, ndim=1)

        open_fn = gzip.open if self.text_file.get_path().endswith(".gz") else open
        count = 0
        with open_fn(self.text_file.get_path(), "rt", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                tokens = line.strip().split()
                if not tokens:
                    continue
                ids = [vocab_dict.get(tok, self.unk_id) for tok in tokens]
                if self.append_eos:
                    ids.append(self.eos_id)
                arr = np.array(ids, dtype="int32")
                seq_tag = f"{self.seq_tag_prefix}-{idx}"
                writer.insert_batch(
                    inputs=arr[None, :],
                    seq_len=[len(ids)],
                    seq_tag=[seq_tag],
                )
                count += 1
                if self.max_sentences is not None and count >= self.max_sentences:
                    break

        writer.close()
        logging.info(f"Dumped {count} sequences to {self.out_hdf.get_path()} (vocab dim={num_classes})")
