"""
Byte Pair Encoding (BPE) Jobs for Unsupervised Neural Machine Translation (UNMT).
Uses subword-nmt (Sennrich et al., ACL 2016) to:
  1. Learn BPE merge operations (TrainBpeJob) independently or jointly on monolingual corpora.
  2. Apply BPE segmentation (ApplyBpeJob) line-by-line preserving exact 1-to-1 line counts.
  3. Reverse BPE segmentation (DeBpeTextJob) by stripping '@@ ' for word-level BLEU evaluation.
"""

from typing import Dict, Iterator, List, Optional, Sequence, Union
import gzip
import logging
import os
import re

from sisyphus import Job, Task, tk


class TrainBpeJob(Job):
    """
    Learns BPE merge operations from one or more tokenized text files using subword-nmt.
    Exports out_bpe_codes.
    """
    _version = 1

    def __init__(
        self,
        input_files: Union[tk.Path, Sequence[tk.Path]],
        num_operations: int = 50000,
        min_frequency: int = 2,
        num_workers: int = 8,
    ):
        super().__init__()
        self.input_files = [input_files] if isinstance(input_files, tk.Path) else list(input_files)
        self.num_operations = num_operations
        self.min_frequency = min_frequency
        self.num_workers = num_workers

        self.out_bpe_codes = self.output_path("bpe.codes")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": self.num_workers, "mem": 32, "time": 6})

    def run(self):
        logging.basicConfig(level=logging.INFO)
        import sys
        import tempfile

        extra_paths = [
            "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/container_libs",
        ]
        for p in extra_paths:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)

        from subword_nmt import learn_bpe

        with tempfile.TemporaryDirectory() as tmp:
            if len(self.input_files) == 1:
                input_path = self.input_files[0].get_path()
            else:
                input_path = os.path.join(tmp, "combined_corpus.txt")
                logging.info(f"Concatenating {len(self.input_files)} corpora into {input_path}...")
                with open(input_path, "wb") as f_out:
                    for p in self.input_files:
                        in_p = p.get_path()
                        open_fn = gzip.open if in_p.endswith(".gz") else open
                        with open_fn(in_p, "rb") as f_in:
                            while True:
                                chunk = f_in.read(1024 * 1024)
                                if not chunk:
                                    break
                                f_out.write(chunk)

            logging.info(
                f"Learning BPE codes ({self.num_operations} operations, min_freq={self.min_frequency}, "
                f"workers={self.num_workers}) from {input_path}..."
            )
            open_fn = gzip.open if input_path.endswith(".gz") else open
            with open_fn(input_path, "rt", encoding="utf-8", errors="replace") as infile, \
                 open(self.out_bpe_codes.get_path(), "w", encoding="utf-8") as outfile:
                learn_bpe.learn_bpe(
                    infile=infile,
                    outfile=outfile,
                    num_symbols=self.num_operations,
                    min_frequency=self.min_frequency,
                    num_workers=self.num_workers,
                )

        logging.info(f"Successfully generated BPE codes -> {self.out_bpe_codes.get_path()}")


class ApplyBpeJob(Job):
    """
    Applies BPE codes to a tokenized text file using subword-nmt.
    Preserves exact 1-to-1 line mapping.
    """
    _version = 1

    def __init__(
        self,
        text_file: tk.Path,
        bpe_codes: tk.Path,
        separator: str = "@@",
    ):
        super().__init__()
        self.text_file = text_file
        self.bpe_codes = bpe_codes
        self.separator = separator

        self.out_bpe_text = self.output_path("text.bpe.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 2})

    def run(self):
        logging.basicConfig(level=logging.INFO)
        import sys

        extra_paths = [
            "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/container_libs",
        ]
        for p in extra_paths:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)

        from subword_nmt import apply_bpe

        with open(self.bpe_codes.get_path(), "r", encoding="utf-8") as cf:
            bpe = apply_bpe.BPE(cf, separator=self.separator)

        in_path = self.text_file.get_path()
        open_fn = gzip.open if in_path.endswith(".gz") else open

        count = 0
        with open_fn(in_path, "rt", encoding="utf-8", errors="replace") as f_in, \
             open(self.out_bpe_text.get_path(), "w", encoding="utf-8") as f_out:
            for line in f_in:
                segmented = bpe.process_line(line.strip())
                f_out.write(segmented + "\n")
                count += 1

        logging.info(f"Applied BPE to {count} sentences -> {self.out_bpe_text.get_path()}")


class DeBpeTextJob(Job):
    """
    Strips subword BPE merge separators ('@@ ') from a text file or RETURNN search result dict,
    recovering word-level text for standard MT evaluation.
    """
    _version = 1

    def __init__(
        self,
        input_file: tk.Path,
        separator: str = "@@",
    ):
        super().__init__()
        self.input_file = input_file
        self.separator = separator

        self.out_debpe_text = self.output_path("text.debpe.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        logging.basicConfig(level=logging.INFO)
        in_path = self.input_file.get_path()
        open_fn = gzip.open if in_path.endswith(".gz") else open
        sep_pattern = self.separator + " "

        count = 0
        with open_fn(in_path, "rt", encoding="utf-8", errors="replace") as f_in, \
             open(self.out_debpe_text.get_path(), "w", encoding="utf-8") as f_out:
            content = f_in.read().strip()
            # If python dict format from RETURNN search_out
            if content.startswith("{") and content.endswith("}"):
                import ast
                try:
                    d = ast.literal_eval(content)
                    if isinstance(d, dict):
                        def _tag_sort_key(item):
                            k = str(item[0])
                            parts = k.replace("/", "-").split("-")
                            for p in reversed(parts):
                                if p.isdigit():
                                    return (0, int(p), k)
                            return (1, 0, k)
                        for k, v in sorted(d.items(), key=_tag_sort_key):
                            v_str = (
                                v[0][1]
                                if (isinstance(v, list) and len(v) > 0 and isinstance(v[0], tuple))
                                else (str(v[0]) if isinstance(v, list) else str(v))
                            )
                            cleaned = v_str.replace(sep_pattern, "").replace(self.separator, "")
                            f_out.write(cleaned + "\n")
                            count += 1
                        logging.info(f"De-BPE'd {count} dictionary entries -> {self.out_debpe_text.get_path()}")
                        return
                except Exception:
                    pass

            for line in content.splitlines():
                cleaned = line.replace(sep_pattern, "").replace(self.separator, "")
                f_out.write(cleaned + "\n")
                count += 1

        logging.info(f"De-BPE'd {count} lines -> {self.out_debpe_text.get_path()}")
