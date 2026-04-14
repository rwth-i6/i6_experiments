from typing import Any, Dict, Iterator, Optional
import copy
import os
import shutil
import subprocess as sp

from sisyphus import Job, Task, tk

import i6_experiments


class PhonemizeTextDataJob(Job):
    def __init__(
        self,
        text_file: tk.Path,
        fairseq_root: tk.Path,
        python_exe: tk.Path,
        language: str,
        sil_prob: float,
        lid_path: tk.Path,
        lexicon_file: Optional[tk.Path],
        phoneme_file: Optional[tk.Path],
        seq_tag_file: Optional[tk.Path] = None,
    ):
        self.text_file = text_file
        self.fairseq_root = fairseq_root
        self.python_exe = python_exe
        self.lid_path = lid_path
        self.lexicon_file = lexicon_file
        self.phoneme_file = phoneme_file
        self.language = language
        self.sil_prob = sil_prob
        self.seq_tag_file = seq_tag_file

        self.out_phoneme_text = self.output_path("text.phonemes.txt")
        if seq_tag_file is not None:
            self.out_seq_tags = self.output_path("seq-tags-after-phonemize.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 8, "time": 2})

    def run(self):
        i6_experiments_mod = i6_experiments.__path__[0]
        normalize_cmd = (
            f"{self.python_exe.get_path()} {i6_experiments_mod}/users/schmitt/experiments/exp2025_10_02_shared_enc/librispeech/data/normalize_and_filter_text.py "
            f"{'--seq-tags-file ' + self.seq_tag_file.get_path() if self.seq_tag_file is not None else ''} "
            f"--lang {self.language} --fasttext-model {self.lid_path.get_path()} < {self.text_file.get_path()} | grep -v '\-\-\-' > lm.upper.lid.txt"
        )
        sp.check_call(normalize_cmd, shell=True)
        # written by normalize_and_filter_text.py
        seq_tag_file = (
            os.path.join(os.getcwd(), "seq-tags-after-norm-and-filter.txt") if self.seq_tag_file is not None else None
        )

        preprocess_cmd = (
            f"python {self.fairseq_root.get_path()}/fairseq_cli/preprocess.py "
            f"--dataset-impl mmap "
            f"--trainpref lm.upper.lid.txt "
            f"--only-source "
            f"--destdir . "
            f"--thresholdsrc 2 "
            f"--padding-factor 1 "
            f"--dict-only"
        )
        sp.check_call(preprocess_cmd, shell=True)

        cut_cmd = f"cut -f1 -d' ' dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' > words.txt"
        sp.check_call(cut_cmd, shell=True)

        phonemize_cmd = (
            f"python {i6_experiments_mod}/users/schmitt/experiments/exp2025_10_02_shared_enc/librispeech/data/phonemize_with_sil.py "
            f"-s {self.sil_prob} "
            f"--surround "
            f"{'--seq-tags-file ' + seq_tag_file if seq_tag_file is not None else ''} "
            f"--lexicon {self.lexicon_file.get()} < lm.upper.lid.txt > lm.phones.filtered.txt"
        )
        sp.check_call(phonemize_cmd, shell=True)

        shutil.move("lm.phones.filtered.txt", self.out_phoneme_text.get_path())
        if self.seq_tag_file is not None:
            shutil.move("seq-tags-after-phonemize.txt", self.out_seq_tags.get_path())
