from typing import Any, Dict, Iterator, Optional, Union
import copy
import os
import shutil
import subprocess as sp
import ast

import numpy as np

from returnn.datasets.hdf import SimpleHDFWriter

from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedBase

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
        min_phoneme_occurrence: int,
        phonemizer_engine: str = "G2P",
        python_env: Optional[tk.Path] = None,
        seq_tag_file: Optional[tk.Path] = None,
    ):
        self.text_file = text_file
        self.fairseq_root = fairseq_root
        self.python_exe = python_exe
        self.lid_path = lid_path
        self.lexicon_file = lexicon_file
        self.language = language
        self.sil_prob = sil_prob
        self.seq_tag_file = seq_tag_file
        self.python_env = python_env
        self.min_phoneme_occurrence = min_phoneme_occurrence
        self.phonemizer_engine = phonemizer_engine

        self.out_lexicon_file = self.output_path("lexicon_filtered.lst")
        self.out_phoneme_text = self.output_path("text.phonemes.txt")
        self.out_phoneme_counts = self.output_path("phoneme_counts.txt")
        self.out_phoneme_vocab = self.output_path("phoneme_vocab.txt")
        if seq_tag_file is not None:
            self.out_seq_tags = self.output_path("seq-tags-after-phonemize.txt")
        else:
            self.out_seq_tags = None

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 8, "time": 2})
        if self.lexicon_file is None:
            yield Task("create_phoneme_vocab", mini_task=True)

    @staticmethod
    def normalize_and_filter_text(
        text_file: str,
        out_text_file: str,
        lang: str,
        lid_threshold: float,
        fasttext_model: str,
        seq_tags_file: Optional[str],
    ):
        import regex
        import fasttext as ft
        import sys

        filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")

        lg = lang.lower()
        lg_label = f"__label__{lg}"
        thresh = lid_threshold
        if seq_tags_file is not None:
            with open(seq_tags_file, "r", encoding="utf-8") as f:
                seq_tags = list(line.strip() for line in f)
        else:
            seq_tags = None

        if os.path.exists(fasttext_model):
            model = ft.load_model(fasttext_model)
        else:
            print(
                f"fasttext language id model {fasttext_model} not found. Proceeding without language filtering. "
                f"To enable language filtering, please download the latest language id model "
                f"from https://fasttext.cc/docs/en/language-identification.html",
                file=sys.stderr,
            )
            model = None

        new_seq_tag_file = open("seq-tags-after-norm-and-filter.txt", "w")
        text_file = open(text_file, "r", encoding="utf-8")
        out_text_file = open(out_text_file, "w", encoding="utf-8")
        lines = text_file.readlines()
        if seq_tags is not None:
            assert len(lines) == len(seq_tags), "Number of lines in text file and seq tag file must be the same"

        for i, line in enumerate(lines):
            line = line.strip()
            line = filter_r.sub(" ", line)
            line = " ".join(line.split())

            if model is not None:
                lid, prob = model.predict(line, k=100)
                try:
                    target_idx = lid.index(lg_label)
                except ValueError:
                    continue
                if target_idx == 0 or prob[target_idx] >= thresh:
                    out_text_file.write(f"{line}\n")
                    if seq_tags is not None:
                        new_seq_tag_file.write(f"{seq_tags[i]}\n")
            else:
                out_text_file.write(f"{line}\n")
                if seq_tags is not None:
                    new_seq_tag_file.write(f"{seq_tags[i]}\n")

        new_seq_tag_file.close()
        text_file.close()
        out_text_file.close()

    @staticmethod
    def phonemize_with_sil(
        text_file: str,
        out_text_file: str,
        sil_prob: float,
        surround: bool,
        seq_tags_file: str,
        lexicon: str,
    ):
        import sys
        import numpy as np

        sil = "<SIL>"

        if seq_tags_file is not None:
            with open(seq_tags_file, "r", encoding="utf-8") as f:
                seq_tags = list(line.strip() for line in f)
        else:
            seq_tags = None

        wrd_to_phn = {}

        with open(lexicon, "r") as lf:
            for line in lf:
                items = line.rstrip().split()
                assert len(items) > 1, line
                assert items[0] not in wrd_to_phn, items
                wrd_to_phn[items[0]] = items[1:]

        new_seq_tag_file = open("seq-tags-after-phonemize.txt", "w", encoding="utf-8")
        out_text_file = open(out_text_file, "w", encoding="utf-8")
        text_file = open(text_file, "r", encoding="utf-8")
        lines = text_file.readlines()

        if seq_tags is not None:
            assert len(lines) == len(seq_tags), "Number of lines in text file and seq tag file must be the same"

        for i, line in enumerate(lines):
            words = line.strip().split()

            if not all(w in wrd_to_phn for w in words):
                continue

            phones = []
            if surround:
                phones.append(sil)

            sample_sil_probs = None
            if sil_prob > 0 and len(words) > 1:
                sample_sil_probs = np.random.random(len(words) - 1)

            for j, w in enumerate(words):
                phones.extend(wrd_to_phn[w])
                if sample_sil_probs is not None and j < len(sample_sil_probs) and sample_sil_probs[j] < sil_prob:
                    phones.append(sil)

            if surround:
                phones.append(sil)

            out_text_file.write(" ".join(phones) + "\n")

            if seq_tags is not None:
                new_seq_tag_file.write(f"{seq_tags[i]}\n")

        new_seq_tag_file.close()
        out_text_file.close()
        text_file.close()

    def run(self):
        import sys

        text_dir = os.path.join(os.path.dirname(self.out_phoneme_text.get_path()), "text")
        seq_tag_file = None
        if self.lexicon_file is None:
            env = os.environ.copy()

            env["PYTHONPATH"] = f"{self.fairseq_root.get_path()}:" + env.get("PYTHONPATH", "")
            env["FAIRSEQ_ROOT"] = self.fairseq_root.get_path()
            env["PATH"] = f"{os.path.dirname(sys.executable)}" + os.pathsep + env["PATH"]

            script_path = (
                "/work/asr4/schmitt/sisyphus_work_dirs/2026_04_09_unsupervised_asr/process_text/phonemize_text.sh"
            )
            sh_call = [
                "zsh",
                script_path,
                self.language,
                self.text_file.get_path(),
                text_dir,
                str(self.min_phoneme_occurrence),
                self.phonemizer_engine,
                self.lid_path.get_path(),
                str(self.sil_prob),
            ]
            # env["VIRTUAL_ENV"] = self.fairseq_python_env.get_path()
            sp.run(sh_call, env=env, check=True)
            lexicon_file = os.path.join(text_dir, "lexicon_filtered.lst")
            shutil.copy(lexicon_file, self.out_lexicon_file.get_path())
            shutil.move(os.path.join(text_dir, "lm.upper.lid.txt"), "lm.upper.lid.txt")
            shutil.move(os.path.join(text_dir, "phones/dict.txt"), self.out_phoneme_counts.get_path())
        else:
            shutil.copy(self.lexicon_file, self.out_lexicon_file.get_path())

            self.normalize_and_filter_text(
                text_file=self.text_file.get_path(),
                out_text_file="lm.upper.lid.txt",
                lang=self.language,
                lid_threshold=0.4,
                fasttext_model=self.lid_path.get_path(),
                seq_tags_file=self.seq_tag_file.get_path() if self.seq_tag_file is not None else None,
            )
            seq_tag_file = (
                os.path.join(os.getcwd(), "seq-tags-after-norm-and-filter.txt")
                if self.seq_tag_file is not None
                else None
            )

            preprocess_cmd = (
                f"{sys.executable} {self.fairseq_root.get_path()}/fairseq_cli/preprocess.py "
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

        self.phonemize_with_sil(
            text_file="lm.upper.lid.txt",
            out_text_file="lm.phones.filtered.txt",
            sil_prob=self.sil_prob,
            surround=True,
            seq_tags_file=seq_tag_file,
            lexicon=self.out_lexicon_file.get_path(),
        )

        shutil.move("lm.phones.filtered.txt", self.out_phoneme_text.get_path())
        if self.seq_tag_file is not None:
            shutil.move("seq-tags-after-phonemize.txt", self.out_seq_tags.get_path())

    def create_phoneme_vocab(self):
        with open(self.out_phoneme_counts.get(), "r") as f:
            vocab = {line.strip().split()[0]: i for i, line in enumerate(f.readlines())}
            vocab["<SIL>"] = len(vocab)
        with open(self.out_phoneme_vocab.get_path(), "w") as f:
            f.write("{\n")
            for phon, i in vocab.items():
                f.write(f'"{phon}": {i},\n')
            f.write("}\n")


class DumpPhonemeIndicesToHdfJob(Job):
    def __init__(
        self,
        text_file: Union[DelayedBase, tk.Path],
        phoneme_vocab: Union[DelayedBase, tk.Path],
        concurrent: int = 10,
        fixed_random_subset: Optional[int] = None,
        seq_tag_file: Optional[tk.Path] = None,
    ):
        """

        Args:
            text_file: each line contains a sequence of phonemes separated by space
            phoneme_vocab:
            concurrent: number of concurrent hdf files to dump
            fixed_random_subset: if given, only use a fixed random subset of the data
        """
        self.text_file = text_file
        self.phoneme_vocab = phoneme_vocab
        self.concurrent = concurrent
        self.fixed_random_subset = fixed_random_subset
        self.seq_tag_file = seq_tag_file

        self.out_hdfs = {i: self.output_path(f"data_{i}.hdf") for i in range(self.concurrent)}

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 4}, args=range(1, self.concurrent + 1))

    def run(self, task_id):
        import gc
        import tempfile
        import random

        with open(self.text_file.get(), "r") as f:
            lines = f.readlines()

        with open(self.phoneme_vocab.get(), "r") as f:
            vocab = ast.literal_eval(f.read())

        if self.seq_tag_file is not None:
            with open(self.seq_tag_file.get(), "r") as f:
                seq_tags = [line.strip() for line in f.readlines()]
                assert len(seq_tags) == len(lines)
        else:
            seq_tags = None
        pairs = list(zip(lines, seq_tags)) if seq_tags is not None else [(line, None) for line in lines]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_hdf = os.path.join(tmp_dir, f"data_{task_id}.hdf")
            hdf_writer = SimpleHDFWriter(filename=tmp_hdf, dim=len(vocab), ndim=1)

            random.Random(42).shuffle(pairs)

            num_lines = len(pairs)
            pairs = [pairs[i] for i in range(num_lines) if (i % self.concurrent) == (task_id - 1)]
            if self.fixed_random_subset is not None:
                pairs = pairs[: self.fixed_random_subset]
            num_lines = len(lines)
            gc.collect()

            for i, (line, seq_tag) in enumerate(pairs):
                phonemes = line.strip().split()
                data = [vocab[p] for p in phonemes]
                data = np.array([data])  # (1, T)

                seq_len = len(phonemes)
                seq_lens = {0: np.array([seq_len])}
                batch_seq_sizes = np.expand_dims(seq_lens[0], 1)

                hdf_writer.insert_batch(
                    data,
                    seq_len=seq_lens,
                    seq_tag=[f"lm-data-{i}" if seq_tag is None else seq_tag],
                    extra={"seq_sizes": batch_seq_sizes},
                )

                if i % 10_000 == 0:
                    gc.collect()
                    print(f"Processed sequence {i}/{num_lines} ({i / num_lines * 100:.1f}%)")

            hdf_writer.close()
            shutil.move(tmp_hdf, self.out_hdfs[task_id - 1].get_path())
