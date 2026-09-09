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
        surround_w_sil: bool = True,
        apply_lid_filter: bool = True,
        collapse_repeats: bool = False,
    ):
        """
        :param apply_lid_filter: run the fasttext language-ID filter over the input lines. This makes sense for
            a raw (web) LM corpus, but for a corpus that is known to be in ``language`` already (e.g. LibriSpeech
            transcriptions) it only drops lines: on dev-other it removes 5 of 2864 utterances. Set to False for
            eval sets, where losing sequences makes the scores incomparable. NB the text normalization itself
            (the character filter) is always applied.
        :param collapse_repeats: merge adjacent identical phonemes of the final phoneme sequence into one
            (``S AH M M IH S`` -> ``S AH M IH S``, mostly across word boundaries; 0.56% of the tokens on the
            LibriSpeech transcripts). Mirrors what a segmentation derived from a frame alignment does: a run
            of identical labels is one segment, so a phoneme repeated across a word boundary is a single
            segment there. Applied after the silence insertion, so inserted ``<SIL>`` tokens are collapsed too.
        """
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
        self.surround_w_sil = surround_w_sil
        self.apply_lid_filter = apply_lid_filter
        self.collapse_repeats = collapse_repeats

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
        apply_lid_filter: bool = True,
    ):
        import regex
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

        if not apply_lid_filter:
            print("Language-ID filtering disabled, keeping all lines.", file=sys.stderr)
            model = None
        elif os.path.exists(fasttext_model):
            # imported lazily so that apply_lid_filter=False does not need fasttext at all
            import fasttext as ft

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
        collapse_repeats: bool = False,
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

            if collapse_repeats:
                phones = [p for i, p in enumerate(phones) if i == 0 or p != phones[i - 1]]

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
                apply_lid_filter=self.apply_lid_filter,
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
            surround=self.surround_w_sil,
            seq_tags_file=seq_tag_file,
            lexicon=self.out_lexicon_file.get_path(),
            collapse_repeats=self.collapse_repeats,
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

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        if parsed_args["surround_w_sil"]:
            del parsed_args["surround_w_sil"]
        # only hash the new option when it deviates from the old (only) behavior, to keep existing job hashes
        if parsed_args.get("apply_lid_filter", True):
            parsed_args.pop("apply_lid_filter", None)
        if not parsed_args.get("collapse_repeats", False):
            parsed_args.pop("collapse_repeats", None)
        return super().hash(parsed_args)


class ExtendLexiconWithG2PJob(Job):
    """
    Add G2P-generated pronunciations for every word of ``text_file`` that is missing from ``lexicon_file``.

    Motivation: :class:`PhonemizeTextDataJob` silently *drops every line* that contains a word which is not in
    the lexicon (``phonemize_with_sil``). On LibriSpeech dev-other that removes 147 of 2864 utterances (83
    distinct OOV words, nearly all proper names absent from the ``lm_minus_librivox`` lexicon), so the reported
    WER/PER is computed on 94.7% of the corpus only. Feeding :class:`PhonemizeTextDataJob` the extended lexicon
    produced here keeps all sequences.

    The words are phonemized with ``g2p_en`` (the same tool fairseq's wav2vec-U ``prepare_text.sh`` uses to
    build its lexicon), run in a separate venv via ``python_exe`` since it is not installed in the sisyphus
    environment. Stress markers are stripped, and every resulting phoneme is asserted to already occur in
    ``lexicon_file`` -- otherwise the phoneme vocab (which stays pinned to the one derived from the LM corpus)
    would not cover it.
    """

    def __init__(
        self,
        text_file: tk.Path,
        lexicon_file: tk.Path,
        python_exe: tk.Path,
        nltk_data: Optional[tk.Path] = None,
    ):
        """
        :param text_file: one sequence per line, in the same form as given to :class:`PhonemizeTextDataJob`
            (the character normalization of ``normalize_and_filter_text`` is reproduced here)
        :param lexicon_file: existing lexicon, ``<word> <phoneme>+`` per line
        :param python_exe: python of a venv with ``g2p_en`` installed, see ``default_tools.get_g2p_python_exe``
        :param nltk_data: pre-downloaded nltk data dir, so no network access is needed on the compute node
        """
        self.text_file = text_file
        self.lexicon_file = lexicon_file
        self.python_exe = python_exe
        self.nltk_data = nltk_data

        self.out_lexicon_file = self.output_path("lexicon_extended.lst")
        self.out_new_entries = self.output_path("new_entries.lst")
        self.out_num_new_words = self.output_var("num_new_words")

    def tasks(self) -> Iterator[Task]:
        # g2p_en does ~1 ms/word (measured, 0.9 ms/word on 300 OOV words), so even the ~600k OOV types of the
        # 33M-line LM corpus fit; the time mostly goes into reading/normalizing the text.
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "time": 4})

    def run(self):
        script = os.path.join(os.getcwd(), "g2p_oov_words.py")
        with open(script, "w") as f:
            f.write(_G2P_OOV_SCRIPT)

        env = os.environ.copy()
        if self.nltk_data is not None:
            env["NLTK_DATA"] = self.nltk_data.get_path()

        sp.check_call(
            [
                self.python_exe.get_path(),
                script,
                "--text-file",
                self.text_file.get_path(),
                "--lexicon-file",
                self.lexicon_file.get_path(),
                "--out-lexicon-file",
                self.out_lexicon_file.get_path(),
                "--out-new-entries",
                self.out_new_entries.get_path(),
                "--out-num-new-words",
                "num_new_words.txt",
            ],
            env=env,
        )

        with open("num_new_words.txt", "r") as f:
            self.out_num_new_words.set(int(f.read().strip()))


# Run in a separate venv (``g2p_en`` is not available in the sisyphus environment), hence a script and not a
# function of the job above.
_G2P_OOV_SCRIPT = r'''
import argparse
import re
import sys

import regex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--lexicon-file", required=True)
    parser.add_argument("--out-lexicon-file", required=True)
    parser.add_argument("--out-new-entries", required=True)
    parser.add_argument("--out-num-new-words", required=True)
    args = parser.parse_args()

    known_words = set()
    known_phonemes = set()
    with open(args.lexicon_file, "r", encoding="utf-8") as f:
        lexicon_lines = f.readlines()
    for line in lexicon_lines:
        items = line.split()
        assert len(items) > 1, line
        known_words.add(items[0])
        known_phonemes.update(items[1:])
    print(f"lexicon: {len(known_words)} words, {len(known_phonemes)} phonemes", file=sys.stderr)

    # same normalization as PhonemizeTextDataJob.normalize_and_filter_text
    filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")
    oov_words = {}  # word -> first line idx, dict for a deterministic (first-occurrence) order
    with open(args.text_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = " ".join(filter_r.sub(" ", line.strip()).split())
            for word in line.split():
                if word not in known_words:
                    oov_words.setdefault(word, i)
    print(f"{len(oov_words)} OOV words in {args.text_file}", file=sys.stderr)

    from g2p_en import G2p

    g2p = G2p()

    new_entries = []
    failed = []
    for word in oov_words:
        phonemes = [re.sub(r"\d", "", p) for p in g2p(word)]
        # g2p_en emits a stray "'" token (plus a blank) for words with a trailing apostrophe ("says'" ->
        # S EH1 Z ' ), which is no phoneme of any inventory; drop it together with the blanks.
        phonemes = [p for p in phonemes if p.strip() and p != "'"]
        if not phonemes or any(p not in known_phonemes for p in phonemes):
            failed.append((word, phonemes))
            continue
        new_entries.append((word, phonemes))

    assert not failed, (
        f"G2P produced no usable pronunciation for {len(failed)} word(s), which would still be dropped by the "
        f"OOV filter of PhonemizeTextDataJob (phonemes outside the lexicon's inventory cannot be represented by "
        f"the pinned phoneme vocab either): {failed[:20]}"
    )

    with open(args.out_lexicon_file, "w", encoding="utf-8") as f:
        for line in lexicon_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        for word, phonemes in new_entries:
            f.write(f"{word}\t{' '.join(phonemes)}\n")
    with open(args.out_new_entries, "w", encoding="utf-8") as f:
        for word, phonemes in new_entries:
            f.write(f"{word}\t{' '.join(phonemes)}\n")
    with open(args.out_num_new_words, "w", encoding="utf-8") as f:
        f.write(f"{len(new_entries)}\n")
    print(f"added {len(new_entries)} lexicon entries", file=sys.stderr)


if __name__ == "__main__":
    main()
'''


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
                    # without an input seq-tag file we invent a tag. It MUST contain the
                    # task/shard id: `i` restarts at 0 in every task, so a bare `lm-data-{i}`
                    # repeats across the `concurrent` output HDFs and any tag-based lookup
                    # (`seq_list_filter_file`, `get_data_by_seq_tag`) over the full set breaks.
                    seq_tag=[f"lm-data-{task_id}-{i}" if seq_tag is None else seq_tag],
                    extra={"seq_sizes": batch_seq_sizes},
                )

                if i % 10_000 == 0:
                    gc.collect()
                    print(f"Processed sequence {i}/{num_lines} ({i / num_lines * 100:.1f}%)")

            hdf_writer.close()
            shutil.move(tmp_hdf, self.out_hdfs[task_id - 1].get_path())
