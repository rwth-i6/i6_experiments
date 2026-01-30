from sisyphus import tk, Job, Task
from sisyphus.delayed_ops import DelayedFormat, DelayedBase

from typing import Tuple
import os
import shutil
import subprocess as sp

from i6_core.tools.download import DownloadJob
from i6_core.corpus.convert import CorpusToTextDictJob, CorpusToTxtJob
from i6_core.text.convert import TextDictToTextLinesJob
from i6_core.text.processing import ConcatenateJob, PipelineJob

import i6_experiments
from i6_experiments.users.schmitt.corpus.seq_tags import GetSeqTagsFromCorpusJob
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import (
    get_fairseq_root,
    PrepareWav2VecTextDataJob,
)
from i6_experiments.common.datasets import librispeech

import numpy as np
import copy
from typing import Optional, Dict, Any, Iterator

from returnn.datasets.hdf import SimpleHDFWriter

from ...default_tools import KENLM_BINARY_PATH


def _get_corpus_text_dict(key: str) -> Tuple[tk.Path, tk.Path]:
    corpus = librispeech.get_bliss_corpus_dict()[key]
    text_dict = CorpusToTextDictJob(corpus, gzip=True).out_dictionary
    seq_tags = GetSeqTagsFromCorpusJob(corpus, gzip=False).out_txt
    return text_dict, seq_tags


def get_corpus_text(key: str, gzip=False) -> Tuple[tk.Path, tk.Path]:
    """train corpus text (used for LM training)"""
    corpus = librispeech.get_bliss_corpus_dict()[key]
    seq_tags = GetSeqTagsFromCorpusJob(corpus, gzip=gzip).out_txt
    text_lines = CorpusToTxtJob(corpus, gzip=gzip).out_txt
    return text_lines, seq_tags


def get_dev_text() -> Tuple[tk.Path, tk.Path]:
    text_dev_other, seq_tags_dev_other = get_corpus_text("dev-other")
    text_dev_clean, seq_tags_dev_clean = get_corpus_text("dev-clean")
    concat_text = ConcatenateJob([text_dev_clean, text_dev_other], zip_out=False).out
    lowercase_text = PipelineJob(
        concat_text,
        pipeline=["tr A-Z a-z"]
    ).out

    concat_seq_tags = ConcatenateJob([seq_tags_dev_clean, seq_tags_dev_other], zip_out=False).out
    return lowercase_text, concat_seq_tags

def get_960_text() -> Tuple[tk.Path, tk.Path]:
    text_960h, seq_tags_960h = get_corpus_text("train-other-960")
    lowercase_text = PipelineJob(
        text_960h,
        pipeline=["tr A-Z a-z"]
    ).out

    return lowercase_text, seq_tags_960h


def get_phonemized_lm_data(
        text_file: tk.Path,
        dump_hdf_concurrent: int = 1,
        fixed_random_subset: Optional[int] = None,
        vocab_size: int = 1000,  # TODO: THIS IS NOT THE VOCAB SIZE, IT IS THE MIN NUMBER OF TIMES A PHONEME NEEDS TO APPEAR FOR IT TO NOT BE DISCARTED
        alias: Optional[str] = None,
        lexicon_file: Optional[tk.Path] = None,
        phoneme_file: Optional[tk.Path] = None,
        seq_tag_file: Optional[tk.Path] = None,
):
    # Text configuration
    language = "en"  # Language of the text data
    tts_engine = "G2P"  # Text-to-speech engine to use for text normalization
    #text_file_path = "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/data/text_raw/BNCCorpus.txt"
    text_file_path = text_file
    sil_prob = 0.25
    fasttext_model = DownloadJob(
        url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", target_filename="lid.176.bin"
    ).out_file
    training_lm_pruning = [0, 0, 1, 4]

    environment = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3")
    fairseq_root = get_fairseq_root(
        python_env=environment,
        fairseq_root=tk.Path(
            "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
        ),
    )

    if lexicon_file and phoneme_file:
        prepare_text_job_training = PhonemizeTextDataJob(
            text_file=text_file_path,
            fairseq_root=fairseq_root,
            python_env=environment,
            lid_path=fasttext_model,
            language=language,
            sil_prob=sil_prob,
            lexicon_file=lexicon_file,
            phoneme_file=phoneme_file,
            seq_tag_file=seq_tag_file,
        )
        text_file = prepare_text_job_training.out_phoneme_text
    else:
        prepare_text_job_training = PrepareWav2VecTextDataJob(
            fairseq_root=fairseq_root,
            language=language,
            text_file_path=text_file_path,
            kenlm_root=KENLM_BINARY_PATH,
            tts_engine=tts_engine,
            fasttext_model=fasttext_model,
            sil_prob=sil_prob,
            fairseq_python_env=environment,
            vocab_size=vocab_size,
            lm_pruning=training_lm_pruning,
        )
        text_file, phoneme_file = [DelayedFormat(
            f"{{}}/{file_name}",
            prepare_text_job_training.processed_phn_data_and_LM
        ) for file_name in ("lm.phones.filtered.txt", "dict.txt")]
        lexicon_file = DelayedFormat(
            f"{{}}/{'lexicon_filtered.lst'}",
            prepare_text_job_training.out_text_dir
        )
    # if alias:
    #     tk.register_output(f"data/librispeech/lm_phon_text/{alias}", prepare_text_job_training.processed_phn_data_and_LM)

    dump_phoneme_indices_job = DumpPhonemeIndicesToHdfJob(
        text_file=text_file,
        phoneme_file=phoneme_file,
        concurrent=dump_hdf_concurrent,
        fixed_random_subset=fixed_random_subset,
        seq_tag_file=seq_tag_file,
    )
    if alias:
        tk.register_output(f"data/librispeech/lm_phon_text_hdf/{alias}", dump_phoneme_indices_job.out_hdfs[0])

    if lexicon_file and phoneme_file and seq_tag_file:
        return list(
            dump_phoneme_indices_job.out_hdfs.values()), dump_phoneme_indices_job.out_vocab, lexicon_file, phoneme_file, prepare_text_job_training.out_seq_tags

    return list(dump_phoneme_indices_job.out_hdfs.values()), dump_phoneme_indices_job.out_vocab, lexicon_file, phoneme_file, None


class PhonemizeTextDataJob(Job):
    def __init__(
        self,
        text_file: tk.Path,
        fairseq_root: tk.Path,
        python_env: tk.Path,
        language: str,
        sil_prob: float,
        lid_path: tk.Path,
        lexicon_file: Optional[tk.Path],
        phoneme_file: Optional[tk.Path],
        seq_tag_file: Optional[tk.Path] = None,
    ):
        self.text_file = text_file
        self.fairseq_root = fairseq_root
        self.python_env = python_env
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
        env = os.environ.copy()
        env["PATH"] = f"{self.python_env.get_path()}/bin:" + env["PATH"]

        i6_experiments_mod = i6_experiments.__path__[0]
        normalize_cmd = (
            # f"python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py "
            f"python {i6_experiments_mod}/users/schmitt/experiments/exp2025_10_02_shared_enc/librispeech/data/normalize_and_filter_text.py "
            f"{'--seq-tags-file ' + self.seq_tag_file.get_path() if self.seq_tag_file is not None else ''} "
            f"--lang {self.language} --fasttext-model {self.lid_path.get_path()} < {self.text_file.get_path()} | grep -v '\-\-\-' > lm.upper.lid.txt"
        )
        sp.check_call(normalize_cmd, shell=True, env=env)
        # written by normalize_and_filter_text.py
        seq_tag_file = os.path.join(os.getcwd(), "seq-tags-after-norm-and-filter.txt") if self.seq_tag_file is not None else None

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
        sp.check_call(preprocess_cmd, shell=True, env=env)

        cut_cmd = (
            f"cut -f1 -d' ' dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' > words.txt"
        )
        sp.check_call(cut_cmd, shell=True, env=env)

        phonemize_cmd = (
            # f"python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py "
            f"python {i6_experiments_mod}/users/schmitt/experiments/exp2025_10_02_shared_enc/librispeech/data/phonemize_with_sil.py "
            f"-s {self.sil_prob} "
            f"--surround "
            f"{'--seq-tags-file ' + seq_tag_file if seq_tag_file is not None else ''} "
            f"--lexicon {self.lexicon_file.get()} < lm.upper.lid.txt > lm.phones.filtered.txt"
        )
        sp.check_call(phonemize_cmd, shell=True, env=env)

        shutil.move("lm.phones.filtered.txt", self.out_phoneme_text.get_path())
        if self.seq_tag_file is not None:
            shutil.move("seq-tags-after-phonemize.txt", self.out_seq_tags.get_path())

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = copy.deepcopy(parsed_args)
        if d["seq_tag_file"] is None:
            d.pop("seq_tag_file")
        return super().hash(d)


class DumpPhonemeIndicesToHdfJob(Job):
    def __init__(
            self,
            text_file: DelayedBase,
            phoneme_file: DelayedBase,
            concurrent: int = 10,
            fixed_random_subset: Optional[int] = None,
            seq_tag_file: Optional[tk.Path] = None,
    ):
        """

        Args:
            text_file: each line contains a sequence of phonemes separated by space
            phoneme_file: line format: <phoneme> <integer (count?)>
            concurrent: number of concurrent hdf files to dump
            fixed_random_subset: if given, only use a fixed random subset of the data
        """
        self.text_file = text_file
        self.phoneme_file = phoneme_file
        self.concurrent = concurrent
        self.fixed_random_subset = fixed_random_subset
        self.seq_tag_file = seq_tag_file

        self.out_hdfs = {
            i: self.output_path(f"data_{i}.hdf") for i in range(self.concurrent)
        }
        self.out_vocab = self.output_path("phonemes.vocab")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 4}, args=range(1, self.concurrent + 1))

    def run(self, task_id):
        import gc
        import tempfile
        import random

        with open(self.phoneme_file.get(), "r") as f:
            vocab = {line.strip().split()[0]: i for i, line in enumerate(f.readlines())}
        with open(self.out_vocab.get_path(), "w") as f:
            f.write("{\n")
            for phon, i in vocab.items():
                f.write(f'"{phon}": {i},\n')
            f.write("}\n")

        if self.seq_tag_file is not None:
            with open(self.seq_tag_file.get(), "r") as f:
                seq_tags = [line.strip() for line in f.readlines()]
        else:
            seq_tags = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_hdf = os.path.join(tmp_dir, f"data_{task_id}.hdf")
            hdf_writer = SimpleHDFWriter(filename=tmp_hdf, dim=len(vocab), ndim=1)

            with open(self.text_file.get(), "r") as f:
                lines = f.readlines()
            random.Random(42).shuffle(lines)
            if seq_tags is not None:
                random.Random(42).shuffle(seq_tags)

            num_lines = len(lines)
            lines = [lines[i] for i in range(num_lines) if (i % self.concurrent) == (task_id - 1)]
            if seq_tags is not None:
                seq_tags = [seq_tags[i] for i in range(num_lines) if (i % self.concurrent) == (task_id - 1)]
            if self.fixed_random_subset is not None:
                lines = lines[:self.fixed_random_subset]
                if seq_tags is not None:
                    seq_tags = seq_tags[:self.fixed_random_subset]
            num_lines = len(lines)
            gc.collect()

            for i, line in enumerate(lines):
                phonemes = line.strip().split()
                data = [vocab[p] for p in phonemes]
                data = np.array([data])  # (1, T)

                seq_len = len(phonemes)
                seq_lens = {0: np.array([seq_len])}
                batch_seq_sizes = np.expand_dims(seq_lens[0], 1)

                hdf_writer.insert_batch(
                    data,
                    seq_len=seq_lens,
                    seq_tag=[f"lm-data-{i}" if seq_tags is None else seq_tags[i]],
                    extra={"seq_sizes": batch_seq_sizes}
                )

                if i % 10_000 == 0:
                    gc.collect()
                    print(f"Processed sequence {i}/{num_lines} ({i / num_lines * 100:.1f}%)")

            hdf_writer.close()
            shutil.move(tmp_hdf, self.out_hdfs[task_id - 1].get_path())

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = copy.deepcopy(parsed_args)

        if d["fixed_random_subset"] is None:
            d.pop("fixed_random_subset")

        if d["seq_tag_file"] is None:
            d.pop("seq_tag_file")

        return super().hash(d)


