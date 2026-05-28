from typing import Optional

from sisyphus import Path, tk
from sisyphus.delayed_ops import DelayedFormat

from i6_experiments.users.schmitt.datasets.utils.phonemize import PhonemizeTextDataJob
from i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.librispeech.data.text import (
    DumpPhonemeIndicesToHdfJob,
)

from ..default_tools import get_fairseq_root, get_lid_model, get_fasttext_python_exe


def get_phonemized_data(
    dataset_name: str,
    corpus_name: str,
    text_file: Path,
    dump_hdf_concurrent: int,
    fixed_random_subset: Optional[int] = None,
    lexicon_file: Optional[Path] = None,
    phoneme_vocab: Optional[Path] = None,
    language: str = "en",
    sil_prob: float = 0.25,
    seq_tag_file: Optional[Path] = None,
):
    prepare_text_job_training = PhonemizeTextDataJob(
        text_file=text_file,
        fairseq_root=get_fairseq_root(),
        python_exe=get_fasttext_python_exe(),
        lid_path=get_lid_model(),
        language=language,
        sil_prob=sil_prob,
        lexicon_file=lexicon_file,
        phoneme_file=phoneme_vocab,
        seq_tag_file=seq_tag_file,
        min_phoneme_occurrence=1000,
    )
    vocab_file = prepare_text_job_training.out_phoneme_vocab
    text_file = prepare_text_job_training.out_phoneme_text
    tk.register_output(f"data/{corpus_name}/text/phonemized/{dataset_name}.txt", text_file)
    lexicon_file = prepare_text_job_training.out_lexicon_file

    dump_phoneme_indices_job = DumpPhonemeIndicesToHdfJob(
        text_file=text_file,
        phoneme_file=vocab_file,
        concurrent=dump_hdf_concurrent,
        fixed_random_subset=fixed_random_subset,
        seq_tag_file=seq_tag_file,
    )

    return (
        list(dump_phoneme_indices_job.out_hdfs.values()),
        vocab_file,
        lexicon_file,
        prepare_text_job_training.out_seq_tags,
    )
