from typing import Optional, Tuple

from sisyphus import tk, Path

from i6_experiments.users.schmitt.text.normalize import NormalizeLBSLMDataJob
from i6_experiments.common.datasets import librispeech
from i6_experiments.users.schmitt.corpus.seq_tags import GetSeqTagsFromCorpusJob

from i6_core.tools.download import DownloadJob
from i6_core.corpus.convert import CorpusToTextDictJob, CorpusToTxtJob
from i6_core.text.processing import ConcatenateJob, PipelineJob

from ....data.text import get_phonemized_data
from ....default_tools import get_wav2letter_root


def get_lm_minus_librivox() -> Tuple[tk.Path, Optional[tk.Path]]:
    lm_minus_librivox = DownloadJob(
        url="https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm_corpus/librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt",
        target_filename="lm_corpus_minus_librivox",
    ).out_file
    tk.register_output("data/librispeech/lm/lbs_lm_minus_librivox.raw", lm_minus_librivox)

    return NormalizeLBSLMDataJob(
        wav2letter_root=get_wav2letter_root(),
        wav2letter_python_exe=tk.Path(""),
        librispeech_lm_corpus=lm_minus_librivox,
    ).out_corpus_norm, None


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
    lowercase_text = PipelineJob(concat_text, pipeline=["tr A-Z a-z"]).out

    concat_seq_tags = ConcatenateJob([seq_tags_dev_clean, seq_tags_dev_other], zip_out=False).out
    return lowercase_text, concat_seq_tags


def get_960_text() -> Tuple[tk.Path, tk.Path]:
    text_960h, seq_tags_960h = get_corpus_text("train-other-960")
    lowercase_text = PipelineJob(text_960h, pipeline=["tr A-Z a-z"]).out

    return lowercase_text, seq_tags_960h


def get_text(data_name: str) -> Tuple[tk.Path, tk.Path]:
    if data_name == "lm_minus_librivox":
        return get_lm_minus_librivox()
    if data_name == "960h":
        return get_960_text()
    if data_name == "dev":
        return get_dev_text()

    raise ValueError(f"Unknown dataset name: {data_name}")


def get_phonemized_text(
    data_name: str,
    dump_hdf_concurrent: int,
    lexicon_file: Optional[Path] = None,
    phoneme_vocab: Optional[Path] = None,
):
    text_data, seq_tags = get_text(data_name)

    return get_phonemized_data(
        dataset_name=data_name,
        corpus_name="librispeech",
        text_file=text_data,
        phoneme_vocab=phoneme_vocab,
        dump_hdf_concurrent=dump_hdf_concurrent,
        lexicon_file=lexicon_file,
        seq_tag_file=seq_tags,
    )
