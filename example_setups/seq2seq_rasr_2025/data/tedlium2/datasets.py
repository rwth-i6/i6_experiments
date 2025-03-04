from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob
from i6_core.text.processing import ConcatenateJob
from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.tedlium2.textual_data import get_text_data_dict
from i6_experiments.common.datasets.tedlium2.vocab import get_subword_nmt_bpe_v2

from ...model_pipelines.common.corpus import ScorableCorpus
from ...tools import subword_nmt_repo
from ..base import LmDataConfig, MetaOggZipDataConfig, OggZipDataConfig
from .bpe import get_default_bpe_target_config


def get_default_bpe_train_data(bpe_size: int) -> MetaOggZipDataConfig:
    return MetaOggZipDataConfig(
        bliss_corpus_files=[get_bliss_corpus_dict("wav")["train"]],
        speed_perturbation=True,
        ogg_segments=40,
        partition_epoch=4,
        seq_ordering="laplace:.1000",
        target_config=get_default_bpe_target_config(bpe_size),
    )


def get_default_bpe_lm_train_data(bpe_size: int) -> LmDataConfig:
    lm_data = get_text_data_dict()
    full_train_text = ConcatenateJob(
        text_files=[lm_data["audio-transcriptions"], lm_data["background-data"]],
        zip_out=True,
    ).out
    bpe_settings = get_subword_nmt_bpe_v2(bpe_size=bpe_size)
    lm_bpe_data_job = ApplyBPEToTextJob(
        text_file=full_train_text,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_count_vocab,
        gzip_output=True,
        subword_nmt_repo=subword_nmt_repo,
        mini_task=False,  # this is a large file, so run in cluster
    )

    return LmDataConfig(
        corpus_file=lm_bpe_data_job.out_bpe_text,
        vocab_file=bpe_settings.bpe_vocab,
        partition_epoch=100,
        seq_ordering="laplace:.100",
    )


def get_default_bpe_cv_data(bpe_size: int) -> MetaOggZipDataConfig:
    return MetaOggZipDataConfig(
        bliss_corpus_files=[get_bliss_corpus_dict("wav")["dev"]],
        speed_perturbation=False,
        ogg_segments=1,
        partition_epoch=1,
        seq_ordering="sorted",
        target_config=get_default_bpe_target_config(bpe_size),
    )


def get_default_bpe_lm_cv_data(bpe_size: int) -> LmDataConfig:
    dev_text = CorpusToTxtJob(bliss_corpus=get_bliss_corpus_dict("wav")["dev"], gzip=True).out_txt
    bpe_settings = get_subword_nmt_bpe_v2(bpe_size=bpe_size)
    lm_bpe_data_job = ApplyBPEToTextJob(
        text_file=dev_text,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_count_vocab,
        gzip_output=True,
        subword_nmt_repo=subword_nmt_repo,
        mini_task=False,  # this is a large file, so run in cluster
    )

    return LmDataConfig(
        corpus_file=lm_bpe_data_job.out_bpe_text,
        vocab_file=bpe_settings.bpe_vocab,
        partition_epoch=1,
        seq_ordering="sorted",
    )


def get_default_prior_data() -> MetaOggZipDataConfig:
    # use 50% of the training corpus to estimate the prior
    train_corpus_file = get_bliss_corpus_dict("wav")["train"]
    segment_file = SegmentCorpusJob(train_corpus_file, 2).out_single_segment_files[1]

    return MetaOggZipDataConfig(
        bliss_corpus_files=[train_corpus_file],
        speed_perturbation=False,
        ogg_segments=40,
        partition_epoch=1,
        seq_ordering="sorted",
        segment_file=segment_file,
    )


def get_default_recog_data(corpus_name: str) -> OggZipDataConfig:
    return OggZipDataConfig(
        bliss_corpus_files=[get_bliss_corpus_dict("wav")[corpus_name]],
        speed_perturbation=False,
        ogg_segments=1,
        partition_epoch=1,
        seq_ordering="sorted",
    )


def get_default_score_corpus(corpus_name: str) -> ScorableCorpus:
    return ScorableCorpus(
        corpus_name=corpus_name,
        bliss_corpus_file=get_bliss_corpus_dict("wav")[corpus_name],
    )
