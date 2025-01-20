from i6_experiments.common.datasets.switchboard.bpe import get_subword_nmt_bpe
from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob, FilterSegmentsByListJob
from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob
from i6_core.corpus.segments import SegmentCorpusJob
from i6_experiments.common.datasets.switchboard.corpus_train import get_train_corpus_object_i6_legacy
from i6_experiments.common.datasets.switchboard.corpus_eval import get_hub5e00_corpus_object, get_hub5e01_corpus_object
from i6_experiments.common.datasets.switchboard.lexicon import get_bliss_lexicon

from ..base import DataConfig, LmDataConfig
from .bpe import get_default_bpe_target_config
from ...tools import subword_nmt_repo


def get_default_bpe_train_data(bpe_size: int) -> DataConfig:
    train_corpus_file = get_train_corpus_object_i6_legacy().corpus_file
    segment_files = SegmentCorpusJob(train_corpus_file, 1).out_single_segment_files
    filtered_segments = FilterSegmentsByListJob(
        segment_files,
        filter_list=[
            "switchboard-1/sw04118A/sw4118A-ms98-a-0045",
            "switchboard-1/sw02663A/sw2663A-ms98-a-0022",
            "switchboard-1/sw02986A/sw2986A-ms98-a-0013",
        ],
    ).out_single_segment_files[1]
    return DataConfig(
        bliss_corpus_files=[train_corpus_file],
        speed_perturbation=True,
        ogg_segments=50,
        partition_epoch=3,
        seq_ordering="laplace:.1000",
        target_config=get_default_bpe_target_config(bpe_size),
        segment_file=filtered_segments,
    )


def get_default_bpe_lm_train_data(bpe_size: int) -> LmDataConfig:
    lm_data = get_train_corpus_object_i6_legacy().corpus_file
    train_text = CorpusToTxtJob(
        bliss_corpus=lm_data,
        gzip=True,
    ).out_txt
    bpe_settings = get_subword_nmt_bpe(bpe_size=bpe_size)
    lm_bpe_data_job = ApplyBPEToTextJob(
        text_file=train_text,
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


def get_default_bpe_cv_data(bpe_size: int) -> DataConfig:
    corpus_file = get_hub5e00_corpus_object().corpus_file
    corpus_file = FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=corpus_file,
        bliss_lexicon=get_bliss_lexicon(),
        all_unknown=False,
    ).out_corpus

    segment_files = SegmentCorpusJob(corpus_file, num_segments=1).out_single_segment_files
    filtered_segments = FilterSegmentsByListJob(
        segment_files=segment_files,
        filter_list=[
            "hub5e_00/en_6189a/36",
            "hub5e_00/en_4852b/77",
            "hub5e_00/en_6189b/66",
            "hub5e_00/en_4938b/39",
            "hub5e_00/en_4170b/71",
            "hub5e_00/en_6189b/55",
            "hub5e_00/en_4938b/58",
            "hub5e_00/en_4938b/27",
            "hub5e_00/en_4910b/61",
            "hub5e_00/en_4622b/50",
            "hub5e_00/en_4938b/23",
            "hub5e_00/en_4170a/4",
        ],
    ).out_single_segment_files[1]
    return DataConfig(
        bliss_corpus_files=[corpus_file],
        speed_perturbation=False,
        ogg_segments=1,
        partition_epoch=1,
        seq_ordering="sorted",
        target_config=get_default_bpe_target_config(bpe_size),
        segment_file=filtered_segments,
    )


def get_default_bpe_lm_cv_data(bpe_size: int) -> LmDataConfig:
    bliss_corpus_file = get_hub5e00_corpus_object().corpus_file
    dev_text = CorpusToTxtJob(bliss_corpus=bliss_corpus_file, gzip=True).out_txt
    bpe_settings = get_subword_nmt_bpe(bpe_size=bpe_size)
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


def get_default_prior_data() -> DataConfig:
    # use 33% of the training corpus to estimate the prior
    train_corpus_file = get_train_corpus_object_i6_legacy().corpus_file
    segment_file = SegmentCorpusJob(train_corpus_file, 3).out_single_segment_files[1]

    return DataConfig(
        bliss_corpus_files=[train_corpus_file],
        speed_perturbation=False,
        ogg_segments=50,
        partition_epoch=1,
        seq_ordering="sorted",
        segment_file=segment_file,
    )


def get_default_recog_data(corpus_name: str) -> DataConfig:
    if corpus_name == "hub5e00":
        corpus_file = get_hub5e00_corpus_object().corpus_file
    elif corpus_name == "hub5e01":
        corpus_file = get_hub5e01_corpus_object().corpus_file
    else:
        raise ValueError(f"Recog corpus name '{corpus_name}' not known.")

    return DataConfig(
        bliss_corpus_files=[corpus_file],
        speed_perturbation=False,
        ogg_segments=1,
        partition_epoch=1,
        seq_ordering="sorted",
    )
