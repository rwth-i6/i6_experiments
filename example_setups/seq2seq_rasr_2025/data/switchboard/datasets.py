from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob, FilterSegmentsByListJob
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob
from .bpe import get_bpe_settings
from i6_experiments.common.datasets.switchboard.corpus_eval import (
    get_hub5e00,
    get_hub5e00_corpus_object,
    get_hub5e01,
    get_hub5e01_corpus_object,
)
from i6_experiments.common.datasets.switchboard.corpus_train import get_train_corpus_object_i6_legacy
from i6_experiments.common.datasets.switchboard.lexicon import get_bliss_lexicon

from ...model_pipelines.common.corpus import ScorableCorpus
from ...tools import subword_nmt_repo
from ..base import DataConfig, LmDataConfig, MetaOggZipDataConfig, OggZipDataConfig, RemoveWordsFromTranscriptionsJob
from .bpe import get_default_bpe_target_config


def get_default_bpe_train_data(bpe_size: int) -> DataConfig:
    train_corpus_file = get_train_corpus_object_i6_legacy().corpus_file
    train_corpus_file = RemoveWordsFromTranscriptionsJob(
        train_corpus_file, ["[NOISE]", "[LAUGHTER]", "[VOCALIZED-NOISE]"]
    ).out_corpus_file
    segment_files = SegmentCorpusJob(train_corpus_file, 1).out_single_segment_files
    filtered_segments = FilterSegmentsByListJob(
        segment_files,
        filter_list=[
            "switchboard-1/sw04118A/sw4118A-ms98-a-0045",
            "switchboard-1/sw02663A/sw2663A-ms98-a-0022",
            "switchboard-1/sw02986A/sw2986A-ms98-a-0013",
        ],
    ).out_single_segment_files[1]
    return MetaOggZipDataConfig(
        bliss_corpus_files=[train_corpus_file],
        speed_perturbation=True,
        ogg_segments=50,
        partition_epoch=6,
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
    bpe_settings = get_bpe_settings(bpe_size)
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
    corpus_file = RemoveWordsFromTranscriptionsJob(corpus_file, ["(%HESITATION)"]).out_corpus_file
    corpus_file = FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=corpus_file,
        bliss_lexicon=get_bliss_lexicon(),
        all_unknown=False,
    ).out_corpus

    segment_files = SegmentCorpusJob(corpus_file, num_segments=1).out_single_segment_files
    filtered_segments = FilterSegmentsByListJob(
        segment_files=segment_files,
        filter_list=[
            "hub5e_00/en_4938b/58",
            "hub5e_00/en_4938b/67",
            "hub5e_00/en_6267a/25",
            "hub5e_00/en_4938b/27",
            "hub5e_00/en_4938a/17",
            "hub5e_00/en_4938b/69",
            "hub5e_00/en_6267a/15",
            "hub5e_00/en_6189b/66",
            "hub5e_00/en_4938b/39",
            "hub5e_00/en_6282b/24",
            "hub5e_00/en_4170b/71",
            "hub5e_00/en_6267a/11",
            "hub5e_00/en_6189b/26",
            "hub5e_00/en_4852b/77",
            "hub5e_00/en_6189a/36",
            "hub5e_00/en_6189b/3",
            "hub5e_00/en_6189a/38",
            "hub5e_00/en_6489a/54",
            "hub5e_00/en_6658b/6",
            "hub5e_00/en_4910b/52",
            "hub5e_00/en_6489a/44",
            "hub5e_00/en_4616a/24",
            "hub5e_00/en_4910b/31",
            "hub5e_00/en_6489b/7",
            "hub5e_00/en_4938b/46",
            "hub5e_00/en_6658b/71",
            "hub5e_00/en_6189b/69",
            "hub5e_00/en_6267a/42",
            "hub5e_00/en_6489b/50",
            "hub5e_00/en_4966a/67",
            "hub5e_00/en_6489b/9",
            "hub5e_00/en_4910b/61",
            "hub5e_00/en_6189a/84",
            "hub5e_00/en_4170b/58",
            "hub5e_00/en_6189b/55",
            "hub5e_00/en_4574b/24",
            "hub5e_00/en_4170b/64",
            "hub5e_00/en_4622b/50",
            "hub5e_00/en_4966b/21",
            "hub5e_00/en_6267a/35",
            "hub5e_00/en_4966a/22",
            "hub5e_00/en_4852b/63",
            "hub5e_00/en_6912a/41",
            "hub5e_00/en_4938b/23",
            "hub5e_00/en_4170a/4",
            "hub5e_00/en_6282b/21",
            "hub5e_00/en_6189b/37",
            "hub5e_00/en_4938b/77",
            "hub5e_00/en_4170a/2",
            "hub5e_00/en_4170a/81",
            "hub5e_00/en_6267a/4",
            "hub5e_00/en_6282a/1",
            "hub5e_00/en_6189b/54",
            "hub5e_00/en_4910a/40",
            "hub5e_00/en_4616b/43",
            "hub5e_00/en_6489a/77",
            "hub5e_00/en_4170a/5",
            "hub5e_00/en_4170a/8",
            "hub5e_00/en_6189b/1",
            "hub5e_00/en_4938b/47",
            "hub5e_00/en_6489a/41",
            "hub5e_00/en_6489b/43",
            "hub5e_00/en_4938b/2",
            "hub5e_00/en_6267b/69",
            "hub5e_00/en_6489b/35",
            "hub5e_00/en_4852a/39",
            "hub5e_00/en_6489a/2",
        ],
    ).out_single_segment_files[1]
    return MetaOggZipDataConfig(
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
    bpe_settings = get_bpe_settings(bpe_size=bpe_size)
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

    return MetaOggZipDataConfig(
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

    return OggZipDataConfig(
        bliss_corpus_files=[corpus_file],
        speed_perturbation=False,
        ogg_segments=1,
        partition_epoch=1,
        seq_ordering="sorted",
    )


def get_default_score_corpus(corpus_name: str) -> ScorableCorpus:
    if corpus_name == "hub5e00":
        dataset = get_hub5e00()
        corpus_file = get_hub5e00_corpus_object().corpus_file
    elif corpus_name == "hub5e01":
        dataset = get_hub5e01
        corpus_file = get_hub5e01_corpus_object().corpus_file
    else:
        raise ValueError(f"Recog corpus name '{corpus_name}' not known.")
    return ScorableCorpus(
        corpus_name=corpus_name,
        bliss_corpus_file=corpus_file,
        stm_file=dataset.stm,
        glm_file=dataset.glm,
        score_job_type="Hub5",
    )
