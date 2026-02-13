from i6_core.corpus import FilterCorpusBySegmentsJob
from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob, FilterSegmentsByListJob
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob
from sisyphus import tk
from .bpe import get_bpe_settings
from i6_experiments.common.datasets.switchboard.corpus_eval import (
    get_hub5e00,
    get_hub5e00_corpus_object,
    get_hub5e01,
    get_hub5e01_corpus_object,
)
from i6_experiments.common.datasets.switchboard.corpus_train import get_train_bliss_corpus_i6_legacy
from .lexicon import get_raw_bliss_lexicon
from .phoneme import get_phoneme_target_hdf_file

from ...model_pipelines.common.corpus import ScorableCorpus, ScoreJobType
from ...tools import subword_nmt_repo
from ..base import (
    HdfDataConfig,
    LmDataConfig,
    MetaOggZipDataConfig,
    MetaOggZipHdfTargetDataConfig,
    OggZipDataConfig,
    RemoveWordsFromTranscriptionsJob,
)
from .bpe import get_default_bpe_target_config


def get_default_bpe_train_data(bpe_size: int) -> MetaOggZipDataConfig:
    train_corpus_file = get_train_bliss_corpus_i6_legacy()
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
    return MetaOggZipDataConfig.from_bliss(
        bliss_corpus_files=[train_corpus_file],
        speed_perturbation=True,
        ogg_segments=50,
        partition_epoch=6,
        seq_ordering="laplace:.1000",
        target_config=get_default_bpe_target_config(bpe_size),
        segment_file=filtered_segments,
    )


def get_default_phoneme_train_data() -> MetaOggZipHdfTargetDataConfig:
    corpus_file = get_train_bliss_corpus_i6_legacy()
    return MetaOggZipHdfTargetDataConfig(
        oggzip_config=OggZipDataConfig.from_bliss(
            bliss_corpus_files=[corpus_file],
            speed_perturbation=True,
            ogg_segments=50,
            partition_epoch=6,
            seq_ordering="laplace:.1000",
            target_config=None,
        ),
        oggzip_target_name=None,
        hdf_config=HdfDataConfig(files=[get_phoneme_target_hdf_file(corpus_file)]),
        hdf_target_name="classes",
    )


def get_default_bpe_phoneme_train_data(bpe_size: int) -> MetaOggZipHdfTargetDataConfig:
    corpus_file = get_train_bliss_corpus_i6_legacy()
    return MetaOggZipHdfTargetDataConfig(
        oggzip_config=OggZipDataConfig.from_bliss(
            bliss_corpus_files=[corpus_file],
            speed_perturbation=True,
            ogg_segments=50,
            partition_epoch=6,
            seq_ordering="laplace:.1000",
            target_config=get_default_bpe_target_config(bpe_size),
        ),
        oggzip_target_name="bpe",
        hdf_config=HdfDataConfig(files=[get_phoneme_target_hdf_file(corpus_file)]),
        hdf_target_name="phoneme",
    )


def get_default_bpe_lm_train_data(bpe_size: int) -> LmDataConfig:
    lm_data = get_train_bliss_corpus_i6_legacy()
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


def _get_hub5e00_cv_file() -> tk.Path:
    corpus_file = get_hub5e00_corpus_object().corpus_file
    corpus_file = RemoveWordsFromTranscriptionsJob(corpus_file, ["(%HESITATION)"]).out_corpus_file
    corpus_file = FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=corpus_file,
        bliss_lexicon=get_raw_bliss_lexicon(),
        all_unknown=False,
    ).out_corpus

    segment_files = SegmentCorpusJob(corpus_file, num_segments=1).out_single_segment_files
    filtered_segments = FilterSegmentsByListJob(
        segment_files=segment_files,
        filter_list=[
            "hub5e_00/en_4170a/10",
            "hub5e_00/en_4170a/12",
            "hub5e_00/en_4170a/24",
            "hub5e_00/en_4170a/46",
            "hub5e_00/en_4170a/66",
            "hub5e_00/en_4170b/53",
            "hub5e_00/en_4183b/55",
            "hub5e_00/en_4404a/21",
            "hub5e_00/en_4574a/3",
            "hub5e_00/en_4574a/4",
            "hub5e_00/en_4574a/12",
            "hub5e_00/en_4616a/4",
            "hub5e_00/en_4616a/9",
            "hub5e_00/en_4616a/21",
            "hub5e_00/en_4616a/25",
            "hub5e_00/en_4616b/47",
            "hub5e_00/en_4622a/2",
            "hub5e_00/en_4622a/5",
            "hub5e_00/en_4622a/14",
            "hub5e_00/en_4622a/17",
            "hub5e_00/en_4622a/20",
            "hub5e_00/en_4622b/58",
            "hub5e_00/en_4910b/50",
            "hub5e_00/en_4938a/23",
            "hub5e_00/en_4938a/37",
            "hub5e_00/en_4938a/50",
            "hub5e_00/en_4938b/44",
            "hub5e_00/en_4938b/56",
            "hub5e_00/en_4938b/66",
            "hub5e_00/en_4966a/28",
            "hub5e_00/en_4966a/53",
            "hub5e_00/en_4966b/59",
            "hub5e_00/en_5011a/18",
            "hub5e_00/en_5011a/22",
            "hub5e_00/en_5011a/23",
            "hub5e_00/en_5011a/28",
            "hub5e_00/en_5011a/34",
            "hub5e_00/en_5011b/13",
            "hub5e_00/en_5017a/18",
            "hub5e_00/en_5017a/42",
            "hub5e_00/en_5017a/53",
            "hub5e_00/en_5017b/4",
            "hub5e_00/en_5017b/20",
            "hub5e_00/en_6189a/34",
            "hub5e_00/en_6189a/59",
            "hub5e_00/en_6189a/81",
            "hub5e_00/en_6267a/7",
            "hub5e_00/en_6267a/28",
            "hub5e_00/en_6267a/29",
            "hub5e_00/en_6267a/55",
            "hub5e_00/en_6267a/61",
            "hub5e_00/en_6267a/62",
            "hub5e_00/en_6489a/62",
            "hub5e_00/en_6489a/64",
            "hub5e_00/en_6489b/25",
            "hub5e_00/en_6489b/61",
            "hub5e_00/en_6489b/63",
            "hub5e_00/en_6658a/27",
            "hub5e_00/en_6658b/62",
            "hub5e_00/en_6912a/4",
            "hub5e_00/sw_4390a/4",
            "hub5e_00/sw_4390a/13",
            "hub5e_00/sw_4390a/35",
            "hub5e_00/sw_4390a/39",
            "hub5e_00/sw_4390b/30",
            "hub5e_00/sw_4484b/12",
            "hub5e_00/sw_4507b/20",
            "hub5e_00/sw_4507b/32",
            "hub5e_00/sw_4507b/45",
            "hub5e_00/sw_4507b/53",
            "hub5e_00/sw_4507b/57",
            "hub5e_00/sw_4537b/22",
            "hub5e_00/sw_4537b/42",
            "hub5e_00/sw_4543a/34",
            "hub5e_00/sw_4547a/36",
            "hub5e_00/sw_4547a/41",
            "hub5e_00/sw_4560a/6",
            "hub5e_00/sw_4560a/11",
            "hub5e_00/sw_4560a/24",
            "hub5e_00/sw_4577a/37",
            "hub5e_00/sw_4580a/18",
            "hub5e_00/sw_4580a/24",
            "hub5e_00/sw_4580a/34",
            "hub5e_00/sw_4580a/47",
            "hub5e_00/sw_4580a/53",
            "hub5e_00/sw_4580b/12",
            "hub5e_00/sw_4580b/13",
            "hub5e_00/sw_4580b/20",
            "hub5e_00/sw_4580b/28",
            "hub5e_00/sw_4604a/6",
            "hub5e_00/sw_4604b/33",
            "hub5e_00/sw_4686b/15",
            "hub5e_00/sw_4686b/23",
            "hub5e_00/sw_4686b/31",
            "hub5e_00/sw_4689a/28",
            "hub5e_00/sw_4694a/2",
            "hub5e_00/sw_4694a/38",
            "hub5e_00/sw_4694b/50",
            "hub5e_00/sw_4824a/19",
            "hub5e_00/sw_4824a/26",
            "hub5e_00/sw_4910a/4",
            "hub5e_00/sw_4910a/25",
            "hub5e_00/sw_4910a/32",
            "hub5e_00/sw_4910a/43",
            "hub5e_00/sw_4910b/1",
            "hub5e_00/sw_4910b/56",
        ],
    ).out_single_segment_files[1]
    corpus_file = FilterCorpusBySegmentsJob(
        bliss_corpus=corpus_file, segment_file=filtered_segments, compressed=True, delete_empty_recordings=True
    ).out_corpus

    return corpus_file


def get_default_bpe_cv_data(bpe_size: int) -> MetaOggZipDataConfig:
    return MetaOggZipDataConfig.from_bliss(
        bliss_corpus_files=[_get_hub5e00_cv_file()],
        speed_perturbation=False,
        ogg_segments=1,
        partition_epoch=1,
        seq_ordering="sorted",
        target_config=get_default_bpe_target_config(bpe_size),
    )


def get_default_phoneme_cv_data() -> MetaOggZipHdfTargetDataConfig:
    corpus_file = _get_hub5e00_cv_file()
    return MetaOggZipHdfTargetDataConfig(
        oggzip_config=OggZipDataConfig.from_bliss(
            bliss_corpus_files=[_get_hub5e00_cv_file()],
            speed_perturbation=False,
            ogg_segments=1,
            partition_epoch=1,
            seq_ordering="sorted",
            target_config=None,
        ),
        oggzip_target_name=None,
        hdf_config=HdfDataConfig(files=[get_phoneme_target_hdf_file(corpus_file)]),
        hdf_target_name="classes",
    )


def get_default_bpe_phoneme_cv_data(bpe_size: int) -> MetaOggZipHdfTargetDataConfig:
    corpus_file = _get_hub5e00_cv_file()
    return MetaOggZipHdfTargetDataConfig(
        oggzip_config=OggZipDataConfig.from_bliss(
            bliss_corpus_files=[corpus_file],
            speed_perturbation=False,
            ogg_segments=1,
            partition_epoch=1,
            seq_ordering="sorted",
            target_config=get_default_bpe_target_config(bpe_size),
        ),
        oggzip_target_name="bpe",
        hdf_config=HdfDataConfig(files=[get_phoneme_target_hdf_file(corpus_file)]),
        hdf_target_name="phoneme",
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


def get_default_prior_data() -> MetaOggZipDataConfig:
    # use 33% of the training corpus to estimate the prior
    train_corpus_file = get_train_bliss_corpus_i6_legacy()
    segment_file = SegmentCorpusJob(train_corpus_file, 3).out_single_segment_files[1]

    return MetaOggZipDataConfig.from_bliss(
        bliss_corpus_files=[train_corpus_file],
        speed_perturbation=False,
        ogg_segments=50,
        partition_epoch=1,
        seq_ordering="sorted",
        segment_file=segment_file,
    )


def get_default_recog_data(corpus_name: str) -> OggZipDataConfig:
    if corpus_name == "hub5e00":
        corpus_file = get_hub5e00_corpus_object().corpus_file
    elif corpus_name == "hub5e01":
        corpus_file = get_hub5e01_corpus_object().corpus_file
    else:
        raise ValueError(f"Recog corpus name '{corpus_name}' not known.")

    return OggZipDataConfig.from_bliss(
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
        dataset = get_hub5e01()
        corpus_file = get_hub5e01_corpus_object().corpus_file
    else:
        raise ValueError(f"Recog corpus name '{corpus_name}' not known.")
    return ScorableCorpus(
        corpus_name=corpus_name,
        bliss_corpus_file=corpus_file,
        stm_file=dataset.stm,
        glm_file=dataset.glm,
        score_job_type=ScoreJobType.Hub5,
    )
