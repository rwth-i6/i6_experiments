from dataclasses import dataclass
from enum import Enum
from functools import cache
import itertools
import os
from typing import Any, Dict, Iterator, Tuple

from sisyphus import Job, Task, tk

from apptek_asr.artefacts import AbstractArtefactRepository
from apptek_asr.audio.encoding import ChangeEncodingJob
from apptek_asr.corpus.info import ComputeCorpusStatisticsJob
from apptek_asr.meta.metrics import AbstractMetric
from apptek_asr.rasr.crp import ApptekCommonRasrParameters
from apptek_asr.segmentation.apply import LegacySegmentAudioJob
from apptek_asr.users.mgunz.corpus.convert import StmToBlissCorpusJob
from apptek_asr.users.mgunz.corpus.filter import FilterCorpusByDurationWordsRatioJob
from i6_core.corpus import MergeCorporaJob, MergeStrategy
from i6_core.lib.corpus import Corpus
from i6_core.rasr import crp_add_default_output


# total: about 29,6kh
train_corpora_def = {
    "corpus.EN_AU.f16kHz": [
        "abc.Australia-batch.1.base.v2",  # 2039h
        "abc.Australia-batch.2.roadmap.v2",  # 678h
    ],
    "corpus.EN_CA.f16kHz": [
        "UPV.ourcommons-batch.1.base.v2",  # 503h
    ],
    "corpus.EN_GB.f16kHz": [
        "100Brit-batch.1.v2",  # 18h
        "BBC-batch.1.v2",  # 2076h
        "BBCradio-batch.1.v2",  # 31h
        "euronews-euro20000.v2",  # 328h
        "euronews-euronews201802.v2",  # 58h
    ],
    "corpus.EN_IN.f8kHz": [
        "IndianAccented-batch.1.v2",  # 128h
    ],
    "corpus.EN_KE.f8kHz": [
        "AfricaAccented-batch.1.v2",  # 143h
        "AfricaAccented-batch.2.v2",  # 205h
    ],
    "corpus.EN_NG.f8kHz": [
        "AfricaAccented-batch.1.v2",  # 53h
    ],
    "corpus.EN_SG.f8kHz": [
        "SingaporeAccented-batch.1.v2",  # 197h
    ],
    "corpus.EN_US.f8kHz": [
        "AAVE-batch.1.v2",  # 95h
        "AAVE-batch.2.v2",  # 9h
        "AAVE-batch.3.v2",  # 266h
        "AAVE-batch.4.v2",  # 100h
        "Callcenter.EasyBreathe-batch.1.v2",  # 480h
        "ChineseAccented-batch.1.v2",  # 93h
        "CustomerService.AAA-batch.1.v2",  # 106h
        "CustomerService.Ignite-batch.1.v2",  # 19h
        "CustomerService.Navigant-batch.1.bank.v2",  # 15h
        "CustomerService.Navigant-batch.1.kodiak.v2",  # 37h
        "Execvision-EV-001.v2",  # 40h
        "Execvision-EV-002.v2",  # 5h
        "Execvision-EV-003.v2",  # 172h
        "Execvision-EV-004.v2",  # 19h
        "Execvision-EV-005.v2",  # 104h
        "Execvision-EV-20190923.v2",  # 211h
        "Execvision-EV-20191030.v2",  # 191h
        "Execvision-EV-20191223.v2",  # 422h
        "Execvision-EV-20200210.v2",  # 346h
        "Execvision-EV-20200831.v2",  # 806h
        "Execvision-EV-production.v2",  # 404h
        "finance_earnings_calls-batch.1.v2",  # 180h
        "finance_intuit_calls-batch.1.v2",  # 53h
        "finance_intuit_calls-batch.2.v2",  # 132h
        "finance_intuit_calls-batch.3.v2",  # 3h
        "Fisher-fsh.1.v2",  # 1012h
        "Fisher-fsh.2.v2",  # 1020h
        "Hispanic-callcenter.v2",  # 92h
        "Hispanic-capital.v2",  # 181h
        "Hispanic-general_telephony.v2",  # 179h
        "Hispanic-real.v2",  # 711h
        "Hispanic-resurge.v2",  # 75h
        "Hispanic-seranova_cs.v2",  # 239h
        "Hispanic-seranova.v2",  # 318h
        "Hispanic-simulated.v2",  # 282h
        "JailCalls-batch.1.collin.v2",  # 60h
        "JailCalls-batch.2.wise.v2",  # 175h
        "SouthernDialects-callcenter.v2",  # 96h
        "SouthernDialects-general_telephony.v2",  # 107h
        "Switchboard-swb.v2",  # 312h
    ],
    "corpus.EN_US.f16kHz": [
        "BOLT-all.v3",  # 70h
        "commonvoice-cv-en-6.1-2020-12-11.batch.1.v2",  # 901h
        "DIVA-batch.1.v3",  # 2h
        "DIVA-batch.2.v3",  # 13h
        "dotsub-batch.1.v4",  # 1190h
        "Entertainment-batch.1.v2",  # 158h
        "Entertainment-batch.2.v2",  # 638h
        "FarmJournal-batch.1.v2",  # 40h
        "Hub4-1996_train.v2",  # 75h
        "Hub4-1997_train.v2",  # 72h
        "IDenTV-batch.1.v3",  # 918h
        "IDenTV-batch.2.v3",  # 2136h
        "LibriSpeech-train-clean-100.v3",  # 100h
        "LibriSpeech-train-clean-360.v3",  # 364h
        "LibriSpeech-train-other-500.v3",  # 497h
        "MedicalBCN-batch.1.v2",  # 279h
        "NEWS.HQ-batch.1.CNN.v3",  # 769h
        "NEWS.HQ-batch.2.NPR.v3",  # 2613h
        "TedTalks-batch.1.v2",  # 335h
        "TVeyes-batch.1.v4",  # 2521h
    ],
    "corpus.EN_ZH.f8kHz": [
        "ChineseAccented-batch.1.v2",  # 138h
        "ChineseAccented-batch.2.v2",  # 136h
        "ChineseAccented-batch.3.v2",  # 23h
    ],
}
cv_corpora_def = {
    "corpus.EN_US.f8kHz": [
        "AAVE-dev.v2",
        "Execvision-EV-20190923.dev.v2",  # 2h
        "Execvision-EV-20191030.dev.v2",  # 2h
        "Execvision-EV-20191223.dev.v2",  # 2h
        "Execvision-EV-20200210.dev.v2",  # 2h
        "Execvision-EV-20200831.dev.v2",  # 2h
        "Execvision-EV-production.dev.v2",  # 2h
    ],
    "corpus.EN_US.f16kHz": [
        "commonvoice-cv-en-6.1-2020-12-11.dev.v2",  # 3h
    ],
}
dev_corpora_def = {
    "test_set.EN_US.f16kHz": [
        "dev-v6",
        "dev_news_202203-v3",
        "dev_keynote_202207-v3",
        "dev_meetings_202207-v3",
        "dev_movies_tvshows_202207-v3",
        "dev_librispeech_dev_other-v3",
    ],
}
test_corpora_def = {
    # "test_set.EN_GB.f8kHz": [
    #     "eval-v3",
    # ],
    "test_set.EN_GB.f16kHz": [
        "eval-v3",
        "eval_entertainment_a-v3",
        "eval_entertainment_b-v3",
        "eval_news_bbc-v3",
        "eval_news_misc-v3",
        "eval_itv_news_talkshow_202203-v2",
    ],
    "test_set.EN_US": [
        "eval_evds_billing_solutions_202403-v3",
        "eval_evds_communication_202404-v3",
        "eval_evds_enterprise_mgmt_202404-v3",
        "eval_evds_entertainment_202404-v3",
        "eval_evds_food_202404-v3",
        "eval_evds_healthcare_202403-v3",
        "eval_evds_hr_202403-v3",
        "eval_evds_it_202404-v3",
        "eval_evds_promotional_offers_202404-v3",
        "eval_evds_sales_202404-v3",
        "eval_evds_sports_202404-v3",
    ],
    "test_set.EN_US.f8kHz": [
        "eval_aave_callcenter-v3",
        "eval_chinese_general_telephony-v3",
        "eval_finance_earnings_calls-v2",
        "eval_hispanic_callcenter-v3",
        "eval_hispanic_general_telephony-v3",
        "eval_singapore-v3",
        # "eval_southern_callcenter-v3", # empty?
        "eval_southern_general_telephony-v3",
        "eval-v6",
        "hub5_2000_eval-v2",
        "mtp_eval-v3",
    ],
    "test_set.EN_US.f16kHz": [
        "eval-v7",
        # "mtp_eval-v5",
        "eval_keynote-v4",
        "eval_keynote_large-v3",
        "eval_news_202201-v4",
        "eval_meetings_202203-v3",
        "eval_movies_tvshows_202203-v3",
        "eval_diva_202206-v4",
        "eval_librispeech_test_clean-v3",
        "eval_librispeech_test_other-v3",
        "eval_racing_202310-v4",  # F1 go brrr
    ],
}

segmenter_artefact = (
    "segmenter.f16kHz",
    "2017-06-tf-am-segmenter-batch",
    # "2022-tf-am-segmenter-batch",
)
segmenter_flow_artefact = (
    "feature_flow",
    # "mel-fb-34-legacy-segmenter-8kHz-batch-with-mean-var-norm",
    "mel-fb-45-legacy-segmenter-16kHz-batch-with-mean-var-norm",
)
rasr_artefact = (
    "rasr",
    # "streaming-rasr-2024-06-21",
    "streaming-rasr-2025-01-31",
)


class SegmenterType(Enum):
    Reference = "ref"
    AppTekLegacy = "aptk_leg"
    Pylasr = "pylasr"
    PylasrE2E = "pylasr_e2e"

    def get_opts(self) -> Dict[str, Any]:
        if self == SegmenterType.Reference:
            return {}
        elif self == SegmenterType.AppTekLegacy:
            return {
                "top_n": 20,
                "buffer_size": 100,
                "min_speech_frames": 50,
                "mini_music_frames": 50,
                "min_silence_gap": 30,
                "max_speech_frames": 2000,
                "max_speech_dur": 2000,
                "hard_stop_max_speech_frames": 3000,
            }
        elif self == SegmenterType.PylasrE2E:
            return {
                "top_n": 1,
                "buffer_size": 100,
                "min_speech_frames": 600,
                "mini_music_frames": 50,
                "min_silence_gap": 300,
                "max_speech_frames": 1000,
                "max_speech_dur": 1000,
                "hard_stop_max_speech_frames": 1500,
            }
        else:
            raise ValueError(f"segmenter type {self} is not implemented yet")

    def __str__(self) -> str:
        return self.value


ALL_SEGMENTER_TYPES = [
    SegmenterType.AppTekLegacy,
    SegmenterType.Reference,
    SegmenterType.PylasrE2E,
]


class WerMeasure(Enum):
    FF_WER = "ff_wer"
    WER = "wer"

    def __str__(self) -> str:
        return self.value


ALL_MEASURE_TYPES = [WerMeasure.FF_WER]


@dataclass(frozen=True)
class EvalInfo:
    corpus: tk.Path
    glm: tk.Path
    measure_type: WerMeasure
    metrics: AbstractMetric
    segmented_corpus: tk.Path
    segmenter_type: SegmenterType
    stm: tk.Path


@dataclass(frozen=True)
class CorpusDef:
    bliss_corpus: tk.Path
    duration: float


@dataclass(frozen=True)
class Corpora:
    cv: tk.Path
    train: Dict[str, CorpusDef]

    dev: Dict[str, EvalInfo]
    test: Dict[str, EvalInfo]

    def get_corpus(self, name: str) -> tk.Path:
        in_self = getattr(self, name, None)
        if in_self is not None:
            return in_self
        in_dev = self.dev.get(name, None)
        if in_dev is not None:
            return in_dev.corpus
        in_test = self.test.get(name, None)
        if in_test is not None:
            return in_test.corpus
        raise ValueError(f"could not find corpus {name} in all english corpora")

    def get_eval_info(self, name: str) -> EvalInfo:
        if name in self.dev:
            return self.dev[name]
        elif name in self.test:
            return self.test[name]
        else:
            raise ValueError(f"unknown eval corpus {name}")


@cache
def _get_train_cv_corpus(namespace: str, corpus_key: str) -> Tuple[tk.Path, float]:
    aar = AbstractArtefactRepository()
    artefact = aar.get_artefact_factory(namespace, corpus_key).build()
    assert "corpus" in artefact
    out_corpus_file = artefact["corpus"].corpus_file
    # if "f8kHz" in namespace:
    #     out_corpus_file = BlissChangeEncodingJob(
    #         out_corpus_file,
    #         output_format="ogg",
    #         sample_rate=16_000,
    #         recover_duration=False,
    #     ).out_corpus
    out_corpus_file = PreprocessEnglishCorpusJob(out_corpus_file, hash_break=1).out_corpus
    return out_corpus_file, artefact["corpus"].duration


@cache
def _get_eval_corpus(
    namespace: str, corpus_key: str, segmenter_type: SegmenterType, measure_type: WerMeasure, alias_prefix: str
) -> EvalInfo:
    aar = AbstractArtefactRepository()

    artefact = aar.get_artefact_factory(namespace, corpus_key).build()
    assert "stm" in artefact

    audio_files = artefact["audio_files"]
    full_corpus = StmToBlissCorpusJob(artefact["stm"], audio_files, skip_non_speech=True).out_bliss
    out_corpus_file = full_corpus

    segmenter_opts = segmenter_type.get_opts()
    seg_opts_as_str = "segmenter"
    for k, v in segmenter_opts.items():
        seg_opts_as_str += f"-{k}-{v}"
    if segmenter_type in [SegmenterType.Pylasr, SegmenterType.PylasrE2E]:
        seg_opts_as_str += "-pylasrSeg"

    if segmenter_type == SegmenterType.Reference:
        pass
    elif segmenter_type == SegmenterType.AppTekLegacy or segmenter_type == SegmenterType.PylasrE2E:
        compile_rasr_job, _ = aar.get_artefact_factory(*rasr_artefact).build()
        crp = ApptekCommonRasrParameters()
        crp.legacy_segmenter_exe = compile_rasr_job.out_apptek_segmenter_recognizer_conf
        crp_add_default_output(crp)
        segmenter_config, segmenter_post_config = aar.get_artefact_factory(*segmenter_artefact).build()
        crp.segmenter_config = segmenter_config
        crp.segmenter_post_config = segmenter_post_config
        segmenter_feature_flow = aar.get_artefact_factory(*segmenter_flow_artefact).build()
        segmenter_feature_flow.add_node(
            "signal-resampling", "resample-audio", {"resample-rate": 16_000}
        )
        segmenter_feature_flow.unlink("demultiplex", "convert")
        segmenter_feature_flow.link("demultiplex", "resample-audio")
        segmenter_feature_flow.link("resample-audio", "convert")
        segment_audio_job = LegacySegmentAudioJob(
            crp=crp,
            feature_flow=segmenter_feature_flow,
            audio_files=artefact["audio_files"],
            rtf=4,
            **segmenter_opts,
        )
        segment_audio_job.add_alias(f"segmenter/{namespace}.{corpus_key}/{seg_opts_as_str}")
        out_corpus_file = segment_audio_job.out_merged_corpus
    elif segmenter_type == SegmenterType.Pylasr:
        from apptek_asr.pylasr.segment import CreateScpFromWavs, SegmentScp

        scp_job = CreateScpFromWavs(audio_files)
        segment_audio_job = SegmentScp(
            scp=scp_job.scp,
            concurrent=10,
            topn=segmenter_opts["top_n"],
            minsilgap=segmenter_opts["min_silence_gap"],
            minspeechframes=segmenter_opts["min_speech_frames"],
            maxspeechframes=segmenter_opts["max_speech_frames"],
            maxspeechdur=segmenter_opts["max_speech_dur"],
            minimusicframes=segmenter_opts["mini_music_frames"],
            hardstopmaxspeechframes=segmenter_opts["hard_stop_max_speech_frames"],
        )
        segment_audio_job.add_alias(f"segmenter/{namespace}.{corpus_key}/{seg_opts_as_str}")
        rename_job = RenameSegmentScpCorpus(segment_audio_job.out_merged_corpus)
        out_corpus_file = rename_job.out_merged_corpus

    out_corpus_file = AddFakeTranscriptionJob(out_corpus_file).out_corpus
    eval_info = EvalInfo(
        corpus=full_corpus,
        glm=artefact["glm"],
        measure_type=measure_type,
        metrics=artefact["metrics"],
        segmented_corpus=out_corpus_file,
        segmenter_type=segmenter_type,
        stm=artefact["stm"],
    )

    for corpus, name in [
        (full_corpus, f"{namespace}.{corpus_key}"),
        (out_corpus_file, f"{namespace}.{corpus_key}.seg.{segmenter_type}"),
    ]:
        tk.register_output(f"{alias_prefix}/corpus/{name}.xml.gz", corpus)

        stats = ComputeCorpusStatisticsJob(corpus, audio_dir=None)
        tk.register_output(f"{alias_prefix}/costa/{name}/avg-seg-length", stats.average_segment_length)
        tk.register_output(f"{alias_prefix}/costa/{name}/duration", stats.corpus_duration)

    return eval_info


def _compute_merged_costa(name: str, corpus_dict: Dict[str, EvalInfo], alias_prefix: str):
    for segmenter_type, dummy_measure_type in itertools.product(ALL_SEGMENTER_TYPES, ALL_MEASURE_TYPES[:1]):
        corpora_to_merge = [
            eval_info.segmented_corpus
            for key, eval_info in corpus_dict.items()
            if key.endswith(f"{segmenter_type}.{dummy_measure_type}")
        ]
        merge_corpora_job = MergeCorporaJob(corpora_to_merge, merge_strategy=MergeStrategy.CONCATENATE, name=name)
        stats_job = ComputeCorpusStatisticsJob(merge_corpora_job.out_merged_corpus, audio_dir=None)
        tk.register_output(
            f"{alias_prefix}/costa/{name}.{segmenter_type}/avg-seg-length",
            stats_job.average_segment_length,
        )
        tk.register_output(
            f"{alias_prefix}/costa/{name}.{segmenter_type}/avg-seg-length-std",
            stats_job.average_segment_length_std,
        )
        tk.register_output(f"{alias_prefix}/costa/{name}.{segmenter_type}/duration", stats_job.corpus_duration)


@cache
def get_corpora(alias_prefix: str = "datasets/english/mbw") -> Corpora:
    train_corpora = {
        f"{corpus_ns}.{corpus}": _get_train_cv_corpus(corpus_ns, corpus)
        for corpus_ns, corpus_list in train_corpora_def.items()
        for corpus in corpus_list
    }
    cv_corpora = {
        f"{corpus_ns}.{corpus}": _get_train_cv_corpus(corpus_ns, corpus)[0]
        for corpus_ns, corpus_list in cv_corpora_def.items()
        for corpus in corpus_list
    }
    dev_corpora = {
        f"{corpus_ns}.{corpus}.{segmenter_type}.{measure_type}": _get_eval_corpus(
            corpus_ns, corpus, segmenter_type, measure_type, alias_prefix
        )
        for corpus_ns, corpus_list in dev_corpora_def.items()
        for corpus in corpus_list
        for segmenter_type in ALL_SEGMENTER_TYPES
        for measure_type in ALL_MEASURE_TYPES
        if (
            "fisher" not in corpus or segmenter_type != SegmenterType.Reference
        )  # exclude fisher corpora w/ ref seg, because there is none -> inference goes OOM
    }
    _compute_merged_costa("dev", dev_corpora, alias_prefix)
    test_corpora = {
        f"{corpus_ns}.{corpus}.{segmenter_type}.{measure_type}": _get_eval_corpus(
            corpus_ns, corpus, segmenter_type, measure_type, alias_prefix
        )
        for corpus_ns, corpus_list in test_corpora_def.items()
        for corpus in corpus_list
        for segmenter_type in ALL_SEGMENTER_TYPES
        for measure_type in ALL_MEASURE_TYPES
        if (
            "fisher" not in corpus or segmenter_type != SegmenterType.Reference
        )  # exclude fisher corpora w/ ref seg, because there is none -> inference goes OOM
    }
    _compute_merged_costa("eval", test_corpora, alias_prefix)

    filtered_train_corpus_jobs = {
        key: FilterCorpusByDurationWordsRatioJob(corp, ratio=0.25) for key, (corp, _dur) in train_corpora.items()
    }
    for key, job in filtered_train_corpus_jobs.items():
        job.add_alias(f"{alias_prefix}/corpus/{key}")
        tk.register_output(f"{alias_prefix}/corpus/{key}.xml.gz", job.out_corpus)
    train_corpus_def = {
        key: CorpusDef(bliss_corpus=filtered_train_corpus_jobs[key].out_corpus, duration=dur)
        for key, (_, dur) in train_corpora.items()
    }

    for key, (corp, _dur) in train_corpora.items():
        costa_job = ComputeCorpusStatisticsJob(corp, audio_dir=None)
        tk.register_output(f"{alias_prefix}/costa/{key}/avg-seg-length", costa_job.average_segment_length)
        tk.register_output(f"{alias_prefix}/costa/{key}/avg-seg-length-std", costa_job.average_segment_length_std)
        tk.register_output(f"{alias_prefix}/costa/{key}/duration", costa_job.corpus_duration)

    train_corpus_list = [job.out_corpus for job in filtered_train_corpus_jobs.values()]
    merged_train_corpus = MergeCorporaJob(train_corpus_list, name="english-mbw-train")
    train_stats = ComputeCorpusStatisticsJob(merged_train_corpus.out_merged_corpus, audio_dir=None)
    tk.register_output(f"{alias_prefix}/costa/train/avg-seg-length", train_stats.average_segment_length)
    tk.register_output(f"{alias_prefix}/costa/train/avg-seg-length-std", train_stats.average_segment_length_std)
    tk.register_output(f"{alias_prefix}/costa/train/duration", train_stats.corpus_duration)

    cv_corpus_merge_job = MergeCorporaJob(
        list(cv_corpora.values()), merge_strategy=MergeStrategy.CONCATENATE, name="english-mbw-cv"
    )
    filter_cv_corpus_job = FilterCorpusByDurationWordsRatioJob(cv_corpus_merge_job.out_merged_corpus, ratio=0.25)
    filter_cv_corpus_job.add_alias(f"{alias_prefix}/corpus/cv")
    cv_corpus = filter_cv_corpus_job.out_corpus
    tk.register_output(f"{alias_prefix}/corpus/cv.xml.gz", cv_corpus)

    cv_stats = ComputeCorpusStatisticsJob(cv_corpus, audio_dir=None)
    tk.register_output(f"{alias_prefix}/costa/cv/avg-seg-length", cv_stats.average_segment_length)
    tk.register_output(f"{alias_prefix}/costa/cv/avg-seg-length-std", cv_stats.average_segment_length_std)
    tk.register_output(f"{alias_prefix}/costa/cv/duration", cv_stats.corpus_duration)

    crp = Corpora(cv=cv_corpus, train=train_corpus_def, dev=dev_corpora, test=test_corpora)
    return crp


class AddFakeTranscriptionJob(Job):
    """
    Set orth to <unk>
    """

    def __init__(self, bliss_corpus: tk.Path, gzip=True):
        self.bliss_corpus = bliss_corpus

        self.out_corpus = self.output_path("corpus.xml" + (".gz" if gzip else ""))

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        corp = Corpus()
        corp.load(self.bliss_corpus.get_path())
        for seg in corp.segments():
            seg.orth = "<unk>"
        corp.dump(self.out_corpus.get_path())


class RenameSegmentScpCorpus(Job):
    """
    copied from here: /home/hwu/setups/2024-04-08_ph-transducer-hwu-es-8khz/recipe/i6_experiments/users/gruev/helpers.py
    """

    def __init__(self, bliss_corpus: tk.Path):
        """
        :param bliss_corpus: Corpus for which recording and segment names should be replaced
        """
        self.bliss_corpus = bliss_corpus
        self.out_merged_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = Corpus()
        c.load(self.bliss_corpus.get_path())

        for rec in c.all_recordings():
            rec.name = rec.name.rsplit("/", 1)[-1]

        for seg in c.segments():
            seg.name = seg.name.rsplit("/", 1)[-1]

        c.dump(self.out_merged_corpus.get_path())


class PreprocessEnglishCorpusJob(Job):
    """
    Drop non-word phones and invalid transcription labels from the Spanish corpora.
    """

    __sis_hash_exclude__ = {"hash_break": 0}

    def __init__(self, bliss_corpus: tk.Path, gzip=True, hash_break=0):
        self.bliss_corpus = bliss_corpus

        self.out_corpus = self.output_path("corpus.xml" + (".gz" if gzip else ""))

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import re

        words_to_drop = [
            "[laughter]",
            "[noise]",
            "[vocalized-noise]",
            "[unknown]",
            "[vocalized-unknown]",
            "<unk>",
        ]
        pattern = re.compile("|".join([re.escape(w) for w in words_to_drop]))

        def preprocess_orth(orth: str) -> str:
            orth = pattern.sub("", orth)
            orth = [w for w in orth.split() if w and not w.startswith("[")]
            return " ".join(orth)

        corp = Corpus()
        corp.load(self.bliss_corpus.get_path())
        for seg in corp.segments():
            seg.orth = preprocess_orth(seg.orth)
        corp.filter_segments(lambda _corp, _rec, seg: seg.orth is not None and len(seg.orth.strip()) > 0)
        corp.dump(self.out_corpus.get_path())
