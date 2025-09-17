from dataclasses import dataclass
from enum import Enum
from functools import cache
import itertools
import os
from typing import Any, Dict, Iterator, List, Tuple
import gzip
import re
import xml.etree.ElementTree as ET
from sisyphus import Job, Task, tk

from apptek_asr.artefacts.factory import AbstractArtefactRepository
from apptek_asr.audio.encoding import ChangeEncodingJob
from apptek_asr.corpus.info import ComputeCorpusStatisticsJob
from apptek_asr.data.ES.process_corpora_gmm import MetaProcessSpanishCorporaGMM
from apptek_asr.meta.metrics import AbstractMetric
from apptek_asr.rasr.crp import ApptekCommonRasrParameters
from apptek_asr.segmentation.apply import LegacySegmentAudioJob
from apptek_asr.users.mgunz.corpus.convert import StmToBlissCorpusJob
from apptek_asr.users.mgunz.corpus.filter import FilterCorpusByDurationWordsRatioJob
from i6_core.corpus import MergeCorporaJob, MergeStrategy
from i6_core.lib.corpus import Corpus
from i6_core.rasr import crp_add_default_output

# Training corpora not used
train_corpora_def = {
    "corpus.ES.f8kHz": [
        "Fisher-batch.1.train.v2",  # 168h
    ],
    "corpus.ES_ES.f8kHz": [
        "Appen-batch.1.v1",  # 96h
    ],
    "corpus.ES_US.f8kHz": [
        "AAA-batch.1.v1",  # 3h
        "CallFriend-batch.1.v1",  # 49h
        "CallHome-batch.1.train.v2",  # 14h
        "CollinCounty-batch.1.v1",  # 90h
        "Ignite.ATNT-batch.1.v1",  # 63h
        "Ignite.ATNT-batch.2.v1",  # 77h
        "Ignite.HomeShopping-batch.1.v1",  # 320h
        "Ignite.LiveAgent-batch.1.v2",  # 20h
        "ListenTrust-batch.1.v1",  # 401h
        "ListenTrust-batch.2.v1",  # 397h
        "ListenTrust-batch.3.capital.v2",  # 27h
        "ListenTrust-batch.4.funeral.v2",  # 10h
        "ListenTrust-batch.5.immigration.v2",  # 39h
        "ListenTrust-batch.6.real.v2",  # 255h
        "ListenTrust-batch.7.simulated.v2",  # 197h
        "NameAddr-batch.1.v1",  # 12h
    ],
}
extended_train_corpora_def = {
    "corpus.ES_AR.f8kHz": [
        "Callcenter.AcVe-batch.1.v2",  # 255h
    ],
    "corpus.ES_CL.f8kHz": [
        "Callcenter.AcVe-batch.1.v2",  # 166h
    ],
    "corpus.ES_CO.f8kHz": [
        "Callcenter.AcVe-batch.1.v2",  # 221h
    ],
    "corpus.ES_PE.f8kHz": [
        "Callcenter.AcVe-batch.1.v2",  # 224h
    ],

}
# CV corpora not used
cv_corpora_def = {
    "corpus.ES.f8kHz": [
        "Fisher-batch.2.dev.v2",  # not sure how many h
    ],
}
#####################
dev_corpora_def = {
    "test_set.ES.f8kHz": [
        "mtp_dev_heldout-v2",
    ],
    "test_set.ES_US.f8kHz": [
        "dev_callhome-v4", # ok
    ],
    "test_set.ES_US.f16kHz": [
        "dev_conversations_202411-v2",
        #"dev-v2",
    ],
}
test_corpora_def = {
    "test_set.ES_ES.f8kHz": [
        "mtp_eval-v2", #present but bug in data loader
    ],
    # "test_set.ES.f16kHz": [
    #     "mtp_eval_heldout-v2",
    # ],
    "test_set.ES_ES.f16kHz": [
        "eval_voice_call-v3", #present but bug in data loader
        "eval_napoli_202210-v3",
    ],
    "test_set.ES_US.f8kHz": [ #ok
        "eval_callcenter_lt-v5",
        "mtp_eval_p1_travel_entertainment-v3",
        "mtp_eval_p2_finance_sales-v3",
        "mtp_eval_p3_retail_realestate-v3",
        "mtp_eval_p4_family_holiday_other-v3",
    ],
    "test_set.ES_US.f16kHz": [
        "eval_movies_tvshows_talks_202303-v3"
        # "mtp_eval-v2",
        # "mtp_eval_p1_news_podcast-v2",
        # "mtp_eval_p2_entertainment_others-v2",
    ],
}

LM_dev_corpora_def = {
    "test_set.ES.f8kHz": [
        "mtp_dev_heldout-v2",
    ],
    "test_set.ES_US.f8kHz": [
        "dev_callhome-v4", # ok
    ],
    "test_set.ES_US.f16kHz": [
        "dev_conversations_202411-v2",
        #"dev-v2",
    ],
}

LM_test_corpora_def = {
    "test_set.ES_ES.f8kHz": [
        "mtp_eval-v2", #present but bug in data loader
    ],
    # "test_set.ES.f16kHz": [
    #     "mtp_eval_heldout-v2",
    # ],
    "test_set.ES_ES.f16kHz": [
        "eval_voice_call-v2", #present but bug in data loader
        #"eval_napoli_202210-v3",
    ],
    "test_set.ES_US.f8kHz": [ #ok
        "eval_callcenter_lt-v5",
        "mtp_eval_p1_travel_entertainment-v3",
        "mtp_eval_p2_finance_sales-v3",
        "mtp_eval_p3_retail_realestate-v3",
        "mtp_eval_p4_family_holiday_other-v3",
    ]
}


segmenter_artefact = (
    "segmenter.f16kHz",
    "2017-06-tf-am-segmenter-batch",
)
segmenter_flow_artefact = (
    "feature_flow",
    "mel-fb-45-legacy-segmenter-16kHz-batch-with-mean-var-norm",
)
rasr_artefact = (
    "rasr",
    "streaming-rasr-2024-06-21",
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
                "top_n": 1, #20
                "buffer_size": 100,
                "min_speech_frames": 600, #50
                "mini_music_frames": 50,
                "min_silence_gap": 300, #30
                "max_speech_frames": 1000, #2000
                "max_speech_dur": 1000, #2000
                "hard_stop_max_speech_frames": 1500, #3000
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
#    SegmenterType.PylasrE2E,
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
class Corpora:
    cv: tk.Path
    train: tk.Path

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
        raise ValueError(f"could not find corpus {name} in all spanish corpora")

    def get_eval_info(self, name: str) -> EvalInfo:
        if name in self.dev:
            return self.dev[name]
        elif name in self.test:
            return self.test[name]
        else:
            raise ValueError(f"unknown eval corpus {name}")


@cache
def _get_train_cv_corpus(namespace: str, corpus_key: str) -> tk.Path:
    aar = AbstractArtefactRepository()
    artefact = aar.get_artefact_factory(namespace, corpus_key).build()
    out_corpus_file = (
        StmToBlissCorpusJob(artefact["stm"], artefact["audio_files"], skip_non_speech=True).out_bliss
        if "stm" in artefact
        else artefact["corpus"].corpus_file
    )
    spanish_mapping_Y_to_y = MetaProcessSpanishCorporaGMM(out_corpus_file)
    spanish_mapping_Y_to_y.run()
    out_corpus_file = spanish_mapping_Y_to_y.jobs["get_final_corpus"].out_corpus
    preprocess_job = PreprocessSpanishCorpusJob(out_corpus_file)
    out_corpus_file = preprocess_job.out_corpus
    return out_corpus_file


@cache
def _get_eval_corpus(
    namespace: str, corpus_key: str, segmenter_type: SegmenterType, measure_type: WerMeasure, alias_prefix: str
) -> EvalInfo:
    aar = AbstractArtefactRepository()

    artefact = aar.get_artefact_factory(namespace, corpus_key).build()
    assert "stm" in artefact

    khz = namespace.split(".")[-1]
    if khz == "f16kHz":
        audio_files = artefact["audio_files"]
    elif khz == "f8kHz":
        resample_job = ChangeEncodingJob(
            file_list=artefact["audio_files"],
            output_filenames=[os.path.basename(p) for p in artefact["audio_files"]],
            output_format="wav",
            sample_rate=16000,
            recover_duration=False,
        )
        audio_files = list(resample_job.out_files.values())
    else:
        raise ValueError(f"unknown sample rate {khz}")
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
        segment_audio_job = LegacySegmentAudioJob(
            crp=crp,
            feature_flow=segmenter_feature_flow,
            audio_files=audio_files,
            **segmenter_opts,
        )
        segment_audio_job.add_alias(f"segmenter/{namespace}.{corpus_key}/{seg_opts_as_str}")
        out_corpus_file = segment_audio_job.out_merged_corpus
        out_corpus_file = AddFakeTranscriptionJob(out_corpus_file).out_corpus
        out_corpus_file = ProjectBlissRefToSegmenterJob(full_corpus, out_corpus_file).out_corpus
        #tk.register_output("test/align_aptk_seg_Corpus",ProjectBlissRefToSegmenterJob(full_corpus, out_corpus_file).out_corpus)
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
        #tk.register_output(f"{alias_prefix}/corpus/{name}.xml.gz", corpus)

        stats = ComputeCorpusStatisticsJob(corpus, audio_dir=None)
        #tk.register_output(f"{alias_prefix}/costa/{name}/avg-seg-length", stats.average_segment_length)
        #tk.register_output(f"{alias_prefix}/costa/{name}/duration", stats.corpus_duration)

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
def get_corpora(
    alias_prefix: str = "datasets/spanish/f16kHz", include_extra_train_data_beyond_2k: bool = False, for_lm: bool = False,
) -> Corpora:
    train_corpora = {
        f"{corpus_ns}.{corpus}": _get_train_cv_corpus(corpus_ns, corpus)
        for corpus_ns, corpus_list in train_corpora_def.items()
        for corpus in corpus_list
    }
    cv_corpora = {
        f"{corpus_ns}.{corpus}": _get_train_cv_corpus(corpus_ns, corpus)
        for corpus_ns, corpus_list in cv_corpora_def.items()
        for corpus in corpus_list
    }
    dev_corpora = {
        f"{corpus_ns}.{corpus}.{segmenter_type}.{measure_type}": _get_eval_corpus(
            corpus_ns, corpus, segmenter_type, measure_type, alias_prefix
        )
        for corpus_ns, corpus_list in (LM_dev_corpora_def.items() if for_lm else dev_corpora_def.items())
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
        for corpus_ns, corpus_list in (LM_test_corpora_def.items() if for_lm else test_corpora_def.items())
        for corpus in corpus_list
        for segmenter_type in ALL_SEGMENTER_TYPES
        for measure_type in ALL_MEASURE_TYPES
        if (
            "fisher" not in corpus or segmenter_type != SegmenterType.Reference
        )  # exclude fisher corpora w/ ref seg, because there is none -> inference goes OOM
    }
    _compute_merged_costa("eval", test_corpora, alias_prefix)

    train_corpus_merge_job = MergeCorporaJob(
        list(train_corpora.values()), merge_strategy=MergeStrategy.CONCATENATE, name="spanish-8k-train"
    )
    filter_train_corpus_job = FilterCorpusByDurationWordsRatioJob(train_corpus_merge_job.out_merged_corpus, ratio=0.25)
    filter_train_corpus_job.add_alias(f"{alias_prefix}/corpus/train")
    train_corpus = filter_train_corpus_job.out_corpus
    #tk.register_output(f"{alias_prefix}/corpus/train.xml.gz", train_corpus)

    train_stats = ComputeCorpusStatisticsJob(train_corpus, audio_dir=None)
    #tk.register_output(f"{alias_prefix}/costa/train/avg-seg-length", train_stats.average_segment_length)
    #tk.register_output(f"{alias_prefix}/costa/train/avg-seg-length-std", train_stats.average_segment_length_std)
    #tk.register_output(f"{alias_prefix}/costa/train/duration", train_stats.corpus_duration)

    cv_corpus_merge_job = MergeCorporaJob(
        list(cv_corpora.values()), merge_strategy=MergeStrategy.CONCATENATE, name="spanish-8k-cv"
    )
    filter_cv_corpus_job = FilterCorpusByDurationWordsRatioJob(cv_corpus_merge_job.out_merged_corpus, ratio=0.25)
    filter_cv_corpus_job.add_alias(f"{alias_prefix}/corpus/cv")
    cv_corpus = filter_cv_corpus_job.out_corpus
    #tk.register_output(f"{alias_prefix}/corpus/cv.xml.gz", cv_corpus)

    cv_stats = ComputeCorpusStatisticsJob(cv_corpus, audio_dir=None)
    #tk.register_output(f"{alias_prefix}/costa/cv/avg-seg-length", cv_stats.average_segment_length)
    #tk.register_output(f"{alias_prefix}/costa/cv/avg-seg-length-std", cv_stats.average_segment_length_std)
    #tk.register_output(f"{alias_prefix}/costa/cv/duration", cv_stats.corpus_duration)

    crp = Corpora(cv=cv_corpus, train=train_corpus, dev=dev_corpora, test=test_corpora)
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


class PreprocessSpanishCorpusJob(Job):
    """
    Drop non-word phones and invalid transcription labels from the Spanish corpora.
    """

    def __init__(self, bliss_corpus: tk.Path, gzip=True):
        self.bliss_corpus = bliss_corpus

        self.out_corpus = self.output_path("corpus.xml" + (".gz" if gzip else ""))

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import re

        word_map = {
            "**pal**": "pal",
            "**jerardina**": "jerardina",
            "hic-": " ",
            "sis-": " ",
            r"#ah": " ",
            r"#eh": " ",
            r"#er": " ",
            r"#uh": " ",
            r"<overlap>#eh": " ",
            r"<overlap>\xc2\xbfy": "y",
            r"<overlap>contestar": "contestar",
            r"<overlap>cuidarnos": "cuidarnos",
            r"<overlap>d": "d",
            r"<overlap>de": "de",
            r"<overlap>eduardo": "eduardo",
            r"<overlap>entonces": "entonces",
            r"<overlap>es": "es",
            r"<overlap>muy": "muy",
            r"<overlap>okay": "okay",
            r"<overlap>que": "que",
            r"<overlap>tenemos": "tenemos",
            r"<overlap>uh-huh": " ",
            r"<overlap>veces": "veces",
            r"\xc2\xbf": " ",
            r"\xc2\xbfa": "a",
            r"\xc2\xbfah": "ah",
            r"\xc2\xbfal\xc3\xb3": "aló",
            r"\xc2\xbfc\xc3\xb3mo": "cómo",
            r"\xc2\xbfcon": "con",
            r"\xc2\xbfcu\xc3\xa1l": "á",
            r"\xc2\xbfcu\xc3\xa1ndo": "á",
            r"\xc2\xbfcu\xc3\xa1ntos": "á",
            r"\xc2\xbfde": "de",
            r"\xc2\xbfel": "el",
            r"\xc2\xbfen": "en",
            r"\xc2\xbfes": "es",
            r"\xc2\xbfest\xc3\xa1": "está",
            r"\xc2\xbfgui\xc3\xb3n": "guión",
            r"\xc2\xbfi": "i",
            r"\xc2\xbfme": "me",
            r"\xc2\xbfn": "n",
            r"\xc2\xbfno": "no",
            r"\xc2\xbfo": "o",
            r"\xc2\xbfokay": "okay",
            r"\xc2\xbfonce": "once",
            r"\xc2\xbfpara": "para",
            r"\xc2\xbfpero": "pero",
            r"\xc2\xbfpor": "por",
            r"\xc2\xbfqu\xc3\xa9": "qué",
            r"\xc2\xbfs\xc3\xad": "sí",
            r"\xc2\xbftodav\xc3\xada": "todavía",
            r"\xc2\xbfusted": "usted",
            r"\xc2\xbfverdad": "verdad",
            r"\xc2\xbfy": "y",
            r"\xc2\xbfya": "ya",
            r"\xc3\xa1rea": "área",
            r"\xc3\xb3igame": "óigame",
        }
        replace_map = {
            "s<num>": " ",
            r"\xc2\xbf": " ",
            r"\xc3\xa1": "á",
            r"\xc3\xa9": "é",
            r"\xc3\xad": "í",
            r"\xc3\xba": "ú",
            r"\xc3\xb3": "ó",
            r"\xc3\xb1": "ñ",
            r".": "",
            "º": "",
            "~": "-",  # mapping to word fragment if word end, so will be deleted in later step
            "</overlap>": " ",
        }
        char_map = str.maketrans(
            {
                "⇢": " ",
                ",": " ",
                "¡": " ",
                "?": " ",
                "¿": " ",
                "*": " ",
                bytes.fromhex("d98c").decode("utf-8"): " ",
                bytes.fromhex("d98e").decode("utf-8"): " ",
            }
        )
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
            orth = orth.translate(char_map)
            for old, new in replace_map.items():
                orth = orth.replace(old, new)
            orth = [word_map.get(w, w) for w in orth.split()]
            orth = [w for w in orth if w and not w.startswith("[")]
            return " ".join(orth)

        corp = Corpus()
        corp.load(self.bliss_corpus.get_path())
        for seg in corp.segments():
            seg.orth = preprocess_orth(seg.orth)
        corp.filter_segments(lambda _corp, _rec, seg: seg.orth is not None and len(seg.orth.strip()) > 0)
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





class ProjectBlissRefToSegmenterJob(Job):
    """
    Project reference <orth> onto segmenter-made segments for ALL <recording>s.

    Inputs (gzipped Bliss):
      - ref_bliss_gz: tk.Path -> *.xml.gz (reference Bliss with <orth>)
      - seg_bliss_gz: tk.Path -> *.xml.gz (segmenter Bliss with <unk>)

    Output:
      - projected.xml.gz  (same structure as segmenter, but with filled <orth>)

    Parameters:
      - min_overlap: ignore ref/seg pairs with less than this overlap (sec)
      - keep_unk_if_empty: keep "<unk>" when no tokens land in a segment
    """

    def __init__(
        self,
        ref_bliss_gz: tk.Path,
        seg_bliss_gz: tk.Path,
        *,
        min_overlap: float = 0.02,
        keep_unk_if_empty: bool = True,
    ):
        self.ref_bliss_gz = ref_bliss_gz
        self.seg_bliss_gz = seg_bliss_gz
        self.min_overlap = float(min_overlap)
        self.keep_unk_if_empty = bool(keep_unk_if_empty)

        self.out_corpus = self.output_path("projected.xml.gz")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 1})

    # ---------------- Internals ----------------

    @dataclass
    class RefSeg:
        start: float
        end: float
        text: str

    @dataclass
    class SegSeg:
        start: float
        end: float
        name: str

    _SPACE_RE = re.compile(r"\s+")

    @staticmethod
    def _clean_text(s: str) -> str:
        return ProjectBlissRefToSegmenterJob._SPACE_RE.sub(" ", (s or "").strip())

    @staticmethod
    def _parse_all_ref_recordings(path_gz: str) -> Dict[str, List["ProjectBlissRefToSegmenterJob.RefSeg"]]:
        """
        Parse ALL <recording>s from reference Bliss.
        Returns: { recording_name: [RefSeg, ...], ... }
        """
        with gzip.open(path_gz, "rt", encoding="utf-8") as f:
            tree = ET.parse(f)
        root = tree.getroot()
        out: Dict[str, List[ProjectBlissRefToSegmenterJob.RefSeg]] = {}
        for rec in root.findall("./recording"):
            rec_name = rec.attrib.get("name", "")
            refs: List[ProjectBlissRefToSegmenterJob.RefSeg] = []
            for seg in rec.findall("./segment"):
                s = float(seg.attrib["start"])
                e = float(seg.attrib["end"])
                orth_el = seg.find("./orth")
                txt = orth_el.text if (orth_el is not None and orth_el.text) else ""
                refs.append(ProjectBlissRefToSegmenterJob.RefSeg(s, e, txt))
            out[rec_name] = refs
        return out

    @staticmethod
    def _parse_all_seg_recordings(path_gz: str):
        """
        Parse ALL <recording>s from segmenter Bliss (and keep the XML nodes to write back).
        Returns:
          tree: ElementTree for the segmenter XML
          seg_map: { recording_name: ( [SegSeg], [segment_nodes] ) }
        """
        with gzip.open(path_gz, "rt", encoding="utf-8") as f:
            tree = ET.parse(f)
        root = tree.getroot()
        seg_map = {}
        for rec in root.findall("./recording"):
            rec_name = rec.attrib.get("name", "")
            segs: List[ProjectBlissRefToSegmenterJob.SegSeg] = []
            nodes = []
            for seg in rec.findall("./segment"):
                s = float(seg.attrib["start"])
                e = float(seg.attrib["end"])
                name = seg.attrib.get("name", "")
                segs.append(ProjectBlissRefToSegmenterJob.SegSeg(s, e, name))
                nodes.append(seg)
            seg_map[rec_name] = (segs, nodes)
        return tree, seg_map

    @staticmethod
    def _interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        s = max(a[0], b[0])
        e = min(a[1], b[1])
        return max(0.0, e - s)

    @staticmethod
    def _tokenize_keep_punct(text: str) -> List[str]:
        text = ProjectBlissRefToSegmenterJob._clean_text(text)
        if not text:
            return []
        return text.split(" ")

    def _assign_tokens_by_midpoint(
        self, rs: float, re: float, tokens: List[str], seg_start: float, seg_end: float
    ) -> List[str]:
        """Assign tokens to a single segment bin by uniform midpoint; return tokens that land in [seg_start, seg_end)."""
        if rs >= re or not tokens:
            return []
        dur = re - rs
        N = len(tokens)
        out = []
        for i, tok in enumerate(tokens, start=1):
            mid = rs + ((i - 0.5) / N) * dur
            if (mid >= seg_start and mid < seg_end) or (abs(mid - seg_end) < 1e-8):
                out.append(tok)
        return out

    def _project_one_recording(
        self,
        refs: List["ProjectBlissRefToSegmenterJob.RefSeg"],
        segs: List["ProjectBlissRefToSegmenterJob.SegSeg"],
    ) -> List[str]:
        # Build overlap candidates per segmenter segment
        per_seg_ref_ids: List[List[int]] = [[] for _ in segs]
        for ri, r in enumerate(refs):
            a = (r.start, r.end)
            for si, s in enumerate(segs):
                b = (s.start, s.end)
                if self._interval_overlap(a, b) >= self.min_overlap:
                    per_seg_ref_ids[si].append(ri)

        out_texts: List[str] = []
        for si, s in enumerate(segs):
            toks_out: List[str] = []
            for ri in per_seg_ref_ids[si]:
                r = refs[ri]
                toks = self._tokenize_keep_punct(r.text)
                toks_here = self._assign_tokens_by_midpoint(r.start, r.end, toks, s.start, s.end)
                toks_out.extend(toks_here)
            out_texts.append(self._clean_text(" ".join(toks_out)))
        return out_texts

    def run(self):
        # Parse full corpora
        ref_map = self._parse_all_ref_recordings(self.ref_bliss_gz.get_path())
        seg_tree, seg_map = self._parse_all_seg_recordings(self.seg_bliss_gz.get_path())

        # For each segmenter recording, project using the matching reference recording (if present)
        for rec_name, (segs, seg_nodes) in seg_map.items():
            refs = ref_map.get(rec_name, [])
            if refs:
                out_texts = self._project_one_recording(refs, segs)
            else:
                # No refs for this recording → keep empty so we can optionally keep <unk>
                out_texts = ["" for _ in segs]

            # Write back into XML nodes
            assert len(seg_nodes) == len(out_texts)
            for node, txt in zip(seg_nodes, out_texts):
                orth = node.find("./orth")
                if orth is None:
                    orth = ET.SubElement(node, "orth")
                if not txt and self.keep_unk_if_empty:
                    orth.text = " <unk> "
                else:
                    orth.text = txt

        # Pretty print if available (Python 3.9+)
        try:
            ET.indent(seg_tree, space="  ")
        except Exception:
            pass

        # Write gzipped output
        with gzip.open(self.out_corpus.get_path(), "wt", encoding="utf-8") as f:
            seg_tree.write(f, encoding="unicode", xml_declaration=True)