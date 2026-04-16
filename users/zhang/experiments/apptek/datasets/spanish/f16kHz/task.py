__all__ = ["get_asr_task", "SpanishTask"]

from dataclasses import dataclass
from functools import partial
import itertools
from typing import Any, Dict, Optional, Sequence

from i6_experiments.users.zhang.common.util.delayed import ToVariableJob
from i6_experiments.users.zhang.common.util.json import DumpAsJsonJob
from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zhang.datasets.score_results import (
    MeasureType,
    RecogOutput,
    ScoreResult,
    ScoreResultCollection,
)
from i6_experiments.users.zhang.datasets.task import Task as AsrTask
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from sisyphus import Path, tk

from apptek_asr.artefacts import AbstractArtefactRepository
from apptek_asr.meta.evaluations.aggregated_scoring import (
    ES_MBW_TEST_SET_SPECS_V1,
    ES_MBW_TEST_SET_SPECS_V2,
    spanish_mbw_metrics_v1,
    spanish_mbw_metrics_v2,
)
from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import get_task_data
from i6_core.returnn.search import SearchWordsToCTMJob
from i6_core.text.label.sentencepiece.train import SentencePieceType
from i6_core.text.processing import PipelineJob
from i6_experiments.users.zhang.sentencepiece import train_sentence_piece_model
from .corpus_lex import ALL_SEGMENTER_TYPES, Corpora, SegmenterType, WerMeasure, get_corpora


ALIAS_PREFIX = "datasets/spanish/f16kHz"
#RETURNN_ROOT = tk.Path("/home/mgunz/setups/2024-07-08--zeyer-setup-apptek/recipe/returnn")

@dataclass
class SpanishTask(AsrTask):
    corpora: Corpora = None  # needs a default for python field order
    sampling_rate: int = 16_000
    spm: SentencePieceModel = None  # needs a default for python field order

    def __post_init__(self):
        super().__post_init__()

        assert self.corpora is not None
        assert self.spm is not None


def get_asr_task(
    *,
    spm_dim: int,
    spm_type: SentencePieceType,
    spm_extra_symbols: Optional[Sequence[str]] = None,
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_partition_epoch: int,
    returnn_root: Path,
    include_extra_train_data_beyond_2k: bool = False,
    **kwargs,
) -> SpanishTask:
    assert spm_dim > 0
    assert train_partition_epoch > 0

    corpora = get_corpora(
        include_extra_train_data_beyond_2k=include_extra_train_data_beyond_2k,
        alias_prefix=ALIAS_PREFIX,
    )
    spm, _vocab = train_sentence_piece_model(
        corpora.train,
        dim=spm_dim,
        ty=spm_type,
        user_defined_symbols=spm_extra_symbols,
        alias_prefix=ALIAS_PREFIX,
    )
    task_data = get_task_data(
        corpora=corpora,
        returnn_root=returnn_root,
        spm=spm,
        train_partition_epoch=train_partition_epoch,
        train_vocab_opts=train_vocab_opts,
        train_dataset_kwargs=kwargs,
        alias_prefix=ALIAS_PREFIX,
    )
    task = SpanishTask(
        name="spanish-16khz",
        corpora=corpora,
        dev_dataset=task_data.cv,
        eval_datasets={**task_data.dev, **task_data.test},
        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="test_set.ES_ES.f8kHz.mtp_eval-v2.ref.ff_wer", #"test_set.ES_US.f8kHz.mtp_eval_p4_family_holiday_other-v3.ref.ff_wer"
        train_dataset=task_data.train,
        train_epoch_split=train_partition_epoch,
        score_recog_output_func=partial(_score_recog_out_v2, corpora=corpora, remove_labels=spm_extra_symbols),
        recog_post_proc_funcs=[_spm_to_words],
        #collect_score_results_func=_score_aggregate_es,
        spm=spm,
    )
    return task

def get_asr_task_given_spm(
    *,
    spm: SentencePieceModel,
    spm_extra_symbols: Optional[Sequence[str]] = None,
    train_partition_epoch: int = 1,
    returnn_root: Path,
    include_extra_train_data_beyond_2k: bool = False,
    aggregate_wer: bool = True,
    **kwargs,
) -> SpanishTask:
    assert train_partition_epoch > 0

    corpora = get_corpora(
        include_extra_train_data_beyond_2k=include_extra_train_data_beyond_2k,
        alias_prefix=ALIAS_PREFIX,
    )
    task_data = get_task_data(
        corpora=corpora,
        returnn_root=returnn_root,
        spm=spm,
        train_partition_epoch=train_partition_epoch,
        train_vocab_opts=None,
        train_dataset_kwargs=kwargs,
        alias_prefix=ALIAS_PREFIX,
    )
    from i6_experiments.users.zhang.experiments.apptek_exp_wer_ppl import seg_key
    task = SpanishTask(
        name="spanish-16khz",
        corpora=corpora,
        dev_dataset=task_data.cv,
        eval_datasets={**task_data.dev, **task_data.test},
        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name=f"test_set.ES_US.f16kHz.dev_conversations_202411-v2.{seg_key}.ff_wer",
        train_dataset=task_data.train,
        train_epoch_split=train_partition_epoch,
        score_recog_output_func=partial(_score_recog_out_v2, corpora=corpora, remove_labels=spm_extra_symbols),
        recog_post_proc_funcs=[_spm_to_words],
        collect_score_results_func=_score_aggregate_es if aggregate_wer else None,
        spm=spm,
    )
    return task

def _score_aggregate_es(results: Dict[str, ScoreResult]) -> ScoreResultCollection:
    results_by_seg = {}
    from i6_experiments.users.zhang.experiments.apptek_exp_wer_ppl import seg_key
    for seg, (version, score_callback, spec) in itertools.product(
        [seg_key],
        [
            #("v1", spanish_mbw_metrics_v1, ES_MBW_TEST_SET_SPECS_V1),
            ("v2", spanish_mbw_metrics_v2, ES_MBW_TEST_SET_SPECS_V2),
        ],
    ):
        suffix = f".{seg}.ff_wer"
        if score_callback in [spanish_mbw_metrics_v2, spanish_mbw_metrics_v1]:
            weighed_results = score_callback(
                AbstractArtefactRepository(),
                ctms={
                    stripped_key: result.ctm
                    for k, result in results.items()
                    if k.endswith(suffix)
                    for stripped_key in [k.replace(suffix, "")]
                    if stripped_key in spec
                },
            )
            results_by_seg[f"{version}.{seg}"] = {
                "by_custom_weight": weighed_results,
            }  # noqa
        else:
            weighed_results, reflen_results = score_callback(
                AbstractArtefactRepository(),
                ctms={
                    stripped_key: result.ctm
                    for k, result in results.items()
                    if k.endswith(suffix)
                    for stripped_key in [k.replace(suffix, "")]
                    if stripped_key in spec
                },
            )
            results_by_seg[f"{version}.{seg}"] = {
                "by_custom_weight": weighed_results,
                "by_reflen": reflen_results,
            }
    return ScoreResultCollection(
        main_measure_value=ToVariableJob(
            results_by_seg[f"v2.{seg_key}"]["by_custom_weight"]["full_file_wer"]
        ).out,
        output=DumpAsJsonJob(
            {"all": {k: v.main_measure_value for k, v in results.items()}, "pooled": results_by_seg}
        ).out,
    )


def _score_recog_out_v2(
    dataset: DatasetConfig, recog_output: RecogOutput, corpora: Corpora, remove_labels: Optional[Sequence[str]] = None
) -> ScoreResult:
    """score"""
    corpus_name = dataset.get_main_name()
    eval_info = corpora.get_eval_info(corpus_name)

    search_ctm = SearchWordsToCTMJob(
        recog_words_file=recog_output.output, bliss_corpus=eval_info.segmented_corpus
    ).out_ctm_file

    # Need to remove one header field to make the count match w/ the fields in the file,
    # otherwise GLM remapping will drop all the contents
    search_ctm = PipelineJob(
        search_ctm,
        [
            "sed 's/<name> <track> <start> <duration> <word> <confidence> \[<n-best>\]/<name> <track> <start> <duration> <word> <confidence>/'"
        ],
        mini_task=True,
    ).out

    if remove_labels:
        replacements = "; ".join(f"s/{sep}//g" for sep in remove_labels)
        search_ctm = PipelineJob(search_ctm, [f"sed '{replacements}'"], mini_task=True).out

    scorer = eval_info.metrics
    scorer.run(
        aar=AbstractArtefactRepository(),
        ctm_path=search_ctm,
        extra_scorer_kwargs={"sctk_binary_path": tools_paths.get_sctk_binary_path()},
    )
    if eval_info.measure_type == WerMeasure.WER:
        scorer_name = "wer_scorer"
    elif eval_info.measure_type == WerMeasure.FF_WER:
        scorer_name = "full_file_wer_scorer"
    else:
        raise ValueError(f"unknown WER measure {eval_info.measure_type}")
    scorer_job = scorer.jobs[scorer_name]
    return ScoreResult(
        dataset_name=corpus_name,
        main_measure_value=scorer_job.out_wer,
        report=scorer_job.out_report_dir,
        score_job=scorer_job,
        ctm=search_ctm,
    )


def _spm_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    from i6_core.returnn.search import SearchOutputRawReplaceJob

    words = SearchOutputRawReplaceJob(bpe.output, [(" ", ""), ("▁", " ")], output_gzip=True).out_search_results
    return RecogOutput(output=words)
