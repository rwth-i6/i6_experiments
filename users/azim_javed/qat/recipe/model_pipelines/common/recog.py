__all__ = [
    "RecogResult",
    "OfflineRecogParameters",
    "StreamingRecogParameters",
    "recog_rasr_offline",
    "recog_rasr_streaming",
]

import json
import os
import pprint
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Iterator, List, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import ReturnnConfig
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.util import DelayedFormat
from i6_experiments.common.setups.serialization import Call, Collection, ExternalImport, Import
from i6_experiments.users.azim_javed.common.jobs.returnn import ResumableReturnnForwardJobV2
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import Tensor, TensorDict
from sisyphus import Job, Task, tk
from sisyphus.hash import sis_hash_helper

from ...data.base import DataConfig
from ...tools import rasr_binary_path, returnn_python_exe, returnn_root
from .corpus import ScorableCorpus
from .serializers import recipe_imports
from .pyscorers import register_pyscorers

# =============================
# ======== Dataclasses ========
# =============================


@dataclass
class RecogResult:
    descriptor: str
    corpus_name: str
    wer: Optional[tk.Variable]
    num_errors: Optional[tk.Variable]
    num_correct: Optional[tk.Variable]
    deletion: Optional[tk.Variable]
    insertion: Optional[tk.Variable]
    substitution: Optional[tk.Variable]
    ref_words: Optional[tk.Variable]
    search_error_rate: Optional[tk.Variable]
    model_error_rate: Optional[tk.Variable]
    correct_rate: Optional[tk.Variable]
    skipped_rate: Optional[tk.Variable]
    enc_rtf: Optional[tk.Variable]
    search_rtf: Optional[tk.Variable]
    total_rtf: Optional[tk.Variable]
    unstable_latency_stats: Optional[tk.Variable]
    stable_latency_stats: Optional[tk.Variable]
    step_hyps_stats: Optional[tk.Variable]
    step_word_end_hyps_stats: Optional[tk.Variable]
    step_trees_stats: Optional[tk.Variable]


@dataclass
class MemristorRecogResult:
    descriptor: str
    corpus_name: str
    average_wer: Optional[tk.Variable]
    stddev_wer: Optional[tk.Variable]
    min_wer: Optional[tk.Variable]
    max_wer: Optional[tk.Variable]
    average_total_rtf: Optional[tk.Variable]
    average_enc_rtf: Optional[tk.Variable]
    average_search_rtf: Optional[tk.Variable]


@dataclass
class OfflineRecogParameters:
    mem_rqmt: int = 16
    gpu_mem_rqmt: int = 0
    batched_decoder: bool = False
    dataloader_num_workers: int = 1
    use_legacy_dataloader_num_workers_0_hash: bool = False
    batch_size_seconds: int = 360
    num_shards: int = 1
    debug_joint: bool = False
    resumable: bool = False
    gpu_name: Optional[str] = None
    hash_salt: Optional[str] = None
    nbest_size: int = 0


@dataclass
class StreamingRecogParameters:
    encoder_frame_shift_seconds: float
    chunk_history_seconds: float = 10.0
    chunk_center_seconds: float = 1.0
    chunk_future_seconds: float = 1.0
    mem_rqmt: int = 16
    gpu_mem_rqmt: int = 0
    batched_decoder: bool = False


class _RecogReturnnConfig(ReturnnConfig):
    def __init__(self, *args, hash_dataloader_num_workers_as_zero: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self._hash_dataloader_num_workers_as_zero = hash_dataloader_num_workers_as_zero

    def _sis_hash(self):
        # Keep compatibility with recognition jobs that were hashed with zero workers.
        config = dict(self.config)
        if self._hash_dataloader_num_workers_as_zero:
            config["torch_dataloader_opts"] = {"num_workers": 0}
        hash_data = {
            "returnn_config": config,
            "python_epilog_hash": self.python_epilog_hash,
            "python_prolog_hash": self.python_prolog_hash,
        }
        if self.staged_network_dict:
            hash_data["returnn_networks"] = self.staged_network_dict
        return sis_hash_helper(hash_data)


def get_returrn_recog_job(
    checkpoint: PtCheckpoint,
    recog_returnn_config: _RecogReturnnConfig,
    returnn_python_exe: str,
    returnn_root: str,
    output_files: List[str],
    params: OfflineRecogParameters,
    descriptor: str,
    recog_corpus: ScorableCorpus,
    shard_num: Optional[int] = None,
):
    job_cls = ResumableReturnnForwardJobV2 if params.resumable else ReturnnForwardJobV2
    recog_job = job_cls(
        model_checkpoint=checkpoint,
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=output_files,
        device="gpu" if params.gpu_mem_rqmt > 0 else "cpu",
        mem_rqmt=params.mem_rqmt,
        time_rqmt=168,
    )
    recog_job.add_alias(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/shard_{shard_num}_{params.num_shards}"
        if shard_num is not None
        else f"recognition/{descriptor}/{recog_corpus.corpus_name}"
    )
    if params.gpu_mem_rqmt > 0:
        recog_job.rqmt["gpu_mem"] = params.gpu_mem_rqmt
        if params.gpu_name is not None:
            recog_job.rqmt["gpu_name"] = params.gpu_name

    if shard_num is not None:
        recog_job.set_keep_value(0)

    return recog_job


def recog_rasr_offline(
    descriptor: str,
    recog_rasr_config_file: tk.Path,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    encoder_serializers: Collection,
    sample_rate: int,
    params: OfflineRecogParameters,
    checkpoint: Optional[PtCheckpoint] = None,
    align_rasr_config_file: Optional[tk.Path] = None,
) -> RecogResult:
    compute_search_errors = align_rasr_config_file is not None
    output_files = ["search_out.ctm", "rtf.py", "rasr.recog.log"]

    if compute_search_errors:
        output_files += ["search_errors.py", "rasr.align.log"]

    if params.debug_joint:
        if params.num_shards > 1:
            raise ValueError("debug_joint is not supported with sharded recognition.")
        output_files += ["joint_activations_debug.npy"]

    model_outputs = {
        "ctm_string": {
            "dtype": "string",
            "feature_dim_axis": None,
            "time_dim_axis": None,
        },
        "audio_time": {
            "dtype": "float32",
            "feature_dim_axis": None,
            "time_dim_axis": None,
        },
        "enc_time": {
            "dtype": "float32",
            "feature_dim_axis": None,
            "time_dim_axis": None,
        },
        "search_time": {
            "dtype": "float32",
            "feature_dim_axis": None,
            "time_dim_axis": None,
        },
    }

    forward_step_kwargs = [
        ("recog_rasr_config_file", DelayedFormat('tk.Path("{}")', recog_rasr_config_file)),
        ("sample_rate", sample_rate),
    ]

    if params.nbest_size > 0:
        if params.resumable:
            raise ValueError("nbest decoding is not supported with resumable recognition.")
        output_files.append("nbest.json")
        model_outputs["nbest_json"] = {
            "dtype": "string",
            "feature_dim_axis": None,
            "time_dim_axis": None,
        }
        forward_step_kwargs.append(("nbest_size", params.nbest_size))
    if compute_search_errors:
        model_outputs.update(
            {
                "search_errors": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "model_errors": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "correct": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "skipped": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
            }
        )
        forward_step_kwargs.append(("align_rasr_config_file", DelayedFormat('tk.Path("{}")', align_rasr_config_file)))

    if params.resumable:
        model_outputs["resume_batch_end"] = {
            "dtype": "bool",
            "feature_dim_axis": None,
            "time_dim_axis": None,
        }
        forward_step_kwargs.append(("resumable", True))

    if params.batched_decoder:
        decoder_epilog = [
            Import(
                f"{OfflineSearchCallback.__module__}.{OfflineSearchCallback.__name__}",
                import_as="forward_callback",
            ),
            Import(
                f"{OfflineRasrRecogForwardStepV2.__module__}.{OfflineRasrRecogForwardStepV2.__name__}",
            ),
            Call(
                OfflineRasrRecogForwardStepV2.__name__,
                kwargs=forward_step_kwargs,
                return_assign_variables="forward_step",
            ),
            "print('batched_decoder_test')",
        ]
    else:
        decoder_epilog = [
            Import(
                f"{OfflineSearchCallback.__module__}.{OfflineSearchCallback.__name__}",
                import_as="forward_callback",
            ),
            Import(
                f"{OfflineRasrRecogForwardStep.__module__}.{OfflineRasrRecogForwardStep.__name__}",
            ),
            Call(
                OfflineRasrRecogForwardStep.__name__,
                kwargs=forward_step_kwargs,
                return_assign_variables="forward_step",
            ),
        ]
    if params.resumable:
        decoder_epilog.insert(
            1,
            Call("forward_callback", kwargs=[("resumable", True)], return_assign_variables="forward_callback"),
        )
    elif params.nbest_size:
        decoder_epilog.insert(
            1,
            Call("forward_callback", kwargs=[("nbest_size", params.nbest_size)], return_assign_variables="forward_callback"),
        )

    recog_config = {
        "extern_data": {
            "data": {"dim": 1, "dtype": "float32"},
            "raw": {"feature_dim_axis": None, "time_dim_axis": None, "dtype": "string"},
        },
        "model_outputs": model_outputs,
        "backend": "torch",
        "batch_size": params.batch_size_seconds * sample_rate,
    }
    if params.hash_salt is not None:
        recog_config["recognition_hash_salt"] = params.hash_salt
    recog_post_config = {}
    if params.dataloader_num_workers != 1:
        recog_post_config["torch_dataloader_opts"] = {"num_workers": params.dataloader_num_workers}

    if params.num_shards != 1:
        shard_segment_job = SegmentCorpusJob(
            bliss_corpus=recog_corpus.bliss_corpus_file, num_segments=params.num_shards
        )
        shard_recog_jobs = []
        for shard_num, segment_file in shard_segment_job.out_single_segment_files.items():
            shared_recog_data_config = replace(recog_data_config, segment_file=segment_file)
            shard_recog_returnn_config = _RecogReturnnConfig(
                config=recog_config,
                post_config=recog_post_config,
                hash_dataloader_num_workers_as_zero=(
                    params.dataloader_num_workers != 1 or params.use_legacy_dataloader_num_workers_0_hash
                ),
                python_prolog=recipe_imports + ["import torch"] + [ExternalImport(rasr_binary_path)],
                python_epilog=[
                    encoder_serializers,
                    Import("sisyphus.tk"),
                ]
                + decoder_epilog,  # type: ignore
                sort_config=False,
            )

            shard_recog_returnn_config.update(shared_recog_data_config.get_returnn_data("forward_data"))
            shard_recog_job = get_returrn_recog_job(
                checkpoint=checkpoint,
                recog_returnn_config=shard_recog_returnn_config,
                returnn_python_exe=returnn_python_exe,
                returnn_root=returnn_root,
                output_files=output_files,
                params=params,
                descriptor=descriptor,
                recog_corpus=recog_corpus,
                shard_num=shard_num,
            )
            shard_recog_jobs.append(shard_recog_job)

        recog_job = MergeShardedReturnnForwardJob(shard_recog_jobs)

    else:
        recog_returnn_config = _RecogReturnnConfig(
            config=recog_config,
            post_config=recog_post_config,
            hash_dataloader_num_workers_as_zero=(
                params.dataloader_num_workers != 1 or params.use_legacy_dataloader_num_workers_0_hash
            ),
            python_prolog=recipe_imports + ["import torch"] + [ExternalImport(rasr_binary_path)],
            python_epilog=[
                encoder_serializers,
                Import("sisyphus.tk"),
            ]
            + decoder_epilog,  # type: ignore
            sort_config=False,
        )

        recog_returnn_config.update(recog_data_config.get_returnn_data("forward_data"))

        recog_job = get_returrn_recog_job(
            checkpoint=checkpoint,
            recog_returnn_config=recog_returnn_config,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            output_files=output_files,
            params=params,
            descriptor=descriptor,
            recog_corpus=recog_corpus
        )

    for output_file in output_files:
        tk.register_output(
            f"recognition/{descriptor}/{recog_corpus.corpus_name}/{output_file}",
            recog_job.out_files[output_file],
        )

    if compute_search_errors:
        search_error_job = ExtractSearchErrorDataJob(recog_job.out_files["search_errors.py"])
        tk.register_output(
            f"recognition/{descriptor}/{recog_corpus.corpus_name}/search_error_rate",
            search_error_job.out_search_error_rate,
        )
        tk.register_output(
            f"recognition/{descriptor}/{recog_corpus.corpus_name}/model_error_rate",
            search_error_job.out_model_error_rate,
        )
        tk.register_output(
            f"recognition/{descriptor}/{recog_corpus.corpus_name}/correct_rate", search_error_job.out_correct_rate
        )
        tk.register_output(
            f"recognition/{descriptor}/{recog_corpus.corpus_name}/skipped_rate", search_error_job.out_skipped_rate
        )

    rtf_job = ExtractRTFDataJob(recog_job.out_files["rtf.py"])
    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/enc_rtf",
        rtf_job.out_enc_rtf,
    )
    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/search_rtf",
        rtf_job.out_search_rtf,
    )
    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/total_rtf",
        rtf_job.out_total_rtf,
    )

    score_job = recog_corpus.score_ctm(recog_job.out_files["search_out.ctm"])
    tk.register_output(f"recognition/{descriptor}/{recog_corpus.corpus_name}/scoring_reports", score_job.out_report_dir)

    extract_rasr_stats_job = ExtractRasrStatisticsJob(rasr_log_file=recog_job.out_files["rasr.recog.log"])

    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/search_space_stats/out_step_hyps",
        extract_rasr_stats_job.out_step_hyps,
    )
    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/search_space_stats/out_step_word_end_hyps",
        extract_rasr_stats_job.out_step_word_end_hyps,
    )
    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/search_space_stats/out_step_trees",
        extract_rasr_stats_job.out_step_trees,
    )

    return RecogResult(
        descriptor=descriptor,
        corpus_name=recog_corpus.corpus_name,
        wer=score_job.out_wer,
        num_correct=score_job.out_num_correct,
        num_errors=score_job.out_num_errors,
        deletion=score_job.out_percent_deletions,
        insertion=score_job.out_percent_insertions,
        substitution=score_job.out_percent_substitution,
        ref_words=score_job.out_ref_words,
        search_error_rate=search_error_job.out_search_error_rate if compute_search_errors else None,
        model_error_rate=search_error_job.out_model_error_rate if compute_search_errors else None,
        correct_rate=search_error_job.out_correct_rate if compute_search_errors else None,
        skipped_rate=search_error_job.out_skipped_rate if compute_search_errors else None,
        enc_rtf=rtf_job.out_enc_rtf,
        search_rtf=rtf_job.out_search_rtf,
        total_rtf=rtf_job.out_total_rtf,
        unstable_latency_stats=None,
        stable_latency_stats=None,
        step_hyps_stats=extract_rasr_stats_job.out_step_hyps,
        step_word_end_hyps_stats=extract_rasr_stats_job.out_step_word_end_hyps,
        step_trees_stats=extract_rasr_stats_job.out_step_trees,
    )


def recog_rasr_streaming(
    descriptor: str,
    recog_rasr_config_file: tk.Path,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    encoder_serializers: Collection,
    sample_rate: int,
    params: StreamingRecogParameters,
    checkpoint: Optional[PtCheckpoint] = None,
) -> RecogResult:
    recog_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
                "raw": {"feature_dim_axis": None, "time_dim_axis": None, "dtype": "string"},
            },
            "model_outputs": {
                "ctm_string": {
                    "dtype": "string",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "unstable_latencies": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                },
                "stable_latencies": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                },
            },
            "backend": "torch",
            "batch_size": 360 * sample_rate,
        },
        python_prolog=recipe_imports + [ExternalImport(rasr_binary_path)],
        python_epilog=[
            encoder_serializers,
            Import(
                f"{StreamingSearchCallback.__module__}.{StreamingSearchCallback.__name__}",
                import_as="forward_callback",
            ),
            Import(
                f"{StreamingRasrRecogForwardStep.__module__}.{StreamingRasrRecogForwardStep.__name__}",
            ),
            Import("sisyphus.tk"),
            Call(
                StreamingRasrRecogForwardStep.__name__,
                kwargs=[
                    ("recog_rasr_config_file", DelayedFormat('tk.Path("{}")', recog_rasr_config_file)),
                    ("sample_rate", sample_rate),
                    ("encoder_frame_shift_seconds", params.encoder_frame_shift_seconds),
                    ("chunk_history_seconds", params.chunk_history_seconds),
                    ("chunk_center_seconds", params.chunk_center_seconds),
                    ("chunk_future_seconds", params.chunk_future_seconds),
                ],
                return_assign_variables="forward_step",
            ),
        ],  # type: ignore
        sort_config=False,
    )

    recog_returnn_config.update(recog_data_config.get_returnn_data("forward_data"))

    output_files = ["search_out.ctm", "unstable_latencies.py", "stable_latencies.py", "rasr.recog.log"]

    recog_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=output_files,
        device="gpu" if params.gpu_mem_rqmt > 0 else "cpu",
        mem_rqmt=params.mem_rqmt,
        time_rqmt=168,
    )
    recog_job.add_alias(f"recognition/{descriptor}/{recog_corpus.corpus_name}")
    if params.gpu_mem_rqmt > 0:
        recog_job.rqmt["gpu_mem"] = params.gpu_mem_rqmt

    for output_file in output_files:
        tk.register_output(
            f"recognition/{descriptor}/{recog_corpus.corpus_name}/{output_file}",
            recog_job.out_files[output_file],
        )

    unstable_latency_stats = ExtractStatisticsJob(recog_job.out_files["unstable_latencies.py"]).out_stats
    stable_latency_stats = ExtractStatisticsJob(recog_job.out_files["stable_latencies.py"]).out_stats

    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/latencies/unstable_latency_stats", unstable_latency_stats
    )
    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/latencies/stable_latency_stats", stable_latency_stats
    )

    score_job = recog_corpus.score_ctm(recog_job.out_files["search_out.ctm"])
    tk.register_output(f"recognition/{descriptor}/{recog_corpus.corpus_name}/scoring_reports", score_job.out_report_dir)

    extract_space_stats_job = ExtractRasrStatisticsJob(rasr_log_file=recog_job.out_files["rasr.recog.log"])
    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/search_space_stats/out_step_hyps",
        extract_space_stats_job.out_step_hyps,
    )
    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/search_space_stats/out_step_word_end_hyps",
        extract_space_stats_job.out_step_word_end_hyps,
    )
    tk.register_output(
        f"recognition/{descriptor}/{recog_corpus.corpus_name}/search_space_stats/out_step_trees",
        extract_space_stats_job.out_step_trees,
    )

    return RecogResult(
        descriptor=descriptor,
        corpus_name=recog_corpus.corpus_name,
        wer=score_job.out_wer,
        deletion=score_job.out_percent_deletions,
        insertion=score_job.out_percent_insertions,
        substitution=score_job.out_percent_substitution,
        search_error_rate=None,
        model_error_rate=None,
        correct_rate=None,
        skipped_rate=None,
        enc_rtf=None,
        search_rtf=None,
        total_rtf=None,
        unstable_latency_stats=unstable_latency_stats,
        stable_latency_stats=stable_latency_stats,
        step_hyps_stats=extract_space_stats_job.out_step_hyps,
        step_word_end_hyps_stats=extract_space_stats_job.out_step_word_end_hyps,
        step_trees_stats=extract_space_stats_job.out_step_trees,
    )


def post_recog_memristor_offline(
    descriptor: str, recog_corpus: ScorableCorpus, recog_results: List[RecogResult]
) -> MemristorRecogResult:
    extract_stats_job = ExtractMemristorRecogStatsJob(
        wer=[result.wer for result in recog_results],
        num_errors=[result.num_errors for result in recog_results],
        ref_words=[result.ref_words for result in recog_results],
        total_rtf=[result.total_rtf for result in recog_results],
        enc_rtf=[result.enc_rtf for result in recog_results],
        search_rtf=[result.search_rtf for result in recog_results],
    )

    output_dir = f"recognition/{descriptor}/{recog_corpus.corpus_name}/memristor_stats"
    extract_stats_job.add_alias(f"recognition/memristor/{descriptor}/{recog_corpus.corpus_name}")
    tk.register_output(f"{output_dir}/average_wer", extract_stats_job.out_average_wer)
    tk.register_output(f"{output_dir}/stddev_wer", extract_stats_job.out_stddev_wer)
    tk.register_output(f"{output_dir}/min_wer", extract_stats_job.out_min_wer)
    tk.register_output(f"{output_dir}/max_wer", extract_stats_job.out_max_wer)
    tk.register_output(f"{output_dir}/average_total_rtf", extract_stats_job.out_average_total_rtf)
    tk.register_output(f"{output_dir}/average_enc_rtf", extract_stats_job.out_average_enc_rtf)
    tk.register_output(f"{output_dir}/average_search_rtf", extract_stats_job.out_average_search_rtf)

    return MemristorRecogResult(
        descriptor=descriptor,
        corpus_name=recog_corpus.corpus_name,
        average_wer=extract_stats_job.out_average_wer,
        stddev_wer=extract_stats_job.out_stddev_wer,
        min_wer=extract_stats_job.out_min_wer,
        max_wer=extract_stats_job.out_max_wer,
        average_total_rtf=extract_stats_job.out_average_total_rtf,
        average_enc_rtf=extract_stats_job.out_average_enc_rtf,
        average_search_rtf=extract_stats_job.out_average_search_rtf,
    )


# =============================
# ========== Helpers ==========
# =============================


class ExtractMemristorRecogStatsJob(Job):
    def __init__(
        self,
        wer: List[tk.Variable],
        num_errors: List[tk.Variable],
        ref_words: List[tk.Variable],
        total_rtf: List[tk.Variable],
        enc_rtf: List[tk.Variable],
        search_rtf: List[tk.Variable],
    ) -> None:
        self.wer = wer
        self.num_errors = num_errors
        self.ref_words = ref_words
        self.total_rtf = total_rtf
        self.enc_rtf = enc_rtf
        self.search_rtf = search_rtf
        self.out_average_wer = self.output_var("average_wer")
        self.out_stddev_wer = self.output_var("stddev_wer")
        self.out_min_wer = self.output_var("min_wer")
        self.out_max_wer = self.output_var("max_wer")
        self.out_average_total_rtf = self.output_var("average_total_rtf")
        self.out_average_enc_rtf = self.output_var("average_enc_rtf")
        self.out_average_search_rtf = self.output_var("average_search_rtf")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        wer_values = [
            100 * float(num_errors.get()) / float(ref_words.get())
            for wer, num_errors, ref_words in zip(self.wer, self.num_errors, self.ref_words)
            if wer.get() is not None
        ]

        total_rtf_values = [float(value.get()) for value in self.total_rtf]
        enc_rtf_values = [float(value.get()) for value in self.enc_rtf]
        search_rtf_values = [float(value.get()) for value in self.search_rtf]

        average_wer = float(np.mean(wer_values))
        stddev_wer = float(np.std(wer_values))
        min_wer = float(np.min(wer_values))
        max_wer = float(np.max(wer_values))

        average_total_rtf = float(np.mean(total_rtf_values))
        average_enc_rtf = float(np.mean(enc_rtf_values))
        average_search_rtf = float(np.mean(search_rtf_values))

        self.out_average_wer.set(average_wer)
        self.out_stddev_wer.set(stddev_wer)
        self.out_min_wer.set(min_wer)
        self.out_max_wer.set(max_wer)
        self.out_average_wer.set(average_wer)
        self.out_stddev_wer.set(stddev_wer)
        self.out_average_total_rtf.set(average_total_rtf)
        self.out_average_enc_rtf.set(average_enc_rtf)
        self.out_average_search_rtf.set(average_search_rtf)


class ExtractSearchErrorDataJob(Job):
    def __init__(self, search_errors_file: tk.Path) -> None:
        self.search_errors_file = search_errors_file
        self.out_total_skipped = self.output_var("total_skipped")
        self.out_skipped_rate = self.output_var("skipped_rate")
        self.out_total_correct = self.output_var("total_correct")
        self.out_correct_rate = self.output_var("correct_rate")
        self.out_total_search_errors = self.output_var("total_search_errors")
        self.out_search_error_rate = self.output_var("search_error_rate")
        self.out_total_model_errors = self.output_var("total_model_errors")
        self.out_model_error_rate = self.output_var("model_error_rate")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.search_errors_file, "rt") as f:
            d = eval(f.read())

        self.out_total_skipped.set(d["total_skipped"])
        self.out_skipped_rate.set(d["skipped_rate"])
        self.out_total_correct.set(d["total_correct"])
        self.out_correct_rate.set(d["correct_rate"])
        self.out_total_search_errors.set(d["total_search_errors"])
        self.out_search_error_rate.set(d["search_error_rate"])
        self.out_total_model_errors.set(d["total_model_errors"])
        self.out_model_error_rate.set(d["model_error_rate"])


class ExtractRTFDataJob(Job):
    def __init__(self, rtf_file: tk.Path) -> None:
        self.rtf_file = rtf_file
        self.out_enc_rtf = self.output_var("enc_rtf")
        self.out_search_rtf = self.output_var("search_rtf")
        self.out_total_rtf = self.output_var("total_rtf")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.rtf_file, "rt") as f:
            d = eval(f.read())

        self.out_enc_rtf.set(d["enc_rtf"])
        self.out_search_rtf.set(d["search_rtf"])
        self.out_total_rtf.set(d["total_rtf"])


class ExtractStatisticsJob(Job):
    def __init__(self, stat_file: tk.Path) -> None:
        self.stat_file = stat_file
        self.out_stats = self.output_var(
            "statistics", backup={"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "p100": 0.0}
        )

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.stat_file, "rt") as f:
            d = eval(f.read())
        self.out_stats.set(d)


def _open_rasr_log(path, mode="rt"):
    import gzip

    with open(path, "rb") as f:
        compressed = f.read(2) == b"\x1f\x8b"
    return gzip.open(path, mode) if compressed else open(path, mode)


class ExtractRasrStatisticsJob(Job):
    def __init__(self, rasr_log_file: tk.Path) -> None:
        self.rasr_log_file = rasr_log_file

        self.out_step_hyps = self.output_var(
            "step_hyps", backup={"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "p100": 0.0}
        )
        self.out_step_word_end_hyps = self.output_var(
            "step_word_end_hyps", backup={"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "p100": 0.0}
        )
        self.out_step_trees = self.output_var(
            "step_trees", backup={"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "p100": 0.0}
        )

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        import re

        step_hyps_counts = []
        step_word_end_hyps_counts = []
        step_trees_counts = []

        with _open_rasr_log(self.rasr_log_file.get(), "rt") as f:
            for line in f:
                for log_str, counter in [
                    ("<num-hyps-after-beam-pruning-1>", step_hyps_counts),
                    ("<num-word-end-hyps-after-beam-pruning>", step_word_end_hyps_counts),
                    ("<num-active-trees>", step_trees_counts),
                ]:
                    if log_str in line:
                        value = int(re.split(r"[<>]+", line)[2].strip())
                        counter.append(value)

        self.out_step_hyps.set(statistics_from_data(step_hyps_counts))
        self.out_step_word_end_hyps.set(statistics_from_data(step_word_end_hyps_counts))
        self.out_step_trees.set(statistics_from_data(step_trees_counts))


class MergeShardedReturnnForwardJob(Job):
    def __init__(self, sharded_forward_jobs: List[ReturnnForwardJobV2]) -> None:
        assert sharded_forward_jobs
        self.shard_files = [job.out_files for job in sharded_forward_jobs]
        self.out_files = {
            name: self.output_path(name)
            for name in self.shard_files[0]
            if name in {"search_out.ctm", "rtf.py", "rasr.recog.log", "search_errors.py", "rasr.align.log", "nbest.json"}
        }
        assert all(set(files) == set(self.shard_files[0]) for files in self.shard_files)

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        import ast
        import gzip
        import shutil

        for name in ("search_out.ctm", "rasr.recog.log", "rasr.align.log"):
            if name not in self.out_files:
                continue
            output_opener = gzip.open if name in ("rasr.recog.log", "rasr.align.log") else open
            output_kwargs = {"compresslevel": 6} if output_opener is gzip.open else {}
            with output_opener(self.out_files[name].get_path(), "wb", **output_kwargs) as merged_file:
                for files in self.shard_files:
                    opener = _open_rasr_log if name in ("rasr.recog.log", "rasr.align.log") else open
                    with opener(files[name].get_path(), "rb") as shard_file:
                        shutil.copyfileobj(shard_file, merged_file)

        if "nbest.json" in self.out_files:
            seen = set()
            with open(self.out_files["nbest.json"].get_path(), "w") as merged_file:
                merged_file.write("{\n")
                for files in self.shard_files:
                    with open(files["nbest.json"].get_path()) as shard_file:
                        records = json.load(shard_file)
                    for seq_tag, record in records.items():
                        if seq_tag in seen:
                            raise ValueError(f"Duplicate N-best utterance across shards: {seq_tag}")
                        if seen:
                            merged_file.write(",\n")
                        merged_file.write(json.dumps(seq_tag) + ": ")
                        json.dump(record, merged_file, allow_nan=False)
                        seen.add(seq_tag)
                merged_file.write("\n}\n")

        totals = {"audio_seconds": 0.0, "enc_seconds": 0.0, "search_seconds": 0.0}
        for files in self.shard_files:
            with open(files["rtf.py"].get_path(), "rt") as shard_file:
                data = ast.literal_eval(shard_file.read())
            for key in totals:
                totals[key] += data[key]

        audio = totals["audio_seconds"]
        enc = totals["enc_seconds"]
        search = totals["search_seconds"]
        if audio <= 0:
            raise ValueError("Cannot compute RTF for shards with zero total audio duration")
        rtf_data = {
            **totals,
            "enc_rtf": enc / audio,
            "enc_rtfx": audio / enc if enc else 0.0,
            "search_rtf": search / audio,
            "search_rtfx": audio / search if search else 0.0,
            "total_seconds": enc + search,
            "total_rtf": (enc + search) / audio,
            "total_rtfx": audio / (enc + search) if enc + search else 0.0,
        }
        with open(self.out_files["rtf.py"].get_path(), "wt") as merged_file:
            merged_file.write(pprint.pformat(rtf_data))

        if "search_errors.py" in self.out_files:
            counts = {
                "total_skipped": 0,
                "total_correct": 0,
                "total_search_errors": 0,
                "total_model_errors": 0,
                "total_seqs": 0,
            }
            for files in self.shard_files:
                with open(files["search_errors.py"].get_path(), "rt") as shard_file:
                    data = ast.literal_eval(shard_file.read())
                for key in counts:
                    counts[key] += data[key]
            total_seqs = counts["total_seqs"]
            if total_seqs <= 0:
                raise ValueError("Cannot compute search-error rates for zero sequences")
            search_error_data = {
                **counts,
                "skipped_rate": counts["total_skipped"] / total_seqs,
                "correct_rate": counts["total_correct"] / total_seqs,
                "search_error_rate": counts["total_search_errors"] / total_seqs,
                "model_error_rate": counts["total_model_errors"] / total_seqs,
            }
            with open(self.out_files["search_errors.py"].get_path(), "wt") as merged_file:
                merged_file.write(pprint.pformat(search_error_data))



class TracebackItem(Protocol):
    lemma: str
    am_score: float
    lm_score: float
    confidence_score: Optional[float]
    start_time: int
    end_time: int


def _samples_to_frames(n_samples: int, sample_rate: int, frame_shift_seconds: float) -> int:
    return int(np.round(n_samples / (sample_rate * frame_shift_seconds)))


def _frames_to_seconds(n_frames: int, frame_shift_seconds: float) -> float:
    return float(n_frames) * frame_shift_seconds


def _seconds_to_samples(n_seconds: float, sample_rate: int) -> int:
    return int(n_seconds * sample_rate)


def _samples_to_seconds(n_samples: int, sample_rate: int) -> float:
    return float(n_samples) / sample_rate


def _traceback_to_string(traceback: List[TracebackItem]) -> str:
    traceback_str = " ".join(item.lemma for item in traceback)
    traceback_str = traceback_str.replace("<s>", "")
    traceback_str = traceback_str.replace("</s>", "")
    traceback_str = traceback_str.replace("<blank>", "")
    traceback_str = traceback_str.replace("[BLANK] [1]", "")
    traceback_str = traceback_str.replace("[BLANK]", "")
    traceback_str = traceback_str.replace("<silence>", "")
    traceback_str = traceback_str.replace("[SILENCE]", "")
    traceback_str = traceback_str.replace("[SENTENCE-END]", "")
    traceback_str = " ".join(traceback_str.split())
    return traceback_str


def _traceback_to_transcription(traceback: List[TracebackItem]) -> str:
    traceback_str = _traceback_to_string(traceback)
    return traceback_str.replace("@@ ", "")


def _traceback_to_ctm_str(traceback: List[TracebackItem], ms_per_frame: int) -> str:
    lines = []
    start_frame: Optional[int] = None
    current_word = ""
    confidences = []
    for item in traceback:
        if hasattr(item, "confidence_score"):
            if item.confidence_score is not None:
                for _ in range(item.start_time, item.end_time):
                    confidences.append(item.confidence_score)
        else:
            for _ in range(item.start_time, item.end_time):
                confidences.append(0.99)
        if item.lemma.startswith("[") and item.lemma.endswith("]"):
            continue
        if item.lemma.startswith("<") and item.lemma.endswith(">"):
            continue
        if start_frame is None:
            start_frame = item.start_time
        current_word += item.lemma.replace("@@", "")
        if item.lemma.endswith("@@"):
            continue
        duration_frames = item.end_time - start_frame

        start_time = start_frame * 0.001 * ms_per_frame
        duration_time = duration_frames * 0.001 * ms_per_frame
        if len(confidences) == 0:
            avg_confidence = 0.99
        else:
            avg_confidence = sum(confidences) / len(confidences)
        lines.append(f"[REC_NAME] 1 {start_time:.3f} {duration_time:.3f} {current_word} {avg_confidence:.2f}")
        start_frame = None
        current_word = ""
        confidences.clear()

    if len(lines) == 0:
        return "[REC_NAME] 1 0.000 0.010 <empty-sequence> 0.99\n"
    else:
        return "\n".join(lines) + "\n"


def _traceback_to_score(traceback: List[TracebackItem]) -> float:
    if len(traceback) > 0:
        return traceback[-1].am_score + traceback[-1].lm_score
    else:
        return float("inf")


def statistics_from_data(data: list[float]) -> dict[str, float]:
    if len(data) == 0:
        avg = 0.0
        p50 = 0.0
        p90 = 0.0
        p99 = 0.0
        p100 = 0.0
    else:
        data_array = np.array(data)
        avg = np.average(data_array).astype(float)
        p50 = np.percentile(data_array, 50).astype(float)
        p90 = np.percentile(data_array, 90).astype(float)
        p99 = np.percentile(data_array, 99).astype(float)
        p100 = np.percentile(data_array, 100).astype(float)

    return {
        "avg": avg,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "p100": p100,
    }


# =======================================================
# ========== Callbacks for result file writing ==========
# =======================================================


def _load_forward_cache():
    if os.path.exists("forward_cache.json"):
        with open("forward_cache.json") as f:
            return json.load(f)
    return {}


class SearchCallback(ForwardCallbackIface):
    def init(self, **_):
        self.ctm_file = open("search_out.ctm", "w")
        self.ctm_file.write(";; <name> <track> <start> <duration> <word> <confidence>\n")

        self.total_skipped = 0
        self.total_correct = 0
        self.total_search_errors = 0
        self.total_model_errors = 0

        self.total_seqs = 0

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        self.total_seqs += 1

        raw_outputs = outputs.as_raw_tensor_dict()
        ctm_str = raw_outputs.get("ctm_string")
        assert ctm_str is not None
        ctm_str = ctm_str.item()
        rec_name = seq_tag.split("/")[1]
        self.ctm_file.write(f";; {seq_tag}\n")
        self.ctm_file.write(ctm_str.replace("[REC_NAME]", rec_name))

        self.total_skipped += raw_outputs.get("skipped", 0)
        self.total_correct += raw_outputs.get("correct", 0)
        self.total_search_errors += raw_outputs.get("search_errors", 0)
        self.total_model_errors += raw_outputs.get("model_errors", 0)

    def finish(self):
        self.ctm_file.close()

        if self.total_skipped + self.total_correct + self.total_search_errors + self.total_model_errors > 0:
            search_error_data = {
                "total_skipped": int(self.total_skipped),
                "skipped_rate": float(self.total_skipped) / float(self.total_seqs),
                "total_correct": int(self.total_correct),
                "correct_rate": float(self.total_correct) / float(self.total_seqs),
                "total_search_errors": int(self.total_search_errors),
                "search_error_rate": float(self.total_search_errors) / float(self.total_seqs),
                "total_model_errors": int(self.total_model_errors),
                "model_error_rate": float(self.total_model_errors) / float(self.total_seqs),
                "total_seqs": int(self.total_seqs),
            }
            print()
            print(
                f"Total search errors: {search_error_data['total_search_errors']} ({search_error_data['search_error_rate']:.2f}%)"
            )
            print(
                f"Total model errors: {search_error_data['total_model_errors']} ({search_error_data['model_error_rate']:.2f}%)"
            )
            print(
                f"Total correct seqs: {search_error_data['total_correct']} ({search_error_data['correct_rate']:.2f}%)"
            )
            print(f"Total skipped: {search_error_data['total_skipped']} ({search_error_data['skipped_rate']:.2f}%)")
            print()

            with open("search_errors.py", "w") as search_error_file:
                search_error_file.write(pprint.pformat(search_error_data))


class OfflineSearchCallback(SearchCallback):
    def __init__(self, *, resumable: bool = False, nbest_size: int = 0):
        self.resumable = resumable
        self.nbest_size = nbest_size

    def __getstate__(self):
        return {"resumable": self.resumable, "nbest_size": self.nbest_size}

    def init(self, *, model):
        super().init(model=model)

        self.total_audio_time = 0
        self.total_enc_time = 0
        self.total_search_time = 0
        if self.resumable:
            self._output_cache = _load_forward_cache()
            self._cache_dirty = False
        if self.nbest_size:
            self.nbest_file = open("nbest.json", "w")
            self.nbest_file.write("{\n")
            self._nbest_first = True

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        super().process_seq(seq_tag=seq_tag, outputs=outputs)
        raw_outputs = outputs.as_raw_tensor_dict()

        self.total_audio_time += raw_outputs.get("audio_time", 0)
        self.total_enc_time += raw_outputs.get("enc_time", 0)
        self.total_search_time += raw_outputs.get("search_time", 0)

        if self.nbest_size:
            if not self._nbest_first:
                self.nbest_file.write(",\n")
            self.nbest_file.write(json.dumps(seq_tag) + ": " + raw_outputs["nbest_json"].item())
            self._nbest_first = False

        if self.resumable:
            if seq_tag not in self._output_cache:
                self._output_cache[seq_tag] = {
                    name: value.item()
                    for name, value in raw_outputs.items()
                    if name != "resume_batch_end"
                }
                self._cache_dirty = True
            if bool(raw_outputs["resume_batch_end"].item()):
                self.ctm_file.flush()
                self._save_cache()

    def _save_cache(self):
        if not self._cache_dirty:
            return

        with open("forward_cache.json.tmp", "w") as f:
            json.dump(self._output_cache, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace("forward_cache.json.tmp", "forward_cache.json")
        self._cache_dirty = False

    def finish(self):
        super().finish()

        if self.nbest_size:
            self.nbest_file.write("\n}\n")
            self.nbest_file.close()

        rtf_data = {
            "audio_seconds": float(self.total_audio_time),
            "enc_seconds": float(self.total_enc_time),
            "enc_rtf": float(self.total_enc_time) / float(self.total_audio_time),
            "enc_rtfx": float(self.total_audio_time) / float(self.total_enc_time),
            "search_seconds": float(self.total_search_time),
            "search_rtf": float(self.total_search_time) / float(self.total_audio_time),
            "search_rtfx": float(self.total_audio_time) / float(self.total_search_time),
            "total_seconds": float(self.total_enc_time) + float(self.total_search_time),
            "total_rtf": float(self.total_enc_time + self.total_search_time) / float(self.total_audio_time),
            "total_rtfx": float(self.total_audio_time) / float(self.total_enc_time + self.total_search_time),
        }

        print(
            f"Total encoder time: {self.total_enc_time:.2f} seconds, AM-RTF: {rtf_data['enc_rtf']}, XRTF: {rtf_data['enc_rtfx']}"
        )
        print(
            f"Total search time: {self.total_search_time:.2f} seconds, search-RTF: {rtf_data['search_rtf']}, RTFX: {rtf_data['search_rtfx']}"
        )
        print(
            f"Total time: {rtf_data['total_seconds']:.2f} seconds, RTF: {rtf_data['total_rtf']}, RTFX: {rtf_data['total_rtfx']}"
        )
        print()

        with open("rtf.py", "w") as rtf_file:
            rtf_file.write(pprint.pformat(rtf_data))


class StreamingSearchCallback(SearchCallback):
    def init(self, *, model):
        super().init(model=model)

        self.unstable_latencies = []
        self.stable_latencies = []

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        super().process_seq(seq_tag=seq_tag, outputs=outputs)

        raw_outputs = outputs.as_raw_tensor_dict()

        self.unstable_latencies.extend(raw_outputs.get("unstable_latencies", []))
        self.stable_latencies.extend(raw_outputs.get("stable_latencies", []))

    def finish(self):
        super().finish()

        unstable_latency_stats = statistics_from_data(self.unstable_latencies)
        stable_latency_stats = statistics_from_data(self.stable_latencies)

        print(f"Unstable latencies: {', '.join(f'{key}={val:.2f}' for key, val in unstable_latency_stats.items())}")
        print(f"Stable latencies: {', '.join(f'{key}={val:.2f}' for key, val in stable_latency_stats.items())}")

        with open("unstable_latencies.py", "w") as latency_file:
            latency_file.write(pprint.pformat(unstable_latency_stats))

        with open("stable_latencies.py", "w") as latency_file:
            latency_file.write(pprint.pformat(stable_latency_stats))


# ==========================================
# ========== Forward step classes ==========
# ==========================================


class EncoderModel(Protocol):
    def forward(
        self, audio_samples: torch.Tensor, audio_samples_size: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: ...


class RasrRecogForwardStep(ABC):
    def __init__(
        self,
        recog_rasr_config_file: tk.Path,
        align_rasr_config_file: Optional[tk.Path] = None,
        sample_rate: int = 16000,
        *,
        resumable: bool = False,
        nbest_size: int = 0,
    ) -> None:
        self.resumable = resumable
        self.nbest_size = nbest_size
        self._output_cache = _load_forward_cache() if resumable else {}
        self.recog_rasr_config_file = recog_rasr_config_file
        self.sample_rate = sample_rate
        self.search_algorithm = self.init_search_algorithm()

        self.align_rasr_config_file = align_rasr_config_file
        self.aligner = self.init_aligner()

    def init_search_algorithm(self):
        if self.recog_rasr_config_file is None:
            return None

        register_pyscorers()

        from librasr import Configuration, SearchAlgorithm

        config = Configuration()
        config.set_from_file(self.recog_rasr_config_file.get())

        return SearchAlgorithm(config=config)

    def init_aligner(self):
        if self.align_rasr_config_file is None:
            return None

        from librasr import Aligner, Configuration

        config = Configuration()
        config.set_from_file(self.align_rasr_config_file.get())

        return Aligner(config=config)

    # Need to remove `search_algorithm` and `aligner` since they are non-picklable pybind objects from librasr
    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["search_algorithm"] = None
        d["aligner"] = None

        return d

    # Recreate `search_algorithm` and `aligner` object from scratch since it's not picklable
    def __setstate__(self, d) -> None:
        self.__dict__ = d
        self.search_algorithm = self.init_search_algorithm()
        self.aligner = self.init_aligner()

    @abstractmethod
    def __call__(self, *, model: EncoderModel, extern_data: TensorDict, **_) -> None: ...

    def _get_nbest_json(self, features, reference):
        candidates = self.search_algorithm.get_current_n_best_list(self.nbest_size)
        record = {
            "reference": str(reference),
            "encoder_frames": int(features.shape[-2]),
            "rasr_config": self.recog_rasr_config_file.get(),
            "requested_nbest": self.nbest_size,
            "score_source": "last traceback item's cumulative am_score + lm_score; not certified complete",
            "candidates": [
                {
                    "rank": rank,
                    "transcription": _traceback_to_transcription(candidate),
                    "traceback_terminal_cost": _traceback_to_score(candidate) if candidate else None,
                    "traceback": [
                        {
                            "lemma": item.lemma,
                            "am_score": float(item.am_score),
                            "lm_score": float(item.lm_score),
                            "confidence_score": (
                                float(item.confidence_score) if item.confidence_score is not None else None
                            ),
                            "start_time": int(item.start_time),
                            "end_time": int(item.end_time),
                        }
                        for item in candidate
                    ],
                }
                for rank, candidate in enumerate(candidates, 1)
            ],
        }
        return json.dumps(record, allow_nan=False)

    def _mark_nbest_output(self, records):
        if not self.nbest_size:
            return
        import returnn.frontend as rf

        rf.get_run_ctx().mark_as_output(
            Tensor(
                name="nbest_json",
                dtype="string",
                raw_tensor=np.array(records, dtype="U"),
                feature_dim_axis=None,
                time_dim_axis=None,
            ),
            name="nbest_json",
        )
    
    
    def _prepare_resume_batch(self, raw_data):
        seq_tags = raw_data["seq_tag"]
        pending = [i for i, tag in enumerate(seq_tags) if tag not in self._output_cache]
        if pending and len(pending) != len(seq_tags):
            raw_data = {
                "data": raw_data["data"][pending],
                "data:size1": raw_data["data:size1"][pending],
                "raw": [raw_data["raw"][i] for i in pending],
                "seq_tag": [seq_tags[i] for i in pending],
            }
        return raw_data, seq_tags, pending

    def _emit_resume_outputs(self, seq_tags, pending, computed_outputs):
        import returnn.frontend as rf

        computed = {
            seq_tags[index]: {name: values[i] for name, values in computed_outputs.items()}
            for i, index in enumerate(pending)
        }
        rows = [computed[tag] if tag in computed else self._output_cache[tag] for tag in seq_tags]
        dtypes = {
            "ctm_string": "string",
            "enc_time": "float32",
            "audio_time": "float32",
            "search_time": "float32",
        }
        if self.aligner is not None:
            dtypes.update({name: "int32" for name in ("search_errors", "model_errors", "skipped", "correct")})

        run_ctx = rf.get_run_ctx()
        for name, dtype in dtypes.items():
            run_ctx.mark_as_output(
                Tensor(
                    name=name,
                    dtype=dtype,
                    raw_tensor=np.array([row[name] for row in rows], dtype="U" if dtype == "string" else dtype),
                    feature_dim_axis=None,
                    time_dim_axis=None,
                ),
                name=name,
            )
        self._mark_resume_batch_end(run_ctx, len(seq_tags))

    def _mark_resume_batch_end(self, run_ctx, batch_size: int):
        if self.resumable:
            run_ctx.mark_as_output(
                Tensor(
                    name="resume_batch_end",
                    dtype="bool",
                    raw_tensor=np.arange(batch_size) == batch_size - 1,
                    feature_dim_axis=None,
                    time_dim_axis=None,
                ),
                name="resume_batch_end",
            )


class OfflineRasrRecogForwardStep(RasrRecogForwardStep):
    def __call__(self, *, model: EncoderModel, extern_data: TensorDict, **_) -> None:
        assert self.search_algorithm is not None

        raw_data = extern_data.as_raw_tensor_dict()
        if self.resumable:
            raw_data, original_seq_tags, pending = self._prepare_resume_batch(raw_data)
            if not pending:
                self._emit_resume_outputs(original_seq_tags, pending, {})
                return
        audio_samples = raw_data["data"]
        audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
        orths = raw_data["raw"]
        seq_tags = raw_data["seq_tag"]

        ctm_strs = []

        nbest_records = []

        search_errors = []
        model_errors = []
        skipped = []
        correct = []

        encoder_times = []
        search_times = []

        for b in range(audio_samples.size(0)):
            seq_samples_size = audio_samples_size[b : b + 1]
            seq_samples = audio_samples[b : b + 1, : seq_samples_size[0]]  # [1, T, 1]

            encoder_start = perf_counter()

            model_output = model.forward(seq_samples, seq_samples_size)
            if isinstance(model_output, (tuple, list)):
                encoder_states, _ = model_output
            else:
                encoder_states = model_output

            encoder_time = perf_counter() - encoder_start
            encoder_times.append(encoder_time)

            ms_per_enc_frame = round((seq_samples_size.item() / encoder_states.size(1)) / self.sample_rate * 1000)

            encoder_states = encoder_states.to(device="cpu")

            search_start = perf_counter()

            traceback = self.search_algorithm.recognize_segment(features=encoder_states)

            if self.nbest_size:
                nbest_records.append(self._get_nbest_json(encoder_states, orths[b]))

            search_time = perf_counter() - search_start
            search_times.append(search_time)

            seq_time = _samples_to_seconds(seq_samples_size[0], self.sample_rate)

            print(f"Recognized sequence {repr(seq_tags[b])}")
            print(f'    Ground truth: "{orths[b]}"', flush=True)
            print(f'    Recognized: "{_traceback_to_transcription(traceback)}"')
            print("    Traceback:")
            for item in traceback:
                # if item.lemma.startswith("<") or item.lemma.startswith("["):
                #     continue
                print(f"        {repr(item)}")
            print(
                f"    Encoder time: {encoder_time:.3f} seconds, RTF {encoder_time / seq_time:.3f}, XRTF {seq_time / encoder_time:.3f}"
            )
            print(
                f"    Search time: {search_time:.3f} seconds, RTF {search_time / seq_time:.3f}, XRTF {seq_time / search_time:.3f}"
            )
            print()

            ctm_strs.append(_traceback_to_ctm_str(traceback, ms_per_enc_frame))

            if self.aligner is not None:
                align_traceback = self.aligner.align_segment(features=encoder_states, orth=orths[b] + " ")
                recog_transcription = _traceback_to_transcription(traceback)
                align_transcription = _traceback_to_transcription(align_traceback)

                recog_score = _traceback_to_score(traceback)
                align_score = _traceback_to_score(align_traceback)

                if align_transcription != orths[b]:
                    print("    Could not successfully compute forced alignment. Transcription may contain OOV words.")
                    skipped.append(1)
                    search_errors.append(0)
                    model_errors.append(0)
                    correct.append(0)
                elif recog_transcription == orths[b]:
                    print("    Correct transcription found.")
                    skipped.append(0)
                    search_errors.append(0)
                    model_errors.append(0)
                    correct.append(1)
                elif align_score < recog_score:
                    print(
                        f"    Encountered search error. Forced alignment has score {align_score} while search has score {recog_score}"
                    )
                    skipped.append(0)
                    search_errors.append(1)
                    model_errors.append(0)
                    correct.append(0)
                else:
                    print(
                        f"    Encountered model error. Forced alignment has score {align_score} while search has score {recog_score}"
                    )
                    skipped.append(0)
                    search_errors.append(0)
                    model_errors.append(1)
                    correct.append(0)

        if self.resumable:
            computed_outputs = {
                "ctm_string": ctm_strs,
                "enc_time": encoder_times,
                "audio_time": (audio_samples_size / self.sample_rate).cpu().numpy(),
                "search_time": search_times,
            }
            if self.aligner is not None:
                computed_outputs.update(
                    search_errors=search_errors, model_errors=model_errors, skipped=skipped, correct=correct,
                )
            self._emit_resume_outputs(original_seq_tags, pending, computed_outputs)
            return

        import returnn.frontend as rf

        run_ctx = rf.get_run_ctx()

        ctm_string_tensor = Tensor(
            name="ctm_string",
            dtype="string",
            raw_tensor=np.array(ctm_strs, dtype="U"),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(ctm_string_tensor, name="ctm_string")
        self._mark_nbest_output(nbest_records)

        if len(search_errors) > 0:
            search_errors_tensor = Tensor(
                name="search_errors",
                dtype="int32",
                raw_tensor=np.array(search_errors, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(search_errors_tensor, name="search_errors")

            model_errors_tensor = Tensor(
                name="model_errors",
                dtype="int32",
                raw_tensor=np.array(model_errors, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(model_errors_tensor, name="model_errors")

            skipped_tensor = Tensor(
                name="skipped",
                dtype="int32",
                raw_tensor=np.array(skipped, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(skipped_tensor, name="skipped")

            correct_tensor = Tensor(
                name="correct",
                dtype="int32",
                raw_tensor=np.array(correct, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(correct_tensor, name="correct")

        enc_time_tensor = Tensor(
            name="enc_time",
            dtype="float32",
            raw_tensor=np.array(encoder_times, dtype=np.float32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(enc_time_tensor, name="enc_time")

        audio_time_tensor = Tensor(
            name="audio_time",
            dtype="float32",
            raw_tensor=audio_samples_size / self.sample_rate,
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(audio_time_tensor, name="audio_time")

        search_time_tensor = Tensor(
            name="search_time",
            dtype="float32",
            raw_tensor=np.array(search_times, dtype=np.float32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(search_time_tensor, name="search_time")


class OfflineRasrRecogForwardStepV2(RasrRecogForwardStep):
    def __call__(self, *, model: EncoderModel, extern_data: TensorDict, **_) -> None:
        assert self.search_algorithm is not None

        raw_data = extern_data.as_raw_tensor_dict()
        if self.resumable:
            raw_data, original_seq_tags, pending = self._prepare_resume_batch(raw_data)
            if not pending:
                self._emit_resume_outputs(original_seq_tags, pending, {})
                return
        audio_samples = raw_data["data"]
        audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
        orths = raw_data["raw"]
        seq_tags = raw_data["seq_tag"]

        B = audio_samples.size(0)

        ctm_strs = []
        nbest_records = []

        search_errors = []
        model_errors = []
        skipped = []
        correct = []

        encoder_start = perf_counter()
        encoder_states, encoder_states_len = model.forward(audio_samples, audio_samples_size)  # [B, T, V]
        encoder_time = perf_counter() - encoder_start
        encoder_states = encoder_states.to(device="cpu")

        enc_time_per_seq = encoder_time / B
        encoder_times = [enc_time_per_seq for _ in range(B)]
        search_times = []

        for b in range(B):
            seq_samples_size_b = audio_samples_size[b]
            T_b = int(encoder_states_len[b])
            es = encoder_states[b : b + 1, :T_b]
            ms_per_enc_frame = round(seq_samples_size_b.item() / T_b / self.sample_rate * 1000)  # per-seq, like V1

            seq_time = _samples_to_seconds(seq_samples_size_b, self.sample_rate)

            search_start = perf_counter()

            traceback = self.search_algorithm.recognize_segment(features=es)

            if self.nbest_size:
                nbest_records.append(self._get_nbest_json(es, orths[b]))
            
            search_time = perf_counter() - search_start
            search_times.append(search_time)

            print(f"Recognized sequence {repr(seq_tags[b])}")
            print(f'    Ground truth: "{orths[b]}"', flush=True)
            print(f'    Recognized: "{_traceback_to_transcription(traceback)}"')
            print("    Traceback:")
            for item in traceback:
                # if item.lemma.startswith("<") or item.lemma.startswith("["):
                #     continue
                print(f"        {repr(item)}")
            print(
                f"    Encoder time: {enc_time_per_seq:.3f} seconds, RTF {enc_time_per_seq / seq_time:.3f}, XRTF {seq_time / enc_time_per_seq:.3f}"
            )
            print(
                f"    Search time: {search_time:.3f} seconds, RTF {search_time / seq_time:.3f}, XRTF {seq_time / search_time:.3f}"
            )
            print()

            ctm_strs.append(_traceback_to_ctm_str(traceback, ms_per_enc_frame))

            if self.aligner is not None:
                align_traceback = self.aligner.align_segment(features=es, orth=orths[b] + " ")
                recog_transcription = _traceback_to_transcription(traceback)
                align_transcription = _traceback_to_transcription(align_traceback)

                recog_score = _traceback_to_score(traceback)
                align_score = _traceback_to_score(align_traceback)

                if align_transcription != orths[b]:
                    print("    Could not successfully compute forced alignment. Transcription may contain OOV words.")
                    skipped.append(1)
                    search_errors.append(0)
                    model_errors.append(0)
                    correct.append(0)
                elif recog_transcription == orths[b]:
                    print("    Correct transcription found.")
                    skipped.append(0)
                    search_errors.append(0)
                    model_errors.append(0)
                    correct.append(1)
                elif align_score < recog_score:
                    print(
                        f"    Encountered search error. Forced alignment has score {align_score} while search has score {recog_score}"
                    )
                    skipped.append(0)
                    search_errors.append(1)
                    model_errors.append(0)
                    correct.append(0)
                else:
                    print(
                        f"    Encountered model error. Forced alignment has score {align_score} while search has score {recog_score}"
                    )
                    skipped.append(0)
                    search_errors.append(0)
                    model_errors.append(1)
                    correct.append(0)

        if self.resumable:
            computed_outputs = {
                "ctm_string": ctm_strs,
                "enc_time": encoder_times,
                "audio_time": (audio_samples_size / self.sample_rate).cpu().numpy(),
                "search_time": search_times,
            }
            if self.aligner is not None:
                computed_outputs.update(
                    search_errors=search_errors, model_errors=model_errors, skipped=skipped, correct=correct,
                )
            self._emit_resume_outputs(original_seq_tags, pending, computed_outputs)
            return

        import returnn.frontend as rf

        run_ctx = rf.get_run_ctx()

        ctm_string_tensor = Tensor(
            name="ctm_string",
            dtype="string",
            raw_tensor=np.array(ctm_strs, dtype="U"),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(ctm_string_tensor, name="ctm_string")
        self._mark_nbest_output(nbest_records)

        if len(search_errors) > 0:
            search_errors_tensor = Tensor(
                name="search_errors",
                dtype="int32",
                raw_tensor=np.array(search_errors, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(search_errors_tensor, name="search_errors")

            model_errors_tensor = Tensor(
                name="model_errors",
                dtype="int32",
                raw_tensor=np.array(model_errors, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(model_errors_tensor, name="model_errors")

            skipped_tensor = Tensor(
                name="skipped",
                dtype="int32",
                raw_tensor=np.array(skipped, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(skipped_tensor, name="skipped")

            correct_tensor = Tensor(
                name="correct",
                dtype="int32",
                raw_tensor=np.array(correct, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(correct_tensor, name="correct")

        enc_time_tensor = Tensor(
            name="enc_time",
            dtype="float32",
            raw_tensor=np.array(encoder_times, dtype=np.float32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(enc_time_tensor, name="enc_time")

        audio_time_tensor = Tensor(
            name="audio_time",
            dtype="float32",
            raw_tensor=audio_samples_size / self.sample_rate,
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(audio_time_tensor, name="audio_time")

        search_time_tensor = Tensor(
            name="search_time",
            dtype="float32",
            raw_tensor=np.array(search_times, dtype=np.float32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(search_time_tensor, name="search_time")


class StreamingRasrRecogForwardStep(RasrRecogForwardStep):
    def __init__(
        self,
        encoder_frame_shift_seconds: float,
        chunk_history_seconds: float,
        chunk_center_seconds: float,
        chunk_future_seconds: float,
        print_intermediate_outputs: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.chunk_history_samples = _seconds_to_samples(chunk_history_seconds, self.sample_rate)
        self.chunk_center_samples = _seconds_to_samples(chunk_center_seconds, self.sample_rate)
        self.chunk_future_samples = _seconds_to_samples(chunk_future_seconds, self.sample_rate)

        self.encoder_frame_shift_seconds = encoder_frame_shift_seconds

        self.print_intermediate_outputs = print_intermediate_outputs

    def __call__(self, *, model: EncoderModel, extern_data: TensorDict, **_) -> None:
        assert self.search_algorithm is not None

        raw_data = extern_data.as_raw_tensor_dict()
        audio_samples = raw_data["data"]
        audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
        orths = raw_data["raw"]
        seq_tags = raw_data["seq_tag"]

        ctm_strs = []

        unstable_latency_arrays = []
        unstable_latency_lengths = []
        stable_latency_arrays = []
        stable_latency_lengths = []

        for b in range(audio_samples.size(0)):
            self.search_algorithm.reset()
            self.search_algorithm.enter_segment()

            seq_samples_size = audio_samples_size[b].item()
            seq_samples = audio_samples[b : b + 1, :seq_samples_size]  # [1, T, 1]

            chunk_center_start_sample = 0
            num_prev_stable_results = 0
            prev_chunk_end_sample = 0

            leftover_processing_time = 0

            unstable_latencies = []
            stable_latencies = []

            while chunk_center_start_sample < seq_samples_size:
                chunk_center_end_sample = min(seq_samples_size, chunk_center_start_sample + self.chunk_center_samples)

                chunk_start_sample = max(0, chunk_center_start_sample - self.chunk_history_samples)
                chunk_end_sample = min(seq_samples_size, chunk_center_end_sample + self.chunk_future_samples)

                # If encoder + search processing time is shorter than the new audio, it gets absorbed. Otherwise, the next
                # processing step has to wait a bit even after all audio is available until the previous processing step is complete.
                leftover_processing_time = max(
                    0,
                    leftover_processing_time
                    - _samples_to_seconds(chunk_end_sample - prev_chunk_end_sample, self.sample_rate),
                )
                prev_chunk_end_sample = chunk_end_sample

                encoder_start = perf_counter()

                model_output = model.forward(seq_samples, seq_samples_size)
                if isinstance(model_output, (tuple, list)):
                    encoder_states, _ = model_output
                else:
                    encoder_states = model_output

                ms_per_enc_frame = round(
                    ((chunk_end_sample - chunk_start_sample) / encoder_states.size(1)) / self.sample_rate * 1000
                )

                encoder_elapsed = perf_counter() - encoder_start

                total_encoder_frames = encoder_states.size(1)

                center_start_frame = max(
                    0,
                    _samples_to_frames(
                        chunk_center_start_sample - chunk_start_sample,
                        self.sample_rate,
                        self.encoder_frame_shift_seconds,
                    ),
                )
                center_end_frame = min(
                    _samples_to_frames(
                        chunk_center_end_sample - chunk_start_sample,
                        self.sample_rate,
                        self.encoder_frame_shift_seconds,
                    ),
                    total_encoder_frames,
                )

                if center_end_frame > center_start_frame:
                    search_start = perf_counter()

                    self.search_algorithm.put_features(encoder_states[:, center_start_frame:center_end_frame])
                    full_traceback = self.search_algorithm.get_current_best_traceback()
                    stable_traceback = self.search_algorithm.get_common_prefix()

                    search_elapsed = perf_counter() - search_start

                    # Realtime where we would be in the stream after sending all samples for current chunk, encoding them and running search
                    processing_finish_time = (
                        _samples_to_seconds(chunk_end_sample, self.sample_rate)
                        + leftover_processing_time
                        + encoder_elapsed
                        + search_elapsed
                    )

                    leftover_processing_time += encoder_elapsed + search_elapsed

                    new_unstable_results = full_traceback[len(stable_traceback) :]
                    new_stable_results = stable_traceback[num_prev_stable_results:]

                    # Count differences from when words in hypothesis have ended to where we are now
                    for stable_result in new_stable_results:
                        stable_latencies.append(
                            processing_finish_time
                            - _frames_to_seconds(stable_result.end_time, self.encoder_frame_shift_seconds)
                        )

                    for unstable_result in new_unstable_results:
                        unstable_latencies.append(
                            processing_finish_time
                            - _frames_to_seconds(unstable_result.end_time, self.encoder_frame_shift_seconds)
                        )

                    if self.print_intermediate_outputs:
                        print("Intermediate hypothesis after processing chunk:")
                        print(
                            f'  Previous stable hypothesis: "{_traceback_to_string(stable_traceback[:num_prev_stable_results])}"'
                        )
                        print(f'  New stable extension: "{_traceback_to_string(new_stable_results)}"')
                        print(f'  New unstable extension: "{_traceback_to_string(new_unstable_results)}"')

                    num_prev_stable_results = len(stable_traceback)

                chunk_center_start_sample = chunk_center_end_sample

            self.search_algorithm.finish_segment()
            final_traceback = self.search_algorithm.get_current_best_traceback()
            ctm_strs.append(_traceback_to_ctm_str(final_traceback, ms_per_enc_frame))

            print(f"Recognized sequence {repr(seq_tags[b])}")
            print(f'    Ground truth: "{orths[b]}"', flush=True)
            print(f'    Recognized: "{_traceback_to_transcription(final_traceback)}"')
            print("    Traceback:")
            for item in final_traceback:
                if item.lemma.startswith("<") or item.lemma.startswith("["):
                    continue
                print(f"        {repr(item)}")
            print()

            unstable_latency_arrays.append(np.array(unstable_latencies, dtype=np.float32))
            stable_latency_arrays.append(np.array(stable_latencies, dtype=np.float32))
            unstable_latency_lengths.append(len(unstable_latencies))
            stable_latency_lengths.append(len(stable_latencies))

        import returnn.frontend as rf

        run_ctx = rf.get_run_ctx()

        ctm_string_tensor = Tensor(
            name="ctm_string",
            dtype="string",
            raw_tensor=np.array(ctm_strs, dtype="U"),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(ctm_string_tensor, name="ctm_string")

        unstable_latency_arrays_padded = [
            np.pad(latency_array, pad_width=(0, np.max(unstable_latency_lengths) - len(latency_array)))
            for latency_array in unstable_latency_arrays
        ]
        unstable_latencies_tensor = Tensor(
            name="unstable_latencies",
            dtype="float32",
            raw_tensor=np.stack(unstable_latency_arrays_padded, axis=0),
            feature_dim_axis=None,
        )
        unstable_latency_lengths_array = np.array(unstable_latency_lengths, dtype=np.int32)
        if run_ctx.expected_outputs is not None:
            assert run_ctx.expected_outputs["unstable_latencies"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs["unstable_latencies"].dims[
                1
            ].dyn_size_ext.raw_tensor = unstable_latency_lengths_array
        run_ctx.mark_as_output(unstable_latencies_tensor, name="unstable_latencies")

        stable_latency_arrays_padded = [
            np.pad(latency_array, pad_width=(0, np.max(stable_latency_lengths) - len(latency_array)))
            for latency_array in stable_latency_arrays
        ]
        stable_latencies_tensor = Tensor(
            name="stable_latencies",
            dtype="float32",
            raw_tensor=np.stack(stable_latency_arrays_padded, axis=0),
            feature_dim_axis=None,
        )
        stable_latency_lengths_array = np.array(stable_latency_lengths, dtype=np.int32)
        if run_ctx.expected_outputs is not None:
            assert run_ctx.expected_outputs["stable_latencies"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs["stable_latencies"].dims[1].dyn_size_ext.raw_tensor = stable_latency_lengths_array
        run_ctx.mark_as_output(stable_latencies_tensor, name="stable_latencies")
