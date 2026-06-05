import copy
import os
import re
from dataclasses import asdict

import numpy as np
from sisyphus import Job, Task, tk

from i6_experiments.common.setups.returnn.datasets import MetaDataset, OggZipDataset
from i6_experiments.common.setups.returnn.datasets.generic import HDFDataset

from ...data.phmm_common import DatasetSettings, TrainingDatasets, get_audio_raw_datastream
from ...experiments.phmm_phon.rasr_table_decoding import OggZipTextToBlissSubsetJob
from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_lm import create_lm_image_for_lexicon
from ...phmm_pipeline import ASRModel, search, training
from ...phmm_rasr import (
    CreateLibrasrVenvJob,
    build_fsa_exporter_config,
    build_librasr_phmm_recognition_config,
)
from ...pytorch_networks.phmm.decoder.rasr_sequence_phmm_v1 import DecoderConfig
from ...pytorch_networks.phmm.sequence_phmm_supervised_generative_cfg import ModelConfig
from ...segment_clustering_jobs import MakeSparseClusterHDFDatasetCompatibleJob


class SequenceSupervisedDecodingSummaryJob(Job):
    def __init__(self, rows):
        self.rows = rows
        self.out_report = self.output_path("sequence_supervised_decoding_summary.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _read_wer(wer_var) -> float:
        if hasattr(wer_var, "get"):
            return float(wer_var.get())
        with open(tk.uncached_path(wer_var), "rt") as f:
            text = f.read().strip()
        matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
        if not matches:
            raise ValueError(f"Could not parse WER from {wer_var}: {text!r}")
        return float(matches[0])

    @staticmethod
    def _get_sclite_dtl_path(wer_var) -> str:
        if not hasattr(wer_var, "get_path"):
            raise TypeError(f"Expected Sisyphus variable with get_path(), got {wer_var!r}")
        return os.path.join(os.path.dirname(wer_var.get_path()), "reports", "sclite.dtl")

    @staticmethod
    def _parse_sclite_error_breakdown(dtl_path: str):
        keys = {
            "Percent Substitution": "sub",
            "Percent Deletions": "del",
            "Percent Insertions": "ins",
        }
        result = {}
        pattern = re.compile(r"^(Percent (?:Substitution|Deletions|Insertions))\s*=\s*([0-9.]+)%\s*\(")
        with open(dtl_path, "rt", errors="ignore") as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    result[keys[match.group(1)]] = float(match.group(2))
        missing = [key for key in ("sub", "del", "ins") if key not in result]
        if missing:
            raise ValueError(f"Could not parse {missing} from {dtl_path}")
        return result

    def run(self):
        results = []
        for row in self.rows:
            breakdown = self._parse_sclite_error_breakdown(self._get_sclite_dtl_path(row["wer_file"]))
            results.append(
                {
                    "lm_scale": row["lm_scale"],
                    "wer": self._read_wer(row["wer_file"]),
                    **breakdown,
                }
            )
        results.sort(key=lambda item: item["wer"])
        with open(self.out_report.get_path(), "wt") as f:
            f.write("lm_scale\twer\tsub_percent\tdel_percent\tins_percent\n")
            for item in results:
                f.write(
                    f"{item['lm_scale']:g}\t"
                    f"{item['wer']:.4f}\t"
                    f"{item['sub']:.4f}\t"
                    f"{item['del']:.4f}\t"
                    f"{item['ins']:.4f}\n"
                )


def _num_str(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def _make_lr_schedule(num_epochs: int, init_lr: float = 1e-5, peak_lr: float = 1e-4):
    epoch_1 = int(num_epochs * 0.45)
    epoch_2 = num_epochs - 2 * epoch_1
    return (
        list(np.linspace(init_lr, peak_lr, epoch_1))
        + list(np.linspace(peak_lr, init_lr, epoch_1))
        + list(np.linspace(init_lr, 1e-6, epoch_2))
    )


def no_eow_phon_phmm_ls960_sequence_supervised_generative():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_sequence_supervised_generative"

    settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.100",
    )

    returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    train_segments = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/corpus/segments/"
        "ShuffleAndSplitSegmentsJob.RorARIMr0Hr0/output/train.segments"
    )
    cv_segments = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/corpus/segments/"
        "ShuffleAndSplitSegmentsJob.RorARIMr0Hr0/output/cv.segments"
    )
    train_ogg_zip = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/returnn/oggzip/"
        "BlissToOggZipJob.mNyFEqof29Q5/output/out.ogg.zip"
    )
    lexicon = tk.Path(
        "/work/asr4/zyang/corpora/librispeech/960/lexicon/"
        "phmm_no_eow_special_phonemes.lexicon.xml.gz"
    )

    cv_bliss_job = OggZipTextToBlissSubsetJob(
        ogg_zip=train_ogg_zip,
        segment_file=cv_segments,
    )
    cv_bliss_job.add_alias(prefix_name + "/train_cv_1pct_reference")
    tk.register_output(prefix_name + "/train_cv_1pct_reference/corpus.xml.gz", cv_bliss_job.out_corpus)

    fsa_exporter_config = build_fsa_exporter_config(
        lexicon_path=lexicon,
        corpus_path=cv_bliss_job.out_corpus,
    )
    base_lm_config, lm_image = create_lm_image_for_lexicon(
        lexicon_file=lexicon,
        scale=1.0,
        output_prefix=prefix_name + "/no_eow_4gram_lm_image",
    )
    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    def make_dataset(*, hdf_files, segment_file, partition_epoch, seq_ordering, input_key: str = "data"):
        zip_dataset = OggZipDataset(
            files=[train_ogg_zip],
            audio_options=audio_datastream.as_returnn_audio_opts(),
            target_options=None,
            segment_file=segment_file,
            partition_epoch=partition_epoch,
            seq_ordering=seq_ordering,
        )
        hdf_dataset = HDFDataset(
            files=hdf_files if isinstance(hdf_files, (list, tuple)) else [hdf_files],
            partition_epoch=partition_epoch,
            segment_file=segment_file,
            seq_ordering=seq_ordering,
        )
        return MetaDataset(
            data_map={
                input_key: ("hdf_dataset", "data"),
                "labels": ("zip_dataset", "orth"),
            },
            datasets={
                "zip_dataset": zip_dataset,
                "hdf_dataset": hdf_dataset,
            },
            seq_order_control_dataset="zip_dataset",
        )

    def make_training_datasets(*, hdf_files, input_key: str = "data"):
        train_dataset = make_dataset(
            hdf_files=hdf_files,
            segment_file=train_segments,
            partition_epoch=settings.train_partition_epoch,
            seq_ordering=settings.train_seq_ordering,
            input_key=input_key,
        )
        cv_dataset = make_dataset(
            hdf_files=hdf_files,
            segment_file=cv_segments,
            partition_epoch=1,
            seq_ordering="sorted",
            input_key=input_key,
        )
        return TrainingDatasets(
            train=train_dataset,
            cv=cv_dataset,
            devtrain=cv_dataset,
            datastreams={},
            prior=None,
        )

    def run_training_and_search(
        *,
        run_name_suffix: str,
        model_config: ModelConfig,
        hdf_files,
        num_epochs: int = 300,
        decode_epochs=(300,),
        lm_scales=(0.4, 0.6, 0.8, 1.0),
        batch_size: int = 40_000,
        max_seqs: int = 1000,
        gpu_mem: int = 48,
        search_batch_size: int = 5_000,
        search_mem: int = 32,
        train_config_overrides=None,
        decoder_config_overrides=None,
    ):
        train_config = {
            "optimizer": {"class": "adamw", "epsilon": 1e-08, "weight_decay": 1e-2},
            "learning_rates": _make_lr_schedule(num_epochs),
            "batch_size": batch_size,
            "max_seqs": max_seqs,
            "num_workers_per_gpu": 0,
            "accum_grad_multiple_step": 4,
            "gradient_clip_norm": 1.0,
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": list(decode_epochs),
            },
        }
        if train_config_overrides:
            train_config.update(copy.deepcopy(train_config_overrides))

        default_decode_layer_index = (model_config.aux_loss_layers or [0])[-1]
        datasets = make_training_datasets(hdf_files=hdf_files, input_key=model_config.input_key)
        training_name = prefix_name + "/" + run_name_suffix
        train_job = training(
            training_name=training_name,
            datasets=datasets,
            train_args={
                "network_module": "phmm.sequence_phmm_supervised_generative",
                "config": train_config,
                "net_args": {"model_config_dict": asdict(model_config)},
                "train_step_args": {
                    "fsa_exporter_config_path": fsa_exporter_config,
                    "label_smoothing_scale": 0.0,
                },
                "use_training_config_v2": True,
            },
            num_epochs=num_epochs,
            returnn_exe=returnn_exe,
            returnn_root=MINI_RETURNN_ROOT,
        )
        train_job.rqmt["gpu_mem"] = gpu_mem
        for checkpoint_epoch in decode_epochs:
            tk.register_output(
                training_name + f"/epoch.{checkpoint_epoch}.pt",
                train_job.out_checkpoints[checkpoint_epoch].path,
            )

        search_dataset = make_dataset(
            hdf_files=hdf_files,
            segment_file=cv_segments,
            partition_epoch=1,
            seq_ordering="sorted",
            input_key=model_config.input_key,
        )
        for checkpoint_epoch in decode_epochs:
            asr_model = ASRModel(
                checkpoint=train_job.out_checkpoints[checkpoint_epoch],
                net_args={"model_config_dict": asdict(model_config)},
                network_module="phmm.sequence_phmm_supervised_generative",
                prior_file=None,
                prior_files=None,
                prefix_name=training_name,
            )
            summary_rows = []
            for lm_scale in lm_scales:
                lm_config = base_lm_config._copy()
                lm_config.scale = lm_scale
                recog_config = build_librasr_phmm_recognition_config(
                    lexicon_path=lexicon,
                    lm_config=lm_config,
                    logfile_suffix=f"sequence_supervised_{run_name_suffix}_ep{checkpoint_epoch}_lm{lm_scale:g}",
                )
                decoder_config_dict = {
                    "rasr_config_file": recog_config,
                    "lexicon": lexicon,
                    "decode_layer_index": default_decode_layer_index,
                    "input_key": model_config.input_key,
                    "prior_scale": 0.0,
                }
                if decoder_config_overrides:
                    decoder_config_dict.update(copy.deepcopy(decoder_config_overrides))
                decoder_config = DecoderConfig(**decoder_config_dict)
                search_name = training_name + f"/rasr_cv_decoding/ep_{checkpoint_epoch}/lm{_num_str(lm_scale)}"
                _search_jobs, wers = search(
                    search_name,
                    forward_config={
                        "batch_size": search_batch_size,
                        "max_seqs": 64,
                        "num_workers_per_gpu": 0,
                    },
                    asr_model=asr_model,
                    decoder_module="phmm.decoder.rasr_sequence_phmm_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples={"train_cv_1pct": (search_dataset, cv_bliss_job.out_corpus)},
                    returnn_exe=returnn_exe,
                    returnn_root=MINI_RETURNN_ROOT,
                    mem_rqmt=search_mem,
                    use_gpu=False,
                )
                summary_rows.append(
                    {
                        "lm_scale": lm_scale,
                        "wer_file": wers[search_name + "/train_cv_1pct"],
                    }
                )

            summary_job = SequenceSupervisedDecodingSummaryJob(summary_rows)
            summary_job.add_alias(training_name + f"/rasr_cv_decoding/ep_{checkpoint_epoch}/summary")
            tk.register_output(
                training_name + f"/rasr_cv_decoding/ep_{checkpoint_epoch}/summary.txt",
                summary_job.out_report,
            )

    for num_clusters in (128, 384, 512):
        cluster_source_prefix = (
            "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
            "ls960_wav2vec2_large_layer15_pca/segment_clustering/"
            f"layer15_pca512_peak_prom0p02_full_train_k{num_clusters}_4m"
        )
        cluster_hdfs = [
            tk.Path(f"{cluster_source_prefix}/cluster_labels_k{num_clusters}.{idx:03d}.hdf")
            for idx in range(10)
        ]
        compat_job = MakeSparseClusterHDFDatasetCompatibleJob(cluster_hdfs)
        compat_job.add_alias(prefix_name + f"/k{num_clusters}_unmerged/make_hdf_returnn_compatible")
        for idx, compat_hdf in enumerate(compat_job.out_hdf_files):
            tk.register_output(
                prefix_name + f"/k{num_clusters}_unmerged/cluster_labels_k{num_clusters}.{idx:03d}.returnn_compat.hdf",
                compat_hdf,
            )
        if num_clusters==128:
            lm_scales=(0.2,0.1,0.05,0.4)
        else:
            lm_scales=(0.2,0.1,0.4,0.6)

        run_training_and_search(
            run_name_suffix=f"sequence_supervised_generative.k{num_clusters}_cluster_kernel4_stride1_h512_ep100",
            model_config=ModelConfig(
                input_type="cluster",
                input_key="data",
                input_vocab_size=num_clusters,
                label_target_size=40,
                hidden_size=512,
                conv_kernel_size=4,
                conv_stride=1,
                conv_dilation=1,
                input_dropout=0.1,
                pre_output_dropout=0.1,
                sampling_type="batch",
                sampling_ratio=0.5,
                share_samples=True,
                ratio_corrector=1.0,
            ),
            lm_scales=lm_scales,
            hdf_files=compat_job.out_hdf_files,
            num_epochs=100,
            decode_epochs=(100,),
            batch_size=80_000,
            gpu_mem=24,
        )

    segment_representation_prefix = (
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_wav2vec2_large_layer15_pca/segment_representations/"
        "layer15_pca512_peak_prom0p02_full_train"
    )
    segment_representation_hdfs = [
        tk.Path(f"{segment_representation_prefix}/segment_representations.{idx:03d}.hdf")
        for idx in range(10)
    ]
    run_training_and_search(
        run_name_suffix="sequence_supervised_generative.pca512_segments_kernel4_stride1_h512_ep100",
        model_config=ModelConfig(
            input_type="vector",
            input_key="data",
            input_dim=512,
            label_target_size=40,
            hidden_size=512,
            conv_kernel_size=4,
            conv_stride=1,
            conv_dilation=1,
            input_dropout=0.1,
            pre_output_dropout=0.1,
            input_time_batch_norm=True,
            input_residual_linear=True,
            sampling_type="batch",
            sampling_ratio=0.5,
            share_samples=True,
            ratio_corrector=1.0,
        ),
        hdf_files=segment_representation_hdfs,
        num_epochs=100,
        decode_epochs=(100,),
        batch_size=80_000,
        gpu_mem=24,
    )


py = no_eow_phon_phmm_ls960_sequence_supervised_generative
