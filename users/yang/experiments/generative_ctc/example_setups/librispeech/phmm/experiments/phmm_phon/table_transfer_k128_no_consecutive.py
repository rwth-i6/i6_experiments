import copy
import os
import re
from dataclasses import asdict

from sisyphus import Job, Task, tk

from ...data.phmm_common import TrainingDatasets
from ...experiments.phmm_phon.rasr_table_decoding import OggZipTextToBlissSubsetJob
from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_lm import create_lm_image_for_lexicon
from ...phmm_pipeline import ASRModel, search, training
from ...phmm_rasr import CreateLibrasrVenvJob, build_librasr_phone_table_recognition_config
from ...pytorch_networks.phmm.decoder.rasr_trainable_table_decoding_v1 import DecoderConfig
from ...pytorch_networks.phmm.table_transfer_fullsum_cfg import ModelConfig
from ...segment_clustering_jobs import MakeSparseClusterHDFDatasetCompatibleJob


class _ReturnnDataset:
    def __init__(self, opts):
        self.opts = opts

    def as_returnn_opts(self):
        return copy.deepcopy(self.opts)


def _num_str(value: float) -> str:
    return ("%g" % value).replace(".", "p")


class TrainableTableDecodingSummaryJob(Job):
    def __init__(self, rows):
        self.rows = rows
        self.out_report = self.output_path("trainable_table_decoding_summary.txt")

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


def _make_hdf_dataset(*, hdf_files, partition_epoch, seq_ordering, segment_file):
    if not isinstance(hdf_files, (list, tuple)):
        hdf_files = [hdf_files]
    return _ReturnnDataset(
        {
            "class": "HDFDataset",
            "files": list(hdf_files),
            "partition_epoch": partition_epoch,
            "seq_ordering": seq_ordering,
            "seq_list_filter_file": segment_file,
        }
    )


def _make_training_datasets(*, hdf_files, train_segments, cv_segments):
    train_dataset = _make_hdf_dataset(
        hdf_files=hdf_files,
        partition_epoch=10,
        seq_ordering="laplace:.100",
        segment_file=train_segments,
    )
    cv_dataset = _make_hdf_dataset(
        hdf_files=hdf_files,
        partition_epoch=1,
        seq_ordering="sorted",
        segment_file=cv_segments,
    )
    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=cv_dataset,
        datastreams={},
        prior=None,
    )


def _warmup_plateau_decay_lr(num_epochs, init_lr, peak_lr, final_lr):
    peak_epoch = round(0.2 * num_epochs)
    decay_start_epoch = round(0.7 * num_epochs)
    lrs = []
    for epoch in range(1, num_epochs + 1):
        if epoch <= peak_epoch:
            if peak_epoch <= 1:
                lr = peak_lr
            else:
                lr = init_lr + (peak_lr - init_lr) * (epoch - 1) / (peak_epoch - 1)
        elif epoch <= decay_start_epoch:
            lr = peak_lr
        else:
            denom = max(1, num_epochs - decay_start_epoch)
            lr = peak_lr + (final_lr - peak_lr) * (epoch - decay_start_epoch) / denom
        lrs.append(lr)
    return lrs


def eow_phon_phmm_ls960_table_transfer_k128_no_consecutive():
    prefix_name = (
        "example_setups/librispeech/phmm_standalone_2024/"
        "ls960_table_transfer_k128_no_consecutive"
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
    lm_table = tk.Path(
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_phoneme_cipher/phoneme_ngram_conv_lm_v2_context3_ep200_log_probs.pt"
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
    base_lm_config, lm_image = create_lm_image_for_lexicon(
        lexicon_file=lexicon,
        scale=1.0,
        output_prefix=prefix_name + "/no_eow_4gram_lm_image",
    )

    k128_source_prefix = (
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_wav2vec2_large_layer15_pca/segment_clustering/"
        "layer15_pca512_peak_prom0p02_full_train_k128_4m"
    )
    original_cluster_hdfs = [
        tk.Path(f"{k128_source_prefix}/cluster_labels_k128.{idx:03d}.hdf")
        for idx in range(10)
    ]
    compat_job = MakeSparseClusterHDFDatasetCompatibleJob(original_cluster_hdfs)
    compat_job.add_alias(prefix_name + "/k128_original/make_hdf_returnn_compatible")
    for idx, compat_hdf in enumerate(compat_job.out_hdf_files):
        tk.register_output(
            prefix_name + f"/k128_original/cluster_labels_k128.{idx:03d}.returnn_compat.hdf",
            compat_hdf,
        )

    merged_cluster_hdf = tk.Path(
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_wav2vec2_large_layer15_pca_merge_cluster_ids/"
        "k128_peak_prom0p02/cluster_labels_k128_merged.hdf"
    )

    num_epochs = 300
    lr_variants = [
        ("const_lr1e-3", [1e-3] * num_epochs),
        (
            "warmup1e-4_peak1e-3_final1e-5",
            _warmup_plateau_decay_lr(num_epochs, init_lr=1e-4, peak_lr=1e-3, final_lr=1e-5),
        ),
    ]
    model_config = ModelConfig(
        input_vocab_size=128,
        output_vocab_size=40,
        lm_table_path=lm_table,
        lm_vocab_size=41,
        lm_context_length=3,
        beam_size=300,
        softmax_temperature=1.0,
        use_lm_silence_score=False,
        lm_scale=0.6,
        am_scale=1.0,
        table_init_scale=0.02,
    )

    for lr_name, learning_rates in lr_variants:
        train_config = {
            "optimizer": {"class": "Adam"},
            "learning_rates": learning_rates,
            "batch_size": 40_000,
            "max_seqs": 200,
            "num_workers_per_gpu": 0,
            "accum_grad_multiple_step": 4,
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": [num_epochs],
            },
        }
        for run_name, hdf_files in (
            ("k128_original", compat_job.out_hdf_files),
            ("k128_merged", [merged_cluster_hdf]),
        ):
            datasets = _make_training_datasets(
                hdf_files=hdf_files,
                train_segments=train_segments,
                cv_segments=cv_segments,
            )
            training_name = prefix_name + f"/table_transfer_no_consecutive.{run_name}_adam_{lr_name}_ep300"
            train_job = training(
                training_name=training_name,
                datasets=datasets,
                train_args={
                    "network_module": "phmm.table_transfer_no_consecutive_fullsum",
                    "config": train_config,
                    "net_args": {"model_config_dict": asdict(model_config)},
                    "use_training_config_v2": True,
                },
                num_epochs=num_epochs,
                returnn_exe=returnn_exe,
                returnn_root=MINI_RETURNN_ROOT,
            )
            train_job.rqmt["gpu_mem"] = 48
            tk.register_output(
                training_name + "/epoch.300.pt",
                train_job.out_checkpoints[num_epochs].path,
            )

            asr_model = ASRModel(
                checkpoint=train_job.out_checkpoints[num_epochs],
                net_args={"model_config_dict": asdict(model_config)},
                network_module="phmm.table_transfer_no_consecutive_fullsum",
                prior_file=None,
                prior_files=None,
                prefix_name=training_name,
            )
            search_dataset = _make_hdf_dataset(
                hdf_files=hdf_files,
                partition_epoch=1,
                seq_ordering="sorted",
                segment_file=cv_segments,
            )
            summary_rows = []
            for lm_scale in (0.4, 0.6, 0.8):
                lm_config = base_lm_config._copy()
                lm_config.scale = lm_scale
                recog_config = build_librasr_phone_table_recognition_config(
                    lexicon_path=lexicon,
                    lm_config=lm_config,
                    collapse_repeated_labels=True,
                    score_threshold=24.0,
                    intermediate_score_threshold=24.0,
                    logfile_suffix=(
                        f"trainable_table_{run_name}_{lr_name}_lm{lm_scale:g}"
                    ),
                )
                decoder_config = DecoderConfig(
                    rasr_config_file=recog_config,
                    lexicon=lexicon,
                    data_key="data",
                    lm_image_file=lm_image,
                )
                search_name = training_name + f"/rasr_cv_decoding/lm{_num_str(lm_scale)}"
                _search_jobs, wers = search(
                    search_name,
                    forward_config={
                        "batch_size": 1000,
                        "max_seqs": 64,
                        "num_workers_per_gpu": 0,
                    },
                    asr_model=asr_model,
                    decoder_module="phmm.decoder.rasr_trainable_table_decoding_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples={"train_cv_1pct": (search_dataset, cv_bliss_job.out_corpus)},
                    returnn_exe=returnn_exe,
                    returnn_root=MINI_RETURNN_ROOT,
                    mem_rqmt=16,
                    use_gpu=False,
                )
                summary_rows.append(
                    {
                        "lm_scale": lm_scale,
                        "wer_file": wers[search_name + "/train_cv_1pct"],
                    }
                )

            summary_job = TrainableTableDecodingSummaryJob(summary_rows)
            summary_job.add_alias(training_name + "/rasr_cv_decoding/summary")
            tk.register_output(
                training_name + "/rasr_cv_decoding/summary.txt",
                summary_job.out_report,
            )

    small_beam_model_config = ModelConfig(
        input_vocab_size=128,
        output_vocab_size=40,
        lm_table_path=lm_table,
        lm_vocab_size=41,
        lm_context_length=3,
        beam_size=200,
        softmax_temperature=1.0,
        use_lm_silence_score=False,
        lm_scale=0.6,
        am_scale=1.0,
        table_init_scale=0.02,
    )
    small_beam_lr_variants = [
        ("const_lr1e-4_b60000_k200", [1e-4] * num_epochs),
        ("const_lr1e-3_b60000_k200", [1e-3] * num_epochs),
    ]
    for lr_name, learning_rates in small_beam_lr_variants:
        train_config = {
            "optimizer": {"class": "Adam"},
            "learning_rates": learning_rates,
            "batch_size": 40_000,
            "max_seqs": 200,
            "num_workers_per_gpu": 0,
            "accum_grad_multiple_step": 4,
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": [num_epochs],
            },
        }
        for run_name, hdf_files in (
            ("k128_original", compat_job.out_hdf_files),
            ("k128_merged", [merged_cluster_hdf]),
        ):
            datasets = _make_training_datasets(
                hdf_files=hdf_files,
                train_segments=train_segments,
                cv_segments=cv_segments,
            )
            training_name = prefix_name + f"/table_transfer_no_consecutive.{run_name}_adam_{lr_name}_ep300"
            train_job = training(
                training_name=training_name,
                datasets=datasets,
                train_args={
                    "network_module": "phmm.table_transfer_no_consecutive_fullsum",
                    "config": train_config,
                    "net_args": {"model_config_dict": asdict(small_beam_model_config)},
                    "use_training_config_v2": True,
                },
                num_epochs=num_epochs,
                returnn_exe=returnn_exe,
                returnn_root=MINI_RETURNN_ROOT,
            )
            train_job.rqmt["gpu_mem"] = 24
            tk.register_output(
                training_name + "/epoch.300.pt",
                train_job.out_checkpoints[num_epochs].path,
            )

            asr_model = ASRModel(
                checkpoint=train_job.out_checkpoints[num_epochs],
                net_args={"model_config_dict": asdict(small_beam_model_config)},
                network_module="phmm.table_transfer_no_consecutive_fullsum",
                prior_file=None,
                prior_files=None,
                prefix_name=training_name,
            )
            search_dataset = _make_hdf_dataset(
                hdf_files=hdf_files,
                partition_epoch=1,
                seq_ordering="sorted",
                segment_file=cv_segments,
            )
            summary_rows = []
            for lm_scale in (0.4, 0.6, 0.8):
                lm_config = base_lm_config._copy()
                lm_config.scale = lm_scale
                recog_config = build_librasr_phone_table_recognition_config(
                    lexicon_path=lexicon,
                    lm_config=lm_config,
                    collapse_repeated_labels=True,
                    score_threshold=24.0,
                    intermediate_score_threshold=24.0,
                    logfile_suffix=(
                        f"trainable_table_{run_name}_{lr_name}_lm{lm_scale:g}"
                    ),
                )
                decoder_config = DecoderConfig(
                    rasr_config_file=recog_config,
                    lexicon=lexicon,
                    data_key="data",
                    lm_image_file=lm_image,
                )
                search_name = training_name + f"/rasr_cv_decoding/lm{_num_str(lm_scale)}"
                _search_jobs, wers = search(
                    search_name,
                    forward_config={
                        "batch_size": 1000,
                        "max_seqs": 64,
                        "num_workers_per_gpu": 0,
                    },
                    asr_model=asr_model,
                    decoder_module="phmm.decoder.rasr_trainable_table_decoding_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples={"train_cv_1pct": (search_dataset, cv_bliss_job.out_corpus)},
                    returnn_exe=returnn_exe,
                    returnn_root=MINI_RETURNN_ROOT,
                    mem_rqmt=16,
                    use_gpu=False,
                )
                summary_rows.append(
                    {
                        "lm_scale": lm_scale,
                        "wer_file": wers[search_name + "/train_cv_1pct"],
                    }
                )

            summary_job = TrainableTableDecodingSummaryJob(summary_rows)
            summary_job.add_alias(training_name + "/rasr_cv_decoding/summary")
            tk.register_output(
                training_name + "/rasr_cv_decoding/summary.txt",
                summary_job.out_report,
            )


py = eow_phon_phmm_ls960_table_transfer_k128_no_consecutive
