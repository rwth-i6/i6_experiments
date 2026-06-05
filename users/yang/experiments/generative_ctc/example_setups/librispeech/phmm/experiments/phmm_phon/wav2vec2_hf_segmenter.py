import copy
from dataclasses import asdict

from sisyphus import tk

from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.common.setups.returnn.datasets import MetaDataset, OggZipDataset

from ...data.phmm_common import DatasetSettings, TrainingDatasets, get_audio_raw_datastream
from ...phmm_config import get_forward_config
from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_pipeline import ASRModel, prepare_asr_model, training
from ...phmm_rasr import CreateLibrasrVenvJob
from ...pytorch_networks.phmm.raw_cnn_segmenter_cfg import ModelConfig as RawCnnSegmenterConfig
from ...pytorch_networks.phmm.wav2vec2_hf_segmenter_cfg import ModelConfig
from ...segmenter_jobs import (
    CompareSegmentStartHDFJob,
    DetectBoundariesFromScoreHDFJob,
    DetectPeaksFromScoreHDFJob,
    DumpGmmAlignmentSegmentStartsJob,
)


def _build_existing_ogg_audio_datasets(settings: DatasetSettings) -> TrainingDatasets:
    """Use already materialized train-other-960 OGG zip to avoid rebuilding LibriSpeech corpus jobs."""
    train_ogg = tk.Path("/u/zyang/setups/mini/work/i6_core/returnn/oggzip/BlissToOggZipJob.mNyFEqof29Q5/output/out.ogg.zip")
    cv_segments = tk.Path("/u/zyang/setups/mini/work/i6_core/corpus/segments/ShuffleAndSplitSegmentsJob.RorARIMr0Hr0/output/cv.segments")
    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    def make_dataset(*, segment_file=None, partition_epoch=None, seq_ordering="sorted_reverse"):
        zip_dataset = OggZipDataset(
            files=[train_ogg],
            audio_options=audio_datastream.as_returnn_audio_opts(),
            target_options=None,
            segment_file=segment_file,
            partition_epoch=partition_epoch,
            seq_ordering=seq_ordering,
        )
        return MetaDataset(
            data_map={"raw_audio": ("zip_dataset", "data")},
            datasets={"zip_dataset": zip_dataset},
            seq_order_control_dataset="zip_dataset",
        )

    train_dataset = make_dataset(
        partition_epoch=settings.train_partition_epoch,
        seq_ordering=settings.train_seq_ordering,
    )
    cv_dataset = make_dataset(segment_file=cv_segments)

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=cv_dataset,
        datastreams={"raw_audio": audio_datastream},
        prior=None,
    )


def eow_phon_phmm_ls960_wav2vec2_segmenter():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_wav2vec2_segmenter"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data = _build_existing_ogg_audio_datasets(settings=train_settings)
    alignment_hdfs = [tk.Path(f"output/lbs_mono_phone_eow_lexicon/alignment_{i}.hdf") for i in range(1, 201)]
    gmm_start_job = DumpGmmAlignmentSegmentStartsJob(alignment_hdfs)
    gmm_start_job.add_alias(prefix_name + "/gmm_alignment_segment_starts")
    tk.register_output(prefix_name + "/gmm_alignment_segment_starts.hdf", gmm_start_job.out_hdf)

    forward_dataset_opts = copy.deepcopy(train_data.cv.as_returnn_opts())
    forward_zip_dataset_opts = forward_dataset_opts["datasets"]["zip_dataset"]
    forward_zip_dataset_opts["partition_epoch"] = 1
    forward_zip_dataset_opts["seq_ordering"] = "sorted_reverse"
    full_train_forward_dataset_opts = copy.deepcopy(train_data.train.as_returnn_opts())
    full_train_forward_zip_dataset_opts = full_train_forward_dataset_opts["datasets"]["zip_dataset"]
    full_train_forward_zip_dataset_opts["partition_epoch"] = 1
    full_train_forward_zip_dataset_opts["seq_ordering"] = "sorted_reverse"

    network_module = "phmm.wav2vec2_hf_segmenter"
    num_epochs = 20
    default_train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 0.01},
        "learning_rates": [1e-4],
        "batch_size": 200 * 16000,
        "max_seq_length": {"raw_audio": 35 * 16000},
        "accum_grad_multiple_step": 2,
        "torch_amp_options": {"dtype": "bfloat16"},
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 4,
            "keep": [num_epochs],
        },
    }
    segmenter_returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    default_returnn = {
        "returnn_exe": segmenter_returnn_exe,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    base_model_config = ModelConfig(
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        apply_spec_augment=False,
        final_dropout=0.0,
        return_layer=9,
    )

    def run_training_and_forward(
        run_suffix: str,
        model_config,
        *,
        run_network_module: str = network_module,
        learning_rate: float = 1e-4,
        batch_size: int = 200,
    ):
        training_name = prefix_name + f"/{run_network_module}.{run_suffix}"
        train_config = dict(default_train_config)
        train_config["learning_rates"] = [learning_rate]
        train_config["batch_size"] = batch_size * 16000
        train_args = {
            "network_module": run_network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "config": train_config,
            "use_training_config_v2": True,
            "debug": False,
        }
        train_job = training(
            training_name,
            train_data,
            train_args,
            num_epochs=num_epochs,
            **default_returnn,
        )
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name,
            train_job,
            train_args,
            with_prior=False,
            get_specific_checkpoint=num_epochs,
        )
        score_hdf = run_score_forward(
            training_name=training_name + f"/ep_{num_epochs}",
            output_alias=prefix_name + f"/segment_scores/{run_suffix}.hdf",
            asr_model=asr_model,
        )
        return asr_model, score_hdf

    def run_score_forward(
        training_name: str,
        output_alias: str,
        asr_model: ASRModel,
        *,
        dataset_opts=None,
        batch_size: int = 200,
        max_seqs: int = 120,
        time_rqmt: int = 24,
    ):
        output_filename = "scores.hdf"
        forward_config = get_forward_config(
            network_module=asr_model.network_module,
            config={
                "forward": dataset_opts or forward_dataset_opts,
                "batch_size": batch_size * 16000,
                "max_seqs": max_seqs,
                "num_workers_per_gpu": 0,
            },
            net_args=asr_model.net_args,
            decoder="phmm.segmenter_boundary_forward",
            decoder_args={
                "config": {
                    "output_mode": "scores",
                    "threshold": 0.0,
                    "output_filename": output_filename,
                    "boundary_frame_offset": 1,
                }
            },
        )
        forward_job = ReturnnForwardJobV2(
            model_checkpoint=asr_model.checkpoint,
            returnn_config=forward_config,
            log_verbosity=5,
            mem_rqmt=24,
            time_rqmt=time_rqmt,
            device="gpu",
            cpu_rqmt=4,
            output_files=[output_filename],
            returnn_python_exe=segmenter_returnn_exe,
            returnn_root=MINI_RETURNN_ROOT,
        )
        forward_job.add_alias(training_name + "/segment_scores_forward")
        tk.register_output(training_name + "/segment_scores.hdf", forward_job.out_files[output_filename])
        tk.register_output(output_alias, forward_job.out_files[output_filename])
        return forward_job.out_files[output_filename]

    def run_full_train_scores_and_peak_detection(run_suffix: str, asr_model: ASRModel):
        score_hdf = run_score_forward(
            training_name=prefix_name + f"/{asr_model.network_module}.{run_suffix}/ep_{num_epochs}/full_train",
            output_alias=prefix_name + f"/segment_scores_full_train/{run_suffix}.hdf",
            asr_model=asr_model,
            dataset_opts=full_train_forward_dataset_opts,
            batch_size=400,
            max_seqs=240,
            time_rqmt=168,
        )
        start_job = DetectPeaksFromScoreHDFJob(
            score_hdf,
            prominence=0.02,
            min_distance=0,
        )
        start_job.add_alias(prefix_name + f"/segment_starts_full_train/{run_suffix}_peak_prom0p02_mindist0")
        tk.register_output(
            prefix_name + f"/segment_starts_full_train/{run_suffix}_peak_prom0p02_mindist0.hdf",
            start_job.out_hdf,
        )
        return score_hdf, start_job.out_hdf

    def run_boundary_detection_and_compare(run_suffix: str, score_hdf: tk.Path, *, min_distance: int = 1):
        output_suffix = run_suffix if min_distance == 1 else f"{run_suffix}_mindist{min_distance}"
        start_job = DetectBoundariesFromScoreHDFJob(score_hdf, min_distance=min_distance)
        start_job.add_alias(prefix_name + f"/segment_starts/{output_suffix}")
        tk.register_output(prefix_name + f"/segment_starts/{output_suffix}.hdf", start_job.out_hdf)

        compare_job = CompareSegmentStartHDFJob(
            source_hdf=start_job.out_hdf,
            target_hdf=gmm_start_job.out_hdf,
            source_score_hdf=score_hdf,
            target_alignment_hdfs=alignment_hdfs,
        )
        compare_job.add_alias(prefix_name + f"/boundary_comparison/{output_suffix}_vs_gmm")
        tk.register_output(prefix_name + f"/boundary_comparison/{output_suffix}_vs_gmm.txt", compare_job.out_report)

    def run_peak_detection_and_compare(
        run_suffix: str,
        score_hdf: tk.Path,
        *,
        prominence: float,
        min_distance: int = 0,
    ):
        prominence_str = f"{prominence:g}".replace(".", "p")
        output_suffix = f"{run_suffix}_peak_prom{prominence_str}_mindist{min_distance}"
        start_job = DetectPeaksFromScoreHDFJob(
            score_hdf,
            prominence=prominence,
            min_distance=min_distance,
        )
        start_job.add_alias(prefix_name + f"/segment_starts/{output_suffix}")
        tk.register_output(prefix_name + f"/segment_starts/{output_suffix}.hdf", start_job.out_hdf)

        compare_job = CompareSegmentStartHDFJob(
            source_hdf=start_job.out_hdf,
            target_hdf=gmm_start_job.out_hdf,
            source_score_hdf=score_hdf,
            target_alignment_hdfs=alignment_hdfs,
        )
        compare_job.add_alias(prefix_name + f"/boundary_comparison/{output_suffix}_vs_gmm")
        tk.register_output(prefix_name + f"/boundary_comparison/{output_suffix}_vs_gmm.txt", compare_job.out_report)

    def run_peak_sweep(run_suffix: str, score_hdf: tk.Path):
        for prominence in [0.02, 0.03, 0.04, 0.05, 0.08, 0.10]:
            run_peak_detection_and_compare(run_suffix, score_hdf, prominence=prominence, min_distance=0)

    run_training_and_forward(
        "default_blocks1_frozenenc",
        base_model_config,
    )

#    trainable_encoder_config = ModelConfig(
#        **{
#            **asdict(base_model_config),
#            "freeze_feature_encoder": False,
#            "freeze_encoder": False,
#        }
#    )
#    run_training_and_forward(
#        "default_blocks1_trainableenc",
#        trainable_encoder_config,
#    )

    blocks2_config = ModelConfig(**asdict(base_model_config))
    blocks2_config.segmenter_num_blocks = 2
    run_training_and_forward(
        "blocks2_frozenenc",
        blocks2_config,
    )

    raw_cnn_config = RawCnnSegmenterConfig()
    _raw_cnn_model, raw_cnn_scores = run_training_and_forward(
        "kreuk_rawcnn",
        raw_cnn_config,
        run_network_module="phmm.raw_cnn_segmenter",
        learning_rate=2e-4,
        batch_size=400,
    )
    run_boundary_detection_and_compare("kreuk_rawcnn", raw_cnn_scores)
    run_boundary_detection_and_compare("kreuk_rawcnn", raw_cnn_scores, min_distance=0)
    run_peak_sweep("kreuk_rawcnn", raw_cnn_scores)

    raw_cnn_no_final_act_config = RawCnnSegmenterConfig(apply_final_norm_activation=False)
    raw_cnn_no_final_act_model, raw_cnn_no_final_act_scores = run_training_and_forward(
        "kreuk_rawcnn_no_final_bn_lrelu",
        raw_cnn_no_final_act_config,
        run_network_module="phmm.raw_cnn_segmenter",
        learning_rate=2e-4,
        batch_size=400,
    )
    run_boundary_detection_and_compare("kreuk_rawcnn_no_final_bn_lrelu", raw_cnn_no_final_act_scores)
    run_boundary_detection_and_compare(
        "kreuk_rawcnn_no_final_bn_lrelu",
        raw_cnn_no_final_act_scores,
        min_distance=0,
    )
    run_peak_sweep("kreuk_rawcnn_no_final_bn_lrelu", raw_cnn_no_final_act_scores)
    run_full_train_scores_and_peak_detection("kreuk_rawcnn_no_final_bn_lrelu", raw_cnn_no_final_act_model)

py = eow_phon_phmm_ls960_wav2vec2_segmenter
