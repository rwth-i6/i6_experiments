"""Transition-probability sweep for wav2vecUseg features.

Warm-starts every run from the epoch-100 table of the sigma=0.02, seed=42,
5-gram vq_unsupervised_long run (silence-free ls-100h) and continues training
on the wav2vecUseg (k-means segmented) ls-100h features for 30 epochs, varying the
label-loop and label-to-silence probabilities.

Since the model already converged on ls100, 30 epochs is enough to see which
transition settings best preserve the model quality on wav2vecUseg features.
silence_loop_probability always equals loop_probability.

Compare the epoch-30 PER across experiments to pick the best settings for a
longer full run in vq_unsupervised_wav2vecUseg.
"""

from sisyphus import tk

from i6_core.text import PipelineJob

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    COLLEAGUE_CENTROIDS_K512,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
    phonetic_lm_dict,
)
from i6_experiments.example_setups.guided_kmeans.setup.statistics_jobs import (
    MixtureDiagnosticsJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import (
    decode_and_score,
    DecodeConfig,
)
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import (
    DatasetConfig,
    SegmentFile,
)
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import (
    LatexTableReport,
    clustering_statistics_per_epoch,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.w2vu_segmentation import (
    FilterHdfByTagsJob,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    build_decode_config,
    build_vq_training,
    silence_free_cv_features,
    silence_free_ls100_features,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised_wav2vecUseg import (
    WAV2VECUSEG_TRAIN_SHARDS,
    LS100H_WAV2VECUSEG_HDF_TAGS,
)

exp_dir = "vq_finetune_wav2vecUseg"
version = 1

# Source run parameters — must match vq_unsupervised_long exactly for all
# hashed fields so the reconstruction below reuses existing jobs.
_SOURCE_SIGMA = 0.02
_SOURCE_SEED = 42
_SOURCE_EXP_NAME = f"ls100-nosil_ours-5gram_sigma-{_SOURCE_SIGMA}_seed-{_SOURCE_SEED}"
_SOURCE_NUM_EPOCHS = 100
_SOURCE_BEAM_SIZE = 1000
_SOURCE_NUM_CHUNKS = 30
_SOURCE_NUM_WORKERS = 9

NUM_EPOCHS = 30
NUM_CHUNKS = 30
BEAM_SIZE = 1000
NUM_WORKERS = 12

DECODE_LM_SCALE = 1.0
DECODE_EPOCHS = [0, 2, 5, 10, 20, 30]

# (loop_prob, label_to_blank_prob, epoch_rqmt)
# silence_loop_probability always equals loop_probability.
# label_to_blank_prob=0.0 effectively forbids silence (neg_log(0) = inf).
EXPERIMENTS = [
    (0.0, 0.0, {"mem": 24}),  # no loops, silence forbidden — original baseline
    (0.3, 0.3, {"mem": 24}),  # loops and silence entry equally penalized
    (0.3, 0.5, {"mem": 24}),  # moderate loops, silence less penalized than loops
    (0.5, 0.3, {"mem": 24}),  # strong loops, silence more penalized than loops
    (0.5, 0.5, {"mem": 24}),  # strong loops, silence and loops equally penalized
]


def _prob_tag(p: float) -> str:
    """Format 0.3 → '03', 0.5 → '05' for use in experiment names."""
    return f"{p:.1f}".replace(".", "")


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)

    # Reconstruct the source training result.  Calling build_vq_training with
    # the same hashed arguments gives the same Sisyphus jobs — no recomputation.
    ls100_features = silence_free_ls100_features()
    _, source_result = build_vq_training(
        features=ls100_features.out_features,
        lm_path=phonetic_lm_dict[5],
        sigma=_SOURCE_SIGMA,
        seed=_SOURCE_SEED,
        num_epochs=_SOURCE_NUM_EPOCHS,
        num_chunks=_SOURCE_NUM_CHUNKS,
        lexicon=lexicon,
        alias_prefix=f"guided_kmeans/vq_unsupervised_long/{_SOURCE_EXP_NAME}",
        num_workers=_SOURCE_NUM_WORKERS,
        rqmt={"mem": 8},
        max_beam_size=_SOURCE_BEAM_SIZE,
    )
    warmup_table = source_result.out_artifacts["table"][_SOURCE_NUM_EPOCHS]

    # wav2vecUseg train features filtered to ls-100h.
    train_features_job = FilterHdfByTagsJob(
        feature_hdfs=WAV2VECUSEG_TRAIN_SHARDS,
        keep_tags=LS100H_WAV2VECUSEG_HDF_TAGS,
        rqmt={"cpu": 2, "mem": 16, "time": 4},
    )
    train_features_job.add_alias(f"guided_kmeans/{exp_dir}/filter_train_to_ls100h")

    # CV: same held-out train-other-960 slice as vq_unsupervised_long/wav2vecUseg.
    cv_features = silence_free_cv_features()
    cv_features.add_alias(f"guided_kmeans/{exp_dir}/features_cv_nosil_ref")
    cv_wav2vecUseg_tags = PipelineJob(
        cv_features.out_segments,
        ["sed 's|^\\([^/]*\\)/[^/]*/\\(.*\\)$|\\1/\\2/\\2|'"],
        zip_output=False,
        mini_task=True,
    )
    cv_features_job = FilterHdfByTagsJob(
        feature_hdfs=WAV2VECUSEG_TRAIN_SHARDS,
        keep_tags=cv_wav2vecUseg_tags.out,
        rqmt={"cpu": 2, "mem": 16, "time": 2},
    )
    cv_features_job.add_alias(f"guided_kmeans/{exp_dir}/filter_cv")

    cv_dataset = DatasetConfig(
        audio_hdf_path=cv_features_job.out_features,
        sampling_method=SegmentFile(cv_features.out_segments),
        precomputed=True,
        apply_whitelist=False,
    )

    latex_report = LatexTableReport(
        columns=[
            "loop_prob", "blank_prob", "epoch",
            "mi", "per", "del", "ins", "sub",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["loop_prob", "blank_prob"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Transition-probability sweep: warm-started from epoch {_SOURCE_NUM_EPOCHS} "
            f"of vq\\_unsupervised\\_long (sigma={_SOURCE_SIGMA}, seed={_SOURCE_SEED}, "
            f"5-gram, silence-free ls-100h), finetuned for {NUM_EPOCHS} epochs on "
            f"wav2vecUseg (k=128) segmented ls-100h. silence\\_loop = loop\\_prob always. "
            f"Decode LM scale {DECODE_LM_SCALE}."
        ),
    )
    recog_results = []

    for loop_prob, label_to_blank_prob, epoch_rqmt in EXPERIMENTS:
        exp_name = (
            f"ls100-wav2vecUseg_finetune-from-5gram-s{_SOURCE_SEED}"
            f"_loop{_prob_tag(loop_prob)}_blank{_prob_tag(label_to_blank_prob)}"
        )
        # sigma/seed are required by the signature but unused when initial_table
        # is provided (NormalTableJob is not created in that branch).
        # Pass label_to_blank_prob=0.0 explicitly so neg_log(0.0)=inf during
        # training — otherwise the None fallback would make silence free.
        _, exp_result = build_vq_training(
            features=train_features_job.out_features,
            lm_path=phonetic_lm_dict[5],
            sigma=_SOURCE_SIGMA,
            seed=_SOURCE_SEED,
            num_epochs=NUM_EPOCHS,
            num_chunks=NUM_CHUNKS,
            lexicon=lexicon,
            alias_prefix=f"guided_kmeans/{exp_dir}/{exp_name}",
            num_workers=NUM_WORKERS,
            rqmt=epoch_rqmt,
            max_beam_size=BEAM_SIZE,
            table=warmup_table,
            loop_prob=loop_prob,
            silence_loop_prob=loop_prob,
            label_to_blank_probability=label_to_blank_prob,
        )

        tk.register_output(
            f"guided_kmeans/{exp_dir}/statistics/{exp_name}.json",
            exp_result.out_statistics,
        )
        statistics = clustering_statistics_per_epoch(
            exp_result.out_epoch_statistics,
            name=exp_name,
            epoch_offset=1,
            lexicon=lexicon,
        )
        diagnostics = {
            epoch: MixtureDiagnosticsJob(exp_result.out_artifacts["table"][epoch])
            for epoch in range(0, NUM_EPOCHS + 1)
        }
        for epoch, job in diagnostics.items():
            tk.register_output(
                f"guided_kmeans/{exp_dir}/table_diagnostics/{exp_name}_ep-{epoch}.json",
                job.out_diagnostics,
            )

        recognition_config_decode = build_decode_config(
            phonetic_lm_dict[5], DECODE_LM_SCALE, loop_prob,
            max_beam_size=BEAM_SIZE,
            forbid_blank=(label_to_blank_prob == 0.0),
            decode_silence_loop_prob=loop_prob,
            label_to_blank_probability=label_to_blank_prob if label_to_blank_prob > 0.0 else None,
        )

        for recog_epoch in DECODE_EPOCHS:
            decode_name = f"{exp_name}_ep-{recog_epoch}"
            decode_config = DecodeConfig(
                centroids=COLLEAGUE_CENTROIDS_K512,
                model_dir=exp_result.out_models[recog_epoch],
                recog_rasr_config=recognition_config_decode,
                distance_scale=1.0,
                write_frame_labels=False,
            )
            res = decode_and_score(
                decode_name,
                "cv",
                decode_config,
                cv_dataset,
                rasr_path=tools.RASR_PATH,
                device="cpu",
                corpus_key="train-other-960",
            )
            tk.register_output(
                f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per
            )
            recog_results.append(res)
            latex_report.add_row(
                result=res,
                params={"loop_prob": loop_prob, "blank_prob": label_to_blank_prob},
                epoch=recog_epoch,
                statistics=statistics,
                values={
                    "mi": diagnostics[recog_epoch].out_mi,
                },
            )

    plain_report = create_report(recog_results)
    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=plain_report,
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def py():
    run()
