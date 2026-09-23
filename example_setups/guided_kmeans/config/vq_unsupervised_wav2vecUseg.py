"""Unsupervised VQ training on wav2vec-U-style k-means segmented features.

Uses precomputed mean-pooled features from schmitt's pipeline, which exactly
mirrors the wav2vec-U paper:
  1. Silence removed from audio (Wav2VecUDeleteSilencesInAudioJob)
  2. wav2vec2-large (60kh, no fine-tuning) features extracted at layer 14
  3. k=128 k-means on 1024-dim features for boundary detection
  4. 512-dim PCA features mean-pooled over those boundaries

Train: schmitt's full ls-960h HDF (DumpNumpyFeaturesToHdfJobV2.1U3cnBFDVhdm)
       filtered to the ls-100h subset via FilterHdfByTagsJob (28,234 sequences).
CV:    same ls-960h HDF filtered to the same CV subset used by vq_unsupervised_long
       (silence_free_cv_features().out_segments, a held-out slice of train-other-960)
       so results are directly comparable across configs.
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
)

exp_dir = "vq_unsupervised_wav2vecUseg"
version = 1

_WAV2VECUSEG_SOURCE_DIR = (
    "/u/schmitt/experiments/2026_04_09_unsupervised_asr/work"
    "/i6_experiments/users/schmitt/experiments/exp2025_10_02_shared_enc"
    "/librispeech/data/wav2vec"
)

# ls-960h train shards (tagged train-other-960/UTT/UTT)
WAV2VECUSEG_TRAIN_SHARDS = [
    tk.Path(
        f"{_WAV2VECUSEG_SOURCE_DIR}/DumpNumpyFeaturesToHdfJobV2.1U3cnBFDVhdm/output/data_{i}.hdf"
    )
    for i in range(10)
]

# Static filter file: ls-100h tags in schmitt's train-other-960/UTT/UTT format.
# Generated once from ls100h-segments.txt cross-referenced with the HDF tags.
LS100H_WAV2VECUSEG_HDF_TAGS = tk.Path(
    "/u/lkleppel/experiments/20260520_unsupervised_asr/output"
    "/guided_kmeans/w2vu_segmentation/ls100h_wav2vecUseg_hdf_tags.txt"
)

EXPERIMENTS = [
    ("ours-5gram", phonetic_lm_dict[5], 0.02, 42, {"mem": 24}),
  #  ("ours-5gram", phonetic_lm_dict[5], 0.02, 43, {"mem": 8}),
   # ("ours-6gram", phonetic_lm_dict[6], 0.02, 42, {"mem": 30}),
   # ("ours-6gram", phonetic_lm_dict[6], 0.02, 43, {"mem": 30}),
]

NUM_EPOCHS = 100
NUM_CHUNKS = 30
BEAM_SIZE = 1000
NUM_WORKERS = 12


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)

    # Filter ls-960h shards to ls-100h
    train_features_job = FilterHdfByTagsJob(
        feature_hdfs=WAV2VECUSEG_TRAIN_SHARDS,
        keep_tags=LS100H_WAV2VECUSEG_HDF_TAGS,
        rqmt={"cpu": 2, "mem": 16, "time": 4},
    )
    train_features_job.add_alias(
        f"guided_kmeans/{exp_dir}/filter_train_to_ls100h"
    )
    tk.register_output(
        f"guided_kmeans/{exp_dir}/train_features_ls100h.hdf",
        train_features_job.out_features,
    )

    # CV: same subset as vq_unsupervised_long (a held-out slice of train-other-960).
    # silence_free_cv_features().out_segments has standard corpus/speaker/utt tags;
    # schmitt's HDF uses corpus/utt/utt — convert before filtering.
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

    decode_lm_scale = 1.0
    decode_loop_prob = 0.3
    decode_silence_loop_prob = 0.3
    decode_label_to_blank_prob = 0.3
    decode_epochs = [0, 10, 25, 50, 75, 100]

    cv_dataset = DatasetConfig(
        audio_hdf_path=cv_features_job.out_features,
        # SegmentFile filters the bliss corpus for ref generation; the HDF is
        # already pre-filtered, so apply_whitelist=False keeps it from being
        # filtered a second time (with mismatched tags).
        sampling_method=SegmentFile(cv_features.out_segments),
        precomputed=True,
        apply_whitelist=False,
    )

    latex_report = LatexTableReport(
        columns=[
            "lm", "sigma", "seed", "epoch",
            "mi", "per", "del", "ins", "sub",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["sigma", "lm", "seed"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Unsupervised discrete-HMM training on wav2vec-U k=128 segmented "
            f"ls-100h features (schmitt pipeline: silence-removed, layer-14, "
            f"512-dim PCA mean-pooled over k-means boundaries), decoded on the "
            f"same held-out train-other-960 CV subset as vq\\_unsupervised\\_long "
            f"at LM scale {decode_lm_scale}. "
            f"Compare against vq\\_unsupervised\\_long to isolate the effect of the "
            f"boundary detector."
        ),
    )
    recog_results = []

    for lm_name, lm_path, sigma, seed, epoch_rqmt in EXPERIMENTS:
        exp_name = f"ls100-wav2vecUseg_{lm_name}_sigma-{sigma}_seed-{seed}"
        _, exp_result = build_vq_training(
            features=train_features_job.out_features,
            lm_path=lm_path,
            sigma=sigma,
            seed=seed,
            num_epochs=NUM_EPOCHS,
            num_chunks=NUM_CHUNKS,
            lexicon=lexicon,
            alias_prefix=f"guided_kmeans/{exp_dir}/{exp_name}",
            num_workers=NUM_WORKERS,
            rqmt=epoch_rqmt,
            max_beam_size=BEAM_SIZE,
            loop_prob=0.3,
            silence_loop_prob=0.3,
            label_to_blank_probability=0.3,
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
            lm_path, decode_lm_scale, decode_loop_prob,
            max_beam_size=BEAM_SIZE, forbid_blank=False,
            decode_silence_loop_prob=decode_silence_loop_prob,
            label_to_blank_probability=decode_label_to_blank_prob,
        )
        for recog_epoch in decode_epochs:
            decode_name = f"{exp_name}_ep-{recog_epoch}"
            decode_config = DecodeConfig(
                centroids=COLLEAGUE_CENTROIDS_K512,
                model_dir=exp_result.out_models[recog_epoch],
                recog_rasr_config=recognition_config_decode,
                distance_scale=1.0,
                write_frame_labels=True,
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
                params={"lm": lm_name, "sigma": sigma, "seed": seed},
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
