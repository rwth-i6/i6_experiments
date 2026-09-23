"""Unsupervised VQ training continued to 100 epochs, cheating initialisation.

Identical to :mod:`.vq_unsupervised_long` in every respect except the starting
table: instead of drawing from ``NormalTableJob(sigma, seed)`` the epoch-0 table
is produced by ``SupervisedVQTableJob`` over all silence-free ls-100h segments,
which counts ``p(codeword | phoneme)`` directly from the GMM alignment.

The question this config answers: does the EM converge to a better solution when
it starts from a supervised table rather than a random one, or does the search
inevitably collapse to the same attractor regardless of initialisation?

Because the initial table is deterministic (no sigma/seed randomness), a single
run per LM is enough to establish whether the initialisation matters.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    COLLEAGUE_CENTROIDS_K512,
    GMM_SEGMENT_PHONEMES_LS960,
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
from i6_experiments.example_setups.guided_kmeans.setup.score import (
    GmmSegmentPhonemesReferenceJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
    FrameClusterAccuracyJob,
    SupervisedVQTableJob,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    build_decode_config,
    build_vq_training,
    silence_free_cv_features,
    silence_free_ls100_features,
    NUM_LABELS,
    TABLE_FLOOR,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised_long import (
    NUM_EPOCHS,
    NUM_CHUNKS,
    NUM_WORKERS,
    BEAM_SIZE,
)

exp_dir = "vq_unsupervised_long_cheat_init"
version = 1

#: Pre-computed supervised table from Daniel's pipeline.
DANIEL_TABLE = tk.Path(
    "/work/asr3/michel/mann/experiments/2025-05-30--marten-unsupervised/"
    "i6_experiments/example_setups/guided_kmeans/setup/vq_baseline/"
    "SupervisedVQTableJob.3MrPwzkjlHLl/output/table.npy"
)

#: (lm_name, lm_path, epoch_rqmt, init_suffix, init_table)
#: init_table=None → use the SupervisedVQTableJob counted here from ls-100h.
EXPERIMENTS = [
    ("ours-5gram", phonetic_lm_dict[5], {"mem": 8}, "cheat-init", None),
    ("ours-5gram", phonetic_lm_dict[5], {"mem": 8}, "cheat-init-daniel", DANIEL_TABLE),
   # ("ours-6gram", phonetic_lm_dict[6], {"mem": 30}, "cheat-init", None),
]


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)

    ls100_features = silence_free_ls100_features()
    ls100_features.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_nosil")
    cv_features = silence_free_cv_features()
    cv_features.add_alias(f"guided_kmeans/{exp_dir}/features_cv_nosil")

    gmm_ref_job = GmmSegmentPhonemesReferenceJob(
        gmm_hdf_files=GMM_SEGMENT_PHONEMES_LS960,
        features_hdf=cv_features.out_features,
        lexicon=lexicon,
    )

    # Cheating initial table: p(codeword | phoneme) counted from all ls-100h
    # training segments under the 960h GMM alignment. heldout_fraction=0 to use
    # every segment; this is initialisation, not evaluation.
    cheat_init_job = SupervisedVQTableJob(
        features_hdf=ls100_features.out_features,
        labels=ls100_features.out_labels,
        centroids=COLLEAGUE_CENTROIDS_K512,
        table_floor=TABLE_FLOOR,
        heldout_fraction=0.0,
    )
    cheat_init_job.add_alias(f"guided_kmeans/{exp_dir}/cheat_init_table")
    tk.register_output(
        f"guided_kmeans/{exp_dir}/cheat_init_table/diagnostics.json",
        cheat_init_job.out_diagnostics,
    )

    decode_lm_scale = 1.0
    decode_loop_prob = 0.0
    decode_epochs = [0, 10, 25, 50, 75, 100]

    cv_dataset = DatasetConfig(
        audio_hdf_path=cv_features.out_features,
        sampling_method=SegmentFile(cv_features.out_segments),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=[
            "lm", "epoch",
            "mi", "frame_err", "per", "del", "ins", "sub",
            "per_gmm", "del_gmm", "ins_gmm", "sub_gmm",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["lm"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Unsupervised VQ training with cheating initialisation from the GMM "
            f"alignment over ls-100h, continued to {NUM_EPOCHS} epochs, decoded on "
            f"silence-free cv at LM scale {decode_lm_scale}. The initial table "
            f"(epoch 0) is p(codeword|phoneme) counted from all training segments; "
            f"subsequent epochs follow the same EM as :mod:`.vq_unsupervised_long`."
        ),
    )
    recog_results = []
    frame_err_vars = []

    for lm_name, lm_path, epoch_rqmt, init_suffix, init_table in EXPERIMENTS:
        if init_table is None:
            init_table = cheat_init_job.out_table
        exp_name = f"ls100-nosil_{lm_name}_{init_suffix}"
        _, exp_result = build_vq_training(
            features=ls100_features.out_features,
            lm_path=lm_path,
            sigma=0.0,   # unused: table overrides NormalTableJob
            seed=0,      # unused: table overrides NormalTableJob
            num_epochs=NUM_EPOCHS,
            num_chunks=NUM_CHUNKS,
            lexicon=lexicon,
            alias_prefix=f"guided_kmeans/{exp_dir}/{exp_name}",
            num_workers=NUM_WORKERS,
            rqmt=epoch_rqmt,
            max_beam_size=BEAM_SIZE,
            table=init_table,
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
            max_beam_size=BEAM_SIZE, forbid_blank=True,
        )
        for recog_epoch in decode_epochs:
            decode_name = f"{exp_name}_ep-{recog_epoch}"
            frame_err_job = FrameClusterAccuracyJob(
                features_hdf=cv_features.out_features,
                alignment=cv_features.out_labels,
                centroids=COLLEAGUE_CENTROIDS_K512,
                table=exp_result.out_artifacts["table"][recog_epoch],
            )
            tk.register_output(
                f"guided_kmeans/{exp_dir}/frame_err/{decode_name}.json",
                frame_err_job.out_diagnostics,
            )

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
                gmm_segment_ref=gmm_ref_job.out_ref,
            )
            tk.register_output(
                f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per
            )
            if res.per_gmm is not None:
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/per_gmm/{decode_name}_per", res.per_gmm
                )
            recog_results.append(res)
            frame_err_vars.append(frame_err_job.out_error_rate)
            latex_report.add_row(
                result=res,
                params={"lm": lm_name},
                epoch=recog_epoch,
                statistics=statistics,
                values={
                    "mi": diagnostics[recog_epoch].out_mi,
                    "frame_err": frame_err_job.out_error_rate,
                },
            )

    plain_report = create_report(recog_results)
    for idx, (res, err_var) in enumerate(zip(recog_results, frame_err_vars), start=1):
        if err_var is not None:
            plain_report.add_entry(
                col="6 Frame err.", row=f"{idx}_{res.descriptor}", var=err_var
            )
    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=plain_report,
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def py():
    run()
