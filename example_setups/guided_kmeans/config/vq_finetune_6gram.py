"""6-gram LM fine-tuning of the converged 5-gram VQ run.

Warm-starts from the epoch-100 table of the sigma=0.02, seed=42, 5-gram run in
:mod:`.vq_unsupervised_long` and continues training on the same silence-free
ls-100h features with the 6-gram LM. Asks whether switching to a sharper LM
after convergence improves further. The supervised table decoded with the
6-gram LM is included as the ceiling to read it against.

The source run is rebuilt here with :mod:`.vq_unsupervised_long`'s own
constants, so it is the same Sisyphus jobs and its finished epochs are reused.
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
    ExternalLogPtTableJob,
    FrameClusterAccuracyJob,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    build_decode_config,
    build_vq_training,
    silence_free_cv_features,
    silence_free_ls100_features,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised_long import (
    BEAM_SIZE,
    NUM_CHUNKS,
    NUM_EPOCHS,
    NUM_WORKERS,
)

exp_dir = "vq_finetune_6gram"
version = 1

SOURCE_SIGMA = 0.02
SOURCE_SEED = 42
SOURCE_EXP_NAME = f"ls100-nosil_ours-5gram_sigma-{SOURCE_SIGMA}_seed-{SOURCE_SEED}"

DECODE_LM_SCALE = 1.0
DECODE_LOOP_PROB = 0.0
DECODE_EPOCHS = [0, 10, 25, 50, 75, 100]


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    ls100_features = silence_free_ls100_features()
    cv_features = silence_free_cv_features()

    gmm_ref_job = GmmSegmentPhonemesReferenceJob(
        gmm_hdf_files=GMM_SEGMENT_PHONEMES_LS960,
        features_hdf=cv_features.out_features,
        lexicon=lexicon,
    )
    cv_dataset = DatasetConfig(
        audio_hdf_path=cv_features.out_features,
        sampling_method=SegmentFile(cv_features.out_segments),
        precomputed=True,
    )

    # Same hashed arguments as the run in vq_unsupervised_long, so these are its
    # finished jobs; rqmt and the alias do not enter the hash.
    _, source_result = build_vq_training(
        features=ls100_features.out_features,
        lm_path=phonetic_lm_dict[5],
        sigma=SOURCE_SIGMA,
        seed=SOURCE_SEED,
        num_epochs=NUM_EPOCHS,
        num_chunks=NUM_CHUNKS,
        lexicon=lexicon,
        alias_prefix=f"guided_kmeans/vq_unsupervised_long/{SOURCE_EXP_NAME}",
        num_workers=NUM_WORKERS,
        rqmt={"mem": 24},
        max_beam_size=BEAM_SIZE,
    )

    latex_report = LatexTableReport(
        columns=[
            "lm", "sigma", "seed", "epoch",
            "mi", "frame_err", "per", "del", "ins", "sub",
            "per_gmm", "del_gmm", "ins_gmm", "sub_gmm",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["sigma", "lm", "seed"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"6-gram LM fine-tuning, warm-started from the epoch-{NUM_EPOCHS} table of "
            f"the 5-gram sigma={SOURCE_SIGMA} seed={SOURCE_SEED} run and trained for "
            f"{NUM_EPOCHS} more epochs on silence-free ls-100h; decoded on silence-free "
            f"cv at LM scale {DECODE_LM_SCALE}, against the supervised table decoded "
            f"with the same 6-gram LM."
        ),
    )
    recog_results = []
    frame_acc_vars = []

    exp_name = f"ls100-nosil_finetune-from-5gram-s{SOURCE_SEED}_ours-6gram"
    # sigma/seed are unused here: `table` replaces the NormalTableJob draw.
    _, exp_result = build_vq_training(
        features=ls100_features.out_features,
        lm_path=phonetic_lm_dict[6],
        sigma=SOURCE_SIGMA,
        seed=SOURCE_SEED,
        num_epochs=NUM_EPOCHS,
        num_chunks=NUM_CHUNKS,
        lexicon=lexicon,
        alias_prefix=f"guided_kmeans/{exp_dir}/{exp_name}",
        num_workers=NUM_WORKERS,
        rqmt={"mem": 30},
        max_beam_size=BEAM_SIZE,
        table=source_result.out_artifacts["table"][NUM_EPOCHS],
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
        phonetic_lm_dict[6], DECODE_LM_SCALE, DECODE_LOOP_PROB,
        max_beam_size=BEAM_SIZE, forbid_blank=True,
    )
    for recog_epoch in DECODE_EPOCHS:
        decode_name = f"{exp_name}_ep-{recog_epoch}"
        frame_acc_job = FrameClusterAccuracyJob(
            features_hdf=cv_features.out_features,
            alignment=cv_features.out_labels,
            centroids=COLLEAGUE_CENTROIDS_K512,
            table=exp_result.out_artifacts["table"][recog_epoch],
        )
        tk.register_output(
            f"guided_kmeans/{exp_dir}/frame_acc/{decode_name}.json",
            frame_acc_job.out_diagnostics,
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
        tk.register_output(f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per)
        if res.per_gmm is not None:
            tk.register_output(
                f"guided_kmeans/{exp_dir}/per_gmm/{decode_name}_per", res.per_gmm
            )
        recog_results.append(res)
        frame_acc_vars.append(frame_acc_job.out_error_rate)
        latex_report.add_row(
            result=res,
            params={"lm": "ours-6gram", "sigma": "finetune", "seed": SOURCE_SEED},
            epoch=recog_epoch,
            statistics=statistics,
            values={
                "mi": diagnostics[recog_epoch].out_mi,
                "frame_err": frame_acc_job.out_error_rate,
            },
        )

    # --- Supervised table decoded with the 6-gram LM, the ceiling for this run ---
    # zyang's GMM-counted table for the k=512 codebook: log p(cluster | phoneme),
    # [512, 40], turned by the job into [40, 512] p(codeword | label). Same job as
    # the supervised baseline in vq_unsupervised_long.
    supervised_table_job = ExternalLogPtTableJob(
        pt_table=tk.Path(
            "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/"
            "generative_ctc/example_setups/librispeech/phmm/"
            "gmm_alignment_vad_filter_jobs/"
            "BuildAmInitFromCountsJob.ctdAwDvFREto/output/"
            "am_init_log_p_cluster_given_phoneme.pt"
        ),
        centroids=COLLEAGUE_CENTROIDS_K512,
    )
    supervised_mi = MixtureDiagnosticsJob(supervised_table_job.out_table)
    supervised_frame_acc = FrameClusterAccuracyJob(
        features_hdf=cv_features.out_features,
        alignment=cv_features.out_labels,
        centroids=COLLEAGUE_CENTROIDS_K512,
        table=supervised_table_job.out_table,
    )
    supervised_decode_config = DecodeConfig(
        centroids=COLLEAGUE_CENTROIDS_K512,
        model_dir=supervised_table_job.out_model,
        recog_rasr_config=build_decode_config(
            phonetic_lm_dict[6], DECODE_LM_SCALE, DECODE_LOOP_PROB,
            max_beam_size=BEAM_SIZE, forbid_blank=True,
        ),
        distance_scale=1.0,
        write_frame_labels=True,
    )
    supervised_decode_name = "supervised_zyang-6gram"
    supervised_res = decode_and_score(
        supervised_decode_name,
        "cv",
        supervised_decode_config,
        cv_dataset,
        rasr_path=tools.RASR_PATH,
        device="cpu",
        corpus_key="train-other-960",
        gmm_segment_ref=gmm_ref_job.out_ref,
    )
    tk.register_output(
        f"guided_kmeans/{exp_dir}/per/{supervised_decode_name}_per", supervised_res.per
    )
    if supervised_res.per_gmm is not None:
        tk.register_output(
            f"guided_kmeans/{exp_dir}/per_gmm/{supervised_decode_name}_per",
            supervised_res.per_gmm,
        )
    recog_results.append(supervised_res)
    frame_acc_vars.append(supervised_frame_acc.out_error_rate)
    latex_report.add_row(
        result=supervised_res,
        params={"lm": "ours-6gram", "sigma": "supervised", "seed": "-"},
        epoch=None,
        values={
            "mi": supervised_mi.out_mi,
            "frame_err": supervised_frame_acc.out_error_rate,
        },
    )

    plain_report = create_report(recog_results)
    for idx, (res, acc_var) in enumerate(zip(recog_results, frame_acc_vars), start=1):
        plain_report.add_entry(col="6 Frame err.", row=f"{idx}_{res.descriptor}", var=acc_var)
    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=plain_report,
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def py():
    run()
