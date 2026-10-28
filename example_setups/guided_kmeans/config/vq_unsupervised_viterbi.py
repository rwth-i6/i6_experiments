"""Unsupervised VQ training with Viterbi (maximum-approximation) search.

Same model as :mod:`.vq_unsupervised_long` — frozen 512-entry codebook, table
``p(codeword | phoneme)`` estimated from Viterbi alignments — but the E-step
uses the linear search (max approximation) instead of the forward-backward
fullsum.

Only the 5-gram LM with two seeds is run here to keep the scope small and to
directly compare against the equivalent FB runs in :mod:`.vq_unsupervised_long`.

**Feature sharing:** :func:`.vq_unsupervised.silence_free_ls100_features` and
:func:`.vq_unsupervised.silence_free_cv_features` build the same
:class:`.SegmentedFeaturesFromAlignmentJob` objects as the FB config, so the
segmented HDF files are reused rather than rebuilt.

**No continuation from the FB runs.** The RASR search config is part of the
epoch job hash (via the recognition config), so Viterbi epoch jobs are distinct
from their FB counterparts and start from scratch.
"""

from typing import Optional

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    COLLEAGUE_CENTROIDS_K512,
    GMM_SEGMENT_PHONEMES_LS960,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    NormalTableJob,
    chunked_clustering,
    vq_flavor,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_recog_rasr_config,
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
)
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
    silence_free_ls100_features,
    build_decode_config,
    NUM_LABELS,
    NUM_CODEWORDS,
    LM_SCALE,
    DISTANCE_SCALE,
    TABLE_FLOOR,
    LOOP_PROB,
    USE_EOW_PHONEMES,
    #BEAM_SIZE as BASE_BEAM_SIZE,
)

exp_dir = "vq_unsupervised_viterbi"
version = 1

EXPERIMENTS = [
    # (lm_name, lm_path, sigma, seed, epoch_rqmt)
    ("ours-5gram", phonetic_lm_dict[5], 0.02, 42, {"mem": 12}),
    ("ours-5gram", phonetic_lm_dict[5], 0.02, 43, {"mem": 12}),
]

NUM_EPOCHS = 100
NUM_CHUNKS = 30
BEAM_SIZE = 1000   # 5-gram state space (40^4 = 2.56M) needs beam pruning
NUM_WORKERS = 9    # request 10 CPUs (num_workers + 1), matches Slurm even-rounding


def build_vq_training_viterbi(
    *,
    features,
    lm_path,
    sigma,
    seed,
    num_epochs,
    num_chunks,
    lexicon,
    alias_prefix,
    num_workers=NUM_WORKERS,
    rqmt=None,
    max_beam_size: int = BEAM_SIZE,
    initial_table: Optional[tk.Path] = None,
):
    """Same as :func:`.vq_unsupervised.build_vq_training` but with Viterbi search.

    ``use_forward_backward=False`` is passed throughout and the linear-search
    RASR binary is used for training. The decode always uses the linear-search
    binary regardless of training mode, so decode is unchanged.
    """
    recognition_config = create_recog_rasr_config(
        lm_scale=LM_SCALE,
        emission_scale=1.0,
        transition_scale=LM_SCALE,
        loop_probability=LOOP_PROB,
        silence_loop_probability=LOOP_PROB,
        use_forward_backward_search=False,
        lm_order=3,
        use_eow_phonemes=USE_EOW_PHONEMES,
        max_beam_size=max_beam_size,
        lm_path=lm_path,
    )
    if initial_table is None:
        initial_table = NormalTableJob(NUM_LABELS, NUM_CODEWORDS, sigma=sigma, seed=seed).out_table
    flavor = vq_flavor(
        centroids=COLLEAGUE_CENTROIDS_K512,
        table=initial_table,
        recognition_config=recognition_config,
        lexicon=lexicon,
        num_clusters=NUM_LABELS,
        distance_scale=DISTANCE_SCALE,
        use_forward_backward=False,
        table_floor=TABLE_FLOOR,
        num_workers=num_workers,
    )
    exp_result = chunked_clustering(
        num_epochs=num_epochs,
        features_hdf=features,
        recognition_config=recognition_config,
        lexicon=lexicon,
        num_clusters=NUM_LABELS,
        flavor=flavor,
        subsampling=None,
        distance_scale=DISTANCE_SCALE,
        use_forward_backward=False,
        rasr_path=tools.RASR_PATH,
        num_chunks=num_chunks,
        num_workers=num_workers,
        rqmt=rqmt,
        alias_prefix=alias_prefix,
    )
    return recognition_config, exp_result


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
            "lm", "sigma", "seed", "epoch",
            "mi", "frame_err", "per", "del", "ins", "sub",
            "per_gmm", "del_gmm", "ins_gmm", "sub_gmm",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["sigma", "lm", "seed"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Unsupervised discrete-HMM training with Viterbi (max-approximation) "
            f"search over a frozen 512-entry codebook, {NUM_EPOCHS} epochs on "
            f"silence-free ls-100h, decoded on silence-free cv at LM scale "
            f"{decode_lm_scale}. Compare against the forward-backward runs in "
            f"vq_unsupervised_long."
        ),
    )
    recog_results = []
    frame_acc_vars = []

    for lm_name, lm_path, sigma, seed, epoch_rqmt in EXPERIMENTS:
        exp_name = f"ls100-nosil_{lm_name}_sigma-{sigma}_seed-{seed}"
        _, exp_result = build_vq_training_viterbi(
            features=ls100_features.out_features,
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
            lm_path, decode_lm_scale, decode_loop_prob, max_beam_size=BEAM_SIZE, forbid_blank=True
        )
        for recog_epoch in decode_epochs:
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
                params={"lm": lm_name, "sigma": sigma, "seed": seed},
                epoch=recog_epoch,
                statistics=statistics,
                values={
                    "mi": diagnostics[recog_epoch].out_mi,
                    "frame_err": frame_acc_job.out_error_rate,
                },
            )

    plain_report = create_report(recog_results)
    for idx, (res, acc_var) in enumerate(zip(recog_results, frame_acc_vars), start=1):
        if acc_var is not None:
            plain_report.add_entry(
                col="6 Frame err.", row=f"{idx}_{res.descriptor}", var=acc_var
            )
    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=plain_report,
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def py():
    run()
