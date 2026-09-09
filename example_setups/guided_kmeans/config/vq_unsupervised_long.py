"""Long continuation of the promising unsupervised VQ runs.

:mod:`.vq_unsupervised` established, over 10 epochs on ls-100h silence-free
features, that the initialization width decides everything:

    sigma  LM        seed   epoch 0 -> 10 PER
    0.1    ours       42     87.9 -> 83.2
    0.1    ours       43     87.6 -> 82.0
    0.1    zijian     42     89.7 -> 82.2
    0.1    zijian     43     89.8 -> 82.8
    1.0    ours       42     86.3 -> 85.2
    1.0    ours       43     85.9 -> 86.0
    1.0    zijian     42     87.0 -> 87.0
    1.0    zijian     43     86.8 -> 87.5

Every sigma=0.1 run moves 5-7 points; no sigma=1.0 run moves at all. Which is
the *opposite* of the naive reading of "low variance is a weak symmetry break" -
at C=512 a Dirichlet(1.0) draw has coefficient of variation near 1.0, so
sigma=1.0 is the *wide* initialization, and it appears to be wide enough that
the first search has no coherent structure to sharpen. sigma=0.1 starts near
uniform but not flat, and that turns out to be the productive regime.

This config continues the interesting subset for 100 epochs.

**Continuation is free and this config relies on it.** ``num_epochs`` is not
part of an epoch job's identity, so epochs 1-10 here are the *same jobs*
already computed by :mod:`.vq_unsupervised` - they are reused, not recomputed,
and only epochs 11-100 are new. That only holds while every other argument
matches exactly, which is why both configs call
:func:`.vq_unsupervised.build_vq_training` rather than keeping two copies of the
same argument list in step by hand.

``num_chunks`` is raised from 50 to 100 for more parallelism per epoch. It is
excluded from the job hash - merging counts is associative, so the partition
cannot change the result - so this does not orphan anything either.

Note on the run count: the selection below is *five* runs, not four. Taking the
request item by item - sigma=1.0 with Zijian and both seeds (2), plus sigma=0.1
with Zijian and one seed (1) and with our LM and both seeds (2) - gives five.
Trim ``EXPERIMENTS`` if four was meant; the arithmetic is the only thing that
disagreed, each item on its own was unambiguous.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    COLLEAGUE_CENTROIDS_K512,
    PHONEME_LM_ZIJIAN_3GRAM,
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

exp_dir = "vq_unsupervised_long"
version = 1

#: (lm_name, lm_path, sigma, seed, epoch_rqmt). Seed 42 for the sigma=0.1
#: Zijian arm because it was the better of the two at epoch 10 (82.16 against
#: 82.77) - a thin margin, so treat it as a tie broken arbitrarily rather than
#: as a result.
#:
#: epoch_rqmt["mem"] is estimated as: base ~5.6 GB + 9 workers * LM_size.
#: LM sizes (uncompressed): 5gram=231MB, 6gram=1.5GB, 7gram=6.3GB.
#: RASR loads each LM independently per worker (no mmap sharing observed).
#: 8gram (~18GB uncompressed) and 9gram (~40GB) are omitted: ~170/366 GB needed.
EXPERIMENTS = [
    ("ours-5gram", phonetic_lm_dict[5], 0.1,  42, {"mem": 8}),
    ("ours-5gram", phonetic_lm_dict[5], 0.1,  43, {"mem": 8}),
    ("ours-5gram", phonetic_lm_dict[5], 0.02, 42, {"mem": 8}),
    ("ours-5gram", phonetic_lm_dict[5], 0.02, 43, {"mem": 8}),
    ("ours-6gram", phonetic_lm_dict[6], 0.02, 42, {"mem": 30}),
    ("ours-6gram", phonetic_lm_dict[6], 0.02, 43, {"mem": 30}),
    # 7gram: base ~5.6 GB + 9 workers * 6.3 GB LM = ~62 GB; 80 GB gives ~30% headroom.
    # Consider reducing NUM_WORKERS (e.g. to 4-5) to bring this down to ~32-37 GB.
    # Uncomment once 6gram results are in and the approach is confirmed.
    # ("ours-7gram", phonetic_lm_dict[7], 0.02, 42, {"mem": 80}),
    # ("ours-7gram", phonetic_lm_dict[7], 0.02, 43, {"mem": 80}),
    # ("ours-8gram", phonetic_lm_dict[8], 0.02, 42, {"mem": 192}),  # ~18GB LM
    # ("ours-8gram", phonetic_lm_dict[8], 0.02, 43, {"mem": 192}),
    # ("ours-9gram", phonetic_lm_dict[9], 0.02, 42, {"mem": 384}),  # ~40GB LM
    # ("ours-9gram", phonetic_lm_dict[9], 0.02, 43, {"mem": 384}),
]

NUM_EPOCHS = 100

# --- scheduling, sized against this cluster's measured behaviour -------------
# None of these change a job hash (see build_vq_training), so the 10 epochs
# already computed are reused whatever they are set to.
#
# The QOS caps this user at 1100 CPUs. With 2 runs, the ceiling per run is
# 1100 / 2 = 550 CPUs. At 10 CPUs per chunk that is 55 chunks; 50 stays just
# under and matches the chunk count used in the 3-gram short runs.
NUM_CHUNKS = 30
# 5-gram LM: state space is 40^4 = 2.56M vs 40^2 = 1600 for 3-gram, so beam
# pruning is needed. 1000 targets a ~30-min epoch, same order as the 3-gram run.
BEAM_SIZE = 1000

# 9, not 8. The task requests num_workers + 1 CPUs, and Slurm rounds an
# allocation up to an even core count - cpu=9 was being given AllocCPUS=10, so
# one core per task was paid for and never used. cpu=10 asks for what was
# already being allocated and gets a ninth search process for free.
NUM_WORKERS = 9

# Per-experiment epoch memory is set in EXPERIMENTS (see above). The CPU count
# does not change: num_workers + 1 = 10 CPUs per chunk regardless of LM order.


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)

    # The same jobs vq_unsupervised builds, so the feature files are shared
    # rather than rebuilt.
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
    # Sparse on purpose: 100 epochs x 5 runs would otherwise be 500 decodes, and
    # the per-epoch statistics already say whether a run is still moving. Epoch
    # 10 is kept so every curve has a point directly comparable to the short run.
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
            f"Unsupervised discrete-HMM training over a frozen 512-entry codebook, "
            f"continued to {NUM_EPOCHS} epochs on silence-free ls-100h and decoded on "
            f"silence-free cv at LM scale {decode_lm_scale}. Epochs 1-10 are the same "
            f"jobs as in the short run and are reused, not recomputed. The supervised "
            f"table over the same codebook reaches 82.4\\% held-out segment accuracy "
            f"and 14.4\\% PER, which is the ceiling these runs are read against. "
            f"'mi' is I(label; codeword) under the model's own table, reference-free "
            f"and 0 for the degenerate all-labels-alike solution."
        ),
    )
    recog_results = []
    frame_acc_vars = []  # parallel list; None entries for rows without frame_acc

    for lm_name, lm_path, sigma, seed, epoch_rqmt in EXPERIMENTS:
        exp_name = f"ls100-nosil_{lm_name}_sigma-{sigma}_seed-{seed}"
        _, exp_result = build_vq_training(
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
        # Every epoch, not just the decoded ones: these are mini-tasks over a
        # [40, 512] array, and a flat I(label; codeword) is what distinguishes a
        # run that has converged from one that never started.
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

    # --- Supervised baseline: zyang's GMM-counted table for the k=512 codebook ---
    # log p(cluster | phoneme), shape [512, 40]. Transposed + exponentiated by the
    # job into [40, 512] p(codeword | label) for VectorQuantizedModel.
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
    supervised_table_job.add_alias(
        f"guided_kmeans/{exp_dir}/supervised_table/zyang_k512"
    )
    supervised_mi = MixtureDiagnosticsJob(supervised_table_job.out_table)
    tk.register_output(
        f"guided_kmeans/{exp_dir}/table_diagnostics/supervised_zyang.json",
        supervised_mi.out_diagnostics,
    )
    supervised_frame_acc = FrameClusterAccuracyJob(
        features_hdf=cv_features.out_features,
        alignment=cv_features.out_labels,
        centroids=COLLEAGUE_CENTROIDS_K512,
        table=supervised_table_job.out_table,
    )
    tk.register_output(
        f"guided_kmeans/{exp_dir}/frame_acc/supervised_zyang.json",
        supervised_frame_acc.out_diagnostics,
    )

    supervised_decode_config = DecodeConfig(
        centroids=COLLEAGUE_CENTROIDS_K512,
        model_dir=supervised_table_job.out_model,
        recog_rasr_config=build_decode_config(
            phonetic_lm_dict[5],
            decode_lm_scale,
            decode_loop_prob,
            max_beam_size=BEAM_SIZE,
            forbid_blank=True,
        ),
        distance_scale=1.0,
        write_frame_labels=True,
    )
    supervised_decode_name = "supervised_zyang-5gram"
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
        params={"lm": "ours-5gram", "sigma": "supervised", "seed": "-"},
        epoch=None,
        values={
            "mi": supervised_mi.out_mi,
            "frame_err": supervised_frame_acc.out_error_rate,
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
