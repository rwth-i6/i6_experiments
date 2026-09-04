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
    GMM_ALIGNMENT_CV,
    COLLEAGUE_CENTROIDS_K512,
    PHONEME_LM_ZIJIAN_3GRAM,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
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
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    build_decode_config,
    build_vq_training,
    silence_free_cv_features,
    silence_free_ls100_features,
)

exp_dir = "vq_unsupervised_long"
version = 1

#: (lm_name, lm_path, sigma, seed). Seed 42 for the sigma=0.1 Zijian arm because
#: it was the better of the two at epoch 10 (82.16 against 82.77) - a thin
#: margin, so treat it as a tie broken arbitrarily rather than as a result.
EXPERIMENTS = [
    ("zijian-3gram", PHONEME_LM_ZIJIAN_3GRAM, 1.0, 42),
    ("zijian-3gram", PHONEME_LM_ZIJIAN_3GRAM, 1.0, 43),
    ("zijian-3gram", PHONEME_LM_ZIJIAN_3GRAM, 0.1, 42),
    ("ours-3gram", None, 0.1, 42),
    ("ours-3gram", None, 0.1, 43),
]

NUM_EPOCHS = 100

# --- scheduling, sized against this cluster's measured behaviour -------------
# None of these change a job hash (see build_vq_training), so the 10 epochs
# already computed are reused whatever they are set to.
#
# The QOS caps this user at 1100 CPUs. An epoch job of 100 tasks x 10 CPU takes
# ~1000 of them, so exactly one epoch job runs at a time and the other four runs
# queue behind it - measured: consecutive arrays started 8 minutes apart, each
# spending ~25 min queued against ~6 min computing. The chunks themselves are
# fine (max/median 1.1x, 89-90% utilisation within an array); what idles the
# quota is the gap between one array ending and the next being scheduled.
#
# So size an epoch job to a fifth of the quota and let all five runs overlap:
# one run's scheduling gap becomes another run's compute.
NUM_CHUNKS = 22          # 22 x 10 CPU = 220; x 5 runs = 1100, the whole quota

# 9, not 8. The task requests num_workers + 1 CPUs, and Slurm rounds an
# allocation up to an even core count - cpu=9 was being given AllocCPUS=10, so
# one core per task was paid for and never used. cpu=10 asks for what was
# already being allocated and gets a ninth search process for free.
NUM_WORKERS = 9

# Measured peak RSS is 5 GB; the 16 GB default reserves three times that. Not
# binding while CPU is, but at 110 concurrent tasks it would reserve 1.76 TB of
# the 2.2 TB QOS memory cap, which is the next thing that would bind.
EPOCH_RQMT = {"mem": 8}


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)

    # The same jobs vq_unsupervised builds, so the feature files are shared
    # rather than rebuilt.
    ls100_features = silence_free_ls100_features()
    ls100_features.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_nosil")
    cv_features = silence_free_cv_features()
    cv_features.add_alias(f"guided_kmeans/{exp_dir}/features_cv_nosil")

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
            "mi", "per", "del", "ins", "sub", "fer",
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

    for lm_name, lm_path, sigma, seed in EXPERIMENTS:
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
            rqmt=EPOCH_RQMT,
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
            lm_path, decode_lm_scale, decode_loop_prob
        )
        for recog_epoch in decode_epochs:
            decode_config = DecodeConfig(
                centroids=COLLEAGUE_CENTROIDS_K512,
                model_dir=exp_result.out_models[recog_epoch],
                recog_rasr_config=recognition_config_decode,
                distance_scale=1.0,
                write_frame_labels=True,
            )
            decode_name = f"{exp_name}_ep-{recog_epoch}"
            res = decode_and_score(
                decode_name,
                "cv",
                decode_config,
                cv_dataset,
                rasr_path=tools.RASR_PATH,
                device="cpu",
                corpus_key="train-other-960",
            )
            if res.frame_labels is not None:
                res.fer = FrameErrorRateJob(
                    res.frame_labels, GMM_ALIGNMENT_CV, lexicon
                ).out_fer
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/eval/{decode_name}_fer", res.fer
                )
            tk.register_output(f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per)
            recog_results.append(res)
            latex_report.add_row(
                result=res,
                params={"lm": lm_name, "sigma": sigma, "seed": seed},
                epoch=recog_epoch,
                statistics=statistics,
                values={
                    k: v
                    for k, v in (("mi", diagnostics[recog_epoch].out_mi), ("fer", res.fer))
                    if v is not None
                },
            )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def py():
    run()
