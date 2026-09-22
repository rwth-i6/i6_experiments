"""VQ unsupervised training with chunked batched updates.

Same idea as :mod:`.vq_unsupervised_batched_ema` - re-estimate B times per pass
over the corpus instead of once - but built out of the ordinary chunked epoch
job, so a batch is still spread over the cluster instead of being streamed
through one node's worker pool.

**Why this exists.** Recognition costs the same per sequence either way; what
differs is how much of it runs at once. Measured on the full-epoch runs, a
sequence costs ~11.9 s of worker CPU, so a pass over the 28.2 k sequences of
silence-free ls-100h is ~93 CPU-hours. The single-task job spends that at its
own ``num_workers`` (10 in :mod:`.vq_unsupervised_batched_ema`, so ~9.3 h per
epoch); the full-epoch chunked job spends it at ``num_chunks * num_workers``
(~30 min). Splitting the *batch* across chunks recovers most of that: at
``BATCHES_PER_EPOCH`` batches of ``NUM_CHUNKS`` tasks the compute per batch is
about 5-6 minutes, and what is left is the barrier - B strictly sequential
steps per epoch, each paying a job transition of a couple of minutes. Expect
roughly 1.5-2 h per epoch here against 9.3 h for the single-task job and 30 min
for a whole-epoch run. Those are predictions from the numbers above; the epoch
jobs log ``[TIMING] chunk`` lines to check them against.

**Both update rules are run.** ``mode="statistics"`` damps the counts
(``S <- a*S + (1-a)*N``, online EM); ``mode="parameters"`` damps the normalized
table, which is what :mod:`.vq_unsupervised_batched_ema` does. They are
different algorithms - see ``lib.guided_kmeans.chunked.ema`` - and which one
converges better here is the open question this config is for.

Batch size and alpha are held at the single-task job's values (2000 sequences,
0.8) so the ``mode="parameters"`` arm is directly comparable to it, and to
:mod:`.vq_unsupervised` at the same epoch counts. They are one knob, not two:
the effective window is ``batch_size / (1 - alpha)`` = 10 000 sequences, so a
run at half the batch size wanting the same experiment needs alpha 0.9, not 0.8.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    GMM_ALIGNMENT_CV,
    COLLEAGUE_CENTROIDS_K512,
    PHONEME_LM_ZIJIAN_3GRAM,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
    create_recog_rasr_config,
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
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    EMAConfig,
    NormalTableJob,
    chunked_clustering,
    vq_flavor,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    build_decode_config,
    silence_free_cv_features,
    silence_free_ls100_features,
    NUM_LABELS,
    NUM_CODEWORDS,
    LM_ORDER,
    USE_FORWARD_BACKWARD,
    SUBSAMPLING,
    LM_SCALE,
    DISTANCE_SCALE,
    LOOP_PROB,
    BEAM_SIZE,
    TABLE_FLOOR,
)

exp_dir = "vq_unsupervised_batched"
version = 1

#: (lm_name, lm_path, sigma, seed) - the vq_unsupervised_batched_ema matrix, so
#: every row here has a single-task counterpart to compare against.
EXPERIMENTS = [
    ("ours-3gram",   None,                    0.1, 42),
    ("ours-3gram",   None,                    0.1, 43),
    # ("zijian-3gram", PHONEME_LM_ZIJIAN_3GRAM, 0.1, 42),
    # ("zijian-3gram", PHONEME_LM_ZIJIAN_3GRAM, 0.1, 43),
]

#: The comparison the config is for. Both arms run the identical schedule and
#: differ only in where the damping is applied.
EMA_MODES = ["statistics", "parameters"]

#: Ten first, as with the single-task job: the per-epoch trajectory against
#: vq_unsupervised decides whether a longer run is worth 8 x 10 x 14 epoch jobs.
NUM_EPOCHS = 10

EMA_ALPHA = 0.8
#: 28.2 k sequences / 14 ~ 2 020 per batch, i.e. the single-task job's
#: EMA_MINIBATCH_SIZE of 2000. Raising this is not free in the way it looks:
#: the B steps of an epoch are strictly sequential, so each one costs a job
#: transition on top of its compute.
BATCHES_PER_EPOCH = 14

#: Chunks **per batch**, not per epoch - a batch is 1/14 of the corpus, so the
#: whole-epoch value (22-30) would leave each task ~70 sequences, less than the
#: startup it has to amortize. Eight tasks of ~250 sequences is ~5-6 minutes of
#: recognition each at NUM_WORKERS, which is worth scheduling.
NUM_CHUNKS = 8

#: 9 RASR workers + 1 overhead, the configuration the full-epoch runs were
#: measured at (9x concurrency, parent process idle at a few percent of a core).
NUM_WORKERS = 9

#: RSS was measured at 6.2 GB against an 8 GB request on the full-epoch runs;
#: the parent's memory scales with the recognizer's in-flight limit, which is
#: 4 x num_workers.
EPOCH_RQMT = {"mem": 12}

decode_epochs = [0, 5, 10]


def build_vq_training_batched(
    *,
    features,
    segments,
    lm_path,
    sigma,
    seed,
    ema_mode,
    num_epochs,
    lexicon,
    alias_prefix,
    num_workers=NUM_WORKERS,
    rqmt=None,
):
    """One unsupervised VQ run updating BATCHES_PER_EPOCH times per epoch."""
    recognition_config = create_recog_rasr_config(
        lm_scale=LM_SCALE,
        emission_scale=1.0,
        transition_scale=LM_SCALE,
        loop_probability=LOOP_PROB,
        silence_loop_probability=LOOP_PROB,
        use_forward_backward_search=USE_FORWARD_BACKWARD,
        lm_order=LM_ORDER,
        use_eow_phonemes=False,
        max_beam_size=BEAM_SIZE,
        lm_path=lm_path,
    )
    flavor = vq_flavor(
        centroids=COLLEAGUE_CENTROIDS_K512,
        table=NormalTableJob(NUM_LABELS, NUM_CODEWORDS, sigma=sigma, seed=seed).out_table,
        recognition_config=recognition_config,
        lexicon=lexicon,
        num_clusters=NUM_LABELS,
        distance_scale=DISTANCE_SCALE,
        use_forward_backward=USE_FORWARD_BACKWARD,
        table_floor=TABLE_FLOOR,
        num_workers=num_workers,
    )
    exp_result = chunked_clustering(
        num_epochs=num_epochs,
        features_hdf=features,
        # Only used to derive the batch partition: each epoch job is built with
        # its own part of this list, never with the whole thing.
        segments=segments,
        recognition_config=recognition_config,
        lexicon=lexicon,
        num_clusters=NUM_LABELS,
        flavor=flavor,
        subsampling=SUBSAMPLING,
        distance_scale=DISTANCE_SCALE,
        use_forward_backward=USE_FORWARD_BACKWARD,
        rasr_path=(
            tools.RASR_PATH_FORWARD_BACKWARD if USE_FORWARD_BACKWARD else tools.RASR_PATH
        ),
        batches_per_epoch=BATCHES_PER_EPOCH,
        ema=EMAConfig(alpha=EMA_ALPHA, mode=ema_mode),
        num_chunks=NUM_CHUNKS,
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

    decode_lm_scale = 1.0
    decode_loop_prob = 0.0

    cv_dataset = DatasetConfig(
        audio_hdf_path=cv_features.out_features,
        sampling_method=SegmentFile(cv_features.out_segments),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=[
            "ema", "lm", "sigma", "seed", "epoch",
            "mi", "per", "del", "ins", "sub", "fer",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["ema", "sigma", "lm", "seed"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Unsupervised discrete-HMM training over a frozen 512-entry codebook, "
            f"re-estimated {BATCHES_PER_EPOCH} times per epoch on "
            f"{BATCHES_PER_EPOCH}-way batches of silence-free ls-100h "
            f"(alpha={EMA_ALPHA}, effective window "
            f"{28234 // BATCHES_PER_EPOCH} / (1 - {EMA_ALPHA}) sequences), "
            f"{NUM_EPOCHS} epochs, decoded on silence-free cv at LM scale "
            f"{decode_lm_scale}. The two arms damp the counts and the normalized "
            f"table respectively; the latter is the rule of vq_unsupervised_batched_ema."
        ),
    )
    recog_results = []

    for ema_mode in EMA_MODES:
        for lm_name, lm_path, sigma, seed in EXPERIMENTS:
            exp_name = f"ls100-nosil_{ema_mode}_{lm_name}_sigma-{sigma}_seed-{seed}"
            _, exp_result = build_vq_training_batched(
                features=ls100_features.out_features,
                segments=ls100_features.out_segments,
                lm_path=lm_path,
                sigma=sigma,
                seed=seed,
                ema_mode=ema_mode,
                num_epochs=NUM_EPOCHS,
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
                    params={"ema": ema_mode, "lm": lm_name, "sigma": sigma, "seed": seed},
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
