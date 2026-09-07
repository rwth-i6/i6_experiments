"""VQ unsupervised training with within-epoch EMA updates — 3-gram test.

Replicates the :mod:`.vq_unsupervised` experiment matrix (both 3-gram LMs,
sigma in {0.1, 1.0}, seeds 42 and 43) but with a different update rule.
Instead of accumulating p(codeword | label) counts over the whole epoch and
normalising once, the corpus is streamed in mini-batches of
``EMA_MINIBATCH_SIZE`` sequences, and after each mini-batch the table is
updated via an exponential moving average:

    table <- EMA_ALPHA * table + (1 - EMA_ALPHA) * table_minibatch

This runs as a **single task** per epoch (no job array). Parallelism comes from
the ``num_workers`` RASR processes that recognise each mini-batch concurrently
inside :class:`.BatchedEMAEpochJob`.

Kept at 10 epochs first so the per-epoch trajectory can be compared directly
against the full-epoch runs in :mod:`.vq_unsupervised` before committing to a
longer run.
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

exp_dir = "vq_unsupervised_batched_ema"
version = 1

#: (lm_name, lm_path, sigma, seed) - mirrors the vq_unsupervised matrix
EXPERIMENTS = [
    ("ours-3gram",   None,                    0.1, 42),
    ("ours-3gram",   None,                    0.1, 43),
    ("zijian-3gram", PHONEME_LM_ZIJIAN_3GRAM, 0.1, 42),
    ("zijian-3gram", PHONEME_LM_ZIJIAN_3GRAM, 0.1, 43),
]

NUM_EPOCHS = 20

EMA_ALPHA = 0.8
EMA_MINIBATCH_SIZE = 2000  # sequences per EMA update step

# 10 RASR workers + 1 overhead = 11 CPUs. Each worker handles roughly
# EMA_MINIBATCH_SIZE / NUM_WORKERS = 200 sequences per mini-batch.
# Single-task job: one allocation per epoch, not a job array.
NUM_WORKERS = 10

# The single task reads the full corpus mini-batch by mini-batch, keeping at
# most EMA_MINIBATCH_SIZE sequences buffered at once. At 2000 seqs * ~120
# frames * 512-D float64 that is ~1 GB; 16 GB leaves comfortable headroom.
EPOCH_RQMT = {"mem": 16}

decode_epochs = [0, 5, 10, 20]


def build_vq_training_batched_ema(
    *,
    features,
    lm_path,
    sigma,
    seed,
    num_epochs,
    lexicon,
    alias_prefix,
    num_workers=NUM_WORKERS,
    rqmt=None,
):
    """One unsupervised VQ run using within-epoch EMA updates."""
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
        num_chunks=1,  # unused by BatchedEMAEpochJob (single task, no job array)
        num_workers=num_workers,
        rqmt=rqmt,
        alias_prefix=alias_prefix,
        ema_alpha=EMA_ALPHA,
        ema_minibatch_size=EMA_MINIBATCH_SIZE,
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
            "lm", "sigma", "seed", "epoch",
            "mi", "per", "del", "ins", "sub", "fer",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["sigma", "lm", "seed"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Unsupervised discrete-HMM training over a frozen 512-entry codebook "
            f"with within-epoch EMA updates (alpha={EMA_ALPHA}, "
            f"{EMA_MINIBATCH_SIZE} sequences per step), {NUM_EPOCHS} epochs on "
            f"silence-free ls-100h, decoded on silence-free cv at LM scale "
            f"{decode_lm_scale}. Both 3-gram LMs, both sigmas, both seeds: "
            f"direct comparison against vq_unsupervised."
        ),
    )
    recog_results = []

    for lm_name, lm_path, sigma, seed in EXPERIMENTS:
        exp_name = f"ls100-nosil_{lm_name}_sigma-{sigma}_seed-{seed}"
        _, exp_result = build_vq_training_batched_ema(
            features=ls100_features.out_features,
            lm_path=lm_path,
            sigma=sigma,
            seed=seed,
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
