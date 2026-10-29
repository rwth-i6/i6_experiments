"""A Gaussian mixture trained whole from the first epoch - with the codebook, and without one.

Every run so far learns the mixture weights against a frozen codebook first; the
densities move, if at all, only after the table has converged
(``vq_gmm_tau_densities``: weights + means + one re-estimated covariance, from
epoch 60 of the weights run, 10.7 -> 9.1). This asks whether the staging is
needed, and then whether the codebook is: means, weights and a tied covariance
re-estimated together from epoch 1, starting from

* the colleague's k512 codebook (``cb-*``), or
* no codebook at all: 512 random training segments (``rand-tied``), or the corpus
  mean plus ``eps * N(0, Sigma)`` - a flat start (``flat-<eps>-tied``).

The table starts where the weights run's did (NormalTableJob, sigma 0.02), so
epoch 1 of ``cb-tau4`` and ``cb-tied`` searches exactly as that run's epoch 1 did.

**Label-blind EM first.** While the weight rows are (near) equal, p(k | l, x) is
the same for every label, so the density update is plain unsupervised EM - the
search's posteriors drop out of it. Run on 200k ls-100 segments in the globally
whitened space, 15 iterations (spread = mean squared distance of the means from
their centroid, per dimension; log-likelihood per segment):

    start                     spread it 0 -> 3 -> 5 -> 15         ll at 15
    codebook, tau=4 fixed     0.212   0.241   0.251   0.281        -883.0
    codebook, tied            0.212   0.239   0.244   0.252        -657.2
    random segments, tied     1.012   0.363   0.349   0.351        -660.4
    flat 0.01, tied           0.0001  0.0001  0.0001  0.093        -680.0
    flat 0.1, tied            0.010   0.125   0.253   0.309        -656.2
    flat 0.1, Sigma fixed     0.010   0.112   0.219   0.267        -682.8

So no start collapses: the codebook spreads (after the epoch-1 offset correction,
4.99 -> 0.41) at either covariance, random segments contract to a codebook-like
geometry in ~5 iterations, and a flat start leaves the symmetric point on the
data's own structure - at 0.1 within 2-3 iterations, at 0.01 after ~6, at the
price of 44 dead densities by 15. The symmetric point (identical densities, an
uninformative AM, the LM's own best path) is a fixed point of EM, but an unstable
one along directions the data provides. What only the guided runs can show is how
that interacts with the table sharpening at the same time (its label contrast
went 0.9 -> ~12 nats within 5 epochs in the weights runs) - which is the
experiment.

Arms (RASR forward-backward, AM scale 1.0, 60 epochs):

``cb-tied``
    codebook, covariance starting at tau * Sigma (tau=4), then one covariance
    re-estimated for all labels (:class:`...LabelCovarianceMixtureModel`, r = inf)
``cb-tau4``
    codebook, covariance fixed at tau * Sigma: the means arm of
    ``vq_gmm_tau_densities``, from epoch 0. The control.
``rand-tied``
    512 random training segments, covariance starting at Sigma, re-estimated tied
``flat-<eps>-tied``
    corpus mean + eps * N(0, Sigma), for eps in ``FLAT_SCALES``, covariance
    starting at Sigma, re-estimated tied. ``eps`` is per dimension: a perturbation
    is ~``eps * sqrt(512)`` standard deviations long, 2.3 at 0.1 - so 0.1 is a
    widely spread start, 0.001 one that sits on the symmetric point.

The reference is ``vq_gmm_tau``'s tau=4 weights run (10.7 at epoch 60), rebuilt
argument for argument so its rows are its finished decodes.

As in ``vq_gmm_tau_densities``, every decode at the training scale on
``TRAIN_SLICE_EPOCHS`` is repeated on a 10% slice of the training sequences:
cv is train-other, training is train-clean-100, and that mismatch was what broke
means_from-40 (cv 9.8 -> 15.7 while its train-slice PER held at ~9).

**Low priority by construction.** Every job this config adds is submitted
``--nice=1600``, and the footprint is capped at 6 arms x ``NUM_CHUNKS`` x 10 CPUs
= 480 of the 1,100 CPUs ``cpu_modern`` allows one user. ``NUM_CHUNKS`` is
unhashed: lower it before launch if ``vq_gmm_tau_densities`` is running too.
"""

from math import inf

from sisyphus import tk

from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    COLLEAGUE_CENTROIDS_K512,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    DuplicateCovsJob,
    GlobalCovarianceJob,
    NormalTableJob,
    PerturbedMeanCentroidsJob,
    RandomCentroidsJob,
    ScaleCovsJob,
    chunked_clustering,
    mixture_flavor,
)
from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.chunked.flavors import (
    label_covariance_flavor,
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
    Column,
    LatexTableReport,
    TextTableReport,
    clustering_statistics_per_epoch,
    param,
    statistic,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
    silence_free_ls100_features,
)
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau as gmm

exp_dir = "vq_gmm_tau_scratch"
version = 1

TAU = 4.0
NUM_EPOCHS = 60
#: The flat starts' perturbation scales, per dimension. All draw the same noise
#: (one seed), so the arms differ in its size and nothing else.
FLAT_SCALES = [0.1, 0.01, 0.001]
SEED = gmm.SEED

#: Dense through the weights run's take-off (epochs 30-40 under RASR), then
#: every ten. Chains do not hash their length: extending resumes.
DECODE_EPOCHS = [5, 10, 15, 20, 25, 30, 40, 50, 60]
SCALE_SWEEP_EPOCHS = [30, 60]
DECODE_SCALES = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
#: Where the cv decode is repeated on TRAIN_SLICE of the training sequences.
TRAIN_SLICE_EPOCHS = [10, 20, 30, 40, 50, 60]
#: As vq_gmm_tau_densities.TRAIN_SLICE: the same slice job, hence the same slice.
TRAIN_SLICE = 0.1

#: As vq_gmm_tau_densities: 8 chunks of 10 CPUs, ~2-2.5 h an epoch on RASR.
NUM_CHUNKS = 8
EPOCH_RQMT = {"mem": 10, "time": 6, "sbatch_args": ["--nice=1600"]}
DECODE_SBATCH_ARGS = ["--nice=1600"]


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    ls100 = silence_free_ls100_features()
    cv = silence_free_cv_features()
    cv_dataset = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )
    train_slice = ShuffleAndSplitSegmentsJob(
        ls100.out_segments, split={"decode": TRAIN_SLICE, "rest": 1.0 - TRAIN_SLICE}
    ).out_segments["decode"]
    train_dataset = DatasetConfig(
        audio_hdf_path=ls100.out_features,
        sampling_method=SegmentFile(train_slice),
        precomputed=True,
    )

    # --- vq_gmm_tau's jobs, argument for argument (its _run) ------------------------
    global_stats = GlobalCovarianceJob(ls100.out_features)
    covs = ScaleCovsJob(
        DuplicateCovsJob(global_stats.out_cov, gmm.NUM_CODEWORDS).out_covs, TAU
    ).out_covs
    init_table = NormalTableJob(
        gmm.NUM_LABELS, gmm.NUM_CODEWORDS, sigma=gmm.SIGMA, seed=SEED
    ).out_table
    train_config = gmm._recog_config(1.0, 0.0, 1.0, use_fb=True)
    decode_config_rasr = gmm._recog_config(1.0, 0.0, None, use_fb=False)
    reference_scale = gmm.train_distance_scale(TAU)

    common = dict(
        features_hdf=ls100.out_features,
        recognition_config=train_config,
        lexicon=lexicon,
        num_clusters=gmm.NUM_LABELS,
        use_forward_backward=True,
        rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
        num_workers=gmm.NUM_WORKERS,
        parent_threads=gmm.PARENT_THREADS,
    )
    flavor_common = dict(
        recognition_config=train_config,
        lexicon=lexicon,
        num_clusters=gmm.NUM_LABELS,
        use_forward_backward=True,
        mixture_floor=1e-2,
        num_workers=gmm.NUM_WORKERS,
    )
    reference_name = f"weights_tau-{TAU}"
    reference = chunked_clustering(
        num_epochs=gmm.WEIGHTS_EPOCHS,
        flavor=mixture_flavor(
            centroids=COLLEAGUE_CENTROIDS_K512,
            covs=covs,
            mixtures=init_table,
            distance_scale=reference_scale,
            update_densities=False,
            update_covariances=None,
            **flavor_common,
        ),
        distance_scale=reference_scale,
        num_chunks=gmm.NUM_CHUNKS,
        rqmt=gmm.EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{gmm.exp_dir}/{reference_name}",
        **common,
    )

    # --- the arms' starting points --------------------------------------------------
    # One covariance per label, all equal: the tied model's own layout.
    label_covs = {
        TAU: ScaleCovsJob(
            DuplicateCovsJob(global_stats.out_cov, gmm.NUM_LABELS).out_covs, TAU
        ).out_covs,
        1.0: DuplicateCovsJob(global_stats.out_cov, gmm.NUM_LABELS).out_covs,
    }
    tied = dict(update_means=True, covariance="full", covariance_relevance=inf)

    def tied_flavor(centroids, cov_scale):
        return label_covariance_flavor(
            centroids=centroids,
            covs=label_covs[cov_scale],
            mixtures=init_table,
            distance_scale=1.0,
            **tied,
            **flavor_common,
        )

    arms = {
        "cb-tied": tied_flavor(COLLEAGUE_CENTROIDS_K512, TAU),
        "cb-tau4": mixture_flavor(
            centroids=COLLEAGUE_CENTROIDS_K512,
            covs=covs,
            mixtures=init_table,
            distance_scale=1.0,
            update_densities=True,
            update_covariances=False,
            **flavor_common,
        ),
        "rand-tied": tied_flavor(
            RandomCentroidsJob(ls100.out_features, gmm.NUM_CODEWORDS, seed=SEED).out_centroids,
            1.0,
        ),
    }
    for eps in FLAT_SCALES:
        arms[f"flat-{eps}-tied"] = tied_flavor(
            PerturbedMeanCentroidsJob(
                global_stats.out_mean, global_stats.out_cov, gmm.NUM_CODEWORDS, eps, seed=SEED
            ).out_centroids,
            1.0,
        )

    # --- reports --------------------------------------------------------------------
    train_columns = [
        Column("train_per", ("Train slice", "PER [%]"),
               lambda row: row.values.get("train_per"), fmt="{:.1f}", block="train"),
        Column("train_sub", ("Train slice", "Sub [%]"),
               lambda row: row.values.get("train_sub"), fmt="{:.1f}", scale=100, block="train"),
    ]
    report = LatexTableReport(
        columns=[param("arm"), "epoch", "per", "del", "ins", "sub", *train_columns, "mi",
                 "log_likelihood"],
        sort_by=["arm"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            "A Gaussian mixture trained whole from epoch 1 - weights, means and (tied) "
            "covariance - from the k512 codebook (cb), from 512 random training "
            "segments (rand), or from the corpus mean plus $\\epsilon \\mathcal{N}(0, "
            "\\Sigma)$ (flat). Reference: vq\\_gmm\\_tau's weights-only run. RASR "
            "forward-backward, decoded on silence-free cv (train-other) at AM scale 1.0 "
            "and on a 10\\% slice of the train-clean-100 sequences it trains on."
        ),
    )
    scales_report = LatexTableReport(
        columns=[param("arm"), "epoch", param("ds", header=("Decode", "AM scale")),
                 "per", "del", "ins", "sub"],
        sort_by=["arm", "ds"],
        epochs=None,
        drop_empty_rows=True,
        caption="As above, the epochs decoded over the full AM-scale sweep.",
    )
    progress_report = TextTableReport(
        columns=[param("arm"), "epoch", "log_likelihood", "posterior_entropy",
                 "cluster_perplexity",
                 statistic("means_shift_mean", ("Mean shift", "[Mahal.]"), fmt="{:.3f}"),
                 statistic("means_kept_previous", ("Dead", "means"), fmt="{:d}"),
                 statistic("cov_logdet_vs_previous_mean", ("log det", "vs prev / D"),
                           fmt="{:+.4f}")],
        sort_by=["arm"],
        drop_empty_rows=True,
        title=(
            "Per-epoch M-step of every arm: the epoch job's search log-likelihood, "
            "posterior entropy and label perplexity (under the previous model), and "
            "what its M-step did - mean Mahalanobis shift of the codewords, codewords "
            "too starved to move, and the change of log det Sigma per dimension."
        ),
    )
    recog_results = []

    def decode(name, arm, epoch, model_dir, mixtures, scales, statistics):
        diagnostics = MixtureDiagnosticsJob(mixtures)
        tk.register_output(
            f"guided_kmeans/{exp_dir}/diagnostics/{name}_ep-{epoch}.json",
            diagnostics.out_diagnostics,
        )
        for scale in scales:
            # vq_gmm_tau's decode arguments, so the reference's decodes are reused.
            dc = DecodeConfig(
                centroids=COLLEAGUE_CENTROIDS_K512,
                model_dir=model_dir,
                recog_rasr_config=decode_config_rasr,
                distance_scale=scale,
                write_frame_labels=True,
            )
            decode_name = f"{name}_ep-{epoch}_ds-{scale}"
            res = decode_and_score(
                decode_name, "cv", dc, cv_dataset,
                rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960",
            )
            # Unhashed, and harmless on the reference's decodes, which are finished.
            res.fwd_job.rqmt["sbatch_args"] = list(DECODE_SBATCH_ARGS)
            tk.register_output(f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per)
            recog_results.append(res)
            if len(scales) > 1:
                scales_report.add_row(result=res, params={"arm": arm, "ds": scale},
                                      epoch=epoch)
            if scale != 1.0:
                continue
            values = {"mi": diagnostics.out_mi}
            if epoch in TRAIN_SLICE_EPOCHS:
                train = decode_and_score(
                    f"{decode_name}_train", "train-slice", dc, train_dataset,
                    rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-clean-100",
                )
                train.fwd_job.rqmt["sbatch_args"] = list(DECODE_SBATCH_ARGS)
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/per/{decode_name}_train_per", train.per)
                recog_results.append(train)
                values.update(train_per=train.per, train_sub=train.substitution)
            report.add_row(result=res, params={"arm": arm}, epoch=epoch,
                           statistics=statistics, values=values)

    ref_stats = clustering_statistics_per_epoch(
        reference.out_epoch_statistics, name=reference_name, epoch_offset=1, lexicon=lexicon)
    for e in gmm.WEIGHTS_DECODE_EPOCHS:
        decode(reference_name, "weights (reference)", e, reference.out_models[e],
               reference.out_artifacts["mixtures"][e],
               DECODE_SCALES if e in gmm.SCALE_SWEEP_EPOCHS else [1.0], ref_stats)

    for arm, flavor in arms.items():
        result = chunked_clustering(
            num_epochs=NUM_EPOCHS,
            flavor=flavor,
            distance_scale=1.0,
            num_chunks=NUM_CHUNKS,
            rqmt=EPOCH_RQMT,
            alias_prefix=f"guided_kmeans/{exp_dir}/{arm}",
            **common,
        )
        tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{arm}.json",
                           result.out_statistics)
        stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=arm, epoch_offset=1, lexicon=lexicon)
        progress = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=arm, epoch_offset=0, lexicon=lexicon)
        for e in range(1, NUM_EPOCHS + 1):
            progress_report.add_row(params={"arm": arm}, epoch=e, statistics=progress)
        for e in DECODE_EPOCHS:
            decode(arm, arm, e, result.out_models[e], result.out_artifacts["mixtures"][e],
                   DECODE_SCALES if e in SCALE_SWEEP_EPOCHS else [1.0], stats)

    recog_report = create_report(recog_results)
    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=recog_report, required=recog_report, update_frequency=60,
    )
    text_report = report.as_text()
    text_report.register(
        f"guided_kmeans/{exp_dir}/report_{version}.txt",
        required=text_report, register_dependencies=False, update_frequency=60,
    )
    scales_text = scales_report.as_text()
    scales_text.register(
        f"guided_kmeans/{exp_dir}/report_scales_{version}.txt",
        required=scales_text, register_dependencies=False, update_frequency=60,
    )
    report.register(
        f"guided_kmeans/{exp_dir}/tex/report_{version}.tex",
        required=report, update_frequency=60,
    )
    progress_report.register(
        f"guided_kmeans/{exp_dir}/convergence_{version}.txt",
        required=progress_report, update_frequency=60,
    )


def py():
    run()
