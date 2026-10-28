"""More of the Gaussian mixture than its weights: when the means move, and covariances per label.

So far only the weights have worked. ``vq_gmm_tau``'s means arm (weights *and*
means re-estimated, shapes frozen at tau * Sigma, branched at the converged
epoch 60) improves for a few epochs and then gives most of it back - at tau=4
10.71 -> 9.50 (ep 5) -> 10.64 (ep 60), with the likelihood rising throughout.
This config tries the two ways out of that: move the means *earlier*, and give
the densities a *shape* that is estimated rather than dialled in.

**What the means arm actually did** (measured on its saved centroids, in units
of the globally whitened space):

* Epoch 1 is mostly an offset correction. The colleague's k512 codebook sits
  4.1 standard deviations off this data's mean, almost entirely along the
  *lowest*-variance direction of the features (eigenvalue 5.4 of 512); the
  first mean update takes that to 1.4, which is most of the 3.3-unit jump of
  epoch 1. Everything after is small steps (0.2-0.8 per epoch).
* After that the codebook does not collapse, it *spreads*: centroid spread
  0.194 -> 0.251, median nearest-neighbour distance 7.1 -> 9.1. That is what a
  mean update under an isotropic (whitened) metric does when the codebook
  explains only 12.6% of the whitened variance: every one of the 512 directions
  pulls equally, and 319 of them are directions the codebook does not explain
  at all (within-codeword variance > 0.9 of the total). Likelihood is gained
  there; PER is not.

**Why per-label covariances.** The within-codeword scatter is far from
isotropic in that space (eigenvalues 0.07 .. 2.4 after centring) and differs
by label: with the oracle labels, a covariance per label smoothed towards the
pooled one (r = 512..2048) gains 62 nats/segment of held-out likelihood over a
single pooled covariance, while a per-label *diagonal* in the pooled
eigenbasis gains only 7. Statistically a full covariance per density is out of
reach (131k parameters, ~6,800 segments per density), per label it is not
(~87k segments per label on average, ~2k for the rarest). Hence the model

    p(x | l) = sum_k w_lk N(x | mu_k, Sigma_l)

(:class:`...chunked.models.LabelCovarianceMixtureModel`): the codebook means
are shared, the covariance is tied per *label*. Its M-step is ECM - means under
the previous covariances, then covariances about the new means
(:class:`...chunked.accumulators.LabelCovarianceAccumulator`).

**The AM/LM balance moves.** Label-specific covariances make the label scores
far more different from each other - measured, one M-step multiplies the
median gap between the best and second-best label by ~10 - and the LM is the
only supervision this method has. See ``COV_TRAIN_SCALE``: each covariance arm
trains at the AM scale that holds the branch point's gap (0.1 for free per-label
covariances), and decodes over a sweep around it. ``cov-label`` also runs at
scale 1.0 (``COV_EM_ARMS``): with the LM and transitions at 1.0 as well, its
E-step is the model's exact posterior, so that arm is EM proper - for the
relevance-smoothed (MAP) objective, see ``COV_RELEVANCE`` - and says whether
the rebalancing was needed at all.

Arms, all branched from ``vq_gmm_tau``'s tau=4 weights arm and trained with its
search - RASR forward-backward on CPU workers, nothing on a GPU:

``means_from-<e>``
    means + weights (shapes frozen at tau * Sigma), branched at epoch ``e`` of
    the weights run: 30 (PER 43.6%, mid take-off), 40 (13.2%, nearly there),
    and 60 (converged). The last *is* ``vq_gmm_tau``'s means arm, rebuilt
    argument for argument, so its 60 epochs are finished jobs; only its decodes
    at epochs 1-3, which locate the early peak, are new.
``cov-<structure>[+means]``
    from epoch 60, weights + a re-estimated covariance per label (+ means):
    ``tied`` - one covariance for all labels (r = inf): does re-estimating the
    *metric* alone help, and does it stop the means drift? ``label`` - free per
    label, smoothed with r = ``COV_RELEVANCE``: the model above, trained at both
    the rebalanced scale and 1.0. ``mllt`` - semi-tied (one transform, a
    diagonal per label), the cheap middle ground.

A per-label diagonal (``covariance="diagonal"``) is implemented but not run:
offline it recovers 7 of the 62 nats a full per-label covariance does.

**The first covariance runs (2026-10-23..26) are void.** The model handed the
recognizers a column-major score matrix, which librasr reads row-major, so every
search - training and decoding - saw the scores scrambled across segments and
labels (91.4% of cv segments decoded as the argmin of the scrambled read, 2.6%
as that of the true scores). Fixed in the model (2026-10-26); those jobs were
removed, and the arms rerun under the same hashes.

**Train-slice decodes.** The means arms all improve the AM on the data they
train on - AM-only accuracy against the oracle 0.79 -> 0.85 from epoch 40 - while
means_from-40 decodes cv 9.8 -> 15.7. On cv its AM drops 0.818 -> 0.756 between
epochs 7 and 10, many phonemes turning into a rare label (OY) with a peripheral
codeword, while on ls-100 nothing moves. cv is train-other, the training data
train-clean-100. So every decode at an arm's training scale is repeated on
``TRAIN_SLICE`` of the training sequences themselves: if the cv PER rises and
the train-slice PER does not, the means are fitting the clean data at the
expense of the other condition, rather than giving up on the acoustics.

**Low priority by construction.** Every job this config adds - epochs and
decodes - is submitted ``--nice=1600``, above the full age weight, so anything
else pending goes first however long these have waited. Nice only orders
*pending* jobs, though, and ``cpu_modern`` caps one user at 1,100 CPUs, so the
footprint is capped too: 8 new arms x ``NUM_CHUNKS`` x 10 CPUs = 640 CPUs at
most, leaving 460 of the cap to the running configs' decodes.

Only finished jobs of other configs are built: ``vq_gmm_tau`` finished both its
tau=4 arms (60 epochs each) and nothing drives them.
"""

from math import inf

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    COLLEAGUE_CENTROIDS_K512,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    DuplicateCovsJob,
    GlobalCovarianceJob,
    NormalTableJob,
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
from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
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

exp_dir = "vq_gmm_tau_densities"
version = 1

#: The reference: vq_gmm_tau's arms at this tau, all of them finished.
TAU = 4.0

# --- means, branched at several points of the weights run -----------------------
#: 60 is vq_gmm_tau's own means arm (60 epochs, finished); the others are new.
MEANS_BRANCH_EPOCHS = [30, 40, gmm.MEANS_FROM_EPOCH]
#: The old arm's drift was mostly done by epoch 20 (9.50 at 5, 10.30 at 20,
#: 10.64 at 60). Chains do not hash their length: extending resumes.
MEANS_EPOCHS = 20
MEANS_DECODE_EPOCHS = [1, 2, 3, 5, 7, 10, 15, 20]

# --- covariances, branched from the converged table ------------------------------
COV_BRANCH_EPOCH = 60
COV_EPOCHS = 20
COV_DECODE_EPOCHS = [1, 2, 3, 5, 7, 10, 15, 20]
#: ``r`` of LabelCovarianceAccumulator: a label with n segments keeps n / (n + r)
#: of its own estimate. 2048 was the best of {512, 2048, 8192} on held-out
#: oracle-label residuals (62 nats/segment over pooled; 512 equal, 8192 48).
#: Labels here hold ~2,000 to ~320,000 segments, so the rarest sit about halfway.
#: 0 would be maximum likelihood, and degenerate for those: ~2,000 segments for a
#: covariance of 131k parameters is invertible but rank-poor, and its log det
#: runs away - the variance collapse of the earlier full-covariance runs.
COV_RELEVANCE = 2048.0
COV_ARMS = {
    "cov-tied": dict(covariance="full", covariance_relevance=inf),
    "cov-tied+means": dict(covariance="full", covariance_relevance=inf, update_means=True),
    "cov-label": dict(covariance="full", covariance_relevance=COV_RELEVANCE),
    "cov-label+means": dict(covariance="full", covariance_relevance=COV_RELEVANCE,
                            update_means=True),
    "cov-mllt": dict(covariance="semi_tied", covariance_relevance=COV_RELEVANCE),
}
#: AM scale each covariance arm trains at: the one that keeps the branch point's
#: median frame gap (2nd-best minus best label cost). Measured by one M-step
#: from vq_gmm_tau_cuda's epoch-60 model - the same tau=4 table trained with the
#: GPU search, PER 10.49 against this reference's 10.71 - on 200k ls-100
#: segments (AM-only posteriors as the E-step), evaluated on the next 100k:
#:
#:     M-step                 gap    x base   d loglik/seg   argmax = base
#:     base (tau=4, ep 60)    2.95    1.00        -              -
#:     means only             2.35    0.80       +0.9          90.9%
#:     tied (r=inf)           2.83    0.96     +215.1          90.0%
#:     label, r=512          35.15   11.92     +290.6          82.2%
#:     label, r=2048         28.87    9.79     +282.6          83.1%
#:     label, r=8192         21.03    7.13     +263.3          85.0%
#:     label + means, 2048   28.64    9.71     +282.1          82.7%
#:     mllt, r=2048 (10 it)   4.47    1.51     +241.0          57.8%
#:     diagonal, r=2048       2.96    1.00     +230.1          50.5%
#:
#: Repeated on this config's own reference (vq_gmm_tau's epoch-60 model): base
#: 2.94, tied x0.96, label (2048) x9.71, label + means x9.63, mllt x1.49 - the
#: same to two digits, so the scales below hold for it.
#:
#: Two readings. The fixed tau=4 covariance is far too broad: re-estimating even
#: one tied covariance shrinks log det by 1.7 per dimension and gains 215 nats a
#: segment, where moving the means gains 0.9. And a free covariance per label
#: multiplies the label contrast by ~10 at the first step - at scale 1.0 the LM,
#: the only supervision this method has, would barely register in the search.
#: The decode sweep runs around each arm's training scale, so the decode side
#: settles the balance by measurement whatever the training side did.
COV_TRAIN_SCALE = {
    "cov-tied": 1.0,
    "cov-tied+means": 1.0,
    "cov-label": 0.1,
    "cov-label+means": 0.1,
    "cov-mllt": 0.7,
}
#: Arms trained a second time at AM scale 1.0, unbalanced: EM proper, with the
#: search's posterior the model's own. Decoded over both scales' sweeps.
COV_EM_ARMS = ["cov-label"]

DECODE_SCALE_FACTORS = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
#: Epochs decoded at every scale in DECODE_SCALE_FACTORS x the arm's training
#: scale; elsewhere at the training scale alone.
SCALE_SWEEP_EPOCHS = [3, 20]

#: Chunks per new arm, i.e. how many 10-CPU tasks it holds at once - see the
#: module docstring for the cap this keeps. vq_gmm_tau's chunks ran 0.63-0.85 h
#: at 22 per epoch (1,284 sequences, 13.4 s of search each over 7 workers), so
#: at 8 a chunk takes ~2-2.5 h and an epoch about that long.
NUM_CHUNKS = 8
#: Unhashed. vq_gmm_tau's 8 GB chunks peaked at 8.2-8.4 GB RSS. The time is a
#: limit, not a request for speed: settings.py doubles it on a timeout.
EPOCH_RQMT = {"mem": 10, "time": 6, "sbatch_args": ["--nice=1600"]}
DECODE_SBATCH_ARGS = ["--nice=1600"]

#: Fraction of the ls-100 training sequences decoded alongside cv - ~2,820 of
#: 28,234, about cv's 2,786, so the two PERs carry the same sampling noise. Drawn
#: from the feature job's own segment list, so every tag has features.
TRAIN_SLICE = 0.1


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
    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    base_covs = DuplicateCovsJob(global_cov, gmm.NUM_CODEWORDS).out_covs
    covs = ScaleCovsJob(base_covs, TAU).out_covs
    init_table = NormalTableJob(
        gmm.NUM_LABELS, gmm.NUM_CODEWORDS, sigma=gmm.SIGMA, seed=gmm.SEED
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

    def gmm_flavor(mixtures, update_covariances):
        """vq_gmm_tau._run's flavor: frozen densities for the weights arm (None),
        means-only for the means arm (False)."""
        return mixture_flavor(
            centroids=COLLEAGUE_CENTROIDS_K512,
            covs=covs,
            mixtures=mixtures,
            distance_scale=reference_scale,
            update_densities=update_covariances is not None,
            update_covariances=update_covariances,
            **flavor_common,
        )

    reference_name = f"weights_tau-{TAU}"
    reference = chunked_clustering(
        num_epochs=gmm.WEIGHTS_EPOCHS,
        flavor=gmm_flavor(init_table, None),
        distance_scale=reference_scale,
        num_chunks=gmm.NUM_CHUNKS,
        rqmt=gmm.EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{gmm.exp_dir}/{reference_name}",
        **common,
    )

    # --- reports --------------------------------------------------------------------
    report = LatexTableReport(
        columns=[param("arm"), param("from", header=("Branch", "epoch")),
                 param("train", header=("Train", "AM scale")), "epoch",
                 param("ds", header=("Decode", "AM scale")), "per", "del", "ins", "sub",
                 Column("train_per", ("Train slice", "PER [%]"),
                        lambda row: row.values.get("train_per"), fmt="{:.1f}", block="train"),
                 Column("train_sub", ("Train slice", "Sub [%]"),
                        lambda row: row.values.get("train_sub"), fmt="{:.1f}", scale=100,
                        block="train"),
                 "log_likelihood"],
        sort_by=["arm", "train"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            "Re-estimating more of the Gaussian mixture than its weights, branched "
            "from vq\\_gmm\\_tau's tau=4 weights arm (frozen k512 codebook, RASR "
            "forward-backward). \\texttt{means}: means + weights under the fixed "
            "tau-scaled corpus covariance; \\texttt{cov}: weights (+ means) under a "
            "covariance per label, $p(x|l) = \\sum_k w_{lk} \\mathcal{N}(x | \\mu_k, "
            "\\Sigma_l)$, tied across labels, free with relevance smoothing, or "
            "semi-tied (MLLT). Decoded on silence-free cv (train-other) at each arm's "
            "training AM scale, and on a 10\\% slice of the train-clean-100 sequences "
            "it trains on (train slice)."
        ),
    )
    scales_report = LatexTableReport(
        columns=[param("arm"), param("from", header=("Branch", "epoch")),
                 param("train", header=("Train", "AM scale")), "epoch",
                 param("ds", header=("Decode", "AM scale")), "per", "del", "ins", "sub"],
        sort_by=["arm", "train", "ds"],
        epochs=None,
        drop_empty_rows=True,
        caption="As above, the epochs decoded over the full AM-scale sweep.",
    )
    progress_report = TextTableReport(
        columns=[param("arm"), param("from"), param("train"), "epoch", "log_likelihood",
                 "posterior_entropy",
                 statistic("means_shift_mean", ("Mean shift", "[Mahal.]"), fmt="{:.3f}"),
                 statistic("cov_logdet_vs_previous_mean", ("log det", "vs prev / D"),
                           fmt="{:+.4f}"),
                 statistic("cov_logdet_vs_pooled_min", ("log det", "vs pool min"),
                           fmt="{:+.3f}"),
                 statistic("cov_logdet_vs_pooled_max", ("log det", "vs pool max"),
                           fmt="{:+.3f}"),
                 statistic("semi_tied_objective_end", ("MLLT", "objective"), fmt="{:.4f}")],
        sort_by=["arm", "from", "train"],
        drop_empty_rows=True,
        title=(
            "Per-epoch M-step of every branch: the epoch job's search log-likelihood "
            "and posterior entropy (under the previous model), and what its M-step did - "
            "mean Mahalanobis shift of the codewords, mean change of log det Sigma_l per "
            "dimension against the previous epoch, the spread of the label covariances' "
            "log det against the pooled one, and the semi-tied objective."
        ),
    )
    recog_results = []

    def decode(name, params, epoch, model_dir, scales, statistics, report_scale):
        for scale in scales:
            # vq_gmm_tau's decode arguments, so its finished decodes are reused.
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
            row = dict(params, ds=scale)
            if scale == report_scale:
                # The same model and scale on training sequences: see TRAIN_SLICE.
                train = decode_and_score(
                    f"{decode_name}_train", "train-slice", dc, train_dataset,
                    rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-clean-100",
                )
                train.fwd_job.rqmt["sbatch_args"] = list(DECODE_SBATCH_ARGS)
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/per/{decode_name}_train_per", train.per)
                recog_results.append(train)
                report.add_row(result=res, params=row, epoch=epoch, statistics=statistics,
                               values={"train_per": train.per, "train_sub": train.substitution})
            if len(scales) > 1:
                scales_report.add_row(result=res, params=row, epoch=epoch)

    def sweep(epoch, sweep_epochs, base_scale, sweep_bases=()):
        """The training scale alone, or at a sweep epoch DECODE_SCALE_FACTORS
        around it - and around every scale in ``sweep_bases`` too."""
        if epoch not in sweep_epochs:
            return [base_scale]
        return sorted({round(f * b, 4) for b in {base_scale, *sweep_bases}
                       for f in DECODE_SCALE_FACTORS})

    def add_branch(name, params, result, num_epochs, decode_epochs, scale, sweep_epochs,
                   sweep_bases=()):
        stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=name, epoch_offset=1, lexicon=lexicon)
        progress = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=name, epoch_offset=0, lexicon=lexicon)
        for e in range(1, num_epochs + 1):
            # Epochs counted from the branch, so the rows line up across arms.
            progress_report.add_row(params=params, epoch=e, statistics=progress)
        for e in decode_epochs:
            decode(name, params, e, result.out_models[e],
                   sweep(e, sweep_epochs, scale, sweep_bases), stats, scale)
            tk.register_output(
                f"guided_kmeans/{exp_dir}/diagnostics/{name}_ep-{e}.json",
                MixtureDiagnosticsJob(result.out_artifacts["mixtures"][e]).out_diagnostics,
            )

    # The reference's own rows: its finished decodes.
    ref_stats = clustering_statistics_per_epoch(
        reference.out_epoch_statistics, name=reference_name, epoch_offset=1, lexicon=lexicon)
    for e in gmm.WEIGHTS_DECODE_EPOCHS:
        scales = gmm.DECODE_DISTANCE_SCALES if e in gmm.SCALE_SWEEP_EPOCHS else [1.0]
        decode(reference_name, {"arm": "weights (reference)", "from": 0, "train": 1.0}, e,
               reference.out_models[e], scales, ref_stats, 1.0)

    # --- means, from several points of the weights run ------------------------------
    for b in MEANS_BRANCH_EPOCHS:
        flavor = gmm_flavor(reference.out_artifacts["mixtures"][b], False)
        if b == gmm.MEANS_FROM_EPOCH:
            # vq_gmm_tau's means arm itself, with its own name, length and resources.
            name, num_epochs = f"means_tau-{TAU}", gmm.MEANS_EPOCHS
            result = chunked_clustering(
                num_epochs=num_epochs, flavor=flavor, distance_scale=reference_scale,
                num_chunks=gmm.NUM_CHUNKS, rqmt=gmm.EPOCH_RQMT,
                alias_prefix=f"guided_kmeans/{gmm.exp_dir}/{name}", **common,
            )
            # Its finished decodes, at its own epochs and sweeps, plus the epochs
            # and sweeps the new arms get, so every arm is compared alike.
            decode_epochs = sorted(set(MEANS_DECODE_EPOCHS) | set(gmm.MEANS_DECODE_EPOCHS))
            sweep_epochs = sorted(set(gmm.SCALE_SWEEP_EPOCHS) | set(SCALE_SWEEP_EPOCHS))
        else:
            name, num_epochs = f"means_from-{b}", MEANS_EPOCHS
            result = chunked_clustering(
                num_epochs=num_epochs, flavor=flavor, distance_scale=reference_scale,
                num_chunks=NUM_CHUNKS, rqmt=EPOCH_RQMT,
                alias_prefix=f"guided_kmeans/{exp_dir}/{name}", **common,
            )
            decode_epochs, sweep_epochs = MEANS_DECODE_EPOCHS, SCALE_SWEEP_EPOCHS
        tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{name}.json",
                           result.out_statistics)
        add_branch(name, {"arm": "means", "from": b, "train": reference_scale}, result,
                   num_epochs, decode_epochs, reference_scale, sweep_epochs)

    # --- a covariance per label, from the converged table ---------------------------
    # The reference's own model, written per label: the first epoch's E-step is
    # exactly the reference's, so only the M-step differs.
    label_covs = ScaleCovsJob(DuplicateCovsJob(global_cov, gmm.NUM_LABELS).out_covs, TAU).out_covs
    for arm, cfg in COV_ARMS.items():
        balanced = COV_TRAIN_SCALE[arm]
        for scale in [balanced] + ([1.0] if arm in COV_EM_ARMS and balanced != 1.0 else []):
            name = f"{arm}_from-{COV_BRANCH_EPOCH}_ds-{scale}"
            result = chunked_clustering(
                num_epochs=COV_EPOCHS,
                flavor=label_covariance_flavor(
                    centroids=reference.out_artifacts["centroids"][COV_BRANCH_EPOCH],
                    covs=label_covs,
                    mixtures=reference.out_artifacts["mixtures"][COV_BRANCH_EPOCH],
                    distance_scale=scale,
                    **cfg,
                    **flavor_common,
                ),
                distance_scale=scale,
                num_chunks=NUM_CHUNKS,
                rqmt=EPOCH_RQMT,
                alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
                **common,
            )
            tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{name}.json",
                               result.out_statistics)
            # The EM arm decodes over the balanced arm's sweep as well, since its
            # best decode scale may well be near there rather than near 1.0.
            add_branch(name, {"arm": arm, "from": COV_BRANCH_EPOCH, "train": scale}, result,
                       COV_EPOCHS, COV_DECODE_EPOCHS, scale, SCALE_SWEEP_EPOCHS,
                       sweep_bases=[balanced] if scale != balanced else [])

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
