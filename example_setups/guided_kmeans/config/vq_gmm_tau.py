"""Does a Gaussian mixture buy anything over the VQ table? A covariance-scale sweep.

The VQ+table model works (PER falling past 50% and still improving at 150
epochs). The question is whether a mixture over the *same* codebook does better,
and the honest answer requires knowing what a mixture can even do differently.

**At small covariance a mixture is exactly the VQ table.** Writing
``Sigma_c = tau * Sigma_0``, as ``tau -> 0``

    score_l(x) -> -log w_{l,c*}  +  C(x, tau)

with ``c*`` the nearest density and ``C`` independent of the label. A
label-independent per-frame term cancels in Viterbi (every path consumes every
frame) and in forward-backward (it divides out of the per-frame posterior). So
the mixture weights *are* the table, and measured on a trained model the VQ
approximation reproduces the mixture's own label ranking on 97.7% of frames.

So ``tau`` is the only knob that buys anything new, and it trades two things
against each other. Measured on these features with the k512 codebook and a
colleague's table:

    tau     label-dependence of the      label contrast seen
            centroid update (mean TV)    by the search (nats)
    0.25          0.045                    14.75  (97% of VQ)
    1.0           0.167                    13.06  (86%)
    2.0             -                      11.56  (76%)
    4.0           0.483                     8.22  (54%)
    8.0             -                       4.35  (29%)
    16.0          0.894                     1.98  (13%)

The left column is the whole reason a mixture can fine-tune centroids where a VQ
model cannot. The mean update is driven by ``sum_l gamma_tl p(c | l, x)``, and
when one density dominates ``p(c*|l,x) = 1`` for *every* label - so the update
collapses to plain k-means and never sees the labels at all. At the corpus
covariance it is ~83% label-blind. Softness is not an alternative to centroid
fine-tuning; it is the mechanism by which label information reaches the means.

The right column is the price: as ``tau`` grows ``score_l -> -log sum_c w_lc = 0``
and the acoustic model stops discriminating. Hence the sweep, and hence the
window tau in [2, 8] where both columns are non-trivial.

**``distance_scale`` is one constant for every arm, and swept at decode.** An
earlier version compensated it per tau from a contrast table measured on the
wrong object - a colleague's trained table rather than this config's near-uniform
sigma=0.02 initialization. The real contrast does not vary with tau at any epoch
(the table absorbs it), so that compensation corrected nothing and instead made
the effective AM/LM balance run 13.7 to 32.4 nats across the arms. See
BASE_DISTANCE_SCALE below for the measurements. Arms are now compared at their
*best* decode scale, which settles the balance question by measurement rather
than by prediction.

**Covariances are never estimated**, for two independent reasons. Statistically,
a full covariance at D=512 has 131,328 free parameters and 512 densities over
ls-100h leaves ~6,800 vectors each - 0.05 per parameter, the regime that
produced an acoustic model scoring below chance. Mechanically, re-estimating
would pull the scale straight back to the data's own within an epoch, so tau
would not survive as a hyperparameter at all.

Two arms:

``weights``
    tau sweep with the densities frozen. Isolates what softness alone does. The
    smallest tau should reproduce the VQ result, which is a free correctness
    check on the whole mixture path.
``means``
    tau sweep with the centroids moving and their shapes frozen, initialized
    from the corresponding ``weights`` run's table. This is the actual
    hypothesis - fine-tuning the codebook once the table is good - and it is
    staged because moving centroids against a bad table walks them toward
    whatever that table happens to say.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
    COLLEAGUE_CENTROIDS_K512,
    PHONEME_LM_ZIJIAN_3GRAM,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    DuplicateCovsJob,
    GlobalCovarianceJob,
    NormalTableJob,
    ScaleCovsJob,
    chunked_clustering,
    mixture_flavor,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_recog_rasr_config,
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
    silence_free_cv_features,
    silence_free_ls100_features,
)

exp_dir = "vq_gmm_tau"
version = 1

# The two settings carried over from the VQ work: a 5-gram rather than a 3-gram,
# and the sharper initialization. Both were established there, not here.
LM_ORDER = 5
SIGMA = 0.02
BEAM_SIZE = 1_000

NUM_LABELS = 40
NUM_CODEWORDS = 512
NUM_CHUNKS = 22
# A task's 10 CPUs, split between RASR search processes and the parent's own
# threads (Mahalanobis scoring, accumulation). Both unhashed - retune freely.
# Measured with 9 + 1: workers ran at 3.5x concurrency waiting on a saturated
# parent (search ~11-14 s/seq, parent ~3-4 s/seq single-threaded), and the
# parent's work scales 1.7x / 2.2x / 2.4x at 2 / 3 / 4 threads - which makes 7 + 3
# the split where neither side waits much. See parent_thread_budget for how to
# read the task log if that stops holding.
NUM_WORKERS = 7
PARENT_THREADS = 3
EPOCH_RQMT = {"mem": 8}

WEIGHTS_EPOCHS = 60          # arm 1: tau sweep, densities frozen
WEIGHTS_DECODE_EPOCHS = [10, 20, 30, 40, 50, 60]
MEANS_EPOCHS = 60            # arm 2: continues from arm 1's table at this epoch
MEANS_FROM_EPOCH = 60
MEANS_DECODE_EPOCHS = [5, 10, 20, 30, 40, 50, 60]

#: Measured median label contrast at each tau on the initial table, against
#: 15.25 nats for the tau->0 (VQ) limit. Used to hold the AM/LM balance fixed
#: across the sweep. Approximate - the contrast drifts as the table sharpens -
#: but far better than leaving it uncompensated.
# --- distance_scale ---------------------------------------------------------
# An earlier version of this config compensated distance_scale per tau, on the
# theory that softer assignment shrinks the label contrast and so shifts the
# AM/LM balance. That compensation was wrong, and the way it was wrong is worth
# recording because the mistake is easy to repeat.
#
# The contrast table it used was measured on a colleague's *trained* k512 table
# (perplexity ~36), where the contrast really does fall with tau: 14.75 nats at
# tau=0.25 down to 4.35 at tau=8. But the initialization these runs actually use
# is NormalTableJob(sigma=0.02), which is near-uniform - perplexity 430-440. The
# contrast of *that* table, measured per epoch on the real runs:
#
#     tau     ep1    ep5   ep10   ep20   ep30
#     0.25    0.9   12.3   13.0   13.2   13.3
#     1.00    0.9   12.4   13.1   13.3   13.3
#     2.00    0.9   12.4   13.1   13.3   13.4
#     4.00    0.8   11.7   12.3   12.7   12.7
#     8.00    0.6    7.5    8.5    9.0    9.3
#
# It does not vary with tau - not at epoch 1, not later. The table absorbs tau:
# a softer E-step concentrates the weights in the M-step (perplexity 27.8 at
# tau=0.25 against 8.6 at tau=8) and that sharpening restores almost exactly the
# discriminability the softness removed.
#
# So the per-tau compensation corrected a spread that never existed, and instead
# injected one: effective contrast ran 13.7 nats at tau=0.25 up to 32.4 at
# tau=8, a 2.4x spread monotone in tau. In effective-LM terms the arms decoded
# at lm_scale 0.97 down to 0.285 - and the supervised sweep puts the optimum at
# 1.0 with 0.0 catastrophic (97.7% PER). The reported PER ordering
# (51.2% -> 86.2% with rising tau) is collinear with that artifact, so those
# numbers cannot separate "softness hurts" from "the AM was over-weighted".
#
# One constant for every arm, therefore. tau is then the only thing that varies.
BASE_DISTANCE_SCALE = 1.0

#: Decode-side sweep. Only the AM/LM *ratio* matters, so sweeping this at fixed
#: lm_scale is equivalent to sweeping lm_scale, and it costs no retraining -
#: distance_scale on DecodeConfig is decode-side only. Comparing arms at their
#: *best* scale removes the balance question instead of trying to predict it,
#: which is what the per-tau table failed at. Widened at both ends after the
#: first sweep put the optimum on its edge for four of five arms (2.0 for
#: tau<=2, 0.7 for tau=8), i.e. never bracketed it.
DECODE_DISTANCE_SCALES = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]

#: Epochs at which the full scale sweep runs. Everywhere else a single scale is
#: used, so the trajectory stays readable without paying len(scales)x for it.
SCALE_SWEEP_EPOCHS = [30, 60]

#: Training-side scale. None restores the legacy per-tau values (and the 480
#: epoch jobs computed under them); a float makes every arm train at one scale.
#:
#: 1.0 since the decode sweep ruled out the decode side. At epoch 30, with each
#: arm decoded at its own best of {0.7, 1.0, 1.4, 2.0}:
#:
#:     tau     0.25   1.0    2.0    4.0    8.0
#:     best    49.13  72.39  74.90  80.58  84.54   (PER, %)
#:     @ds     2.0    2.0    2.0    1.0    0.7
#:
#: The decode scale moves PER by 1-4 points within an arm and leaves the
#: ordering across arms untouched, so the decode-side artifact does not explain
#: it. What remains is the *training* scale, which ran 1.03 at tau=0.25 up to
#: 3.51 at tau=8 - and the best decode scale drifting the opposite way is what
#: tables learned under a stronger AM would do. Retraining at one scale is the
#: only way to separate that from a genuine tau effect.
TRAIN_DISTANCE_SCALE = 1.0

_LEGACY_LABEL_CONTRAST = {0.25: 14.75, 1.0: 13.06, 2.0: 11.56, 4.0: 8.22, 8.0: 4.35, 16.0: 1.98}
_LEGACY_VQ_CONTRAST = 15.25
DISTANCE_SCALE_FOR_TAU = {
    t: _LEGACY_VQ_CONTRAST / c for t, c in _LEGACY_LABEL_CONTRAST.items()
}


def train_distance_scale(tau):
    """The scale the training search runs at - see TRAIN_DISTANCE_SCALE."""
    return DISTANCE_SCALE_FOR_TAU[tau] if TRAIN_DISTANCE_SCALE is None else TRAIN_DISTANCE_SCALE

WEIGHTS_TAUS = [0.25, 1.0, 2.0, 4.0, 8.0]
# WEIGHTS_TAUS = [3.25]
MEANS_TAUS = [1.0, 2.0, 4.0]
# MEANS_TAUS = []
SEED = 42


def _recog_config(lm_scale, loop_prob, transition_scale, use_fb):
    return create_recog_rasr_config(
        lm_scale=lm_scale,
        emission_scale=1.0,
        transition_scale=transition_scale,
        loop_probability=loop_prob,
        silence_loop_probability=loop_prob,
        use_forward_backward_search=use_fb,
        lm_order=LM_ORDER,
        use_eow_phonemes=False,
        max_beam_size=BEAM_SIZE,
        lm_path=PHONEME_LM_ZIJIAN_3GRAM if LM_ORDER == 3 else None,
    )


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    ls100 = silence_free_ls100_features()
    ls100.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_nosil")
    cv = silence_free_cv_features()
    cv.add_alias(f"guided_kmeans/{exp_dir}/features_cv_nosil")

    # The covariance every tau scales. Computed on the features the run uses, so
    # tau=1 really is "the shape of this data" rather than an arbitrary unit.
    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    base_covs = DuplicateCovsJob(global_cov, NUM_CODEWORDS).out_covs

    cv_dataset = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )
    train_config = _recog_config(1.0, 0.0, 1.0, use_fb=True)
    decode_config_rasr = _recog_config(1.0, 0.0, None, use_fb=False)

    latex_report = LatexTableReport(
        columns=[
            "tau", "epoch", "per", "del", "ins", "sub", "log_likelihood"
        ],
        sort_by=["tau"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Gaussian mixture over the frozen k512 codebook at covariance scale tau, "
            f"$$ p(x|k) = \\mathcal{{N}}(x | \\mu_k, \\tau \\Sigma), $$"
            f"decoded on silence-free cv. tau->0 is the VQ table exactly, so the "
            f"smallest tau is a control rather than an experiment."
            f"The densities are frozen; only the weights are updated."
        ),
    )
    latex_report_means = LatexTableReport(
        columns=[
            "tau", "epoch", "per", "del", "ins", "sub", "log_likelihood"
        ],
        sort_by=["tau"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Gaussian mixture over the frozen k512 codebook at covariance scale tau, "
            f"decoded on silence-free cv. tau->0 is the VQ table exactly, so the "
            f"smallest tau is a control rather than an experiment."
            f"Only the centroids move with their shapes fixed, "
            f"starting from the matching weights-only run's table at epoch "
            f"{MEANS_FROM_EPOCH}."
        ),
    )
    reports = {"weights": latex_report, "means": latex_report_means}
    recog_results = []

    def _decode(exp_result, name, arm, tau, epochs, statistics):
        diags = {
            e: MixtureDiagnosticsJob(exp_result.out_artifacts["mixtures"][e]) for e in epochs
        }
        for e, job in diags.items():
            tk.register_output(
                f"guided_kmeans/{exp_dir}/diagnostics/{name}_ep-{e}.json", job.out_diagnostics
            )
        for e in epochs:
            # The full sweep only at the designated epochs: arms are compared
            # best-against-best there, while the other epochs keep one scale so
            # the trajectory stays readable without costing len(scales)x.
            scales = (
                DECODE_DISTANCE_SCALES if e in SCALE_SWEEP_EPOCHS else [BASE_DISTANCE_SCALE]
            )
            for scale in scales:
                dc = DecodeConfig(
                    centroids=COLLEAGUE_CENTROIDS_K512,
                    model_dir=exp_result.out_models[e],
                    recog_rasr_config=decode_config_rasr,
                    # Decode-side only, so sweeping it costs no retraining. Only
                    # the AM/LM ratio matters, so this is equivalent to sweeping
                    # lm_scale and covers the balance question directly instead
                    # of predicting it from a contrast measurement.
                    distance_scale=scale,
                    write_frame_labels=True,
                )
                decode_name = f"{name}_ep-{e}_ds-{scale}"
                res = decode_and_score(
                    decode_name, "cv", dc, cv_dataset,
                    rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960",
                )
                if res.frame_labels is not None:
                    res.fer = FrameErrorRateJob(
                        res.frame_labels, GMM_ALIGNMENT_CV, lexicon
                    ).out_fer
                    tk.register_output(
                        f"guided_kmeans/{exp_dir}/eval/{decode_name}_fer", res.fer
                    )
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per
                )
                recog_results.append(res)
                if scale == BASE_DISTANCE_SCALE:
                    reports[arm].add_row(
                        result=res,
                        params={"tau": tau},
                        epoch=e,
                        statistics=statistics,
                        values={k: v for k, v in (("mi", diags[e].out_mi), ("fer", res.fer))
                                if v is not None},
                    )

    def _run(*, arm, tau, mixtures, num_epochs, update_covariances):
        name = f"{arm}_tau-{tau}"
        flavor = mixture_flavor(
            centroids=COLLEAGUE_CENTROIDS_K512,
            covs=ScaleCovsJob(base_covs, tau).out_covs,
            mixtures=mixtures,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=NUM_LABELS,
            distance_scale=train_distance_scale(tau),
            use_forward_backward=True,
            # Frozen for the 'weights' arm; means-only for the 'means' arm. The
            # covariances are never estimated in either - see the module docstring.
            update_densities=update_covariances is not None,
            update_covariances=update_covariances,
            mixture_floor=1e-2,
            num_workers=NUM_WORKERS,
        )
        result = chunked_clustering(
            num_epochs=num_epochs,
            features_hdf=ls100.out_features,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=NUM_LABELS,
            flavor=flavor,
            distance_scale=train_distance_scale(tau),
            use_forward_backward=True,
            rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
            num_chunks=NUM_CHUNKS,
            num_workers=NUM_WORKERS,
            parent_threads=PARENT_THREADS,
            rqmt=EPOCH_RQMT,
            alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
        )
        tk.register_output(
            f"guided_kmeans/{exp_dir}/statistics/{name}.json", result.out_statistics
        )
        stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=name, epoch_offset=1, lexicon=lexicon
        )
        return name, result, stats

    init_table = NormalTableJob(
        NUM_LABELS, NUM_CODEWORDS, sigma=SIGMA, seed=SEED
    ).out_table

    weights_runs = {}
    for tau in WEIGHTS_TAUS:
        name, result, stats = _run(
            arm="weights", tau=tau, mixtures=init_table,
            num_epochs=WEIGHTS_EPOCHS, update_covariances=None,
        )
        weights_runs[tau] = result
        _decode(result, name, "weights", tau,
                # sorted({0, WEIGHTS_EPOCHS // 2, WEIGHTS_EPOCHS}), stats)
                sorted(WEIGHTS_DECODE_EPOCHS), stats)

    for tau in MEANS_TAUS:
        # Not a continuation in the job-reuse sense: update_covariances=False is
        # a different accumulator, so this is a fresh chain whose starting table
        # is the weights arm's. That staging is the point - the centroids should
        # only start moving once the table directing them is worth following.
        name, result, stats = _run(
            arm="means", tau=tau,
            mixtures=weights_runs[tau].out_artifacts["mixtures"][MEANS_FROM_EPOCH],
            num_epochs=MEANS_EPOCHS, update_covariances=False,
        )
        _decode(result, name, "means", tau,
                MEANS_DECODE_EPOCHS, stats)

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results), required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")
    latex_report_means.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.means.tex")


def py():
    run()
