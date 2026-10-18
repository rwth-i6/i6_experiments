"""How wrong may the boundaries be before training fails?

Two runs bracket the question, and neither is new. ``vq_gmm_tau_oracle_seg``
(am 0.35) is given the GMM alignment's segmentation and converges: PER 74.5,
31.1 and 10.8 at epochs 10, 20 and 30. ``vq_gmm_tau_unseg`` (S = 1, loop 0.75,
am 0.35) searches for the boundaries itself and does not: PER 83.3, 82.9 and
83.3. Everything else about the two is the same: frozen k512 codebook, tau = 4,
initial table, LM, AM scale and frames. This config degrades the oracle by
degrees and finds where between the two the training stops working.

**Every arm is the failing run plus evidence about the boundaries.** The search
is the unseg arm's, a loop-0.75 duration prior included. Per-frame transition
scores from ``OracleBoundaryTransitions`` are added to it (the op's SPEC.md
14.1). Both endpoints are therefore runs that already exist, exactly and not
approximately (both checked in ``test_chunked``):

* hard boundaries make the prior a constant per segment, so the posteriors are
  the oracle run's;
* as the evidence goes flat, the prior is all that is left, so the posteriors
  are the unseg run's.

Both are rebuilt below with the arguments they ran with, so their jobs are
reused and their rows are already filled.

**Three ways to degrade the oracle**, one family of arms each:

* ``boundary_error`` eps: the alignment's boundaries as a noisy observation,
  each frame wrong with probability eps, independently. Each disagreement with
  the alignment costs lambda = log((1 - eps)/eps) nats: dropping or adding a
  boundary one lambda, moving one two. The arms, eps = 0.01 and 0.1, are
  lambda = 4.6 and 2.2, and eps -> 0.5 is lambda -> 0, the free segmentation.
  This is the axis between the two endpoints.
* ``jitter`` sigma: hard boundaries, each moved by round(sigma N(0, 1)) frames,
  one draw per sequence, the same in every epoch: a worse aligner. The number of
  segments and their phonemes stay right, only the positions are wrong.
  The arm is sigma = 1. Measured on every 10th ls-100 utterance (one-frame
  segments clip many shifts):

      sigma   boundaries moved   mean |shift|   frames in the wrong segment
        0.5         31%              0.31                  7.8%
        1.0         60%              0.73                 17.9%
        2.0         78%              1.45                 33.5%

* ``blur`` sigma (postponed, no arms yet): the boundary probability widened
  into a Gaussian of width sigma frames around each alignment boundary, peaking
  at 1 - eps on it, at eps = 0.01. Moving a boundary gets cheaper the closer
  it stays. Unlike jitter this cannot keep the count: per-frame scores cannot
  say "exactly one boundary near here", and frames within ~1.18 sigma of a
  boundary favour a second boundary of their own, held back only by the LM's
  price for one more phoneme (see ``transitions.boundary_transitions``). Read
  these arms with that in mind.

**Cost.** An arm is the unseg arm's op work: ~6.5 GPU-hours per epoch, six
~1.4 h chunks. ``EPOCHS`` = 20 is where the endpoints have separated (31.1
against 82.9), so it is enough to tell a working arm from a failing one, if
not to see a slow arm converge. The epochs are a chain, so raising it later
continues an arm and recomputes nothing. At 20 epochs each arm is ~130
GPU-hours. The three arms run now (two error rates and one jitter, chosen in
the comments at the grid) are ~390; the full grid of ten was ~1,300.
"""

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
    BackoffFBSearch,
)
from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.chunked.spec import Spec
from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.chunked.transitions import (
    OracleBoundaryTransitions,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
    create_fb_lexicon,
    phonetic_lm_dict,
)
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import (
    decode_and_score,
    DecodeConfig,
)
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import (
    DatasetConfig,
    SegmentFile,
)
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import (
    TextTableReport,
    clustering_statistics_per_epoch,
    param,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
)
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_unseg as unseg
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_oracle_trans as check

exp_dir = "vq_gmm_tau_boundary_noise"
version = 1

AM_SCALE = 0.35
#: The unseg arm every arm here is built on: S = 1 at the MLE mean duration of
#: 4 frames, p = 0.75.
NUM_STATES = 1
LOOP = unseg.loop_probability(unseg.MEAN_DURATION_FRAMES, NUM_STATES)

EPOCHS = 20
DECODE_EPOCHS = [10, 20]

#: The grid, one list per way of degrading the oracle; every arm is
#: independent of the others. Chosen 2026-10-12 for a first, cheaper round of
#: three arms.
#:
#: The two error rates bracket what a boundary has to pay for itself here. The
#: LM charges ~1.89 nats for one more phoneme (vq_gmm_tau_bonus), and the
#: loop-0.75 prior a further logit(1/4) = 1.10 nats for an exit rather than a
#: loop, so a boundary costs ~3 nats. The acoustics pay back only 0.03-0.05
#: nats of that in the first epochs, ~1 at the free run's plateau and 2.4 in
#: the oracle run (vq_gmm_tau_lm, at S=2). The evidence pays lambda =
#: log((1 - eps)/eps): 4.6 nats at eps = 0.01, above the price, and 2.2 at 0.1,
#: below it. If the first converges and the second does not, 0.05 (2.9 nats,
#: at the price) is the arm to run next; the others in the original grid were
#: 0.2 and 0.3.
BOUNDARY_ERRORS = [0.01, 0.1]
#: 1 frame moves 60% of the boundaries and puts 18% of the frames in the wrong
#: segment: a usable but imperfect aligner. The original grid also had 0.5
#: (7.8%) and 2 (33.5%).
JITTERS = [1.0]
#: Postponed on 2026-10-12. The code stays, and the arms come back by filling
#: this list; the original grid was [1.0, 2.0] at BLUR_BOUNDARY_ERROR.
BLURS = []
BLUR_BOUNDARY_ERROR = 0.01


def arms():
    """``(family, value, OracleBoundaryTransitions kwargs)`` per arm. Only the
    knobs an arm sets go into the spec."""
    out = [("eps", e, {"boundary_error": e}) for e in BOUNDARY_ERRORS]
    out += [("jitter", j, {"jitter": j}) for j in JITTERS]
    out += [("blur", b, {"boundary_error": BLUR_BOUNDARY_ERROR, "blur": b}) for b in BLURS]
    return out


def _free_arm(*, transitions, name, alias_dir, ls100, covs, init_table, train_config,
              lexicon, fb_lexicon):
    """The unseg S = 1, loop 0.75, am 0.35 arm, with ``transitions`` added.

    With ``transitions=None`` this must stay argument for argument what
    vq_gmm_tau_unseg runs, so its jobs are reused.
    """
    backoff_fb = BackoffFBSearch(
        backoff_fb_root=tools.BACKOFF_FB,
        arpa_path=phonetic_lm_dict[unseg.LM_ORDER],
        lexicon=fb_lexicon,
        lm_scale=1.0,
        emission_scale=1.0,
        loop_probability=LOOP,
        silence_loop_probability=None,
        transition_scale=1.0,
        num_states=NUM_STATES,
        apply_sentence_end=True,
        transitions=transitions,
        batch_size=unseg.BATCH_SIZE,
        max_frames=unseg.MAX_FRAMES,
        checkpoint_interval=-1,
        storage_dtype="float32",
    )
    flavor = mixture_flavor(
        centroids=COLLEAGUE_CENTROIDS_K512,
        covs=covs,
        mixtures=init_table,
        recognition_config=train_config,
        lexicon=lexicon,
        num_clusters=unseg.NUM_LABELS,
        distance_scale=AM_SCALE,
        use_forward_backward=True,
        update_densities=False,
        update_covariances=None,
        mixture_floor=1e-2,
        num_workers=unseg.NUM_WORKERS,
        backoff_fb=backoff_fb,
    )
    return chunked_clustering(
        num_epochs=EPOCHS,
        features_hdf=ls100.out_features,
        recognition_config=train_config,
        lexicon=lexicon,
        num_clusters=unseg.NUM_LABELS,
        flavor=flavor,
        distance_scale=AM_SCALE,
        use_forward_backward=True,
        rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
        num_chunks=unseg.NUM_CHUNKS,
        num_workers=unseg.NUM_WORKERS,
        parent_threads=unseg.PARENT_THREADS,
        rqmt=unseg.EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{alias_dir}/{name}",
    )


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    ls100 = unseg.frame_level_ls100()
    cv = silence_free_cv_features()

    # As in both endpoint configs, so every arm shares these jobs with them.
    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    covs = ScaleCovsJob(
        DuplicateCovsJob(global_cov, unseg.NUM_CODEWORDS).out_covs, unseg.TAU
    ).out_covs
    init_table = NormalTableJob(
        unseg.NUM_LABELS, unseg.NUM_CODEWORDS, sigma=unseg.SIGMA, seed=unseg.SEED
    ).out_table
    train_config = unseg._recog_config(1.0, 0.0, 1.0, use_fb=True)
    decode_config_rasr = unseg._recog_config(1.0, 0.0, None, use_fb=False)
    cv_dataset = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )
    common = dict(ls100=ls100, covs=covs, init_table=init_table, train_config=train_config,
                  lexicon=lexicon, fb_lexicon=fb_lexicon)

    runs = []  # (family, value, name, result)
    _, oracle = check._arm(per_frame=False, num_epochs=EPOCHS, **common)
    runs.append(("oracle", 0.0, "oracle", oracle))
    for family, value, knobs in arms():
        name = f"weights_tau-{unseg.TAU}_s-{NUM_STATES}_loop-{round(LOOP, 4)}_am-{AM_SCALE}_{family}-{value}"
        transitions = Spec(OracleBoundaryTransitions,
                           {"frame_labels": ls100.out_labels, **knobs})
        result = _free_arm(transitions=transitions, name=name, alias_dir=exp_dir, **common)
        tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{name}.json",
                           result.out_statistics)
        runs.append((family, value, name, result))
    free_name = f"weights_tau-{unseg.TAU}_s-{NUM_STATES}_loop-{round(LOOP, 4)}_am-{AM_SCALE}"
    free = _free_arm(transitions=None, name=free_name, alias_dir=unseg.exp_dir, **common)
    runs.append(("free", 0.5, "free", free))
    # Table order: the eps axis from one endpoint to the other, then the
    # families that leave it.
    rank = {"oracle": 0, "eps": 1, "free": 2, "jitter": 3, "blur": 4}

    report = TextTableReport(
        columns=[param("family", header=("Oracle",)), param("value", header=("Value",)),
                 "epoch", "per", "del", "ins", "sub", "log_likelihood_gain",
                 "posterior_entropy", "cluster_perplexity", "max_cluster_fraction",
                 "segment_duration_argmax", "boundary_rate", "dead_clusters", "l1"],
        sort_by=["rank", "value"],
        drop_empty_rows=True,
        title=(
            f"The oracle segmentation degraded by degrees, tau={unseg.TAU}, am={AM_SCALE}. "
            f"Every arm is vq_gmm_tau_unseg's S=1 loop-{LOOP} search plus per-frame "
            f"boundary evidence from the GMM alignment: 'eps' a per-frame boundary "
            f"error rate, 'jitter' hard boundaries moved by round(sigma N(0,1)) "
            f"frames, 'blur' boundaries widened into a Gaussian of width sigma at "
            f"eps={BLUR_BOUNDARY_ERROR}. 'oracle' (vq_gmm_tau_oracle_seg) and 'free' "
            f"(vq_gmm_tau_unseg) are the endpoints, eps=0 and eps->0.5. Row k is "
            f"model k: the statistics of the pass that recognized with it, and at "
            f"the decode epochs its PER on mean-pooled cv. The alignment's own "
            f"segments average 3.99 frames at a boundary rate of 0.251."
        ),
    )
    for family, value, name, result in runs:
        stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=f"{exp_dir}_{name}", epoch_offset=1,
            lexicon=lexicon,
        )
        decodes = {}
        for e in DECODE_EPOCHS:
            dc = DecodeConfig(
                centroids=COLLEAGUE_CENTROIDS_K512,
                model_dir=result.out_models[e],
                recog_rasr_config=decode_config_rasr,
                distance_scale=unseg.BASE_DECODE_SCALE,
                write_frame_labels=True,
            )
            decodes[e] = decode_and_score(
                f"{name}_ep-{e}_ds-{unseg.BASE_DECODE_SCALE}", "cv", dc, cv_dataset,
                rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960",
            )
        for epoch in range(0, EPOCHS + 1):
            report.add_row(result=decodes.get(epoch),
                           params={"rank": rank[family], "family": family, "value": value},
                           epoch=epoch, statistics=stats)

    report.register(
        f"guided_kmeans/{exp_dir}/sweep_{version}.txt",
        required=report, update_frequency=60,
    )


def py():
    run()
