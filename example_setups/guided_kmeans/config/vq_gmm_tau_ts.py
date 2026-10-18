"""Does the transition model need scaling down? A short probe, not a sweep.

``vq_gmm_tau_unseg`` converged confidently to the wrong answer: PER flat at
81.7-84.4, posterior entropy down to 0.86-1.14, the objective dead flat past
epoch 31, no dead clusters and near-uniform label usage. The duration statistics
say why, and they say it the same way in all four arms.

**The measurement.** Phoneme segments come out 1.3-2.1x too long, and every arm
overshoots its *own* duration prior:

    arm                prior   argmax dur   boundary rate   PER
    S=2, p=1/2           4        5.17          0.194       81.7
    S=1, p=2/3           3        6.78          0.147       82.4
    S=1, p=3/4           4        7.87          0.127       83.5
    S=1, p=4/5           5        8.51          0.118       84.4
    GMM reference        -        3.99          0.251          -

Order the arms by how far their durations sit from the reference and the PER
order falls out exactly. Asked for 3 frames the model produces 6.8; asked for 5
it produces 8.5. It is not merely obeying the duration prior, it is going past
it - which is what happens when the acoustics are too weak to break a segment.

**Why that is a scaling problem.** The emission and transition terms are both
accumulated once per frame, but this setup scales them independently: emissions
by ``distance_scale`` (0.35), transitions by ``transition_scale`` (1.0). So the
duration prior carries ~2.9x the weight of the acoustics in a decision the two
are supposed to share. Measured on the trained epoch-30 model: crossing a
boundary costs 1.099 nats at ``transition_scale = 1.0``, while the median
acoustic gap between the best and second-best label at ``am = 0.35`` is 0.325
nats - a 3.4x imbalance. Tying the transition scale to the AM gives 0.385 nats,
which is the balance one would want.

**And the 1.0 was never a choice.** With ``loop_probability = 0`` - which
``vq_gmm_tau``, ``vq_gmm_tau_cuda`` and ``vq_gmm_tau_oracle_seg`` all use - the
transition weights are -inf and 0 whatever the scale, so ``transition_scale``
has been *inert in every arm this family has ever run*. ``vq_gmm_tau_unseg`` is
the first config to activate it, at a value inherited from configs where it did
nothing. This is the first test of it, not an extrapolation from a tested one.

**The readout is duration, not PER.** That is what makes this a probe. The
duration statistics exist from epoch 1 and settle fast - in the finished run the
boundary rate at epoch 31 was within 2% of its epoch-40 value - so ten epochs
answer the question that forty epochs of PER would. What each outcome means,
stated before the fact:

* boundary rate rises toward 0.251 and duration falls toward 3.99, PER improves
  -> the hypothesis holds, and the winner is worth extending to 40 epochs;
* durations correct but PER does not move -> segmentation was not the binding
  constraint and the label-identity problem lies elsewhere (the LM scale or the
  frozen codebook next). This is the outcome that saves a long detour;
* boundary rate overshoots 0.251 and PER worsens -> the scale is now too low and
  the answer lies between 0.35 and 1.0, bracketed from below.

``am_scale`` is held at 0.35 throughout. Holding it fixed is what isolates the
transition scale, and the oracle sweep argues against moving it anyway (am-0.5
was worse than am-0.35 there).

**No control arm:** ``vq_gmm_tau_unseg`` *is* the ``transition_scale = 1.0``
control, at the same AM scale and the same two topologies, forty epochs deep.

**One consequence of the parameterisation.** ``transition_scale`` enters only as
``ts * log p``, so at ``ts = 0`` the loop probability drops out of the model
entirely and every S=1 loop value collapses to the same thing. The S=1 flat-prior
arm is therefore *one* arm, not three, and its ``p`` is arbitrary - set to 2/3
only so that it differs from its ``ts = 0.35`` partner in exactly one parameter.
S=2 still differs at ``ts = 0``, because it structurally requires two frames per
phoneme. Note also that ``p = 0`` must not be used here: that is the no-loop
segment-level model, not a flat prior.

Everything else - frozen k512 codebook, tau=4 on the frame-level covariance,
weights-only updates, the 5-gram, silence-free frames, and decoding on
mean-pooled cv at loop 0 - is ``vq_gmm_tau_unseg``'s, unchanged, so PER stays
comparable across the whole ladder.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
    GMM_ALIGNMENT_LS960_FRAME,
    COLLEAGUE_CENTROIDS_K512,
)
from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
    SegmentedFeaturesFromAlignmentJob,
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
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_recog_rasr_config,
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
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import (
    LatexTableReport,
    TextTableReport,
    clustering_statistics_per_epoch,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
)

exp_dir = "vq_gmm_tau_ts"
version = 1

# Carried over from vq_gmm_tau_oracle_seg unchanged.
LM_ORDER = 5
SIGMA = 0.02
BEAM_SIZE = 1_000
NUM_LABELS = 40
NUM_CODEWORDS = 512
SEED = 42
TAU = 4.0
#: Ten, because the readout is the duration statistics and those settle inside
#: ten epochs (the finished run's boundary rate at epoch 31 was within 2% of its
#: epoch-40 value). PER will not be converged here and is not meant to be - it
#: is read for direction only, which the finished run shows is already right at
#: epoch 10. Epoch chains do not hash this count, so extending a winner to 40
#: resumes rather than recomputes.
WEIGHTS_EPOCHS = 20
WEIGHTS_DECODE_EPOCHS = [5, 10, 15, 20]

#: Decode-side sweep, identical to the oracle config so PER compares directly.
BASE_DECODE_SCALE = 1.0
DECODE_DISTANCE_SCALES = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
SCALE_SWEEP_EPOCHS = [10]

#: Fixed, not swept: holding the AM scale is what isolates the transition scale.
#: 0.35 is the value the oracle sweep settled on (converged at 10.79 PER by
#: epoch 30, against 11.97 at epoch 40 for 0.5) and the one every arm of
#: vq_gmm_tau_unseg ran at, which is what makes that run this probe's control.
AM_SCALE = 0.35

#: 20 ms per frame, and ~80 ms per phoneme from the literature. The GMM
#: alignment measures 3.99 frames (79.8 ms) per silence-free segment over this
#: corpus, so the prior and the data agree and d = 4 is the MLE anchor.
FRAME_RATE_MS = 20
MEAN_PHONEME_MS = 80
MEAN_DURATION_FRAMES = MEAN_PHONEME_MS / FRAME_RATE_MS

#: Silence is not phoneme-shaped -- its run lengths over the GMM alignment
#: measure 6.98 frames (140 ms) against the phonemes' 4.02 -- but these features
#: have none, so no arm here uses this. Recorded because it is the value the
#: unsegmented variant would need, and because the same measurement is what
#: confirms the phoneme anchor from data rather than from the literature: 4.02
#: frames implies p = 0.7510 at S = 1 and 0.5019 at S = 2, i.e. the 3/4 and 1/2
#: this sweep is built on.
MEAN_SILENCE_FRAMES = 7.0


def loop_probability(mean_frames: float, num_states: int) -> float:
    """The loop probability giving a mean phoneme duration of `mean_frames`.

    Duration is negative binomial with ``r = num_states``, mean
    ``num_states / (1 - p)``; inverting gives ``p = 1 - num_states / d``. At
    ``num_states = 1`` this is the geometric MLE.
    """
    p = 1.0 - num_states / float(mean_frames)
    if not 0.0 <= p < 1.0:
        raise ValueError(
            f"mean duration {mean_frames} frames is unreachable with "
            f"{num_states} state(s): it must be >= {num_states} (each state "
            f"consumes at least one frame), and p came out {p}")
    return p


#: ``(num_states, mean duration in frames, transition_scale)``, one entry per
#: arm. The two topologies are the best of vq_gmm_tau_unseg (S=2 at 81.7, and
#: S=1 p=2/3 at 82.4); each is run at the AM-tied scale and at zero, which
#: brackets the existing 1.0 control from below.
#:
#: The S=1 zero-scale arm's duration is written as the same 3.0 as its partner
#: purely so the pair differs in one parameter - at transition_scale 0 the loop
#: probability has no effect at all (see the module docstring), so any nonzero
#: value would give the identical model. That also makes a free correctness
#: check available: a second S=1 arm at zero with a different duration must
#: produce bit-identical statistics.
ARMS = [
    (1, 3.0, 0.35),
    (1, 3.0, 0.0),
    (2, MEAN_DURATION_FRAMES, 0.35),
    (2, MEAN_DURATION_FRAMES, 0.0),
]

#: Chunks per epoch per arm. The work is 4x the oracle config's: there the
#: recognizer summed a segment's ~4 frames into one observation before calling
#: the op, so the op saw 3.46M observations per epoch; here every one of the
#: 13.81M frames is its own observation. From the two configs' measured chunk
#: times the cost decomposes into 1.63 ms per op observation and 0.44 ms per
#: frame of parent scoring, i.e. ~7.9 GPU-hours per epoch per arm.
#:
#: Measured on this config rather than estimated: S=1 runs 1.70 ms/frame and
#: S=2 2.51 ms (1.48x, so the chain update is a larger share of the op than the
#: state count alone suggests), i.e. 6.5 and 9.6 GPU-hours per epoch per arm.
#: Four arms make that ~29 GPU-hours per epoch, so wall-clock is set by how many
#: GPUs are held and not by this number -- what it actually controls is
#: how much of the per-task startup gets paid. Each task cf-caches the 28.5 GB
#: feature file (~4 min) and compiles the LM graph (~105 s), so at 6 chunks that
#: fixed cost is 7% of a ~1.4 h task; at 12 it would be 13% for no gain.
#:
#: Batch size is not the lever it looks like: measured across all three bench
#: pairs, doubling B costs 1.79-1.81x the time, so it buys ~11% throughput and
#: the op is bandwidth-bound rather than launch-bound. Wall-clock here is linear
#: in GPUs held and in epochs run, and in essentially nothing else short of
#: pruned mode (SPEC.md section 7), which would trade away the exactness that is
#: this op's entire advantage over the RASR arm.
#:
#: Raised from 6 to 16 on 2026-10-07 to admit gpu_11gb (below): a 1080 Ti runs
#: the op ~3x slower than an L40S, so at 6 chunks one S=2 task there would take
#: ~5 h and hold its whole epoch back; at 16 it is ~2 h, and the extra startup
#: is paid on cards that would otherwise sit idle. Unhashed, and an epoch whose
#: directory is already set up keeps the count it started with
#: (GuidedClusteringEpochJob._scheduled_num_chunks), so this takes effect from
#: each arm's next epoch and can be set back the same way.
NUM_CHUNKS = 16
NUM_WORKERS = 0
PARENT_THREADS = 4

#: ``time`` is set explicitly and is not decoration: settings.py's
#: check_engine_limits defaults it to **2 hours**, and a chunk here runs ~1.4 h
#: on an L40S and ~1.8 h on a V100 -- close enough to the default that a slow
#: node would be killed mid-task. 4 h leaves headroom on either card.
#:
#: Both GPU partitions are requested because the per-user QOS caps, not the
#: partition sizes, bound occupancy: gpu_48gb_qos allows one user 5 of its 16
#: L40S, while gpu_32gb_qos caps nothing. Slurm places each chunk on whichever
#: frees first. An explicit "-p" also bypasses the gpu_mem ladder, so gpu_mem is
#: deliberately absent rather than set and ignored.
#:
#: gpu_24gb (A10, its own QOS cap of 6) is added because gpu_32gb was down to
#: one down and one draining node on 2026-10-07, leaving the 5 L40S as the only
#: cards and one epoch's 6 chunks enough to fill them. An A10 runs the op at
#: 3.17 ms/frame against the L40S's 1.32 (tools/backoff_fb/README.md), so a
#: chunk there takes ~3.4 h -- hence ``time`` 8 rather than 4.
#:
#: gpu_11gb (1080 Ti / 2080 Ti, 11 GB, QOS cap 70) is admitted by halving the
#: S=2 arms' batch (BATCH below). Op-only bench, S=1 B=16 / S=2 B=8, T=1000,
#: ms per sequence-frame: 1080 Ti 3.24 / 5.69, 2080 Ti 2.99 / 4.92, against the
#: V100's 1.19 at S=1 (/work/asr4/mann/tmp/bench_11gb_2026-10-07). ``mem`` is
#: 12 because chunks peak at 6.7 GB RSS and the 1080 Ti nodes have 62 GB for
#: four cards -- 16 would fit only three tasks per node.
#:
#: Memory: peak is checkpointed alpha-hat, measured at 5.34 GB for B=16, T=1000,
#: S=1 (bench/README.md). The longest sequence here is 1094 frames and S=2
#: doubles NQ to 1.86M, so the worst arm peaks near 11 GB -- comfortable on
#: either card. ``mem`` is host RAM and unrelated.
EPOCH_RQMT = {
    "mem": 12,
    "gpu": 1,
    "cpu": 4,
    "time": 8,
    # Faulty nodes (cn-245's dead 1080 Ti, 2026-10-07) are excluded by
    # settings.py from node_blacklist.json, not here.
    "sbatch_args": ["-p", "gpu_48gb,gpu_24gb,gpu_11gb,gpu_32gb"],
}

#: Sequences per op call. Unlike the oracle config, T is now a frame count
#: (mean 489, max 1094), not a segment count, so MAX_FRAMES rather than
#: BATCH_SIZE is what binds: 16 x 1094 = 17,504 is why it sits at 18,000. At the
#: oracle config's 4,000 the longest batches would collapse to B=3 and the GPU
#: would idle. Both unhashed - throughput only.
#:
#: Per arm, keyed by num_states, so every arm fits an 11 GB card. Peak memory
#: is linear in B x T and in S: S=1 at 16 x 1000 measures 5.34 GB, S=2 at
#: 8 x 1000 5.04 GB, while S=2 at the old 16 / 18,000 would need ~11 GB. The
#: op is exact per sequence, so batch composition changes no result beyond
#: float summation order; the cost is ~11% throughput for S=2 on the large
#: cards (doubling B costs 1.8x the time). To go back, set S=2 to (16, 18_000).
BATCH = {
    1: (16, 18_000),
    2: (8, 9_000),
}


def frame_level_ls100():
    """ls-100h frames with silence dropped: 28,234 sequences, 13,810,231 frames.

    The same job ``vq_gmm_tau_oracle_seg`` and ``vq_unsupervised_frames`` build,
    so the 28.5 GB file and its covariance are shared rather than recomputed.

    Dropping silence uses the alignment, so it *is* oracle information, and the
    alternative was available: the raw ``wav2vec2_ls100h.hdf`` needs no
    alignment at all and has 15,160,794 frames, 9% of them silence. That was
    measured and rejected on purpose. Taking it would change two things at once
    against the oracle-segmented run -- the boundaries and the presence of
    silence -- and the two cannot be separated afterwards, because removing
    silence is itself an oracle operation. Keeping the silence-free input
    isolates the one variable this config exists to test, which is whether the
    search can find the boundaries. Silence is a separate experiment.

    ``out_labels`` is deliberately *not* used here, and that omission is the
    whole difference from the oracle-segmented run.
    """
    return SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["ls-100"]["features"],
        alignment=GMM_ALIGNMENT_LS960_FRAME,
        exclude_labels=(0,),
        pooling="none",
        rqmt={"cpu": 2, "mem": 16, "time": 12},
    )


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
    )


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    ls100 = frame_level_ls100()
    ls100.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_frames_nosil")
    cv = silence_free_cv_features()
    cv.add_alias(f"guided_kmeans/{exp_dir}/features_cv_nosil")

    # The frame-level covariance, the same job the oracle config uses, so tau=4
    # means the same thing in both and the comparison is not confounded by the
    # metric.
    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    base_covs = DuplicateCovsJob(global_cov, NUM_CODEWORDS).out_covs
    covs = ScaleCovsJob(base_covs, TAU).out_covs
    init_table = NormalTableJob(
        NUM_LABELS, NUM_CODEWORDS, sigma=SIGMA, seed=SEED
    ).out_table

    cv_dataset = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )
    # Inert for training: passing an explicit `flavor` means chunked_clustering
    # never calls _flavor_from_flags, which is the only reader of this, and the
    # op supplies the search. Kept identical to the oracle config so the two
    # differ only where they are meant to. The topology that actually runs is
    # the BackoffFBSearch below.
    train_config = _recog_config(1.0, 0.0, 1.0, use_fb=True)
    # Decoding is unchanged from every other arm in the family: Viterbi, loop 0,
    # on mean-pooled cv. That is what keeps PER comparable across the ladder.
    decode_config_rasr = _recog_config(1.0, 0.0, None, use_fb=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)

    # The diagnostic set, chosen for what a forward-backward run can actually
    # report. Its counter produces no Viterbi traceback, so the loop-count,
    # segment-duration and AM/LM-split columns are empty here however useful they
    # look in the library -- what it does produce is the objective, the posterior
    # entropy, and the label distribution, which between them separate the three
    # ways this run could be failing:
    #
    #   log_likelihood_gain   is the objective moving at all? (the raw value is
    #                         swamped by the Gaussian normalisation; see the column)
    #   posterior_entropy     is the search committing to labels, or hedging?
    #   label perplexity      is it using the inventory, or concentrating?
    #   dead_clusters         has anything collapsed to exactly zero?
    #   l1                    how far is the label prior from the phoneme prior?
    #
    # A run that is confidently wrong looks quite unlike one that is undecided, and
    # these tell them apart; PER alone does not.
    latex_report = LatexTableReport(
        columns=["states", "loop", "transition_scale", "am_scale", "epoch", "per",
                 "del", "ins", "sub", "log_likelihood", "log_likelihood_gain",
                 "posterior_entropy", "cluster_perplexity",
                 "max_cluster_fraction", "loop_frequency", "segment_duration",
                 "segment_duration_argmax", "boundary_rate", "dead_clusters", "l1"],
        sort_by=["states", "transition_scale"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Probing the transition-model scale on silence-free ls-100h "
            f"frames with free segmentation. The emission and transition terms "
            f"are both per-frame but scaled independently, and at "
            f"transition\\_scale 1.0 crossing a boundary costs 1.099 nats "
            f"against a median acoustic gap of 0.325 at am={AM_SCALE} -- a "
            f"3.4x imbalance. Each of the two best topologies from "
            f"vq\\_gmm\\_tau\\_unseg is run with the scale tied to the AM "
            f"and at zero; that run is the 1.0 control. Ten epochs, because the "
            f"duration statistics rather than PER are the readout. Decoded on "
            f"mean-pooled cv, unchanged. "
        ),
    )
    recog_results = []

    # A statistics-only table, one row per *recognition pass* rather than per
    # decode. The table above can only have rows where a decode exists
    # (WEIGHTS_DECODE_EPOCHS), which leaves the diagnostics invisible for every
    # epoch in between -- and its last row is structurally blank, because with
    # epoch_offset=1 a decode of centroids[e] reads statistics key e+1 and the
    # final epoch has no successor pass. Convergence is exactly what one wants
    # to read epoch by epoch, so it gets its own table, and since no decode is
    # involved every epoch that has finished appears in it.
    #
    # epoch_offset=0 here, deliberately differing from the decode table: a row
    # labelled epoch k shows pass k, the pass that ran with centroids[k-1].
    # Aligning it to the decode instead would relabel every row by one and make
    # the two tables disagree about what "epoch" means for the same numbers.
    progress_report = TextTableReport(
        columns=["states", "loop", "transition_scale", "am_scale", "epoch",
                 "log_likelihood", "log_likelihood_gain", "posterior_entropy",
                 "cluster_perplexity", "max_cluster_fraction",
                 "loop_frequency", "segment_duration",
                 "segment_duration_argmax", "boundary_rate", "dead_clusters",
                 "l1"],
        sort_by=["states", "transition_scale"],
        drop_empty_rows=True,
        title=(
            f"Per-epoch convergence for the transition-scale probe. One row "
            f"per recognition pass, so it fills in as the run goes rather than "
            f"only at the decoded epochs. Duration columns: the soft estimate is "
            f"an expectation under a frame-independence assumption (an upper "
            f"bound on the boundary rate), the argmax one is the run length of "
            f"the argmax label sequence (a lower bound); the GMM alignment's own "
            f"mean is 3.99 frames at a boundary rate of 0.251, and "
            f"vq\\_gmm\\_tau\\_unseg reached 5.17-8.51 at 0.118-0.194. This "
            f"is the table the probe is read from."
        ),
    )

    for num_states, mean_dur, trans_scale in ARMS:
        p = loop_probability(mean_dur, num_states)
        am_scale = AM_SCALE
        name = (f"weights_tau-{TAU}_s-{num_states}"
                f"_loop-{round(p, 4)}_ts-{trans_scale}_am-{am_scale}")
        backoff_fb = BackoffFBSearch(
            backoff_fb_root=tools.BACKOFF_FB,
            arpa_path=phonetic_lm_dict[LM_ORDER],
            lexicon=fb_lexicon,
            # No frame_labels: this is the plain BackoffFBRecognizer,
            # which searches over boundaries instead of being handed
            # them. That single omission is the experiment.
            lm_scale=1.0,
            emission_scale=1.0,
            loop_probability=p,
            # Silence is dropped from these features, so <SIL> has no
            # frames to win and its duration model is immaterial;
            # tying it to the phoneme loop keeps each arm a single
            # knob rather than adding a second, untested axis. It
            # would need its own value on unsegmented input, where
            # silence runs measure 6.98 frames against 4.02.
            silence_loop_probability=None,
            # The parameter under test. It multiplies the finite
            # transition log-weights only -- a forbidden transition
            # stays forbidden at any scale -- so 0.0 is a flat duration
            # prior rather than a broken graph.
            transition_scale=trans_scale,
            num_states=num_states,
            apply_sentence_end=True,
            batch_size=BATCH[num_states][0],
            max_frames=BATCH[num_states][1],
            checkpoint_interval=-1,
            storage_dtype="float32",
        )
        flavor = mixture_flavor(
            centroids=COLLEAGUE_CENTROIDS_K512,
            covs=covs,
            mixtures=init_table,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=NUM_LABELS,
            distance_scale=am_scale,
            use_forward_backward=True,
            update_densities=False,
            update_covariances=None,
            mixture_floor=1e-2,
            num_workers=NUM_WORKERS,
            backoff_fb=backoff_fb,
        )
        result = chunked_clustering(
            num_epochs=WEIGHTS_EPOCHS,
            features_hdf=ls100.out_features,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=NUM_LABELS,
            flavor=flavor,
            distance_scale=am_scale,
            use_forward_backward=True,
            rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
            num_chunks=NUM_CHUNKS,
            num_workers=NUM_WORKERS,
            parent_threads=PARENT_THREADS,
            rqmt=EPOCH_RQMT,
            alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
        )
        tk.register_output(
            f"guided_kmeans/{exp_dir}/statistics/{name}.json",
            result.out_statistics,
        )
        stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=name, epoch_offset=1,
            lexicon=lexicon,
        )
        # Same extraction jobs as `stats` (identical inputs, so sisyphus
        # reuses them); only the row->key alignment differs.
        progress_stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=name, epoch_offset=0,
            lexicon=lexicon,
        )
        arm_params = {
            "states": num_states,
            "loop": round(p, 4),
            "transition_scale": trans_scale,
            "am_scale": am_scale,
        }
        for epoch_key in sorted(int(k) for k in result.out_epoch_statistics):
            progress_report.add_row(
                params=dict(arm_params), epoch=epoch_key,
                statistics=progress_stats,
            )

        for e in WEIGHTS_DECODE_EPOCHS:
            scales = (
                DECODE_DISTANCE_SCALES if e in SCALE_SWEEP_EPOCHS
                else [BASE_DECODE_SCALE]
            )
            for scale in scales:
                dc = DecodeConfig(
                    centroids=COLLEAGUE_CENTROIDS_K512,
                    model_dir=result.out_models[e],
                    recog_rasr_config=decode_config_rasr,
                    distance_scale=scale,
                    write_frame_labels=True,
                )
                decode_name = f"{name}_ep-{e}_ds-{scale}"
                res = decode_and_score(
                    decode_name, "cv", dc, cv_dataset,
                    rasr_path=tools.RASR_PATH, device="cpu",
                    corpus_key="train-other-960",
                )
                if res.frame_labels is not None:
                    res.fer = FrameErrorRateJob(
                        res.frame_labels, GMM_ALIGNMENT_CV, lexicon
                    ).out_fer
                    tk.register_output(
                        f"guided_kmeans/{exp_dir}/eval/{decode_name}_fer",
                        res.fer,
                    )
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/per/{decode_name}_per",
                    res.per,
                )
                recog_results.append(res)
                if scale == BASE_DECODE_SCALE:
                    latex_report.add_row(
                        result=res,
                        params={
                            "states": num_states,
                            "loop": round(p, 4),
                            "transition_scale": trans_scale,
                            "am_scale": am_scale,
                        },
                        epoch=e,
                        statistics=stats,
                        values={"fer": res.fer} if res.fer is not None else {},
                    )

    recog_report = create_report(recog_results)
    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=recog_report, required=recog_report, update_frequency=60,
    )
    latex_report.register(
        f"guided_kmeans/{exp_dir}/tex/report_{version}.tex",
        required=latex_report, update_frequency=60,
    )
    # The same table as plain text, for reading a run in flight without a LaTeX
    # pass. as_text() shares the rows, columns, grouping and cell resolution with
    # the .tex above -- only the printer differs -- so the two cannot disagree.
    # register_dependencies=False because the .tex already registered the
    # statistics Variables; doing it twice would duplicate the .deps tree.
    text_report = latex_report.as_text()
    text_report.register(
        f"guided_kmeans/{exp_dir}/report_{version}.txt",
        required=text_report, register_dependencies=False, update_frequency=60,
    )
    # register_dependencies=True here: these rows reference the epoch_offset=0
    # statistics, and nothing else in the config keeps those alive.
    progress_report.register(
        f"guided_kmeans/{exp_dir}/convergence_{version}.txt",
        required=progress_report, update_frequency=60,
    )


def py():
    run()
