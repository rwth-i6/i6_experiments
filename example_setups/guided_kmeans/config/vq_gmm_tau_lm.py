"""Does the LM have to be cheaper while the acoustic classes form?

``vq_gmm_tau_unseg`` and the transition-scale probe both converged confidently to
the wrong answer. Measured directly (``setup.emission_balance``), the reason is
not that the acoustic model never becomes confident - it does - but what it
becomes confident *about*.

**What the oracle run shows.** It shares this family's initial table, codebook,
covariance, AM scale and LM; only the segmentation is given. Its emission model
ends up strong enough to live with ``lm_scale = 1.0``: at convergence the
acoustic evidence for a segment's label is 2.19 nats against the LM's 0.55-nat
pull between two plausible candidates, and the acoustics would gain a median
2.39 nats by placing a true boundary, against the 1.89 nats the LM charges for
one more phoneme - 58% of true boundaries payable. So 1.0 is *not* wrong at
convergence, which is why it looked so stable in every segmented run: there the
LM never had to price a boundary at all.

**What the unsegmented runs show.** Their acoustic models become confident too -
by epoch 40 the frame-level gap is 70% of the oracle's - but the boundary
payback stops growing at about epoch 25, at 0.69 (S=1) and 0.95 (S=2), with
29-35% of true boundaries payable. Confident labels that do not separate
*adjacent* phonemes: the segmentation merged early, the merged segments trained
merged classes, and merged classes cannot pay for the boundaries that would undo
them. Waiting does not help; that is a fixed point, not a slow start.

    epoch   payback oracle   payback unseg S=2     PER oracle   PER unseg S=2
      5          0.15              0.03               84.8          85.5
     10          1.04              0.26               74.5          83.1
     20          1.99              0.85               31.1          81.8
     30          2.38              0.96               10.8          81.7

The runs part company between epochs 5 and 15, while the classes form. During
that window the unsegmented run can afford almost none of the true boundaries at
the LM's price (0% at epoch 5, 8% at epoch 10). The hypothesis here is that it is
the LM price *during that window* that locks the merged segmentation in.

**Why S=2 at p=1/2.** In a fixed-length utterance a topology with S states and
loop probability p contributes ``T log p + S n log((1-p)/p)`` for n phonemes, so
it acts on the segmentation only as a per-phoneme price ``S log(p/(1-p))``
(``emission_balance.transition_boundary_penalty``). At p=1/2 that price is zero.
The LM is then the *only* boundary price in the model, which makes this a
single-variable sweep - and S=2 was the best arm of ``vq_gmm_tau_unseg``. The
transition scale is irrelevant at p=1/2 (it only adds ``T ts log 1/2`` to log Z)
and is left at 1.0 so that the control below is the same job.

**The control is free.** The ``lm_scale = 1.0`` arm hashes identically to
``vq_gmm_tau_unseg``'s S=2 arm, so its epochs are already computed and it costs
nothing but one decode at epoch 15.

**Readout.** ``boundary_payback`` against ``boundary_price`` and
``payable_boundaries``, in the convergence table and beside PER in the decode
table. Breaking out means payback rising past the control's ~0.95 plateau toward
the oracle's 2.39. What each outcome means, stated before the fact:

* payback breaks out and PER follows -> the LM price was the lock. The natural
  next step is to continue the winner at ``lm_scale = 1.0`` from its model, on
  the oracle's evidence that a confident model can sustain it;
* payback breaks out but PER does not -> the boundaries came back but label
  identity paid for the weaker LM; the same continuation at 1.0 is then the
  test of whether it recovers;
* payback does not move -> the fixed point is not about the LM price, and the
  next suspect is the starting point (e.g. initialising from oracle-trained
  mixtures).

Only the *training* LM scale changes. Decoding stays at ``lm_scale = 1.0`` on
mean-pooled cv, as everywhere in the ladder, so PER remains comparable.
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
from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import (
    DatasetConfig,
    SegmentFile,
)
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import (
    LatexTableReport,
    TextTableReport,
    clustering_statistics_per_epoch,
    param,
)
from i6_experiments.example_setups.guided_kmeans.setup.emission_balance import (
    BalanceFramesJob,
    EmissionBalanceJob,
    PhonemeLMCostJob,
    transition_boundary_penalty,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
)

exp_dir = "vq_gmm_tau_lm"
version = 1

# Carried over from vq_gmm_tau_oracle_seg unchanged.
LM_ORDER = 5
SIGMA = 0.02
BEAM_SIZE = 1_000
NUM_LABELS = 40
NUM_CODEWORDS = 512
SEED = 42
TAU = 4.0
#: Twenty. The oracle and unsegmented runs part company between epochs 5 and 15
#: and the unsegmented payback has plateaued by about 25, so whether a lower LM
#: price breaks the plateau should be readable by epoch 15 and settled by 20.
#: Epoch chains do not hash this count: extending resumes rather than recomputes.
WEIGHTS_EPOCHS = 40
WEIGHTS_DECODE_EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40]

#: Epochs at which EmissionBalanceJob measures each arm's model. Dense through
#: the window where the oracle and unsegmented runs diverge, and a superset of
#: the decode epochs so PER and payback sit in the same decode-table row. Each is
#: a mini task of a few seconds over a matrix BalanceFramesJob computes once.
BALANCE_EPOCHS = [1, 3, 5, 8, 10, 13, 15, 20, 23, 25, 28, 30, 33, 35, 38, 40]

#: The standard decode is handed cv's true boundaries (mean-pooled per GMM
#: segment), so it cannot show a segmentation error - hypothesis length comes
#: out at 0.954-0.963 of the reference for every model, converged or stuck - and
#: segmentation is what this sweep is about. Every decoded model is therefore
#: decoded a second time, on cv *frames* with free segmentation: the same
#: decode_and_score routine and RASR Viterbi search, only the input and the loop
#: differ. A segmentation error then shows as deletions, and as a
#: hypothesis/reference ratio below one (that ratio is 1 - Del + Ins exactly, so
#: it is derived from the scorer rather than computed separately).
#:
#: At one fixed LM scale for every arm, so that differences between arms are the
#: models' and not the decoder's: a model trained at 0.25 decoded at 0.25 would
#: also get a decoder that inserts boundaries more cheaply, whatever it learned.
#: 1.0 because it is the standard decode's scale, and because it is the test of
#: whether a model trained under a cheaper LM can then live with the full one.
MATCHED_DECODE_LM = 1.0

#: Loop probability of that decode. RASR's topology has one state per phoneme,
#: and in a fixed-length utterance a loop probability p acts only as a
#: per-phoneme price log(p / (1 - p)); at 1/2 that is zero, so - as in this
#: sweep's training topology - the LM is the only thing pricing a boundary.
FREE_SEG_LOOP = 0.5

#: Share of cv decoded with free segmentation: a seeded ~10% (~280 sequences),
#: the same for every arm and every decode scale. Measured on
#: vq_unsupervised_frames' frame-level probe decodes, this decode runs ~8 ms per
#: frame (557 sequences in 36 min), so the full 1.3M cv frames would take ~3 h -
#: inside the decode job's fixed 4 h limit but without headroom for a slow node,
#: and ~70 decode-hours over the 23 decodes this sweep and the oracle grid ask
#: for. This subset takes ~18 min each. A 200-sequence subset reproduced the
#: standard decode's full-cv PER to within 0.6.
FREE_SEG_FRACTION = 0.1

#: The attribution check: the *control* model (this sweep's 1.0 arm, at its last
#: decoded epoch) decoded at each of the sweep's LM scales. Read against the
#: arms above. If decoding the unchanged control model at 0.25 already closes the
#: deletion gap, the gap is the search's price and not the model; if it does not,
#: the merged classes are in the model and training is the right place to fix
#: them. vq_gmm_tau_oracle_seg runs the same grid on its converged model.
ATTRIBUTION_LM_SCALES = [1.0, 0.5, 0.25, 0.1]
ATTRIBUTION_EPOCH = 20

#: Decode-side sweep, identical to the oracle config so PER compares directly.
BASE_DECODE_SCALE = 1.0
DECODE_DISTANCE_SCALES = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
SCALE_SWEEP_EPOCHS = [20]

#: Fixed, not swept: holding the AM scale is what isolates the LM scale. 0.35 is
#: the value the oracle sweep settled on and the one every unsegmented arm ran
#: at, which is what lets vq_gmm_tau_unseg serve as this sweep's control.
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


#: ``(num_states, mean duration in frames, transition_scale, lm_scale)``.
#:
#: S=2 at a 4-frame mean, i.e. p=1/2, throughout: zero transition penalty, so
#: the LM is the only boundary price (see the module docstring). The 1.0 arm is
#: vq_gmm_tau_unseg's S=2 arm job for job. 0.5 and 0.25 put the LM's 1.89-nat
#: phoneme price at 0.94 and 0.47; 0.1 brackets from below at 0.19, where the
#: LM's pull on label identity (0.055 nats) is gone for practical purposes and
#: the label sequence is the acoustics' alone.
ARMS = [
    (2, MEAN_DURATION_FRAMES, 1.0, 1.0),
    (2, MEAN_DURATION_FRAMES, 1.0, 0.5),
    (2, MEAN_DURATION_FRAMES, 1.0, 0.25),
    (2, MEAN_DURATION_FRAMES, 1.0, 0.1),
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

    # Identical to vq_gmm_tau_unseg's, so tau=4 means the same thing and the
    # lm_scale=1.0 arm is that run's S=2 arm, job for job.
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
    # Inert for training (an explicit flavor bypasses its only reader); kept as
    # vq_gmm_tau_unseg has it so the control arm hashes identically.
    train_config = _recog_config(1.0, 0.0, 1.0, use_fb=True)
    # Decoding is unchanged from every other arm in the family, LM scale
    # included: only the *training* LM scale is under test.
    decode_config_rasr = _recog_config(1.0, 0.0, None, use_fb=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)

    # Emission vs LM, the quantity this sweep is read on. The true segmentation
    # enters here as an evaluation reference only, as the GMM alignment does for
    # FER; nothing on the training path reads it. Both jobs are shared with any
    # other config that builds them over the same features and densities.
    balance_frames = BalanceFramesJob(
        features_hdf=ls100.out_features,
        frame_labels=ls100.out_labels,
        centroids=COLLEAGUE_CENTROIDS_K512,
        covs=covs,
    ).out_frames
    lm_cost = PhonemeLMCostJob(
        arpa_path=phonetic_lm_dict[LM_ORDER],
        lexicon=fb_lexicon,
        backoff_fb_root=tools.BACKOFF_FB,
    ).out_cost

    # Free-segmentation evaluation (MATCHED_DECODE_LM): the standard decode
    # routine on cv frames with silence dropped -- the same job
    # vq_unsupervised_frames builds, so it is shared -- over a fixed seeded subset
    # of cv's own segment list (FREE_SEG_FRACTION). decode_and_score filters the
    # reference to the same subset.
    cv_frames = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["cv"]["features"],
        alignment=GMM_ALIGNMENT_CV,
        exclude_labels=(0,),
        pooling="none",
    ).out_features
    free_seg_segments = ShuffleAndSplitSegmentsJob(
        cv.out_segments,
        split={"decode": FREE_SEG_FRACTION, "rest": 1.0 - FREE_SEG_FRACTION},
    ).out_segments["decode"]
    cv_frames_dataset = DatasetConfig(
        audio_hdf_path=cv_frames,
        sampling_method=SegmentFile(free_seg_segments),
        precomputed=True,
    )

    def matched_values(model_dir, decode_lm, decode_name):
        """Decode cv frames with free segmentation at `decode_lm` and return the
        values the matched_* columns read."""
        dc = DecodeConfig(
            centroids=COLLEAGUE_CENTROIDS_K512,
            model_dir=model_dir,
            recog_rasr_config=_recog_config(decode_lm, FREE_SEG_LOOP, 1.0, use_fb=False),
            # Emissions are per frame here, as in training, so the training AM
            # scale -- not the standard decode's 1.0 on segment means.
            distance_scale=AM_SCALE,
        )
        res = decode_and_score(
            decode_name, "cv", dc, cv_frames_dataset,
            rasr_path=tools.RASR_PATH, device="cpu",
            corpus_key="train-other-960",
        )
        tk.register_output(f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per)
        return {
            "matched_per": res.per,
            "matched_del": res.deletion,
            "matched_ins": res.insertion,
            "matched_sub": res.substitution,
            # hyp = ref - deletions + insertions, so the length ratio follows
            # from the scorer's own counts.
            "matched_hyp_ref": 1 - res.deletion + res.insertion,
        }

    lm_column = param("lm_scale", header=("Train", "LM scale"))
    matched_columns = ["matched_per", "matched_del", "matched_ins",
                       "matched_sub", "matched_hyp_ref"]
    balance_columns = ["segment_gap", "boundary_payback", "boundary_price",
                       "payable_boundaries"]

    latex_report = LatexTableReport(
        columns=[lm_column, "epoch", "per", "del", "ins", "sub"]
                + matched_columns
                + ["posterior_entropy", "segment_duration_argmax",
                   "boundary_rate"] + balance_columns,
        sort_by=["lm_scale"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Training LM scale on silence-free ls-100h frames with free "
            f"segmentation, S=2 at p=1/2 so that the LM is the only boundary "
            f"price, am={AM_SCALE}, tau={TAU}. The 1.0 arm is "
            f"vq\\_gmm\\_tau\\_unseg's S=2 arm. Boundary payback is what the "
            f"acoustics gain by placing a true boundary, against the price of "
            f"one more phoneme; the oracle-segmented run reaches 2.39 against "
            f"1.89 at convergence, the unsegmented control plateaus at 0.95. "
            f"Decoded on mean-pooled cv at LM scale 1.0, unchanged; the "
            f"free-segmentation columns run the same decode on cv frames with no "
            f"boundaries given (loop {FREE_SEG_LOOP}, LM scale {MATCHED_DECODE_LM} "
            f"for every arm, a fixed {int(100 * FREE_SEG_FRACTION)}% of cv), so "
            f"they can show a segmentation error -- as deletions and a "
            f"hypothesis/reference ratio below one -- that the standard decode "
            f"cannot."
        ),
    )
    recog_results = []

    # Control model decoded at each sweep scale; see ATTRIBUTION_LM_SCALES.
    attribution_report = TextTableReport(
        columns=[param("decode_lm", header=("Decode", "LM scale"))] + matched_columns,
        sort_by=["decode_lm"],
        drop_empty_rows=True,
        title=(
            f"Attribution: the control model (training LM scale 1.0, epoch "
            f"{ATTRIBUTION_EPOCH}) decoded with free segmentation at each of the "
            f"sweep's LM scales. Read against the sweep arms decoded at "
            f"{MATCHED_DECODE_LM}. If a cheaper decode alone closes the deletion "
            f"gap here, the gap is the search's price, not the model's classes."
        ),
    )

    # One row per recognition pass, so it fills in as the run goes. epoch_offset=0:
    # row k shows pass k (which ran with model k-1); the balance columns are
    # measured on model k, the one that pass produced -- the pass's outcome,
    # where its statistics are its input.
    progress_report = TextTableReport(
        columns=[lm_column, "epoch", "log_likelihood_gain", "posterior_entropy",
                 "cluster_perplexity", "segment_duration_argmax",
                 "boundary_rate", "dead_clusters"] + balance_columns,
        sort_by=["lm_scale"],
        drop_empty_rows=True,
        title=(
            f"Per-epoch convergence for the training-LM-scale sweep (S=2, "
            f"p=1/2, am={AM_SCALE}). Read boundary payback against boundary "
            f"price: the unsegmented control plateaus at a payback of ~0.95 from "
            f"epoch ~25 with 35% of true boundaries payable; the oracle-segmented "
            f"run reaches 2.39 with 58%. A lower LM scale that breaks the fixed "
            f"point shows here as payback climbing past the control's plateau. "
            f"Balance columns are measured on the model each pass produced and "
            f"are blank at epochs not in BALANCE_EPOCHS."
        ),
    )

    for num_states, mean_dur, trans_scale, lm_scale in ARMS:
        p = loop_probability(mean_dur, num_states)
        am_scale = AM_SCALE
        name = (f"weights_tau-{TAU}_s-{num_states}_loop-{round(p, 4)}"
                f"_ts-{trans_scale}_lm-{lm_scale}_am-{am_scale}")
        backoff_fb = BackoffFBSearch(
            backoff_fb_root=tools.BACKOFF_FB,
            arpa_path=phonetic_lm_dict[LM_ORDER],
            lexicon=fb_lexicon,
            # The parameter under test. It scales the LM log-probabilities in the
            # compiled graph, and with it the price of every extra phoneme.
            lm_scale=lm_scale,
            emission_scale=1.0,
            loop_probability=p,
            silence_loop_probability=None,
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
        progress_stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=name, epoch_offset=0,
            lexicon=lexicon,
        )

        penalty = transition_boundary_penalty(num_states, p, trans_scale)
        balance = {}
        for e in BALANCE_EPOCHS:
            if e not in result.out_artifacts["mixtures"]:
                continue
            job = EmissionBalanceJob(
                frames=balance_frames,
                mixtures=result.out_artifacts["mixtures"][e],
                am_scale=am_scale,
                lm_cost=lm_cost,
                lm_scale=lm_scale,
                transition_penalty=penalty,
            )
            job.add_alias(f"guided_kmeans/{exp_dir}/{name}/balance/epoch_{e:03d}")
            balance[e] = {
                "segment_gap": job.out_segment_gap,
                "boundary_payback": job.out_payback,
                "boundary_price": job.out_price,
                "payable_boundaries": job.out_payable,
            }

        arm_params = {"lm_scale": lm_scale}
        for epoch_key in sorted(int(k) for k in result.out_epoch_statistics):
            progress_report.add_row(
                params=dict(arm_params), epoch=epoch_key,
                statistics=progress_stats,
                values=dict(balance.get(epoch_key, {})),
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
                    # A decode of model e and the balance of model e: the same
                    # model, so this row has no epoch skew between its columns.
                    values = dict(balance.get(e, {}))
                    values.update(matched_values(
                        result.out_models[e], MATCHED_DECODE_LM,
                        f"{name}_ep-{e}_freeseg-lm-{MATCHED_DECODE_LM}"))
                    if res.fer is not None:
                        values["fer"] = res.fer
                    latex_report.add_row(
                        result=res, params=dict(arm_params), epoch=e,
                        statistics=stats, values=values,
                    )

        if lm_scale == 1.0:
            for decode_lm in ATTRIBUTION_LM_SCALES:
                attribution_report.add_row(
                    params={"decode_lm": decode_lm},
                    values=matched_values(
                        result.out_models[ATTRIBUTION_EPOCH], decode_lm,
                        f"{name}_ep-{ATTRIBUTION_EPOCH}_freeseg-lm-{decode_lm}"),
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
    text_report = latex_report.as_text()
    text_report.register(
        f"guided_kmeans/{exp_dir}/report_{version}.txt",
        required=text_report, register_dependencies=False, update_frequency=60,
    )
    progress_report.register(
        f"guided_kmeans/{exp_dir}/convergence_{version}.txt",
        required=progress_report, update_frequency=60,
    )
    attribution_report.register(
        f"guided_kmeans/{exp_dir}/attribution_{version}.txt",
        required=attribution_report, update_frequency=60,
    )


def py():
    run()
