"""The tau=4.0 arm on frames again, but with the segmentation *searched for*.

``vq_gmm_tau_oracle_seg`` trains on every frame with a GMM alignment telling the
search where the phoneme boundaries are. This removes that oracle entirely: the
HMM has to find the boundaries itself, through a self-loop.

That makes this the first arm in the family whose search is a real
frame-synchronous HMM decode rather than a relabelling of a known segmentation,
and the loop probability is the new knob it introduces.

**The input keeps one piece of oracle information, deliberately.** The features
are the silence-free frames ``vq_gmm_tau_oracle_seg`` and
``vq_unsupervised_frames`` use -- 13,810,231 of them -- and dropping silence
took the alignment to do. That is cheating, and it is the cheaper of two bad
options. The honest alternative, raw ``wav2vec2_ls100h.hdf`` with no alignment
anywhere, carries 15,160,794 frames of which 9% are silence, and taking it would
change *two* things at once against the oracle-segmented run: the boundaries and
the presence of silence. Those cannot be separated afterwards, because removing
silence is itself an oracle operation, so the run would answer neither question
cleanly.

Holding silence out isolates the single variable this config exists to test --
whether the search can find the boundaries on its own. What silence costs a
frame-synchronous model is a real question, and a separate one; the measurement
it needs is already here (silence runs average 6.98 frames against the phonemes'
4.02, so it wants its own loop probability, not the phoneme one).

**Where the loop probabilities come from.** A phoneme's duration under this
topology is the sum of ``S`` geometric state occupancies -- negative binomial
with ``r = S`` -- so its mean is ``S / (1 - p)`` and

    p = 1 - S / d        (d = mean duration in frames)

The literature prior is ~80 ms per phoneme, and at this feature rate (20 ms per
frame) that is d = 4. The corpus agrees without being asked: phoneme run lengths
over the GMM alignment average 4.02 frames (80.4 ms), which implies p = 0.7510
at S = 1 and 0.5019 at S = 2. So d = 4 is the MLE anchor from both directions
and 3/4 and 1/2 are the values this sweep is built around. (The alignment is
used *here*, offline, to choose a prior -- not by the pipeline, which never
reads it.)

**Why S = 2 is worth the arm.** At a matched mean of 4 frames the two topologies
disagree about short phonemes, not about average ones. The geometric puts its
mode at *one* frame and 25% of its mass there; the negative binomial with r = 2
cannot emit a one-frame phoneme at all and peaks at 2. Since a one-frame
phoneme is the failure mode a frame-synchronous search falls into when the
acoustics are ambiguous, that difference is the hypothesis, and holding the mean
fixed is what isolates it from a plain duration-prior change.

**What is swept and what is not.** S = 1 gets three loop probabilities spanning
mean durations of 3, 4 and 5 frames; S = 2 gets the MLE only. Both run at the
two training AM scales the oracle sweep left standing (0.35 and 0.5) -- that
sweep had not separated them when this was built, so carrying both is a hedge
rather than a second hypothesis. Eight arms.

**What stays fixed, so the comparison means something.** Frozen k512 codebook,
tau = 4 on the same frame-level covariance the oracle config scales,
weights-only updates, the 5-gram, and decoding on mean-pooled cv with the RASR
Viterbi search at loop 0. Every number
here is therefore comparable with ``vq_gmm_tau`` (RASR search, pooled features),
``vq_gmm_tau_cuda`` (op, pooled features) and ``vq_gmm_tau_oracle_seg`` (op,
frames, oracle boundaries) -- which is the ladder this run adds its rung to.

The AM scale carries over from the oracle config for a reason worth stating: the
oracle recognizer *sums* a segment's ~4 frame scores into one observation, and
here those same ~4 frames are scored individually along a path. Either way a
phoneme accumulates the sum of its frames' acoustic scores against one LM
transition, so the AM/LM balance is the same quantity and 0.35/0.5 mean the same
thing in both. What is genuinely new is that the path may now *choose* how many
frames a phoneme gets, which is what the loop probability prices.
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

exp_dir = "vq_gmm_tau_unseg"
version = 1

# Carried over from vq_gmm_tau_oracle_seg unchanged.
LM_ORDER = 5
SIGMA = 0.02
BEAM_SIZE = 1_000
NUM_LABELS = 40
NUM_CODEWORDS = 512
SEED = 42
TAU = 4.0
#: 40, not 60. Nothing in this family has moved much past epoch 40:
#: vq_gmm_tau_cuda sat at 10.49 for epochs 40, 50 and 60 at ds-1.0, and the
#: oracle run's winning arm was converged by 30. Epochs are a chain of jobs
#: whose hashes do not depend on this count (see chunked_clustering's
#: num_epochs docstring), so lowering it keeps every epoch already computed and
#: simply stops the chain earlier -- and raising it again later resumes rather
#: than recomputes.
WEIGHTS_EPOCHS = 40
WEIGHTS_DECODE_EPOCHS = [5, 10, 20, 30, 40]

#: Decode-side sweep, identical to the oracle config so PER compares directly.
BASE_DECODE_SCALE = 1.0
DECODE_DISTANCE_SCALES = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
SCALE_SWEEP_EPOCHS = [30, 40]

#: One scale, not two. The oracle sweep was still undecided when this config
#: was written and both were carried as a hedge; it has since separated them
#: decisively, at ds-1.0 on cv:
#:
#:     epoch     am-0.25   am-0.35   am-0.5
#:        20       72.39     31.07    59.03
#:        30       62.83     10.79    26.98
#:        40       42.24         -    11.97
#:
#: 0.35 is converged by epoch 30 at 10.79 -- level with the rest of the ladder
#: (vq_gmm_tau 10.34, vq_gmm_tau_cuda 10.13) -- where 0.5 needs epoch 40 to
#: reach 11.97. Dropping 0.5 halves this sweep.
#:
#: Worth knowing if it is ever revisited: the *early* epochs invert this. At
#: ep-5 and ep-10 the oracle run had 0.5 ahead of 0.35, and the order only
#: settled from ep-20. So an early number from this config is not evidence for
#: reopening the choice; ep-20 is the first epoch that would be.
TRAIN_DISTANCE_SCALES = [0.35]

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


#: ``(num_states, [mean durations in frames])``. S=1 is swept one frame either
#: side of the anchor; S=2 takes the anchor alone, which is the budget the
#: comparison needs -- it asks whether the duration *shape* matters at a matched
#: mean, not where S=2's own optimum lies.
TOPOLOGIES = [
    (1, [3.0, MEAN_DURATION_FRAMES, 5.0]),
    (2, [MEAN_DURATION_FRAMES]),
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
NUM_CHUNKS = 6
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
#: Memory: peak is checkpointed alpha-hat, measured at 5.34 GB for B=16, T=1000,
#: S=1 (bench/README.md). The longest sequence here is 1094 frames and S=2
#: doubles NQ to 1.86M, so the worst arm peaks near 11 GB -- comfortable on
#: either card. ``mem`` is host RAM and unrelated.
EPOCH_RQMT = {
    "mem": 16,
    "gpu": 1,
    "cpu": 4,
    "time": 4,
    "sbatch_args": ["-p", "gpu_48gb,gpu_32gb"],
}

#: Sequences per op call. Unlike the oracle config, T is now a frame count
#: (mean 489, max 1094), not a segment count, so MAX_FRAMES rather than
#: BATCH_SIZE is what binds: 16 x 1094 = 17,504 is why it sits at 18,000. At the
#: oracle config's 4,000 the longest batches would collapse to B=3 and the GPU
#: would idle. Both unhashed - throughput only.
BATCH_SIZE = 16
MAX_FRAMES = 18_000


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
        columns=["states", "loop", "mean_dur", "am_scale", "epoch", "per",
                 "del", "ins", "sub", "log_likelihood", "log_likelihood_gain",
                 "posterior_entropy", "cluster_perplexity",
                 "max_cluster_fraction", "loop_frequency", "segment_duration",
                 "segment_duration_argmax", "boundary_rate", "dead_clusters", "l1"],
        sort_by=["states", "loop", "am_scale"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"The tau={TAU} arm on silence-free ls-100h frames with the "
            f"segmentation searched for rather than given, over HMM states per "
            f"phoneme and self-loop probability. $p$ is set from a target mean "
            f"duration by $p = 1 - S/d$; $d = 4$ frames ($80$ ms) is the MLE "
            f"anchor, matching both the literature prior and this corpus' "
            f"measured $3.99$. Decoded on mean-pooled cv, unchanged, so PER "
            f"compares directly with vq\\_gmm\\_tau and its oracle-segmented "
            f"counterpart."
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
        columns=["states", "loop", "mean_dur", "am_scale", "epoch",
                 "log_likelihood", "log_likelihood_gain", "posterior_entropy",
                 "cluster_perplexity", "max_cluster_fraction",
                 "loop_frequency", "segment_duration",
                 "segment_duration_argmax", "boundary_rate", "dead_clusters",
                 "l1"],
        sort_by=["states", "loop", "am_scale"],
        drop_empty_rows=True,
        title=(
            f"Per-epoch convergence for the tau={TAU} unsegmented arms. One row "
            f"per recognition pass, so it fills in as the run goes rather than "
            f"only at the decoded epochs. Duration columns: the soft estimate is "
            f"an expectation under a frame-independence assumption (an upper "
            f"bound on the boundary rate), the argmax one is the run length of "
            f"the argmax label sequence (a lower bound); the GMM alignment's own "
            f"mean is 3.99 frames at a boundary rate of 0.251. Both are blank for "
            f"passes that ran before the duration counters existed."
        ),
    )

    for num_states, durations in TOPOLOGIES:
        for mean_dur in durations:
            p = loop_probability(mean_dur, num_states)
            for am_scale in TRAIN_DISTANCE_SCALES:
                name = (f"weights_tau-{TAU}_s-{num_states}"
                        f"_loop-{round(p, 4)}_am-{am_scale}")
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
                    transition_scale=1.0,
                    num_states=num_states,
                    apply_sentence_end=True,
                    batch_size=BATCH_SIZE,
                    max_frames=MAX_FRAMES,
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
                    "mean_dur": mean_dur,
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
                                    "mean_dur": mean_dur,
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
