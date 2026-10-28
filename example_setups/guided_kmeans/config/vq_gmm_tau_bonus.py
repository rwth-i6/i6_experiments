"""A cheaper boundary with the LM left at full strength, and a frame-level codebook.

Three experiments from ``docs/free_segmentation/findings_and_directions.md``
(§6.1, §6.4, §6.6), sharing one control. Everything not named below is
``vq_gmm_tau_lm``'s S=2 arm unchanged: silence-free ls-100h frames with free
segmentation, S=2, LM scale 1.0, am=0.35, tau=4, frozen k512 codebook,
weights-only training.

**1. Exactly uniform initialisation (the control).** Every free-segmentation run
so far started from ``NormalTableJob(sigma=0.02)``. In the segmented runs the
more uniform the table the faster training converged, exactly uniform best and
sigma=0.1 breaking training; measured on the first EM step here, sigma=0.02 adds
only ~8% to I(label; codeword) over sigma=0 (doc §3.8). So every arm in this
config starts from ``sigma=0.0``, and its zero-bonus arm is the new control - the
LM sweep's 1.0 arm with only the initialisation changed. Every other arm is read
against it, not against the sigma=0.02 runs.

**2. A label-independent boundary bonus.** In the LM sweep the LM was both the
teacher of label identity and the only price of a boundary, coupled through one
scale: lowering it cheapened boundaries but stopped the learning (doc §3.6, §4).
Here the LM stays at 1.0 and the *topology* pays part of each boundary back.

The topology reaches the posterior only through ``lambda = ts * logit(p)``, a
per-state, per-phoneme penalty (``vq_unsupervised_frames`` derived this and
checked it against RASR; ``latex_report.loop_probability``): over a fixed-length
utterance the loop term ``ts * T * log p`` is the same for every segmentation and
cancels. Re-checked for the CUDA op at S=2 with the reference implementation:
``(p, ts)`` pairs with equal lambda give posteriors equal to 1e-15, unequal
lambdas differ by up to 0.12. So **the transition scale is not a separate knob
here**: a sweep over it at fixed p is a sweep over lambda. The arms therefore
sweep lambda directly, at the transition scale of 0.35 that HMM full-sum
experience suggests, and the tables also quote the p that gives the same lambda
at ts=1.

The bonus per phoneme is ``-S * lambda``. It is set as a fraction of the LM's
measured cost for one more phoneme, 1.888 nats on typical histories
(``PhonemeLMCostJob``), so the net boundary price is ``(1 - fraction) * 1.888``:

    fraction   bonus   net price   lambda   p @ ts=0.35   p @ ts=1   LM-sweep arm at the same price
     -0.25    -0.472     2.360     +0.236      0.662        0.559     (lm 1.25, not run)
      0        0.000     1.888      0.000      0.500        0.500     lm 1.0 (the control)
      0.25     0.472     1.416     -0.236      0.338        0.441     (lm 0.75, not run)
      0.5      0.944     0.944     -0.472      0.206        0.384     lm 0.5
      0.75     1.416     0.472     -0.708      0.117        0.330     lm 0.25

The 0.5 and 0.75 arms charge what the LM sweep's 0.5 and 0.25 arms charged, so
each pair differs in one thing: whether the LM still teaches at full strength.
*Prediction if doc §4 is right:* segment gap grows like the control's while
payback and the payable share rise above its plateau (0.95 / 35% for the
sigma=0.02 run). *Risk:* the bonus could fragment segments while the acoustics
are still uninformative - S=2 forbids single-frame phonemes, which bounds it, and
``segment_duration_argmax`` / ``boundary_rate`` show it from epoch 1.

**A negative bonus** is a surcharge: the topology then adds to the LM's price
instead of paying part of it back (lambda > 0, p > 1/2 - the direction the
unsegmented loop sweep ran in, where PER rose with the price). Added on
2026-10-12 because the first epochs ranked the arms by *smaller* bonus, and
whether the trend continues past zero says whether the control's price is
already too low rather than too high.

**Run on past 30 epochs** (2026-10-14). That early ranking reversed between
epochs 15 and 20. At 30 the 0.5 arm decodes at 72.2% PER and the 0.75 arm at
76.7% (0.25: 75.8% at 25), all still falling, against the control's 79.8%, which
had stopped moving by 25. The positive-bonus arms run on to ``EXTENDED_EPOCHS``;
the others stay at 30. At 60 (2026-10-17) the 0.25 arm had flattened at 73.7%,
while the 0.5 arm, at 65.8%, was still losing 0.8 points every five epochs; it
runs on to 100 (``LONG_ARMS``), and from 2026-10-20 to 200 (62.0% at 95).

**3. A frame-level codebook.** The k512 codebook is a colleague's FAISS k-means
over mean-pooled GMM *segment* representations: it carries oracle segmentation
into the densities, describes segment means rather than the frames scored here,
and was fit under plain L2 while this model scores under the shared covariance
``tau * Sigma`` - i.e. in the globally whitened space. The replacement is fit on
the same silence-free ls-100h frames the runs train on, by unguided k-means under
the same global covariance (``unguided_flavor``, as ``codebook_mixture``'s stage
1 but on frames instead of pooled segments), so metric, units and data all match
the model. Its guided arm is the control with only the codebook swapped. If the
segment codebook's "phoneme-shaped" cells are what carry the free runs, this arm
does worse; if the mismatch was holding them back, better.

A second frame-codebook arm at bonus 0.5 (added 2026-10-14, when the zero-bonus
frame arm led the segment control by 2.4 PER points at epoch 20 and 0.5 was the
best bonus) asks whether the two effects add up. It is also the tied arm of
``vq_gmm_tau_untied``.
"""

import math

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    GMM_ALIGNMENT_CV,
    COLLEAGUE_CENTROIDS_K512,
    INPUT_DATA as input_data,
)
from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
    SegmentedFeaturesFromAlignmentJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    DuplicateCovsJob,
    GlobalCovarianceJob,
    NormalTableJob,
    RandomCentroidsJob,
    ScaleCovsJob,
    chunked_clustering,
    mixture_flavor,
    unguided_flavor,
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
    loop_probability as loop_probability_column,
    param,
    relabel,
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
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_lm import (
    frame_level_ls100,
)

exp_dir = "vq_gmm_tau_bonus"
version = 1

# Carried over from vq_gmm_tau_lm unchanged.
LM_ORDER = 5
BEAM_SIZE = 1_000
NUM_LABELS = 40
NUM_CODEWORDS = 512
SEED = 42
TAU = 4.0
AM_SCALE = 0.35
LM_SCALE = 1.0
NUM_STATES = 2

#: Exactly uniform: NormalTableJob at sigma=0 writes 1/512 everywhere. The same
#: job vq_unigram_init's forward-backward cross-check builds.
INIT_SIGMA = 0.0

#: Only lambda = ts * logit(p) reaches the posterior (module docstring), so this
#: picks the units the loop probabilities are written in, not an experimental
#: setting: the 0.35 that HMM full-sum practice suggests.
TRANSITION_SCALE = 0.35

#: The LM's cost of one more phoneme on typical histories, in nats at LM scale
#: 1.0: PhonemeLMCostJob.BvnFwAIhWYXl's out_cost (5-gram, 8,000 positions).
#: Hard-coded rather than read from the job because it sets hashed loop
#: probabilities, which have to be known when the graph is built.
LM_PHONEME_COST = 1.888

#: Share of LM_PHONEME_COST the topology pays back per phoneme; negative is a
#: surcharge on top of it. 0.5 and 0.75 match the boundary prices of
#: vq_gmm_tau_lm's 0.5 and 0.25 arms.
BONUS_FRACTIONS = [-0.25, 0.0, 0.25, 0.5, 0.75]

#: Thirty. The sigma=0.02 control's payback had plateaued by ~25, and in the LM
#: sweep nothing moved after that through epoch 40. Epoch chains do not hash
#: this count: extending resumes rather than recomputes.
WEIGHTS_EPOCHS = 30
WEIGHTS_DECODE_EPOCHS = [5, 10, 15, 20, 25, 30]
BALANCE_EPOCHS = [1, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]

#: The bonus arms that had not converged by epoch 30 run on to 60, resuming
#: from 30. Between epochs 25 and 30 the 0.5 and 0.75 arms still gained 2.6 and
#: 2.0 PER points (0.25: 2.1 between 20 and 25), while the control gained 0.1
#: and the frame-codebook and -0.25 arms were flattening out by epoch 20.
#: Decoded and balanced every five epochs from there. Applies to both codebooks:
#: the 0.5 frame-codebook arm, added later, runs to 60 from the start.
EXTENDED_FRACTIONS = [0.25, 0.5, 0.75]
EXTENDED_EPOCHS = 60
EXTENDED_DECODE_EPOCHS = [35, 40, 45, 50, 55, 60]
EXTENDED_BALANCE_EPOCHS = [35, 40, 45, 50, 55, 60]

#: Arms run on past EXTENDED_EPOCHS: ``{(fraction, codebook): (epochs, chunks)}``.
#: The 0.5 segment arm lost 0.8 PER points in each five epochs from 45 to 60.
#: Twice the chunks because an epoch lasts as long as its GTX 1080 chunks: at
#: epoch 60 those took 110 min for a 16th of the corpus, the L40S ones 35, all
#: starting at once. gpu_11gb's per-user cap of 70 cards leaves room for 32 next
#: to the two frame-codebook arms' 16 each. Unhashed, and pinned per epoch
#: (``GuidedClusteringEpochJob._pin_num_chunks``): epochs 1-60 keep their 16.
#: 64 from epoch 93 or 94 (2026-10-20), to finish by the next day with the other
#: arms paused: a 1080 chunk then takes ~31 min instead of ~60, and 64 is still
#: under gpu_11gb's cap of 70 should every chunk land there.
LONG_ARMS = {(0.5, "segments"): (200, 64)}
#: Every five epochs up to the longest arm's end. A decode for an epoch that has not
#: been reached yet just waits in the graph.
LONG_DECODE_EPOCHS = list(range(65, 201, 5))
LONG_BALANCE_EPOCHS = list(range(65, 201, 5))

#: Every tenth epoch, for the paper table only: the long arm alone is decoded at
#: 40 epochs, a page of table for a curve that is read off every second point
#: just as well. Derived from the three decode-epoch lists rather than written
#: out, so it cannot name an epoch that is never decoded, and it follows when
#: they are extended. The .txt report keeps every decoded epoch.
LATEX_EPOCHS = sorted(
    e
    for e in set(WEIGHTS_DECODE_EPOCHS + EXTENDED_DECODE_EPOCHS + LONG_DECODE_EPOCHS)
    if e % 10 == 0
)

#: Unguided k-means for the frame-level codebook. Batch Lloyd's over 13.8M
#: frames; codebook_mixture's segment-level runs had flattened well within 10.
#: No search, so a chunk is CPU-only and I/O-bound.
CODEBOOK_EPOCHS = 10
CODEBOOK_CHUNKS = 20

#: Free-segmentation decode, as in vq_gmm_tau_lm: loop 1/2 (zero topology price
#: in RASR's one-state topology), LM 1.0 for every arm, a fixed seeded 10% of cv.
#: The same subset and settings, so these columns compare with that sweep's.
MATCHED_DECODE_LM = 1.0
FREE_SEG_LOOP = 0.5
FREE_SEG_FRACTION = 0.1

BASE_DECODE_SCALE = 1.0

#: Unhashed operational settings, as vq_gmm_tau_lm (see the notes there).
NUM_CHUNKS = 16
NUM_WORKERS = 0
PARENT_THREADS = 4
EPOCH_RQMT = {
    "mem": 12,
    "gpu": 1,
    "cpu": 4,
    "time": 8,
    "sbatch_args": ["-p", "gpu_48gb,gpu_24gb,gpu_11gb,gpu_32gb"],
}
#: S=2 at B=8 fits an 11 GB card (5.04 GB at T=1000).
BATCH_SIZE, MAX_FRAMES = 8, 9_000
#: RASR search processes per decode; the job asks for one more CPU and 4 GB per
#: CPU. Unhashed. The default 7 kept all 8 CPUs busy for ~2h05 on the epoch-100
#: cv decode, while the parent's main thread used ~15% of a core (2026-10-20), so
#: the search scales with workers. A decode uses about 3 GB plus 0.6 GB per worker.
DECODE_WORKERS = 23


def bonus_lambda(fraction: float) -> float:
    """Per-state lambda giving a per-phoneme bonus of ``fraction`` of the LM cost.

    The topology charges ``S * lambda`` per phoneme, so a bonus is a negative
    lambda and a negative fraction (a surcharge) a positive one. Rounded so that the hashed loop probability derived from it is
    reproducible to the bit.
    """
    # + 0.0 turns the zero arm's -0.0 into 0.0.
    return round(-fraction * LM_PHONEME_COST / NUM_STATES, 3) + 0.0


def loop_for_lambda(lam: float, transition_scale: float) -> float:
    """``p = sigmoid(lambda / ts)``: the loop probability giving ``lambda`` at ``ts``."""
    return 1.0 / (1.0 + math.exp(-lam / transition_scale))


#: ``(bonus fraction, codebook)``. The frame-level codebook runs at zero bonus,
#: the control with one thing changed, and at 0.5, the best bonus with one
#: thing changed.
ARMS = [(fraction, "segments") for fraction in BONUS_FRACTIONS] + [
    (0.0, "frames"), (0.5, "frames")]


def frame_codebook(features, base_covs):
    """§6.6: unguided k-means on the training frames, under the model's covariance.

    The ``chunked_clustering`` run; its ``out_centroids[CODEBOOK_EPOCHS]`` is the
    codebook. A function so that ``vq_gmm_tau_untied`` builds the very same jobs
    (the alias stays this config's, wherever it is called from).

    :param features: the silence-free ls-100h frames, ``frame_level_ls100()``
    :param base_covs: the global covariance duplicated over the codebook,
        unscaled - the k-means metric, not the model's tau-scaled densities
    """
    return chunked_clustering(
        num_epochs=CODEBOOK_EPOCHS,
        features_hdf=features,
        num_clusters=NUM_CODEWORDS,
        flavor=unguided_flavor(
            centroids=RandomCentroidsJob(features, NUM_CODEWORDS, seed=SEED).out_centroids,
            covs=base_covs,
            num_clusters=NUM_CODEWORDS,
        ),
        num_chunks=CODEBOOK_CHUNKS,
        alias_prefix=f"guided_kmeans/{exp_dir}/codebook-frames-{NUM_CODEWORDS}",
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
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    ls100 = frame_level_ls100()
    ls100.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_frames_nosil")
    cv = silence_free_cv_features()
    cv.add_alias(f"guided_kmeans/{exp_dir}/features_cv_nosil")

    # The family's covariance. It is also the metric the frame-level codebook is
    # fit under, so that codebook partitions the space the way the model scores it.
    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    base_covs = DuplicateCovsJob(global_cov, NUM_CODEWORDS).out_covs
    covs = ScaleCovsJob(base_covs, TAU).out_covs
    init_table = NormalTableJob(
        NUM_LABELS, NUM_CODEWORDS, sigma=INIT_SIGMA, seed=SEED
    ).out_table

    # --- §6.6: the frame-level codebook ------------------------------------------
    codebook_run = frame_codebook(ls100.out_features, base_covs)
    frame_centroids = codebook_run.out_centroids[CODEBOOK_EPOCHS]
    tk.register_output(
        f"guided_kmeans/{exp_dir}/codebooks/frames-{NUM_CODEWORDS}_centroids.npy",
        frame_centroids,
    )
    tk.register_output(
        f"guided_kmeans/{exp_dir}/codebooks/frames-{NUM_CODEWORDS}_statistics.json",
        codebook_run.out_statistics,
    )
    codebooks = {"segments": COLLEAGUE_CENTROIDS_K512, "frames": frame_centroids}

    cv_dataset = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )
    # Inert for training (an explicit flavor bypasses its only reader).
    train_config = _recog_config(LM_SCALE, 0.0, 1.0, use_fb=True)
    decode_config_rasr = _recog_config(1.0, 0.0, None, use_fb=False)

    # Emission balance, per codebook: BalanceFramesJob scores frames against the
    # densities, so the two codebooks need one each. The segment one is the job
    # vq_gmm_tau_lm built, shared.
    balance_frames = {
        key: BalanceFramesJob(
            features_hdf=ls100.out_features,
            frame_labels=ls100.out_labels,
            centroids=centroids,
            covs=covs,
        ).out_frames
        for key, centroids in codebooks.items()
    }
    lm_cost = PhonemeLMCostJob(
        arpa_path=phonetic_lm_dict[LM_ORDER],
        lexicon=fb_lexicon,
        backoff_fb_root=tools.BACKOFF_FB,
    ).out_cost

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

    def matched_values(centroids, model_dir, decode_name):
        dc = DecodeConfig(
            centroids=centroids,
            model_dir=model_dir,
            recog_rasr_config=_recog_config(MATCHED_DECODE_LM, FREE_SEG_LOOP, 1.0, use_fb=False),
            distance_scale=AM_SCALE,
            num_workers=DECODE_WORKERS,
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
            "matched_hyp_ref": 1 - res.deletion + res.insertion,
        }

    arm_columns = [
        param("codebook", header=("Code-", "book")),
        param("bonus", header=("Bonus", "frac."), fmt="{:g}"),
        loop_probability_column("lambda", TRANSITION_SCALE),
        loop_probability_column("lambda", 1.0),
    ]
    matched_columns = ["matched_per", "matched_del", "matched_ins",
                       "matched_sub", "matched_hyp_ref"]
    balance_columns = ["segment_gap", "boundary_payback", "boundary_price",
                       "payable_boundaries"]

    # The full table: every arm, every column. The .txt report is this one, and
    # the .tex below is a narrowed view of it - they share the rows but not the
    # column selection, so trimming the paper table takes nothing away here.
    report = LatexTableReport(
        columns=arm_columns + ["epoch", "per", "del", "ins", "sub"]
                + matched_columns
                + ["posterior_entropy", "segment_duration_argmax",
                   "boundary_rate"] + balance_columns,
        sort_by=["codebook", "bonus"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Label-independent boundary bonus and frame-level codebook, free "
            f"segmentation on silence-free ls-100h frames, S={NUM_STATES}, LM "
            f"scale {LM_SCALE}, am={AM_SCALE}, tau={TAU}, exactly uniform initial "
            f"table. The bonus pays back the given fraction of the LM's "
            f"{LM_PHONEME_COST}-nat cost of one more phoneme through the loop "
            f"probability, quoted at ts={TRANSITION_SCALE:g} and at ts=1 (equal "
            f"lambda, identical runs). Codebook 'segments' is the frozen k512 "
            f"used so far, 'frames' unguided k-means on the training frames under "
            f"the model's covariance. Decoded on mean-pooled cv at LM scale 1.0; "
            f"the free-segmentation columns decode cv frames with no boundaries "
            f"given (loop {FREE_SEG_LOOP}, LM scale {MATCHED_DECODE_LM}, a fixed "
            f"{int(100 * FREE_SEG_FRACTION)}\\% of cv)."
        ),
    )
    #: The paper table: the segment-codebook arms only - lambda alone identifies
    #: an arm there - and the free-segmentation columns they are read by. A view,
    #: so the rows below are added once.
    latex_report = report.view(
        # Every row here is free segmentation and the caption says so, so the
        # library's "Free-seg." qualifier comes off the headers; relabel keeps
        # each column's value source, format and x100 scale.
        columns=[
            loop_probability_column("lambda", TRANSITION_SCALE),
            "epoch",
            relabel("matched_per", r"PER [\%]"),
            relabel("matched_del", r"Del [\%]"),
            relabel("matched_ins", r"Ins [\%]"),
            relabel("matched_sub", r"Sub [\%]"),
            relabel("matched_hyp_ref", "hyp/ref"),
            "log_likelihood",
            "posterior_entropy", "segment_duration_argmax"
        ],
        sort_by=["lambda"],
        epochs=LATEX_EPOCHS,
        row_filter=lambda row: row.params.get("codebook") == "segments",
        caption=(
            f"Free segmentation on silence-free ls-100h frames, HMM-substates={NUM_STATES}, LM "
            f"scale {LM_SCALE}, am={AM_SCALE}, exactly uniform initial table."
        ),
    )
    recog_results = []

    progress_report = TextTableReport(
        columns=arm_columns + ["epoch", "log_likelihood_gain", "posterior_entropy",
                               "cluster_perplexity", "segment_duration_argmax",
                               "boundary_rate", "dead_clusters"] + balance_columns,
        sort_by=["codebook", "bonus"],
        drop_empty_rows=True,
        title=(
            f"Per-epoch convergence: boundary bonus and frame-level codebook (S=2, "
            f"LM {LM_SCALE}, am={AM_SCALE}, uniform init). Read boundary payback "
            f"against boundary price (which the bonus lowers): the sigma=0.02 "
            f"control plateaued at a payback of ~0.95 with 35% of true boundaries "
            f"payable, the oracle-segmented run reaches 2.39 with 58%. Watch "
            f"segment duration and boundary rate on the bonus arms for "
            f"fragmentation (reference 3.99 frames, 0.251). Balance columns are "
            f"measured on the model each pass produced and are blank at epochs "
            f"not in BALANCE_EPOCHS."
        ),
    )

    for fraction, codebook in ARMS:
        lam = bonus_lambda(fraction)
        extended = fraction in EXTENDED_FRACTIONS
        num_epochs = EXTENDED_EPOCHS if extended else WEIGHTS_EPOCHS
        num_chunks = NUM_CHUNKS
        if (fraction, codebook) in LONG_ARMS:
            num_epochs, num_chunks = LONG_ARMS[(fraction, codebook)]
        p = loop_for_lambda(lam, TRANSITION_SCALE)
        centroids = codebooks[codebook]
        name = (f"weights_tau-{TAU}_s-{NUM_STATES}_bonus-{fraction}_loop-{round(p, 4)}"
                f"_ts-{TRANSITION_SCALE}_lm-{LM_SCALE}_am-{AM_SCALE}"
                f"_init-uniform_cb-{codebook}")
        backoff_fb = BackoffFBSearch(
            backoff_fb_root=tools.BACKOFF_FB,
            arpa_path=phonetic_lm_dict[LM_ORDER],
            lexicon=fb_lexicon,
            lm_scale=LM_SCALE,
            emission_scale=1.0,
            loop_probability=p,
            silence_loop_probability=None,
            transition_scale=TRANSITION_SCALE,
            num_states=NUM_STATES,
            apply_sentence_end=True,
            batch_size=BATCH_SIZE,
            max_frames=MAX_FRAMES,
            checkpoint_interval=-1,
            storage_dtype="float32",
        )
        flavor = mixture_flavor(
            centroids=centroids,
            covs=covs,
            mixtures=init_table,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=NUM_LABELS,
            distance_scale=AM_SCALE,
            use_forward_backward=True,
            update_densities=False,
            update_covariances=None,
            mixture_floor=1e-2,
            num_workers=NUM_WORKERS,
            backoff_fb=backoff_fb,
        )
        result = chunked_clustering(
            num_epochs=num_epochs,
            features_hdf=ls100.out_features,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=NUM_LABELS,
            flavor=flavor,
            distance_scale=AM_SCALE,
            use_forward_backward=True,
            rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
            num_chunks=num_chunks,
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

        # Negative for a bonus: the balance job's price is LM cost plus this.
        penalty = transition_boundary_penalty(NUM_STATES, p, TRANSITION_SCALE)
        balance = {}
        for e in BALANCE_EPOCHS + EXTENDED_BALANCE_EPOCHS + LONG_BALANCE_EPOCHS:
            if e not in result.out_artifacts["mixtures"]:
                continue
            job = EmissionBalanceJob(
                frames=balance_frames[codebook],
                mixtures=result.out_artifacts["mixtures"][e],
                am_scale=AM_SCALE,
                lm_cost=lm_cost,
                lm_scale=LM_SCALE,
                transition_penalty=penalty,
            )
            job.add_alias(f"guided_kmeans/{exp_dir}/{name}/balance/epoch_{e:03d}")
            balance[e] = {
                "segment_gap": job.out_segment_gap,
                "boundary_payback": job.out_payback,
                "boundary_price": job.out_price,
                "payable_boundaries": job.out_payable,
            }

        arm_params = {"codebook": codebook, "bonus": fraction, "lambda": lam}
        for epoch_key in sorted(int(k) for k in result.out_epoch_statistics):
            progress_report.add_row(
                params=dict(arm_params), epoch=epoch_key,
                statistics=progress_stats,
                values=dict(balance.get(epoch_key, {})),
            )

        for e in WEIGHTS_DECODE_EPOCHS + EXTENDED_DECODE_EPOCHS + LONG_DECODE_EPOCHS:
            if e not in result.out_models:
                continue
            dc = DecodeConfig(
                centroids=centroids,
                model_dir=result.out_models[e],
                recog_rasr_config=decode_config_rasr,
                distance_scale=BASE_DECODE_SCALE,
                write_frame_labels=True,
                num_workers=DECODE_WORKERS,
            )
            decode_name = f"{name}_ep-{e}_ds-{BASE_DECODE_SCALE}"
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
                    f"guided_kmeans/{exp_dir}/eval/{decode_name}_fer", res.fer,
                )
            tk.register_output(
                f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per,
            )
            recog_results.append(res)
            values = dict(balance.get(e, {}))
            values.update(matched_values(
                centroids, result.out_models[e],
                f"{name}_ep-{e}_freeseg-lm-{MATCHED_DECODE_LM}"))
            if res.fer is not None:
                values["fer"] = res.fer
            report.add_row(
                result=res, params=dict(arm_params), epoch=e,
                statistics=stats, values=values,
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
    text_report = report.as_text()
    text_report.register(
        f"guided_kmeans/{exp_dir}/report_{version}.txt",
        required=text_report, register_dependencies=False, update_frequency=60,
    )
    progress_report.register(
        f"guided_kmeans/{exp_dir}/convergence_{version}.txt",
        required=progress_report, update_frequency=60,
    )


def py():
    run()
