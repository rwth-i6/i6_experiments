"""Unsegmented VQ training: learn the segmentation together with the labels.

The segmented VQ runs (:mod:`.vq_unsupervised_long`) train on one mean-pooled
vector per phoneme segment, where the segments come from a GMM alignment. That
is two kinds of cheating at once: the alignment removes silence, and it tells the
search where every phoneme starts and ends. This config keeps the first and drops
the second. Silence frames are still removed with the GMM alignment, but every
remaining frame is kept as-is, so the search has to find the phoneme boundaries
itself - which is what a loop in the transition model is for.

Everything else follows the segmented runs deliberately, so that the only new
thing is the segmentation: the same 512-entry codebook (itself trained on the
cheating segmentation - frame-level codebooks at 1024+ are a later step), the
same discrete table model, forward-backward training, sigma=0.02 initialization,
and the 5-gram count LM at beam 1000.

**What the transition model actually controls.** Per frame the transition scorer
costs ``ts * (-log p)`` to stay on a label and ``ts * (-log(1-p))`` to advance
to a new one. Summed over an utterance of ``T`` frames segmented into ``K``
phonemes that is::

    T * ts * (-log p)   +   K * ts * logit(p)

The first term is the same for every segmentation of the utterance and cancels -
in Viterbi, in forward-backward, and in beam pruning, since all hypotheses alive
at frame ``t`` share ``t * ts * (-log p)``. So loop probability and transition
scale reach the search **only through their product lambda = ts * logit(p)**, a
per-phoneme insertion penalty. Verified on the real RASR search rather than
assumed, on a 499-frame silence-free cv utterance at acoustic scale 1.0:

    p      ts      lambda   phonemes   output
    0.9    1.0     2.197       32      45f6999156
    0.75   2.0     2.197       32      45f6999156   (bit-identical)
    0.6    5.42    2.197       32      c3e26aa6b8   (same labels; one boundary
                                                     off by one frame)
    0.75   1.0     1.099       50
    0.9    3.0     6.592        8

The one difference among equal-lambda settings is a single tie broken the other
way, because ts=5.42 inflates the accumulated scores. Hence ``ts`` is held at 1
here - the smallest magnitudes, the fewest such ties - and ``p = sigmoid(lambda)``,
which is also exactly what ``create_recog_rasr_config(loop_log_odds=...)`` does. A
grid over loop x transition scale would spend most of its points re-running the
same lambda.

(The reduction needs the blank transitions to mirror the label ones, i.e.
``silence_loop_probability == loop_probability``; the shared builder does that.)

**What was missing from the sweep: the acoustic scale.** Only two ratios matter
among the three costs - acoustic per frame (``distance_scale``), penalty per
phoneme (lambda), LM per phoneme (``lm_scale``) - so with the LM as the unit the
space is (lambda, distance_scale). The segmented optimum at scale 1.0 does not
carry over: there each phoneme contributed one acoustic score, here it
contributes ~4, correlated, and the table is dense (a phoneme's frames spread
over many codewords), so how much acoustic evidence a boundary gets is not
predictable from the segmented setting. ``PROBE_*`` below maps that plane.

How correlated, measured in codeword space on 300 silence-free cv utterances
(35.9k phoneme segments, mean 4.03 frames): consecutive frames inside a phoneme
share their codeword 56% of the time (30% across a boundary), and a segment holds
2.3 codeword runs. A repeated codeword repeats its evidence exactly, so a segment
carries between one and ~2.3 frames' worth: a calibrated acoustic scale lies
between 1/4.03 = 0.25 (the 1/duration rule) and 2.3/4.03 = 0.58.

**Probe first, train second.** Training here costs ~4x a segmented epoch (four
times the frames), so a wrong regime wastes days. The probe decodes held-out
silence-free cv with a *supervised* frame-level table over the (lambda, scale)
plane - decode-only, no training - and shows where segmentation works at all
with a good table. It is an oracle instrument used only to place the training
grid and the decode points; its numbers are a ceiling, not a result.

**The probe is Viterbi, training is full-sum.** Viterbi keeps the best path, so
every phoneme pays lambda plus its LM cost and the scaled acoustics must pay that
back; at low scale they cannot, and the probe's best lambda falls with the scale.
Forward-backward sums over labels and boundary positions instead: at ts=1 and LM
scale 1 the loop probability and the LM form an (almost) normalized prior over
segmentations, with geometric durations of mean 1/(1-p). Lambda then stays a
duration prior - 1.1 is p=0.75, the measured 4 frames - and the acoustic scale is
the posterior temperature. The two readings agree where posteriors are sharp
(scale >= 1) and part at the calibrated scales above, which the Viterbi probe
therefore cannot rule out. ``TRAIN_SETTINGS`` covers both.

**Decoding.** Every trained table is decoded at the same probe-chosen points
(``DECODE_POINT``, ``RIDGE_POINTS``), not at its own training setting: a model
trained at (1.1, 0.25) and decoded there sits in Viterbi's deletion regime (62.7%
PER with the oracle table) whatever its table is worth. ``mi`` and dead clusters
do not depend on decoding at all.

**Cost, and why an epoch is four updates.** Measured on the first epoch of the
runs below: a chunk task of 1284 utterances took ~11 h on a quiet node and was
still running after 19 h on a node carrying nine of them, i.e. ~24-30 h per
epoch, against 0.65 h per chunk task in the segmented runs. That is 17x for 3.9x
the vectors, because the search now has to place a boundary at every frame; the
nodes run ``ThreadsPerCore=2``, so the 1100-CPU quota is ~550 physical cores.

Convergence, though, arrives late rather than gradually - the segmented tau runs
sit at 80.6% PER at epoch 10, 73.7 at 20, 43.4 at 30 and 14.7 at 40 - so cutting
epochs would cut exactly the part that matters. The cost comes out of the epoch
instead: ``BATCHES_PER_EPOCH`` re-estimates the table four times per pass, each
from a quarter of the corpus, which is the same compute per pass and four times
the updates. ``EMA`` damps a quarter-update against the running statistics at
``alpha=0.75``, putting the averaging window (``batch / (1 - alpha)``) at one
full corpus: as smooth as the whole-epoch update it replaces, four times as
often. ``NUM_EPOCHS`` and ``DECODE_EPOCHS`` count passes, so 25 passes are 100
updates.

**Reading the results.** Segmentation errors show up directly in the PER split:
under-segmentation as deletions, over-segmentation as insertions. The frame
error rate is scored against the *silence-free* frame alignment written by the
features job, so hypothesis and reference correspond frame for frame -
``FrameErrorRateJob`` compares by position and never checks lengths, which is
why the FER in the segmented reports (segment labels against a frame alignment)
is not meaningful.
"""

import math

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
    GMM_ALIGNMENT_LS960_FRAME,
    COLLEAGUE_CENTROIDS_K512,
)
from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
    SegmentedFeaturesFromAlignmentJob,
    SupervisedVQTableJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
)
from i6_experiments.example_setups.guided_kmeans.setup.statistics_jobs import (
    MixtureDiagnosticsJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    EMAConfig,
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
    loop_probability,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    build_decode_config,
    build_vq_training,
)

exp_dir = "vq_unsupervised_frames"
version = 1

# --- carried over from the segmented runs ------------------------------------
LM_ORDER = 5
BEAM_SIZE = 1_000
SIGMA = 0.02
SEED = 42
LM_SCALE = 1.0
TRANSITION_SCALE = 1.0   # held fixed: only lambda = ts * logit(p) reaches the search

# --- scheduling --------------------------------------------------------------
NUM_EPOCHS = 25          # passes over the corpus; 4 table updates each
#: model re-estimations per pass, each from 1/B of the corpus
BATCHES_PER_EPOCH = 4
#: alpha puts the averaging window (batch / (1 - alpha)) at one full corpus
EMA = EMAConfig(alpha=0.75, mode="statistics")
NUM_CHUNKS = 22          # per *batch*: ~320 utterances a task, ~3 h; 220 CPU a run
NUM_WORKERS = 9
EPOCH_RQMT = {"mem": 12}  # measured tree RSS 8.6 GB, over the 8 GB first asked for

# --- the (lambda, acoustic scale) plane --------------------------------------
# Probe PER [%], report_1 (Viterbi, oracle frame table, held-out cv):
#
#     scale   lam=-1.1    0.0    1.1    2.2    3.3
#     0.25       37.3    24.0   62.7   93.7   98.8
#     0.5        21.6    14.0   17.4   28.1   48.9
#     1.0        19.4    14.0   13.2*  14.7   17.9
#     2.0        24.3    19.7   17.4   16.3   16.4
#     4.0        35.5    31.8   28.8   26.6   24.9
#
# The best lambda rises with the scale. At 0.25 the errors flip between two grid
# points, from 23.7% insertions (lambda=-1.1) to 13.3% deletions (0.0), so its
# balance point (~-0.5) was not probed - hence -0.5.
PROBE_LAMBDAS = [-1.1, -0.5, 0.0, 1.1, 2.2, 3.3]
PROBE_DISTANCE_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
RUN_TRANSITION_PROBE = True

#: (lambda, distance_scale), five runs so they fit the quota together. Lambda=1.1
#: over the calibrated scales and the probe optimum follows the full-sum reading;
#: the second run at 0.25 is the Viterbi reading where the two differ most, and
#: (2.2, 2.0) is the sharp-posterior end, where they agree.
TRAIN_SETTINGS = [
    (1.1, 0.25),   # duration prior p=0.75, 1/duration scale
    (1.1, 0.5),    # upper end of the calibrated range
    (1.1, 1.0),    # probe optimum
    (-0.5, 0.25),  # Viterbi balance point at 0.25
    (2.2, 2.0),    # sharp posteriors, on the probe ridge
]

#: The rest of the diagonal band: at every scale from 0.25 to 2.0, the lambdas
#: from 1.1 to the probe's best, plus one grid step outside. Left out of the full
#: 6x5 grid: scale 4.0 (7-16x the calibrated scale, so full-sum is Viterbi there
#: and the probe's best is 24.9% at the grid edge); lambda 2.2+ at 0.25, 3.3 at
#: 0.5 and -1.1 below scale 1 - duration priors of 9-28 or 1.3 frames against the
#: weakest evidence; and at scales 1-2, where the probe is a good guide, the
#: lambdas it ranks below their neighbours.
RUN_WAVE_2 = False
TRAIN_SETTINGS_WAVE_2 = [
    (0.0, 0.25),
    (-0.5, 0.5), (0.0, 0.5), (2.2, 0.5),
    (0.0, 1.0), (2.2, 1.0),
    (0.0, 2.0), (1.1, 2.0), (3.3, 2.0),
]

#: passes, i.e. 0, 4, 8, 20, 32, 48, 76 and 100 table updates
DECODE_EPOCHS = [0, 1, 2, 5, 8, 12, 19, 25]
DECODE_BEAM_SIZE = 500
#: (lambda, scale) at which every trained table is decoded, whatever it was
#: trained at: the probe optimum at every DECODE_EPOCH, and its ridge
#: neighbours at RIDGE_EPOCHS.
DECODE_POINT = (1.1, 1.0)
RIDGE_POINTS = [(0.0, 0.5), (2.2, 2.0)]
RIDGE_EPOCHS = [2, 8, 25]


def loop_prob_for(lam):
    """p = sigmoid(lambda): the loop probability giving penalty lambda at ts=1."""
    return 1.0 / (1.0 + math.exp(-lam))


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)

    # Silence frames removed with the alignment; nothing else collapsed. The
    # labels output of each is the silence-free *frame* alignment, which is the
    # reference the frame error rate has to be scored against.
    ls100 = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["ls-100"]["features"],
        alignment=GMM_ALIGNMENT_LS960_FRAME,
        exclude_labels=(0,),
        pooling="none",
        # ~13.9M frames, ~28.5 GB written: four times the segmented file.
        rqmt={"cpu": 2, "mem": 16, "time": 12},
    )
    ls100.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_frames_nosil")
    cv = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["cv"]["features"],
        alignment=GMM_ALIGNMENT_CV,
        exclude_labels=(0,),
        pooling="none",
    )
    cv.add_alias(f"guided_kmeans/{exp_dir}/features_cv_frames_nosil")
    for name, job in (("ls100", ls100), ("cv", cv)):
        tk.register_output(
            f"guided_kmeans/{exp_dir}/features/{name}_frames_statistics.json",
            job.out_statistics,
        )

    latex_report = LatexTableReport(
        columns=[
            # lambda is what the sweep varies, but it is read as a loop
            # probability at a transition scale: both columns restate the same
            # setting, and either pair reproduces the run.
            "arm", "lambda",
            loop_probability("lambda", 1.0), loop_probability("lambda", 0.5),
            "scale", "decode_lambda", "decode_scale", "epoch",
            "mi", "per", "del", "ins", "sub", "fer",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["arm", "lambda", "scale", "decode_lambda", "decode_scale"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            "Unsegmented discrete-HMM training on silence-free ls-100h frames, decoded on "
            "silence-free cv frames. The GMM alignment removes silence but no longer gives "
            "the segmentation. lambda is the per-phoneme insertion penalty ts*logit(p) with "
            "ts=1; 'scale' is the acoustic scale; 'lambda'/'scale' are the training setting "
            "and 'decode lambda'/'decode scale' the one decoded at. Under-segmentation shows "
            "as deletions, over-segmentation as insertions. The 'probe' arm decodes with a "
            "supervised frame table and is an oracle ceiling, not a result."
        ),
    )
    latex_report_train = LatexTableReport(
        columns=[
            # lambda is what the sweep varies, but it is read as a loop
            # probability at a transition scale: both columns restate the same
            # setting, and either pair reproduces the run.
            "lambda", loop_probability("lambda", 1.0), loop_probability("lambda", 0.5),
            "scale", "epoch", "per", "del", "ins", "sub", "fer", "log_likelihood"
        ],
        sort_by=["lambda", "scale"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            "Unsegmented discrete-HMM training on silence-free ls-100h frames, decoded on "
            "silence-free cv frames. The GMM alignment removes silence but no longer gives "
            "the segmentation. lambda is the per-phoneme insertion penalty ts*logit(p) with "
            "ts=1; 'scale' is the acoustic scale; 'lambda'/'scale' are the training setting "
            "and 'decode lambda'/'decode scale' the one decoded at. Under-segmentation shows "
            "as deletions, over-segmentation as insertions. The 'probe' arm decodes with a "
            "supervised frame table and is an oracle ceiling, not a result."
        ),
    )
    recog_results = []

    def _decode(*, name, model_dir, dataset, lam, scale, arm, epoch, train=None, statistics=None, mi=None):
        """Decode at (lam, scale); ``train`` is the (lambda, scale) the table was trained at."""
        dc = DecodeConfig(
            centroids=COLLEAGUE_CENTROIDS_K512,
            model_dir=model_dir,
            recog_rasr_config=build_decode_config(
                None, LM_SCALE, loop_prob_for(lam),
                lm_order=LM_ORDER, beam_size=DECODE_BEAM_SIZE, transition_scale=TRANSITION_SCALE,
            ),
            distance_scale=scale,
            write_frame_labels=True,
        )
        res = decode_and_score(
            name, "cv", dc, dataset,
            rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960",
            alias_prefix=f"guided_kmeans/{exp_dir}",
        )
        res.fwd_job.rqmt["time"] = 8
        if res.frame_labels is not None:
            # Against the silence-free frame alignment, so positions correspond.
            res.fer = FrameErrorRateJob(res.frame_labels, cv.out_labels, lexicon).out_fer
            tk.register_output(f"guided_kmeans/{exp_dir}/eval/{name}_fer", res.fer)
        tk.register_output(f"guided_kmeans/{exp_dir}/per/{name}_per", res.per)
        recog_results.append(res)
        params = {"arm": arm, "decode_lambda": lam, "decode_scale": scale}
        if train is not None:
            train_params = {"lambda": train[0], "scale": train[1]}
            params.update(train_params)
            if (lam, decode_scale) == DECODE_POINT:
                latex_report_train.add_row(
                    result=res,
                    params=train_params,
                    epoch=epoch,
                    statistics=statistics,
                    values={k: v for k, v in (("mi", mi), ("fer", res.fer)) if v is not None},
                )
        latex_report.add_row(
            result=res,
            params=params,
            epoch=epoch,
            statistics=statistics,
            values={k: v for k, v in (("mi", mi), ("fer", res.fer)) if v is not None},
        )

    # --- probe: where does segmentation work at all, given a good table? -----
    if RUN_TRANSITION_PROBE:
        oracle = SupervisedVQTableJob(
            features_hdf=cv.out_features,
            labels=cv.out_labels,
            centroids=COLLEAGUE_CENTROIDS_K512,
            num_labels=40,
            table_floor=1e-2,
            heldout_fraction=0.2,
            split_seed=SEED,
        )
        oracle.add_alias(f"guided_kmeans/{exp_dir}/probe_oracle_frame_table")
        heldout = DatasetConfig(
            audio_hdf_path=cv.out_features,
            sampling_method=SegmentFile(oracle.out_heldout_segments),
            precomputed=True,
        )
        for lam in PROBE_LAMBDAS:
            for scale in PROBE_DISTANCE_SCALES:
                _decode(
                    name=f"probe_lambda-{lam}_scale-{scale}",
                    model_dir=oracle.out_model, dataset=heldout,
                    lam=lam, scale=scale, arm="probe", epoch=0,
                )

    # --- training ------------------------------------------------------------
    full_cv = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )
    for lam, scale in TRAIN_SETTINGS + (TRAIN_SETTINGS_WAVE_2 if RUN_WAVE_2 else []):
        name = f"train_lambda-{lam}_scale-{scale}"
        _, result = build_vq_training(
            features=ls100.out_features,
            lm_path=None,
            sigma=SIGMA,
            seed=SEED,
            num_epochs=NUM_EPOCHS,
            num_chunks=NUM_CHUNKS,
            lexicon=lexicon,
            alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
            num_workers=NUM_WORKERS,
            rqmt=EPOCH_RQMT,
            lm_order=LM_ORDER,
            beam_size=BEAM_SIZE,
            lm_scale=LM_SCALE,
            transition_scale=TRANSITION_SCALE,
            loop_prob=loop_prob_for(lam),
            distance_scale=scale,
            segments=ls100.out_segments,
            batches_per_epoch=BATCHES_PER_EPOCH,
            ema=EMA,
        )
        tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{name}.json", result.out_statistics)
        statistics = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=name, epoch_offset=1, lexicon=lexicon
        )
        for epoch in DECODE_EPOCHS:
            mi = MixtureDiagnosticsJob(result.out_artifacts["table"][epoch]).out_mi
            points = [DECODE_POINT] + (RIDGE_POINTS if epoch in RIDGE_EPOCHS else [])
            for decode_lam, decode_scale in points:
                _decode(
                    name=f"{name}_ep-{epoch}_dlambda-{decode_lam}_dscale-{decode_scale}",
                    model_dir=result.out_models[epoch], dataset=full_cv,
                    lam=decode_lam, scale=decode_scale, arm="train", epoch=epoch,
                    train=(lam, scale), statistics=statistics, mi=mi,
                )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")
    latex_report_train.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.train_only.tex")


def py():
    run()
