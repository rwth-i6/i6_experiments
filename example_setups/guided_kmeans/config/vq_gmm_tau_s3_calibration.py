"""Calibrating the loop probability for S=3 on trained S=2 models, before an S=3 run.

``docs/free_segmentation/findings_and_directions.md`` §6.10 and §7 (item 3).

The S=2 bonus sweep found that what decides a free-segmentation run is where its
*posterior* phoneme duration settles: the 0.5 arm lands at 4.04-4.13 frames against
the alignment's 3.99 and is the one that took off; longer (0.25: 4.61, control: 5.1)
plateaus, shorter (0.75: 3.74) is slower. For S=3 the two readings of the topology
disagree by a factor of ~8 on the bonus that would reproduce this - matching the
per-phoneme bonus gives 0.94 nats, matching the duration ~7.7 - and a wrong guess
costs a plateau of tens of epochs. This measures it instead.

**One E-step per setting, on a trained model.** The 0.5 arm's tied table works for
any S, so its model recognizes a fixed 2% of the silence-free ls-100h frames (~560
sequences, ~70k phonemes) with S=3 at a range of bonuses, and the epoch job's
statistics give the argmax duration, boundary rate and entropy. Two models: epoch
20, when the 0.5 arm's durations had just settled (classes still poor, 78% PER), and
epoch 160 (38%). If both put the same bonus at ~4.0-4.1 frames, the choice does not
depend on how far training has got.

**S=2 rows check the method.** At the 0.5 arm's own setting, the one-step duration
must reproduce what the run measured on the full corpus with the same model (4.04
after both epoch 20 and 160). At the 0.25 and 0.75 settings it is the response of a
model *not* trained under them; the 0.25 / 0.75 arms themselves settled at 4.48 /
3.88 (epoch 21) and 4.61 / 3.74 (epoch 56). The gap between the two is how much a
run's own training amplifies the topology's push, and it says how to read the S=3
rows: if the one-step spread is narrower than the runs', the S=3 run will move
further from the 0.5 model's response than the one-step number suggests.

Bonuses are fractions of the LM's 1.888-nat phoneme cost, as in
``vq_gmm_tau_bonus``; per state the topology then charges ``lambda = -fraction *
1.888 / S`` at ts=0.35. The report also gives the generative mean ``S / (1 - p)`` at
the equivalent ts=1 loop.

Cost: an S=3 E-step on 2% of the corpus is ~1 GPU-h with startup; 20 jobs, ~20
GPU-h, on gpu_11gb at the lowest priority, behind ``vq_gmm_tau_online``.
"""

import math

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
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
    create_fb_lexicon,
    phonetic_lm_dict,
)
from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import (
    Column,
    TextTableReport,
    clustering_statistics_per_epoch,
    loop_probability as loop_probability_column,
    param,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_lm import (
    frame_level_ls100,
)
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_bonus as bonus

exp_dir = "vq_gmm_tau_s3_calibration"
version = 1

#: The arm whose models are calibrated on, in vq_gmm_tau_bonus's terms.
REFERENCE_FRACTION = 0.5
REFERENCE_CODEBOOK = "segments"
#: Models after these epochs. The reference chain is rebuilt only up to the
#: last, all finished, so this config never submits the bonus run's epochs.
MODEL_EPOCHS = [20, 160]

#: ``{S: bonus fractions}``. S=2 at the bonus arms' own settings (the check);
#: S=3 from the per-phoneme-matched 0.5 up to past the duration-matched estimate
#: (~4, i.e. ~7.6 nats).
FRACTIONS = {
    2: [0.25, 0.5, 0.75],
    3: [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
}

#: Share of the silence-free ls-100h sequences recognized, fixed by the split's
#: seed so every setting sees the same ones.
CALIBRATION_FRACTION = 0.02

#: Unhashed. Two tasks of ~280 sequences each.
NUM_CHUNKS = 2
#: Unhashed. The op's memory grows linearly in S; S=2 at B=8 and 9,000 frames
#: fits an 11 GB card, so S=3 keeps the same states x frames.
MAX_FRAMES = {2: bonus.MAX_FRAMES, 3: bonus.MAX_FRAMES * 2 // 3}
#: As vq_gmm_tau_online: gpu_11gb only, behind every other pending GPU job.
EPOCH_RQMT = dict(bonus.EPOCH_RQMT, sbatch_args=["-p", "gpu_11gb", "--nice=1600"])


def calibration_lambda(fraction: float, num_states: int) -> float:
    """``bonus.bonus_lambda`` for any S: a per-phoneme bonus of ``fraction`` of the LM cost."""
    return round(-fraction * bonus.LM_PHONEME_COST / num_states, 3) + 0.0


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    ls100 = frame_level_ls100()

    # The reference chain is vq_gmm_tau_bonus's, argument for argument, so that
    # its epochs are that config's finished jobs.
    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    base_covs = DuplicateCovsJob(global_cov, bonus.NUM_CODEWORDS).out_covs
    covs = ScaleCovsJob(base_covs, bonus.TAU).out_covs
    init_table = NormalTableJob(
        bonus.NUM_LABELS, bonus.NUM_CODEWORDS, sigma=bonus.INIT_SIGMA, seed=bonus.SEED
    ).out_table
    train_config = bonus._recog_config(bonus.LM_SCALE, 0.0, 1.0, use_fb=True)

    def search(num_states, loop_probability, max_frames):
        return BackoffFBSearch(
            backoff_fb_root=tools.BACKOFF_FB,
            arpa_path=phonetic_lm_dict[bonus.LM_ORDER],
            lexicon=fb_lexicon,
            lm_scale=bonus.LM_SCALE,
            emission_scale=1.0,
            loop_probability=loop_probability,
            silence_loop_probability=None,
            transition_scale=bonus.TRANSITION_SCALE,
            num_states=num_states,
            apply_sentence_end=True,
            batch_size=bonus.BATCH_SIZE,
            max_frames=max_frames,
            checkpoint_interval=-1,
            storage_dtype="float32",
        )

    def flavor(centroids, covs, mixtures, backoff_fb):
        return mixture_flavor(
            centroids=centroids,
            covs=covs,
            mixtures=mixtures,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=bonus.NUM_LABELS,
            distance_scale=bonus.AM_SCALE,
            use_forward_backward=True,
            update_densities=False,
            update_covariances=None,
            mixture_floor=1e-2,
            num_workers=bonus.NUM_WORKERS,
            backoff_fb=backoff_fb,
        )

    common = dict(
        features_hdf=ls100.out_features,
        recognition_config=train_config,
        lexicon=lexicon,
        num_clusters=bonus.NUM_LABELS,
        distance_scale=bonus.AM_SCALE,
        use_forward_backward=True,
        rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
        num_workers=bonus.NUM_WORKERS,
        parent_threads=bonus.PARENT_THREADS,
    )

    reference_lam = bonus.bonus_lambda(REFERENCE_FRACTION)
    reference_p = bonus.loop_for_lambda(reference_lam, bonus.TRANSITION_SCALE)
    reference_name = (
        f"weights_tau-{bonus.TAU}_s-{bonus.NUM_STATES}_bonus-{REFERENCE_FRACTION}"
        f"_loop-{round(reference_p, 4)}_ts-{bonus.TRANSITION_SCALE}_lm-{bonus.LM_SCALE}"
        f"_am-{bonus.AM_SCALE}_init-uniform_cb-{REFERENCE_CODEBOOK}")
    _, reference_chunks = bonus.LONG_ARMS[(REFERENCE_FRACTION, REFERENCE_CODEBOOK)]
    reference = chunked_clustering(
        num_epochs=max(MODEL_EPOCHS),
        flavor=flavor(COLLEAGUE_CENTROIDS_K512, covs, init_table,
                      search(bonus.NUM_STATES, reference_p, bonus.MAX_FRAMES)),
        num_chunks=reference_chunks,
        rqmt=bonus.EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{bonus.exp_dir}/{reference_name}",
        **common,
    )

    calibration_segments = ShuffleAndSplitSegmentsJob(
        ls100.out_segments,
        split={"calibration": CALIBRATION_FRACTION, "rest": 1.0 - CALIBRATION_FRACTION},
    ).out_segments["calibration"]

    generative_mean = Column(
        "generative_mean", ("Generative", "mean (frames)"),
        lambda row: row.values.get("generative_mean"), fmt="{:.2f}")
    report = TextTableReport(
        columns=[param("model", header=("Model", "epoch")),
                 param("states", header=("HMM", "states")),
                 param("bonus", header=("Bonus", "frac."), fmt="{:g}"),
                 loop_probability_column("lambda", bonus.TRANSITION_SCALE),
                 loop_probability_column("lambda", 1.0),
                 generative_mean,
                 "segment_duration_argmax", "boundary_rate", "posterior_entropy",
                 "log_likelihood"],
        sort_by=["model", "states", "bonus"],
        drop_empty_rows=True,
        title=(
            f"S=3 calibration: one E-step of vq_gmm_tau_bonus's bonus-{REFERENCE_FRACTION} "
            f"arm (S={bonus.NUM_STATES}, segment codebook, tied) after epochs "
            f"{' and '.join(map(str, MODEL_EPOCHS))}, on a fixed "
            f"{CALIBRATION_FRACTION:.0%} of silence-free ls-100h, at each topology. Bonus = "
            f"share of the LM's {bonus.LM_PHONEME_COST}-nat phoneme cost paid back; "
            f"generative mean = S / (1 - p) at the ts=1 loop. Target: the 0.5 arm's "
            f"argmax duration, 4.04 frames after both epochs on the full corpus. Full "
            f"runs at the S=2 settings (next epoch's job): 0.25 -> 4.48 (ep 21) / 4.61 "
            f"(ep 56), 0.75 -> 3.88 / 3.74; read the S=2 rows' spread against these."
        ),
    )

    for model_epoch in MODEL_EPOCHS:
        for num_states, fractions in FRACTIONS.items():
            for fraction in fractions:
                lam = calibration_lambda(fraction, num_states)
                p = bonus.loop_for_lambda(lam, bonus.TRANSITION_SCALE)
                name = (f"{reference_name}_model-ep{model_epoch}_s-{num_states}"
                        f"_bonus-{fraction}_loop-{round(p, 4)}")
                calibration = chunked_clustering(
                    num_epochs=1,
                    flavor=flavor(
                        reference.out_artifacts["centroids"][model_epoch],
                        reference.out_artifacts["covs"][model_epoch],
                        reference.out_artifacts["mixtures"][model_epoch],
                        search(num_states, p, MAX_FRAMES[num_states]),
                    ),
                    segments=calibration_segments,
                    num_chunks=NUM_CHUNKS,
                    rqmt=EPOCH_RQMT,
                    alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
                    **common,
                )
                statistics = clustering_statistics_per_epoch(
                    calibration.out_epoch_statistics, name=name, epoch_offset=0,
                    lexicon=lexicon)
                loop_at_ts1 = 1.0 / (1.0 + math.exp(-lam))
                report.add_row(
                    params={"model": model_epoch, "states": num_states,
                            "bonus": fraction, "lambda": lam},
                    epoch=1,
                    statistics=statistics,
                    values={"generative_mean": num_states / (1.0 - loop_at_ts1)},
                )

    report.register(
        f"guided_kmeans/{exp_dir}/calibration_{version}.txt",
        required=report, update_frequency=60,
    )


def py():
    run()
