"""Calibrating the silence topology by forward passes only, before trusting a setting.

``docs/free_segmentation/findings_and_directions.md`` §6.12. As
``vq_gmm_tau_s3_calibration`` did for S=3: one E-step of a trained model per topology,
on a fixed 2% of the ls-100h VAD frames with their silence kept, reading the posterior
statistics - no model is updated. ``FBStatisticsCounter`` splits the argmax durations
into silence (label 0) and speech, and gives the silence share soft and by argmax.

**Targets** (the GMM alignment on these frames, corpus-level scalars as the 3.99
frames per phoneme are): silence 8.9% of the frames, in runs of 6.9 frames on
average; speech durations where the bonus-0.5 arm had them, 4.03-4.13 frames.

**Two ways of writing the topology.** On a fixed-length utterance the transitions
reduce to a per-phoneme term, a per-silence-segment term ``S * ts * ln((1-q)/q) + b``
and a per-silence-frame term ``f = ts * ln(q / p)`` against a speech frame.

* *Exit offsets* (``vq_gmm_tau_silence``'s arms): speech loop 0.5 with the bonus as a
  0.944-nat exit offset per phoneme, silence loop q with exit offset b. A grid over q
  and b covers every (f, segment term) reachable at ts = 0.35.
* *Loops only* (the bonus-0.5 arm's way): speech loop 0.2061 at ts = 0.35, i.e. the
  same bonus written as a low loop, silence loop q, no offsets. q = 0.2061 is that
  arm's own topology; higher q shows the per-frame pull f toward silence that this
  way of writing it adds. One row at ts = 1 with the same per-phoneme bonus (speech
  loop 0.384) shows what the transition scale does once the loops differ.

**Two LMs** (added 2026-10-27): every setting runs with the original LM and with the
one re-silenced to the VAD frames (``vq_gmm_tau_silence_lm``), which the LM's random
silence - ~57% of the data's segments in the first branch updates - asked for.

**Models.** The converged bonus-0.5 model (epoch 200), never trained on silence. Its
``<SIL>`` is what the LM's random silences made of speech frames, so this calibrates
the topology's push, not a trained silence model. Models of ``vq_gmm_tau_silence``'s
VAD arms are added to ``MODELS`` once their updates have *finished*: this config runs
under its own manager and must not build that config's pending jobs.

Cost: an S=2 E-step on 2% of the corpus, ~0.5-1 GPU-h with the graph compile; 24
settings per model, on gpu_11gb at ``--nice=1600``.
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
    param,
    statistic,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_lm import (
    frame_level_ls100,
)
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_bonus as bonus
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_silence as silence
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_silence_lm import (
    vad_silence_lm,
)

exp_dir = "vq_gmm_tau_silence_calibration"
version = 1

#: Models recognized with, by label: only finished jobs (module docstring).
REFERENCE_EPOCHS = 200
MODELS = ["epoch 200 (trained without silence)"]

#: Exit-offset grid: silence loop (generative mean S / (1 - q) = 4, 5, 6.9, 10, 20
#: frames at S=2) by silence exit offset in nats.
SILENCE_LOOPS = [0.5, 0.6, 0.71, 0.8, 0.9]
SILENCE_EXIT_OFFSETS = [-1.0, 0.0, 1.0, 2.0]
#: Loops only: ``(transition scale, silence loop)``; the speech loop gives the
#: bonus-0.5 arm's per-phoneme bonus at that scale, and q=None is its own topology.
LOOP_ONLY = [(bonus.TRANSITION_SCALE, None), (bonus.TRANSITION_SCALE, 0.71),
             (bonus.TRANSITION_SCALE, 0.9), (1.0, 0.71)]

#: Share of the VAD-frame sequences recognized, fixed by the split's seed.
CALIBRATION_FRACTION = 0.02
#: Unhashed. Two tasks of ~280 sequences each.
NUM_CHUNKS = 2
EPOCH_RQMT = dict(bonus.EPOCH_RQMT, sbatch_args=["-p", "gpu_11gb", "--nice=1600"])


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    nosil = frame_level_ls100()
    vad = silence.vad_frames()
    train_config = bonus._recog_config(bonus.LM_SCALE, 0.0, 1.0, use_fb=True)

    def search(loop_probability, silence_loop_probability=None, exit_offset=None,
               transition_scale=bonus.TRANSITION_SCALE, arpa_path=None):
        return BackoffFBSearch(
            backoff_fb_root=tools.BACKOFF_FB,
            arpa_path=phonetic_lm_dict[bonus.LM_ORDER] if arpa_path is None else arpa_path,
            lexicon=fb_lexicon,
            lm_scale=bonus.LM_SCALE,
            emission_scale=1.0,
            loop_probability=loop_probability,
            silence_loop_probability=silence_loop_probability,
            transition_scale=transition_scale,
            num_states=bonus.NUM_STATES,
            apply_sentence_end=True,
            batch_size=bonus.BATCH_SIZE,
            max_frames=bonus.MAX_FRAMES,
            checkpoint_interval=-1,
            storage_dtype="float32",
            exit_offset=exit_offset,
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
        recognition_config=train_config,
        lexicon=lexicon,
        num_clusters=bonus.NUM_LABELS,
        distance_scale=bonus.AM_SCALE,
        use_forward_backward=True,
        rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
        num_workers=bonus.NUM_WORKERS,
        parent_threads=bonus.PARENT_THREADS,
    )

    # The bonus-0.5 arm, argument for argument, capped at its finished end.
    ref_covs = ScaleCovsJob(
        DuplicateCovsJob(GlobalCovarianceJob(nosil.out_features).out_cov,
                         bonus.NUM_CODEWORDS).out_covs, bonus.TAU).out_covs
    uniform = NormalTableJob(
        bonus.NUM_LABELS, bonus.NUM_CODEWORDS, sigma=bonus.INIT_SIGMA, seed=bonus.SEED
    ).out_table
    reference_lam = bonus.bonus_lambda(silence.REFERENCE_FRACTION)
    reference_p = bonus.loop_for_lambda(reference_lam, bonus.TRANSITION_SCALE)
    reference_name = (
        f"weights_tau-{bonus.TAU}_s-{bonus.NUM_STATES}_bonus-{silence.REFERENCE_FRACTION}"
        f"_loop-{round(reference_p, 4)}_ts-{bonus.TRANSITION_SCALE}_lm-{bonus.LM_SCALE}"
        f"_am-{bonus.AM_SCALE}_init-uniform_cb-{silence.REFERENCE_CODEBOOK}")
    _, reference_chunks = bonus.LONG_ARMS[
        (silence.REFERENCE_FRACTION, silence.REFERENCE_CODEBOOK)]
    reference = chunked_clustering(
        num_epochs=REFERENCE_EPOCHS,
        features_hdf=nosil.out_features,
        flavor=flavor(COLLEAGUE_CENTROIDS_K512, ref_covs, uniform, search(reference_p)),
        num_chunks=reference_chunks,
        rqmt=bonus.EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{bonus.exp_dir}/{reference_name}",
        **common,
    )
    models = {
        MODELS[0]: {name: reference.out_artifacts[name][REFERENCE_EPOCHS]
                    for name in ("centroids", "covs", "mixtures")},
    }
    #: The training LM: the original, and the one re-silenced to the VAD frames.
    lms = {"original": None, "VAD silence": vad_silence_lm(bonus.LM_ORDER)}

    calibration_segments = ShuffleAndSplitSegmentsJob(
        vad.out_segments,
        split={"calibration": CALIBRATION_FRACTION, "rest": 1.0 - CALIBRATION_FRACTION},
    ).out_segments["calibration"]

    # (LM, label, search, transition scale, speech loop, silence loop, silence exit
    # offset). The original LM's labels carry no LM tag, so its jobs keep their names.
    settings = []
    for lm_name, arpa_path in lms.items():
        tag = "" if lm_name == "original" else "_lm-vadsil"
        for q in SILENCE_LOOPS:
            for b in SILENCE_EXIT_OFFSETS:
                offsets = {**{phoneme: silence.SPEECH_BONUS for phoneme in silence.PHONEMES},
                           silence.SILENCE: b}
                settings.append((lm_name, f"exit-offsets_q-{q:g}_b-{b:g}{tag}",
                                 search(silence.SPEECH_LOOP, q, offsets, arpa_path=arpa_path),
                                 bonus.TRANSITION_SCALE, silence.SPEECH_LOOP, q, b))
        for ts, q in LOOP_ONLY:
            p = bonus.loop_for_lambda(reference_lam, ts)
            q_eff = p if q is None else q
            settings.append((lm_name, f"loops_ts-{ts:g}_p-{round(p, 4)}_q-{round(q_eff, 4)}{tag}",
                             search(p, q, transition_scale=ts, arpa_path=arpa_path),
                             ts, p, q_eff, 0.0))

    headers = {
        "frame_term": ("Sil. frame", "term f"),
        "segment_term": ("Sil. segment", "term"),
        "generative_mean": ("Sil. gen.", "mean"),
    }

    def value(key, fmt):
        return Column(key, headers[key], lambda row: row.values.get(key), fmt=fmt)

    report = TextTableReport(
        columns=[param("model", header=("Model",)),
                 param("lm", header=("LM",)),
                 param("encoding", header=("Encoding",)),
                 param("ts", header=("ts",), fmt="{:g}"),
                 param("speech_loop", header=("Speech", "loop"), fmt="{:.4f}"),
                 param("silence_loop", header=("Sil.", "loop"), fmt="{:.4f}"),
                 param("silence_exit", header=("Sil.", "exit"), fmt="{:+g}"),
                 value("frame_term", "{:+.3f}"),
                 value("segment_term", "{:+.3f}"),
                 value("generative_mean", "{:.1f}"),
                 statistic("silence_share_soft", ("Sil. share", "soft [%]"), fmt="{:.1f}",
                           scale=100),
                 statistic("silence_share_argmax", ("Sil. share", "argmax [%]"),
                           fmt="{:.1f}", scale=100),
                 statistic("silence_duration_argmax", ("Sil. dur.", "(argmax)"),
                           fmt="{:.2f}"),
                 statistic("speech_duration_argmax", ("Phon. dur.", "(argmax)"),
                           fmt="{:.2f}"),
                 "posterior_entropy", "log_likelihood"],
        sort_by=["model", "lm", "encoding", "ts", "silence_loop", "silence_exit"],
        drop_empty_rows=True,
        title=(
            f"Silence calibration: one E-step per topology on a fixed "
            f"{CALIBRATION_FRACTION:.0%} of the ls-100h VAD frames, silence kept. Targets "
            f"from the GMM alignment: silence 8.9% of the frames in runs of 6.9 frames; "
            f"speech 4.03-4.13 frames (the bonus-0.5 arm). Terms in nats per utterance "
            f"path: f per silence frame against a speech frame, ts ln(q/p); per silence "
            f"segment S ts ln((1-q)/q) + exit offset (negative is a price). Generative "
            f"mean S / (1 - q) of silence at the given loop. Exit-offset rows: speech "
            f"loop {silence.SPEECH_LOOP} plus a {silence.SPEECH_BONUS:g}-nat exit bonus "
            f"(vq_gmm_tau_silence's arms run q = {silence.SILENCE_LOOP}, b = "
            f"{silence.SILENCE_EXIT_OFFSET:g}); loop rows: the same bonus as a low "
            f"speech loop, no offsets."
        ),
    )

    for model_label, start in models.items():
        for lm_name, label, backoff_fb, ts, p, q, b in settings:
            name = f"{reference_name}_model-ep{REFERENCE_EPOCHS}_{label}"
            calibration = chunked_clustering(
                num_epochs=1,
                features_hdf=vad.out_features,
                flavor=flavor(start["centroids"], start["covs"], start["mixtures"],
                              backoff_fb),
                segments=calibration_segments,
                num_chunks=NUM_CHUNKS,
                rqmt=EPOCH_RQMT,
                alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
                **common,
            )
            statistics = clustering_statistics_per_epoch(
                calibration.out_epoch_statistics, name=name, epoch_offset=0,
                lexicon=lexicon)
            report.add_row(
                params={"model": model_label, "lm": lm_name, "encoding": label.split("_")[0],
                        "ts": ts, "speech_loop": p, "silence_loop": q, "silence_exit": b},
                epoch=1,
                statistics=statistics,
                values={
                    "frame_term": ts * math.log(q / p),
                    "segment_term": bonus.NUM_STATES * ts * math.log((1 - q) / q) + b,
                    "generative_mean": bonus.NUM_STATES / (1 - q),
                },
            )

    report.register(
        f"guided_kmeans/{exp_dir}/calibration_{version}.txt",
        required=report, update_frequency=60,
    )


def py():
    run()
