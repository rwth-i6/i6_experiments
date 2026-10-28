"""Detector boundaries in the early epochs: do the classes form faster?

``docs/free_segmentation/findings_and_directions.md`` §6.11 and §7 (item T3).

The bonus-0.5 segment-codebook arm of ``vq_gmm_tau_bonus`` is the one free-segmentation
run that took off. Its training argmax has almost no boundaries in epochs 1-10, and
through epoch ~20 its boundaries are worse than those of a two-frame change-point
detector on the raw frames. That detector reaches F1 0.73 within one frame against the
GMM alignment, where chance is 0.53, while the run is at 0.59 at epoch 15 and 0.66 at
epoch 20. From epoch ~50 to the takeoff at ~130 the two are level, while the run's frame
accuracy climbs slowly. The plateau is held by the classes, so the detector cannot be
expected to shorten it. This run asks the narrower question: given detector boundaries
from epoch 1, do the classes form faster in the early epochs, and does the head start
last once the detector is removed?

**One arm, one difference.** Same init (exact uniform table, frozen k512 codebook), the
same S=2 bonus-0.5 topology, and the same LM, AM scale and frames as the reference.
Added are per-frame transition scores from ``ChangePointTransitions`` (the op's SPEC.md
14.1): each utterance's ``round(0.25 T)`` strongest local maxima of the detector are its
boundaries, read as a noisy observation with error rate 0.1. Each disagreement with the
detector then costs log 9 = 2.2 nats, on top of the run's own duration prior. Two phases
make one chain:

* epochs 1-``DETECTOR_EPOCHS`` with the detector;
* ``FREE_EPOCHS`` more without it, from the last of those models: the reference's own
  search from there on.

The reference rows are the 0.5 arm's own epochs, which all exist. It is rebuilt with
that config's exact arguments (64 chunks, its rqmt) and capped at ``REFERENCE_EPOCHS``,
well inside what the bonus manager has finished, so this config submits none of its
jobs. Decodes are at 5-30 as the reference's.

**Reading it.**

* PER (standard and free-segmentation decode) at 5-20 against the reference:
  78.0 / 78.3 at epoch 20, 84.6 / 89.5 at epoch 10.
* The progress rows' argmax durations and boundary rates. During the detector phase they
  describe the *guided* segmentation, the detector's included, not the model's own.
* The rows after epoch 20, for whether the gain holds without the detector.
* Frame accuracy and boundary F1 per epoch are not in the statistics. They come from
  ``tools/backoff_fb/build/boundaries/argmax_epochs.py`` (a ~1 GPU-h job, as for §6.11).

**Cost.** An epoch is ~32 GPU-h: 16 chunks of ~2 h on a 1080. The detector costs
nothing measurable. 30 epochs are ~1,000 GPU-h, about 2.5 days on 16 gpu_11gb cards,
and the decodes are CPU only. The epochs are a chain whose length is not hashed, so a
shorter first look, e.g. ``DETECTOR_EPOCHS = 10`` with no free phase (~320 GPU-h), can
be extended later without recomputing anything.
"""
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
    ScaleCovsJob,
    chunked_clustering,
    mixture_flavor,
)
from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.chunked.flavors import (
    BackoffFBSearch,
)
from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.chunked.spec import Spec
from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.chunked.transitions import (
    ChangePointTransitions,
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
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_lm import (
    frame_level_ls100,
)
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_bonus as bonus

exp_dir = "vq_gmm_tau_detector"
version = 1

#: The arm this one copies, in vq_gmm_tau_bonus's terms.
REFERENCE_FRACTION = 0.5
REFERENCE_CODEBOOK = "segments"
#: The last decoded epoch (30) plus one, whose statistics describe the epoch-30
#: model. Must stay within what the bonus manager has finished.
REFERENCE_EPOCHS = 31

#: Epochs with the detector, then epochs without it from its last model.
DETECTOR_EPOCHS = 20
FREE_EPOCHS = 10
#: As the reference decodes (vq_gmm_tau_bonus.WEIGHTS_DECODE_EPOCHS).
DECODE_EPOCHS = [e for e in bonus.WEIGHTS_DECODE_EPOCHS if e <= DETECTOR_EPOCHS + FREE_EPOCHS]

#: ChangePointTransitions, as measured in findings §6.11 (F1 0.73 within a frame).
DETECTOR = dict(window=2, distance="euclidean", rate=0.25, boundary_error=0.1)

#: Unhashed. 16 chunks of 1/16 of the corpus, ~2 h each on a 1080.
NUM_CHUNKS = 16
#: A copy, so vq_gmm_tau_bonus's own dict stays as it is. As vq_gmm_tau_topology and
#: vq_gmm_tau_online: gpu_11gb only, leaving the capped L40S and A10 slots to the
#: running arms, and a nice above the age weight's maximum (1500), so these tasks stay
#: behind every other pending GPU job of this user however long they wait.
EPOCH_RQMT = dict(bonus.EPOCH_RQMT, sbatch_args=["-p", "gpu_11gb", "--nice=1600"])


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    ls100 = frame_level_ls100()
    cv = silence_free_cv_features()

    # Everything up to the reference run is vq_gmm_tau_bonus's, argument for
    # argument, so that the reference's epochs are that config's jobs.
    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    base_covs = DuplicateCovsJob(global_cov, bonus.NUM_CODEWORDS).out_covs
    covs = ScaleCovsJob(base_covs, bonus.TAU).out_covs
    init_table = NormalTableJob(
        bonus.NUM_LABELS, bonus.NUM_CODEWORDS, sigma=bonus.INIT_SIGMA, seed=bonus.SEED
    ).out_table
    centroids = COLLEAGUE_CENTROIDS_K512
    train_config = bonus._recog_config(bonus.LM_SCALE, 0.0, 1.0, use_fb=True)
    decode_config_rasr = bonus._recog_config(1.0, 0.0, None, use_fb=False)

    lam = bonus.bonus_lambda(REFERENCE_FRACTION)
    p = bonus.loop_for_lambda(lam, bonus.TRANSITION_SCALE)
    reference_name = (
        f"weights_tau-{bonus.TAU}_s-{bonus.NUM_STATES}_bonus-{REFERENCE_FRACTION}"
        f"_loop-{round(p, 4)}_ts-{bonus.TRANSITION_SCALE}_lm-{bonus.LM_SCALE}"
        f"_am-{bonus.AM_SCALE}_init-uniform_cb-{REFERENCE_CODEBOOK}")
    reference_epochs_total, reference_chunks = bonus.LONG_ARMS[
        (REFERENCE_FRACTION, REFERENCE_CODEBOOK)]
    assert REFERENCE_EPOCHS <= reference_epochs_total

    def flavor(centroids, covs, mixtures, transitions=None):
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
            backoff_fb=BackoffFBSearch(
                backoff_fb_root=tools.BACKOFF_FB,
                arpa_path=phonetic_lm_dict[bonus.LM_ORDER],
                lexicon=fb_lexicon,
                lm_scale=bonus.LM_SCALE,
                emission_scale=1.0,
                loop_probability=p,
                silence_loop_probability=None,
                transition_scale=bonus.TRANSITION_SCALE,
                num_states=bonus.NUM_STATES,
                apply_sentence_end=True,
                batch_size=bonus.BATCH_SIZE,
                max_frames=bonus.MAX_FRAMES,
                checkpoint_interval=-1,
                storage_dtype="float32",
                transitions=transitions,
            ),
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
    reference = chunked_clustering(
        num_epochs=REFERENCE_EPOCHS,
        flavor=flavor(centroids, covs, init_table),
        num_chunks=reference_chunks,
        rqmt=bonus.EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{bonus.exp_dir}/{reference_name}",
        **common,
    )

    detector_name = (
        f"{reference_name}_detector-w{DETECTOR['window']}-{DETECTOR['distance']}"
        f"-r{DETECTOR['rate']}-eps{DETECTOR['boundary_error']}")
    guided = chunked_clustering(
        num_epochs=DETECTOR_EPOCHS,
        flavor=flavor(centroids, covs, init_table,
                      transitions=Spec(ChangePointTransitions, dict(DETECTOR))),
        num_chunks=NUM_CHUNKS,
        rqmt=EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{exp_dir}/{detector_name}",
        **common,
    )
    free_name = f"{detector_name}_free-from-{DETECTOR_EPOCHS}"
    free = chunked_clustering(
        num_epochs=FREE_EPOCHS,
        flavor=flavor(
            guided.out_artifacts["centroids"][DETECTOR_EPOCHS],
            guided.out_artifacts["covs"][DETECTOR_EPOCHS],
            guided.out_artifacts["mixtures"][DETECTOR_EPOCHS],
        ),
        num_chunks=NUM_CHUNKS,
        rqmt=EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{exp_dir}/{free_name}",
        **common,
    ) if FREE_EPOCHS > 0 else None
    # One chain in the run's own numbering: the free phase's epoch k is epoch
    # DETECTOR_EPOCHS + k.
    arm_models = {int(e): path for e, path in guided.out_models.items()}
    arm_statistics = {int(e): path for e, path in guided.out_epoch_statistics.items()}
    if free is not None:
        arm_models.update({DETECTOR_EPOCHS + int(e): path for e, path in free.out_models.items()})
        arm_statistics.update({DETECTOR_EPOCHS + int(e): path
                               for e, path in free.out_epoch_statistics.items()})

    # Decodes as vq_gmm_tau_bonus builds them, so the reference's are that config's
    # finished jobs.
    cv_dataset = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )
    cv_frames = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["cv"]["features"],
        alignment=GMM_ALIGNMENT_CV,
        exclude_labels=(0,),
        pooling="none",
    ).out_features
    free_seg_segments = ShuffleAndSplitSegmentsJob(
        cv.out_segments,
        split={"decode": bonus.FREE_SEG_FRACTION, "rest": 1.0 - bonus.FREE_SEG_FRACTION},
    ).out_segments["decode"]
    cv_frames_dataset = DatasetConfig(
        audio_hdf_path=cv_frames,
        sampling_method=SegmentFile(free_seg_segments),
        precomputed=True,
    )

    def decode(name, model_dir, epoch):
        dc = DecodeConfig(
            centroids=centroids,
            model_dir=model_dir,
            recog_rasr_config=decode_config_rasr,
            distance_scale=bonus.BASE_DECODE_SCALE,
            write_frame_labels=True,
            num_workers=bonus.DECODE_WORKERS,
        )
        decode_name = f"{name}_ep-{epoch}_ds-{bonus.BASE_DECODE_SCALE}"
        res = decode_and_score(
            decode_name, "cv", dc, cv_dataset,
            rasr_path=tools.RASR_PATH, device="cpu",
            corpus_key="train-other-960",
        )
        tk.register_output(f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per)
        matched = DecodeConfig(
            centroids=centroids,
            model_dir=model_dir,
            recog_rasr_config=bonus._recog_config(
                bonus.MATCHED_DECODE_LM, bonus.FREE_SEG_LOOP, 1.0, use_fb=False),
            distance_scale=bonus.AM_SCALE,
            num_workers=bonus.DECODE_WORKERS,
        )
        free_res = decode_and_score(
            f"{name}_ep-{epoch}_freeseg-lm-{bonus.MATCHED_DECODE_LM}", "cv", matched,
            cv_frames_dataset, rasr_path=tools.RASR_PATH, device="cpu",
            corpus_key="train-other-960",
        )
        values = {
            "matched_per": free_res.per,
            "matched_del": free_res.deletion,
            "matched_ins": free_res.insertion,
            "matched_sub": free_res.substitution,
            "matched_hyp_ref": 1 - free_res.deletion + free_res.insertion,
        }
        return res, values

    report = LatexTableReport(
        columns=[param("boundaries", header=("Bound-", "aries")), "epoch", "per", "del",
                 "ins", "sub", "matched_per", "matched_del", "matched_ins", "matched_sub",
                 "matched_hyp_ref", "log_likelihood", "posterior_entropy",
                 "segment_duration_argmax", "boundary_rate"],
        sort_by=["boundaries"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Change-point boundaries in the early epochs: the vq\\_gmm\\_tau\\_bonus "
            f"{REFERENCE_FRACTION} arm ({REFERENCE_CODEBOOK} codebook, S={bonus.NUM_STATES}, "
            f"LM {bonus.LM_SCALE}, am=ts={bonus.AM_SCALE}, loop {p:.4f}) as it ran, and with "
            f"per-frame transition scores from a change-point detector on the frames in "
            f"epochs 1-{DETECTOR_EPOCHS} (window {DETECTOR['window']}, "
            f"{DETECTOR['distance']}, rate {DETECTOR['rate']}, error rate "
            f"{DETECTOR['boundary_error']}), then without them. The statistics columns "
            f"describe the decoded model as the next epoch's search sees it."
        ),
    )
    progress_report = TextTableReport(
        columns=[param("boundaries"), "epoch", "log_likelihood", "posterior_entropy",
                 "segment_duration_argmax", "boundary_rate", "dead_clusters"],
        sort_by=["boundaries"],
        drop_empty_rows=True,
        title=(
            f"Per-epoch convergence with and without detector boundaries (rows are the "
            f"epoch jobs, which recognize with the previous epoch's model). In epochs "
            f"1-{DETECTOR_EPOCHS} of the detector arm, durations and boundary rate describe "
            f"the detector-guided posteriors, not the model's own segmentation. Reference "
            f"(findings 6.11): argmax boundary rate 0.006 at epoch 1, 0.08 at 10 and 0.25 "
            f"at 20; the alignment has 0.25."
        ),
    )
    recog_results = []

    def add_rows(params, name, models, epoch_statistics):
        """``models`` and ``epoch_statistics`` keyed by the run's own epoch."""
        stats = clustering_statistics_per_epoch(
            epoch_statistics, name=name, epoch_offset=1, lexicon=lexicon)
        progress = clustering_statistics_per_epoch(
            epoch_statistics, name=name, epoch_offset=0, lexicon=lexicon)
        for e in sorted(epoch_statistics):
            if e <= DETECTOR_EPOCHS + FREE_EPOCHS:
                progress_report.add_row(params=dict(params), epoch=e, statistics=progress)
        for e in DECODE_EPOCHS:
            res, values = decode(name, models[e], e)
            recog_results.append(res)
            report.add_row(result=res, params=dict(params), epoch=e, statistics=stats,
                           values=values)

    add_rows({"boundaries": "free (reference)"}, reference_name,
             {int(e): path for e, path in reference.out_models.items()},
             {int(e): path for e, path in reference.out_epoch_statistics.items()})
    add_rows({"boundaries": f"detector to ep {DETECTOR_EPOCHS}"}, detector_name,
             arm_models, arm_statistics)

    tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{detector_name}.json",
                       guided.out_statistics)
    if free is not None:
        tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{free_name}.json",
                           free.out_statistics)
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
