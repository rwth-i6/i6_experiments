"""Trained loop probabilities: EM on the HMM topology, branched at epoch 20.

Every run so far has held the topology fixed: one loop probability ``p``, set
from a target duration or swept as a boundary bonus (``vq_gmm_tau_bonus``). The
``backoff_fb`` op now also returns how often a path stays in and leaves every
(phoneme, state), its transition counts (milestone T5, SPEC.md section 15 in the
op's checkout). Summed over the corpus they are the statistics of the EM update
of ``p`` with the emissions and the LM held fixed, ``p = sum stay / sum (stay +
move)`` over each tying group, and an epoch now makes that update next to the
emission one (T6: ``BackoffFBSearch(train_topology=...)``). The update is EM at
any transition scale, so it cannot lower the summed log Z it was computed under.

**Two arms, one difference each from the reference.** Both are
``vq_gmm_tau_bonus``'s zero-bonus frame-codebook arm (S=2, LM 1.0, am = ts =
0.35, tau=4, frozen frame k512 codebook, weights only, p = 0.5), with the
topology trained: one ``p`` for every state (``global``), or one per phoneme
(``phoneme``). Both start from that run's model after epoch 20 (77.8% PER,
5.26 frames per phoneme in the argmax) at its ``p = 0.5`` and run 10 epochs,
numbered 21-30 here. The reference rows are the run's own epochs 21-30, which
exist already (77.9-78.0%). Branching skips the early epochs, whose ~120-frame
durations would push the boundary price up (SPEC.md 15.6). Training from epoch
1 is the planned follow-up, with a Beta prior.

**Regularization: the floor only.** A group with fewer than 1,000 expected
frames keeps its previous ``p``, so the arms measure plain EM.

**Prediction (SPEC.md 15.6).** Global EM sets ``p = 1 - S / d``, ``d`` the
model's own mean phoneme duration, so it moves the topology toward the
durations the model already produces rather than toward the alignment's 3.99
frames. The argmax durations (5.26 frames) put the global arm's first update
at about ``1 - 2/5.26 = 0.62``. Measured on 71 training sequences with the
epoch-20 model, the expected duration is 4.91 frames and one update gives
0.592, i.e. lambda = s * logit(p) = +0.131 per state: a surcharge, the bonus
sweep's weakest direction, and it is expected to drift further toward ~+0.2.
With phoneme tying, ``s * ln p_a`` per frame no longer cancels across paths and
acts as a label prior in favour of long phonemes.

**Branch epoch 21 is a consistency check.** It recognizes with the same model
and ``p`` as the reference's epoch 21, so its statistics and its emission model
must match the reference's to rounding (bitwise only if the chunks also land on
the same GPU types). ``p`` moves only from epoch 22 on.

**What the numbers show.** The decodes (25 and 30, as in the reference) use
fixed RASR loop settings, so a trained ``p`` reaches none of them: its effect
shows in the training statistics - the EM estimate and its lambda, the expected
frames per phoneme, ``segment_duration_argmax`` and ``boundary_rate`` - and in
the emissions training produces. The topology each epoch trained is in
``out_topologies``, and every group's estimate in its ``statistics.json``.

Cost: the reference epoch ran 16 chunks of about 1.9 h, and counting adds ~15%,
so ~30-35 GPU-h per epoch and arm, 600-700 GPU-h for both.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    GMM_ALIGNMENT_CV,
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
    statistic,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_lm import (
    frame_level_ls100,
)
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_bonus as bonus

exp_dir = "vq_gmm_tau_topology"
version = 1

#: The arm both branches start from, in vq_gmm_tau_bonus's terms.
REFERENCE_FRACTION = 0.0
REFERENCE_CODEBOOK = "frames"
#: The reference's own length; its epochs 21-30 are the comparison rows.
REFERENCE_EPOCHS = bonus.WEIGHTS_EPOCHS

#: Branch after this epoch's model, and run this many epochs from it.
BRANCH_EPOCH = 20
BRANCH_EPOCHS = 10
#: In the run's own numbering, as the reference decodes.
DECODE_EPOCHS = [25, 30]

#: ``BackoffFBSearch.train_topology`` per arm.
TYINGS = ["global", "phoneme"]
#: The floor: a group with fewer expected frames keeps its previous p. Only
#: silence comes near it on these silence-free frames.
MIN_FRAMES = 1_000.0

#: Unhashed operational settings, as vq_gmm_tau_bonus. Counting adds ~15% to a
#: chunk, well inside its 8 h.
NUM_CHUNKS = bonus.NUM_CHUNKS
#: A copy, so vq_gmm_tau_bonus's own dict stays as it is. These chunks must not
#: slow the runs already going (2026-10-19):
#: - gpu_11gb only, so the L40S, A10 and V100 slots, all under per-user caps or
#:   full, stay with those runs. A 1080 chunk takes about 2 h with counting.
#: - ``--nice`` keeps them behind every other pending GPU job, the running runs'
#:   fresh epochs included. Age adds at most PriorityWeightAge = 1500 (over 7
#:   days), so a nice above that holds however long they wait.
EPOCH_RQMT = dict(bonus.EPOCH_RQMT, sbatch_args=["-p", "gpu_11gb", "--nice=1600"])


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    ls100 = frame_level_ls100()
    cv = silence_free_cv_features()

    # Everything below up to the reference run is vq_gmm_tau_bonus's, so that
    # the reference's 30 epochs are that config's jobs, not new ones.
    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    base_covs = DuplicateCovsJob(global_cov, bonus.NUM_CODEWORDS).out_covs
    covs = ScaleCovsJob(base_covs, bonus.TAU).out_covs
    init_table = NormalTableJob(
        bonus.NUM_LABELS, bonus.NUM_CODEWORDS, sigma=bonus.INIT_SIGMA, seed=bonus.SEED
    ).out_table
    centroids = bonus.frame_codebook(ls100.out_features, base_covs).out_centroids[
        bonus.CODEBOOK_EPOCHS]
    train_config = bonus._recog_config(bonus.LM_SCALE, 0.0, 1.0, use_fb=True)
    decode_config_rasr = bonus._recog_config(1.0, 0.0, None, use_fb=False)

    lam = bonus.bonus_lambda(REFERENCE_FRACTION)
    p = bonus.loop_for_lambda(lam, bonus.TRANSITION_SCALE)
    reference_name = (
        f"weights_tau-{bonus.TAU}_s-{bonus.NUM_STATES}_bonus-{REFERENCE_FRACTION}"
        f"_loop-{round(p, 4)}_ts-{bonus.TRANSITION_SCALE}_lm-{bonus.LM_SCALE}"
        f"_am-{bonus.AM_SCALE}_init-uniform_cb-{REFERENCE_CODEBOOK}")

    def search(**training):
        return BackoffFBSearch(
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
            **training,
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

    def clustering(num_epochs, flavor, alias):
        return chunked_clustering(
            num_epochs=num_epochs,
            features_hdf=ls100.out_features,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=bonus.NUM_LABELS,
            flavor=flavor,
            distance_scale=bonus.AM_SCALE,
            use_forward_backward=True,
            rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
            num_chunks=NUM_CHUNKS,
            num_workers=bonus.NUM_WORKERS,
            parent_threads=bonus.PARENT_THREADS,
            rqmt=EPOCH_RQMT,
            alias_prefix=alias,
        )

    reference = clustering(
        REFERENCE_EPOCHS,
        flavor(centroids, covs, init_table, search()),
        f"guided_kmeans/{bonus.exp_dir}/{reference_name}",
    )

    # The free-segmentation decode of vq_gmm_tau_bonus, on the same cv subset.
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
        # write_frame_labels as vq_gmm_tau_bonus sets it, so that the
        # reference's decodes are that config's jobs. Its FER is left out: it
        # sits at ~96.4 in every row of that family and says nothing here.
        dc = DecodeConfig(
            centroids=centroids,
            model_dir=model_dir,
            recog_rasr_config=decode_config_rasr,
            distance_scale=bonus.BASE_DECODE_SCALE,
            write_frame_labels=True,
        )
        decode_name = f"{name}_ep-{epoch}_ds-{bonus.BASE_DECODE_SCALE}"
        res = decode_and_score(
            decode_name, "cv", dc, cv_dataset,
            rasr_path=tools.RASR_PATH, device="cpu",
            corpus_key="train-other-960",
        )
        matched = DecodeConfig(
            centroids=centroids,
            model_dir=model_dir,
            recog_rasr_config=bonus._recog_config(
                bonus.MATCHED_DECODE_LM, bonus.FREE_SEG_LOOP, 1.0, use_fb=False),
            distance_scale=bonus.AM_SCALE,
        )
        free = decode_and_score(
            f"{name}_ep-{epoch}_freeseg-lm-{bonus.MATCHED_DECODE_LM}", "cv", matched,
            cv_frames_dataset, rasr_path=tools.RASR_PATH, device="cpu",
            corpus_key="train-other-960",
        )
        values = {
            "matched_per": free.per,
            "matched_del": free.deletion,
            "matched_ins": free.insertion,
            "matched_sub": free.substitution,
            "matched_hyp_ref": 1 - free.deletion + free.insertion,
        }
        return res, values

    topology_columns = [
        statistic("topology_p_global", ("EM loop", "prob."), fmt="{:.4f}"),
        statistic("topology_lambda_global", ("EM", "lambda"), fmt="{:+.3f}"),
        statistic("topology_mean_duration", ("Exp. dur.", "(frames)"), fmt="{:.2f}"),
    ]
    report = LatexTableReport(
        columns=[param("topology", header=("Topo-", "logy")), "epoch", "per", "del", "ins",
                 "sub", "matched_per", "matched_del", "matched_ins", "matched_sub",
                 "matched_hyp_ref", "log_likelihood", "posterior_entropy",
                 "segment_duration_argmax", "boundary_rate"] + topology_columns,
        sort_by=["topology"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Trained loop probabilities: EM on the topology from epoch {BRANCH_EPOCH + 1}, "
            f"global or per phoneme, against the run they branch from "
            f"(vq\\_gmm\\_tau\\_bonus, zero bonus, frame codebook: S={bonus.NUM_STATES}, "
            f"LM {bonus.LM_SCALE}, am=ts={bonus.AM_SCALE}, p={p:g} fixed). The decodes use "
            f"RASR's fixed loop settings, so a trained p reaches none of them. The EM "
            f"columns are the global estimate from the counts of the epoch after the "
            f"decoded model, and the expected frames per phoneme under it."
        ),
    )
    progress_report = TextTableReport(
        columns=[param("topology"), "epoch", "log_likelihood", "posterior_entropy",
                 "segment_duration_argmax", "boundary_rate", "dead_clusters"] + topology_columns,
        sort_by=["topology"],
        drop_empty_rows=True,
        title=(
            f"Per-epoch convergence of the trained topology (rows are the epoch jobs, "
            f"which recognize with the previous epoch's model and table). EM loop prob. "
            f"is what the global update sets from that epoch's counts, 1 - S/d for d the "
            f"expected frames per phoneme (Exp. dur.; the alignment has 3.99). "
            f"Measured before the run: one global update from 0.5 gives 0.592 "
            f"(lambda +0.131) on a sample; expected to drift toward ~+0.2."
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
            if e > BRANCH_EPOCH:
                progress_report.add_row(params=dict(params), epoch=e, statistics=progress)
        for e in DECODE_EPOCHS:
            res, values = decode(name, models[e], e)
            tk.register_output(f"guided_kmeans/{exp_dir}/per/{name}_ep-{e}_per", res.per)
            recog_results.append(res)
            report.add_row(result=res, params=dict(params), epoch=e, statistics=stats,
                           values=values)

    add_rows({"topology": f"fixed p={p:g}"}, reference_name, reference.out_models,
             {int(e): path for e, path in reference.out_epoch_statistics.items()})

    for tying in TYINGS:
        name = (f"{reference_name}_branch-{BRANCH_EPOCH}_em-{tying}"
                f"_floor-{int(MIN_FRAMES)}")
        branch = clustering(
            BRANCH_EPOCHS,
            flavor(
                reference.out_artifacts["centroids"][BRANCH_EPOCH],
                reference.out_artifacts["covs"][BRANCH_EPOCH],
                reference.out_artifacts["mixtures"][BRANCH_EPOCH],
                search(train_topology=tying, topology_min_frames=MIN_FRAMES),
            ),
            f"guided_kmeans/{exp_dir}/{name}",
        )
        # The branch counts its epochs from 1; the rows use the run's numbering.
        models = {BRANCH_EPOCH + int(e): path for e, path in branch.out_models.items()}
        epoch_statistics = {BRANCH_EPOCH + int(e): path
                            for e, path in branch.out_epoch_statistics.items()}
        for e, topology in branch.out_topologies.items():
            tk.register_output(
                f"guided_kmeans/{exp_dir}/topology/{name}/epoch_{BRANCH_EPOCH + e:03d}",
                topology)
        tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{name}.json",
                           branch.out_statistics)
        add_rows({"topology": f"EM {tying}"}, name, models, epoch_statistics)

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
