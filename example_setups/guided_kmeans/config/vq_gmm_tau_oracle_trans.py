"""The oracle segmentation through per-frame transition scores: the end-to-end check.

``vq_gmm_tau_oracle_seg`` trains on frames with the GMM alignment's segmentation,
by summing each segment into one observation (``OracleSegmentFBRecognizer``).
That trick needs no transition input, but it is limited to one HMM state and a
hard oracle. The ``backoff_fb`` op now takes per-frame transition scores (its
SPEC.md section 14, milestones T1-T2), and ``OracleBoundaryTransitions`` states
the same oracle through them: an exit is forced on the last frame of every
alignment segment and forbidden on every other frame. The graph uses the zero
topology, so these scores are the whole transition model, and the search runs on
the frames themselves.

**The check.** At one state with hard boundaries the two routes are the same
model (SPEC.md 14.1, consequence 4), so they have the same posteriors and the
same log Z, to rounding. Measured on 71 ls-100 sequences (35,252 frames) with
the 5-gram and the am-0.35 epoch-40 model, on an L40S: in float64 storage the
two agree to 6.5e-13 in Gamma and 1.7e-15 relative in log Z; in float32, the
storage training uses, to 1.1e-4 and 7e-9. The per-frame route costs 2.5x the
segment route's search time (25 s against 10 s there), because it steps through
four times as many observations.

This config is the same comparison through training: the am-0.35 arm of
``vq_gmm_tau_oracle_seg`` again, through the new route, for ``CHECK_EPOCHS``
epochs, next to the existing run. The existing run is rebuilt here with exactly
the arguments it was run with, so its jobs are reused rather than recomputed,
and its rows are already filled. Expect the two rows of every epoch to agree
in every statistic, up to the drift that float32 storage of per-frame rather
than summed scores introduces. A difference beyond that is a bug in the new
route.

**What it does not test.** More than one state. With hard boundaries, a segment
shorter than S frames has no path. 8% of the alignment's segments are one frame
long, and 99.1% of ls-100 utterances contain at least one, so a hard oracle at
S = 2 trains on less than 1% of the corpus (all 71 sequences of the check
above die). An S > 1 arm needs ``OracleBoundaryTransitions(boundary_error >
0)``, which lets a path move, drop or add a boundary at a price. Which error rate to use is an experimental
choice and is not made here.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
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
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_oracle_seg as seg
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_unseg as unseg

exp_dir = "vq_gmm_tau_oracle_trans"
version = 1

#: The arm both routes run. 0.35 is the one vq_gmm_tau_oracle_seg runs its
#: attribution decodes on.
AM_SCALE = 0.35
#: Enough for a trajectory to diverge if the routes differ, short of the 60 the
#: reference ran. The two routes share every job input except the search, so a
#: longer run would compare the same thing at more cost.
CHECK_EPOCHS = 10
DECODE_EPOCHS = [5, 10]


def _arm(*, per_frame: bool, ls100, covs, init_table, train_config, lexicon, fb_lexicon,
         num_epochs: int = CHECK_EPOCHS):
    """One arm of vq_gmm_tau_oracle_seg at AM_SCALE, through either route.

    ``per_frame=False`` must stay argument for argument what that config runs,
    so the existing jobs are reused. The two routes differ only in the
    BackoffFBSearch, and that is the whole point. The scheduling knobs
    (batching, rqmt) are unhashed: the per-frame route takes the frame-level
    ones of vq_gmm_tau_unseg, whose chunks do the same op work.
    """
    search = dict(
        backoff_fb_root=tools.BACKOFF_FB,
        arpa_path=phonetic_lm_dict[seg.LM_ORDER],
        lexicon=fb_lexicon,
        lm_scale=1.0,
        emission_scale=1.0,
        transition_scale=1.0,
        apply_sentence_end=True,
        checkpoint_interval=-1,
        storage_dtype="float32",
    )
    if per_frame:
        # 18,000, not the oracle config's 4,000: T is now frames, and at 4,000
        # the long batches would collapse to B = 3 (see vq_gmm_tau_unseg).
        search.update(batch_size=unseg.BATCH_SIZE, max_frames=unseg.MAX_FRAMES)
    else:
        search.update(batch_size=seg.BATCH_SIZE, max_frames=seg.MAX_FRAMES)
    if per_frame:
        backoff_fb = BackoffFBSearch(
            **search,
            transitions=Spec(OracleBoundaryTransitions, {"frame_labels": ls100.out_labels}),
            zero_topology=True,
        )
    else:
        backoff_fb = BackoffFBSearch(
            **search,
            frame_labels=ls100.out_labels,
            loop_probability=0.0,
            silence_loop_probability=0.0,
        )
    flavor = mixture_flavor(
        centroids=COLLEAGUE_CENTROIDS_K512,
        covs=covs,
        mixtures=init_table,
        recognition_config=train_config,
        lexicon=lexicon,
        num_clusters=seg.NUM_LABELS,
        distance_scale=AM_SCALE,
        use_forward_backward=True,
        update_densities=False,
        update_covariances=None,
        mixture_floor=1e-2,
        num_workers=seg.NUM_WORKERS,
        backoff_fb=backoff_fb,
    )
    name = f"weights_tau-{seg.TAU}_am-{AM_SCALE}"
    return name, chunked_clustering(
        num_epochs=num_epochs,
        features_hdf=ls100.out_features,
        recognition_config=train_config,
        lexicon=lexicon,
        num_clusters=seg.NUM_LABELS,
        flavor=flavor,
        distance_scale=AM_SCALE,
        use_forward_backward=True,
        rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
        num_chunks=seg.NUM_CHUNKS,
        num_workers=seg.NUM_WORKERS,
        parent_threads=seg.PARENT_THREADS,
        # A per-frame chunk runs ~1.4 h on an L40S, which the 2 h default time
        # limit would cut on a slower card; vq_gmm_tau_unseg's rqmt asks for 4.
        rqmt=unseg.EPOCH_RQMT if per_frame else seg.EPOCH_RQMT,
        # The reference keeps the alias tree it already has.
        alias_prefix=f"guided_kmeans/{exp_dir if per_frame else seg.exp_dir}/{name}",
    )


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    ls100 = seg.frame_level_ls100()
    cv = silence_free_cv_features()

    # As in vq_gmm_tau_oracle_seg, so both arms share these jobs with it.
    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    covs = ScaleCovsJob(DuplicateCovsJob(global_cov, seg.NUM_CODEWORDS).out_covs, seg.TAU).out_covs
    init_table = NormalTableJob(
        seg.NUM_LABELS, seg.NUM_CODEWORDS, sigma=seg.SIGMA, seed=seg.SEED
    ).out_table
    train_config = seg._recog_config(1.0, 0.0, 1.0, use_fb=True)
    decode_config_rasr = seg._recog_config(1.0, 0.0, None, use_fb=False)
    cv_dataset = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )

    report = TextTableReport(
        columns=[param("route", header=("Route",)), "epoch", "per", "log_likelihood",
                 "log_likelihood_gain", "posterior_entropy", "cluster_perplexity",
                 "max_cluster_fraction", "segment_duration_argmax", "boundary_rate",
                 "dead_clusters", "l1"],
        sort_by=["route"],
        drop_empty_rows=True,
        title=(
            f"The oracle segmentation by two routes, tau={seg.TAU}, am={AM_SCALE}: "
            f"'segments' sums each alignment segment into one observation "
            f"(vq_gmm_tau_oracle_seg, reused), 'per-frame' runs the search on frames "
            f"with hard boundaries as per-frame transition scores on the zero "
            f"topology. The same model at one HMM state, so the two rows of an epoch "
            f"must agree up to float32 rounding. Row k is model k: the statistics "
            f"of the pass that recognized with it (k + 1), and at the decode epochs "
            f"its PER on mean-pooled cv, decoded as in vq_gmm_tau_oracle_seg."
        ),
    )
    recog_results = []
    for route, per_frame in (("segments", False), ("per-frame", True)):
        name, result = _arm(
            per_frame=per_frame, ls100=ls100, covs=covs, init_table=init_table,
            train_config=train_config, lexicon=lexicon, fb_lexicon=fb_lexicon,
        )
        if per_frame:
            tk.register_output(
                f"guided_kmeans/{exp_dir}/statistics/{name}.json", result.out_statistics
            )
        # epoch_offset=1: row k is model k, beside the statistics of the pass
        # that recognized with it (pass k + 1), as vq_gmm_tau_oracle_seg's
        # decode table reads them. The last model has no such pass.
        stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=f"{name}_{route}", epoch_offset=1,
            lexicon=lexicon,
        )
        decodes = {}
        for e in DECODE_EPOCHS:
            dc = DecodeConfig(
                centroids=COLLEAGUE_CENTROIDS_K512,
                model_dir=result.out_models[e],
                recog_rasr_config=decode_config_rasr,
                distance_scale=seg.BASE_DECODE_SCALE,
                write_frame_labels=True,
            )
            res = decode_and_score(
                f"{name}_{route}_ep-{e}_ds-{seg.BASE_DECODE_SCALE}", "cv", dc, cv_dataset,
                rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960",
            )
            decodes[e] = res
            recog_results.append(res)
        for epoch in range(0, CHECK_EPOCHS + 1):
            report.add_row(
                result=decodes.get(epoch), params={"route": route}, epoch=epoch,
                statistics=stats,
            )

    report.register(
        f"guided_kmeans/{exp_dir}/comparison_{version}.txt",
        required=report, update_frequency=60,
    )


def py():
    run()
