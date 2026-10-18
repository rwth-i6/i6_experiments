"""The tau=4.0 arm again, trained on *frames* with the segmentation given.

``vq_gmm_tau`` scores one mean-pooled vector per oracle segment. This trains on
every frame instead, with the same oracle telling the search where the segments
are. The model is identical (frozen k512 codebook, tau=4, weights-only), the LM
is identical, the decode path is identical -- only what the acoustic model
*sees* changes: 13.8M frames rather than 3.46M means.

Why this needs no new kernel. With one HMM state per phoneme, fixing the
boundaries leaves a path free only to choose which phoneme each segment holds,
so its score over a segment ``[b, e)`` is

    sum_{t in [b,e)} o_t(a)  +  log P(a | history)

and that is exactly what the op computes from one observation whose score is
that sum. Brute-forcing the frame-level trellis against the oracle on summed
scores agrees to 2e-16. :class:`OracleSegmentFBRecognizer` therefore sums the
per-frame costs per segment, runs the op unchanged, and broadcasts each
segment's posterior back over its frames -- which is the point, because the
accumulator then weights every frame by its segment's posterior instead of
seeing one mean.

**The AM scale is the whole sweep.** Summing ~4 frames per segment multiplies
the acoustic score by ~4, so the 1.0 that ``vq_gmm_tau`` trains at would swamp
the LM here and the run would look catastrophic for a reason that has nothing to
do with the hypothesis. 1/3.99 = 0.25 is the scale that holds the AM/LM balance
where the reference had it, and the sweep brackets it. Note it is not a pure
rescale: long segments now carry proportionally more acoustic weight than short
ones, which is exactly what frame-level modelling should do.

What to compare against: ``vq_gmm_tau`` (RASR search, pooled features) and
``vq_gmm_tau_cuda`` (same op, pooled features). Against the latter, this run
differs *only* in frames-vs-means, so the difference is attributable.

Two limits of the trick, both wanting the per-frame transition tensor the op
does not have, and both deferred:

* a **finite** boundary penalty, to measure sensitivity to boundary error. The
  equivalence needs a hard oracle.
* **S > 1**. A segment would span several states with distinct emissions, which
  summing collapses.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
    GMM_ALIGNMENT_LS960_FRAME,
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
from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
    SegmentedFeaturesFromAlignmentJob,
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
from i6_experiments.example_setups.guided_kmeans.setup.emission_balance import (
    BalanceFramesJob,
    EmissionBalanceJob,
    PhonemeLMCostJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import (
    LatexTableReport,
    TextTableReport,
    clustering_statistics_per_epoch,
    param,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
)

exp_dir = "vq_gmm_tau_oracle_seg"
version = 1

# Carried over from vq_gmm_tau unchanged.
LM_ORDER = 5
SIGMA = 0.02
BEAM_SIZE = 1_000
NUM_LABELS = 40
NUM_CODEWORDS = 512
SEED = 42
TAU = 4.0
WEIGHTS_EPOCHS = 60
#: Decoded early as well as late: the sweep's job is to pick an AM scale, and
#: that ordering is usually clear by epoch 10-20. Epochs are a chain, so
#: stopping the losers there costs nothing already spent.
WEIGHTS_DECODE_EPOCHS = [5, 10, 20, 30, 40, 50, 60]

#: Training AM scales. 0.25 = 1/3.99 is where the AM/LM balance sits at the
#: reference's 1.0 once ~4 frames are summed per segment; the rest bracket it.
# TRAIN_DISTANCE_SCALES = [0.15, 0.25, 0.35, 0.5]
TRAIN_DISTANCE_SCALES = [0.25, 0.35, 0.5, 0.65, 0.75, 1.0]

#: Epochs at which EmissionBalanceJob measures each arm's model, for the
#: convergence table. This run is the reference for the unsegmented ones: its
#: boundary payback (what the acoustics would gain by placing a true boundary)
#: is the trajectory a free-segmentation run has to match, so it is measured
#: densely while the classes form and sparsely after.
BALANCE_EPOCHS = [1, 3, 5, 8, 10, 13, 15, 20, 25, 30, 35, 40, 50, 60]

#: The converged model of the am-0.35 arm, decoded on cv *frames* with free
#: segmentation at each of vq_gmm_tau_lm's LM scales, with that sweep's
#: free-segmentation decode: the standard decode routine on frames, loop 1/2
#: (no per-phoneme price besides the LM's), training AM scale, the same seeded
#: 10% of cv. This model learned per-phoneme classes; the grid shows how much of
#: what it loses to a free segmentation is the LM's price alone. It is the
#: reference for vq_gmm_tau_lm's attribution table, which runs the same grid on
#: the unsegmented control model.
ATTRIBUTION_ARM_SCALE = 0.35
ATTRIBUTION_EPOCH = 40
ATTRIBUTION_LM_SCALES = [1.0, 0.5, 0.25, 0.1]
FREE_SEG_LOOP = 0.5
FREE_SEG_FRACTION = 0.1

#: Decode is unchanged from vq_gmm_tau -- mean-pooled cv, linear search -- so
#: PER is directly comparable to it. The frozen codebook is what makes that
#: defensible: only the 40x512 table is learned here, and it is a table over
#: the same densities either way.
BASE_DECODE_SCALE = 1.0
DECODE_DISTANCE_SCALES = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
SCALE_SWEEP_EPOCHS = [30, 60]

#: Chunk count is set by how many GPUs this account can actually hold at once,
#: which is a per-user QOS limit and not the partition's size:
#:
#:   gpu_48gb  16 L40S, but gpu_48gb_qos has MaxTRESPU gres/gpu=5
#:   gpu_32gb  32 V100, gpu_32gb_qos has no per-user limit at all
#:
#: so the 48 GB partition is the faster card and the 32 GB one is the bigger
#: allowance. Neither alone is enough for a four-arm sweep, hence the explicit
#: two-partition ``-p`` below; Slurm places each chunk on whichever frees first.
#: Every log_* partition, including the A100s in log_gpu_80gb, is
#: AllowAccounts=log while this setup runs under hlt, so a job sent there is
#: rejected at submit with (PartitionConfig) - they are not an option.
#:
#: With ~8 slots reachable in practice, 4 arms x 6 chunks = 24 tasks runs in
#: three waves of ~1.1 h, against two waves of ~2.1 h at 3 chunks. Going past 6
#: is counterproductive: each task cf-caches the 28.5 GB frame-level feature
#: file (~4 min, measured) and builds the LM graph (~85 s), so at 6 chunks that
#: fixed cost is already 8% of a task.
NUM_CHUNKS = 6
NUM_WORKERS = 0
PARENT_THREADS = 4
#: The working set is ~4 GB, so nothing here needs 32 or 48 GB of device memory
#: - the partition choice is about throughput and allowance, and an explicit
#: ``-p`` is the only way to ask for two partitions. It also bypasses the
#: gpu_mem ladder in settings.py:check_engine_limits (which honours an
#: sbatch_args that already contains "-p"), so gpu_mem is deliberately absent:
#: it would have no effect and would only suggest otherwise.
#:
#: On card choice: the L40S is the faster of the two for this op at 1.320
#: ms/frame against the V100's 1.670 (an A10 measures 3.174). That is contrary
#: to what the FP64 column suggests - the L40S runs FP64 at 1:64 and the V100 at
#: 1:2 - because the op is bandwidth-bound rather than double-precision-bound.
#: A 27% slower card that can be held in unlimited numbers still wins on
#: aggregate throughput here, which is why both are listed rather than just one.
EPOCH_RQMT = {
    "mem": 16,
    "gpu": 1,
    "cpu": 4,
    "sbatch_args": ["-p", "gpu_48gb,gpu_32gb"],
}

BATCH_SIZE = 16
MAX_FRAMES = 4_000


def frame_level_ls100():
    """ls-100h frames with silence dropped, and the per-frame labels that give
    the segmentation. Same job ``vq_unsupervised_frames`` builds, so it is
    shared rather than recomputed."""
    job = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["ls-100"]["features"],
        alignment=GMM_ALIGNMENT_LS960_FRAME,
        exclude_labels=(0,),
        pooling="none",
        rqmt={"cpu": 2, "mem": 16, "time": 12},
    )
    return job


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

    # The covariance is the one tau scales, computed on the features the run
    # uses -- frames here, so tau=1 means "the shape of the frame distribution"
    # rather than of the segment means. That is the honest choice even though it
    # makes tau not quite the same quantity as in vq_gmm_tau.
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
    train_config = _recog_config(1.0, 0.0, 1.0, use_fb=True)
    decode_config_rasr = _recog_config(1.0, 0.0, None, use_fb=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)

    # Emission vs LM (setup.emission_balance). Here there is no boundary decision
    # to make, so the price is the hypothetical one -- one more phoneme at this
    # run's LM scale, no transition term -- and the number answers "could this
    # model hold the true segmentation if it had to". Both prep jobs are shared
    # with vq_gmm_tau_lm, which builds them over the same features and densities.
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

    cv_frames = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["cv"]["features"],
        alignment=GMM_ALIGNMENT_CV,
        exclude_labels=(0,),
        pooling="none",
    ).out_features
    # Built exactly as in vq_gmm_tau_lm, so both grids decode the same subset.
    free_seg_segments = ShuffleAndSplitSegmentsJob(
        cv.out_segments,
        split={"decode": FREE_SEG_FRACTION, "rest": 1.0 - FREE_SEG_FRACTION},
    ).out_segments["decode"]
    cv_frames_dataset = DatasetConfig(
        audio_hdf_path=cv_frames,
        sampling_method=SegmentFile(free_seg_segments),
        precomputed=True,
    )
    attribution_report = TextTableReport(
        columns=[param("decode_lm", header=("Decode", "LM scale")), "matched_per",
                 "matched_del", "matched_ins", "matched_sub", "matched_hyp_ref"],
        sort_by=["decode_lm"],
        drop_empty_rows=True,
        title=(
            f"Attribution reference: the converged oracle-segmented model "
            f"(am {ATTRIBUTION_ARM_SCALE}, epoch {ATTRIBUTION_EPOCH}) decoded "
            f"with free segmentation (standard decode on cv frames, loop "
            f"{FREE_SEG_LOOP}, {int(100 * FREE_SEG_FRACTION)}% of cv) at each LM "
            f"scale of vq_gmm_tau_lm. Compare with that sweep's attribution table "
            f"for the unsegmented control model."
        ),
    )

    latex_report = LatexTableReport(
        # Same diagnostic set as vq_gmm_tau_unseg, so the two tables can be read
        # side by side -- this is the run that converged, and it is only useful as
        # a reference for the one that did not if the columns match.
        columns=["am_scale", "epoch", "per", "del", "ins", "sub",
                 "log_likelihood", "posterior_entropy",
                ],
        sort_by=["am_scale"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Trained on silence-free ls-100h *frames* with the segmentation taken "
            f"from the GMM alignment, swept over the training AM scale. Summing "
            f"$\\sim$4 frames per segment scales the acoustic score by the same "
            f"factor, so 0.25 is where the AM/LM balance matches 1.0 for the "
            f"segmented experiments. Decoded on mean-pooled cv for backward comparability."
        )
    )
    recog_results = []

    # Statistics per recognition pass, not per decode. The table above only has
    # rows at WEIGHTS_DECODE_EPOCHS, and with epoch_offset=1 its last row reads a
    # key the run never produces, so the diagnostics are invisible for most
    # epochs and always for the final one. epoch_offset=0 here: a row labelled
    # epoch k shows pass k.
    #
    # This run is also where the duration counters are checked against a known
    # answer. OracleSegmentFBRecognizer broadcasts each segment's posterior back
    # over its frames, so adjacent frames inside a segment are identical and the
    # measured duration must reproduce the oracle segmentation -- 3.99 frames at
    # a boundary rate of 0.251. It does: the argmax estimate comes back
    # 4.000-4.025 (the excess is adjacent segments that happen to share a label,
    # which merges two runs into one), and the boundary rate 0.2500.
    progress_report = TextTableReport(
        columns=["am_scale", "epoch", "log_likelihood", "log_likelihood_gain",
                 "posterior_entropy", "cluster_perplexity",
                 "max_cluster_fraction", "loop_frequency", "segment_duration",
                 "segment_duration_argmax", "boundary_rate", "dead_clusters",
                 "l1", "segment_gap", "boundary_payback", "boundary_price",
                 "payable_boundaries"],
        sort_by=["am_scale"],
        drop_empty_rows=True,
        title=(
            f"Per-epoch convergence for the tau={TAU} oracle-segmented arms, and "
            f"the reference check on the duration counters: with the "
            f"segmentation given, the argmax duration must come back at the "
            f"oracle's own 3.99 frames and boundary rate 0.251. The soft "
            f"estimate is biased low by construction and the bias tracks the "
            f"posterior entropy -- about -8% at entropy 0.06, about -50% at "
            f"0.85 -- so read the argmax column where the posteriors are soft. "
            f"Duration columns are blank for passes that predate the counters."
        ),
    )

    for am_scale in TRAIN_DISTANCE_SCALES:
        name = f"weights_tau-{TAU}_am-{am_scale}"
        backoff_fb = BackoffFBSearch(
            backoff_fb_root=tools.BACKOFF_FB,
            arpa_path=phonetic_lm_dict[LM_ORDER],
            lexicon=fb_lexicon,
            # The per-frame alignment is what turns this into the oracle-segment
            # recognizer; the features have to be the matching pooling="none".
            frame_labels=ls100.out_labels,
            lm_scale=1.0,
            emission_scale=1.0,
            loop_probability=0.0,
            silence_loop_probability=0.0,
            transition_scale=1.0,
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
            f"guided_kmeans/{exp_dir}/statistics/{name}.json", result.out_statistics
        )
        stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=name, epoch_offset=1, lexicon=lexicon
        )
        # Same extraction jobs as `stats`; only the row-to-key alignment differs.
        progress_stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=name, epoch_offset=0, lexicon=lexicon
        )
        balance = {}
        for e in BALANCE_EPOCHS:
            if e not in result.out_artifacts["mixtures"]:
                continue
            job = EmissionBalanceJob(
                frames=balance_frames,
                mixtures=result.out_artifacts["mixtures"][e],
                am_scale=am_scale,
                lm_cost=lm_cost,
                lm_scale=1.0,
                transition_penalty=0.0,
            )
            job.add_alias(f"guided_kmeans/{exp_dir}/{name}/balance/epoch_{e:03d}")
            balance[e] = {
                "segment_gap": job.out_segment_gap,
                "boundary_payback": job.out_payback,
                "boundary_price": job.out_price,
                "payable_boundaries": job.out_payable,
            }
        for epoch_key in sorted(int(k) for k in result.out_epoch_statistics):
            progress_report.add_row(
                params={"am_scale": am_scale}, epoch=epoch_key,
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
                        f"guided_kmeans/{exp_dir}/eval/{decode_name}_fer", res.fer
                    )
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per
                )
                recog_results.append(res)
                if scale == BASE_DECODE_SCALE:
                    latex_report.add_row(
                        result=res, params={"am_scale": am_scale}, epoch=e,
                        statistics=stats,
                        values={"fer": res.fer} if res.fer is not None else {},
                    )

        if am_scale == ATTRIBUTION_ARM_SCALE:
            for decode_lm in ATTRIBUTION_LM_SCALES:
                dc = DecodeConfig(
                    centroids=COLLEAGUE_CENTROIDS_K512,
                    model_dir=result.out_models[ATTRIBUTION_EPOCH],
                    recog_rasr_config=_recog_config(decode_lm, FREE_SEG_LOOP, 1.0, use_fb=False),
                    distance_scale=am_scale,
                )
                decode_name = f"{name}_ep-{ATTRIBUTION_EPOCH}_freeseg-lm-{decode_lm}"
                res = decode_and_score(
                    decode_name, "cv", dc, cv_frames_dataset,
                    rasr_path=tools.RASR_PATH, device="cpu",
                    corpus_key="train-other-960",
                )
                tk.register_output(f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per)
                attribution_report.add_row(
                    params={"decode_lm": decode_lm},
                    values={
                        "matched_per": res.per,
                        "matched_del": res.deletion,
                        "matched_ins": res.insertion,
                        "matched_sub": res.substitution,
                        "matched_hyp_ref": 1 - res.deletion + res.insertion,
                    },
                )

    recog_report = create_report(recog_results)
    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=recog_report, required=recog_report, update_frequency=60,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex", required=latex_report, update_frequency=60)
    # Plain-text twin of the same table; see the note in vq_gmm_tau_unseg.
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
