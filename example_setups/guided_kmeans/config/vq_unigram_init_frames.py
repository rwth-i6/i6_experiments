"""The analytic first EM step, transferred to the unsegmented case.

:mod:`.vq_unigram_init` initializes a segmented VQ run with the positional
unigram of the label prior instead of a random draw, and it works: **13.3% PER
at 20 epochs and 12.2% at 40**, past the 14.4% of a table counted from a
reference alignment, with the table's I(label; codeword) going 0.0009 -> 2.98
nats. Epoch 0 decodes at 86.6%, so none of that is in the initialization itself -
it is in how fast the trajectory leaves it.

That run had one label per frame. Here a label spans several frames, and the
question the addendum answers is what ``gamma`` even means then.

What changes, and what does not
    §8.1: the duration model factors out into an **alignment kernel**

        gamma_t(c) = sum_i A[t, i] * delta(c_i = c),
        A[t, i]    = p(frame t belongs to token i | total length T)

    so the label sequence enters only through the one-hot incidence, and the
    entire duration model lives in ``A``. With geometric durations every
    composition of ``T`` into ``L`` positive parts is equally likely once the
    total is fixed - the likelihood ``a^(T-L) (1-a)^L`` does not depend on which
    composition it is - so ``A`` is the *same hypergeometric* the segmented case
    already used, with ``S -> L``, and **the self-loop probability cancels**.
    Nothing has to be fitted, measured, or matched to ``lambda``.

    Two things do change. Bands go by token count, not frame count (§9.2): the
    profile width is set by the chain length, and banding by frames would put
    segments of very different resolution in one cell. And the codeword
    histogram has to be banded the same way, which is why it now takes the token
    sequences too.

Sub-states, and why m = 3 (§10.1)
    A geometric duration has its mode at 1 frame and a monotonically decreasing
    pmf, which no phone does. Splitting each token into ``m`` sub-states with
    tied self-loops makes the duration negative binomial - unimodal, mode away
    from 1 - and costs nothing: build the kernel at chain length ``m*L`` and sum
    each token's ``m`` columns back. ``a`` still cancels.

Resolution, measured on this corpus rather than assumed
    The addendum quotes ``sd_tau ~ 1/(2 sqrt(mL))`` and a floor at ``L ~ 16``.
    That is the ``mL << T`` limit, and **this corpus is not in it**: ls-100h
    frames run at 13.81M frames over 28,234 sequences against 3.52M transcript
    phonemes, i.e. ``T/L = 3.92``. At ``m = 3`` the sub-state chain covers 77% of
    the frames, the exact width carries a ``sqrt(1 - (mL-1)/(T-1))`` factor the
    asymptotic drops, and the profiles come out **about twice as sharp** as the
    formula says:

        m = 1:  sd_tau 0.0385  (asymptotic 0.0447)   ~2.5 bins at B = 66
        m = 3:  sd_tau 0.0125  (asymptotic 0.0258)   ~0.8 bins at B = 66

    So resolution is not the binding constraint here - the opposite. At ``m = 3``
    the kernel is narrower than one bin, which is close to §10.3's deterministic
    limit: maximal rank in ``G``, but overconfident, and it puts hard zeros where
    the duration model is only a guess. Hence both are run, and ``m`` is the one
    knob this config sweeps.

Cost
    An unsegmented epoch is ~24-30 h against 0.65 h per chunk task segmented,
    because the search has to place a boundary at every frame. The scheduling
    here is copied wholesale from :mod:`.vq_unsupervised_frames` - four table
    updates per pass damped by an EMA whose window is one corpus - so that the
    only difference from those runs is the initialization.
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
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
)
from i6_experiments.example_setups.guided_kmeans.setup.positional_unigram import (
    positional_unigram_table,
)
from i6_experiments.example_setups.guided_kmeans.setup.statistics_jobs import (
    MixtureDiagnosticsJob,
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
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    build_decode_config,
    build_vq_training,
)

# Everything about the run that is not the initialization, so the comparison
# against those runs is row for row.
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised_frames import (
    BATCHES_PER_EPOCH,
    BEAM_SIZE,
    DECODE_BEAM_SIZE,
    DECODE_EPOCHS,
    DECODE_POINT,
    EMA,
    EPOCH_RQMT,
    LM_ORDER,
    LM_SCALE,
    NUM_CHUNKS,
    NUM_EPOCHS,
    NUM_WORKERS,
    RIDGE_EPOCHS,
    RIDGE_POINTS,
    TRANSITION_SCALE,
    loop_prob_for,
)

exp_dir = "vq_unigram_init_frames"
version = 1

NUM_LABELS = 40

#: §9.2: B ~ 4 sqrt(max L), clipped to [32, 128]. The transcripts top out at
#: L = 274 tokens, so 4 * sqrt(274) = 66.
NUM_BINS = 66

#: §10.1's m. 1 is the plain geometric duration model the search itself assumes;
#: 3 is the standard phone-model split and the addendum's recommended default.
#: Both are run because this corpus sits near the deterministic limit at m = 3
#: (see the module docstring) - which the addendum's own §10.3 flags as
#: overconfident - so which side of that trade wins is a measurement.
SUB_STATES = [1, 3]

#: Frames are 4x the segmented corpus, so the one pass that quantizes them gets
#: 4x the chunks. Unhashed.
HISTOGRAM_CHUNKS = 64

#: (lambda, distance_scale) for training. The probe optimum of the random-init
#: runs, held fixed so that m is the only thing varying here. The rest of that
#: config's grid is deliberately not repeated - it placed the operating point
#: and the point is now placed.
TRAIN_SETTINGS = [(1.1, 1.0)]


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)

    # Identical to vq_unsupervised_frames' jobs, so these are shared rather than
    # recomputed: silence removed with the alignment, nothing else collapsed.
    ls100 = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["ls-100"]["features"],
        alignment=GMM_ALIGNMENT_LS960_FRAME,
        exclude_labels=(0,),
        pooling="none",
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

    full_cv = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=[
            "init", "sub_states", "lambda", "scale", "decode_lambda", "decode_scale",
            "epoch", "mi", "per", "del", "ins", "sub", "fer",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["init", "sub_states", "lambda", "scale", "decode_lambda", "decode_scale"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            "Unsegmented discrete-HMM training on silence-free ls-100h frames, "
            "initialized with the analytic first EM step under a geometric duration "
            "model (addendum §9): each transcript token is spread over the frames by "
            "the alignment kernel A[t,i] = p(frame t belongs to token i | length T), "
            "which under geometric durations is the same hypergeometric the segmented "
            "case used and does not depend on the self-loop probability. sub\\_states "
            "is §10.1's m, splitting each token into m tied sub-states to make the "
            "duration negative binomial. The segmented counterpart of this "
            "initialization reaches 12.2\\% PER at 40 epochs against 14.4\\% for a "
            "table counted from a reference alignment. Decoded on silence-free cv "
            "frames at the random-init runs' probe optimum, so the initialization is "
            "the only difference from those."
        ),
    )
    recog_results = []

    def _decode(*, name, model_dir, dataset, lam, scale, epoch, params, statistics, mi):
        dc = DecodeConfig(
            centroids=COLLEAGUE_CENTROIDS_K512,
            model_dir=model_dir,
            recog_rasr_config=build_decode_config(
                None, LM_SCALE, loop_prob_for(lam),
                lm_order=LM_ORDER, beam_size=DECODE_BEAM_SIZE,
                transition_scale=TRANSITION_SCALE,
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
        latex_report.add_row(
            result=res,
            params={**params, "decode_lambda": lam, "decode_scale": scale},
            epoch=epoch,
            statistics=statistics,
            values={k: v for k, v in (("mi", mi), ("fer", res.fer)) if v is not None},
        )

    for sub_states in SUB_STATES:
        # --- the initialization: transcripts, one quantization pass, one GEMM --
        # No search anywhere in here, and no epoch.
        unigram = positional_unigram_table(
            features_hdf=ls100.out_features,
            centroids=COLLEAGUE_CENTROIDS_K512,
            lexicon=lexicon,
            num_labels=NUM_LABELS,
            corpus_key="train-clean-100",
            segments=ls100.out_segments,
            num_bins=NUM_BINS,
            num_chunks=HISTOGRAM_CHUNKS,
            token_mode=True,
            sub_states=sub_states,
            # §9.5: the kernel already smooths, so §3.3 gives way. At m = 3 its
            # width is 0.8 bins and at m = 1 it is 2.5, so neither wants the base
            # spec's 1.0 and the addendum's 0.5 covers both.
            sigma_bins=0.5,
            alias_prefix=f"guided_kmeans/{exp_dir}/m-{sub_states}",
        )
        for label, path in (
            ("band_edges", unigram.band_edges),
            ("gamma_diagnostics.json", unigram.gamma_diagnostics),
            ("table_diagnostics.json", unigram.table_diagnostics),
            ("length_agreement.json", unigram.length_agreement),
            ("profiles", unigram.plots),
            ("table.npy", unigram.table),
        ):
            tk.register_output(f"guided_kmeans/{exp_dir}/m-{sub_states}/{label}", path)
        tk.register_output(
            f"guided_kmeans/{exp_dir}/m-{sub_states}/table_mi.json",
            MixtureDiagnosticsJob(unigram.table).out_diagnostics,
        )

        for lam, scale in TRAIN_SETTINGS:
            name = f"m-{sub_states}_lambda-{lam}_scale-{scale}"
            _, result = build_vq_training(
                features=ls100.out_features,
                lm_path=None,
                table=unigram.table,
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
            tk.register_output(
                f"guided_kmeans/{exp_dir}/statistics/{name}.json", result.out_statistics
            )
            statistics = clustering_statistics_per_epoch(
                result.out_epoch_statistics, name=name, epoch_offset=1, lexicon=lexicon
            )
            params = {"init": "unigram", "sub_states": sub_states,
                      "lambda": lam, "scale": scale}
            for epoch in DECODE_EPOCHS:
                mi = MixtureDiagnosticsJob(result.out_artifacts["table"][epoch]).out_mi
                points = [DECODE_POINT] + (RIDGE_POINTS if epoch in RIDGE_EPOCHS else [])
                for decode_lam, decode_scale in points:
                    _decode(
                        name=(
                            f"{name}_ep-{epoch}_dlambda-{decode_lam}"
                            f"_dscale-{decode_scale}"
                        ),
                        model_dir=result.out_models[epoch], dataset=full_cv,
                        lam=decode_lam, scale=decode_scale, epoch=epoch,
                        params=params, statistics=statistics, mi=mi,
                    )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def py():
    run()
