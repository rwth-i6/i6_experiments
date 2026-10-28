"""Over-relaxed online EM from the bonus-0.5 arm's epoch 100: can the plateau pass faster?

``docs/free_segmentation/findings_and_directions.md`` §7 (item 1b).

``vq_gmm_tau_online`` showed that four undamped updates per pass track the
full-batch run update for update (within 0.1 PER points to update 135), so an
update now costs a quarter of a pass. This asks whether each update can also go
further. The mixture M-step steps ``eta`` times along the EM update, in log
space per label row (``MixtureGaussianAccumulator(over_relaxation=eta)``):

    w <- normalize(w_prev**(1 - eta) * w_em**eta)

Linearized, EM shrinks the error along direction i by a factor ``l_i`` per update,
the fraction of missing information there; over-relaxation makes it
``1 - eta*(1 - l_i)``. The slow directions - the plateau, a drift the objective
barely registers - move ``eta`` times faster, and every direction stays stable for
``eta < 2``. Above that, only directions with ``l_i > 1 - 2/eta`` are stable, so a
larger eta works only if EM is slow everywhere. On a weights-only toy EM with
overlapping densities (all directions slow) the speed-up was exactly eta up to
3, monotone; 5 diverged.

**Two arms, one difference each from the online run.** Same branch point, same
quarters, same undamped steps; the M-step over-relaxed at

* ``eta = 1.5``: inside the guaranteed range, expected 1.5x;
* ``eta = 3.0``: outside it - 3x if no direction of this table is fast, an
  oscillation (step-to-step sign changes, a growing ``mixture_kl_em_step``)
  within a few updates if one is.

No damping: an EMA moves the slow direction ``eta*(1-a) / (1-a + eta*a)`` times
as fast as undamped EM, which undoes most of the extrapolation, and the pipeline
refuses the combination (``chunked_clustering``).

**Reading it.** By update against the full-batch reference, whose epochs and
decodes run to 200 (61.3% at 100, 38.2% at 160, 11.8% at 200); the online run
equals it to 0.1 points, so it stands for the eta=1 arm too. An arm that is
``eta`` times faster at update 100 + k sits where the reference is at
100 + eta*k. The progress table shows each step's proposed and applied size
(mean KL from the previous table, nats) and the share of weights held at the
floor.

**Only finished jobs of other configs are built**: the reference is
vq_gmm_tau_bonus's run, all 200 epochs finished. The online branch is still
running under its own manager, so its rows are not in this config's tables.

Cost: 40 updates per arm, ~8 GPU-h each - ~640 GPU-h for both, about a day on 32
gpu_11gb cards.
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
    Column,
    LatexTableReport,
    TextTableReport,
    clustering_statistics_per_epoch,
    param,
    relabel,
    statistic,
)
from i6_experiments.example_setups.guided_kmeans.setup.emission_balance import (
    BalanceFramesJob,
    EmissionBalanceJob,
    PhonemeLMCostJob,
    transition_boundary_penalty,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_lm import (
    frame_level_ls100,
)
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_bonus as bonus
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_online as online

exp_dir = "vq_gmm_tau_overrelax"
version = 1

#: The reference's whole run: vq_gmm_tau_bonus finished all of it, so nothing
#: here is a job another manager still drives.
REFERENCE_EPOCHS = 200
REFERENCE_DECODE_EPOCHS = list(range(online.BRANCH_EPOCH, REFERENCE_EPOCHS + 1, 5))

#: ``MixtureGaussianAccumulator.over_relaxation`` per arm (module docstring).
ETAS = [1.5, 3.0]
#: Passes over the corpus per arm; online.BATCHES_PER_EPOCH updates each, so 40
#: updates. Epoch chains do not hash their length: extending resumes.
BRANCH_EPOCHS = 10
DECODE_STEPS = list(range(5, online.BATCHES_PER_EPOCH * BRANCH_EPOCHS + 1, 5))


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    ls100 = frame_level_ls100()
    cv = silence_free_cv_features()

    # Up to the reference run, vq_gmm_tau_bonus's jobs argument for argument
    # (as in vq_gmm_tau_online).
    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    base_covs = DuplicateCovsJob(global_cov, bonus.NUM_CODEWORDS).out_covs
    covs = ScaleCovsJob(base_covs, bonus.TAU).out_covs
    init_table = NormalTableJob(
        bonus.NUM_LABELS, bonus.NUM_CODEWORDS, sigma=bonus.INIT_SIGMA, seed=bonus.SEED
    ).out_table
    centroids = COLLEAGUE_CENTROIDS_K512
    train_config = bonus._recog_config(bonus.LM_SCALE, 0.0, 1.0, use_fb=True)
    decode_config_rasr = bonus._recog_config(1.0, 0.0, None, use_fb=False)

    lam = bonus.bonus_lambda(online.REFERENCE_FRACTION)
    p = bonus.loop_for_lambda(lam, bonus.TRANSITION_SCALE)
    reference_name = (
        f"weights_tau-{bonus.TAU}_s-{bonus.NUM_STATES}_bonus-{online.REFERENCE_FRACTION}"
        f"_loop-{round(p, 4)}_ts-{bonus.TRANSITION_SCALE}_lm-{bonus.LM_SCALE}"
        f"_am-{bonus.AM_SCALE}_init-uniform_cb-{online.REFERENCE_CODEBOOK}")
    reference_epochs_total, reference_chunks = bonus.LONG_ARMS[
        (online.REFERENCE_FRACTION, online.REFERENCE_CODEBOOK)]
    assert REFERENCE_EPOCHS <= reference_epochs_total

    def flavor(centroids, covs, mixtures, over_relaxation=1.0):
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
            over_relaxation=over_relaxation,
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
    branch_start = {
        name: reference.out_artifacts[name][online.BRANCH_EPOCH]
        for name in ("centroids", "covs", "mixtures")
    }

    # Decodes and emission balance as vq_gmm_tau_bonus builds them, so the
    # reference's are that config's finished jobs.
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
    balance_frames = BalanceFramesJob(
        features_hdf=ls100.out_features,
        frame_labels=ls100.out_labels,
        centroids=centroids,
        covs=covs,
    ).out_frames
    lm_cost = PhonemeLMCostJob(
        arpa_path=phonetic_lm_dict[bonus.LM_ORDER],
        lexicon=fb_lexicon,
        backoff_fb_root=tools.BACKOFF_FB,
    ).out_cost
    penalty = transition_boundary_penalty(bonus.NUM_STATES, p, bonus.TRANSITION_SCALE)

    def decode(name, model_dir, label):
        dc = DecodeConfig(
            centroids=centroids,
            model_dir=model_dir,
            recog_rasr_config=decode_config_rasr,
            distance_scale=bonus.BASE_DECODE_SCALE,
            write_frame_labels=True,
            num_workers=bonus.DECODE_WORKERS,
        )
        decode_name = f"{name}_{label}_ds-{bonus.BASE_DECODE_SCALE}"
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
        free = decode_and_score(
            f"{name}_{label}_freeseg-lm-{bonus.MATCHED_DECODE_LM}", "cv", matched,
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

    def balance(mixtures):
        job = EmissionBalanceJob(
            frames=balance_frames,
            mixtures=mixtures,
            am_scale=bonus.AM_SCALE,
            lm_cost=lm_cost,
            lm_scale=bonus.LM_SCALE,
            transition_penalty=penalty,
        )
        return {
            "segment_gap": job.out_segment_gap,
            "boundary_payback": job.out_payback,
            "boundary_price": job.out_price,
            "payable_boundaries": job.out_payable,
        }

    passes = Column("passes", ("Passes", f"since {online.BRANCH_EPOCH}"),
                    lambda row: row.values.get("passes"), fmt="{:g}")
    update = relabel("epoch", "Update")
    step_columns = [
        statistic("mixture_kl_em_step", ("EM step", "KL [nats]"), fmt="{:.2e}"),
        statistic("mixture_kl_applied_step", ("Applied", "KL [nats]"), fmt="{:.2e}"),
        statistic("mixture_clipped_fraction", ("At floor", "[%]"), fmt="{:.1f}", scale=100),
    ]
    report = LatexTableReport(
        columns=[param("rule", header=("Update", "rule")), update, passes, "per", "del",
                 "ins", "sub", "matched_per", "matched_del", "matched_ins",
                 "matched_sub", "matched_hyp_ref", "posterior_entropy",
                 "segment_duration_argmax", "boundary_rate", "segment_gap",
                 "boundary_payback", "payable_boundaries"],
        sort_by=["rule"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Over-relaxed online EM against full-batch EM, by update: "
            f"vq\\_gmm\\_tau\\_bonus's bonus-{online.REFERENCE_FRACTION} arm (S="
            f"{bonus.NUM_STATES}, segment codebook), branched after epoch "
            f"{online.BRANCH_EPOCH} into {online.BATCHES_PER_EPOCH} undamped updates "
            f"per pass, each from its own fixed quarter, with the mixture M-step taken "
            f"eta times in log space. Full-batch rows are the run's own epochs (one "
            f"update per pass); online EM at eta = 1 tracks them to 0.1 points. "
            f"Decoded on mean-pooled cv at LM scale 1.0; the free-segmentation "
            f"columns decode cv frames with no boundaries given (loop "
            f"{bonus.FREE_SEG_LOOP}, LM scale {bonus.MATCHED_DECODE_LM}, a fixed "
            f"{int(100 * bonus.FREE_SEG_FRACTION)}\\% of cv)."
        ),
    )
    progress_report = TextTableReport(
        columns=[param("rule"), update, passes, "posterior_entropy", "cluster_perplexity",
                 "segment_duration_argmax", "boundary_rate", "dead_clusters"]
                + step_columns,
        sort_by=["rule"],
        drop_empty_rows=True,
        title=(
            f"Per-update convergence from epoch {online.BRANCH_EPOCH} of the bonus-"
            f"{online.REFERENCE_FRACTION} arm: full-batch EM (one update per pass) "
            f"against over-relaxed online EM ({online.BATCHES_PER_EPOCH} updates per "
            f"pass, step eta). A row is the job of that update, measured with the "
            f"previous update's model - on the whole corpus for full-batch rows, on "
            f"that step's quarter for online ones. The step columns are the mean "
            f"KL(new || previous table) over labels of the step EM proposed and the one "
            f"taken (about eta^2 apart while small), and the share of weights held at "
            f"the plain M-step's floor. An oscillating arm shows a growing EM step. "
            f"Reference at {online.BRANCH_EPOCH}: 61.3% PER; 160: 38.2%; 200: 11.8%."
        ),
    )
    recog_results = []

    # --- full-batch reference: vq_gmm_tau_bonus's own epochs --------------------
    rule = {"rule": "full-batch"}
    ref_stats = clustering_statistics_per_epoch(
        reference.out_epoch_statistics, name=reference_name, epoch_offset=1,
        lexicon=lexicon)
    ref_progress = clustering_statistics_per_epoch(
        reference.out_epoch_statistics, name=reference_name, epoch_offset=0,
        lexicon=lexicon)
    for e in range(online.BRANCH_EPOCH + 1, REFERENCE_EPOCHS + 1):
        progress_report.add_row(params=dict(rule), epoch=e, statistics=ref_progress,
                                values={"passes": e - online.BRANCH_EPOCH})
    for e in REFERENCE_DECODE_EPOCHS:
        res, values = decode(reference_name, reference.out_models[e], f"ep-{e}")
        values.update(balance(reference.out_artifacts["mixtures"][e]))
        values["passes"] = e - online.BRANCH_EPOCH
        recog_results.append(res)
        report.add_row(result=res, params=dict(rule), epoch=e, statistics=ref_stats,
                       values=values)

    # --- over-relaxed online arms ---------------------------------------------------
    for eta in ETAS:
        name = (f"{reference_name}_branch-{online.BRANCH_EPOCH}"
                f"_online-b{online.BATCHES_PER_EPOCH}_ema-{online.EMA.alpha:g}_relax-{eta:g}")
        branch = chunked_clustering(
            num_epochs=BRANCH_EPOCHS,
            flavor=flavor(**branch_start, over_relaxation=eta),
            segments=ls100.out_segments,
            batches_per_epoch=online.BATCHES_PER_EPOCH,
            ema=online.EMA,
            num_chunks=online.NUM_CHUNKS,
            rqmt=online.EPOCH_RQMT,
            alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
            **common,
        )
        # Keyed by update, as the reference's epochs are (vq_gmm_tau_online).
        jobs = {online.BRANCH_EPOCH + k: job for k, job in enumerate(branch.jobs, start=1)}
        rule = {"rule": f"online B={online.BATCHES_PER_EPOCH}, eta={eta:g}"}
        step_statistics = {u: job.out_statistics for u, job in jobs.items()}
        stats = clustering_statistics_per_epoch(
            step_statistics, name=name, epoch_offset=1, lexicon=lexicon)
        progress = clustering_statistics_per_epoch(
            step_statistics, name=name, epoch_offset=0, lexicon=lexicon)
        for u in jobs:
            progress_report.add_row(
                params=dict(rule), epoch=u, statistics=progress,
                values={"passes": (u - online.BRANCH_EPOCH) / online.BATCHES_PER_EPOCH})
        for k in DECODE_STEPS:
            u = online.BRANCH_EPOCH + k
            res, values = decode(name, jobs[u].out_model, f"upd-{u}")
            values.update(balance(jobs[u].artifact("mixtures")))
            values["passes"] = k / online.BATCHES_PER_EPOCH
            recog_results.append(res)
            report.add_row(result=res, params=dict(rule), epoch=u, statistics=stats,
                           values=values)

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
