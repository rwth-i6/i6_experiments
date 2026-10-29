"""S=3 from scratch, online EM, at the calibrated loop probability.

``docs/free_segmentation/findings_and_directions.md`` §6.10 and §7 (item 3).

The 0.5 arm of ``vq_gmm_tau_bonus`` (S=2) converged at 11.8% PER (epoch 200). This is
the same run with three states per phoneme instead of two, and nothing else changed
that can be avoided: exactly uniform table, frozen segment codebook, tied states,
5-gram at LM scale 1.0, am = ts = 0.35, silence-free ls-100h frames.

**The loop probability comes from ``vq_gmm_tau_s3_calibration``.** One E-step of the
0.5 arm's models with S=3 at seven bonuses showed the posterior duration landing on
the generative mean ``S / (1 - p)`` (ratio 0.97-1.09, where S=2's is ~1.25), and put
the 0.5 arm's own 4.05 frames between a bonus of 1.8 (epoch-20 model) and 2.15
(epoch-160 model). At 2.0 - 3.776 nats per phoneme, twice the LM's cost - the two
models give 3.94 and 4.10. Read against S=2's own one-step numbers, whose runs are
known (0.25: 4.22-4.30 -> plateau; 0.5: 4.05 -> converged; 0.75: 3.82-3.90 -> slower),
1.5 (4.23-4.31) sits in the plateau regime and 3.0 (3.54-3.77) beyond the slow one.
2.0 errs on the short side, which in the S=2 sweep cost speed, while the long side
cost the run.

**Online EM, B=4, undamped.** ``vq_gmm_tau_online`` showed it tracking the full-batch
run update for update from epoch 100 through the takeoff (60.5 / 58.1 / 54.2 against
60.6 / 58.2 / 54.2 at updates 105 / 120 / 135), at a quarter of the compute per
update. It has not been checked from a uniform start, where the classes form, so a
second arm runs S=2 the same way for the first 40 updates: if it tracks the
full-batch 0.5 arm there too, S=3 can be read against that arm's epochs 1-200 by
update throughout.

Rows are by update for all three: the full-batch S=2 reference (its epoch e is
update e; all finished), the S=2 online control and the S=3 run.

Cost: an S=3 pass is ~1.3x an S=2 one by the calibration's task times, ~45 GPU-h;
200 updates are 50 passes, ~2,200 GPU-h, ~4-5 days at 24 chunks a step. S=3 runs at
the bonus run's priority on all four GPU partitions: the 0.5 arm has finished, and
this is the main run now. The control (~330 GPU-h) runs behind the side runs (see
``ARMS``).
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
    EMAConfig,
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
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_lm import (
    frame_level_ls100,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_s3_calibration import (
    calibration_lambda,
)
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_bonus as bonus

exp_dir = "vq_gmm_tau_s3"
version = 1

#: The full-batch reference, in vq_gmm_tau_bonus's terms. Finished at 200.
REFERENCE_FRACTION = 0.5
REFERENCE_CODEBOOK = "segments"
REFERENCE_EPOCHS = 200

#: The bonus run's own: all four GPU partitions, no nice.
MAIN_RQMT = bonus.EPOCH_RQMT
#: As the side runs (vq_gmm_tau_online, _detector, _overrelax): gpu_11gb only,
#: behind every other pending GPU job of this user.
SIDE_RQMT = dict(bonus.EPOCH_RQMT, sbatch_args=["-p", "gpu_11gb", "--nice=1600"])

#: ``(name, S, bonus fraction, passes, chunks per step, max frames, rqmt)``. Passes
#: are epochs of BATCHES_PER_EPOCH updates each and do not hash; a chain resumes
#: when extended. Max frames keeps states x frames at S=2's 9,000 x 2 (the
#: calibration ran S=3 at 6,000 on 11 GB cards). Chunks and rqmt are unhashed.
#: At launch (2026-10-23) every GPU partition was full and four nice'd side runs
#: asked for 64 gpu_11gb cards against the ~42 this user actually got: S=3 takes
#: 24 at normal priority (~34 min a step on 1080s), the control 8 behind the side
#: runs - it is read against the finished full-batch run, so it is not in a hurry.
#:
#: 2026-10-28: the bonus-2.0 arm reached update 200 at 52.1%, where the S=2 run was
#: at 140 and accelerating the same way, so it runs on to 280. Next to it a small
#: sweep, 1.5 and 2.5 (+-0.94 nats), from scratch and otherwise identical, for 100
#: updates: their one-step duration offsets from the calibration (+0.18-0.21 /
#: -0.16-0.20 frames) are those of S=2's 0.25 and 0.75 arms, which stalled and
#: slowed, so the sweep asks whether S=3 tolerates what S=2 did not. Nice'd, next to
#: the other side runs: the 2.0 arm is the main run.
S3_MAX_FRAMES = bonus.MAX_FRAMES * 2 // 3
ARMS = [
    ("S=3 online", 3, 2.0, 70, 24, S3_MAX_FRAMES, MAIN_RQMT),
    ("S=2 online", 2, REFERENCE_FRACTION, 10, 8, bonus.MAX_FRAMES, SIDE_RQMT),
    ("S=3 online", 3, 1.5, 25, 16, S3_MAX_FRAMES, SIDE_RQMT),
    ("S=3 online", 3, 2.5, 25, 16, S3_MAX_FRAMES, SIDE_RQMT),
]
BATCHES_PER_EPOCH = 4
#: Undamped, as vq_gmm_tau_online.
EMA = EMAConfig(alpha=0.0)
#: Every five updates, as the reference decodes.
DECODE_EVERY = 5


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

    def flavor(backoff_fb):
        return mixture_flavor(
            centroids=centroids,
            covs=covs,
            mixtures=init_table,
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
    reference_epochs_total, reference_chunks = bonus.LONG_ARMS[
        (REFERENCE_FRACTION, REFERENCE_CODEBOOK)]
    assert REFERENCE_EPOCHS <= reference_epochs_total
    reference = chunked_clustering(
        num_epochs=REFERENCE_EPOCHS,
        flavor=flavor(search(bonus.NUM_STATES, reference_p, bonus.MAX_FRAMES)),
        num_chunks=reference_chunks,
        rqmt=bonus.EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{bonus.exp_dir}/{reference_name}",
        **common,
    )

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

    def balance(mixtures, num_states, p):
        job = EmissionBalanceJob(
            frames=balance_frames,
            mixtures=mixtures,
            am_scale=bonus.AM_SCALE,
            lm_cost=lm_cost,
            lm_scale=bonus.LM_SCALE,
            transition_penalty=transition_boundary_penalty(
                num_states, p, bonus.TRANSITION_SCALE),
        )
        return {
            "segment_gap": job.out_segment_gap,
            "boundary_payback": job.out_payback,
            "boundary_price": job.out_price,
            "payable_boundaries": job.out_payable,
        }

    passes = Column("passes", ("Corpus", "passes"),
                    lambda row: row.values.get("passes"), fmt="{:g}")
    update = relabel("epoch", "Update")
    arm_columns = [
        param("run", header=("Run",)),
        param("states", header=("HMM", "states")),
        param("bonus", header=("Bonus", "frac."), fmt="{:g}"),
        loop_probability_column("lambda", bonus.TRANSITION_SCALE),
    ]
    balance_columns = ["segment_gap", "boundary_payback", "boundary_price",
                       "payable_boundaries"]
    report = LatexTableReport(
        columns=arm_columns + [update, passes, "per", "del", "ins", "sub", "matched_per",
                               "matched_del", "matched_ins", "matched_sub",
                               "matched_hyp_ref", "posterior_entropy",
                               "segment_duration_argmax", "boundary_rate"]
                + balance_columns,
        sort_by=["states", "bonus", "run"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Three HMM states per phoneme against two, free segmentation on "
            f"silence-free ls-100h frames, LM scale {bonus.LM_SCALE}, am=ts="
            f"{bonus.AM_SCALE}, exactly uniform initial table, segment codebook, tied "
            f"states. The S=3 bonus is twice the LM's {bonus.LM_PHONEME_COST}-nat "
            f"phoneme cost, calibrated so that its posterior duration matches the S=2 "
            f"bonus-{REFERENCE_FRACTION} arm's. Rows by update: the full-batch S=2 run "
            f"(one update per pass) against online EM ({BATCHES_PER_EPOCH} undamped "
            f"updates per pass). Decoded on mean-pooled cv at LM scale 1.0; the "
            f"free-segmentation columns decode cv frames with no boundaries given "
            f"(loop {bonus.FREE_SEG_LOOP}, LM scale {bonus.MATCHED_DECODE_LM}, a fixed "
            f"{int(100 * bonus.FREE_SEG_FRACTION)}\\% of cv)."
        ),
    )
    progress_report = TextTableReport(
        columns=arm_columns + [update, passes, "posterior_entropy", "cluster_perplexity",
                               "segment_duration_argmax", "boundary_rate",
                               "dead_clusters"],
        sort_by=["states", "bonus", "run"],
        drop_empty_rows=True,
        title=(
            f"Per-update convergence, S=3 against S=2 (bonus-{REFERENCE_FRACTION} arm). "
            f"A row is the job of that update, measured with the previous update's "
            f"model - on the whole corpus for the full-batch rows, on that step's "
            f"quarter for the online ones. Watch the argmax duration against the "
            f"calibration's 3.94 (epoch-20 model) / 4.10 (epoch-160 model) and the "
            f"S=2 run's 4.03-4.13; the alignment has 3.99."
        ),
    )
    recog_results = []

    # --- full-batch S=2 reference: vq_gmm_tau_bonus's own epochs ----------------
    params = {"run": "S=2 full-batch", "states": bonus.NUM_STATES,
              "bonus": REFERENCE_FRACTION, "lambda": reference_lam}
    ref_stats = clustering_statistics_per_epoch(
        reference.out_epoch_statistics, name=reference_name, epoch_offset=1,
        lexicon=lexicon)
    ref_progress = clustering_statistics_per_epoch(
        reference.out_epoch_statistics, name=reference_name, epoch_offset=0,
        lexicon=lexicon)
    for e in range(1, REFERENCE_EPOCHS + 1):
        progress_report.add_row(params=dict(params), epoch=e, statistics=ref_progress,
                                values={"passes": e})
    for e in range(DECODE_EVERY, REFERENCE_EPOCHS + 1, DECODE_EVERY):
        res, values = decode(reference_name, reference.out_models[e], f"ep-{e}")
        values.update(balance(reference.out_artifacts["mixtures"][e], bonus.NUM_STATES,
                              reference_p))
        values["passes"] = e
        recog_results.append(res)
        report.add_row(result=res, params=dict(params), epoch=e, statistics=ref_stats,
                       values=values)

    # --- online arms from scratch --------------------------------------------------
    for label, num_states, fraction, num_passes, num_chunks, max_frames, rqmt in ARMS:
        lam = calibration_lambda(fraction, num_states)
        p = bonus.loop_for_lambda(lam, bonus.TRANSITION_SCALE)
        name = (f"weights_tau-{bonus.TAU}_s-{num_states}_bonus-{fraction}_loop-{round(p, 4)}"
                f"_ts-{bonus.TRANSITION_SCALE}_lm-{bonus.LM_SCALE}_am-{bonus.AM_SCALE}"
                f"_init-uniform_cb-{REFERENCE_CODEBOOK}_online-b{BATCHES_PER_EPOCH}"
                f"_ema-{EMA.alpha:g}")
        result = chunked_clustering(
            num_epochs=num_passes,
            flavor=flavor(search(num_states, p, max_frames)),
            segments=ls100.out_segments,
            batches_per_epoch=BATCHES_PER_EPOCH,
            ema=EMA,
            num_chunks=num_chunks,
            rqmt=rqmt,
            alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
            **common,
        )
        # Step k recognizes with the model of update k - 1 and produces update k.
        jobs = {k: job for k, job in enumerate(result.jobs, start=1)}
        params = {"run": label, "states": num_states, "bonus": fraction, "lambda": lam}
        statistics = {k: job.out_statistics for k, job in jobs.items()}
        stats = clustering_statistics_per_epoch(
            statistics, name=name, epoch_offset=1, lexicon=lexicon)
        progress = clustering_statistics_per_epoch(
            statistics, name=name, epoch_offset=0, lexicon=lexicon)
        for k in jobs:
            progress_report.add_row(params=dict(params), epoch=k, statistics=progress,
                                    values={"passes": k / BATCHES_PER_EPOCH})
        for k in range(DECODE_EVERY, len(jobs) + 1, DECODE_EVERY):
            job = jobs[k]
            res, values = decode(name, job.out_model, f"upd-{k}")
            values.update(balance(job.artifact("mixtures"), num_states, p))
            values["passes"] = k / BATCHES_PER_EPOCH
            recog_results.append(res)
            report.add_row(result=res, params=dict(params), epoch=k, statistics=stats,
                           values=values)
        tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{name}.json",
                           result.out_statistics)

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
