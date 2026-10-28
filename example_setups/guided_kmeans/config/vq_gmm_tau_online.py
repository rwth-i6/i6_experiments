"""Online EM from the bonus-0.5 arm's epoch 100: does progress follow M-steps or data?

``docs/free_segmentation/findings_and_directions.md`` §6.1 and §7 (item 1a).

The one free-segmentation run that broke out (``vq_gmm_tau_bonus``, S=2, bonus 0.5,
segment codebook) needed ~130 epochs to take off, with the objective nearly flat
the whole way: slow EM along a consistent direction. Every epoch there is a single
full-batch M-step over all of ls-100h, so its pace is one update per pass. If what
moves the model is the number of updates rather than the data behind each, then
re-estimating four times per pass, each time from a quarter of the corpus, gets four
updates for the compute of one - and every later experiment costs a quarter.

**One arm, branched, against existing epochs.** The run's model after epoch 100
(61.3% PER, on the plateau, 30 epochs before the takeoff) starts an online run with
``batches_per_epoch=4``: each step recognizes a fixed, shuffled quarter of the corpus
with the current model and re-estimates the table from it alone. No damping
(``EMAConfig(alpha=0.0)``): a quarter is ~3.4M frames for a 40x512 table, and
damping is exactly what would take the speed back - an EMA over one corpus (the
``vq_unsupervised_frames`` setting) moves the table by about one full EM step per
pass, the reference's pace. That variant is the fallback if this one is too noisy,
not a second arm.

The comparison is by update count. The branch's k-th step is update 100 + k, the
reference's epoch e is update e, and the reference's epochs 101-160 already exist
(61.3 -> 38.2%), so they are the control and nothing is rerun. The ``Passes`` column
says what each row cost in corpus passes since the branch.

**Reading it.** Per-update statistics (entropy, durations) are measured on the
quarter each step saw, so the branch's progress rows describe a quarter of the
corpus each; the decodes are on the full cv set for both, as in
``vq_gmm_tau_bonus``. If the branch tracks the reference per update, online EM is
the cheap platform for everything in §7; if it tracks per pass, it buys nothing;
if it is noisy (rows jumping with the batch index), damping comes back.

**Only finished reference jobs are built.** The reference is rebuilt with
``vq_gmm_tau_bonus``'s exact arguments, so its epochs, decodes and balance jobs are
that config's, all finished when this was written (the bonus manager was at epoch
~165). ``REFERENCE_EPOCHS`` must not go past what that manager has finished, or
this config would submit its epochs too.

Cost: one pass is ~30-35 GPU-h, 15 passes (60 updates) ~500 GPU-h - about a day
and a half on 16 gpu_11gb cards, plus ~2 min of scheduling per step. Decodes are
CPU only. Can be stopped early: the decodes every five updates show the
comparison as it goes.
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
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_bonus as bonus

exp_dir = "vq_gmm_tau_online"
version = 1

#: The arm the branch starts from, in vq_gmm_tau_bonus's terms.
REFERENCE_FRACTION = 0.5
REFERENCE_CODEBOOK = "segments"
#: The last decoded reference epoch (160) plus one, whose statistics describe the
#: epoch-160 model. Must stay within what the bonus manager has finished.
REFERENCE_EPOCHS = 161

#: Branch after this epoch's model; the reference's decodes from here on are the
#: comparison rows.
BRANCH_EPOCH = 100
#: Passes over the corpus; BATCHES_PER_EPOCH updates each. 15 x 4 = 60 updates,
#: the reference's epochs 101-160. Epoch chains do not hash their length.
BRANCH_EPOCHS = 15
BATCHES_PER_EPOCH = 4
#: Undamped: every quarter replaces the table (module docstring).
EMA = EMAConfig(alpha=0.0)
#: In updates since the branch, matching the reference's decodes at 105-160.
DECODE_STEPS = list(range(5, BATCHES_PER_EPOCH * BRANCH_EPOCHS + 1, 5))
REFERENCE_DECODE_EPOCHS = list(range(BRANCH_EPOCH, REFERENCE_EPOCHS, 5))

#: Unhashed. 16 chunks per step makes a chunk 1/64 of the corpus, the size the
#: reference has run at since epoch 93 (~30 min on a 1080).
NUM_CHUNKS = 16
#: A copy, so vq_gmm_tau_bonus's own dict stays as it is. As vq_gmm_tau_topology:
#: gpu_11gb only, leaving the capped L40S and A10 slots to the running arm, and a
#: nice above the age weight's maximum (1500), so these tasks stay behind every
#: other pending GPU job of this user however long they wait.
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

    def flavor(centroids, covs, mixtures):
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

    branch_name = (f"{reference_name}_branch-{BRANCH_EPOCH}_online-b{BATCHES_PER_EPOCH}"
                   f"_ema-{EMA.alpha:g}")
    branch = chunked_clustering(
        num_epochs=BRANCH_EPOCHS,
        flavor=flavor(
            reference.out_artifacts["centroids"][BRANCH_EPOCH],
            reference.out_artifacts["covs"][BRANCH_EPOCH],
            reference.out_artifacts["mixtures"][BRANCH_EPOCH],
        ),
        segments=ls100.out_segments,
        batches_per_epoch=BATCHES_PER_EPOCH,
        ema=EMA,
        num_chunks=NUM_CHUNKS,
        rqmt=EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{exp_dir}/{branch_name}",
        **common,
    )
    # Keyed by update, as the reference's epochs are: step k recognizes with the
    # model of update BRANCH_EPOCH + k - 1 and produces update BRANCH_EPOCH + k.
    branch_jobs = {BRANCH_EPOCH + k: job for k, job in enumerate(branch.jobs, start=1)}

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

    passes = Column("passes", ("Passes", f"since {BRANCH_EPOCH}"),
                    lambda row: row.values.get("passes"), fmt="{:g}")
    update = relabel("epoch", "Update")
    balance_columns = ["segment_gap", "boundary_payback", "payable_boundaries"]
    report = LatexTableReport(
        columns=[param("rule", header=("Update", "rule")), update, passes, "per", "del",
                 "ins", "sub", "matched_per", "matched_del", "matched_ins",
                 "matched_sub", "matched_hyp_ref", "posterior_entropy",
                 "segment_duration_argmax", "boundary_rate"] + balance_columns,
        sort_by=["rule"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Online EM against full-batch EM, by update: vq\\_gmm\\_tau\\_bonus's "
            f"bonus-{REFERENCE_FRACTION} arm (S={bonus.NUM_STATES}, segment codebook), "
            f"branched after epoch {BRANCH_EPOCH} into {BATCHES_PER_EPOCH} undamped "
            f"updates per pass over the corpus, each from its own fixed quarter. "
            f"Full-batch rows are the run's own epochs (one update per pass). "
            f"Decoded on mean-pooled cv at LM scale 1.0; the free-segmentation "
            f"columns decode cv frames with no boundaries given (loop "
            f"{bonus.FREE_SEG_LOOP}, LM scale {bonus.MATCHED_DECODE_LM}, a fixed "
            f"{int(100 * bonus.FREE_SEG_FRACTION)}\\% of cv)."
        ),
    )
    progress_report = TextTableReport(
        columns=[param("rule"), update, passes, "posterior_entropy", "cluster_perplexity",
                 "segment_duration_argmax", "boundary_rate", "dead_clusters"],
        sort_by=["rule"],
        drop_empty_rows=True,
        title=(
            f"Per-update convergence from epoch {BRANCH_EPOCH} of the bonus-"
            f"{REFERENCE_FRACTION} arm: full-batch (one update per pass) against "
            f"online EM ({BATCHES_PER_EPOCH} updates per pass). A row is the job of that "
            f"update, measured with the previous update's model - on the whole corpus "
            f"for full-batch rows, on that step's quarter for online ones, so the "
            f"online rows may wobble with the batch index. Reference at "
            f"{BRANCH_EPOCH}: 61.3% PER; at 160: 38.2%."
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
    for e in range(BRANCH_EPOCH + 1, REFERENCE_EPOCHS + 1):
        progress_report.add_row(params=dict(rule), epoch=e, statistics=ref_progress,
                                values={"passes": e - BRANCH_EPOCH})
    for e in REFERENCE_DECODE_EPOCHS:
        res, values = decode(reference_name, reference.out_models[e], f"ep-{e}")
        values.update(balance(reference.out_artifacts["mixtures"][e]))
        values["passes"] = e - BRANCH_EPOCH
        recog_results.append(res)
        report.add_row(result=res, params=dict(rule), epoch=e, statistics=ref_stats,
                       values=values)

    # --- online branch -------------------------------------------------------------
    rule = {"rule": f"online B={BATCHES_PER_EPOCH}"}
    branch_statistics = {u: job.out_statistics for u, job in branch_jobs.items()}
    branch_stats = clustering_statistics_per_epoch(
        branch_statistics, name=branch_name, epoch_offset=1, lexicon=lexicon)
    branch_progress = clustering_statistics_per_epoch(
        branch_statistics, name=branch_name, epoch_offset=0, lexicon=lexicon)
    for u in branch_jobs:
        progress_report.add_row(params=dict(rule), epoch=u, statistics=branch_progress,
                                values={"passes": (u - BRANCH_EPOCH) / BATCHES_PER_EPOCH})
    for k in DECODE_STEPS:
        u = BRANCH_EPOCH + k
        job = branch_jobs[u]
        res, values = decode(branch_name, job.out_model, f"upd-{u}")
        values.update(balance(job.artifact("mixtures")))
        values["passes"] = k / BATCHES_PER_EPOCH
        recog_results.append(res)
        report.add_row(result=res, params=dict(rule), epoch=u, statistics=branch_stats,
                       values=values)

    tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{branch_name}.json",
                       branch.out_statistics)
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
