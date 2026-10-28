"""Silence without the oracle: training on the VAD frames with their silence kept.

``docs/free_segmentation/findings_and_directions.md`` §0 and §7. The bonus-0.5 arm
converged at 11.8% PER (21.0% free-segmentation) on ls-100h frames whose silence the
GMM alignment had removed - an oracle step. The features themselves are already
VAD-trimmed (the fairseq wav2vec-U pipeline runs rVAD before extraction): 15.16M
frames for ls-100h, ~84% of the audio. On them the alignment still marks 8.9% of the
frames silence: 95% of utterances start with ~6 frames of it, 65% end with ~4, and
5.3 pauses of ~7.4 frames sit inside each (measured 2026-10-26; rVAD cut everything
much longer). This config keeps those frames and lets ``<SIL>`` - label 0, already in
the LM and the table - model them.

**Two pairs of arms, each pair differing only in whether the oracle drops silence**
(``SegmentedFeaturesFromAlignmentJob`` with ``exclude_labels=(0,)`` or ``()``; the
same job otherwise, so the tags, their order and the online quarters coincide):

* *From scratch*: exactly uniform table, a frame codebook fit on the VAD frames with
  silence (unguided k-means under their global covariance, as ``vq_gmm_tau_bonus``'s
  frame codebook but on the silence-kept data), that covariance for the model,
  online EM (B=4, undamped) with over-relaxation 1.5 - the speed-up defaults - for
  ``PASSES`` passes. The oracle-silence arm is the control; it is also the first
  from-scratch run at eta = 1.5.
* *Branch*: the converged bonus-0.5 model (epoch 200, segment codebook, its own
  covariance) continued for ``BRANCH_PASSES`` passes. It asks whether ``<SIL>``
  takes the silence frames once the phoneme classes are good. Its codebook has no
  cells fit to silence, so it is not a preview of the from-scratch arm.

**The LM is the original** (``SilenceModel(between_word=0.1, sentence_bounds=0.1)``:
``<SIL>`` inserted at random, 10% at the sentence bounds, against 95% / 65% in this
data). *Added 2026-10-27:* in the first branch updates ``<SIL>`` settled at ~57% of the
data's silence segments, about the LM's rate relative to the data's, so a branch and a
from-scratch arm on the VAD frames also train with the LM re-silenced to the data
(``vq_gmm_tau_silence_lm``: 0.952 / 0.646 at the sentence begin / end, 0.158 per word
boundary; per-phoneme cost 1.859 against 1.888, so the 0.944-nat bonus stays). Each is
its original-LM VAD arm with only the training LM changed; decodes keep the original LM,
so every arm is evaluated the same way.

**The topology changes how the bonus is written, not what it is.** The bonus-0.5 arm
paid its 0.944 nats per phoneme through a low loop (p = 0.2061 at ts = 0.35). That is
equivalent only while every label shares the loop: with a second loop class, each
silence frame would gain ts * ln(q / p) over a speech frame (~0.44 nats at q = 0.71),
more than the converged model's label evidence per frame (~0.3). Here speech loops at
0.5 and the same 0.944 nats is an exit offset per phoneme - the identical speech
posterior (checked to 1e-9 on real data, ``build/silence/silence_check.py`` in the
op's checkout) - and silence gets its own loop, ``SILENCE_LOOP``, from the measured
6.9 frames: S / (1 - q) = 6.9. Its per-frame advantage is then ts * ln(q / 0.5) =
0.12 nats, and its exit carries ``SILENCE_EXIT_OFFSET``. The control runs the same
topology, so the pair differs in the data alone.

**Evaluation on both cv conditions.** Every decode runs on cv as before (silence
removed by the alignment, comparable with every earlier number) and on cv as the VAD
left it (silence kept, the honest condition). RASR's lexicon maps ``[SILENCE]`` to
the ``<SIL>`` LM token with an empty evaluation token, so decoded silence never
scores. The progress table splits the argmax durations into silence and speech
(``FBStatisticsCounter``, added for this config): with silence kept, the pooled
duration no longer compares with the alignment's 3.99 frames per phoneme.

Cost: an S=2 pass ~32 GPU-h (VAD frames ~10% more), 40 passes per from-scratch arm,
~1,300 GPU-h each; the branch arms ~170 GPU-h each. On gpu_11gb at ``--nice=1600``,
next to the S=3 run (24 cards) - 16 chunks a step from scratch, 8 for the branches.
With the re-silenced arms, three from-scratch arms (48 chunks) and S=3's 18 make 66 of
the 70-card cap once the branches are done; until then the extra tasks queue.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    GMM_ALIGNMENT_CV,
    GMM_ALIGNMENT_LS960_FRAME,
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
    RandomCentroidsJob,
    ScaleCovsJob,
    chunked_clustering,
    mixture_flavor,
    unguided_flavor,
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
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_silence_lm import (
    vad_silence_lm,
)

exp_dir = "vq_gmm_tau_silence"
version = 1

#: The converged run the branch arms start from, in vq_gmm_tau_bonus's terms. All
#: of its 200 epochs are finished, so nothing built here is another manager's.
REFERENCE_FRACTION = 0.5
REFERENCE_CODEBOOK = "segments"
REFERENCE_EPOCHS = 200

#: The speech side of the bonus-0.5 arm, written as loop 0.5 plus an exit offset:
#: S * lambda = 2 * 0.472 nats per phoneme (``bonus.bonus_lambda``).
SPEECH_LOOP = 0.5
SPEECH_BONUS = -bonus.NUM_STATES * bonus.bonus_lambda(REFERENCE_FRACTION)
#: The ARPAbet inventory of the lexicon and the LM; ``<SIL>`` is the 40th symbol.
#: Spelled out because an exit-offset dict leaves every symbol it omits at 0.
PHONEMES = (
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T "
    "TH UH UW V W Y Z ZH").split()
SILENCE = "<SIL>"
#: S / (1 - q) = 6.9 frames, the alignment's mean silence run on ls-100h (initial
#: 6.0, internal 7.4, final 4.3; 2026-10-26).
SILENCE_LOOP = 0.71
#: Nats on <SIL>'s exit, after the transition scale. 0: silence pays its loop's own
#: per-segment price, S * ts * ln((1 - q) / q) = -0.63 nats, plus the LM's.
SILENCE_EXIT_OFFSET = 0.0

#: Online EM and over-relaxation, the defaults for new weights-only runs.
BATCHES_PER_EPOCH = 4
EMA = EMAConfig(alpha=0.0)
OVER_RELAXATION = 1.5

#: Passes per arm (BATCHES_PER_EPOCH updates each). Epoch chains do not hash their
#: length, so an arm is extended by raising these.
PASSES = 40
BRANCH_PASSES = 5
#: Decodes by update: every tenth from scratch, every fifth for the branches.
DECODE_EVERY = 10
BRANCH_DECODE_EVERY = 5

#: Unhashed scheduling: gpu_11gb behind every other pending GPU job of this user
#: (the S=3 run, which also uses the capped A10 / L40S slots).
RQMT = dict(bonus.EPOCH_RQMT, sbatch_args=["-p", "gpu_11gb", "--nice=1600"])
CHUNKS = 16
BRANCH_CHUNKS = 8


def vad_frames():
    """ls-100h frames with silence kept: ``frame_level_ls100()`` with nothing excluded.

    The alignment only packages the frames here (tags, order, per-frame labels for
    the balance measurement); no frame is dropped.
    """
    job = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["ls-100"]["features"],
        alignment=GMM_ALIGNMENT_LS960_FRAME,
        exclude_labels=(),
        pooling="none",
        rqmt={"cpu": 2, "mem": 16, "time": 12},
    )
    job.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_frames_vad")
    return job


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    data = {"oracle": frame_level_ls100(), "VAD": vad_frames()}
    train_config = bonus._recog_config(bonus.LM_SCALE, 0.0, 1.0, use_fb=True)
    decode_config_rasr = bonus._recog_config(1.0, 0.0, None, use_fb=False)

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

    def search(loop_probability, silence_loop_probability=None, exit_offset=None,
               arpa_path=None):
        return BackoffFBSearch(
            backoff_fb_root=tools.BACKOFF_FB,
            arpa_path=phonetic_lm_dict[bonus.LM_ORDER] if arpa_path is None else arpa_path,
            lexicon=fb_lexicon,
            lm_scale=bonus.LM_SCALE,
            emission_scale=1.0,
            loop_probability=loop_probability,
            silence_loop_probability=silence_loop_probability,
            transition_scale=bonus.TRANSITION_SCALE,
            num_states=bonus.NUM_STATES,
            apply_sentence_end=True,
            batch_size=bonus.BATCH_SIZE,
            max_frames=bonus.MAX_FRAMES,
            checkpoint_interval=-1,
            storage_dtype="float32",
            exit_offset=exit_offset,
        )

    def flavor(centroids, covs, mixtures, backoff_fb, over_relaxation=1.0):
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
            backoff_fb=backoff_fb,
        )

    exit_offsets = {**{phoneme: SPEECH_BONUS for phoneme in PHONEMES}, SILENCE: SILENCE_EXIT_OFFSET}
    silence_searches = {
        "original": search(SPEECH_LOOP, SILENCE_LOOP, exit_offsets),
        "VAD silence": search(SPEECH_LOOP, SILENCE_LOOP, exit_offsets,
                              arpa_path=vad_silence_lm(bonus.LM_ORDER)),
    }

    # --- the branch point: vq_gmm_tau_bonus's run, argument for argument ----------
    nosil = data["oracle"]
    ref_global_cov = GlobalCovarianceJob(nosil.out_features).out_cov
    ref_covs = ScaleCovsJob(
        DuplicateCovsJob(ref_global_cov, bonus.NUM_CODEWORDS).out_covs, bonus.TAU).out_covs
    uniform = NormalTableJob(
        bonus.NUM_LABELS, bonus.NUM_CODEWORDS, sigma=bonus.INIT_SIGMA, seed=bonus.SEED
    ).out_table
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
        features_hdf=nosil.out_features,
        flavor=flavor(COLLEAGUE_CENTROIDS_K512, ref_covs, uniform, search(reference_p)),
        num_chunks=reference_chunks,
        rqmt=bonus.EPOCH_RQMT,
        alias_prefix=f"guided_kmeans/{bonus.exp_dir}/{reference_name}",
        **common,
    )
    branch_start = {
        name: reference.out_artifacts[name][REFERENCE_EPOCHS]
        for name in ("centroids", "covs", "mixtures")
    }

    # --- the from-scratch codebook and covariance: VAD frames, silence kept -------
    vad = data["VAD"]
    vad_global_cov = GlobalCovarianceJob(vad.out_features).out_cov
    vad_base_covs = DuplicateCovsJob(vad_global_cov, bonus.NUM_CODEWORDS).out_covs
    vad_covs = ScaleCovsJob(vad_base_covs, bonus.TAU).out_covs
    codebook = chunked_clustering(
        num_epochs=bonus.CODEBOOK_EPOCHS,
        features_hdf=vad.out_features,
        num_clusters=bonus.NUM_CODEWORDS,
        flavor=unguided_flavor(
            centroids=RandomCentroidsJob(
                vad.out_features, bonus.NUM_CODEWORDS, seed=bonus.SEED).out_centroids,
            covs=vad_base_covs,
            num_clusters=bonus.NUM_CODEWORDS,
        ),
        num_chunks=bonus.CODEBOOK_CHUNKS,
        alias_prefix=f"guided_kmeans/{exp_dir}/codebook-frames-vad-{bonus.NUM_CODEWORDS}",
    ).out_centroids[bonus.CODEBOOK_EPOCHS]

    # --- evaluation: cv with silence removed by the alignment, and as the VAD left it
    cv_segments = silence_free_cv_features().out_segments
    free_seg_segments = ShuffleAndSplitSegmentsJob(
        cv_segments,
        split={"decode": bonus.FREE_SEG_FRACTION, "rest": 1.0 - bonus.FREE_SEG_FRACTION},
    ).out_segments["decode"]
    cv_sets = {}
    for condition, exclude in (("oracle", (0,)), ("VAD", ())):
        pooled = SegmentedFeaturesFromAlignmentJob(
            features_hdf=input_data["cv"]["features"],
            alignment=GMM_ALIGNMENT_CV,
            exclude_labels=exclude,
            pooling="mean",
        )
        frames = SegmentedFeaturesFromAlignmentJob(
            features_hdf=input_data["cv"]["features"],
            alignment=GMM_ALIGNMENT_CV,
            exclude_labels=exclude,
            pooling="none",
        )
        cv_sets[condition] = (
            DatasetConfig(audio_hdf_path=pooled.out_features,
                          sampling_method=SegmentFile(cv_segments), precomputed=True),
            DatasetConfig(audio_hdf_path=frames.out_features,
                          sampling_method=SegmentFile(free_seg_segments), precomputed=True),
        )

    def decode(name, model_dir, label, centroids, condition):
        """vq_gmm_tau_bonus's two decodes, on one cv condition. On the oracle
        condition they are that config's jobs for its own models."""
        pooled, frames = cv_sets[condition]
        suffix = "" if condition == "oracle" else "_cv-vad"
        dc = DecodeConfig(
            centroids=centroids,
            model_dir=model_dir,
            recog_rasr_config=decode_config_rasr,
            distance_scale=bonus.BASE_DECODE_SCALE,
            write_frame_labels=True,
            num_workers=bonus.DECODE_WORKERS,
        )
        decode_name = f"{name}_{label}_ds-{bonus.BASE_DECODE_SCALE}{suffix}"
        res = decode_and_score(
            decode_name, "cv", dc, pooled,
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
            f"{name}_{label}_freeseg-lm-{bonus.MATCHED_DECODE_LM}{suffix}", "cv", matched,
            frames, rasr_path=tools.RASR_PATH, device="cpu",
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

    # Emission balance on the alignment's silence-free measurement frames, for each
    # codebook. The price is the same 0.944-nat bonus either way of writing it.
    lm_cost = PhonemeLMCostJob(
        arpa_path=phonetic_lm_dict[bonus.LM_ORDER],
        lexicon=fb_lexicon,
        backoff_fb_root=tools.BACKOFF_FB,
    ).out_cost
    penalty = transition_boundary_penalty(bonus.NUM_STATES, reference_p, bonus.TRANSITION_SCALE)
    balance_frames = {
        "segments": BalanceFramesJob(features_hdf=nosil.out_features, frame_labels=nosil.out_labels,
                                     centroids=COLLEAGUE_CENTROIDS_K512, covs=ref_covs).out_frames,
        "frames-vad": BalanceFramesJob(features_hdf=nosil.out_features,
                                       frame_labels=nosil.out_labels,
                                       centroids=codebook, covs=vad_covs).out_frames,
    }

    def balance(mixtures, codebook_name):
        job = EmissionBalanceJob(
            frames=balance_frames[codebook_name],
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

    # --- reports ----------------------------------------------------------------
    passes = Column("passes", ("Corpus", "passes"),
                    lambda row: row.values.get("passes"), fmt="{:g}")
    update = relabel("epoch", "Update")
    arm_columns = [param("run", header=("Run",)), param("data", header=("Training", "silence")),
                   param("lm", header=("Training", "LM"))]
    silence_columns = [
        statistic("silence_share_soft", ("Sil.", "share [%]"), fmt="{:.1f}", scale=100),
        statistic("silence_share_argmax", ("Sil. share", "argmax [%]"), fmt="{:.1f}", scale=100),
        statistic("silence_duration_argmax", ("Sil. dur.", "(argmax)"), fmt="{:.2f}"),
        statistic("speech_duration_argmax", ("Phon. dur.", "(argmax)"), fmt="{:.2f}"),
    ]
    step_columns = [
        statistic("mixture_kl_em_step", ("EM step", "KL [nats]"), fmt="{:.2e}"),
        statistic("mixture_kl_applied_step", ("Applied", "KL [nats]"), fmt="{:.2e}"),
        statistic("mixture_clipped_fraction", ("At floor", "[%]"), fmt="{:.1f}", scale=100),
    ]
    report = LatexTableReport(
        columns=arm_columns + [param("cv", header=("cv", "silence")), update, passes,
                               "per", "del", "ins", "sub", "matched_per", "matched_del",
                               "matched_ins", "matched_sub", "matched_hyp_ref",
                               "posterior_entropy"]
                + silence_columns
                + ["boundary_rate", "segment_gap", "boundary_payback", "payable_boundaries"],
        sort_by=["run", "data", "lm", "cv"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Silence kept (VAD) against silence removed by the alignment (oracle), in "
            f"training and in cv. S={bonus.NUM_STATES}, LM scale {bonus.LM_SCALE}; training "
            f"LM the original (random silence) or re-silenced to the VAD frames' "
            f"statistics, every decode with the original; am=ts={bonus.AM_SCALE}; "
            f"speech loop {SPEECH_LOOP} "
            f"with a {SPEECH_BONUS:g}-nat exit bonus per phoneme (the bonus-0.5 arm's), "
            f"silence loop {SILENCE_LOOP} with a {SILENCE_EXIT_OFFSET:g}-nat exit offset. "
            f"From scratch: uniform table, frame codebook fit on the VAD frames, online "
            f"EM ({BATCHES_PER_EPOCH} undamped updates per pass, over-relaxation "
            f"{OVER_RELAXATION:g}). Branch: the bonus-0.5 arm after epoch "
            f"{REFERENCE_EPOCHS} (segment codebook), continued the same way; update 0 is "
            f"that model. Decoded on mean-pooled cv at LM scale 1.0; the "
            f"free-segmentation columns decode cv frames with no boundaries given (loop "
            f"{bonus.FREE_SEG_LOOP}, LM scale {bonus.MATCHED_DECODE_LM}, a fixed "
            f"{int(100 * bonus.FREE_SEG_FRACTION)}\\% of cv)."
        ),
    )
    progress_report = TextTableReport(
        columns=arm_columns + [update, passes, "posterior_entropy", "cluster_perplexity"]
                + silence_columns + ["boundary_rate", "dead_clusters"] + step_columns,
        sort_by=["run", "data", "lm"],
        drop_empty_rows=True,
        title=(
            f"Per-update convergence with silence kept (VAD) or removed (oracle). A row "
            f"is the job of that update, measured with the previous update's model on "
            f"that step's quarter. Targets from the alignment: silence 8.9% of the VAD "
            f"frames in runs of 6.9 frames, phonemes 3.99 frames (the bonus-0.5 arm "
            f"converged at 4.03-4.13). On oracle-trimmed data every silence frame is "
            f"taken from speech; the converged arm gave <SIL> 3.4% there."
        ),
    )
    recog_results = []

    def add_decodes(name, params, model_dir, label, centroids, epoch, statistics, values):
        for condition in cv_sets:
            res, decoded = decode(name, model_dir, label, centroids, condition)
            recog_results.append(res)
            report.add_row(result=res, params=dict(params, cv=condition), epoch=epoch,
                           statistics=statistics, values={**values, **decoded})

    # --- update 0 of the branches: the converged model, on both cv conditions -----
    # Its row shows the reference's last pass (silence-free data, model 199).
    ref_stats = clustering_statistics_per_epoch(
        reference.out_epoch_statistics, name=reference_name, epoch_offset=REFERENCE_EPOCHS,
        lexicon=lexicon)
    add_decodes(reference_name, {"run": "branch", "lm": "original",
                                 "data": f"epoch {REFERENCE_EPOCHS} (trained without silence)"},
                reference.out_models[REFERENCE_EPOCHS], f"ep-{REFERENCE_EPOCHS}",
                COLLEAGUE_CENTROIDS_K512, 0, ref_stats,
                {**balance(reference.out_artifacts["mixtures"][REFERENCE_EPOCHS], "segments"),
                 "passes": 0})

    # --- the arms -------------------------------------------------------------------
    # The original LM on both data conditions; the re-silenced one on the VAD frames,
    # where it is the only difference from the original-LM VAD arm.
    from_scratch_start = {"centroids": codebook, "covs": vad_covs, "mixtures": uniform}
    arms = []
    for condition, frames in data.items():
        arms.append(("branch", condition, frames, branch_start, "segments", BRANCH_PASSES,
                     BRANCH_CHUNKS, BRANCH_DECODE_EVERY, "original"))
        arms.append(("from scratch", condition, frames, from_scratch_start, "frames-vad",
                     PASSES, CHUNKS, DECODE_EVERY, "original"))
    arms.append(("branch", "VAD", vad, branch_start, "segments", BRANCH_PASSES,
                 BRANCH_CHUNKS, BRANCH_DECODE_EVERY, "VAD silence"))
    arms.append(("from scratch", "VAD", vad, from_scratch_start, "frames-vad",
                 PASSES, CHUNKS, DECODE_EVERY, "VAD silence"))
    for (run_name, condition, frames, start, codebook_name, num_passes, num_chunks, every,
         lm_name) in arms:
        prefix = (f"{reference_name}_branch-{REFERENCE_EPOCHS}" if run_name == "branch" else
                  f"weights_tau-{bonus.TAU}_s-{bonus.NUM_STATES}_init-uniform"
                  f"_cb-{codebook_name}")
        name = (f"{prefix}_data-{condition.lower()}_speech-{SPEECH_LOOP:g}-exit-{SPEECH_BONUS:g}"
                f"_sil-{SILENCE_LOOP:g}-exit-{SILENCE_EXIT_OFFSET:g}"
                f"_online-b{BATCHES_PER_EPOCH}_ema-{EMA.alpha:g}_relax-{OVER_RELAXATION:g}"
                + ("" if lm_name == "original" else "_lm-vadsil"))
        result = chunked_clustering(
            num_epochs=num_passes,
            features_hdf=frames.out_features,
            flavor=flavor(**start, backoff_fb=silence_searches[lm_name],
                          over_relaxation=OVER_RELAXATION),
            segments=frames.out_segments,
            batches_per_epoch=BATCHES_PER_EPOCH,
            ema=EMA,
            num_chunks=num_chunks,
            rqmt=RQMT,
            alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
            **common,
        )
        # Step k recognizes with the model of update k - 1 and produces update k.
        jobs = {k: job for k, job in enumerate(result.jobs, start=1)}
        params = {"run": run_name, "data": condition, "lm": lm_name}
        statistics = {k: job.out_statistics for k, job in jobs.items()}
        stats = clustering_statistics_per_epoch(
            statistics, name=name, epoch_offset=1, lexicon=lexicon)
        progress = clustering_statistics_per_epoch(
            statistics, name=name, epoch_offset=0, lexicon=lexicon)
        for k in jobs:
            progress_report.add_row(params=dict(params), epoch=k, statistics=progress,
                                    values={"passes": k / BATCHES_PER_EPOCH})
        for k in range(every, len(jobs) + 1, every):
            job = jobs[k]
            add_decodes(name, params, job.out_model, f"upd-{k}", start["centroids"], k, stats,
                        {**balance(job.artifact("mixtures"), codebook_name),
                         "passes": k / BATCHES_PER_EPOCH})
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
