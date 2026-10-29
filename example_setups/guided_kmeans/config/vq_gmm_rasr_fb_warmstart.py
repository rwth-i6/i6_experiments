"""The successful 2-states HMM run, continued from epoch 100 with the RASR forward-backward search.

The reference is ``vq_gmm_tau_bonus``'s bonus-0.5 arm with the segment codebook: a
Gaussian mixture over the frozen k512 codebook (weights only, tau=4), trained on the
silence-free ls-100h frames with free segmentation, S=2 tied states, loop 0.2061 at
transition scale 0.35, LM 5-gram at scale 1.0 with the sentence-end score, AM scale
0.35, from an exactly uniform table, by the ``backoff_fb`` CUDA op. Its free-segmentation
PER sits on a plateau for long (69.0% at 100, 58.6% at 150) and drops to 21.0% by 200.

Here the same model is trained on from his epoch-100 checkpoint, with only the training
search swapped: RASR's forward-backward search with ``num_hmm_states=2``
(``tools.RASR_PATH_FORWARD_BACKWARD_HMM_STATES``) instead of the CUDA op. Epoch k of this
run produces the model of update 100 + k, and it is decoded exactly as his run is, every
5 updates from 105 to 200, so the two PER curves can be laid on top of each other.

**Equivalence with the CUDA op**, checked on his epoch-100 model for 5 cv sequences
against the op's exact float64 reference: with silence as an ordinary label
(``silence_as_label``: 2 states, label transitions, LM token ``<SIL>``, as the op models
it), the only difference left is RASR's beam. At beam 100000 log Z agrees to 0.05 nats
and the per-frame label posteriors to 0.0008 (total variation); at beam 10000 to 0.9 nats
and 0.015, at beam 1000 to 5 nats and 0.064. As the blank RASR uses by default, silence
has one state and gets half the op's posterior mass, which leaves a 0.024 difference at
any beam.
"""

import math

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    COLLEAGUE_CENTROIDS_K512,
    GMM_ALIGNMENT_CV,
    GMM_ALIGNMENT_LS960_FRAME,
    INPUT_DATA as input_data,
)
from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
    SegmentedFeaturesFromAlignmentJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    chunked_clustering,
    mixture_flavor,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
    create_recog_rasr_config,
)
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import (
    decode_and_score,
    DecodeConfig,
)
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import (
    DatasetConfig,
    SegmentFile,
)
from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
)

exp_dir = "vq_gmm_rasr_fb_warmstart"
version = 1

#: His epoch-100 model (GuidedClusteringEpochJob alias epoch_100 of the bonus-0.5
#: segment-codebook arm): centroids.npy (the frozen k512 codebook), covs.npy (tau * the
#: global covariance) and mixtures.npy (the trained weights). In his work directory.
BRANCH_EPOCH = 100
BRANCH_MODEL = (
    "/work/asr4/mann/experiments/2026-06-09--guided-k-means/work/i6_experiments/example_setups/"
    "guided_kmeans/setup/chunked_clustering/GuidedClusteringEpochJob.FQnnugTx2gAG/output/model"
)

# -- the reference arm, as in vq_gmm_tau_bonus --------------------------------------
NUM_LABELS = 40
LM_ORDER = 5
LM_SCALE = 1.0
AM_SCALE = 0.35
NUM_STATES = 2
TRANSITION_SCALE = 0.35
#: The LM's measured cost of one more phoneme, and the share the topology pays back.
LM_PHONEME_COST = 1.888
BONUS_FRACTION = 0.5
MIXTURE_FLOOR = 1e-2


def bonus_lambda(fraction: float) -> float:
    """Per-state lambda for a per-phoneme bonus of ``fraction`` of the LM cost,
    rounded exactly as in vq_gmm_tau_bonus so the loop probability is the same to the bit."""
    return round(-fraction * LM_PHONEME_COST / NUM_STATES, 3) + 0.0


def loop_for_lambda(lam: float, transition_scale: float) -> float:
    """``p = sigmoid(lambda / ts)``."""
    return 1.0 / (1.0 + math.exp(-lam / transition_scale))


LOOP_PROB = loop_for_lambda(bonus_lambda(BONUS_FRACTION), TRANSITION_SCALE)  # 0.2061...

# -- the RASR training search ---------------------------------------------------------
#: States kept per frame, one arm each. The CUDA op is exact; see the module docstring
#: for what the beam costs in accuracy. 1000 is the practical setting, 10000 close to
#: the op's exact sum and ~11x slower per sequence: if the 1000 arm leaves his curve,
#: the 10000 arm says whether the beam is why.
BEAM_SIZES = [1000, 10000]
#: Warm-start epochs: 100 reaches update 200, his 21% model.
NUM_EPOCHS = 100

# -- evaluation, as in vq_gmm_tau_bonus -----------------------------------------------
#: Updates decoded, as his LONG_DECODE_EPOCHS from the branch on. The branch point
#: itself is decoded too: it is his model, so it must reproduce his 69.0% exactly.
DECODE_UPDATES = list(range(BRANCH_EPOCH, BRANCH_EPOCH + NUM_EPOCHS + 1, 5))
DECODE_BEAM_SIZE = 1000
MATCHED_DECODE_LM = 1.0
FREE_SEG_LOOP = 0.5
FREE_SEG_FRACTION = 0.1
BASE_DECODE_SCALE = 1.0
DECODE_WORKERS = 8

# -- resources, as the RASR FB frame-level runs (vq_frames_cheat_init) ----------------
NUM_CHUNKS = 30
NUM_WORKERS = 9
#: Per beam size (unhashed). Beam 1000 peaked at ~10.3 GB in its first epoch, so 16 GB
#: (the VQ frame-level runs got by with 12, but the GMM scoring here is heavier). Beam 10000
#: keeps ~2 MB more per frame and worker (beam test: ~200 B per kept state and frame), so
#: 9 workers on the longest sequences (p99 692 frames, max 1094) need roughly 17-28 GB.
EPOCH_RQMT = {1000: {"mem": 16}, 10000: {"mem": 32}}
#: Threads of the task's own process (unhashed; cpu = NUM_WORKERS + this). It scores the
#: features against the 512 full-covariance densities, which at 1 thread kept the 9 beam-1000
#: workers at ~1.2x concurrency (parent at 1.0 cores, ~2 h per chunk). The Mahalanobis pass
#: scales to ~3.5x at 4 threads (cpu_modern) and little beyond. Beam 10000's workers are
#: busy on their own (8.95x), so its parent stays at 1.
PARENT_THREADS = {1000: 4, 10000: 1}


def _decode_rasr_config(lm_scale, loop_prob, transition_scale):
    """vq_gmm_tau_bonus' decode configs: 1-state linear search."""
    return create_recog_rasr_config(
        lm_scale=lm_scale,
        emission_scale=1.0,
        transition_scale=transition_scale,
        loop_probability=loop_prob,
        silence_loop_probability=loop_prob,
        lm_order=LM_ORDER,
        use_eow_phonemes=False,
        max_beam_size=DECODE_BEAM_SIZE,
    )


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    # The training frames and cv sets of vq_gmm_tau_bonus, job for job.
    ls100 = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["ls-100"]["features"],
        alignment=GMM_ALIGNMENT_LS960_FRAME,
        exclude_labels=(0,),
        pooling="none",
        rqmt={"cpu": 2, "mem": 16, "time": 12},
    )
    cv = silence_free_cv_features()
    cv_frames = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["cv"]["features"],
        alignment=GMM_ALIGNMENT_CV,
        exclude_labels=(0,),
        pooling="none",
    ).out_features
    free_seg_segments = ShuffleAndSplitSegmentsJob(
        cv.out_segments,
        split={"decode": FREE_SEG_FRACTION, "rest": 1.0 - FREE_SEG_FRACTION},
    ).out_segments["decode"]
    cv_dataset = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )
    cv_frames_dataset = DatasetConfig(
        audio_hdf_path=cv_frames,
        sampling_method=SegmentFile(free_seg_segments),
        precomputed=True,
    )

    centroids = tk.Path(f"{BRANCH_MODEL}/centroids.npy")
    decode_ds = _decode_rasr_config(1.0, 0.0, None)
    decode_freeseg = _decode_rasr_config(MATCHED_DECODE_LM, FREE_SEG_LOOP, 1.0)
    recog_results = []

    def decode(name, model_dir):
        """Both of his decodes of one model: oracle segments ("_ds-1.0", cv pooled per
        alignment segment) and free segmentation ("_freeseg-lm-1.0", 10% of the
        silence-free cv frames, where his 21% comes from)."""
        ds_name = f"{name}_ds-{BASE_DECODE_SCALE}"
        res = decode_and_score(
            ds_name, "cv",
            DecodeConfig(
                centroids=centroids,
                model_dir=model_dir,
                recog_rasr_config=decode_ds,
                distance_scale=BASE_DECODE_SCALE,
                write_frame_labels=True,
                num_workers=DECODE_WORKERS,
            ),
            cv_dataset,
            rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960",
        )
        if res.frame_labels is not None:
            res.fer = FrameErrorRateJob(res.frame_labels, GMM_ALIGNMENT_CV, lexicon).out_fer
            tk.register_output(f"guided_kmeans/{exp_dir}/eval/{ds_name}_fer", res.fer)
        tk.register_output(f"guided_kmeans/{exp_dir}/per/{ds_name}_per", res.per)
        recog_results.append(res)

        fs_name = f"{name}_freeseg-lm-{MATCHED_DECODE_LM}"
        res = decode_and_score(
            fs_name, "cv",
            DecodeConfig(
                centroids=centroids,
                model_dir=model_dir,
                recog_rasr_config=decode_freeseg,
                distance_scale=AM_SCALE,
                num_workers=DECODE_WORKERS,
            ),
            cv_frames_dataset,
            rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960",
        )
        tk.register_output(f"guided_kmeans/{exp_dir}/per/{fs_name}_per", res.per)
        recog_results.append(res)

    base_name = (f"weights_tau-4.0_s-{NUM_STATES}_bonus-{BONUS_FRACTION}_loop-{round(LOOP_PROB, 4)}"
                 f"_ts-{TRANSITION_SCALE}_lm-{LM_SCALE}_am-{AM_SCALE}_cb-segments")
    # The branch point is his model, shared by every arm: decoded once.
    decode(f"{base_name}_ep-{BRANCH_EPOCH}", tk.Path(BRANCH_MODEL))

    for beam_size in BEAM_SIZES:
        train_config = create_recog_rasr_config(
            lm_scale=LM_SCALE,
            emission_scale=1.0,
            transition_scale=TRANSITION_SCALE,
            loop_probability=LOOP_PROB,
            silence_loop_probability=LOOP_PROB,
            use_forward_backward_search=True,
            lm_order=LM_ORDER,
            use_eow_phonemes=False,
            max_beam_size=beam_size,
            apply_sentence_end_score=True,
            num_hmm_states=NUM_STATES,
            silence_as_label=True,
        )
        flavor = mixture_flavor(
            centroids=centroids,
            covs=tk.Path(f"{BRANCH_MODEL}/covs.npy"),
            mixtures=tk.Path(f"{BRANCH_MODEL}/mixtures.npy"),
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=NUM_LABELS,
            distance_scale=AM_SCALE,
            use_forward_backward=True,
            update_densities=False,
            update_covariances=None,
            mixture_floor=MIXTURE_FLOOR,
            num_workers=NUM_WORKERS,
        )
        name = f"{base_name}_branch-{BRANCH_EPOCH}_rasr-fb_beam-{beam_size}"
        result = chunked_clustering(
            num_epochs=NUM_EPOCHS,
            features_hdf=ls100.out_features,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=NUM_LABELS,
            flavor=flavor,
            distance_scale=AM_SCALE,
            use_forward_backward=True,
            rasr_path=tools.RASR_PATH_FORWARD_BACKWARD_HMM_STATES,
            num_chunks=NUM_CHUNKS,
            num_workers=NUM_WORKERS,
            parent_threads=PARENT_THREADS[beam_size],
            rqmt=EPOCH_RQMT[beam_size],
            alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
        )
        tk.register_output(f"guided_kmeans/{exp_dir}/statistics/{name}.json", result.out_statistics)

        # Epoch e of this run produces the model of update BRANCH_EPOCH + e. Epoch 0 is
        # the initial model, i.e. his, which is decoded above already.
        for update in DECODE_UPDATES:
            epoch = update - BRANCH_EPOCH
            if epoch > 0 and epoch in result.out_models:
                decode(f"{name}_ep-{update}", result.out_models[epoch])

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )


def py():
    run()
