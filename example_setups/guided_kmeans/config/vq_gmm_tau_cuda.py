"""The tau=4.0 arm of :mod:`.vq_gmm_tau`, twice, with the training search on the GPU.

Same model, same data, same LM, same decode path. The one thing that changes is
*how the training forward-backward is computed*: the RASR worker pool is
replaced by the ``backoff_fb`` CUDA op (``tools/backoff_fb``, SPEC.md there).

Why this arm. ``vq_gmm_tau`` is a run that works -- PER falls and keeps falling
-- so a change that breaks the search shows up as a worse number rather than as
nothing at all. It also runs a *standard* HMM topology: the features are
segment-pooled with silence dropped, one vector per alignment segment, so
``loop_probability = 0`` and one observation is one phoneme. That is exactly
what the op models, which makes the two searches comparable rather than merely
similar.

What should and should not move:

* **Wall-clock is the point.** The RASR arm spends its epoch in a 7-process
  pool; this one spends it in batched GPU calls. Measured on 400 sequences with
  a realistic length distribution (median 96 segments, 44,017 frames):
  **184 ms/sequence on a V100**, against RASR's own reported 11-14 s/sequence
  of worker time at ~3.5x effective concurrency, i.e. roughly 3.6 s/sequence of
  wall time. Call it 10-20x, and note the parent's Mahalanobis scoring moves to
  the GPU as well now that the task has one.
* **The reported log-likelihood is not comparable to the RASR arm's**, and
  lower here does not mean worse. Two measured reasons, neither a bug:
  RASR's forward-backward omits ``ln P(</s> | g)`` from the final weight (its
  log Z converges from below onto ``apply_sentence_end=False`` as the beam
  grows), and at ``BEAM_SIZE = 1000`` against 932,459 graph states it loses a
  further 2.5-4.4 nats to pruning. This op is exact and applies the term.
* **PER and FER are the comparison that means something.** They come from the
  same decode path in both arms, so they can be compared directly.
* **The parent got faster too, for free.** Its Mahalanobis scoring was 3-4 s
  per sequence single-threaded on CPU in the RASR arm; with a GPU in the rqmt
  ``GaussianModelNumpy`` uses it, and it measures 17 ms. The search now
  dominates the chunk task 11:1, which is what makes ``NUM_CHUNKS`` the knob
  that matters.

Two arms, differing only in ``apply_sentence_end``:

``eos``
    ``True`` -- a path pays ``ln P(</s> | g)`` to terminate. This is the model
    spec 2.3 defines and what the KenLM agreement (2.3e-5) validates.
``noeos``
    ``False`` -- the term is dropped, which is RASR's convention. Measured: its
    log Z converges from below onto this value as its beam grows, and with both
    corrections applied the two agree on 100% / 98.4% of frames.

So ``noeos`` is the strict like-for-like against ``vq_gmm_tau`` and ``eos`` is
the corrected model. Nobody has trained with the sentence-end term before, so
whether it helps is an open question rather than a formality -- which is why
both run.

``vq_gmm_tau`` is unchanged and stays the reference.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
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
    silence_free_cv_features,
    silence_free_ls100_features,
)

exp_dir = "vq_gmm_tau_cuda"
version = 1

# Everything below is copied from vq_gmm_tau so the two runs differ in the
# search and nothing else. Keep them in step if that config moves.
LM_ORDER = 5
SIGMA = 0.02
BEAM_SIZE = 1_000
NUM_LABELS = 40
NUM_CODEWORDS = 512
#: Chunks per epoch, which is exactly how many GPUs the arm holds at once:
#: each chunk is one array task and each task takes one card. The work divides
#: cleanly, so this is a straight trade of wall-clock against occupancy.
#:
#: Timed on a V100 by running one real chunk through run_chunk: 142 sequences,
#: 17,496 frames, 34 s in the scoring-and-recognition loop, i.e. 0.24 s per
#: sequence including the parent's share. The corpus is 28,234 sequences, so
#: per epoch, per arm (plus ~1.7 min of per-task startup: model build, ARPA
#: parse, graph compile, upload):
#:
#:     chunks   epoch    60 epochs   GPUs (both arms)
#:        4     29.9 min   29.9 h          8
#:        6     20.5 min   20.5 h         12
#:        8     15.8 min   15.8 h         16     <- here
#:       12     11.1 min   11.1 h         24
#:       16      8.8 min    8.8 h         32     (all of gpu_32gb)
#:
#: 8 puts both arms at 16 of gpu_32gb's 32 V100s, which leaves room for other
#: people. Drop to 6 or 4 when the partition is busy; going much above 12 buys
#: little, because the startup stops amortizing -- at 16 chunks it is a fifth
#: of the epoch, and it is pure repetition, the same graph built once per task.
NUM_CHUNKS = 6
SEED = 42
BASE_DISTANCE_SCALE = 1.0
TRAIN_DISTANCE_SCALE = 1.0
DECODE_DISTANCE_SCALES = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
SCALE_SWEEP_EPOCHS = [30, 60]

TAU = 4.0
WEIGHTS_EPOCHS = 60
WEIGHTS_DECODE_EPOCHS = [10, 20, 30, 40, 50, 60]

# The GPU replaces the worker pool, so the CPU request drops to what the parent
# still does itself (Mahalanobis scoring, accumulation) and one GPU is added.
# settings.py:worker_wrapper turns a GPU in the rqmt into `apptainer --nv`.
NUM_WORKERS = 0
PARENT_THREADS = 4
# gpu_mem = 32 selects gpu_32gb (V100) through settings.py:check_engine_limits.
# It is the card that is being asked for, not the memory: the working set here
# is ~4 GB, but the V100 and the A100 are the only GPUs on this cluster with
# FP64 at 1:2 rather than 1:32 or 1:64, and the LM step runs in float64
# (spec 3.5). Measured on 400 sequences, 44,017 frames: 1.670 ms/frame on the
# V100 against 3.174 on an A10.
#
# Leaving gpu_mem out is not neutral -- it routes to gpu_11gb with an explicit
# --gres=gpu:gtx_1080, which is compute capability 6.1 and 11 GB.
EPOCH_RQMT = {"mem": 16, "gpu": 1, "gpu_mem": 32, "cpu": 4}

#: Sequences per op call. These are segment-pooled features, so T is a phoneme
#: count (O(100)), not a frame count. Unhashed - throughput only.
BATCH_SIZE = 16
#: Caps B * T_max, which is what actually bounds peak memory. NQ is 932,459 at
#: order 5, so alpha-hat costs 4 * (B*T_max) * NQ bytes before checkpointing.
MAX_FRAMES = 4_000


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


#: The two arms: suffix -> apply_sentence_end.
ARMS = {"eos": True, "noeos": False}


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    ls100 = silence_free_ls100_features()
    ls100.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_nosil")
    cv = silence_free_cv_features()
    cv.add_alias(f"guided_kmeans/{exp_dir}/features_cv_nosil")

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
    # Still built: chunked_clustering takes it, and the decode side is
    # unchanged. The training search no longer reads it.
    train_config = _recog_config(1.0, 0.0, 1.0, use_fb=True)
    decode_config_rasr = _recog_config(1.0, 0.0, None, use_fb=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)

    latex_report = LatexTableReport(
        columns=["arm", "epoch", "per", "del", "ins", "sub", "log_likelihood"],
        sort_by=["arm"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"The tau={TAU} arm of vq\\_gmm\\_tau with the training "
            f"forward-backward computed exactly on the GPU (backoff\\_fb). "
            f"``eos`` applies $\\ln P(\\langle/s\\rangle \\mid g)$ to terminate, "
            f"``noeos`` drops it as RASR's search does. Model, data, LM and "
            f"decode path are identical across both."
        ),
    )
    recog_results = []

    for arm, apply_eos in ARMS.items():
        name = f"weights_tau-{TAU}_{arm}"
        # The same numbers train_config carries, as numbers the graph compiler
        # can use. The lexicon must be the forward-backward one: it fixes the
        # score column order (silence at 0, phonemes 1..N).
        backoff_fb = BackoffFBSearch(
            backoff_fb_root=tools.BACKOFF_FB,
            arpa_path=phonetic_lm_dict[LM_ORDER],
            lexicon=fb_lexicon,
            lm_scale=1.0,
            emission_scale=1.0,   # as passed to create_recog_rasr_config above
            loop_probability=0.0,
            silence_loop_probability=0.0,
            transition_scale=1.0,
            apply_sentence_end=apply_eos,
            batch_size=BATCH_SIZE,
            max_frames=MAX_FRAMES,
            checkpoint_interval=-1,   # the spec-8.1 optimum per batch
            storage_dtype="float32",
        )
        flavor = mixture_flavor(
            centroids=COLLEAGUE_CENTROIDS_K512,
            covs=covs,
            mixtures=init_table,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=NUM_LABELS,
            distance_scale=TRAIN_DISTANCE_SCALE,
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
            distance_scale=TRAIN_DISTANCE_SCALE,
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

        for e in WEIGHTS_DECODE_EPOCHS:
            scales = (
                DECODE_DISTANCE_SCALES if e in SCALE_SWEEP_EPOCHS
                else [BASE_DISTANCE_SCALE]
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
                if scale == BASE_DISTANCE_SCALE:
                    latex_report.add_row(
                        result=res, params={"arm": arm}, epoch=e,
                        statistics=stats,
                        values={"fer": res.fer} if res.fer is not None else {},
                    )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results), required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def py():
    run()
