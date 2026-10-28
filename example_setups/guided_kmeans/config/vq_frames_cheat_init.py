"""Frame-level (unsegmented) VQ training from a cheating initialization.

Like :mod:`.vq_unsupervised_frames`, the features are the silence-free ls-100h
*frames*: the GMM alignment removes silence but no longer gives the phoneme
boundaries, so the search has to place them itself via the label loop.

Differences to :mod:`.vq_unsupervised_frames`:

* the initial table is supervised, counted from all silence-free ls-100h frames
  and their frame alignment (the frame-level counterpart of
  :mod:`.vq_unsupervised_long_cheat_init`), instead of a sigma draw;
* LM, transition and acoustic scale are all 1 and the loop probability is set
  directly, instead of the combined penalty ``lambda = ts * logit(p)``;
* one table update per epoch (no batches / EMA), as in the segment-level runs;
* silence (the blank label) is forbidden in training and decoding, since the
  features contain no silence frames;
* training uses the memory-efficient forward-backward build
  (``tools.RASR_PATH_FORWARD_BACKWARD_MEM_EFFICIENT``): the default build keeps
  every state and arc of an utterance until its backward pass, ~5.3 MB per frame
  per worker, which exhausted even 32 GB per task at frame level;
* the LM sentence-end score is applied at the end of every training utterance.
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
    SupervisedVQTableJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
    phonetic_lm_dict,
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
    TABLE_FLOOR,
    build_decode_config,
    build_vq_training,
)

exp_dir = "vq_frames_cheat_init"
version = 1

LM_ORDER = 5
LM_PATH = phonetic_lm_dict[LM_ORDER]
LM_SCALE = 1.0
TRANSITION_SCALE = 1.0
ACOUSTIC_SCALE = 1.0
BEAM_SIZE = 1000

#: Self-loop probability of a label (silence uses the same). The mean phoneme
#: duration in silence-free frames is ~4, i.e. p = 0.75. Add more values once the
#: epoch time of the first run is known.
LOOP_PROBS = [0.5, 0.6, 0.75, 0.9]

NUM_EPOCHS = 50
NUM_CHUNKS = 30
NUM_WORKERS = 9
EPOCH_RQMT = {"mem": 12}

DECODE_EPOCHS = [0, 1, 2, 5, 10, 20, 30, 40, 50]
DECODE_BEAM_SIZE = 1000


def frame_features():
    """Silence-free ls-100h and cv *frames* (silence removed, not pooled), as
    ``(ls100, cv)`` SegmentedFeaturesFromAlignmentJobs; the same jobs as in
    vq_unsupervised_frames."""
    ls100 = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["ls-100"]["features"],
        alignment=GMM_ALIGNMENT_LS960_FRAME,
        exclude_labels=(0,),
        pooling="none",
        rqmt={"cpu": 2, "mem": 16, "time": 12},
    )
    cv = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["cv"]["features"],
        alignment=GMM_ALIGNMENT_CV,
        exclude_labels=(0,),
        pooling="none",
    )
    return ls100, cv


def _for_loop(value, loop_prob, default):
    """``value`` for this loop probability: a per-loop dict, a plain value, or None for the default."""
    if isinstance(value, dict):
        value = value.get(loop_prob)
    return default if value is None else value


def run_frames(
    *,
    exp_dir,
    init_name,
    init_description,
    make_table,
    sigma=None,
    seed=None,
    loop_probs=None,
    subsampling=None,
    pooling_function=None,
    pooling_kernel_size=None,
    num_epochs=None,
    decode_epochs=None,
):
    """All frame-level runs of this setup for one initialization.

    :param init_name: part of every run name, e.g. ``cheat-init``
    :param init_description: how the report caption describes the initial table
    :param make_table: ``(ls100 features job, exp_dir, pooling kwargs) -> initial
        table`` (or None, then the ``sigma``/``seed`` NormalTableJob draw of
        :func:`.build_vq_training` is used)
    :param loop_probs: defaults to ``LOOP_PROBS``
    :param num_epochs, decode_epochs: default to ``NUM_EPOCHS`` / ``DECODE_EPOCHS``;
        either one value for every loop probability or a ``{loop_prob: value}`` dict.
        The epoch count is not part of an epoch job's hash, so raising it continues
        a run and reuses its finished epochs.
    :param subsampling, pooling_function, pooling_kernel_size: pool the frames
        over time (see :class:`...util.PoolingRegistry`), identically in training,
        the initial table, decoding and the FER; ``None`` keeps the full rate
    """
    if tools.RASR_PATH_FORWARD_BACKWARD_MEM_EFFICIENT is None:
        raise RuntimeError(
            "the frame-level runs need the memory-efficient forward-backward build "
            "(tools.RASR_PATH_FORWARD_BACKWARD_MEM_EFFICIENT), which is not set for this user"
        )
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    ls100, cv = frame_features()

    pooling_kwargs = (
        {"subsampling": subsampling, "pooling_function": pooling_function, "pooling_kernel_size": pooling_kernel_size}
        if subsampling
        else {}
    )
    initial_table = make_table(ls100, exp_dir, pooling_kwargs)

    cv_dataset = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=[
            "loop", "epoch",
            "mi", "per", "del", "ins", "sub", "fer",
            "log_likelihood", "posterior_entropy", "dead_clusters",
        ],
        sort_by=["loop"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            f"Frame-level discrete-HMM training on silence-free ls-100h frames, started "
            f"from {init_description}, decoded on silence-free cv frames. LM, "
            f"transition and acoustic scale are all 1; 'loop' is the label self-loop "
            f"probability, used for training and decoding alike. Under-segmentation shows "
            f"as deletions, over-segmentation as insertions."
        ),
    )
    recog_results = []

    for loop_prob in LOOP_PROBS if loop_probs is None else loop_probs:
        exp_name = f"ls100-frames_{init_name}_loop-{loop_prob}"
        _, exp_result = build_vq_training(
            features=ls100.out_features,
            lm_path=LM_PATH,
            sigma=sigma,
            seed=seed,
            num_epochs=_for_loop(num_epochs, loop_prob, NUM_EPOCHS),
            num_chunks=NUM_CHUNKS,
            lexicon=lexicon,
            alias_prefix=f"guided_kmeans/{exp_dir}/{exp_name}",
            num_workers=NUM_WORKERS,
            rqmt=EPOCH_RQMT,
            lm_order=LM_ORDER,
            beam_size=BEAM_SIZE,
            lm_scale=LM_SCALE,
            transition_scale=TRANSITION_SCALE,
            loop_prob=loop_prob,
            forbid_blank=True,
            apply_sentence_end_score=True,
            rasr_path=tools.RASR_PATH_FORWARD_BACKWARD_MEM_EFFICIENT,
            distance_scale=ACOUSTIC_SCALE,
            table=initial_table,
            **pooling_kwargs,
        )
        tk.register_output(
            f"guided_kmeans/{exp_dir}/statistics/{exp_name}.json",
            exp_result.out_statistics,
        )
        statistics = clustering_statistics_per_epoch(
            exp_result.out_epoch_statistics, name=exp_name, epoch_offset=1, lexicon=lexicon
        )

        recognition_config_decode = build_decode_config(
            LM_PATH, LM_SCALE, loop_prob,
            lm_order=LM_ORDER, beam_size=DECODE_BEAM_SIZE, transition_scale=TRANSITION_SCALE,
            forbid_blank=True,
        )
        for epoch in _for_loop(decode_epochs, loop_prob, DECODE_EPOCHS):
            decode_name = f"{exp_name}_ep-{epoch}"
            mi = MixtureDiagnosticsJob(exp_result.out_artifacts["table"][epoch]).out_mi
            decode_config = DecodeConfig(
                centroids=COLLEAGUE_CENTROIDS_K512,
                model_dir=exp_result.out_models[epoch],
                recog_rasr_config=recognition_config_decode,
                distance_scale=ACOUSTIC_SCALE,
                write_frame_labels=True,
                **pooling_kwargs,
            )
            res = decode_and_score(
                decode_name, "cv", decode_config, cv_dataset,
                rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960",
                alias_prefix=f"guided_kmeans/{exp_dir}",
            )
            res.fwd_job.rqmt["time"] = 8
            # Against the silence-free frame alignment (at the pooled rate if pooled),
            # so positions correspond.
            res.fer = FrameErrorRateJob(
                res.frame_labels, cv.out_labels, lexicon,
                subsampling=subsampling, pooling_kernel_size=pooling_kernel_size,
            ).out_fer
            tk.register_output(f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per)
            tk.register_output(f"guided_kmeans/{exp_dir}/eval/{decode_name}_fer", res.fer)
            recog_results.append(res)
            latex_report.add_row(
                result=res,
                params={"loop": loop_prob},
                epoch=epoch,
                statistics=statistics,
                values={"mi": mi, "fer": res.fer},
            )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def cheat_init_table(ls100, exp_dir, pooling_kwargs):
    """The supervised table counted from ls-100h, pooled like the training features."""
    cheat_init_job = SupervisedVQTableJob(
        features_hdf=ls100.out_features,
        labels=ls100.out_labels,
        centroids=COLLEAGUE_CENTROIDS_K512,
        table_floor=TABLE_FLOOR,
        heldout_fraction=0.0,
        **pooling_kwargs,
    )
    cheat_init_job.add_alias(f"guided_kmeans/{exp_dir}/cheat_init_table")
    tk.register_output(
        f"guided_kmeans/{exp_dir}/cheat_init_table/diagnostics.json",
        cheat_init_job.out_diagnostics,
    )
    return cheat_init_job.out_table


def run():
    run_frames(
        exp_dir=exp_dir,
        init_name="cheat-init",
        init_description="a supervised frame table",
        make_table=cheat_init_table,
    )


def py():
    run()
