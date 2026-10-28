"""Supervised comparison of pooling methods for frame subsampling.

For every subsampling factor and pooling method, a table is counted from the
pooled silence-free ls-100h frames and the reference label at the center of each
pooling window (supervised, no EM), and silence-free cv is decoded with it. The
window is 2k frames (overlapping, the pipeline default) or k (non-overlapping). This
is the ceiling each pooling allows before any unsupervised training, i.e. the
counterpart of epoch 0 of :mod:`.vq_frames_cheat_init`.

Decoding as in :mod:`.vq_frames_cheat_init`: 5-gram LM, all scales 1, silence
forbidden. The loop probabilities follow the factor, since a phoneme of ~4 frames
covers ~4/k pooled steps (``DECODE_LOOP_PROBS``). FER is scored at the pooled
rate, against the reference frame at each window center.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import COLLEAGUE_CENTROIDS_K512
from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import SupervisedVQTableJob
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import create_lexicon
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import (
    decode_and_score,
    DecodeConfig,
)
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import (
    DatasetConfig,
    SegmentFile,
)
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import LatexTableReport
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    TABLE_FLOOR,
    build_decode_config,
)
from i6_experiments.example_setups.guided_kmeans.config.vq_frames_cheat_init import (
    ACOUSTIC_SCALE,
    DECODE_BEAM_SIZE,
    LM_ORDER,
    LM_PATH,
    LM_SCALE,
    TRANSITION_SCALE,
    frame_features,
)

exp_dir = "vq_frames_pooling_probe"
version = 1

FACTORS = [1, 2, 4]
POOLINGS = ["maxpool_time_np", "avg_time_np", "normed_avg_time_np", "cosine_medoid_pool_np"]
#: Also run with non-overlapping windows (window = factor instead of 2 * factor); only
#: for the poolings that did well with overlapping ones.
NON_OVERLAPPING_POOLINGS = ["avg_time_np", "normed_avg_time_np"]
#: Mean phoneme duration ~4 frames: p = 0.75 at full rate (best in vq_frames_cheat_init),
#: ~2 steps at k=2 (p = 0.5), ~1 step at k=4. The first grid's best values were its lowest
#: ones (0.25 at k=2, 0.1 at k=4), so it extends down to 0, where every pooled step is a
#: new phoneme.
DECODE_LOOP_PROBS = {
    1: [0.75],
    2: [0.0, 0.1, 0.2, 0.25, 0.5, 0.75],
    4: [0.0, 0.01, 0.05, 0.1, 0.25, 0.5],
}


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    ls100, cv = frame_features()
    cv_dataset = DatasetConfig(
        audio_hdf_path=cv.out_features,
        sampling_method=SegmentFile(cv.out_segments),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=["factor", "pooling", "window", "loop", "table_acc", "per", "del", "ins", "sub", "fer"],
        sort_by=["factor", "pooling", "window", "loop"],
        epochs=None,
        drop_empty_rows=True,
        caption=(
            "Supervised tables counted from pooled silence-free ls-100h frames, decoded on "
            "silence-free cv frames with the same pooling. 'table\\_acc' is the counted "
            "table's accuracy on its training data; FER is scored at the pooled rate."
        ),
    )
    recog_results = []

    for factor in FACTORS:
        # Pooling does not apply at full rate: a single unpooled arm. kernel None is the
        # default window of 2 * factor, kernel = factor makes the windows non-overlapping.
        arms = (
            [(pooling, None) for pooling in POOLINGS]
            + [(pooling, factor) for pooling in NON_OVERLAPPING_POOLINGS]
            if factor > 1
            else [(None, None)]
        )
        for pooling, kernel in arms:
            subsampling = factor if factor > 1 else None
            name = (
                f"ls100-frames_supervised_sub-{factor}"
                + (f"_{pooling}" if pooling else "")
                + (f"_win-{kernel}" if kernel else "")
            )
            pooling_kwargs = (
                {"subsampling": subsampling, "pooling_function": pooling, "pooling_kernel_size": kernel}
                if subsampling
                else {}
            )
            table_job = SupervisedVQTableJob(
                features_hdf=ls100.out_features,
                labels=ls100.out_labels,
                centroids=COLLEAGUE_CENTROIDS_K512,
                table_floor=TABLE_FLOOR,
                heldout_fraction=0.0,
                **pooling_kwargs,
            )
            table_job.add_alias(f"guided_kmeans/{exp_dir}/tables/{name}")
            tk.register_output(f"guided_kmeans/{exp_dir}/tables/{name}.json", table_job.out_diagnostics)

            for loop_prob in DECODE_LOOP_PROBS[factor]:
                decode_name = f"{name}_loop-{loop_prob}"
                decode_config = DecodeConfig(
                    centroids=COLLEAGUE_CENTROIDS_K512,
                    model_dir=table_job.out_model,
                    recog_rasr_config=build_decode_config(
                        LM_PATH, LM_SCALE, loop_prob,
                        lm_order=LM_ORDER, beam_size=DECODE_BEAM_SIZE, transition_scale=TRANSITION_SCALE,
                        forbid_blank=True,
                    ),
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
                res.fer = FrameErrorRateJob(
                    res.frame_labels, cv.out_labels, lexicon,
                    subsampling=subsampling, pooling_kernel_size=kernel,
                ).out_fer
                tk.register_output(f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per)
                tk.register_output(f"guided_kmeans/{exp_dir}/eval/{decode_name}_fer", res.fer)
                recog_results.append(res)
                latex_report.add_row(
                    result=res,
                    params={
                        "factor": factor,
                        "pooling": pooling or "none",
                        "window": kernel or (2 * factor if subsampling else 1),
                        "loop": loop_prob,
                    },
                    epoch=None,
                    values={"table_acc": table_job.out_accuracy, "fer": res.fer},
                )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_{version}.tex")


def py():
    run()
