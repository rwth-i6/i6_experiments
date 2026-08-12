"""Chunked guided k-means: FB search, Euclidean model, cheating segmentation, random init."""

from itertools import product

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
    PHONEME_FREQUENCIES_LS100H,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    RandomCentroidsJob,
    chunked_clustering,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_recog_rasr_config,
    create_lexicon,
)
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import decode_and_score, DecodeConfig
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import DatasetConfig, SegmentFile
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import LatexTableReport, clustering_statistics
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.centroid_metrics import (
    CentroidCosineSimilarityJob,
    PhonemeL1DistanceJob,
    AverageNamedScoreJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob

exp_dir = "fb_cheat_seg"


def run():
    use_eow_phonemes = False
    num_epochs = 10
    num_clusters = 40 if not use_eow_phonemes else 79
    input_data_key = "ls-100-segmented"

    num_chunks = 20
    num_workers = 8
    lm_order = 3
    subsampling = None
    seed = 42

    lm_scales = [1.0]
    loop_prob = 0.0  # forced by cheating segmentation
    distance_scales = [0.1]  # AM weight applied to emission scores before FB search

    decode_lm_scales = [5000.0]
    decode_loop_prob = 0.4
    decode_distance_scale = 1.0

    initial_centroids = RandomCentroidsJob(
        input_data[input_data_key]["features"], num_clusters, seed=seed
    ).out_centroids

    lexicon = create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False)

    cv_dataset_config = DatasetConfig(
        audio_hdf_path=input_data["cv"]["features"],
        sampling_method=SegmentFile(input_data["cv"]["segment_file"]),
        precomputed=True,
    )
    cv_seg_dataset_config = DatasetConfig(
        audio_hdf_path=input_data["cv-segmented"]["features"],
        sampling_method=SegmentFile(input_data["cv-segmented"]["segment_file"]),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=["lm_scale", "am_scale", "dec_lm", "epoch", "per", "del", "ins", "sub"],
        sort_by=["lm_scale", "am_scale", "dec_lm"],
        epochs=(0, num_epochs - 1),
        caption=f"Chunked FB k-means, cheating seg, random init: epoch 0 vs epoch {num_epochs - 1}.",
    )
    latex_report_seg = LatexTableReport(
        columns=["lm_scale", "am_scale", "dec_lm", "epoch", "per", "del", "ins", "sub"],
        sort_by=["lm_scale", "am_scale", "dec_lm"],
        epochs=(0, num_epochs - 1),
        caption=f"Chunked FB k-means, cheating seg decode, random init: epoch 0 vs epoch {num_epochs - 1}.",
    )
    recog_results = []
    recog_results_seg = []

    for lm_scale, distance_scale in product(lm_scales, distance_scales):
        exp_name = f"lm-{lm_order}-{lm_scale}_loop-{loop_prob}_am-{distance_scale}"

        recognition_config = create_recog_rasr_config(
            lm_scale=lm_scale,
            emission_scale=1.0,
            transition_scale=lm_scale,
            loop_probability=loop_prob,
            silence_loop_probability=loop_prob,
            use_forward_backward_search=True,
            lm_order=lm_order,
            use_eow_phonemes=use_eow_phonemes,
        )

        exp_result = chunked_clustering(
            num_epochs=num_epochs,
            features_hdf=input_data[input_data_key]["features"],
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            initial_centroids=initial_centroids,
            subsampling=subsampling,
            distance_scale=distance_scale,
            use_forward_backward=True,
            rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
            num_chunks=num_chunks,
            num_workers=num_workers,
            alias_prefix=f"guided_kmeans/{exp_dir}/{exp_name}",
        )

        tk.register_output(
            f"guided_kmeans/{exp_dir}/statistics/{exp_name}.json", exp_result.out_statistics
        )
        statistics = clustering_statistics(exp_result.out_statistics, name=exp_name, epoch_offset=1)

        for decode_lm_scale in decode_lm_scales:
            decode_name = exp_name + f"_dec-{decode_lm_scale}"
            recognition_config_decode = create_recog_rasr_config(
                lm_scale=decode_lm_scale,
                emission_scale=1.0,
                transition_scale=None,
                loop_probability=decode_loop_prob,
                silence_loop_probability=decode_loop_prob,
                lm_order=lm_order,
                use_eow_phonemes=use_eow_phonemes,
            )
            recognition_config_decode_seg = create_recog_rasr_config(
                lm_scale=decode_lm_scale,
                emission_scale=1.0,
                transition_scale=None,
                loop_probability=0.0,
                silence_loop_probability=0.0,
                lm_order=lm_order,
                use_eow_phonemes=use_eow_phonemes,
            )
            for recog_epoch in range(num_epochs + 1):
                decode_config = DecodeConfig(
                    centroids=exp_result.out_centroids[recog_epoch],
                    recog_rasr_config=recognition_config_decode,
                    distance_scale=decode_distance_scale,
                    subsampling=subsampling,
                    write_frame_labels=True,
                )
                res = decode_and_score(
                    decode_name + f"_epoch-{recog_epoch}",
                    "cv",
                    decode_config,
                    cv_dataset_config,
                    rasr_path=tools.RASR_PATH,
                    device="cpu",
                    corpus_key="train-other-960",
                )
                res.mean_cos_sim = CentroidCosineSimilarityJob(exp_result.out_centroids[recog_epoch]).out_mean_cos_sim
                res.l1_dist = PhonemeL1DistanceJob(exp_result.out_statistics, recog_epoch, PHONEME_FREQUENCIES_LS100H).out_l1_dist
                res.avg_am_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_am_score").out_avg_score
                res.avg_transition_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_transition_score").out_avg_score
                res.avg_lm_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_lm_score").out_avg_score
                if res.frame_labels is not None:
                    res.fer = FrameErrorRateJob(res.frame_labels, GMM_ALIGNMENT_CV, lexicon).out_fer
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/recognition/{decode_name}_epoch-{recog_epoch}_per",
                    res.per,
                )
                recog_results.append(res)
                latex_report.add_row(
                    result=res,
                    params={"lm_scale": lm_scale, "am_scale": distance_scale, "dec_lm": decode_lm_scale},
                    epoch=recog_epoch,
                    statistics=statistics,
                )

                decode_config_seg = DecodeConfig(
                    centroids=exp_result.out_centroids[recog_epoch],
                    recog_rasr_config=recognition_config_decode_seg,
                    distance_scale=decode_distance_scale,
                    subsampling=subsampling,
                )
                res_seg = decode_and_score(
                    decode_name + f"_seg_epoch-{recog_epoch}",
                    "cv",
                    decode_config_seg,
                    cv_seg_dataset_config,
                    rasr_path=tools.RASR_PATH,
                    device="cpu",
                    corpus_key="train-other-960",
                )
                res_seg.mean_cos_sim = CentroidCosineSimilarityJob(exp_result.out_centroids[recog_epoch]).out_mean_cos_sim
                res_seg.l1_dist = PhonemeL1DistanceJob(exp_result.out_statistics, recog_epoch, PHONEME_FREQUENCIES_LS100H).out_l1_dist
                res_seg.avg_am_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_am_score").out_avg_score
                res_seg.avg_transition_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_transition_score").out_avg_score
                res_seg.avg_lm_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_lm_score").out_avg_score
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/recognition/{decode_name}_seg_epoch-{recog_epoch}_per",
                    res_seg.per,
                )
                recog_results_seg.append(res_seg)
                latex_report_seg.add_row(
                    result=res_seg,
                    params={"lm_scale": lm_scale, "am_scale": distance_scale, "dec_lm": decode_lm_scale},
                    epoch=recog_epoch,
                    statistics=statistics,
                )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report.txt",
        values=create_report(recog_results),
        required=True,
    )
    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_seg.txt",
        values=create_report(recog_results_seg),
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/recognition/report_first_last.tex")
    latex_report_seg.register(f"guided_kmeans/{exp_dir}/recognition/report_seg_first_last.tex")


def py():
    run()
