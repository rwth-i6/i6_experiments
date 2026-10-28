"""Chunked guided k-means: Viterbi search, Gaussian model, standard segmentation, random init."""

from itertools import product

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
    PHONEME_FREQUENCIES_LS100H,
)
from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    IdentityCovsJob,
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

exp_dir = "viterbi_cov"
version = 1


def run():
    use_eow_phonemes = False
    num_epochs = 10
    num_clusters = 40 if not use_eow_phonemes else 79
    feature_dim = 512
    input_data_key = "ls-100"

    num_chunks = 20
    num_workers = 8
    lm_order = 3
    subsampling = None
    seed = 42

    lm_scales = [50.0]
    transition_scale = None
    loop_probs = [0.4]

    decode_lm_scales = [5000.0]
    transition_scale_decode = None
    decode_loop_prob = 0.4

    train_beam_size = 100_000
    train_score_threshold = None
    decode_beam_size = 100_000
    decode_score_threshold = None


    initial_centroids = RandomCentroidsJob(
        input_data[input_data_key]["features"], num_clusters, seed=seed
    ).out_centroids
    initial_covs = IdentityCovsJob(num_clusters, feature_dim=feature_dim).out_covs

    lexicon = create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False)

    cv_dataset_config = DatasetConfig(
        audio_hdf_path=input_data["cv"]["features"],
        sampling_method=SegmentFile(input_data["cv"]["segment_file"]),
        precomputed=True,
    )

    latex_report = LatexTableReport(
        columns=["lm_scale", "loop_prob", "epoch", "per", "del", "ins", "sub"],
        sort_by=["lm_scale", "loop_prob"],
        epochs=(0, num_epochs - 1),
        caption=f"Chunked Viterbi k-means + cov, random init: epoch 0 vs epoch {num_epochs - 1}.",
    )
    recog_results = []

    for lm_scale, loop_prob in product(lm_scales, loop_probs):
        _ts = transition_scale if transition_scale is not None else lm_scale
        _beam = "inf" if train_beam_size is None else (f"{train_beam_size // 1000}k" if train_beam_size >= 1000 else str(train_beam_size))
        exp_name = f"seed-{seed}_lm-{lm_order}-{lm_scale}_ts-{_ts}_loop-{loop_prob}_beam-{_beam}"

        recognition_config = create_recog_rasr_config(
            lm_scale=lm_scale,
            emission_scale=1.0,
            transition_scale=transition_scale if transition_scale is not None else lm_scale,
            loop_probability=loop_prob,
            silence_loop_probability=loop_prob,
            lm_order=lm_order,
            use_eow_phonemes=use_eow_phonemes,
            max_beam_size=train_beam_size,
            score_threshold=train_score_threshold,
        )

        exp_result = chunked_clustering(
            num_epochs=num_epochs,
            features_hdf=input_data[input_data_key]["features"],
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            initial_centroids=initial_centroids,
            initial_covs=initial_covs,
            subsampling=subsampling,
            rasr_path=tools.RASR_PATH,
            num_chunks=num_chunks,
            num_workers=num_workers,
            alias_prefix=f"guided_kmeans/{exp_dir}/{exp_name}",
        )

        tk.register_output(
            f"guided_kmeans/{exp_dir}/statistics/{exp_name}.json", exp_result.out_statistics
        )
        statistics = clustering_statistics(exp_result.out_statistics, name=exp_name, epoch_offset=1)

        for decode_lm_scale in decode_lm_scales:
            _dts = transition_scale_decode if transition_scale_decode is not None else decode_lm_scale
            _dbeam = "inf" if decode_beam_size is None else (f"{decode_beam_size // 1000}k" if decode_beam_size >= 1000 else str(decode_beam_size))
            decode_name = exp_name + f"_dec-{decode_lm_scale}_dts-{_dts}_dloop-{decode_loop_prob}_dbeam-{_dbeam}"
            recognition_config_decode = create_recog_rasr_config(
                lm_scale=decode_lm_scale,
                emission_scale=1.0,
                transition_scale=transition_scale_decode,
                loop_probability=decode_loop_prob,
                silence_loop_probability=decode_loop_prob,
                lm_order=lm_order,
                use_eow_phonemes=use_eow_phonemes,
                max_beam_size=decode_beam_size,
                score_threshold=decode_score_threshold,
            )
            for recog_epoch in range(num_epochs + 1):
                decode_config = DecodeConfig(
                    centroids=exp_result.out_centroids[recog_epoch],
                    recog_rasr_config=recognition_config_decode,
                    distance_scale=1.0,
                    subsampling=subsampling,
                    write_frame_labels=True,
                    covs=exp_result.out_covs[recog_epoch],
                )
                res = decode_and_score(
                    decode_name + f"_ep-{recog_epoch}",
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
                tk.register_output(f"guided_kmeans/{exp_dir}/eval/{decode_name}_ep-{recog_epoch}_cos_sim", res.mean_cos_sim)
                tk.register_output(f"guided_kmeans/{exp_dir}/eval/{decode_name}_ep-{recog_epoch}_l1_dist", res.l1_dist)
                tk.register_output(f"guided_kmeans/{exp_dir}/eval/{decode_name}_ep-{recog_epoch}_avg_am_score", res.avg_am_score)
                tk.register_output(f"guided_kmeans/{exp_dir}/eval/{decode_name}_ep-{recog_epoch}_avg_transition_score", res.avg_transition_score)
                tk.register_output(f"guided_kmeans/{exp_dir}/eval/{decode_name}_ep-{recog_epoch}_avg_lm_score", res.avg_lm_score)
                if res.frame_labels is not None:
                    tk.register_output(f"guided_kmeans/{exp_dir}/eval/{decode_name}_ep-{recog_epoch}_fer", res.fer)
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/per/{decode_name}_ep-{recog_epoch}_per",
                    res.per,
                )
                recog_results.append(res)
                latex_report.add_row(
                    result=res,
                    params={"lm_scale": lm_scale, "loop_prob": loop_prob},
                    epoch=recog_epoch,
                    statistics=statistics,
                )

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )
    latex_report.register(f"guided_kmeans/{exp_dir}/tex/report_first_last_{version}.tex")


def py():
    run()
