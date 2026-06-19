from sisyphus import tk

import sys

from i6_experiments.example_setups.guided_kmeans.setup.clustering_config import (
    clustering,
    ClusteringCallbackConfig,
    LateInitConfig,
    StreamingStandardInitializerConfig,
    PickleCheatingCentroidInitializerConfig
)

from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import create_recog_rasr_config, create_lexicon
from i6_experiments.example_setups.guided_kmeans.setup.phoneme_frequency import get_sampled_segments_file
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import decode_and_score, DecodeConfig
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import DatasetConfig, RandomNumber, All
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import RecogConfig

verbosity = 1
num_epochs = 10

#rasr_path = tk.Path("/work/asr3/michel/mann/tools/rasr/librasr_recog/arch/linux-x86_64-standard")
rasr_path = tk.Path("/work/asr4/lkleppel/rasr_dev/ngram_linear_search/rasr2/arch/linux-x86_64-standard")    # for linear search

lexicon = create_lexicon()

initializer_configs = {
    "random": StreamingStandardInitializerConfig(seed=42),
    "cheating": PickleCheatingCentroidInitializerConfig(
       # centroids_path=tk.Path("/u/jxu/setups/unsupervised/2025-05-30--marten-unsupervised/output/cheating_centroids.pkl"),
        centroids_path=tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids_ls_100.pkl"),   # LS 100h
        lexicon_path=lexicon,
    )
}

parameters = [
            (10.0, 0.2, 0.1), (15.0, 0.2, 0.1), (20.0, 0.2, 0.1),
            (10.0, 0.1, 0.05), (15.0, 0.1, 0.05), (20.0, 0.1, 0.05),
             ]

def py():
    for name, initializer_config in initializer_configs.items():
        recog_results = []
        for lm_scale, loop_probability, silence_loop_probability in parameters:
            exp_name = f"lm-{lm_scale}_loop-{loop_probability}-sil-loop-{silence_loop_probability}"

            recognition_config = create_recog_rasr_config(
                lm_scale=lm_scale,
                emission_scale=1.0,
                transition_scale=None,
                loop_probability=loop_probability,
                silence_loop_probability=silence_loop_probability,
                use_tree_search=False
            )

            clustering_callback_config = ClusteringCallbackConfig(
                num_clusters=41,
                initializer_config=LateInitConfig(),
                recognition_config=recognition_config,
                lexicon_path=create_lexicon(),
                subsampling=3,
                rasr_path=rasr_path,
            )
            clustering_callback_config.initializer_config = initializer_config

            exp_result = clustering(
                num_epochs=num_epochs,
                sampled_segments=get_sampled_segments_file(min_phoneme_count=sys.maxsize),
                cluster_callback_config=clustering_callback_config,
            )

            fwd_job = exp_result.fwd_job

            fwd_job.add_alias(f"guided_kmeans/ls-100/{name}/{exp_name}")

            tk.register_output(
                f"guided_kmeans/statistics/ls-100/{name}/{exp_name}.json",
                exp_result.out_statistics
            )

           # tk.register_output(
           #     f"guided_kmeans/clustering_fwd_job_returnn_config_{name}",
           #     fwd_job.out_returnn_config_file
           # )

            recog_config = RecogConfig(
                lm_scale=lm_scale,
                loop_probability=loop_probability,
                silence_loop_probability=silence_loop_probability
            )

            dataset_config = DatasetConfig(
                #audio_hdf_path=tk.Path("/work/asr4/jxu/setups/pretraining/2025-02-28--best-rq-pretraining/work/i6_core/returnn/hdf/BlissToPcmHDFJob.vExsEVfudAcd/output/audio.hdf"),
                #sampling_method=RandomNumber(20)
                audio_hdf_path=tk.Path("/work/asr4/jxu/setups/pretraining/2025-02-28--best-rq-pretraining/work/i6_core/returnn/hdf/BlissToPcmHDFJob.Yl7xJWHh0bgs/output/audio.hdf"), # dev-clean
                sampling_method=All()
            )

            decode_config = DecodeConfig(
                centroids=exp_result.out_centroids[num_epochs-1],
                recog_config=recog_config,
                distance_scale=1.0,
                # dataset_config=dataset_config,
                subsampling=3,
            )

            res = decode_and_score(exp_name, "dev-clean", decode_config, dataset_config, rasr_path=rasr_path)
            tk.register_output(f"guided_kmeans/recognition/ls-100/{name}/{exp_name}_per", res.per)
            recog_results.append(res)

        tk.register_report(f"guided_kmeans/recognition/ls-100/{name}/report.txt", values=create_report(recog_results), required=True)

    return recog_results
