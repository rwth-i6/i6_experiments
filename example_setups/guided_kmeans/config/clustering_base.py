from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.clustering_config import (
    clustering,
    ClusteringCallbackConfig,
    LateInitConfig,
    StreamingStandardInitializerConfig,
    PickleCheatingCentroidInitializerConfig
)

from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import create_recog_rasr_config, create_lexicon
from i6_experiments.example_setups.guided_kmeans.setup.phoneme_frequency import py as get_sampled_segments_file

verbosity = 1
num_epochs = 3

rasr_path = tk.Path("/work/asr3/michel/mann/tools/rasr/librasr_recog/arch/linux-x86_64-standard")

lexicon = create_lexicon()
recognition_config = create_recog_rasr_config(
    lm_scale=1.0,
    emission_scale=1.0,
    transition_scale=None,
    loop_probability=0.1,
)

initailizer_configs = {
    "random": StreamingStandardInitializerConfig(seed=42),
    "cheating": PickleCheatingCentroidInitializerConfig(
        centroids_path=tk.Path("/u/jxu/setups/unsupervised/2025-05-30--marten-unsupervised/output/cheating_centroids.pkl"),
        lexicon_path=lexicon,
    )
}

clustering_callback_config = ClusteringCallbackConfig(
    num_clusters=41,
    initializer_config=LateInitConfig(),
    recognition_config=recognition_config,
    lexicon_path=create_lexicon(),
    subsampling=3,
    rasr_path=rasr_path,
)

def py():
    for name, initializer_config in initailizer_configs.items():
        clustering_callback_config.initializer_config = initializer_config
        exp_result = clustering(
            num_epochs=num_epochs,
            sampled_segments=get_sampled_segments_file(),
            cluster_callback_config=clustering_callback_config,
        )

        fwd_job = exp_result.fwd_job

        fwd_job.add_alias(f"guided_kmeans/{name}")

        tk.register_output(
            f"test/clustering_fwd_job_returnn_config_{name}",
            fwd_job.out_returnn_config_file
        )