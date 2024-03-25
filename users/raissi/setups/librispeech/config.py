import dataclasses

from sisyphus import tk

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import DecodingTensorMap


RAISSI_ALIGNMENT_10ms = tk.Path(
    "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/mm/alignment/AlignmentJob.hK21a0UU4iiJ/output/alignment.cache.bundle",
    cached=True,
)
SCRATCH_ALIGNMENT_10ms = tk.Path(
    "/u/mgunz/gunz/dependencies/alignments/ls-960/scratch/daniel-with-dc-detection/alignment.cache.bundle", cached=True
)

CV_SEGMENTS = tk.Path(
    "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/cv_segments_new_recipes/dev-clean-other.segments",
    cached=True,
)
P_HMM_AM7T1_ALIGNMENT_40ms = tk.Path(
    "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/cv_alignment_phmmAm0.7T0.1/alignment.cache.bundle",
    cached=True,
)


BLSTM_CHUNKING = "64:32"
CONF_CHUNKING = "400:200"

CONF_FOCAL_LOSS = 2.0
CONF_LABEL_SMOOTHING = 0.2
CONF_NUM_EPOCHS = 600

CONF_SA_CONFIG = {
    "max_reps_time": 20,
    "min_reps_time": 0,
    "max_len_time": 20,  # 200*0.05
    "max_reps_feature": 1,
    "min_reps_feature": 0,
    "max_len_feature": 15,
}

L2 = 1e-6
LABEL_SMOOTHING = 0.2
