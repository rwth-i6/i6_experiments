import dataclasses
from i6_experiments.users.raissi.setups.common.decoder.factored_hybrid_search import DecodingTensorMap


RAISSI_ALIGNMENT = "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/mm/alignment/AlignmentJob.hK21a0UU4iiJ/output/alignment.cache.bundle"
SCRATCH_ALIGNMENT = (
    "/u/mgunz/gunz/dependencies/alignments/ls-960/scratch/daniel-with-dc-detection/alignment.cache.bundle"
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
