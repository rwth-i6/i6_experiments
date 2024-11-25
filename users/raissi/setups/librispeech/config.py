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


SMBR_FEATUREFLOW_SS = "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/train.ss.feature.flow"
SMBR_FEATUREFLOW_SS_IVEC = "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/train.ss_ivec.feature.flow"
