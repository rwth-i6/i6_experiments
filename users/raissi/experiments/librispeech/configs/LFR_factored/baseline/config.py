import dataclasses

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import DecodingTensorMap

ALIGN_30MS_BLSTM_MP = "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/30ms/blstm-1-lr-v6-ss-4-mp-2,3-mp-2,4-bw-0.3-pC0.6-tdp-1.0-v2/alignment.cache.bundle"
ALIGN_40MS_BLSTM_MP = "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/40ms/blstm-1-lr-v6-ss-4-mp2,3-mp2,4-bw0.3-pC0.6-tdp1.0/alignment.cache.bundle"

ZHOU_SUBSAMPLED_ALIGNMENT = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/alignments/ls-960/scratch/zhou-subsample-4-dc-detection/alignment.cache.bundle"
ZHOU_ALLOPHONES = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/allophones/ls-960/zhou-allophones"

ALIGN_GMM_MONO_10MS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/alignments/ls-960/mono/AlignmentJob.lZiXlFMiSb0C/output/alignment.cache.bundle"
ALIGN_GMM_TRI_10MS = "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"
# "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/mm/alignment/AlignmentJob.hK21a0UU4iiJ/output/alignment.cache.bundle"
ALIGN_GMM_TRI_ALLOPHONES = "/work/asr4/raissi/setups/librispeech/960-ls/2022-03--adapt_pipeline/work/i6_core/lexicon/allophones/StoreAllophonesJob.bJ8Qty3dD2cO/output/allophones"
ALIGN_BLSTM_40MS = "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/40ms/blstm-1-lr-v6-ss-4-mp2,3-mp2,4-bw0.3-pC0.6-tdp1.0/alignment.cache.bundle"

FROM_SCRATCH_CV_INFO = {
    "train_segments": "/work/asr3/raissi/shared_workspaces/gunz/dependencies/segments/ls-segment-names-to-librispeech/ShuffleAndSplitSegmentsJob.hPMsdZr1PSjY/output/train.segments",
    "train-dev_corpus": "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/zhou-corpora/train-dev.corpus.xml",
    "cv_corpus": "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/zhou-corpora/train-dev.corpus.xml",
    "cv_segments": "/work/asr3/raissi/shared_workspaces/gunz/dependencies/segments/ls-segment-names-to-librispeech/ShuffleAndSplitSegmentsJob.hPMsdZr1PSjY/output/cv.segments",
    # "features_postpath_cv": "/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/FeatureExtraction.Gammatone.qrINHi3yh3GH/output/gt.cache.bundle",
    "features_postpath_cv": "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/zhou-corpora/FeatureExtraction.Gammatone.yly3ZlDOfaUm/output/gt.cache.bundle",
    "features_tkpath_train": "/work/asr_archive/assis/luescher/best-models/librispeech/960h_2019-04-10/FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.bundle",
}

BLAS_LIB = "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so"

RASR_ROOT_NO_TF_APPTAINER = "/work/tools/users/raissi/shared/mgunz/rasr_apptainer_ss_no_tf"
RASR_ROOT_NO_TF_WORK_TOOLS = "/work/tools/users/raissi/shared/mgunz/rasr_no_tf"
RASR_ROOT_U16_APPTAINER = "/work/tools/users/raissi/shared/mgunz/rasr_apptainer_tf2.3_u16"
RASR_ROOT_NO_TF = RASR_ROOT_U16_APPTAINER

RASR_ROOT_TF2_APPTAINER = "/work/tools/users/raissi/shared/mgunz/rasr_apptainer_tf2.8"
RASR_ROOT_TF2_WORK_TOOLS = "/work/tools/users/raissi/shared/mgunz/rasr_tf2"
RASR_ROOT_TF2 = RASR_ROOT_U16_APPTAINER

RETURNN_PYTHON_TF2_12 = "/u/mgunz/src/bin/returnn_tf2.12_launcher.sh"
RETURNN_PYTHON_APPTAINER = "/u/mgunz/src/bin/returnn_tf2.8_apptainer_launcher.sh"
RETURNN_PYTHON_APPTAINER_2_3 = "/u/mgunz/src/bin/returnn_tf2.3_apptainer_u16_launcher.sh"
RETURNN_PYTHON = RETURNN_PYTHON_APPTAINER_2_3

BLSTM_CHUNKING = "64:32"

CONF_CHUNKING_10MS = "400:200"  # divides cleanly by 2, 4, 8
CONF_CHUNKING_30MS = "402:201"  # divides cleanly 3
CONF_CHUNKING_60MS = "402:198"  # divides cleanly 6

CONF_FH_DECODING_TENSOR_CONFIG_TF1 = dataclasses.replace(
    DecodingTensorMap.default(),
    in_encoder_output="length_masked/strided_slice",
    in_seq_length="extern_data/placeholders/centerState/centerState_dim0_size",
    out_encoder_output="encoder-output/output_batch_major",
    out_right_context="right-output/output_batch_major",
    out_left_context="left-output/output_batch_major",
    out_center_state="center-output/output_batch_major",
)


CONF_FH_DECODING_TENSOR_CONFIG = dataclasses.replace(
    DecodingTensorMap.default(),
    in_encoder_output="encoder/add",
    in_seq_length="extern_data/placeholders/data/data_dim0_size",
    out_encoder_output="encoder__output/output_batch_major",
    out_right_context="right__output/output_batch_major",
    out_left_context="left__output/output_batch_major",
    out_center_state="center__output/output_batch_major",
    out_joint_diphone="output/output_batch_major",
)


BLSTM_FH_DECODING_TENSOR_CONFIG = dataclasses.replace(
    CONF_FH_DECODING_TENSOR_CONFIG,
    in_encoder_output="concat_lstm_fwd_6_lstm_bwd_6/concat_sources/concat",
)
BLSTM_FH_TINA_DECODING_TENSOR_CONFIG = dataclasses.replace(
    CONF_FH_DECODING_TENSOR_CONFIG,
    in_encoder_output="concat_fwd_6_bwd_6/concat_sources/concat",
)
MLP_FH_DECODING_TENSOR_CONFIG = dataclasses.replace(
    CONF_FH_DECODING_TENSOR_CONFIG,
    in_encoder_output="linear__6/activation/Relu",
)
TDNN_FH_DECODING_TENSOR_CONFIG = dataclasses.replace(
    CONF_FH_DECODING_TENSOR_CONFIG,
    in_encoder_output="gated__6/output/Add",
)

BLSTM_FH_DECODING_TENSOR_CONFIG_TF2 = dataclasses.replace(
    CONF_FH_DECODING_TENSOR_CONFIG,
    in_encoder_output="concat_lstm_fwd_6_lstm_bwd_6/concat_sources/concat",
    in_seq_length="extern_data/placeholders/data/data_dim0_size",
)
