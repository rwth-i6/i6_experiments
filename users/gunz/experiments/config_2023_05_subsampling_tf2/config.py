import dataclasses

from ...setups.fh.decoder.search import DecodingTensorMap

CART_TREE_DI = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/di.tree.xml.gz"
CART_TREE_DI_NUM_LABELS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/di.num_labels"
CART_TREE_TRI = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.tree.xml.gz"
CART_TREE_TRI_NUM_LABELS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.num_labels"

ALIGN_30MS_CONF_V1 = "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/30ms/conf-1-lr-v6-ss-3-fs-3-bw-0.3-pC-0.6-tdp-0.1/alignment.cache.bundle"
ALIGN_30MS_CONF_V2 = "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/30ms/conf-1-lr-v6-ss-3-fs-3-bw-0.3-pC-0.6-tdp-0.1-v2/alignment.cache.bundle"
ALIGN_30MS_CONF_V3 = "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/30ms/conf-1-lr-v6-ss-3-fs-3-bw-0.3-pC-0.6-tdp-1.0-v3/alignment.cache.bundle"
ALIGN_30MS_BLSTM_V1 = "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/30ms/blstm-1-lr-v6-ss-3-fs-3-bw-0.3-pC-0.6-tdp-0.1-v1/alignment.cache.bundle"  # BLSTM bad
ALIGN_30MS_BLSTM_V2 = "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/30ms/blstm-1-lr-v6-ss-4-mp-2,3-mp-2,4-bw-0.3-pC0.6-tdp-1.0-v2/alignment.cache.bundle"  # BLSTM good
ALIGN_30MS_BLSTM_V3 = "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/30ms/blstm-1-tina-v3/alignment.cache.bundle"  # tina bad
ALIGN_30MS_BLSTM_V4 = "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/30ms/blstm-1-tina-v4/alignment.cache.bundle"  # tina good?

ZHOU_SUBSAMPLED_ALIGNMENT = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/alignments/ls-960/scratch/zhou-subsample-4-dc-detection/alignment.cache.bundle"
ZHOU_ALLOPHONES = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/allophones/ls-960/zhou-allophones"

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
RASR_ARCH = "linux-x86_64-standard"

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

CONF_CHUNKING_10MS = "400:200"  # divides cleanly by
CONF_CHUNKING_30MS = "402:201"  # divides cleanly 3

CONF_FOCAL_LOSS = 2.0
CONF_LABEL_SMOOTHING = 0.0
CONF_NUM_EPOCHS = 600
TEST_EPOCH = 65

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

CONF_FH_DECODING_TENSOR_CONFIG = dataclasses.replace(
    DecodingTensorMap.default(),
    in_encoder_output="encoder/add",
    in_seq_length="extern_data/placeholders/data/data_dim0_size",
    out_encoder_output="encoder__output/output_batch_major",
    out_right_context="right__output/output_batch_major",
    out_left_context="left__output/output_batch_major",
    out_center_state="center__output/output_batch_major",
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
