import dataclasses

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import DecodingTensorMap

CART_TREE_DI = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/di.tree.xml.gz"
CART_TREE_DI_NUM_LABELS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/di.num_labels"
CART_TREE_TRI = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.tree.xml.gz"
CART_TREE_TRI_NUM_LABELS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.num_labels"

GMM_TRI_ALIGNMENT = "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/mm/alignment/AlignmentJob.hK21a0UU4iiJ/output/alignment.cache.bundle"
SCRATCH_ALIGNMENT = (
    "/work/asr3/raissi/shared_workspaces/gunz/dependencies/alignments/ls-960/scratch/10ms/alignment.cache.bundle"
)
SCRATCH_ALIGNMENT_DANIEL = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/alignments/ls-960/scratch/daniel-with-dc-detection/alignment.cache.bundle"

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

RASR_ROOT_NO_TF_APPTAINER = "/u/berger/software/rasr_apptainer_no_tf"
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

CONF_CHUNKING = "400:200"
CONF_FOCAL_LOSS = 2.0
CONF_LABEL_SMOOTHING = 0.0
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

CONF_FH_DECODING_TENSOR_CONFIG = dataclasses.replace(
    DecodingTensorMap.default(),
    in_encoder_output="length_masked/strided_slice",
    in_seq_length="extern_data/placeholders/centerState/centerState_dim0_size",
    out_encoder_output="encoder__output/output_batch_major",
    out_right_context="right__output/output_batch_major",
    out_left_context="left__output/output_batch_major",
    out_center_state="center__output/output_batch_major",
)
BLSTM_FH_DECODING_TENSOR_CONFIG = dataclasses.replace(
    CONF_FH_DECODING_TENSOR_CONFIG,
    in_encoder_output="concat_lstm_fwd_6_lstm_bwd_6/concat_sources/concat",
)
MLP_FH_DECODING_TENSOR_CONFIG = dataclasses.replace(
    CONF_FH_DECODING_TENSOR_CONFIG,
    in_encoder_output="linear__6/activation/Relu",
)
