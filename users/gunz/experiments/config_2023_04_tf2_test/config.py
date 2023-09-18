import dataclasses

from ...setups.fh.decoder.search import DecodingTensorMap

CART_TREE_DI = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/di.tree.xml.gz"
CART_TREE_DI_NUM_LABELS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/di.num_labels"
CART_TREE_TRI = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.tree.xml.gz"
CART_TREE_TRI_NUM_LABELS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.num_labels"

RAISSI_ALIGNMENT = "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/mm/alignment/AlignmentJob.hK21a0UU4iiJ/output/alignment.cache.bundle"
SCRATCH_ALIGNMENT = (
    "/u/mgunz/gunz/dependencies/alignments/ls-960/scratch/daniel-with-dc-detection/alignment.cache.bundle"
)

FROM_SCRATCH_CV_INFO = {
    "train_segments": "/work/asr3/raissi/shared_workspaces/gunz/dependencies/segments/ls-segment-names-to-librispeech/ShuffleAndSplitSegmentsJob.hPMsdZr1PSjY/output/train.segments",
    "train-dev_corpus": "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/zhou-corpora/train-dev.corpus.xml",
    "cv_corpus": "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/zhou-corpora/train-dev.corpus.xml",
    "cv_segments": "/work/asr3/raissi/shared_workspaces/gunz/dependencies/segments/ls-segment-names-to-librispeech/ShuffleAndSplitSegmentsJob.hPMsdZr1PSjY/output/cv.segments",
    # "features_postpath_cv": "/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/FeatureExtraction.Gammatone.qrINHi3yh3GH/output/gt.cache.bundle",
    "features_postpath_cv": "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/data/zhou-corpora/FeatureExtraction.Gammatone.yly3ZlDOfaUm/output/gt.cache.bundle",
    "features_tkpath_train": "/work/asr_archive/assis/luescher/best-models/librispeech/960h_2019-04-10/FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.bundle",
}

RASR_ROOT_FH_GUNZ = "/u/mgunz/src/fh_rasr/"
RASR_ROOT_TF2 = "/u/raissi/dev/rasr_github/rasr_tf2/"
RETURNN_PYTHON_TF23 = "/u/mgunz/src/bin/returnn_tf2.3_launcher.sh"

# Ubuntu 22
RASR_ARCH = "linux-x86_64-standard"
RASR_ROOT_NO_TF = "/work/tools/users/raissi/shared/mgunz/rasr_no_tf"
RASR_ROOT_TF2_U22 = "/work/tools/users/raissi/shared/mgunz/rasr_tf2"

RETURNN_PYTHON_TF2_12 = "/u/mgunz/src/bin/returnn_tf2.12_launcher.sh"
# End ubuntu 22

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
CONF_CART_DECODING_TENSOR_CONFIG = dataclasses.replace(
    CONF_FH_DECODING_TENSOR_CONFIG,
    in_seq_length="extern_data/placeholders/classes/classes_dim0_size",
    out_encoder_output="length_masked/output_batch_major",
    out_center_state="output/output_batch_major",
)
