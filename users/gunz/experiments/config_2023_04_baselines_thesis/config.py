import dataclasses

from ...setups.fh.decoder.search import DecodingTensorMap

CART_TREE_DI = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/di.tree.xml.gz"
CART_TREE_DI_NUM_LABELS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/di.num_labels"
CART_TREE_TRI = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.tree.xml.gz"
CART_TREE_TRI_NUM_LABELS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.num_labels"

RAISSI_ALIGNMENT = "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/mm/alignment/AlignmentJob.hK21a0UU4iiJ/output/alignment.cache.bundle"

RASR_ROOT_FH_GUNZ = "/u/mgunz/src/fh_rasr/"
RASR_ROOT_RS_RASR_GUNZ = "/u/mgunz/src/rs_rasr/"

RETURNN_PYTHON_TF15 = "/u/mgunz/src/bin/returnn_tf1.15_launcher.sh"

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

FH_DECODING_TENSOR_CONFIG = dataclasses.replace(
    DecodingTensorMap.default(),
    in_encoder_output="length_masked/strided_slice",
    in_seq_length="extern_data/placeholders/centerState/centerState_dim0_size",
    out_encoder_output="encoder__output/output_batch_major",
    out_right_context="right__output/output_batch_major",
    out_left_context="left__output/output_batch_major",
    out_center_state="center__output/output_batch_major",
)
