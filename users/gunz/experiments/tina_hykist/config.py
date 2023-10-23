import dataclasses
from dataclasses import dataclass


CART_TREE_DI = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/di.tree.xml.gz"
CART_TREE_DI_NUM_LABELS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/di.num_labels"
CART_TREE_TRI = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.tree.xml.gz"
CART_TREE_TRI_NUM_LABELS = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.num_labels"

RAISSI_ALIGNMENT = "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/mm/alignment/AlignmentJob.hK21a0UU4iiJ/output/alignment.cache.bundle"

RASR_ROOT_FH_GUNZ = "/u/mgunz/src/fh_rasr/"
RASR_ROOT_RS_RASR_GUNZ = "/u/mgunz/src/rs_rasr/"

RETURNN_PYTHON_TF15 = "/u/mgunz/src/bin/returnn_tf1.15_launcher.sh"

BLSTM_CHUNKING = "64:32"

CONF_CHUNKING = "256:128"
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

@dataclass(eq=True, frozen=True)
class DecodingTensorMap:
    """Map of tensor names used during decoding."""

    in_classes: str
    """Name of the input tensor carrying the classes."""

    in_encoder_output: str
    """
    Name of input tensor for feeding in the previously obtained encoder output while
    processing the state posteriors.

    This can be different from `out_encoder_output` because the tensor name for feeding
    in the intermediate data for computing the output softmaxes can be different from
    the tensor name where the encoder-output is provided.
    """

    in_delta_encoder_output: str
    """
    Name of input tensor for feeding in the previously obtained delta encoder output.

    See `in_encoder_output` for further explanation and why this can be different
    from `out_delta_encoder_output`.
    """

    in_data: str
    """Name of the input tensor carrying the audio features."""

    in_seq_length: str
    """Tensor name of the tensor where the feature length is fed in (as a dimension)."""

    out_encoder_output: str
    """Name of output tensor carrying the raw encoder output (before any softmax)."""

    out_delta_encoder_output: str
    """Name of the output tensor carrying the raw delta encoder output (before any softmax)."""

    out_left_context: str
    """Tensor name of the softmax for the left context."""

    out_right_context: str
    """Tensor name of the softmax for the right context."""

    out_center_state: str
    """Tensor name of the softmax for the center state."""

    out_delta: str
    """Tensor name of the softmax for the delta output."""

    @classmethod
    def default(cls) -> "DecodingTensorMap":
        return DecodingTensorMap(
            in_classes="extern_data/placeholders/classes/classes",
            in_data="extern_data/placeholders/data/data",
            in_seq_length="extern_data/placeholders/data/data_dim0_size",
            in_delta_encoder_output="delta-ce/output_batch_major",
            in_encoder_output="encoder-output/output_batch_major",
            out_encoder_output="encoder-output/output_batch_major",
            out_delta_encoder_output="deltaEncoder-output/output_batch_major",
            out_left_context="left-output/output_batch_major",
            out_right_context="right-output/output_batch_major",
            out_center_state="center-output/output_batch_major",
            out_delta="delta-ce/output_batch_major",
        )

FH_DECODING_TENSOR_CONFIG = dataclasses.replace(
    DecodingTensorMap.default(),
    in_encoder_output="length_masked/strided_slice",
    in_seq_length="extern_data/placeholders/centerState/centerState_dim0_size",
    out_encoder_output="encoder__output/output_batch_major",
    out_right_context="right__output/output_batch_major",
    out_left_context="left__output/output_batch_major",
    out_center_state="center__output/output_batch_major",
)
