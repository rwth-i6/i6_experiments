from ...setups.fh.factored import PhoneticContext, PhonemeStateClasses
from ...setups.fh.decoder.search import DecodingTensorMap

N_PHONES = [1, 2, 3]
RAISSI_ALIGNMENT = "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/mm/alignment/AlignmentJob.hK21a0UU4iiJ/output/alignment.cache.bundle"

RASR_ROOT_FH_GUNZ = "/u/mgunz/src/fh_rasr/"
RASR_ROOT_RS_RASR_GUNZ = "/u/mgunz/src/rs_rasr/"

RETURNN_PYTHON_TF15 = "/u/mgunz/src/bin/returnn_tf1.15_launcher.sh"
RETURNN_PYTHON_GENERIC_TF15 = "/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1/bin/python3.8"

CONF_CHUNKING = "400:200"
CONF_NUM_TRAIN_EPOCHS = [600]
CONF_SIZES = [512]
CONF_LABEL_SMOOTHING = 0.2

CONF_SA_CONFIG = {
    "max_reps_time": 20,
    "min_reps_time": 0,
    "max_len_time": 20,  # 200*0.05
    "max_reps_feature": 1,
    "min_reps_feature": 0,
    "max_len_feature": 15,
}

FH_LOSS_SHARE_FINAL_OUT = 0.6
FH_LOSS_VARIANTS_MONO = [[(6, PhoneticContext.monophone, True)]]

FH_BOUNDARY_WE_CLASS = [PhonemeStateClasses.word_end]

L2 = 1e-6
LABEL_SMOOTHING = 0.2

FH_DECODING_TENSOR_CONFIG: DecodingTensorMap = {
    "in_encoder_output": "length_masked/strided_slice",
    "in_seq_length": "extern_data/placeholders/centerState/centerState_dim0_size",
    "out_encoder_output": "encoder__output/output_batch_major",
    "out_right_context": "right__output/output_batch_major",
    "out_left_context": "left__output/output_batch_major",
    "out_center_state": "center__output/output_batch_major",
}
