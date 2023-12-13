#!/bin/bash

if [ $# -ne 2 ]; then
    echo -e "Usage: rename_vars_checkpoint.sh checkpoint_path output_path"
    exit
fi

checkpoint_path=$1
output_path=$2

apptainer_image="/work/asr4/hilmes/apptainer/u16.sif"
returnn_folder="/u/atanas.gruev/setups/librispeech/2023-08-08-zhou-conformer-transducer/returnn/tools"

# pos_clipping mismatch: 16 (2*16+1 == 33) vs. 32 (2*32+1 == 65)
# "conformer_block_(\\d+)_self_att_ln_rel_pos_enc(.*):conformer_\1_mhsa_mod_relpos_encoding\2,"\


rules="conformer_block_0:conformer_block_,"\
"conformer_block_(\\d+)_conv_mod_depthwise_conv2(.*):conformer_\1_conv_mod_depthwise_conv\2,"\
"conformer_block_(\\d+)_conv_mod_pointwise_conv(\\d+)(.*):conformer_\1_conv_mod_pointwise_conv_\2\3,"\
"conformer_block_(\\d+)_ffmod_(\\d+)_ff1(.*):conformer_\1_ffmod_\2_linear_swish\3,"\
"conformer_block_(\\d+)_ffmod_(\\d+)_ff2(.*):conformer_\1_ffmod_\2_dropout_linear\3,"\
"conformer_block_(\\d+)_ln(.*):conformer_\1_output\2,"\
"conformer_block_(\\d+)_self_att_linear(.*):conformer_\1_mhsa_mod_att_linear\2,"\
"conformer_block_(\\d+)_self_att_ln(.*):conformer_\1_mhsa_mod_ln\2,"\
"conformer_block_(\\d+)_self_att(.*):conformer_\1_mhsa_mod_self_attention\2,"\
"conformer_block:conformer,"\
"source_linear:input_linear,"\
"subsample_conv0:conv_2,"\
"subsample_conv1:conv_3,"\
"conv0p:conv0p,"\
"conv0:conv_1"


cd ${returnn_folder}
apptainer exec -B /work/asr3 ${apptainer_image} python3 tf_rename_vars_checkpoint.py --checkpoint_path ${checkpoint_path} --output_path ${output_path} --rules ${rules}


