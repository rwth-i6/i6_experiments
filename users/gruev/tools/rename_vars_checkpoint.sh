#!/bin/bash

if [ $# -ne 2 ]; then
    echo -e "Usage: rename_vars_checkpoint.sh checkpoint_path output_path"
    exit
fi

checkpoint_path=$1
output_path=$2

apptainer_image="/work/asr4/hilmes/apptainer/u16.sif"
returnn_folder="/u/atanas.gruev/setups/librispeech/2023-08-08-zhou-conformer-transducer/returnn/tools"

rules="conformer_block_(\\d+)_conv_mod_depthwise_conv2(.*):conformer_{}_conv_mod_depthwise_conv{},"\
"conformer_block_(\\d+)_conv_mod_pointwise_conv(\\d+)(.*):conformer_{}_conv_mod_pointwise_conv_{}{},"\
"conformer_block_(\\d+)_ffmod_(\\d+)_ff1(.*):conformer_{}_ffmod_{}_linear_swish{},"\
"conformer_block_(\\d+)_ffmod_(\\d+)_ff2(.*):conformer_{}_ffmod_{}_dropout_linear{},"\
"conformer_block_(\\d+)_ln(.*):conformer_{}_output{},"\
"conformer_block_(\\d+)_self_att_ln(.*):conformer_{}_mhsa_mod_ln{},"\
"conformer_block_(\\d+)_self_att_ln_rel_pos_enc(.*):conformer_{}_mhsa_mod_relpos_encoding{},"\
"conformer_block_(\\d+)_self_att(.*):conformer_{}_mhsa_mod_self_attention{},"\
"conformer_block_(\\d+)_self_att_linear(.*):conformer_{}_mhsa_mod_linear{}"

cd ${returnn_folder}
apptainer exec -B /work/asr3 ${apptainer_image} python3 tf_rename_vars_checkpoint.py --checkpoint_path ${checkpoint_path} --output_path ${output_path} --rules ${rules}


