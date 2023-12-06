# Chunked Attention-based Encoder-Decoder Model for Streaming Speech Recognition
# https://arxiv.org/abs/2309.08436

# ted2: C=20, L=10:
# /work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-06-14--streaming-conf/work/i6_core/returnn/training/ReturnnTrainingJob.PbZzH3tZhCgT
# chunked_att_chunk-35_step-20_linDecay120_0.0002_decayPt0.3333333333333333_bs15000_accum2_winLeft0_endSliceStart0_endSlice20_memVariant1_memSize2_convCache1_useCachedKV_memSlice0-20_L0_C20_R15          7.52    7.14  avg

# libri: C=20, R=15:
# /work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-06-14--streaming-conf/work/i6_core/returnn/training/ReturnnTrainingJob.ThjimHBUuNFA
# chunked_att_chunk-35_step-20_linDecay300_0.0002_decayPt0.3333333333333333_bs15000_accum2_winLeft0_endSliceStart0_endSlice20_memVariant1_memSize2_convCache2_useCachedKV_memSlice0-20_L0_C20_R15         2.44         6.38          2.66          6.28  avg
# checkpoint: /u/zeineldeen/setups/ubuntu_22_setups/2023-06-14--streaming-conf/work/i6_core/returnn/training/AverageTFCheckpointsJob.5r6TB06ypiVq/output/model/average

# i6_experiments/users/zeineldeen/models/asr/encoder/conformer_encoder.py
# i6_experiments/users/zeineldeen/experiments/chunkwise_att_2023/librispeech_960/chunkwise_attention_asr_config.py
