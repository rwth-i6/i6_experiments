## Differences:

- ( my (tf2.3 + rasr2.3 ) baseline ) : `/u/schupp/setups/ping_setup_refactor_tf23/output/conformer/baseline/conf2__baseline_chunk200.100_bs-6144_l2-7e-05_drop-0.05_ffdim-1538/returnn.config`

WER:
- dev-other (best ep 190):  9.32
- dev-clean      (ep 190):  3.61
- test-other     (ep 190):  9.53
- test-clean     (ep 190):  3.94
- num_params: 28.1449 mio

- ( config 1, ping best hybrid + CE + 4gramLM ) : `/work/asr3/luescher/hiwis/pzheng/librispeech/transformer_conformer_21_11_10/work/crnn/sprint_training/CRNNSprintTrainingJob.R6Ivh9zimxO3/output/crnn.config`
- or original config name path  : `output/conformer/specaug/conformer_standard_best_combination_down_up_3_layernorm_epochsplit_40/crnn.config`

WER: 
- dev-other      (ep ?):  6.7
- dev-clean      (ep ?):  2.9
- test-other     (ep ?):  7.1
- test-clean     (ep ?):  3.2
- num_params: 86.8285 mio

## Training (mine) -> (ping):

full epochs:
10 -> 15
(200 split: 20) -> (600 split 40)

chunking:
200:100 -> 400:200

ping has extra sprint dataset options:
- segment-order-sort-by-time-length-chunk-size=-1
- segment-order-sort-by-time-length=true

behavior_version:
12 -> None (0)

### learning rate:

ping:
- linear 0.0002 -> 0.0005 ( 10 sepochs )
- const 0.0005 ( 190 sepochs )
- loglr from 0.0005 -> 0.00001 ( 400 sepochs )

mine:
- 0.0002 (1 sepoch)
- 0.0005 constant ( 18 sepochs )
- loglr from 0.0005 ->  1e-5 ( 181 sepochs )


## General (mine) -> (ping)

residual dim:
256 -> 512

## Preprocessing (mine) -> (ping):

l2 on convolutions:
7e-5 -> None

dropout on embedding:
0.05 -> 0.0

2VGG preprocessing: identical

*extra:*
Ping uses feature stacking:
conv_merged -> stacking_window ( stride: 2, window_left: 2, window_right: 0, window_size: 3) -> merge_dims 2, 3 (F, F)
Results in 3x time downsampling

Ping uses auxilary loss:
After 6th conformer block:
transposed_conv ( upsample x3 ) -> reinterpred data ( size classes) -> relu, nout=256 -> linear nout=256 -> Softmax CE layer

Ping has transposed convolution upsampling 3x after conformer encoder layer_norm

## Conformer Modules ( mine ) -> ( ping )

### feed forward module:

ffdim:
1538 -> 2048

dropout on fflinear2 and ff out:
0.05 -> 0.1

L2 on fflinear2 and ff out:
7e-5 -> None

### self att module:

att-dim:
256 -> 512

num_heads:
4 -> 8

self attention:
left_only=True -> left_only=False

positional encoding:
None -> relpos as keyshift on attention (clipping 400 non fixed)

output dropout:
0.05 -> 0.1

l2 on att-linear and output:
7e-5 -> None

### convolution module:

first pointwise conv:
n_out 512 -> 1024 ( cause of residual dim 512)

norm:
batch_norm -> layer_norm

dropout on conv output:
0.05 -> 0.1


## Specaug settings

ping:
    max_len_feature = 15
    max_len_time = 20
    max_reps_feature = 1
    max_reps_time = 20
    min_reps_feature = 0
    min_reps_time = 0

mine:
    Not set using defaults ( as calculated in transform function )

    min_reps_time = 1
    max_reps_time = 10? # not sure tf.maximum(tf.shape(x)[1] // (max_len_time or 20), 1) # // 100, 1)

    max_reps_feature = 2
    min_reps_feature = 1
    max_len_time = 20
    max_len_feature = 10 # nor sure : ?tf.shape(x)[-1] // 5