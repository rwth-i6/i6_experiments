

from returnn_common.nn.encoder.blstm_cnn_specaug import BlstmCnnSpecAugEncoder


def get_net_dict(pretrain_idx):
    """
    :param int|None pretrain_idx: starts at 0. note that this has a default repetition factor of 6
    :return: net_dict or None if pretrain should stop
    :rtype: dict[str,dict[str]|int]|None
    """
    # Note: epoch0 is 0-based here! I.e. in contrast to elsewhere, where it is 1-based.
    # Also, we never use #repetition here, such that this is correct.
    # This is important because of sub-epochs and storing the HDF files,
    # to know exactly which HDF files cover the dataset completely.
    epoch0 = pretrain_idx
    net_dict = {}

    # network
    # (also defined by num_inputs & num_outputs)
    EncKeyTotalDim = 200
    AttNumHeads = 1  # must be 1 for hard-att
    AttentionDropout = 0.1
    EncKeyPerHeadDim = EncKeyTotalDim // AttNumHeads
    EncValueTotalDim = 2048
    EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
    LstmDim = EncValueTotalDim // 2
    l2 = 0.0001

    if pretrain_idx is not None:
        net_dict["#config"] = {}

        # Do this in the very beginning.
        #lr_warmup = [0.0] * EpochSplit  # first collect alignments with existing model, no training
        lr_warmup = list(numpy.linspace(learning_rate * 0.1, learning_rate, num=10))
        if pretrain_idx < len(lr_warmup):
            net_dict["#config"]["learning_rate"] = lr_warmup[pretrain_idx]


    # We import the model, thus no growing.
    start_num_lstm_layers = 2
    final_num_lstm_layers = 6
    num_lstm_layers = final_num_lstm_layers
    if pretrain_idx is not None:
        pretrain_idx = max(pretrain_idx, 0) // 5  # Repeat a bit.
        num_lstm_layers = pretrain_idx + start_num_lstm_layers
        pretrain_idx = num_lstm_layers - final_num_lstm_layers
        num_lstm_layers = min(num_lstm_layers, final_num_lstm_layers)

    if final_num_lstm_layers > start_num_lstm_layers:
        start_dim_factor = 0.5
        grow_frac = 1.0 - float(final_num_lstm_layers - num_lstm_layers) / (final_num_lstm_layers - start_num_lstm_layers)
        dim_frac = start_dim_factor + (1.0 - start_dim_factor) * grow_frac
    else:
        dim_frac = 1.

    time_reduction = [3, 2] if num_lstm_layers >= 3 else [6]

    if pretrain_idx is not None and pretrain_idx <= 1 and "learning_rate" not in net_dict["#config"]:
        # Fixed learning rate for the beginning.
        net_dict["#config"]["learning_rate"] = learning_rate

    net_dict["#info"] = {
        "epoch0": epoch0,  # Set this here such that a new construction for every pretrain idx is enforced in all cases.
        "num_lstm_layers": num_lstm_layers,
        "dim_frac": dim_frac,
    }

    # We use this pretrain construction during the whole training time (epoch0 > num_epochs).
    if pretrain_idx is not None and epoch0 % EpochSplit == 0 and epoch0 > num_epochs:
        # Stop pretraining now.
        return None

    net_dict.update({
        "source": {"class": "eval", "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"},
        "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

        # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
        "conv0": {"class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None, "with_bias": True, "trainable": True},  # (T,40,32)
        "conv0p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0"},  # (T,20,32)
        "conv1": {"class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None, "with_bias": True, "trainable": True},  # (T,20,32)
        "conv1p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1"},  # (T,10,32)
        "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)

        # Encoder LSTMs added below, resulting in "encoder0".

        "encoder": {"class": "copy", "from": "encoder0"},
        "enc_ctx0": {"class": "linear", "from": "encoder", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim, "L2": l2, "dropout": 0.2},
        "enc_ctx_win": {"class": "window", "from": "enc_ctx0", "window_size": 5},  # [B,T,W,D]
        "enc_val": {"class": "copy", "from": "encoder"},
        "enc_val_win": {"class": "window", "from": "enc_val", "window_size": 5},  # [B,T,W,D]

        "enc_ctx": {"class": "linear", "from": "encoder", "activation": "tanh", "n_out": EncKeyTotalDim, "L2": l2, "dropout": 0.2},

        "enc_seq_len": {"class": "length", "from": "encoder", "sparse": True},

        # for task "search" / search_output_layer
        "output_wo_b0": {
          "class": "masked_computation", "unit": {"class": "copy"},
          "from": "output", "mask": "output/output_emit"},
        "output_wo_b": {"class": "reinterpret_data", "from": "output_wo_b0", "set_sparse_dim": target_num_labels},
        "decision": {
            "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": target,
            'only_on_search': True},
    })

    # Add encoder BLSTM stack.
    src = "conv_merged"
    if num_lstm_layers >= 1:
        net_dict.update({
            "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": 1, "from": src, "trainable": True},
            "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": -1, "from": src, "trainable": True}})
        src = ["lstm0_fw", "lstm0_bw"]
    for i in range(1, num_lstm_layers):
        red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
        net_dict.update({
            "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
        src = "lstm%i_pool" % (i - 1)
        net_dict.update({
            "lstm%i_fw" % i: {"class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": 1, "from": src, "dropout": 0.3 * dim_frac, "trainable": True},
            "lstm%i_bw" % i: {"class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": -1, "from": src, "dropout": 0.3 * dim_frac, "trainable": True}})
        src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
    net_dict["encoder0"] = {"class": "copy", "from": src}  # dim: EncValueTotalDim

    # This is used for training.
    net_dict["lm_input0"] = {"class": "copy", "from": "data:%s" % target}
    net_dict["lm_input1"] = {"class": "prefix_in_time", "from": "lm_input0", "prefix": targetb_blank_idx}
    net_dict["lm_input"] = {"class": "copy", "from": "lm_input1"}

    def get_output_dict(train, search, targetb, beam_size=beam_size):
        return {
        "class": "rec",
        "from": "encoder",  # time-sync
        "include_eos": True,
        "back_prop": (task == "train") and train,
        "unit": {
            "am": {"class": "copy", "from": "data:source"},

            "enc_ctx_win": {"class": "gather_nd", "from": "base:enc_ctx_win", "position": ":i"},  # [B,W,D]
            "enc_val_win": {"class": "gather_nd", "from": "base:enc_val_win", "position": ":i"},  # [B,W,D]
            "att_query": {"class": "linear", "from": "am", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim},
            'att_energy': {"class": "dot", "red1": "f", "red2": "f", "var1": "static:0", "var2": None,
                           "from": ['enc_ctx_win', 'att_query']},  # (B, W)
            'att_weights0': {"class": "softmax_over_spatial", "axis": "static:0", "from": 'att_energy',
                            "energy_factor": EncKeyPerHeadDim ** -0.5},  # (B, W)
            'att_weights1': {"class": "dropout", "dropout_noise_shape": {"*": None},
                             "from": 'att_weights0', "dropout": AttentionDropout},
            "att_weights": {"class": "merge_dims", "from": "att_weights1", "axes": "except_time"},
            'att': {"class": "dot", "from": ['att_weights', 'enc_val_win'],
                    "red1": "static:0", "red2": "static:0", "var1": None, "var2": "f"},  # (B, V)


            "prev_out_non_blank": {
                "class": "reinterpret_data", "from": "prev:output", "set_sparse_dim": target_num_labels},
                # "class": "reinterpret_data", "from": "prev:output_wo_b", "set_sparse_dim": target_num_labels},  # [B,]
            "lm_masked": {"class": "masked_computation",
                "mask": "prev:output_emit",
                "from": "prev_out_non_blank",  # in decoding
                "masked_from": "base:lm_input" if task == "train" else None,  # enables optimization if used

                "unit": {
                "class": "subnetwork", "from": "data", "trainable": True,
                "subnetwork": {
                    "input_embed": {"class": "linear", "n_out": 256, "activation": "identity", "trainable": True, "L2": l2, "from": "data"},
                    "lstm0": {"class": "rec", "unit": "nativelstm2", "dropout": 0.2, "n_out": 1024, "L2": l2, "from": "input_embed", "trainable": True},
                    "output": {"class": "copy", "from": "lstm0"}
                    #"output": {"class": "linear", "from": "lstm1", "activation": "softmax", "dropout": 0.2, "n_out": target_num_labels, "trainable": False}
                }}},
            # "lm_embed_masked": {"class": "linear", "activation": None, "n_out": 256, "from": "lm_masked"},
            #"lm_unmask": {"class": "unmask", "from": "lm_masked", "mask": "prev:output_emit"},
            # "lm_embed_unmask": {"class": "unmask", "from": "lm_embed_masked", "mask": "prev:output_emit"},
            "lm": {"class": "copy", "from": "lm_embed_unmask"},  # [B,L]


            # joint network: (W_enc h_{enc,t} + W_pred * h_{pred,u} + b)
            # train : (T-enc, B, F|2048) ; (U+1, B, F|256)
            # search: (B, F|2048) ; (B, F|256)
            "readout_in": {"class": "linear", "from": ["am", "att", "lm_masked"], "activation": None, "n_out": 1000, "L2": l2, "dropout": 0.2,
            "out_type": {"batch_dim_axis": 2 if task == "train" else 0, "shape": (None, None, 1000) if task == "train" else (1000,),
            "time_dim_axis": 0 if task == "train" else None}}, # (T, U+1, B, 1000)

            "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},

            "label_log_prob": {
                "class": "linear", "from": "readout", "activation": "log_softmax", "dropout": 0.3, "n_out": target_num_labels},  # (B, T, U+1, 1030)
            "emit_prob0": {"class": "linear", "from": "readout", "activation": None, "n_out": 1, "is_output_layer": True},  # (B, T, U+1, 1)
            "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},  # (B, T, U+1, 1)
            "blank_log_prob": {"class": "eval", "from": "emit_prob0", "eval": "tf.log_sigmoid(-source(0))"},  # (B, T, U+1, 1)
            "label_emit_log_prob": {"class": "combine", "kind": "add", "from": ["label_log_prob", "emit_log_prob"]},  # (B, T, U+1, 1), scaling factor in log-space
            "output_log_prob": {"class": "copy", "from": ["blank_log_prob", "label_emit_log_prob"]},  # (B, T, U+1, 1031)

            "output_prob": {
                "class": "eval", "from": ["output_log_prob", "base:data:" + target, "base:encoder"], "eval": rna_loss,
                "out_type": rna_loss_out, "loss": "as_is",
            },



            # this only works when the loop has been optimized, i.e. log-probs are (B, T, U, V)
            "rna_alignment" : {"class": "eval", "from": ["output_log_prob", "base:data:"+target, "base:encoder"],
                    "eval": rna_alignment, "out_type": rna_alignment_out, "is_output_layer": True} if task == "train"  # (B, T)
            else {"class": "copy", "from": "output_log_prob"},

            # During training   : targetb = "target"  (RNA-loss)
            # During recognition: targetb = "targetb"
            'output': {
                'class': 'choice', 'target': targetb, 'beam_size': beam_size,
                'from': "output_log_prob", "input_type": "log_prob",
                "initial_output": 0,
                "cheating": "exclusive" if task == "train" else None,
                #"explicit_search_sources": ["prev:u"] if task == "train" else None,
                #"custom_score_combine": targetb_recomb_train if task == "train" else None
                "explicit_search_sources": ["prev:out_str", "prev:output"] if task == "search" else None,
                "custom_score_combine": targetb_recomb_recog if task == "search" else None
                },

            "out_str": {
                "class": "eval", "from": ["prev:out_str", "output_emit", "output"],
                "initial_output": None, "out_type": {"shape": (), "dtype": "string"},
                "eval": out_str},

            "output_is_not_blank": {"class": "compare", "from": "output", "value": targetb_blank_idx, "kind": "not_equal", "initial_output": True},

            # initial state=True so that we are consistent to the training and the initial state is correctly set.
            "output_emit": {"class": "copy", "from": "output_is_not_blank", "initial_output": True, "is_output_layer": True},

            "const0": {"class": "constant", "value": 0, "collocate_with": ["du", "dt"]},
            "const1": {"class": "constant", "value": 1, "collocate_with": ["du", "dt"]},

            # pos in target, [B]
            # "du": {"class": "switch", "condition": "output_emit", "true_from": "const1", "false_from": "const0"},
            # "u": {"class": "combine", "from": ["prev:u", "du"], "kind": "add", "initial_output": 0},

            # pos in input, [B]
            # RNA is time-sync, so we always advance t
            # output label: stay in t, otherwise advance t (encoder)
            # "dt": {"class": "switch", "condition": "output_is_not_blank", "true_from": "const0", "false_from": "const1"},
            # "t": {"class": "combine", "from": ["prev:t", "dt"], "kind": "add", "initial_output": 0},

            # stop at U+T
            # in recog: stop when all input has been consumed
            # in train: defined by target.
            # "end": {"class": "compare", "from": ["t", "base:enc_seq_len"], "kind": "greater_equal"},
            },
            # "target": targetb,
            # "size_target": targetb if task == "train" else None,
            # "max_seq_len": "max_len_from('base:encoder') * 2"  # actually N+T
        }

    if task == "train":
        net_dict["output"] = get_output_dict(train=True, search=False, targetb=target)
    else:
        net_dict["output"] = get_output_dict(train=True, search=True, targetb="targetb")

    if task in ("train", "forward"):
        net_dict["rna_alignment"] =  {"class": "copy", "from": ["output/rna_alignment"]}  # (B, T)

    return net_dict



network = get_net_dict(pretrain_idx=None)
search_output_layer = "decision"
debug_print_layer_output_template = True

# trainer
batching = "random"
# Seq-length 'data' Stats:
#  37867 seqs
#  Mean: 447.397258827
#  Std dev: 350.353162012
#  Min/max: 15 / 2103
# Seq-length 'bpe' Stats:
#  37867 seqs
#  Mean: 14.1077719386
#  Std dev: 13.3402518828
#  Min/max: 2 / 82
log_batch_size = True
batch_size = 4000
max_seqs = 200
max_seq_length = {"bpe": 75}
#chunking = ""  # no chunking
truncation = -1

def custom_construction_algo(idx, net_dict):
    # For debugging, use: python3 ./crnn/Pretrain.py config...
    return get_net_dict(pretrain_idx=idx)

# No repetitions here. We explicitly do that in the construction.
pretrain = {"copy_param_mode": "subset", "construction_algo": custom_construction_algo}


num_epochs = 250
model = "net-model/network"
cleanup_old_models = True
gradient_clip = 0
#gradient_clip_global_norm = 1.0
adam = True
optimizer_epsilon = 1e-8
accum_grad_multiple_step = 3
#debug_add_check_numerics_ops = True
#debug_add_check_numerics_on_output = True
stop_on_nonfinite_train_score = False
tf_log_memory_usage = True
gradient_noise = 0.0
# lr set above
learning_rate_control = "newbob_multi_epoch"
learning_rate_control_error_measure = "dev_error_output/output_prob"
learning_rate_control_relative_error_relative_lr = True
learning_rate_control_min_num_epochs_per_new_lr = 3
use_learning_rate_control_always = True
newbob_multi_num_epochs = 6
newbob_multi_update_interval = 1
newbob_learning_rate_decay = 0.7
learning_rate_file = "newbob.data"

# log
#log = "| /u/zeyer/dotfiles/system-tools/bin/mt-cat.py >> log/crnn.seq-train.%s.log" % task
log = "log/crnn.%s.log" % task
log_verbosity = 5



