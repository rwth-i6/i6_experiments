
# via:
# /u/irie/setups/switchboard/2018-02-13--end2end-zeyer/config-train/bpe_1k.multihead-mlp-h1.red8.enc6l.encdrop03.decbs.ls01.pretrain2.nbd07.config
# Kazuki BPE1k baseline, from Interspeech paper.
# based on rna3c-attlstmr-irielm2.nohdf.dec5la-ff.mlr50.emit.rep.ctcalign.config
# differences:
# see notes-andre.txt

import os
import numpy
from subprocess import check_output, CalledProcessError
from TFUtil import DimensionTag

# task
use_tensorflow = True
task = config.value("task", "train")
device = "gpu"
full_sum_train = True

debug_mode = False
if int(os.environ.get("DEBUG", "0")):
    # print("** DEBUG MODE")
    debug_mode = True

if config.has("beam_size"):
    beam_size = config.int("beam_size", 0)
    print("** beam_size %i" % beam_size)
else:
    if task == "train":
        beam_size = 4
    else:
        beam_size = 12

_cf_cache = {}

def cf(filename):
    """Cache manager"""
    if filename in _cf_cache:
        return _cf_cache[filename]
    if debug_mode or check_output(["hostname"]).strip().decode("utf8") in ["cluster-cn-211", "sulfid"]:
        print("use local file: %s" % filename)
        return filename  # for debugging
    try:
        cached_fn = check_output(["cf", filename]).strip().decode("utf8")
    except CalledProcessError:
        print("Cache manager: Error occured, using local file")
        return filename
    assert os.path.exists(cached_fn)
    _cf_cache[filename] = cached_fn
    return cached_fn

# data
target = "bpe"
target_num_labels = 1030
targetb_num_labels = target_num_labels + 1  # with blank
targetb_blank_idx = 0
time_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="time")
output_len_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="output-len")  # it's downsampled time
# use "same_dim_tags_as": {"t": time_tag} if same time tag ("data" and "alignment"). e.g. for RNA. not for RNN-T.
extern_data = {
    "data": {"dim": 40, "same_dim_tags_as": {"t": time_tag}},  # Gammatone 40-dim
    target: {"dim": target_num_labels, "sparse": True},  # see vocab
    "alignment": {"dim": targetb_num_labels, "sparse": True, "same_dim_tags_as": {"t": output_len_tag}},
    "align_score": {"shape": (1,), "dtype": "float32"},
}
if task != "train":
    # During train, we add this via the network (from prev alignment, or linear seg). Otherwise it's not available.
    extern_data["targetb"] = {"dim": targetb_num_labels, "sparse": True, "available_for_inference": False}
EpochSplit = 6

_alignment = None

def get_sprint_dataset(data, hdf_files=None):
    assert data in {"train", "devtrain", "cv", "dev", "hub5e_01", "rt03s"}
    epoch_split = {"train": EpochSplit}.get(data, 1)
    corpus_name = {"cv": "train", "devtrain": "train"}.get(data, data)  # train, dev, hub5e_01, rt03s
    hdf_files = None
    if not full_sum_train and data in {"train", "cv", "devtrain"}:
        hdf_files = ["base/dump-align/data/%s.data-%s.hdf" % (_alignment, {"cv": "dev", "devtrain": "train"}.get(data, data))]

    # see /u/tuske/work/ASR/switchboard/corpus/readme
    # and zoltans mail https://mail.google.com/mail/u/0/#inbox/152891802cbb2b40
    files = {}
    files["config"] = "config/training.config"
    files["corpus"] = "/work/asr3/irie/data/switchboard/corpora/%s.corpus.gz" % corpus_name
    if data in {"train", "cv", "devtrain"}:
        files["segments"] = "dependencies/seg_%s" % {"train":"train", "cv":"cv_head3000", "devtrain": "train_head3000"}[data]
    files["features"] = "/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.%s.bundle" % corpus_name
    for k, v in sorted(files.items()):
        assert os.path.exists(v), "%s %r does not exist" % (k, v)
    estimated_num_seqs = {"train": 227047, "cv": 3000, "devtrain": 3000}  # wc -l segment-file

    args = [
        "--config=" + files["config"],
        lambda: "--*.corpus.file=" + cf(files["corpus"]),
        lambda: "--*.corpus.segments.file=" + (cf(files["segments"]) if "segments" in files else ""),
        lambda: "--*.feature-cache-path=" + cf(files["features"]),
        "--*.log-channel.file=/dev/null",
        "--*.window-size=1",
    ]
    if not hdf_files:
        args += [
            "--*.corpus.segment-order-shuffle=true",
            "--*.segment-order-sort-by-time-length=true",
            "--*.segment-order-sort-by-time-length-chunk-size=%i" % {"train": epoch_split * 1000}.get(data, -1),
        ]
    d = {
        "class": "ExternSprintDataset", "sprintTrainerExecPath": "sprint-executables/nn-trainer",
        "sprintConfigStr": args,
        "suppress_load_seqs_print": True,  # less verbose
    }
    d.update(sprint_interface_dataset_opts)
    partition_epochs_opts = {
        "partition_epoch": epoch_split,
        "estimated_num_seqs": (estimated_num_seqs[data] // epoch_split) if data in estimated_num_seqs else None,
    }
    if hdf_files:
        align_opts = {
            "class": "HDFDataset", "files": hdf_files,
            "use_cache_manager": True,
            "seq_list_filter_file": files["segments"],  # otherwise not right selection
            #"unique_seq_tags": True  # dev set can exist multiple times
            }
        align_opts.update(partition_epochs_opts)  # this dataset will control the seq list
        if data == "train":
            align_opts["seq_ordering"] = "laplace:%i" % (estimated_num_seqs[data] // 1000)
            align_opts["seq_order_seq_lens_file"] = "/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"
        d = {
            "class": "MetaDataset",
            "datasets": {"sprint": d, "align": align_opts},
            "data_map": {
                "data": ("sprint", "data"),
                # target: ("sprint", target),
                "alignment": ("align", "data"),
                #"align_score": ("align", "scores")
                },
            "seq_order_control_dataset": "align",  # it must support get_all_tags
        }
    else:
        d.update(partition_epochs_opts)
    return d

sprint_interface_dataset_opts = {
    "input_stddev": 3.,
    "bpe": {
        'bpe_file': '/work/asr3/irie/data/switchboard/subword_clean/ready/swbd_clean.bpe_code_1k',
        'vocab_file': '/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k',
        # 'seq_postfix': [0]  # no EOS needed for RNN-T
    }}

train = get_sprint_dataset("train")
dev = get_sprint_dataset("cv")
eval_datasets = {"devtrain": get_sprint_dataset("devtrain")}
cache_size = "0"
window = 1


# Note: We control the warmup in the pretrain construction.
learning_rate = 0.001
learning_rates = list(numpy.linspace(learning_rate * 0.1, learning_rate, num=10))  # warmup (not in original?)
min_learning_rate = learning_rate / 50.


def targetb_linear(source, **kwargs):
    from TFUtil import get_rnnt_linear_aligned_output
    enc = source(1, as_data=True, auto_convert=False)
    dec = source(0, as_data=True, auto_convert=False)
    enc_lens = enc.get_sequence_lengths()
    dec_lens = dec.get_sequence_lengths()
    out, out_lens = get_rnnt_linear_aligned_output(
        input_lens=enc_lens,
        target_lens=dec_lens, targets=dec.get_placeholder_as_batch_major(),
        blank_label_idx=targetb_blank_idx,
        targets_consume_time=True)
    return out

def targetb_linear_out(sources, **kwargs):
    from TFUtil import Data
    enc = sources[1].output
    dec = sources[0].output
    size = enc.get_sequence_lengths() #  + dec.get_sequence_lengths()
    #output_len_tag.set_tag_on_size_tensor(size)
    return Data(name="targetb_linear", sparse=True, dim=targetb_num_labels, size_placeholder={0: size})

def targetb_search_or_fallback(source, **kwargs):
    import tensorflow as tf
    from TFUtil import where_bc
    ts_linear = source(0)  # (B,T)
    ts_search = source(1)  # (B,T)
    l = source(2, auto_convert=False)  # (B,)
    return where_bc(tf.less(l[:, None], 0.01), ts_search, ts_linear)


#import_model_train_epoch1 = "base/data-train/base2.conv2l.specaug4a/net-model/network.160"
#_train_setup_dir = "data-train/base2.conv2l.specaug4a"
#model = _train_setup_dir + "/net-model/network"
preload_from_files = {
  #"base": {
  #  "init_for_train": True,
  #  "ignore_missing": True,
  #  "filename": "/u/zeyer/setups/switchboard/2018-10-02--e2e-bpe1k/data-train/base2.conv2l.specaug4a/net-model/network.160",
  #},
  #"encoder": {
  #  "init_for_train": True,
  #  "ignore_missing": True,
  #  "filename": "/u/zeyer/setups/switchboard/2017-12-11--returnn/data-train/#dropout01.l2_1e_2.6l.n500.inpstddev3.fl2.max_seqs100.grad_noise03.nadam.lr05e_3.nbm6.nbrl.grad_clip_inf.nbm3.run1/net-model/network.077",
  #},
  # "encoder": {
    # "init_for_train": True,
    # "ignore_missing": True,
    # "ignore_params_prefixes": {"output/"},
    # "filename": "/u/zeyer/setups/switchboard/2019-10-22--e2e-bpe1k/data-train/base2.conv2l.specaug4a.ctc.devtrain/net-model/network.150",
  # }
}
# lm_model_filename = "/work/asr3/irie/experiments/lm/switchboard/2018-01-23--lmbpe-zeyer/data-train/bpe1k_clean_i256_m2048_m2048.sgd_b16_lr0_cl2.newbobabs.d0.2/net-model/network.023"


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
    l2 = 0.0001
    EncKeyPerHeadDim = EncKeyTotalDim // AttNumHeads
    EncValueTotalDim = 2048
    EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
    LstmDim = EncValueTotalDim // 2

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



