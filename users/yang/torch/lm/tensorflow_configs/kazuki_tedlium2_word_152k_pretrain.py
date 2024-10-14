#!crnn/rnn.py
# kate: syntax python;
# multisetup: finished True; finished_reason 'unstable';

import os
from subprocess import check_output
from Util import cleanup_env_var_path
cleanup_env_var_path("LD_LIBRARY_PATH", "/u/zeyer/tools/glibc217")
#
# pretrain score:
#
# 40: EpochData(learningRate=0.20390682574579033, error={
# 'dev_error': 0.7438210640662316,
# 'dev_score_output/output': 4.232838513695084,
# 'dev_score_output/output:exp': 68.91256407344456,
# 'train_error': 0.7232437768053858,
# 'train_score_output/output': 4.204414941688513,
# 'train_score_output/output:exp': 66.98139817671094,
# }),

# task
use_tensorflow = True
task = "train"
device = "gpu"
multiprocessing = True
update_on_device = True

_cf_cache = {}

def cf(filename):
    """Cache manager"""
    if filename in _cf_cache:
        return _cf_cache[filename]
    if check_output(["hostname"]).strip() in ["cluster-cn-211", "sulfid"]:
        print("use local file: %s" % filename)
        return filename  # for debugging
    cached_fn = check_output(["cf", filename]).strip().decode("utf8")
    assert os.path.exists(cached_fn)
    _cf_cache[filename] = cached_fn
    return cached_fn

data_files = {
    "train": ["/work/asr3/irie/data/tedlium2/word_150k/data/train.en.gz",
              "/work/asr3/irie/data/tedlium2/word_150k/data/commoncrawl-9pc.en.gz",
              "/work/asr3/irie/data/tedlium2/word_150k/data/europarl-v7-6pc.en.gz",
              "/work/asr3/irie/data/tedlium2/word_150k/data/giga-fren-4pc.en.gz",
              "/work/asr3/irie/data/tedlium2/word_150k/data/news-18pc.en.gz",
              "/work/asr3/irie/data/tedlium2/word_150k/data/news-commentary-v8-9pc.en.gz",
              "/work/asr3/irie/data/tedlium2/word_150k/data/yandex-1m-31pc.en.gz"],
    "cv": ["/work/asr3/irie/data/tedlium2/word_150k/data/dev.en.gz"],
    "test": ["/work/asr3/irie/data/tedlium2/word_150k/data/eval.en.gz"]}
vocab_file = "/work/asr3/irie/data/tedlium2/word_150k/data/vocab.returnn.txt"

orth_replace_map_file = ""

num_inputs = 152267

# train dataset len: LmDataset, sequences: 2392275, frames: NumbersDict(numbers_dict={'data': 26700802}, broadcast_value=0)
#train_num_seqs = 40699501
#train_num_seqs = 40418260
seq_order_bin_size = 100
train_epoch_split = 10

epoch_split = {"train": train_epoch_split, "cv": 1, "test": 1}
seq_order = {
    "train": "random",
    "cv": "sorted",
    "test": "sorted"}

def get_dataset(data):
    assert data in ["train", "cv", "test"]
    return {
        "class": "LmDataset",
        "corpus_file": lambda: list(map(cf, data_files[data])),
        "orth_symbols_map_file": lambda: cf(vocab_file),
        "orth_replace_map_file": orth_replace_map_file,
        "word_based": True,
        "seq_end_symbol": "<sb>",
        "auto_replace_unknown_symbol": True,
        "unknown_symbol": "<unk>",
        "add_delayed_seq_data": True,
        "delayed_seq_data_start_symbol": "<sb>",
        "seq_ordering": seq_order[data],
        "partition_epoch": epoch_split[data]
    }

train = get_dataset("train")
dev = get_dataset("cv")
eval = get_dataset("test")
cache_size = "0"
window = 1

# network
# (also defined by num_inputs & num_outputs)
num_outputs = {"data": {"dim": num_inputs, "sparse": True, "dtype": "int32"}}  # sparse data
num_outputs["delayed"] = num_outputs["data"]
target = "data"

tf_session_opts = {'allow_soft_placement': True, 'log_device_placement': False}

# Transformer params.
num_layers = 8 
num_ff_per_block = 3
ff_dim = 4096
num_heads = 12
emb_dim = 128
qk_dim = 768
v_dim = qk_dim
trans_out_dim = qk_dim
dropout = 0.1
att_dropout = 0.1
act_func = "relu"

# Universal.
tied_params = False

# Output layer.
output_sampling_loss = False
output_num_sampled = 4096
output_use_full_softmax = False

# Input embedding.
place_emb_on_cpu = False  # E.g. for adagrad.

# Initializer.
forward_weights_initializer = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=1.0)"

network = {
'output': {'class': 'rec',
  'from': ['data:delayed'],
  'target': 'data',
  'trainable': True,
  'unit': {'target_embed_raw': { 'activation': None,
                                 "param_device": "CPU" if place_emb_on_cpu else None,
                                 'class': 'linear',
                                 'forward_weights_init': forward_weights_initializer,
                                 'from': ['data:source'],
                                 'n_out': emb_dim,
                                 'with_bias': False},
           'target_embed': {'class': 'dropout', 'dropout': dropout, 'from': ['target_embed_raw']},
           'target_embed_lin': { 'activation': None,
                                  'class': 'linear',
                                  'forward_weights_init': forward_weights_initializer,
                                  'from': ['target_embed'],
                                  'n_out': trans_out_dim,
                                  'with_bias': False},
           'dec_0': {'class': 'copy', 'from': ['dec_0_ff_out']},
           'dec_0_self_att_laynorm': {'class': 'layer_norm', 'from': ['target_embed_lin']},
           'dec_0_self_att_att': { 'attention_dropout': dropout,
                                    'attention_left_only': True,
                                    'class': 'self_attention',
                                    'forward_weights_init': forward_weights_initializer,
                                    'from': ['dec_0_self_att_laynorm'],
                                    'n_out': v_dim,
                                    'num_heads': num_heads,
                                    'total_key_dim': qk_dim},
           'dec_0_self_att_lin': { 'activation': None,
                                    'class': 'linear',
                                    'forward_weights_init': forward_weights_initializer,
                                    'from': ['dec_0_self_att_att'],
                                    'n_out': trans_out_dim,
                                    'with_bias': False},
           'dec_0_self_att_drop': {'class': 'dropout', 'dropout': dropout, 'from': ['dec_0_self_att_lin']},
           'dec_0_att_out': { 'class': 'combine',
                               'from': ['target_embed_lin', 'dec_0_self_att_drop'],
                               'kind': 'add',
                               'n_out': trans_out_dim,
                               'trainable': True},
           'dec_0_ff_laynorm': {'class': 'layer_norm', 'from': ['dec_0_att_out']},
           'dec_0_ff_conv1': { 'activation': act_func,
                                'class': 'linear',
                                'forward_weights_init': forward_weights_initializer,
                                'from': ['dec_0_ff_laynorm'],
                                'n_out': ff_dim,
                                'with_bias': True},
           'dec_0_ff_conv2': { 'activation': None,
                                'class': 'linear',
                                'dropout': dropout,
                                'forward_weights_init': forward_weights_initializer,
                                'from': ['dec_0_ff_conv1'],
                                'n_out': trans_out_dim,
                                'with_bias': True},
           'dec_0_ff_drop': {'class': 'dropout', 'dropout': dropout, 'from': ['dec_0_ff_conv2']},
           'dec_0_ff_out': {'class': 'combine', 'from': ['dec_0_att_out', 'dec_0_ff_drop'], 'kind': 'add', 'n_out': trans_out_dim},}}}


def add_trafo_layer(cur_lay_id, prev_lay_id, name_input):
  network['output']['unit']['dec_%(cur_lay_id)s' % {'cur_lay_id': cur_lay_id} ] = {
    'class': 'copy', 'from': ['dec_%(cur_lay_id)s_ff_out' % {'cur_lay_id': cur_lay_id} ]}  # Just renaming the block output to dec_N.

  network['output']['unit']['dec_%(cur_lay_id)s_self_att_laynorm' % {'cur_lay_id': cur_lay_id} ] = {
    'class': 'layer_norm', 'from': [name_input]}  # Actual input layer of the block.

  network['output']['unit']['dec_%(cur_lay_id)s_self_att_att' % {'cur_lay_id': cur_lay_id} ] = {
    'attention_dropout': dropout,
    'attention_left_only': True,
    'reuse_params': 'dec_0_self_att_att' if tied_params else None,
    'class': 'self_attention',
    'forward_weights_init': forward_weights_initializer,
    'from': ['dec_%(cur_lay_id)s_self_att_laynorm' % {'cur_lay_id': cur_lay_id}],
    'n_out': v_dim,
    'num_heads': num_heads,
    'total_key_dim': qk_dim}
  network['output']['unit']['dec_%(cur_lay_id)s_self_att_lin' % {'cur_lay_id': cur_lay_id} ] = {
    'activation': None,
    'class': 'linear',
    'reuse_params': 'dec_0_self_att_lin' if tied_params else None,
    'forward_weights_init': forward_weights_initializer,
    'from': ['dec_%(cur_lay_id)s_self_att_att' % {'cur_lay_id': cur_lay_id}],
    'n_out': trans_out_dim,
    'with_bias': False}
  network['output']['unit']['dec_%(cur_lay_id)s_self_att_drop' % {'cur_lay_id': cur_lay_id} ] = {
    'class': 'dropout', 'dropout': dropout, 'from': ['dec_%(cur_lay_id)s_self_att_lin' % {'cur_lay_id': cur_lay_id}]}
  network['output']['unit']['dec_%(cur_lay_id)s_att_out' % {'cur_lay_id': cur_lay_id} ] = {
    'class': 'combine',
    'from': [name_input, 'dec_%(cur_lay_id)s_self_att_drop' % {'cur_lay_id': cur_lay_id}],
    'kind': 'add',
    'n_out': trans_out_dim,
    'trainable': True}
  network['output']['unit']['dec_%(cur_lay_id)s_ff_laynorm' % {'cur_lay_id': cur_lay_id}] = {
    'class': 'layer_norm', 'from': ['dec_%(cur_lay_id)s_att_out' % {'cur_lay_id': cur_lay_id}]}
  network['output']['unit']['dec_%(cur_lay_id)s_ff_conv1' % {'cur_lay_id': cur_lay_id}] = {
                       'class': 'linear',
                       'activation': act_func,
                       'forward_weights_init': forward_weights_initializer,
                       'reuse_params': 'dec_0_ff_conv1' if tied_params else None,
                       'from': ['dec_%(cur_lay_id)s_ff_laynorm' % {'cur_lay_id': cur_lay_id}],
                       'n_out': ff_dim,
                       'with_bias': True}
  network['output']['unit']['dec_%(cur_lay_id)s_ff_conv2' % {'cur_lay_id': cur_lay_id} ] = {
                       'class': 'linear',
                       'activation': None,
                       'dropout': dropout,
                       'reuse_params': 'dec_0_ff_conv2' if tied_params else None,
                       'forward_weights_init': forward_weights_initializer,
                       'from': ['dec_%(cur_lay_id)s_ff_conv1' % {'cur_lay_id': cur_lay_id}],
                       'n_out': trans_out_dim,
                       'with_bias': True}
  network['output']['unit']['dec_%(cur_lay_id)s_ff_drop' % {'cur_lay_id': cur_lay_id}] = {
    'class': 'dropout', 'dropout': dropout, 'from': ['dec_%(cur_lay_id)s_ff_conv2' % {'cur_lay_id': cur_lay_id}]}
  network['output']['unit']['dec_%(cur_lay_id)s_ff_out' % {'cur_lay_id': cur_lay_id}] = {
    'class': 'combine', 'from': ['dec_%(cur_lay_id)s_att_out' % {'cur_lay_id': cur_lay_id}, 'dec_%(cur_lay_id)s_ff_drop' % {'cur_lay_id': cur_lay_id}],
    'kind': 'add', 'n_out': trans_out_dim}

  return 'dec_%(cur_lay_id)s' % {'cur_lay_id': cur_lay_id}



def add_dnn_layer(cur_lay_id, prev_lay_id, dnn_id, name_input):
  network['output']['unit']['dec_%(cur_lay_id)s_dnn_%(dnn_id)s' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id} ] = {
    'class': 'copy', 'from': ['dec_%(cur_lay_id)s_dnn_%(dnn_id)s_ff_out' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id} ]}  # Just renaming the block output to dec_N.

  network['output']['unit']['dec_%(cur_lay_id)s_dnn_%(dnn_id)s_ff_laynorm' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id}] = {
    'class': 'layer_norm', 'from': [name_input]}

  network['output']['unit']['dec_%(cur_lay_id)s_dnn_%(dnn_id)s_ff_conv1' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id}] = {
                       'class': 'linear',
                       'activation': 'relu',
                       'forward_weights_init': forward_weights_initializer,
                       'reuse_params': 'dec_0_ff_conv1' if tied_params else None,
                       'from': ['dec_%(cur_lay_id)s_dnn_%(dnn_id)s_ff_laynorm' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id}],
                       'n_out': ff_dim,
                       'with_bias': True}

  network['output']['unit']['dec_%(cur_lay_id)s_dnn_%(dnn_id)s_ff_conv2' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id} ] = {
                       'class': 'linear',
                       'activation': None,
                       'dropout': dropout,
                       'reuse_params': 'dec_0_ff_conv2' if tied_params else None,
                       'forward_weights_init': forward_weights_initializer,
                       'from': ['dec_%(cur_lay_id)s_dnn_%(dnn_id)s_ff_conv1' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id}],
                       'n_out': trans_out_dim,
                       'with_bias': True}

  network['output']['unit']['dec_%(cur_lay_id)s_dnn_%(dnn_id)s_ff_drop' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id}] = {
    'class': 'dropout', 'dropout': dropout, 'from': ['dec_%(cur_lay_id)s_dnn_%(dnn_id)s_ff_conv2' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id}]}

  network['output']['unit']['dec_%(cur_lay_id)s_dnn_%(dnn_id)s_ff_out' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id}] = {
    'class': 'combine', 'from': [name_input, 'dec_%(cur_lay_id)s_dnn_%(dnn_id)s_ff_drop' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id}],
    'kind': 'add', 'n_out': trans_out_dim}

  return 'dec_%(cur_lay_id)s_dnn_%(dnn_id)s' % {'cur_lay_id': cur_lay_id, 'dnn_id': dnn_id}



# Stack layers.
cur_lay_id = 1
prev_lay_id = 0
name_input = "dec_0"
for i in range(num_layers-1):
  name_input = add_trafo_layer(cur_lay_id, prev_lay_id, name_input)
  for j in range(num_ff_per_block):
    name_input = add_dnn_layer(cur_lay_id, prev_lay_id, j, name_input)
  cur_lay_id += 1
  prev_lay_id += 1

# Add the final layer norm.
network['output']['unit']['decoder'] = {'class': 'layer_norm', 'from': [name_input]}

# Add output layer and loss.
if output_sampling_loss:
  network['output']['unit']['output'] = {
    'class': 'softmax', 'dropout': dropout, 'use_transposed_weights': True,
    'loss_opts': {'num_sampled': output_num_sampled, 'use_full_softmax': output_use_full_softmax, 'nce_loss': False},
    'forward_weights_init': forward_weights_initializer,
    'loss': 'sampling_loss', 'target': 'data', 'from': ['decoder']}
else:
  network['output']['unit']['output'] = {
    'class': 'softmax',
    'dropout': dropout,
    'forward_weights_init': forward_weights_initializer,
    'from': ['decoder'],
    'loss': 'ce',
    'target': 'data',
    'with_bias': True}

############
# Training hyper-params.
batching = "random"
batch_size = 900
max_seq_length = 602
max_seqs = 64
chunking = "0"
#truncation = 15
num_epochs = 60
#pretrain = "default"
#pretrain_construction_algo = "from_input"
#gradient_clip = 1.
gradient_clip_global_norm = 1.
#gradient_clip = 0
#adam = True
#optimizer = {'beta1': 0.9, 'beta2': 0.999, 'class': 'Adam', 'epsilon': 1e-08}
gradient_noise = 0.
#learning_rate = 0.1
#learning_rate = 1.
learning_rate = 1.
learning_rate_control = "newbob_rel"
learning_rate_control_relative_error_relative_lr = False
newbob_multi_num_epochs = train_epoch_split
#newbob_multi_update_interval = 1
newbob_relative_error_div_by_old = True

newbob_learning_rate_decay = 0.95
newbob_relative_error_threshold = -0.007
newbob_multi_update_interval = 1
learning_rate_control_error_measure = "dev_score_output:exp"

learning_rate_file = "newbob.data"
model = "net-model/network"

calculate_exp_loss = True
cleanup_old_models = {"keep_best_n": 1, "keep_last_n": 2}

# log
log = "log/crnn.%s.log" % task
log_verbosity = 4

