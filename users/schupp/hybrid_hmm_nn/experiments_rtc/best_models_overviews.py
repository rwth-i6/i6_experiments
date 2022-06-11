# This contains all the 'best models'
# Note that the results are from the train configs `baseline_*_big_short` and `baseline_*_big_long`
# But this file should be a smaller overview of the best models
from typing import OrderedDict
from sisyphus import gs

OUTPUT_PATH = "conformer/all_baselines/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH
BASE = "all_baselines"

# This contains *all* params for Baseline II, ( note that these are progressively modified in other baselines )
class original_args_baseline_II:
    def lr(self, warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
            min_lr_ratio=1/50, decay_factor=0.99):
        import numpy
        import math
        num_lr = int(math.log(min_lr_ratio, decay_factor))
        return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                        [start] * constant_subepoch + \
                        list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                        [min_lr_ratio * start]
    def __init__(self):
        self.EP_SPLIT = 40
        self.specaug_args = OrderedDict(
            max_len_feature = 15,
            max_len_time = 20,
            max_reps_feature = 1,
            max_reps_time = 20,
            min_learning_rate = 1e-05,
            min_reps_feature = 0,
            min_reps_time = 0,)
        self.config_args =  {
            'task': "train",
            'extra_tag_tim_setup' : "best-models",
            'use_tensorflow': True,
            'multiprocessing': True,
            'update_on_device': True,
            'stop_on_nonfinite_train_score': False,
            'log_batch_size': True,
            'debug_print_layer_output_template': True,
            'tf_log_memory_usage': True,
            'start_epoch': "auto",
            'start_batch': "auto",
            'batching': "sort_bin_shuffle:.64",
            'batch_size': 6144,
            'chunking': "400:200",
            'truncation': -1,
            'cache_size': "0",
            'window': 1,
            'num_inputs': 50,
            'num_outputs': {
                'data': [50, 2],
                'classes': [12001, 1]},
            'target': 'classes',
            'optimizer' : {"class" : "nadam"},
            'optimizer_epsilon': 1e-8,
            'gradient_noise': 0.0,
            'behavior_version' : 12, 
            'learning_rate_control': "constant",
            'learning_rate_file': "learning_rates",
            'learning_rates' : self.lr(),#
            **self.specaug_args}
        self.returnn_rasr_args_defaults = OrderedDict(
            feature_name = 'gammatone',
            alignment_name = 'align_hmm',
            num_classes = 12001,
            num_epochs = 120,
            partition_epochs = {'train': self.EP_SPLIT, 'dev': 1},
            shuffle_data = True)
        self.returnn_train_post_config = OrderedDict(
            cleanup_old_models =  {'keep': [40, 60, 80, 100, 110, 120], 'keep_best_n': 3, 'keep_last_n': 3})
        self.conformer_defaults = OrderedDict(
            num_blocks = 12)
        self.sampling_default_args = OrderedDict(
            time_reduction=1,
            unsampling_strides = 3,
            embed_l2 = 0.0,
            embed_dropout = 0.0,
            stacking_stride = 3,
            window_size = 3,
            window_left = 2,
            window_right = 0)
        self.ff_default_args = OrderedDict(
            ff_dim = 2048,
            ff_activation = "swish",
            ff_activation_dropout = 0.1,
            ff_post_dropout = 0.1,
            ff_half_ratio = 0.5)
        self.sa_default_args = OrderedDict(
            num_heads = 8,
            key_dim = 512,
            value_dim = 512,
            attention_left_only = False,
            sa_dropout = 0.1,
            linear_mapping_bias = False,
            sa_post_dropout = 0.1,
            fixed_pos = False,
            clipping = 400)
        self.conv_default_args = OrderedDict(
            kernel_size = 32,
            conv_act = "swish",
            conv_post_dropout = 0.1,
        )
        self.shared_network_args = OrderedDict(
            model_dim = 512,
            initialization = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)")
        self.auxilary_loss_args = OrderedDict(
            aux_dim = 256,
            aux_strides = 3)

def baseline_short_v0():
    NAME = "baseline_II_S_v0"
    from .baseline_04_big_short import make_experiment_03_rqmt
    args = original_args_baseline_II()
    make_experiment_03_rqmt(args, NAME)

# + shuffle data
def baseline_short_v1():
    NAME = "baseline_II_S_v1"
    from .baseline_04_big_short import make_experiment_04_seq_orders
    args = original_args_baseline_II()
    del args.returnn_rasr_args_defaults["shuffle_data"]
    params = OrderedDict(
            segment_order_shuffle = True,
            segment_order_sort_by_time_length = False)
    args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}
    make_experiment_04_seq_orders(args, NAME)

# + shorter lr
def baseline_short_v2():
    NAME = "baseline_II_S_v2"
    from .baseline_05_big_short import make_experiment_04_seq_orders
    args = original_args_baseline_II()
    del args.returnn_rasr_args_defaults["shuffle_data"]
    params = OrderedDict(
            segment_order_shuffle = True,
            segment_order_sort_by_time_length = False)
    args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}
    args.config_args["learning_rates"] = args.lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
    make_experiment_04_seq_orders(args, NAME)

# + SE block conv-mod
def baseline_short_v3():
    NAME = "baseline_II_S_v3"
    from .baseline_07_big_short import make_experiment_07_se_block
    args = original_args_baseline_II()
    del args.returnn_rasr_args_defaults["shuffle_data"]
    params = OrderedDict(
            segment_order_shuffle = True,
            segment_order_sort_by_time_length = False)
    args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}
    args.config_args["learning_rates"] = args.lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
    make_experiment_07_se_block(args, NAME, se_block_for_module = ["conv_mod"])

# + Increased Conformer blocks to 16 ( aux layer at middle 6 -> 8)
# + Reduces block dropout to 0.03
def baseline_short_v4():
    NAME = "baseline_II_S_v4"
    from .baseline_08_big_short import make_experiment_07_se_block
    args = original_args_baseline_II()
    del args.returnn_rasr_args_defaults["shuffle_data"]
    params = OrderedDict(
            segment_order_shuffle = True,
            segment_order_sort_by_time_length = False)
    args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}
    args.config_args["learning_rates"] = args.lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
    args.conformer_defaults["num_blocks"] = 16
    drop = 0.03
    args.sa_default_args["sa_dropout"] = drop
    args.sa_default_args["sa_post_dropout"] = drop
    args.conv_default_args["conv_post_dropout"] = drop
    args.ff_default_args["ff_activation_dropout"] = drop
    args.ff_default_args["ff_post_dropout"] = drop
    data = make_experiment_07_se_block(args, NAME, aux_loss_layers = [8], se_block_for_module = ["conv_mod"])

# + skip connections all even layers
# Note this is not yet included in Baseline II-S v5
def baseline_short_v4_skip_even():
    NAME = "baseline_II_S_v4+skip-all-even"
    from .baseline_08_big_short import make_experiment_10_se_l2_skip, even_space_skip
    args = original_args_baseline_II()
    del args.returnn_rasr_args_defaults["shuffle_data"]
    params = OrderedDict(
            segment_order_shuffle = True,
            segment_order_sort_by_time_length = False)
    args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}
    args.config_args["learning_rates"] = args.lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
    args.conformer_defaults["num_blocks"] = 16
    drop = 0.03
    args.sa_default_args["sa_dropout"] = drop
    args.sa_default_args["sa_post_dropout"] = drop
    args.conv_default_args["conv_post_dropout"] = drop
    args.ff_default_args["ff_activation_dropout"] = drop
    args.ff_default_args["ff_post_dropout"] = drop
    args.conformer_defaults['skip_con_after_layer'] = even_space_skip(2, args.conformer_defaults["num_blocks"])
    make_experiment_10_se_l2_skip(args, NAME, aux_loss_layers = [8], se_block_for_module = ["conv_mod"])

# + Aux lyer at 4, 12 ( and still at 8 )
# + subsampling act GELU
def baseline_short_v5():
    NAME = "baseline_II_S_v5"
    from .baseline_08_big_short import make_experiment_07_se_block
    args = original_args_baseline_II()
    del args.returnn_rasr_args_defaults["shuffle_data"]
    params = OrderedDict(
            segment_order_shuffle = True,
            segment_order_sort_by_time_length = False)
    args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}
    args.config_args["learning_rates"] = args.lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
    args.conformer_defaults["num_blocks"] = 16
    drop = 0.03
    args.sa_default_args["sa_dropout"] = drop
    args.sa_default_args["sa_post_dropout"] = drop
    args.conv_default_args["conv_post_dropout"] = drop
    args.ff_default_args["ff_activation_dropout"] = drop
    args.ff_default_args["ff_post_dropout"] = drop
    args.sampling_default_args["sampling_activation"] = "gelu"
    data = make_experiment_07_se_block(args, NAME, aux_loss_layers = [4, 8, 12], se_block_for_module = ["conv_mod"])


# All baseline start with to run with sispiphus
def run_all_experiments():
    baseline_short_v0()
    baseline_short_v1()
    baseline_short_v2()
    baseline_short_v3()
    baseline_short_v4()
    baseline_short_v4_skip_even()
    baseline_short_v5()