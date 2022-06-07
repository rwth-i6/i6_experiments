from typing import OrderedDict
import math
import numpy

class original_args_big_baseline_00: # Convenient as class


  def __init__(self):
  
    def lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
          min_lr_ratio=1/50, decay_factor=0.99):

      num_lr = int(math.log(min_lr_ratio, decay_factor))
      return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                      [start] * constant_subepoch + \
                      list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                      [min_lr_ratio * start]

    self.EP_SPLIT = 40

    self.specaug_args = OrderedDict(
          max_len_feature = 15,
          max_len_time = 20,
          max_reps_feature = 1,
          max_reps_time = 20,
          min_learning_rate = 1e-05,
          min_reps_feature = 0,
          min_reps_time = 0,
    )

    self.config_args =  {
          'task': "train",
          'extra_tag_tim_setup' : "baseline-big-short",
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
            'classes': [12001, 1]
          },
          'target': 'classes',
          'optimizer' : {"class" : "nadam"},
          'optimizer_epsilon': 1e-8,
          'gradient_noise': 0.0,  # 0.1
          'behavior_version' : 12, # DIFFERENCE ( this is supposed to be the only difference here )
          'learning_rate_control': "constant",
          'learning_rate_file': "learning_rates",
          'learning_rates' : lr(),#
          **self.specaug_args
    }

    self.returnn_rasr_args_defaults = OrderedDict(
        feature_name = 'gammatone',
        alignment_name = 'align_hmm',
        num_classes = 12001,
        num_epochs = 120,
        partition_epochs = {'train': self.EP_SPLIT, 'dev': 1},
        shuffle_data = True, # Adds some etra ars to the sprint train call
    )

    self.returnn_train_post_config = OrderedDict(
      cleanup_old_models =  {'keep': [50, 80, 110, 120], 'keep_best_n': 3, 'keep_last_n': 3}
    )

    # --------------- Conformer overall args -----------------

    self.conformer_defaults = OrderedDict(
      num_blocks = 12
    )

    # -------------- Sampling args --------------

    self.sampling_default_args = OrderedDict(
      time_reduction=1,
      unsampling_strides = 3,
      embed_l2 = 0.0,
      embed_dropout = 0.0,
      stacking_stride = 3,
      window_size = 3,
      window_left = 2,
      window_right = 0,
    )

    # -------------- Feed forward -----------------

    self.ff_default_args = OrderedDict(
        ff_dim = 2048,
        ff_activation = "swish",
        ff_activation_dropout = 0.1,
        ff_post_dropout = 0.1,
        ff_half_ratio = 0.5,
    )

    # ------------ Self attention ------------------

    self.sa_default_args = OrderedDict(
        num_heads = 8,
        key_dim = 512,
        value_dim = 512,
        attention_left_only = False,
        sa_dropout = 0.1,
        linear_mapping_bias = False,
        sa_post_dropout = 0.1,
        fixed_pos = False,
        clipping = 400,
    )

    # -------------- Conv mod -----------------------

    self.conv_default_args = OrderedDict(
        kernel_size = 32,
        conv_act = "swish",
        conv_post_dropout = 0.1,
    )

    # ---------------- Shared -----------------------

    # Args are shared with layers
    self.shared_network_args = OrderedDict(
      model_dim = 512,
      initialization = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
    )

    self.auxilary_loss_args = OrderedDict(
      aux_dim = 256,
      aux_strides = 3
    )

gt_options = {
  'minfreq': 100,
  'maxfreq': 7500,
  'channels': 50,  # 68
  'warp_freqbreak': None,  # 3700
  'tempint_type': 'hanning',
  'tempint_shift': .01,
  'tempint_length': .025,
  'flush_before_gap': True,
  'do_specint': False,  # True
  'specint_type': 'hanning',
  'specint_shift': 4,
  'specint_length': 9,
  'normalize': True,
  'preemphasis': True,
  'legacy_scaling': False,
  'without_samples': False,
  'samples_options': {'audio_format': "wav",
                      'dc_detection': False},
  'normalization_options': {},
}