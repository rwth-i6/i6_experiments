config_baseline_00 = {
      'task': "train",
      'use_tensorflow': True,
      'multiprocessing': True,
      'update_on_device': True,
      'stop_on_nonfinite_train_score': False,
      'log_batch_size': True,
      'debug_print_layer_output_template': True,
      'tf_log_memory_usage': True,
      'start_epoch': "auto",
      'start_batch': "auto",
      'batching': "sort_bin_shuffle:.64",  # f"laplace:{num_seqs//1000}"
      'batch_size': 6144,
      'chunking': "200:100",
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
      'learning_rate_control': "constant",
      'learning_rate_file': "learning_rates",
}