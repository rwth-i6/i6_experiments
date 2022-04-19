# Tim Schupp
# This is a first Conformer Implementation using returnn_common

import recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline.librispeech_hybrid_system as librispeech_hybrid_system

system = librispeech_hybrid_system.LibrispeechHybridSystem()
gs.ALIAS_AND_OUTPUT_SUBDIR = "conformer/baseline_rtc/"


return_config_train_args = dict(
    num_epochs = 200
    num_inputs = 50
    num_outputs = {'classes': [12001, 1], 'data': [50, 2]}
    optimizer = {'class': 'nadam'}
    optimizer_epsilon = 1e-08
    save_interval = 1
    start_batch = 'auto' # not needed ?
    start_epoch = 'auto' # not needed ?
    stop_on_nonfinite_train_score = False
    target = 'classes'
    task = 'train'
    tf_log_memory_usage = True
    truncation = -1
    update_on_device = True
    use_tensorflow = True
    window = 1
    batch_size = 6144
    batching = 'sort_bin_shuffle:.64'
    behavior_version = 12
    cache_size = '0'
    chunking = '200:100'
    cleanup_old_models = {'keep': [10, 20, 40, 80, 90, 100, 110, 116, 120, 140, 160, 180, 190, 200], 'keep_best_n': 3, 'keep_last_n': 3}
    debug_print_layer_output_template = True
    device = 'gpu'
    gradient_noise = 0.0
    learning_rate = 1e-05 # not needed ? or should be added later?
    learning_rate_control = 'constant'
    log_batch_size = True
    log_verbosity = 5
    min_learning_rate = 1e-05 # not needed ? or should be added later?
)


