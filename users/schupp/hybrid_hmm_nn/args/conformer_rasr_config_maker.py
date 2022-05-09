from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline.librispeech_hybrid_tim_refactor import LibrispeechHybridSystemTim


# TODO FIXME ( these ares should also be moved to the default args file )
def get_returnn_rasr_args(
    system : LibrispeechHybridSystemTim,
    train_corpus_key = None,
    feature_name = None,
    alignment_name = None,
    num_classes = None,
    num_epochs = None,
    partition_epochs = None,
    shuffle_data = None,
):
    assert system.rasr_am_config_is_created, "please use system.create_rasr_am_config(...) first"

    import copy

    train_feature_flow = system.feature_flows[train_corpus_key][feature_name]
    train_alignment = system.alignments[train_corpus_key][alignment_name]
    
    train_crp = copy.deepcopy(system.crp[train_corpus_key + '_train'])
    dev_crp = copy.deepcopy(system.crp[train_corpus_key + '_dev'])

    if shuffle_data is None: # Is default ( was addes later will change the hash)
        pass
    elif shuffle_data: # First check is unecesarry but leavint this here so you know this changed
        for t in [train_crp, dev_crp]:
            t.corpus_config.segment_order_shuffle = True
            t.corpus_config.segment_order_sort_by_time_length = True
            t.corpus_config.segment_order_sort_by_time_length_chunk_size = -1
    
    return {
        'train_crp': train_crp,
        'dev_crp': dev_crp,
        'feature_flow' : train_feature_flow,
        'alignment' : train_alignment,
        'num_classes': num_classes,
        'num_epochs' : num_epochs,
        'partition_epochs': partition_epochs
    }


# + devtrain
# + shuffle_data=True
def get_returnn_rasr_args_02_devtrain(
    system : LibrispeechHybridSystemTim,
    train_corpus_key = None,
    feature_name = None,
    alignment_name = None,
    num_classes = None,
    num_epochs = None,
    partition_epochs = None,
    shuffle_data = True, # This is set to True per defalt here
):
    assert system.rasr_am_config_is_created, "please use system.create_rasr_am_config(...) first"

    import copy

    train_feature_flow = system.feature_flows[train_corpus_key][feature_name]
    train_alignment = system.alignments[train_corpus_key][alignment_name]
    
    train_crp = copy.deepcopy(system.crp[train_corpus_key + '_train'])
    dev_crp = copy.deepcopy(system.crp[train_corpus_key + '_dev'])
    devtrain_crp = copy.deepcopy(system.crp['devtrain2000'])

    if shuffle_data is None: # Is default ( was addes later will change the hash)
        pass
    elif shuffle_data: # First check is unecesarry but leavint this here so you know this changed
        for t in [train_crp, dev_crp, devtrain_crp]:
            t.corpus_config.segment_order_shuffle = True
            t.corpus_config.segment_order_sort_by_time_length = True
            t.corpus_config.segment_order_sort_by_time_length_chunk_size = -1
    
    return {
        'train_crp': train_crp,
        'dev_crp': dev_crp,
        'devtrain_crp': devtrain_crp,
        'feature_flow' : train_feature_flow,
        'alignment' : train_alignment,
        'num_classes': num_classes,
        'num_epochs' : num_epochs,
        'partition_epochs': partition_epochs
    }