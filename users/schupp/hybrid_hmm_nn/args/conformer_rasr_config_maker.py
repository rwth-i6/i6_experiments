from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline.librispeech_hybrid_tim_refactor import LibrispeechHybridSystemTim


# TODO FIXME ( these ares should also be moved to the default args file )
def get_returnn_rasr_args(
    system : LibrispeechHybridSystemTim,
    train_corpus_key = None,
    feature_name = None,
    alignment_name = None,
    num_classes = None,
    num_epochs = None,
    partition_epochs = None
):
    assert system.rasr_am_config_is_created, "please use system.create_rasr_am_config(...) first"

    train_feature_flow = system.feature_flows[train_corpus_key][feature_name]
    train_alignment = system.alignments[train_corpus_key][alignment_name]
    
    return {
        'train_crp': system.crp[train_corpus_key + '_train'],
        'dev_crp': system.crp[train_corpus_key + '_dev'],
        'feature_flow' : train_feature_flow,
        'alignment' : train_alignment,
        'num_classes': num_classes,
        'num_epochs' : num_epochs,
        'partition_epochs': partition_epochs
    }