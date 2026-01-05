from sisyphus import tk
import numpy as np

from i6_core.text.processing import ConcatenateJob

from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data

from ..data.word import build_training_data
from ..data.common import get_librispeech_train_cv_data
from ..config import get_training_config
from ..pipeline import train
from ...default_tools import RETURNN_EXE, RETURNN_ROOT

def test_train_lm(lex):
    prefix = "experiments/domain_testing_2024/tf/lm"

    ls_base_lexicon = get_bliss_lexicon()
    ls_train_text, cv_text = get_librispeech_train_cv_data()
    
    ufal_medical_version_1_text = tk.Path(
        "/work/asr4/rossenbach/domain_data/UFAL_medical_shuffled/clean_version_1/all_en_medical_uniq_sorted_final.txt.gz",
        hash_overwrite="UFAL_medical_shuffled/clean_version_1/all_en_medical_uniq_sorted_final.txt.gz"
    )
    # this is for cv
    wmt22_medline_v1 = tk.Path(
        "/work/asr4/rossenbach/domain_data/wmt_medline_test_data/wmt22_medline_v1.txt",
        hash_overwrite="wmt22_medline_v1.txt"
    )

    MTG_trial4_train = tk.Path(
        "/work/asr4/rossenbach/domain_data/MTG/MTG_trial4_train.txt",
        hash_overwrite="MTG/MTG_trial4_train.txt"
    )
    # this is for cv
    MTG_trial4_dev = tk.Path(
        "/work/asr4/rossenbach/domain_data/MTG/MTG_trial4_dev.txt",
        hash_overwrite="MTG/MTG_trial4_dev.txt"
    )

    training_data_bin_shuffle, ls_index_vocab_bin_shuffle = build_training_data(
        prefix=prefix,
        lexicon_bliss=ls_base_lexicon,
        train_text=ls_train_text,
        cv_text=cv_text,
        partition_epoch=100,
        seq_ordering="sort_bin_shuffle:.32",
    )

    training_data, ls_index_vocab = build_training_data(
        prefix=prefix,
        lexicon_bliss=ls_base_lexicon,
        train_text=ls_train_text,
        cv_text=cv_text,
        partition_epoch=100,
        seq_ordering="random",
    )

    ufal_training_data_bin_shuffle, ufal_index_vocab_bin_shuffle = build_training_data(
        prefix=prefix,
        train_text=ufal_medical_version_1_text,
        lexicon_bliss=lex["ufal_v1_3more_only_nols"],
        cv_text=wmt22_medline_v1,
        partition_epoch=25,
        seq_ordering="sort_bin_shuffle:.32",
    )

    ufal_training_data_random, ufal_index_vocab_bin_random = build_training_data(
        prefix=prefix,
        train_text=ufal_medical_version_1_text,
        lexicon_bliss=lex["ufal_v1_3more_only_nols"],
        cv_text=wmt22_medline_v1,
        partition_epoch=25,
        seq_ordering="random",
    )

    MTG_training_data_random, MTG_index_vocab_random = build_training_data(
        prefix=prefix,
        train_text=MTG_trial4_train,
        lexicon_bliss=lex["MTG_trial4_nols"],
        cv_text=MTG_trial4_dev,
        partition_epoch=1,
        seq_ordering="random",
    )

    from ..networks import LstmLmSettings,get_lstm_lm_network

    # Different network settings

    settings = LstmLmSettings(
        embedding_dim=128,
        lstm_sizes=[1024, 1024],
        bottleneck=None,
        lstm_dropout=0.05,
        output_dropout=0.2
    )
    lstm_net = get_lstm_lm_network(settings)

    settings_lstmdrop_01 = LstmLmSettings(
        embedding_dim=128,
        lstm_sizes=[1024, 1024],
        bottleneck=None,
        lstm_dropout=0.1,
        output_dropout=0.2
    )
    lstm_net_lstmdrop_01 = get_lstm_lm_network(settings_lstmdrop_01)

    settings_lstmdrop_05 = LstmLmSettings(
        embedding_dim=128,
        lstm_sizes=[1024, 1024],
        bottleneck=None,
        lstm_dropout=0.5,
        output_dropout=0.5
    )
    lstm_net_lstmdrop_05 = get_lstm_lm_network(settings_lstmdrop_05)

    settings_2x2k_1kbot = LstmLmSettings(
        embedding_dim=128,
        lstm_sizes=[2048, 2048],
        bottleneck=1024,
        lstm_dropout=0.1,
        output_dropout=0.2
    )
    lstm_net_2x2k_1kbot = get_lstm_lm_network(settings_2x2k_1kbot)

    name = "word/ls960_emb128_2x1k_lstmdrop005_outdrop_02_wronglr"
    # First test, like mini 2.4 config
    config = get_training_config(training_data_bin_shuffle, lstm_net)
    training_config = {
        "batch_size": 1200,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": [1.0] * 49 + list(np.linspace(1e0, 1e-7, 201))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name,  config, num_epochs=300)
    train_job.rqmt["gpu_mem"] = 48

    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)


    ###### Corrected LR

    name = "word/ls960_emb128_2x1k_lstmdrop005_outdrop_02_bin_shuffle"
    config = get_training_config(training_data_bin_shuffle, lstm_net)
    training_config = {
        "batch_size": 1200,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": [1.0] * 49 + list(np.linspace(1e0, 1e-7, 251))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name,  config, num_epochs=300)
    train_job.rqmt["gpu_mem"] = 48

    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)

    # Random instead of bin-shuffle

    # name = "word/ls960_emb128_2x1k_lstmdrop005_outdrop_02_random"
    # config = get_training_config(training_data, lstm_net)
    # training_config = {
    #     "batch_size": 1200,
    #     "max_seqs": 32,
    #     "gradient_clip_global_norm": 2.0,
    #     "learning_rates": [1.0] * 49 + list(np.linspace(1e0, 1e-7, 251))
    # }
    # config.config.update(training_config)
    # train_job = train(prefix + "/" + name,  config, num_epochs=300)
    # train_job.rqmt["gpu_mem"] = 48

    # tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)

    # Larger model

    name = "word/ls960_emb128_2x2k_1kbot_lstmdrop01_outdrop_02_bin_shuffle"
    config = get_training_config(training_data_bin_shuffle, lstm_net_2x2k_1kbot)
    training_config = {
        "batch_size": 1200,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": [1.0] * 49 + list(np.linspace(1e0, 1e-7, 251))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name,  config, num_epochs=300)
    train_job.rqmt["gpu_mem"] = 48

    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)

    ##### short training

    lstm_export_net = get_lstm_lm_network(settings, for_export=True)
    name = "word/ufalv1_3more_emb128_2x1k_lstmdrop005_outdrop_02_short"
    config = get_training_config(ufal_training_data_bin_shuffle, lstm_net)
    training_config = {
        "batch_size": 1200,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": list(np.linspace(1e0, 1e-3, 50))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name,  config, num_epochs=50)
    train_job.rqmt["gpu_mem"] = 48

    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)
    from i6_core.returnn.compile import CompileTFGraphJob
    export_config = get_training_config(ufal_training_data_bin_shuffle, lstm_export_net)
    compile_job = CompileTFGraphJob(
        returnn_config=export_config,
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
    )
    tk.register_output(prefix + "/" + name + "/compile_config.py", compile_job.out_returnn_config)
    tk.register_output(prefix + "/" + name + "/compile_graph.meta", compile_job.out_returnn_config)
    tk.register_output(prefix + "/" + name + "/index_vocab", ufal_index_vocab_bin_shuffle.vocab)

    # with larger dropout

    name = "word/ufalv1_3more_emb128_2x1k_lstmdrop01_outdrop_02_bin_shuffle_short"
    config = get_training_config(ufal_training_data_bin_shuffle, lstm_net_lstmdrop_01)
    training_config = {
        "batch_size": 1200,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": list(np.linspace(1e0, 1e-3, 50))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name,  config, num_epochs=50)
    train_job.rqmt["gpu_mem"] = 48

    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)

    name = "word/ufalv1_3more_emb128_2x1k_lstmdrop01_outdrop_02_random_short"
    config = get_training_config(ufal_training_data_random, lstm_net_lstmdrop_01)
    training_config = {
        "batch_size": 1200,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": list(np.linspace(1e0, 1e-3, 50))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name,  config, num_epochs=50)
    train_job.rqmt["gpu_mem"] = 48

    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)

    name = "word/ufalv1_3more_emb128_2x1k_lstmdrop01_outdrop_02_long"
    config = get_training_config(ufal_training_data_bin_shuffle, lstm_net_lstmdrop_01)
    training_config = {
        "batch_size": 1200,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": [1.0] * 49 + list(np.linspace(1e0, 1e-7, 251))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name,  config, num_epochs=300)
    train_job.rqmt["gpu_mem"] = 48

    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)

    name = "word/MTG_trial4_emb128_2x1k_lstmdrop01_outdrop_02_long"
    config = get_training_config(MTG_training_data_random, lstm_net_lstmdrop_01)
    training_config = {
        "batch_size": 1200,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": [1.0] * 49 + list(np.linspace(1e0, 1e-7, 251))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name, config, num_epochs=300)
    train_job.rqmt["gpu_mem"] = 48

    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)

    name = "word/MTG_trial4_emb128_2x1k_lstmdrop05_outdrop_05_long"
    config = get_training_config(MTG_training_data_random, lstm_net_lstmdrop_05)
    training_config = {
        "batch_size": 1200,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": [1.0] * 49 + list(np.linspace(1e0, 1e-7, 251))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name, config, num_epochs=300)
    train_job.rqmt["gpu_mem"] = 48

    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)