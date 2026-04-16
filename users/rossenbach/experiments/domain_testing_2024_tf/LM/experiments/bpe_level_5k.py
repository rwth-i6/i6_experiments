from sisyphus import tk
import numpy as np

from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

from ..data.bpe import build_training_data
from ..data.common import get_librispeech_train_cv_data
from ..config import get_training_config
from ..pipeline import train

from ..networks import LstmLmSettings, get_lstm_lm_network, get_lstm_lm_network_for_shallow_fusion
from ...storage import add_lm, LmModel


def test_train_bpe_5k_lm():
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

    training_data, datastream, bpe_settings = build_training_data(
        prefix=prefix,
        train_text=ls_train_text,
        cv_text=cv_text,
        partition_epoch=100,
        bpe_size=5000,
        seq_ordering="random",
    )
    
    ufal_training_data, datastream, bpe_settings = build_training_data(
        prefix=prefix,
        train_text=ufal_medical_version_1_text,
        cv_text=wmt22_medline_v1,
        partition_epoch=25,
        bpe_size=5000,
        seq_ordering="random",
        allow_unknown=True,
    )
    
    MTG_training_data_random, datastream, bpe_settings = build_training_data(
        prefix=prefix,
        train_text=MTG_trial4_train,
        cv_text=MTG_trial4_dev,
        partition_epoch=1,
        bpe_size=5000,
        seq_ordering="random",
        allow_unknown=True,
    )
    
    from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob
    bpe_dev = ApplyBPEToTextJob(
        MTG_trial4_dev,
        bpe_settings.bpe_codes,
        bpe_settings.bpe_count_vocab,
        subword_nmt_repo=get_returnn_subword_nmt(),
        mini_task=True,
    )
    tk.register_output(prefix + "/PPL/MTG_trial4_dev.word.txt", MTG_trial4_dev)
    tk.register_output(prefix + "/PPL/MTG_trial4_dev.5kbpe.txt", bpe_dev.out_bpe_text)
    
    bpe_dev = ApplyBPEToTextJob(
        wmt22_medline_v1,
        bpe_settings.bpe_codes,
        bpe_settings.bpe_count_vocab,
        subword_nmt_repo=get_returnn_subword_nmt(),
        mini_task=True,
    )
    tk.register_output(prefix + "/PPL/wmt22_medline_v1.5kbpe.txt", bpe_dev.out_bpe_text)
    
    bpe_dev = ApplyBPEToTextJob(
        cv_text,
        bpe_settings.bpe_codes,
        bpe_settings.bpe_count_vocab,
        subword_nmt_repo=get_returnn_subword_nmt(),
        mini_task=True,
    )
    tk.register_output(prefix + "/PPL/ls_cv.5kbpe.txt", bpe_dev.out_bpe_text)

    settings = LstmLmSettings(
        embedding_dim=128,
        lstm_sizes=[1024, 1024],
        bottleneck=None,
        lstm_dropout=0.05,
        output_dropout=0.2
    )
    lstm_net = get_lstm_lm_network(settings)
    lstm_fusion_net = get_lstm_lm_network_for_shallow_fusion(settings, out_dim=datastream.vocab_size)

    ##### Full LS training

    name = "bpe5k/ls960_emb128_2x1k_lstmdrop005_outdrop_02_long"
    config = get_training_config(training_data, lstm_net)
    training_config = {
        "batch_size": 1700,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": [1.0] * 49 + list(np.linspace(1e0, 1e-7, 251))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name,  config, num_epochs=300)
    train_job.rqmt["gpu_mem"] = 48
    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)

    add_lm("bpe5k_ls_2x1k", lm_model=LmModel(network=lstm_fusion_net, model=train_job.out_models[300].model))

    training_config = {
        "batch_size": 1700,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": [1.0] * 49 + list(np.linspace(1e0, 1e-7, 251))
    }
    name = "bpe5k/ufalv1_emb128_2x1k_lstmdrop005_outdrop_02_long"
    ufal_config = get_training_config(ufal_training_data, lstm_net)
    ufal_config.config.update(training_config)
    train_job = train(prefix + "/" + name,  ufal_config, num_epochs=300)
    train_job.rqmt["gpu_mem"] = 48
    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)

    add_lm("bpe5k_medline_2x1k", lm_model=LmModel(network=lstm_fusion_net, model=train_job.out_models[300].model))

    # MTG
    name = "bpe5k/MTGv4_emb128_2x1k_lstmdrop05_outdrop_05"
    settings = LstmLmSettings(
        embedding_dim=128,
        lstm_sizes=[1024, 1024],
        bottleneck=None,
        lstm_dropout=0.5,
        output_dropout=0.5
    )
    lstm_net = get_lstm_lm_network(settings)
    config = get_training_config(MTG_training_data_random, lstm_net)
    training_config = {
        "batch_size": 1700,
        "max_seqs": 32,
        "gradient_clip_global_norm": 2.0,
        "learning_rates": [1.0] * 49 + list(np.linspace(1e0, 1e-7, 51))
    }
    config.config.update(training_config)
    train_job = train(prefix + "/" + name,  config, num_epochs=100)
    train_job.rqmt["gpu_mem"] = 48
    tk.register_output(prefix + "/" + name + "/learning_rates", train_job.out_learning_rates)
