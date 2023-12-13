from sisyphus import tk
import numpy as np

import copy
from dataclasses import asdict
from .data import build_training_datasets, TrainingDatasetSettings, build_test_dataset
from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from .pipeline import training, search, get_best_checkpoint

from .config import get_training_config, get_search_config

def conformer_baseline():
    BPE_SIZE = 2000
    prefix_name = "experiments/librispeech/librispeech_100_bpe_ctc/standalone_pt_2023"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        "train-clean-100",
        bpe_size=BPE_SIZE,
        preemphasis=None,
        settings=train_settings
    )

    # build testing datasets
    test_dataset_tuples = {}
    #for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for testset in ["dev-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
            bpe_size=BPE_SIZE,
            preemphasis=None
        )


        # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function


    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    from i6_experiments.users.rossenbach.lexicon.bpe_lexicon import CreateBPELexiconJob
    from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
    ls_lexicon = get_bliss_lexicon(use_stress_marker=False, add_unknown_phoneme_and_mapping=True)
    from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
    
    bpe_lexicon = CreateBPELexiconJob(
        ls_lexicon,
        bpe_codes=train_data.datastreams["bpe_labels"].codes,
        bpe_vocab=train_data.datastreams["bpe_labels"].vocab,
        subword_nmt_repo=get_returnn_subword_nmt(),
    ).out_lexicon
    bpe_lexicon = BlissLexiconToWordLexicon(bpe_lexicon).out_lexicon

    config = {

    }

    def run_exp(ft_name, datasets, train_args, search_args=None, best=False, last=True, speed_perturbation=False):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, speed_perturbation=speed_perturbation, **train_args)
        if speed_perturbation:
            from i6_core.returnn.config import CodeWrapper
            returnn_config.config["train"]["datasets"]["zip_dataset"]["audio"]["pre_process"] = CodeWrapper("speed_perturbation")
        returnn_search_config = get_search_config(**train_args, search_args=search_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=300)

        #averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        best_checkpoint = get_best_checkpoint(train_job)

        if last:
            search(ft_name + "/default_300", returnn_search_config, train_job.out_checkpoints[300], test_dataset_tuples, RETURNN_EXE, MINI_RETURNN_ROOT)
        if best:
            search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, MINI_RETURNN_ROOT)
        #search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

    # Better training with OCLR / Batch size / Data len fix
    train_args = {
        "net_args": {},
        "network_module": "basic_conformer_static",
        "debug": True,
        "config": {
            "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            #############
            "batch_size": 200 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args, last=False, best=True)


    ## ADAMW
    # for weight_decay in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    for weight_decay in [1e-1, 1e-2, 1e-3]:
        train_args = {
            "net_args": {},
            "network_module": "basic_conformer_static",
            "debug": True,
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-8, "weight_decay": weight_decay},
                "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
                #############
                "batch_size": 200 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
            },
        }
        for lm_weight in [1.2, 1.4, 1.6]:
            search_args = {
                "lexicon": bpe_lexicon,
                "beam_size": 20,
                "arpa_lm": get_arpa_lm_dict()["4gram"],
                "lm_weight": lm_weight,
            }
            run_exp(prefix_name + "/test_adamw%1.E_bs20_lm%.1f" % (weight_decay, lm_weight), datasets=train_data, train_args=train_args, search_args=search_args)

        if weight_decay == 1e-2:
            for blank_penalty in [0.25, 0.5, 0.75, 0.9]:
                search_args = {
                    "lexicon": bpe_lexicon,
                    "beam_size": 20,
                    "arpa_lm": get_arpa_lm_dict()["4gram"],
                    "lm_weight": lm_weight,
                    "blank_log_penalty": -np.log(blank_penalty)
                }
                run_exp(prefix_name + "/test_adamw%1.E_bs20_lm%.1f_blankscale%.2f" % (weight_decay, lm_weight, blank_penalty), datasets=train_data,
                        train_args=train_args, search_args=search_args)



    # Try output dropout
    train_args = {
        "net_args": {},
        "network_module": "basic_conformer_static_outdrop",
        "debug": True,
        "config": {
            "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            #############
            "batch_size": 200 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_outdrop_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)


    train_args = {
        "net_args": {},
        "network_module": "basic_conformer_static",
        "debug": True,
        "config": {
            "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            #############
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_large_batch_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)
        

    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw_02 = {
        "net_args": {},
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            #############
            "batch_size": 200 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
    }
       
        
    # Try output dropout and speed perturbation with AdamW
    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_outdrop",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_outdrop_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)
        run_exp(prefix_name + "/test_adamw1E-02_outdrop_speedpert_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args,
            search_args=search_args, speed_perturbation=True)

    # second run for verification
    train_args = copy.deepcopy({
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_outdrop",
        "debug": True,
    })
    train_args["config"]["batch_size"] = (200*16000) + 1
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_outdrop_bs20_lm%.1f_r2" % lm_weight, datasets=train_data,
                train_args=train_args, search_args=search_args)


    # Try returnn-style specaugment
    # Try output dropout and speed perturbation with AdamW
    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_outdrop_newspec",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)


    # second run for verification
    train_args = copy.deepcopy({
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_outdrop_newspec",
        "debug": True,
    })
    train_args["config"]["batch_size"] = (200*16000) + 1
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_outdropnewspec_bs20_lm%.1f_r2" % lm_weight, datasets=train_data,
                train_args=train_args, search_args=search_args)

    # Try returnn-style specaugment
    # Try output dropout and with AdamW
    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_convfirst_outdrop_newspec",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_convfirst_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)

    # Conv first with 31
    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_convfirst31_outdrop_newspec",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_convfirst31_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data,
                train_args=train_args, search_args=search_args)
        
    # normal conv with 31
    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_conv31_outdrop_newspec",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_conv31_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data,
                train_args=train_args, search_args=search_args)


    # No positional encoding
    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_nope_static_outdrop_newspec",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_nope_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)

    # With Silu activation in frontend
    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_frntact_static_outdrop_newspec",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_frntact_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)

    # As above but with Mohammads frontend (diverged)
    # train_args = {
    #     **train_args_adamw_02,
    #     "network_module": "basic_conformer_mhmdfrontend_static_outdrop_newspec",
    #     "debug": True,
    # }
    # for lm_weight in [1.2, 1.4, 1.6]:
    #     search_args = {
    #         "lexicon": bpe_lexicon,
    #         "beam_size": 20,
    #         "arpa_lm": get_arpa_lm_dict()["4gram"],
    #         "lm_weight": lm_weight,
    #     }
    #     run_exp(prefix_name + "/test_adamw1E-02_mhmdfrontend_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)


    # # As above but with additional batch norm (diverged)
    # train_args = {
    #     **train_args_adamw_02,
    #     "network_module": "basic_conformer_mhmdfrontend_bn_static_outdrop_newspec",
    #     "debug": True,
    # }
    # for lm_weight in [1.2, 1.4, 1.6]:
    #     search_args = {
    #         "lexicon": bpe_lexicon,
    #         "beam_size": 20,
    #         "arpa_lm": get_arpa_lm_dict()["4gram"],
    #         "lm_weight": lm_weight,
    #     }
    #     run_exp(prefix_name + "/test_adamw1E-02_mhmdfrontend_bn_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)

    # # No batch norm, but absolute positional encoding (diverged)
    # train_args = {
    #     **train_args_adamw_02,
    #     "network_module": "basic_conformer_mhmdfrontend_pe_static_outdrop_newspec",
    #     "debug": True,
    # }
    # for lm_weight in [1.2, 1.4, 1.6]:
    #     search_args = {
    #         "lexicon": bpe_lexicon,
    #         "beam_size": 20,
    #         "arpa_lm": get_arpa_lm_dict()["4gram"],
    #         "lm_weight": lm_weight,
    #     }
    #     run_exp(prefix_name + "/test_adamw1E-02_mhmdfrontend_pe_outdropnewspec_bs20_lm%.1f" % lm_weight,
    #             datasets=train_data, train_args=train_args, search_args=search_args)
        
    # Atanas-frontend (diverged)
    # train_args = {
    #     **train_args_adamw_02,
    #     "network_module": "basic_conformer_atanasfrontend_static_outdrop_newspec",
    #     "debug": True,
    # }
    # for lm_weight in [1.2, 1.4, 1.6]:
    #     search_args = {
    #         "lexicon": bpe_lexicon,
    #         "beam_size": 20,
    #         "arpa_lm": get_arpa_lm_dict()["4gram"],
    #         "lm_weight": lm_weight,
    #     }
    #     run_exp(prefix_name + "/test_adamw1E-02_atanasfrontend_pe_outdropnewspec_bs20_lm%.1f" % lm_weight,
    #             datasets=train_data, train_args=train_args, search_args=search_args)
        
    # ESPNet-frontend
    # train_args = {
    #     **train_args_adamw_02,
    #     "network_module": "basic_conformer_espnetsub_static_outdrop_newspec",
    #     "debug": True,
    # }
    # for lm_weight in [1.2, 1.4, 1.6]:
    #     search_args = {
    #         "lexicon": bpe_lexicon,
    #         "beam_size": 20,
    #         "arpa_lm": get_arpa_lm_dict()["4gram"],
    #         "lm_weight": lm_weight,
    #     }
    #     run_exp(prefix_name + "/test_adamw1E-02_espnetsub_outdropnewspec_bs20_lm%.1f" % lm_weight,
    #             datasets=train_data, train_args=train_args, search_args=search_args)
        
    
    # Jingjing mhsa with flat warmup
    train_args = copy.deepcopy({
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_jingjingrelmhsa_outdrop_newspec",
        "debug": True,
    })
    train_args["config"]["learning_rates"] = [1e-6] * 10 + list(np.linspace(1e-6, 1e-3, 140)) + list(np.linspace(1e-3, 1e-6, 150))
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_jingjingrelmhsa_warmconst_outdropnewspec_bs20_lm%.1f" % lm_weight,
                datasets=train_data, train_args=train_args, search_args=search_args)
    
    # Jingjing mhsa with flat warmup
    train_args = copy.deepcopy({
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_jingjingrelmhsa_outdrop_newspec",
        "debug": True,
    })
    train_args["config"]["learning_rates"] = [1e-5] * 10 + list(np.linspace(1e-5, 1e-3, 140)) + list(np.linspace(1e-3, 1e-6, 150))
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_jingjingrelmhsa_warmconst_1e5_outdropnewspec_bs20_lm%.1f" % lm_weight,
                datasets=train_data, train_args=train_args, search_args=search_args)
    
    # Jingjing mhsa with normal warmup and transparent attention
    # train_args = copy.deepcopy({
    #     **train_args_adamw_02,
    #     "network_module": "basic_conformer_static_jingjingrelmhsa_transparent_outdrop_newspec",
    #     "debug": True,
    # })
    # for lm_weight in [1.2, 1.4, 1.6]:
    #     search_args = {
    #         "lexicon": bpe_lexicon,
    #         "beam_size": 20,
    #         "arpa_lm": get_arpa_lm_dict()["4gram"],
    #         "lm_weight": lm_weight,
    #     }
    #     run_exp(prefix_name + "/test_adamw1E-02_jingjingrelmhsa_transparent_outdropnewspec_bs20_lm%.1f" % lm_weight,
    #             datasets=train_data, train_args=train_args, search_args=search_args)
      
        
    # Stacking-frontend
    # train_args = {
    #     **train_args_adamw_02,
    #     "network_module": "basic_conformer_static_stacking_outdrop_newspec",
    #     "debug": True,
    # }
    # for lm_weight in [1.2, 1.4, 1.6]:
    #     search_args = {
    #         "lexicon": bpe_lexicon,
    #         "beam_size": 20,
    #         "arpa_lm": get_arpa_lm_dict()["4gram"],
    #         "lm_weight": lm_weight,
    #     }
    #     run_exp(prefix_name + "/test_adamw1E-02_stacking_outdropnewspec_bs20_lm%.1f" % lm_weight,
    #             datasets=train_data, train_args=train_args, search_args=search_args)

    # Try returnn-style specaugment
    # Try output dropout and with AdamW
    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_convfirst_transparent_outdrop_newspec",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_convfirst_transparent_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)
 
    # Try returnn-style specaugment
    # Try output dropout and with AdamW
    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_convfirst31_transparent_outdrop_newspec",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_convfirst31_transparent_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args)

    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_convfirst31_transparent_fixed_outdrop_newspec",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_convfirst31_transparentv2_outdropnewspec_bs20_lm%.1f" % lm_weight,
                datasets=train_data, train_args=train_args, search_args=search_args)


    # normal conv with 31 but with transparent attention
    train_args = {
        **train_args_adamw_02,
        "network_module": "basic_conformer_static_conv31_transparent_outdrop_newspec",
        "debug": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/test_adamw1E-02_conv31_transparent_outdropnewspec_bs20_lm%.1f" % lm_weight, datasets=train_data,
                train_args=train_args, search_args=search_args)


def conformer_tuning():
    BPE_SIZE = 2000
    prefix_name = "experiments/librispeech/librispeech_100_bpe_ctc/standalone_pt_2023"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        "train-clean-100",
        bpe_size=BPE_SIZE,
        preemphasis=None,
        settings=train_settings,
    )

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for testset in ["dev-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
            bpe_size=BPE_SIZE,
            preemphasis=None
        )

        # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function

    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    from i6_experiments.users.rossenbach.lexicon.bpe_lexicon import CreateBPELexiconJob
    from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
    ls_lexicon = get_bliss_lexicon(use_stress_marker=False, add_unknown_phoneme_and_mapping=True)
    from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

    bpe_lexicon = CreateBPELexiconJob(
        ls_lexicon,
        bpe_codes=train_data.datastreams["bpe_labels"].codes,
        bpe_vocab=train_data.datastreams["bpe_labels"].vocab,
        subword_nmt_repo=get_returnn_subword_nmt(),
    ).out_lexicon
    bpe_lexicon = BlissLexiconToWordLexicon(bpe_lexicon).out_lexicon

    config = {

    }

    def run_exp(ft_name, datasets, train_args, search_args=None, best=False, last=True, speed_perturbation=False):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, speed_perturbation=speed_perturbation,
                                             **train_args)
        if speed_perturbation:
            from i6_core.returnn.config import CodeWrapper
            returnn_config.config["train"]["datasets"]["zip_dataset"]["audio"]["pre_process"] = CodeWrapper(
                "speed_perturbation")
        returnn_search_config = get_search_config(**train_args, search_args=search_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=300)

        # averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        best_checkpoint = get_best_checkpoint(train_job)

        if last:
            search(ft_name + "/default_300", returnn_search_config, train_job.out_checkpoints[300], test_dataset_tuples,
                   RETURNN_EXE, MINI_RETURNN_ROOT)
        if best:
            search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE,
                   MINI_RETURNN_ROOT)
        # search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

    from .pytorch_networks.ctc_conformer_0923.conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit_cfg import SpecaugConfig, TwoLayer1DFrontendConfig, ModelConfig

    specaug_config = SpecaugConfig(
        repeat_per_n_frames = 50,
        max_dim_time = 20,
        max_dim_feat = 8,
        num_repeat_feat = 5,
    )
    frontend_config = TwoLayer1DFrontendConfig(
        in_features=80,
        conv1_channels=512,
        conv2_channels=512,
        conv1_kernel_size=5,
        conv2_kernel_size=5,
        conv1_stride=3,
        conv2_stride=2,
        dropout=0.1,
    )
    model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        conformer_size=512,
        num_layers=8,
        num_heads=4,
        ff_dim=512,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )
    
    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw_02 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            #############
            "batch_size": 200 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
    }
    
    # normal conv with 31 but with transparent attention
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
        "with_devtrain": True,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/tune/conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/baseline_bs20_lm%.1f" % lm_weight, datasets=train_data,
                train_args=train_args, search_args=search_args)
        
        
    # same, but with num_workers=2
    train_args_mp = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
        "with_devtrain": True,
        "num_workers": 2,
    }
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/tune/conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/baseline_mp2_bs20_lm%.1f" % lm_weight, datasets=train_data,
                train_args=train_args_mp, search_args=search_args)
        
    model_config_12x384 = copy.deepcopy((model_config))
    model_config_12x384.ff_dim = 1536
    model_config_12x384.num_layers = 12
    model_config_12x384.conformer_size = 384
    model_config_12x384.frontend_config.conv1_channels = 256
    model_config_12x384.frontend_config.conv2_channels = 384
    train_args = copy.deepcopy(train_args)
    train_args["net_args"]["model_config_dict"] = asdict(model_config_12x384)
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/tune/conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/12x384_bs20_lm%.1f" % lm_weight, datasets=train_data,
                train_args=train_args, search_args=search_args)

    search_args = {
        "lexicon": bpe_lexicon,
        "beam_size": 1,
        "arpa_lm": None,
        "lm_weight": 0.0,
    }
    run_exp(
        prefix_name + "/tune/conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/12x384_greedy",
        datasets=train_data,
        train_args=train_args, search_args=search_args)
        
        
    model_config_12x384_morespecaug = copy.deepcopy((model_config_12x384))
    model_config_12x384_morespecaug.specaug_config.repeat_per_n_frames = 25
    model_config_12x384_morespecaug.specaug_config.max_dim_feat = 16
    train_args = copy.deepcopy(train_args)
    train_args["net_args"]["model_config_dict"] = asdict(model_config_12x384_morespecaug)
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/tune/conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/12x384_morespec_bs20_lm%.1f" % lm_weight, datasets=train_data,
                train_args=train_args, search_args=search_args)
        
    model_config_12x384_moredropout = copy.deepcopy((model_config_12x384))
    model_config_12x384_moredropout.frontend_config.dropout = 0.2
    model_config_12x384_moredropout.final_dropout = 0.3
    model_config_12x384_moredropout.conv_dropout = 0.3
    model_config_12x384_moredropout.ff_dropout = 0.3
    model_config_12x384_moredropout.mhsa_dropout = 0.3
    train_args = copy.deepcopy(train_args)
    train_args["net_args"]["model_config_dict"] = asdict(model_config_12x384_moredropout)
    for lm_weight in [1.2, 1.4, 1.6]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(prefix_name + "/tune/conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/12x384_moredropout_bs20_lm%.1f" % lm_weight, datasets=train_data,
                train_args=train_args, search_args=search_args)


def conformer_small_bpe():
    BPE_SIZE = 300
    prefix_name = "experiments/librispeech/librispeech_100_bpe_ctc/standalone_pt_2023"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        "train-clean-100",
        bpe_size=BPE_SIZE,
        preemphasis=None,
        settings=train_settings,
        use_v2_subnmt=True,
    )

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for testset in ["dev-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
            bpe_size=BPE_SIZE,
            preemphasis=None,
            use_v2_subnmt=True,
        )

        # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function

    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    from i6_experiments.users.rossenbach.lexicon.bpe_lexicon import CreateBPELexiconJob
    from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
    ls_lexicon = get_bliss_lexicon(use_stress_marker=False, add_unknown_phoneme_and_mapping=True)
    from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

    bpe_lexicon = CreateBPELexiconJob(
        ls_lexicon,
        bpe_codes=train_data.datastreams["bpe_labels"].codes,
        bpe_vocab=train_data.datastreams["bpe_labels"].vocab,
        subword_nmt_repo=get_returnn_subword_nmt(),
    ).out_lexicon
    bpe_lexicon = BlissLexiconToWordLexicon(bpe_lexicon).out_lexicon

    config = {

    }

    def run_exp(ft_name, datasets, train_args, search_args=None, best=False, last=True, speed_perturbation=False):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, speed_perturbation=speed_perturbation,
                                             **train_args)
        if speed_perturbation:
            from i6_core.returnn.config import CodeWrapper
            returnn_config.config["train"]["datasets"]["zip_dataset"]["audio"]["pre_process"] = CodeWrapper(
                "speed_perturbation")
        returnn_search_config = get_search_config(**train_args, search_args=search_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=300)

        # averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        best_checkpoint = get_best_checkpoint(train_job)

        if last:
            search(ft_name + "/default_300", returnn_search_config, train_job.out_checkpoints[300], test_dataset_tuples,
                   RETURNN_EXE, MINI_RETURNN_ROOT)
        if best:
            search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE,
                   MINI_RETURNN_ROOT)
        # search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

    from .pytorch_networks.ctc_conformer_0923.conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit_cfg import \
        SpecaugConfig, TwoLayer1DFrontendConfig, ModelConfig

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=50,
        max_dim_time=20,
        max_dim_feat=8,
        num_repeat_feat=5,
    )
    frontend_config = TwoLayer1DFrontendConfig(
        in_features=80,
        conv1_channels=512,
        conv2_channels=512,
        conv1_kernel_size=5,
        conv2_kernel_size=5,
        conv1_stride=3,
        conv2_stride=2,
        dropout=0.1,
    )
    model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        conformer_size=512,
        num_layers=8,
        num_heads=4,
        ff_dim=512,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )

    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw_02 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            #############
            "batch_size": 200 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
    }

    # normal conv with 31 but with transparent attention
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
        "with_devtrain": True,
    }

    model_config_12x384 = copy.deepcopy((model_config))
    model_config_12x384.ff_dim = 1536
    model_config_12x384.num_layers = 12
    model_config_12x384.conformer_size = 384
    model_config_12x384.frontend_config.conv1_channels = 256
    model_config_12x384.frontend_config.conv2_channels = 384
    train_args = copy.deepcopy(train_args)
    train_args["net_args"]["model_config_dict"] = asdict(model_config_12x384)
    for lm_weight in [1.0, 1.2, 1.4, 1.6, 1.8]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(
            prefix_name + "/bpe_tune/conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/12x384_bs20_lm%.1f" % lm_weight,
            datasets=train_data,
            train_args=train_args, search_args=search_args)


    model_config_12x384_sub4 = copy.deepcopy(model_config_12x384)
    model_config_12x384_sub4.frontend_config.conv1_stride = 2
    train_args = copy.deepcopy(train_args)
    train_args["net_args"]["model_config_dict"] = asdict(model_config_12x384_sub4)
    for lm_weight in [1.0, 1.2, 1.4, 1.6, 1.8]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 20,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
        }
        run_exp(
            prefix_name + "/bpe_tune/conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/sub4_12x384_bs20_lm%.1f" % lm_weight,
            datasets=train_data,
            train_args=train_args, search_args=search_args)