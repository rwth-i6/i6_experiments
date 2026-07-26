import copy
from dataclasses import dataclass
from sisyphus import tk
from typing import List, Optional, cast


from dataclasses import asdict
import numpy as np

#from i6_experiments.users.jxu.experiments.transducer.voxpopuli.data import get_voxpopuli_data
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datasets import Dataset
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.data.bpe import build_bpe_training_datasets
from i6_experiments.users.nikolov.experiments.voxpopuli.datasets.voxpopuli.get_data import get_voxpopuli_data_per_lang, get_csfleurs_data_per_set, get_fleurs_data_per_set, get_switchlingua_data_per_set, get_data_hdf, get_miami_data_per_set
from i6_experiments.users.nikolov.experiments.voxpopuli.datasets.voxpopuli.lexicon import get_text_lexicon
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.train_util import build_test_dataset, TrainingDatasetSettings



# NOTE: Switched to config_noreturnn because Engram functions pass keep_epochs=,
# which is supported by config_noreturnn.get_training_config but not config.get_training_config.
# Pre-existing functions (conformer_rnnt_baseline, conformer_noreturnn) have local
# imports that override this module-level binding.
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config_noreturnn import get_training_config, get_search_config, get_prior_config
#from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config import get_training_config, get_search_config, get_prior_config
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.engram_acoustic_ctc import get_model_config
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.gpt2_ngram_ctc import (
    get_model_config as get_gpt2_model_config,
)


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset

@dataclass(frozen=True)
class CTCTrainingDatasets(TrainingDatasets):
    train: Dataset
    cv: Dataset
    prior: Optional[Dataset]


def conformer_rnnt_baseline(
    bpe_size: int, 
    batch_size: int = 120,
    learning_rates = [2e-4, 3e-4, 4e-4, 5e-4, 6e-4],
    vocab_size: Optional[int] = 10396):
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.i6modelsV1_VGG4LayerActFrontendV1_v7 import get_model_config
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config import get_training_config, get_search_config, get_prior_config
    from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pipeline import training, search, compute_prior
    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_rnnt/baseline/"

    train_data = TrainingDatasets(
        train=get_voxpopuli_data("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="train",
                                    partition_epoch=20),
        cv=get_voxpopuli_data("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="dev",
                                    partition_epoch=1)
    )
    RETURNN_EXE = tk.Path("/usr/bin/python3")
    RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/git/returnn")

    # ---------------------------------------------------------------------------------------------------------------- #
    def run_exp(ft_name, datasets, train_args, search_args=None, num_epochs=600,
                decoder="rnnt.decoder.experimental_rnnt_decoder", with_prior=False, evaluate_epoch=None):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, RETURNN_ROOT, num_epochs=num_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs

        returnn_search_config = get_search_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)
        return train_job

    model_config = get_model_config(vocab_size_without_blank=vocab_size,network_args={})


    for peak_lr in learning_rates:
        train_args_adamw03_accum2_jjlr = {
            "config": {
                "extern_data" : {"data": {"dim": 1}, "targets": {"dim": vocab_size, "sparse": True}},
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr/100, peak_lr, 270)) + list(
        np.linspace(peak_lr, peak_lr/100, 270)) + list(np.linspace(peak_lr/100, 1e-8, 60)),
                "batch_size": 180 * 16000,
                "max_seq_length": {"data": 35 * 16000},
                "min_seq_length": {"data": 640},
                "accum_grad_multiple_step": 2,
            },
            "debug": True,
        }

        train_args = {
            **copy.deepcopy(train_args_adamw03_accum2_jjlr),
            "network_module": "i6modelsV1_VGG4LayerActFrontendV1_v7",
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        train_args["config"]["batch_size"] = batch_size * 16000
        train_args["config"]["accum_grad_multiple_step"] = 1
        search_args = {
        }

        train_job = run_exp(
            prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_transparent/bs12/{batch_size}_{bpe_size}_lr{peak_lr}",
            datasets=train_data, train_args=train_args, search_args=search_args, with_prior=False)

        train_job.rqmt["gpu_mem"] = 48
        tk.register_output(f"output/{batch_size}_{bpe_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
        tk.register_output(f"output/{batch_size}_{bpe_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)



def conformer_noreturnn(
    bpe_size: int, 
    batch_size: int = 120,
    vocab_size: Optional[int] = 10396,
    learning_rates = [8e-5, 1e-4, 2e-4],
    gpu_mem: int = 24,
    lang_list = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"],
    vocab_name: Optional[str] = None):
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config_noreturnn import get_training_config, get_search_config, get_prior_config
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.i6modelsV1_VGG4LayerActFrontendV1_v7_onnx_exportable import get_model_config
    from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pipeline import training, search, compute_prior

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_rnnt/baseline/"
    network_module = "i6modelsV1_VGG4LayerActFrontendV1_v7_onnx_exportable" 
    
    if not vocab_name:
            vocab_name = f"bpe_{bpe_size}.vocab" 


    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=5,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    #label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = vocab_size

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev", "test"]:
    splits = ['train', 'test', 'dev']
    
    for lang in lang_list:
        #tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")
        test_dataset_tuples[lang] = get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="test",
                                    lang_list=[lang],
                                    partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")


    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")
    #MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/git/returnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    # ---------------------------------------------------------------------------------------------------------------- #
    def run_exp(ft_name, datasets, train_args, search_args=None, recog_args=None, num_epochs=600,
                decoder="rnnt.decoder.experimental_rnnt_decoder", network_module = "i6modelsV1_VGG4LayerActFrontendV1_v7_onnx_exportable", with_prior=False, evaluate_epoch=None):
        #training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        #returnn_config.black_formatting = False
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs


        returnn_search_config = get_search_config(**recog_args, decoder_args=search_args,
                                                  decoder=decoder)
        # _, _, search_jobs = search(ft_name + "/default_%i" % evaluate_epoch, returnn_search_config,
        #                            train_job.out_checkpoints[evaluate_epoch], test_dataset_tuples, RETURNN_EXE,
        #                            MINI_RETURNN_ROOT, use_gpu=search_args.get("use_gpu", False))
        # return train_job, search_jobs

        _, _, search_jobs = search(ft_name + "/default_%i" % evaluate_epoch, returnn_search_config,
                           train_job.out_checkpoints[evaluate_epoch], test_dataset_tuples, RETURNN_EXE,
                           MINI_RETURNN_ROOT, use_gpu=search_args.get("use_gpu", False))

        return train_job, search_jobs

    model_config = get_model_config(vocab_size_without_blank=vocab_size,network_args={})


    for peak_lr in learning_rates:
        train_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr/100, peak_lr, 270)) + list(
        np.linspace(peak_lr, peak_lr/100, 270)) + list(np.linspace(peak_lr/100, 1e-8, 60)),
                #############
                "batch_size": 180 * 16000,
                "max_seq_length": {"data": 35 * 16000},
                "min_seq_length": {"data": 640},
                "accum_grad_multiple_step": 2,
                "torch_amp_options": {"dtype": "bfloat16"},
            },
            "debug": True,
        }

        recog_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr/100, peak_lr, 270)) + list(
        np.linspace(peak_lr, peak_lr/100, 270)) + list(np.linspace(peak_lr/100, 1e-8, 60)),
                #############
                "batch_size": 180 * 16000,
                "accum_grad_multiple_step": 2,
            },
            "debug": True,
        }

        train_args = {
            **copy.deepcopy(train_args_adamw03_accum2_jjlr),
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        train_args["config"]["batch_size"] = batch_size * 16000
        train_args["config"]["accum_grad_multiple_step"] = 1

        recog_args = {
            **copy.deepcopy(recog_args_adamw03_accum2_jjlr),
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        recog_args["config"]["batch_size"] = batch_size * 16000
        recog_args["config"]["accum_grad_multiple_step"] = 1
        search_args = {
        "beam_size": 12,
        "returnn_vocab": tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),  
    }

        if len(lang_list) < 16:
            train_data = TrainingDatasets(
                train=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="train",
                                    lang_list=lang_list,
                                    partition_epoch=20),
                cv=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="dev",
                                    lang_list=lang_list,
                                    partition_epoch=1)
    )

            train_job, search_job = run_exp(
            prefix_name + f"conformer_new/{network_module}/rnnt/bpe_{bpe_size}/{'_'.join(lang_list)}/{batch_size}_{vocab_size}_lr{peak_lr}",
            datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args, with_prior=False, evaluate_epoch=600)
            
            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(f"output/rnnt/bpe_{bpe_size}/{'_'.join(lang_list)}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
            tk.register_output(f"output/rnnt/bpe_{bpe_size}/{'_'.join(lang_list)}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)
        
        else:
            train_data = TrainingDatasets(
                train=get_voxpopuli_data("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="train",
                                    partition_epoch=20),
                cv=get_voxpopuli_data("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="dev",
                                    partition_epoch=1)
    )

            train_job, search_job = run_exp(
            prefix_name + f"conformer_new/{network_module}/rnnt/bpe_{bpe_size}/{batch_size}_{vocab_size}_lr{peak_lr}",
            datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args, with_prior=False, evaluate_epoch=600)
            
            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(f"output/rnnt/bpe_{bpe_size}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
            tk.register_output(f"output/rnnt/bpe_{bpe_size}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)


def conformer_ctc_noreturnn(
    bpe_size: dict[str, int], 
    lexicon_path: str = None,
    batch_size: int = 120,
    gpu_mem: int = 48,
    recog_mem: int = 10,
    vocab_size: int = 10396,
    learning_rates = [8e-5, 1e-4, 2e-4],
    lang_list = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"],
    add_prefix: bool = False,
    eval_epoch: int = 600,
    keep_epochs = None,
    test_set: str = "voxpopuli",
    test_set_hdf: str = None,
    vocab_name:str=None,
    separate_heads: bool = False):
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config_noreturnn import get_training_config, get_search_config, get_prior_config

    from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pipeline import training
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pipeline_flashlight import search, prepare_asr_model

    if lexicon_path == None:
        lexicon_path = f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size['base']}"
    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_ctc/baseline/"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=5,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    #label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    if not vocab_name:
            vocab_name = f"bpe_{vocab_size}.vocab" 

    vocab_size_without_blank = vocab_size
    

    # build testing datasets
    test_dataset_tuples = {}
    splits = ['train', 'test', 'dev']
    
    # TODO: update to new get_data_per_set()
    if test_set == "voxpopuli":
        for lang in lang_list:
            test_dataset_tuples[lang] = get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                        lexicon_path,
                                        split="test",
                                        lang_list=[lang],
                                        partition_epoch=1,
                                        separate_heads=separate_heads), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")
    elif test_set == "csfleurs":
        datasets = ["mms", "read", "xtts"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_csfleurs_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/csfleurs_asr",
                                        split="test",
                                        set_list=[dataset],
                                        partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/csfleurs/corpus/{dataset}.test.corpus.xml.gz")
    elif test_set == "fleurs":
        langs = ["en_us", "es_419"]

        for lang in langs:
            test_dataset_tuples[lang] = get_fleurs_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/fleurs_asr",
                                        split="test",
                                        set_list=lang,
                                        partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/fleurs/corpus/{lang}/test.corpus.xml.gz")
    elif test_set == "miami":
        datasets = ["full", "spa", "eng"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_miami_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr",
                                        split="test",
                                        set_list=[dataset],
                                        partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz")
    elif test_set == "switchlingua":
        test_dataset_tuples['dev'] = get_switchlingua_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
                                    split="dev",
                                    partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")
    elif test_set == "switchlingua-tts":
        test_dataset_tuples['train'] = get_switchlingua_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr_tts",
                                    target_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr_tts",
                                    split="train",
                                    partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/TTS/generated_audio/generated/train.corpus.xml.gz")
    elif test_set == "TTS":
        test_dataset_tuples['test'] = get_data_hdf(test_set_hdf),  tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")

    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    # ---------------------------------------------------------------------------------------------------------------- #
    def run_exp(ft_name, datasets, train_args, keep_epochs=None, search_args=None, recog_args=None, num_epochs=600, lexicon_path: str = None,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", with_prior=False, evaluate_epoch=None, recog_mem:int = 10, vocab_name:str=None, test_set_name:str = "voxpopuli"):
        #training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, keep_epochs=keep_epochs, **train_args)
        #returnn_config.black_formatting = False
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs

        returnn_search_config = get_search_config(**recog_args, decoder_args=search_args,
                                                  decoder=decoder)
        from ..ctc_rnnt_standalone_2024.pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig

        default_decoder_config = DecoderConfig(
        lexicon= get_text_lexicon(ft_name + "/text_lex", bpe_size, add_prefix, lexicon_path),
        returnn_vocab=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        #arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        )
        asr_model = prepare_asr_model(
        ft_name, train_job, train_args, with_prior=True, datasets=datasets, get_specific_checkpoint=evaluate_epoch
        )

        search_jobs, wers = search(
                           prefix_name=ft_name + "/default_%i" % evaluate_epoch, 
                           forward_config={},
                           asr_model=asr_model,
                           decoder_module=decoder, 
                           decoder_args={"config": asdict(default_decoder_config)}, 
                           test_dataset_tuples=test_dataset_tuples, 
                           returnn_exe=RETURNN_EXE,
                           returnn_root=MINI_RETURNN_ROOT, use_gpu=search_args.get("use_gpu", False),
                           mem_rqmt=recog_mem)

        return train_job, search_jobs

    if separate_heads:
        from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.ctc_separate_lid import get_model_config
        model_config = get_model_config(vocab_size_without_blank=vocab_size,network_args={})
        recog_model_config = get_model_config(vocab_size_without_blank=vocab_size,network_args={})
        network_module = "ctc.conformer_new.ctc_separate_lid"
        flashlight_decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2_splithead"
    else:
        from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.i6modelsV1_VGG4LayerActFrontendV1_v6_onnx_exportable import get_model_config
        model_config = get_model_config(vocab_size_without_blank=vocab_size,network_args={})
        recog_model_config = get_model_config(vocab_size_without_blank=vocab_size,network_args={})
        network_module = "ctc.conformer_new.i6modelsV1_VGG4LayerActFrontendV1_v6_onnx_exportable"
        flashlight_decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2"


    for peak_lr in learning_rates:
        train_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr/100, peak_lr, 270)) + list(
        np.linspace(peak_lr, peak_lr/100, 270)) + list(np.linspace(peak_lr/100, 1e-8, 60)),
                #############
                "batch_size": 180 * 16000,
                "max_seq_length": {"data": 35 * 16000},
                "min_seq_length": {"data": 640},
                "accum_grad_multiple_step": 2,
                "torch_amp_options": {"dtype": "bfloat16"},
            },
            "debug": True,
        }

        recog_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr/100, peak_lr, 270)) + list(
        np.linspace(peak_lr, peak_lr/100, 270)) + list(np.linspace(peak_lr/100, 1e-8, 60)),
                #############
                "batch_size": 180 * 16000,
                "accum_grad_multiple_step": 2,
            },
            "debug": True,
        }
    

        train_args = {
            **copy.deepcopy(train_args_adamw03_accum2_jjlr),
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        train_args["config"]["batch_size"] = batch_size * 16000
        train_args["config"]["accum_grad_multiple_step"] = 1

        recog_args = {
            **copy.deepcopy(recog_args_adamw03_accum2_jjlr),
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(recog_model_config)},
        }

        recog_args["config"]["batch_size"] = batch_size * 16000
        recog_args["config"]["accum_grad_multiple_step"] = 1
        search_args = {
        "beam_size": 12,
        "returnn_vocab": tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
    }

        if len(lang_list) < 16:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    lexicon_path,
                                    split="train",
                                    lang_list=lang_list,
                                    partition_epoch=20),
                cv=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    lexicon_path,
                                    split="dev",
                                    lang_list=lang_list,
                                    partition_epoch=1),
                prior=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    lexicon_path,
                                    split="train",
                                    lang_list=lang_list,
                                    partition_epoch=1),
    )

            train_job, search_job = run_exp(
            prefix_name + f"conformer_new/i6modelsV1_VGG4LayerActFrontendV1_v6/ctc/{vocab_name.split('.')[0] + ('_prefixed' if vocab_name.split('.')[-1] == 'prefixed' else '')}/{'_'.join(lang_list)}/{batch_size}_{vocab_size}_lr{peak_lr}",
            datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args,lexicon_path=lexicon_path, with_prior=False, evaluate_epoch=eval_epoch,decoder=flashlight_decoder, vocab_name=vocab_name, test_set_name=test_set, keep_epochs=keep_epochs)
            
            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
            tk.register_output(f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)
        
        else:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    lexicon_path,
                                    split="train",
                                    partition_epoch=20,
                                    separate_heads=separate_heads),
                cv=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    lexicon_path,
                                    split="dev",
                                    partition_epoch=1,
                                    separate_heads=separate_heads),
                prior=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    lexicon_path,
                                    split="train",
                                    lang_list=lang_list,
                                    partition_epoch=1,
                                    separate_heads=separate_heads),
    )
            train_job, search_job = run_exp(
            prefix_name + f"conformer_new/{network_module.split('.')[-1]}/ctc/{vocab_name.split('.')[0]}/{test_set}/{batch_size}_{vocab_size}_lr{peak_lr}",
            datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args, lexicon_path=lexicon_path, with_prior=False, evaluate_epoch=eval_epoch, recog_mem=recog_mem, decoder=flashlight_decoder, vocab_name=vocab_name, test_set_name=test_set, keep_epochs=keep_epochs)
            
            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
            tk.register_output(f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)


def lid_aware_sc_ctc(
        bpe_size: int | dict[str, int],
        lexicon_path: str = None,
        batch_size: int = 120,
        gpu_mem: int = 48,
        recog_mem: int = 10,
        vocab_size: int = 10396,
        learning_rates=[8e-5, 1e-4, 2e-4],
        lang_list=["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"],
        add_prefix: bool = False,
        eval_epoch: int = 600,
        keep_epochs=None,
        test_set: str = "voxpopuli",
        vocab_name: str = None,
        specaug_start_epoch: int = 1):

    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config_noreturnn import get_training_config
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.jxu_language_aware_sc_ctc_onnx import get_model_config
    from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pipeline import training
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pipeline_flashlight import search, prepare_asr_model


    base_bpe_size = bpe_size["base"] if isinstance(bpe_size, dict) else bpe_size
    if lexicon_path is None:
        lexicon_path = f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{base_bpe_size}"
    if not vocab_name:
        vocab_name = f"bpe_{vocab_size}.vocab"
        
    # build testing datasets
    test_dataset_tuples = {}
    splits = ['train', 'test', 'dev']
    
    # TODO: update to new get_data_per_set()
    if test_set == "voxpopuli":
        for lang in lang_list:
            test_dataset_tuples[lang] = get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                        lexicon_path,
                                        split="test",
                                        lang_list=[lang],
                                        partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")
    elif test_set == "csfleurs":
        datasets = ["mms", "read", "xtts"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_csfleurs_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/csfleurs_asr",
                                        split="test",
                                        set_list=[dataset],
                                        partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/csfleurs/corpus/{dataset}.test.corpus.xml.gz")
    elif test_set == "fleurs":
        langs = ["en_us", "es_419"]

        for lang in langs:
            test_dataset_tuples[lang] = get_fleurs_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/fleurs_asr",
                                        split="test",
                                        set_list=lang,
                                        partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/fleurs/corpus/{lang}/test.corpus.xml.gz")
    elif test_set == "miami":
        datasets = ["full", "spa", "eng"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_miami_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr",
                                        split="test",
                                        set_list=[dataset],
                                        partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz")
    elif test_set == "switchlingua":
        test_dataset_tuples['dev'] = get_switchlingua_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
                                    split="dev",
                                    partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")
    elif test_set == "switchlingua-tts":
        test_dataset_tuples['train'] = get_switchlingua_data_per_set("/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr_tts",
                                    target_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr_tts",
                                    split="train",
                                    partition_epoch=1), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/TTS/generated_audio/generated/train.corpus.xml.gz")
    elif test_set == "TTS":
        test_dataset_tuples['test'] = get_data_hdf(test_set_hdf),  tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")


    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_ctc/baseline/"
    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")
    network_module = "ctc.conformer_new.jxu_language_aware_sc_ctc"
    vocab_stem = vocab_name.split(".")[0] + ("_prefixed" if vocab_name.split(".")[-1] == "prefixed" else "")

    def run_exp(ft_name, datasets, train_args, keep_epochs=None, search_args=None, recog_args=None, num_epochs=600, lexicon_path: str = None,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", with_prior=False, evaluate_epoch=None, recog_mem:int = 10, vocab_name:str=None, test_set_name:str = "voxpopuli"):
        
        search_args = search_args if search_args is not None else {}
        returnn_config = get_training_config(training_datasets=datasets, keep_epochs=keep_epochs, **train_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)
        
        if not evaluate_epoch:
            evaluate_epoch = num_epochs
        
        returnn_search_config = get_search_config(**recog_args, decoder_args=search_args,
                                                  decoder=decoder)
        from ..ctc_rnnt_standalone_2024.pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig

        default_decoder_config = DecoderConfig(
        lexicon= get_text_lexicon(ft_name + "/text_lex", bpe_size, add_prefix, lexicon_path),
        returnn_vocab=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        #arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        )
        asr_model = prepare_asr_model(
        ft_name, train_job, train_args, with_prior=True, datasets=datasets, get_specific_checkpoint=evaluate_epoch
        )

        search_jobs, wers = search(
                           prefix_name=ft_name + "/default_%i" % evaluate_epoch, 
                           forward_config={},
                           asr_model=asr_model,
                           decoder_module=decoder, 
                           decoder_args={"config": asdict(default_decoder_config)}, 
                           test_dataset_tuples=test_dataset_tuples, 
                           returnn_exe=RETURNN_EXE,
                           returnn_root=MINI_RETURNN_ROOT, use_gpu=search_args.get("use_gpu", False),
                           mem_rqmt=recog_mem)
        
        
        return train_job, search_jobs

    model_config = get_model_config(
        vocab_size_without_blank=vocab_size,
        network_args={"specauc_start_epoch": specaug_start_epoch, "lid_sc_layer": 3, "sc_layer": [6, 9]},
    )

    for peak_lr in learning_rates:
        train_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr / 100, peak_lr, 270))
                + list(np.linspace(peak_lr, peak_lr / 100, 270))
                + list(np.linspace(peak_lr / 100, 1e-8, 60)),
                "batch_size": 180 * 16000,
                "max_seq_length": {"data": 35 * 16000},
                "min_seq_length": {"data": 640},
                "accum_grad_multiple_step": 2,
                "torch_amp_options": {"dtype": "bfloat16"},
                "extern_data": {
                    "data": {"dim": 1},
                    "targets": {"dim": vocab_size + 1, "sparse": True},
                    "language": {"dim": 16, "sparse": True},
                },
            },
            "debug": True,
        }

        train_args = {
            **copy.deepcopy(train_args_adamw03_accum2_jjlr),
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        train_args["config"]["batch_size"] = batch_size * 16000
        train_args["config"]["accum_grad_multiple_step"] = 1

        common_dataset_args = {
            "audio_base_dir": "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
            "target_base_dir": lexicon_path,
            "partition_epoch": 1,
            "separate_heads": True,
        }
        recog_model_config = model_config
        flashlight_decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2"
        recog_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr/100, peak_lr, 270)) + list(
        np.linspace(peak_lr, peak_lr/100, 270)) + list(np.linspace(peak_lr/100, 1e-8, 60)),
                #############
                "batch_size": 180 * 16000,
                "accum_grad_multiple_step": 2,
            },
            "debug": True,
        }
        recog_args = {
            **copy.deepcopy(recog_args_adamw03_accum2_jjlr),
            "network_module": network_module + "_onnx",
            "net_args": {"model_config_dict": asdict(recog_model_config)},
        }

        recog_args["config"]["batch_size"] = batch_size * 16000
        recog_args["config"]["accum_grad_multiple_step"] = 1
        search_args = {
            "beam_size": 12,
            "returnn_vocab": tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        }


        if len(lang_list) < 16:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang(
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=20,
                    separate_heads=True,
                    audio_base_dir=common_dataset_args["audio_base_dir"],
                    target_base_dir=common_dataset_args["target_base_dir"],
                ),
                cv=get_voxpopuli_data_per_lang(
                    split="dev",
                    lang_list=lang_list,
                    **common_dataset_args,
                ),
                prior=get_voxpopuli_data_per_lang(
                    split="train",
                    lang_list=lang_list,
                    **common_dataset_args,
                ),
            )
            exp_name = prefix_name + (
                f"conformer_new/{network_module.split('.')[-1]}/ctc/{vocab_stem}/{'_'.join(lang_list)}/"
                f"{batch_size}_{vocab_size}_lr{peak_lr}"
            )
            output_prefix = f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}"
        else:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang(
                    split="train",
                    partition_epoch=20,
                    separate_heads=True,
                    audio_base_dir=common_dataset_args["audio_base_dir"],
                    target_base_dir=common_dataset_args["target_base_dir"],
                ),
                cv=get_voxpopuli_data_per_lang(
                    split="dev",
                    **common_dataset_args,
                ),
                prior=get_voxpopuli_data_per_lang(
                    split="train",
                    lang_list=lang_list,
                    **common_dataset_args,
                ),
            )
            exp_name = prefix_name + (
                f"conformer_new/{network_module.split('.')[-1]}/ctc/{vocab_name.split('.')[0]}/{test_set}/{batch_size}_{vocab_size}_lr{peak_lr}"
            )
            output_prefix = f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}"

        train_job, search_jobs = run_exp(exp_name, datasets=train_data, search_args=search_args, recog_args=recog_args, recog_mem=recog_mem, train_args=train_args, vocab_name=vocab_name)
        train_job.rqmt["gpu_mem"] = gpu_mem
        tk.register_output(f"{output_prefix}/learning_rates", train_job.out_learning_rates)
        tk.register_output(f"{output_prefix}/out_model_dir", train_job.out_model_dir)
def engram_v2_ctc_noreturnn(
    bpe_size: int | dict[str, int],
    lexicon_path: str = None,
    batch_size: int = 120,
    gpu_mem: int = 48,
    recog_mem: int = 10,
    vocab_size: int = 10396,
    learning_rates = [8e-5, 1e-4, 2e-4],
    lang_list = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"],
    add_prefix: bool = False,
    eval_epoch: int = 600,
    keep_epochs = None,
    test_set: str = "voxpopuli",
    test_set_hdf: str = None,
    vocab_name: str = None,
    separate_heads: bool = False,
    # Engram-v2 specific args
    engram_table_size: int = 2**12,
    engram_num_heads: int = 8,
    engram_ngram_orders: List[List[int]] = [[2, 3], [2, 3], [3, 4]],
    engram_mem_dim: int = 1280,
    engram_layers: List[int] = [2, 6, 10],
    acoustic_num_bins: int = 32,
    sample_rate: int = 16000,
    frame_ms: int = 40,
    conformer_size: int = 512,
    num_layers: int = 12,
    specaug_start_epoch: int = 1,
):
    """
    Train an Engram-v2 augmented CTC Conformer with encoder-free architecture.

    Unlike v1 (which uses mel-spectrogram + VGG frontend + acoustic quantizer),
    v2 uses Gemma 4-style raw waveform embedding:
      - Raw audio -> 40ms frames -> Linear projection -> Conformer
      - No mel-spectrogram, no VGG frontend
      - Engram keys evolve: Layer 2 (amplitude bins) -> Layer 6 (LID) -> Layer 10 (BPE)

    Uses the same data pipeline as v1 (requires language ID data).
    """
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.engram_v2_ctc_onnx import get_model_config as get_recog_config

    base_bpe_size = bpe_size["base"] if isinstance(bpe_size, dict) else bpe_size
    if lexicon_path is None:
        lexicon_path = f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{base_bpe_size}"
    if not vocab_name:
        vocab_name = f"bpe_{vocab_size}.vocab"

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_ctc/engram_v2/"
    network_module = "ctc.conformer_new.engram_v2_ctc"
    vocab_stem = vocab_name.split(".")[0] + ("_prefixed" if vocab_name.split(".")[-1] == "prefixed" else "")

    # Build test datasets (same as v1)
    test_dataset_tuples = {}
    if test_set == "voxpopuli":
        for lang in lang_list:
            test_dataset_tuples[lang] = get_voxpopuli_data_per_lang(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                lexicon_path,
                split="test",
                lang_list=[lang],
                partition_epoch=1,
                separate_heads=separate_heads,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")
    elif test_set == "switchlingua":
        test_dataset_tuples['dev'] = get_switchlingua_data_per_set(
            "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
            split="dev",
            partition_epoch=1,
        ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")
    elif test_set == "miami":
        datasets = ["full", "spa", "eng"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_miami_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr",
                split="test",
                set_list=[dataset],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz")
    elif test_set == "csfleurs":
        datasets = ["mms", "read", "xtts"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_csfleurs_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/csfleurs_asr",
                split="test",
                set_list=[dataset],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/csfleurs/corpus/{dataset}.test.corpus.xml.gz")
    elif test_set == "fleurs":
        langs = ["en_us", "es_419"]
        for lang in langs:
            test_dataset_tuples[lang] = get_fleurs_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/fleurs_asr",
                split="test",
                set_list=lang,
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/fleurs/corpus/{lang}/test.corpus.xml.gz")

    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    def run_exp(ft_name, datasets, train_args, search_args=None, recog_args=None, num_epochs=600,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", with_prior=False, evaluate_epoch=None, recog_mem: int = 10,
                vocab_name: str = None, test_set_name: str = "voxpopuli", keep_epochs=None):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, keep_epochs=keep_epochs, **train_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs

        returnn_search_config = get_search_config(**recog_args, decoder_args=search_args, decoder=decoder)

        default_decoder_config = DecoderConfig(
            lexicon=get_text_lexicon(ft_name + "/text_lex", bpe_size, add_prefix, lexicon_path),
            returnn_vocab=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
            beam_size=1024,
            beam_size_token=12,
            beam_threshold=14,
        )
        asr_model = prepare_asr_model(
            ft_name, train_job, train_args, with_prior=True, datasets=datasets, get_specific_checkpoint=evaluate_epoch
        )

        search_jobs, wers = search(
            prefix_name=ft_name + "/default_%i" % evaluate_epoch,
            forward_config={},
            asr_model=asr_model,
            decoder_module=decoder,
            decoder_args={"config": asdict(default_decoder_config)},
            test_dataset_tuples=test_dataset_tuples,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT, use_gpu=search_args.get("use_gpu", False),
            mem_rqmt=recog_mem,
        )

        return train_job, search_jobs

    # Build model configs
    model_config = get_model_config(
        vocab_size_without_blank=vocab_size,
        network_args={
            "sample_rate": sample_rate,
            "frame_ms": frame_ms,
            "conformer_size": conformer_size,
            "num_layers": num_layers,
            "lid_sc_layer": 3,
            "lid_classes": 17,
            "engram_layers": engram_layers,
            "engram_ngram_orders": engram_ngram_orders,
            "engram_num_heads": engram_num_heads,
            "engram_mem_dim": engram_mem_dim,
            "engram_table_size": engram_table_size,
            "acoustic_num_bins": acoustic_num_bins,
            "specaug_start_epoch": specaug_start_epoch,
            "bpe_key_warmup_steps": 50000,
            "bpe_key_temperature": 1.0,
        },
    )
    recog_model_config = get_recog_config(
        vocab_size_without_blank=vocab_size,
        network_args={
            "sample_rate": sample_rate,
            "frame_ms": frame_ms,
            "conformer_size": conformer_size,
            "num_layers": num_layers,
            "lid_sc_layer": 3,
            "lid_classes": 17,
            "engram_layers": engram_layers,
            "engram_ngram_orders": engram_ngram_orders,
            "engram_num_heads": engram_num_heads,
            "engram_mem_dim": engram_mem_dim,
            "engram_table_size": engram_table_size,
            "acoustic_num_bins": acoustic_num_bins,
            "specaug_start_epoch": specaug_start_epoch,
            "bpe_key_warmup_steps": 50000,
            "bpe_key_temperature": 1.0,
        },
    )

    for peak_lr in learning_rates:
        train_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr / 100, peak_lr, 270))
                + list(np.linspace(peak_lr, peak_lr / 100, 270))
                + list(np.linspace(peak_lr / 100, 1e-8, 60)),
                "batch_size": 180 * 16000,
                "max_seq_length": {"data": 35 * 16000},
                "min_seq_length": {"data": 640},
                "accum_grad_multiple_step": 2,
                "torch_amp_options": {"dtype": "bfloat16"},
                "extern_data": {
                    "data": {"dim": 1},
                    "targets": {"dim": vocab_size + 1, "sparse": True},
                    "language": {"dim": 16, "sparse": True},
                },
            },
            "debug": True,
        }

        recog_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr / 100, peak_lr, 270))
                + list(np.linspace(peak_lr, peak_lr / 100, 270))
                + list(np.linspace(peak_lr / 100, 1e-8, 60)),
                "batch_size": 180 * 16000,
                "accum_grad_multiple_step": 2,
            },
            "debug": True,
        }

        train_args = {
            **copy.deepcopy(train_args_adamw03_accum2_jjlr),
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        train_args["config"]["batch_size"] = batch_size * 16000
        train_args["config"]["accum_grad_multiple_step"] = 1

        recog_args = {
            **copy.deepcopy(recog_args_adamw03_accum2_jjlr),
            "network_module": network_module + "_onnx",
            "net_args": {"model_config_dict": asdict(recog_model_config)},
        }
        recog_args["config"]["batch_size"] = batch_size * 16000
        recog_args["config"]["accum_grad_multiple_step"] = 1
        search_args = {
            "beam_size": 12,
            "returnn_vocab": tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        }

        if len(lang_list) < 16:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=20,
                    separate_heads=separate_heads,
                ),
                cv=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="dev",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=separate_heads,
                ),
                prior=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=separate_heads,
                ),
            )

            train_job, search_job = run_exp(
                prefix_name + f"conformer_new/engram_v2_ctc/ctc/{vocab_stem}/{'_'.join(lang_list)}/{batch_size}_{vocab_size}_lr{peak_lr}",
                datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args,
                with_prior=False, evaluate_epoch=eval_epoch, decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2",
                vocab_name=vocab_name, test_set_name=test_set, keep_epochs=keep_epochs,
            )

            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(
                f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates",
                train_job.out_learning_rates,
            )
            tk.register_output(
                f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir",
                train_job.out_model_dir,
            )
        else:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=20,
                ),
                cv=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="dev",
                    lang_list=lang_list,
                    partition_epoch=1,
                ),
                prior=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=1,
                ),
            )

            train_job, search_job = run_exp(
                prefix_name + f"conformer_new/engram_v2_ctc/ctc/{vocab_name.split('.')[0]}/{test_set}/{batch_size}_{vocab_size}_lr{peak_lr}",
                datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args,
                with_prior=False, evaluate_epoch=eval_epoch, recog_mem=recog_mem,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", vocab_name=vocab_name, test_set_name=test_set,
                keep_epochs=keep_epochs,
            )

            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(
                f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates",
                train_job.out_learning_rates,
            )
            tk.register_output(
                f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir",
                train_job.out_model_dir,
            )

    return train_job, search_job
def gpt2_ngram_ctc_noreturnn(
    bpe_size: int | dict[str, int],
    lexicon_path: str = None,
    batch_size: int = 120,
    gpu_mem: int = 48,
    recog_mem: int = 10,
    vocab_size: int = 10396,
    learning_rates = [1e-4, 5e-5, 1e-5],
    lang_list = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"],
    add_prefix: bool = False,
    eval_epoch: int = 600,
    keep_epochs = None,
    test_set: str = "voxpopuli",
    vocab_name: str = None,
    separate_heads: bool = False,
    # GPT-2 config
    gpt2_n_layer: int = 12,
    gpt2_n_head: int = 12,
    gpt2_n_embd: int = 768,
    gpt2_ff_dim: int = 3072,
    # Engram config
    engram_layers: List[int] = [2, 6],
    engram_lid_layer: int = 6,
    engram_audio_bins: int = 32,
    engram_lid_classes: int = 17,
    engram_ngram_orders: List[int] = [2, 3],
    engram_num_heads: int = 8,
    engram_mem_dim: int = 1280,
    engram_table_size: int = 2**12,
    engram_dropout: float = 0.0,
    # Audio config
    sample_rate: int = 16000,
    frame_ms: int = 40,
):
    """
    Train a GPT-2 Ngram CTC model — encoder-free, decoder-only transformer
    with Engram conditional memory.

    Unlike the Engram-acoustic CTC model, this model:
      - Does NOT use a Conformer encoder (GPT-2 blocks process raw audio directly)
      - Does NOT use log-mel feature extraction (linear projection from raw frames)
      - Does NOT require external LID data (LID is predicted internally at layer 6)
      - Uses the standard CTC data pipeline (extern_data without "language")

    Architecture:
      Raw audio -> 40ms frames -> Linear(640->768) -> GPT-2 blocks
      |- Layer 2: Engram (keys = quantized audio amplitudes)
      +- Layer 6: LID head -> Engram (keys = LID predictions)
      -> LayerNorm -> CTC head

    Args:
        bpe_size: BPE vocabulary size (int or dict with 'base' key)
        batch_size: Training batch size
        gpu_mem: GPU memory allocation in GB
        vocab_size: Vocabulary size (without blank token)
        learning_rates: List of peak learning rates for schedule
        lang_list: Languages to include in training/testing
        eval_epoch: Epoch to evaluate (checkpoint to use for recognition)
        keep_epochs: Which epochs to keep (for cleanup)
        test_set: Which test set to decode
        engram_layers: Layer numbers (1-indexed) for Engram injection
        engram_lid_layer: Layer number for LID head (also Engram layer)
        engram_audio_bins: Number of quantization bins for audio keys
        gpt2_n_layer: Number of GPT-2 transformer blocks
        gpt2_n_head: Number of attention heads
        gpt2_n_embd: Model dimension
        gpt2_ff_dim: Feed-forward inner dimension
    """
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.gpt2_ngram_ctc_onnx import (
        get_model_config as get_recog_config,
    )

    base_bpe_size = bpe_size["base"] if isinstance(bpe_size, dict) else bpe_size
    if lexicon_path is None:
        lexicon_path = f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{base_bpe_size}"
    if not vocab_name:
        vocab_name = f"bpe_{vocab_size}.vocab"

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_ctc/gpt2_ngram/"
    network_module = "ctc.conformer_new.gpt2_ngram_ctc"

    # Build test datasets
    test_dataset_tuples = {}
    if test_set == "voxpopuli":
        for lang in lang_list:
            test_dataset_tuples[lang] = get_voxpopuli_data_per_lang(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                lexicon_path,
                split="test",
                lang_list=[lang],
                partition_epoch=1,
                separate_heads=separate_heads,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")
    elif test_set == "switchlingua":
        test_dataset_tuples['dev'] = get_switchlingua_data_per_set(
            "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
            split="dev",
            partition_epoch=1,
        ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")
    elif test_set == "miami":
        datasets = ["full", "spa", "eng"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_miami_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr",
                split="test",
                set_list=[dataset],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz")
    elif test_set == "csfleurs":
        datasets = ["mms", "read", "xtts"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_csfleurs_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/csfleurs_asr",
                split="test",
                set_list=[dataset],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/csfleurs/corpus/{dataset}.test.corpus.xml.gz")
    elif test_set == "fleurs":
        langs = ["en_us", "es_419"]
        for lang in langs:
            test_dataset_tuples[lang] = get_fleurs_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/fleurs_asr",
                split="test",
                set_list=lang,
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/fleurs/corpus/{lang}/test.corpus.xml.gz")

    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    def run_exp(ft_name, datasets, train_args, search_args=None, recog_args=None,
                num_epochs=600, decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2",
                with_prior=False, evaluate_epoch=None, recog_mem: int = 10,
                vocab_name: str = None, test_set_name: str = "voxpopuli",
                keep_epochs=None):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, keep_epochs=keep_epochs, **train_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs

        returnn_search_config = get_search_config(**recog_args, decoder_args=search_args, decoder=decoder)

        default_decoder_config = DecoderConfig(
            lexicon=get_text_lexicon(ft_name + "/text_lex", bpe_size, add_prefix, lexicon_path),
            returnn_vocab=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
            beam_size=1024,
            beam_size_token=12,
            beam_threshold=14,
        )
        asr_model = prepare_asr_model(
            ft_name, train_job, train_args, with_prior=True, datasets=datasets,
            get_specific_checkpoint=evaluate_epoch,
        )

        search_jobs, wers = search(
            prefix_name=ft_name + "/default_%i" % evaluate_epoch,
            forward_config={},
            asr_model=asr_model,
            decoder_module=decoder,
            decoder_args={"config": asdict(default_decoder_config)},
            test_dataset_tuples=test_dataset_tuples,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            use_gpu=search_args.get("use_gpu", False),
            mem_rqmt=recog_mem,
        )

        return train_job, search_jobs

    # Build model configs
    gpt2_network_args = {
        "n_layer": gpt2_n_layer,
        "n_head": gpt2_n_head,
        "n_embd": gpt2_n_embd,
        "ff_dim": gpt2_ff_dim,
        "engram_layers": engram_layers,
        "engram_lid_layer": engram_lid_layer,
        "engram_audio_bins": engram_audio_bins,
        "engram_lid_classes": engram_lid_classes,
        "engram_ngram_orders": engram_ngram_orders,
        "engram_num_heads": engram_num_heads,
        "engram_mem_dim": engram_mem_dim,
        "engram_table_size": engram_table_size,
        "engram_dropout": engram_dropout,
        "sample_rate": sample_rate,
        "frame_ms": frame_ms,
    }

    model_config = get_gpt2_model_config(
        vocab_size_without_blank=vocab_size,
        network_args=gpt2_network_args,
    )
    recog_model_config = get_recog_config(
        vocab_size_without_blank=vocab_size,
        network_args=gpt2_network_args,
    )

    # Training arguments — extern_data does NOT include "language"
    # (LID is internal/emergent in the GPT-2 model)
    for peak_lr in learning_rates:
        train_args_template = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr / 100, peak_lr, 270))
                + list(np.linspace(peak_lr, peak_lr / 100, 270))
                + list(np.linspace(peak_lr / 100, 1e-8, 60)),
                "batch_size": 180 * 16000,
                "max_seq_length": {"data": 35 * 16000},
                "min_seq_length": {"data": 640},
                "accum_grad_multiple_step": 2,
                "torch_amp_options": {"dtype": "bfloat16"},
                "extern_data": {
                    "data": {"dim": 1},
                    "targets": {"dim": vocab_size + 1, "sparse": True},
                },
            },
            "debug": True,
        }

        recog_args_template = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr / 100, peak_lr, 270))
                + list(np.linspace(peak_lr, peak_lr / 100, 270))
                + list(np.linspace(peak_lr / 100, 1e-8, 60)),
                "batch_size": 180 * 16000,
                "accum_grad_multiple_step": 2,
            },
            "debug": True,
        }

        train_args = {
            **copy.deepcopy(train_args_template),
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        train_args["config"]["batch_size"] = batch_size * 16000
        train_args["config"]["accum_grad_multiple_step"] = 1

        recog_args = {
            **copy.deepcopy(recog_args_template),
            "network_module": network_module + "_onnx",
            "net_args": {"model_config_dict": asdict(recog_model_config)},
        }
        recog_args["config"]["batch_size"] = batch_size * 16000
        recog_args["config"]["accum_grad_multiple_step"] = 1

        search_args = {
            "beam_size": 12,
            "returnn_vocab": tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        }

        # Datasets
        if len(lang_list) < 16:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=20,
                    separate_heads=separate_heads,
                ),
                cv=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="dev",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=separate_heads,
                ),
                prior=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=separate_heads,
                ),
            )

            train_job, search_job = run_exp(
                prefix_name + f"gpt2_ngram_ctc/ctc/{vocab_name.split('.')[0]}"
                + ("_prefixed" if add_prefix else "")
                + f"/{'_'.join(lang_list)}/{batch_size}_{vocab_size}_lr{peak_lr}",
                datasets=train_data, train_args=train_args, search_args=search_args,
                recog_args=recog_args, with_prior=False, evaluate_epoch=eval_epoch,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2",
                vocab_name=vocab_name, test_set_name=test_set, keep_epochs=keep_epochs,
            )

            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(
                f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates",
                train_job.out_learning_rates,
            )
            tk.register_output(
                f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir",
                train_job.out_model_dir,
            )
        else:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=20,
                    separate_heads=separate_heads,
                ),
                cv=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="dev",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=separate_heads,
                ),
                prior=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=separate_heads,
                ),
            )

            train_job, search_job = run_exp(
                prefix_name + f"gpt2_ngram_ctc/ctc/{vocab_name.split('.')[0]}/{test_set}/{batch_size}_{vocab_size}_lr{peak_lr}",
                datasets=train_data, train_args=train_args, search_args=search_args,
                recog_args=recog_args, with_prior=False, evaluate_epoch=eval_epoch,
                recog_mem=recog_mem, decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2",
                vocab_name=vocab_name, test_set_name=test_set, keep_epochs=keep_epochs,
            )

            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(
                f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates",
                train_job.out_learning_rates,
            )
            tk.register_output(
                f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir",
                train_job.out_model_dir,
            )

    return train_job, search_job
def engram_acoustic_ctc_noreturnn(
    bpe_size: int | dict[str, int],
    lexicon_path: str = None,
    batch_size: int = 120,
    gpu_mem: int = 48,
    recog_mem: int = 10,
    vocab_size: int = 10396,
    learning_rates = [8e-5, 1e-4, 2e-4],
    lang_list = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"],
    add_prefix: bool = False,
    eval_epoch: int = 600,
    keep_epochs = None,
    test_set: str = "voxpopuli",
    test_set_hdf: str = None,
    vocab_name: str = None,
    separate_heads: bool = False,
    # Engram-specific args
    engram_table_size: int = 2**12,
    engram_num_heads: int = 8,
    engram_ngram_orders: List[int] = [2, 3],
    engram_mem_dim: int = 1280,
    acoustic_codebook_size: int = 256,
    acoustic_feat_dim: int = 80,
    acoustic_codebook_dim: int = 64,
    specaug_start_epoch: int = 1,
):
    """
    Train an Engram-augmented CTC Conformer with acoustic quantization.

    This function mirrors conformer_ctc_noreturnn but uses the Engram model
    with acoustic keys instead of the standard CTC or LID+SC CTC models.

    The Engram model requires language ID data (same extern_data format as
    jxu_language_aware_sc_ctc), so the data pipeline is identical.
    """
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.engram_acoustic_ctc_onnx import get_model_config as get_recog_config

    base_bpe_size = bpe_size["base"] if isinstance(bpe_size, dict) else bpe_size
    if lexicon_path is None:
        lexicon_path = f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{base_bpe_size}"
    if not vocab_name:
        vocab_name = f"bpe_{vocab_size}.vocab"

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_ctc/engram/"
    network_module = "ctc.conformer_new.engram_acoustic_ctc"
    vocab_stem = vocab_name.split(".")[0] + ("_prefixed" if vocab_name.split(".")[-1] == "prefixed" else "")

    # Build test datasets
    test_dataset_tuples = {}
    if test_set == "voxpopuli":
        for lang in lang_list:
            test_dataset_tuples[lang] = get_voxpopuli_data_per_lang(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                lexicon_path,
                split="test",
                lang_list=[lang],
                partition_epoch=1,
                separate_heads=separate_heads,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")
    elif test_set == "switchlingua":
        test_dataset_tuples['dev'] = get_switchlingua_data_per_set(
            "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
            split="dev",
            partition_epoch=1,
        ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")
    elif test_set == "miami":
        datasets = ["full", "spa", "eng"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_miami_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr",
                split="test",
                set_list=[dataset],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz")
    elif test_set == "csfleurs":
        datasets = ["mms", "read", "xtts"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_csfleurs_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/csfleurs_asr",
                split="test",
                set_list=[dataset],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/csfleurs/corpus/{dataset}.test.corpus.xml.gz")
    elif test_set == "fleurs":
        langs = ["en_us", "es_419"]
        for lang in langs:
            test_dataset_tuples[lang] = get_fleurs_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/fleurs_asr",
                split="test",
                set_list=lang,
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/fleurs/corpus/{lang}/test.corpus.xml.gz")

    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    def run_exp(ft_name, datasets, train_args, search_args=None, recog_args=None, num_epochs=600,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", with_prior=False, evaluate_epoch=None, recog_mem: int = 10,
                vocab_name: str = None, test_set_name: str = "voxpopuli", keep_epochs=None):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, keep_epochs=keep_epochs, **train_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs

        returnn_search_config = get_search_config(**recog_args, decoder_args=search_args, decoder=decoder)

        default_decoder_config = DecoderConfig(
            lexicon=get_text_lexicon(ft_name + "/text_lex", bpe_size, add_prefix, lexicon_path),
            returnn_vocab=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
            beam_size=1024,
            beam_size_token=12,
            beam_threshold=14,
        )
        asr_model = prepare_asr_model(
            ft_name, train_job, train_args, with_prior=True, datasets=datasets, get_specific_checkpoint=evaluate_epoch
        )

        search_jobs, wers = search(
            prefix_name=ft_name + "/default_%i" % evaluate_epoch,
            forward_config={},
            asr_model=asr_model,
            decoder_module=decoder,
            decoder_args={"config": asdict(default_decoder_config)},
            test_dataset_tuples=test_dataset_tuples,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT, use_gpu=search_args.get("use_gpu", False),
            mem_rqmt=recog_mem,
        )

        return train_job, search_jobs

    # Build model config with Engram-specific parameters
    model_config = get_model_config(
        vocab_size_without_blank=vocab_size,
        network_args={
            "specaug_start_epoch": specaug_start_epoch,
            "lid_sc_layer": 3,
            "sc_layer": [6, 9],
            "engram_ngram_orders": engram_ngram_orders,
            "engram_num_heads": engram_num_heads,
            "engram_mem_dim": engram_mem_dim,
            "engram_table_size": engram_table_size,
            "acoustic_feat_dim": acoustic_feat_dim,
            "acoustic_codebook_dim": acoustic_codebook_dim,
            "acoustic_codebook_size": acoustic_codebook_size,
        },
    )
    recog_model_config = get_recog_config(
        vocab_size_without_blank=vocab_size,
        network_args={
            "specaug_start_epoch": specaug_start_epoch,
            "lid_sc_layer": 3,
            "sc_layer": [6, 9],
            "engram_ngram_orders": engram_ngram_orders,
            "engram_num_heads": engram_num_heads,
            "engram_mem_dim": engram_mem_dim,
            "engram_table_size": engram_table_size,
            "acoustic_feat_dim": acoustic_feat_dim,
            "acoustic_codebook_dim": acoustic_codebook_dim,
            "acoustic_codebook_size": acoustic_codebook_size,
        },
    )

    for peak_lr in learning_rates:
        train_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr / 100, peak_lr, 270))
                + list(np.linspace(peak_lr, peak_lr / 100, 270))
                + list(np.linspace(peak_lr / 100, 1e-8, 60)),
                "batch_size": 180 * 16000,
                "max_seq_length": {"data": 35 * 16000},
                "min_seq_length": {"data": 640},
                "accum_grad_multiple_step": 2,
                "torch_amp_options": {"dtype": "bfloat16"},
                "extern_data": {
                    "data": {"dim": 1},
                    "targets": {"dim": vocab_size + 1, "sparse": True},
                    "language": {"dim": 16, "sparse": True},
                },
            },
            "debug": True,
        }

        recog_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr / 100, peak_lr, 270))
                + list(np.linspace(peak_lr, peak_lr / 100, 270))
                + list(np.linspace(peak_lr / 100, 1e-8, 60)),
                "batch_size": 180 * 16000,
                "accum_grad_multiple_step": 2,
            },
            "debug": True,
        }

        train_args = {
            **copy.deepcopy(train_args_adamw03_accum2_jjlr),
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        train_args["config"]["batch_size"] = batch_size * 16000
        train_args["config"]["accum_grad_multiple_step"] = 1

        recog_args = {
            **copy.deepcopy(recog_args_adamw03_accum2_jjlr),
            "network_module": network_module + "_onnx",
            "net_args": {"model_config_dict": asdict(recog_model_config)},
        }
        recog_args["config"]["batch_size"] = batch_size * 16000
        recog_args["config"]["accum_grad_multiple_step"] = 1
        search_args = {
            "beam_size": 12,
            "returnn_vocab": tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        }

        if len(lang_list) < 16:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=20,
                    separate_heads=separate_heads,
                ),
                cv=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="dev",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=separate_heads,
                ),
                prior=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=separate_heads,
                ),
            )

            train_job, search_job = run_exp(
                prefix_name + f"conformer_new/engram_acoustic_ctc/ctc/{vocab_stem}/{'_'.join(lang_list)}/{batch_size}_{vocab_size}_lr{peak_lr}",
                datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args,
                with_prior=False, evaluate_epoch=eval_epoch, decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2",
                vocab_name=vocab_name, test_set_name=test_set, keep_epochs=keep_epochs,
            )

            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(
                f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates",
                train_job.out_learning_rates,
            )
            tk.register_output(
                f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir",
                train_job.out_model_dir,
            )
        else:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=20,
                ),
                cv=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="dev",
                    lang_list=lang_list,
                    partition_epoch=1,
                ),
                prior=get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=1,
                ),
            )

            train_job, search_job = run_exp(
                prefix_name + f"conformer_new/engram_acoustic_ctc/ctc/{vocab_name.split('.')[0]}/{test_set}/{batch_size}_{vocab_size}_lr{peak_lr}",
                datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args,
                with_prior=False, evaluate_epoch=eval_epoch, recog_mem=recog_mem,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", vocab_name=vocab_name, test_set_name=test_set,
                keep_epochs=keep_epochs,
            )

            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(
                f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates",
                train_job.out_learning_rates,
            )
            tk.register_output(
                f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir",
                train_job.out_model_dir,
            )

    return train_job, search_job


# ===========================================================================
# Engram-v2 Training Function (Encoder-Free, Progressive-Key)
# ===========================================================================

