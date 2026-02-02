import copy
from dataclasses import dataclass
from sisyphus import tk
from typing import Optional
from typing import cast


from dataclasses import asdict
import numpy as np

from i6_experiments.users.jxu.experiments.transducer.voxpopuli.data import get_voxpopuli_data
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datasets import Dataset
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.data.bpe import build_bpe_training_datasets
from i6_experiments.users.nikolov.experiments.voxpopuli.datasets.voxpopuli.get_data import get_voxpopuli_data_per_lang, get_csfleurs_data_per_set, get_fleurs_data_per_set, get_switchlingua_data_per_set, get_data_hdf, get_miami_data_per_set
from i6_experiments.users.nikolov.experiments.voxpopuli.datasets.voxpopuli.lexicon import get_text_lexicon
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.train_util import build_test_dataset, TrainingDatasetSettings



from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config import get_training_config, get_search_config, get_prior_config
#from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.config import get_training_config, get_search_config, get_prior_config


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
    lang_list = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]):
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config_noreturnn import get_training_config, get_search_config, get_prior_config
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.i6modelsV1_VGG4LayerActFrontendV1_v7 import get_model_config
    from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pipeline import training, search, compute_prior

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_rnnt/baseline/"

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
                decoder="rnnt.decoder.experimental_rnnt_decoder", with_prior=False, evaluate_epoch=None):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        #returnn_config.black_formatting = False
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

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
            "network_module": "i6modelsV1_VGG4LayerActFrontendV1_v7",
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        train_args["config"]["batch_size"] = batch_size * 16000
        train_args["config"]["accum_grad_multiple_step"] = 1

        recog_args = {
            **copy.deepcopy(recog_args_adamw03_accum2_jjlr),
            "network_module": "i6modelsV1_VGG4LayerActFrontendV1_v7",
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        recog_args["config"]["batch_size"] = batch_size * 16000
        recog_args["config"]["accum_grad_multiple_step"] = 1
        search_args = {
        "beam_size": 12,
        "returnn_vocab": tk.Path(f"/u/kaloyan.nikolov/experiments/multilang_0325/bpe_{bpe_size}.vocab"),
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
            prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_transparent/bs12/{'_'.join(lang_list)}/{batch_size}_{bpe_size}_lr{peak_lr}",
            datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args, with_prior=False, evaluate_epoch=600)
            
            train_job.rqmt["gpu_mem"] = 48
            tk.register_output(f"output/{'_'.join(lang_list)}/{batch_size}_{bpe_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
            tk.register_output(f"output/{'_'.join(lang_list)}/{batch_size}_{bpe_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)
        
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
            prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v7_JJLR_sub6_start20_lstm512_transparent/bs12/{batch_size}_{bpe_size}_lr{peak_lr}",
            datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args, with_prior=False, evaluate_epoch=600)
            
            train_job.rqmt["gpu_mem"] = 48
            tk.register_output(f"output/{batch_size}_{bpe_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
            tk.register_output(f"output/{batch_size}_{bpe_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)


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
    vocab_name:str=None):
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config_noreturnn import get_training_config, get_search_config, get_prior_config
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.i6modelsV1_VGG4LayerActFrontendV1_v6_onnx_exportable import get_model_config
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

    model_config = get_model_config(vocab_size_without_blank=vocab_size,network_args={})
    recog_model_config = get_model_config(vocab_size_without_blank=vocab_size,network_args={})


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
            "network_module": "ctc.conformer_new.i6modelsV1_VGG4LayerActFrontendV1_v6_onnx_exportable",
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        train_args["config"]["batch_size"] = batch_size * 16000
        train_args["config"]["accum_grad_multiple_step"] = 1

        recog_args = {
            **copy.deepcopy(recog_args_adamw03_accum2_jjlr),
            "network_module": "ctc.conformer_new.i6modelsV1_VGG4LayerActFrontendV1_v6_onnx_exportable",
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
            datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args,lexicon_path=lexicon_path, with_prior=False, evaluate_epoch=eval_epoch, vocab_name=vocab_name, test_set_name=test_set, keep_epochs=keep_epochs)
            
            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
            tk.register_output(f"output/{vocab_name}/{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)
        
        else:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    lexicon_path,
                                    split="train",
                                    partition_epoch=20),
                cv=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    lexicon_path,
                                    split="dev",
                                    partition_epoch=1),
                prior=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    lexicon_path,
                                    split="train",
                                    lang_list=lang_list,
                                    partition_epoch=1),
    )
            train_job, search_job = run_exp(
            prefix_name + f"conformer_new/i6modelsV1_VGG4LayerActFrontendV1_v6/ctc/{vocab_name.split('.')[0]}/{test_set}/{batch_size}_{vocab_size}_lr{peak_lr}",
            datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args, lexicon_path=lexicon_path, with_prior=False, evaluate_epoch=eval_epoch, recog_mem=recog_mem, vocab_name=vocab_name, test_set_name=test_set, keep_epochs=keep_epochs)
            
            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
            tk.register_output(f"output/ctc/{vocab_name}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)





