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
from i6_experiments.users.nikolov.experiments.voxpopuli.datasets.voxpopuli.get_data import get_voxpopuli_data_per_lang, get_csfleurs_data_per_set, get_miami_data_per_set, get_fleurs_data_per_set, get_switchlingua_data_per_set, get_data_per_set
from i6_experiments.users.nikolov.experiments.voxpopuli.datasets.voxpopuli.lexicon import get_text_lexicon
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.train_util import build_test_dataset, TrainingDatasetSettings
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



def conformer_ctc_noreturnn_finetune(
    bpe_size: int, 
    batch_size: int = 120,
    gpu_mem: int = 48,
    recog_mem: int = 10,
    vocab_size: Optional[int] = 10396,
    learning_rates = [8e-5, 1e-4, 2e-4],
    lang_list = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"],
    eval_epoch: int = 600,
    finetune_epochs: int = 20,
    vocab_name:str=None):
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config_noreturnn import get_training_config, get_search_config, get_prior_config
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.i6modelsV1_VGG4LayerActFrontendV1_v6_onnx_exportable import get_model_config
    from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pipeline import training
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pipeline_flashlight import search, prepare_asr_model
    #from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.data.common import TrainingDatasets

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_ctc/finetune/"

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
        vocab_name = f"bpe_{bpe_size}.vocab"
            
    vocab_size_without_blank = vocab_size

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev", "test"]
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
    def run_exp(ft_name, datasets, train_args, search_args=None, recog_args=None, extra_config=None, num_epochs=600,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", with_prior=False, finetune_epochs=0, evaluate_epoch=None, recog_mem:int = 10, vocab_name:str=None):
        evaluate_epoch = 0
        num_epochs = 0 # TODO: remove if it works
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args, extra_config=extra_config)
        #returnn_config.black_formatting = False
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs + finetune_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs

        returnn_search_config = get_search_config(**recog_args, decoder_args=search_args,
                                                  decoder=decoder)
        # _, _, search_jobs = search(ft_name + "/default_%i" % evaluate_epoch, returnn_search_config,
        #                            train_job.out_checkpoints[evaluate_epoch], test_dataset_tuples, RETURNN_EXE,
        #                            MINI_RETURNN_ROOT, use_gpu=search_args.get("use_gpu", False))
        # return train_job, search_jobs
        from ..ctc_rnnt_standalone_2024.pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig

        default_decoder_config = DecoderConfig(
        lexicon= get_text_lexicon(ft_name + "/text_lex", bpe_size),
        returnn_vocab=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        #arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        )
        asr_model = prepare_asr_model(
        ft_name, train_job, train_args, with_prior=True, datasets=datasets, get_specific_checkpoint=evaluate_epoch + finetune_epochs
        )

        search_jobs, wers = search(
                           prefix_name=ft_name + "/default_%i" % (evaluate_epoch + finetune_epochs), 
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
        
        extra_config = {
            "preload_from_files":{
                "codebook": {
                    "filename": f"/u/kaloyan.nikolov/experiments/multilang_0325/output/output/ctc/{batch_size}_{bpe_size}_lr0.0002/out_model_dir/epoch.{eval_epoch}.pt",  # your checkpoint file, mandatory
                    "init_for_train": True,
                    "checkpoint_key": "model",
                    "ignore_missing": True,  # if the checkpoint only partly covers your model, default is False
                    "ignore_params_prefixes": ["final_linear_list"],
                }
            }
        }

        default_search_args = {
        "lexicon": tk.Path(f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}"),  # TODO: cleanup
        "returnn_vocab": tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        "beam_size": 1024,
        "beam_threshold": 14,
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
            "net_args": {"model_config_dict": asdict(model_config)},
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
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="train",
                                    lang_list=lang_list,
                                    partition_epoch=20),
                cv=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="dev",
                                    lang_list=lang_list,
                                    partition_epoch=1),
                prior=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="train",
                                    lang_list=lang_list,
                                    partition_epoch=1),
    )

            train_job, search_job = run_exp(
            prefix_name + f"conformer_new/i6modelsV1_VGG4LayerActFrontendV1_v6/ctc/{'_'.join(lang_list)}/{batch_size}_{bpe_size}_lr{peak_lr}",
            datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args, extra_config=extra_config, with_prior=False, finetune_epochs=finetune_epochs, evaluate_epoch=eval_epoch, recog_mem=recog_mem, vocab_name=vocab_name)
            
            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(f"output/{'_'.join(lang_list)}/ctc/finetune/{batch_size}_{bpe_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
            tk.register_output(f"output/{'_'.join(lang_list)}/ctc/finetune/{batch_size}_{bpe_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)
        
        else:
            train_data = CTCTrainingDatasets(
                train=get_voxpopuli_data("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="train",
                                    partition_epoch=20),
                cv=get_voxpopuli_data("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="dev",
                                    partition_epoch=1),
                prior=get_voxpopuli_data_per_lang("/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                                    f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{bpe_size}",
                                    split="train",
                                    lang_list=lang_list,
                                    partition_epoch=1),
    )

            train_job, search_job = run_exp(
            prefix_name + f"conformer_new/i6modelsV1_VGG4LayerActFrontendV1_v6/ctc/{batch_size}_{bpe_size}_lr{peak_lr}",
            datasets=train_data, train_args=train_args, search_args=search_args, recog_args=recog_args,extra_config=extra_config, with_prior=False, finetune_epochs=finetune_epochs, evaluate_epoch=eval_epoch, recog_mem=recog_mem, vocab_name=vocab_name)
            
            train_job.rqmt["gpu_mem"] = gpu_mem
            tk.register_output(f"output/ctc/finetune/{batch_size}_{bpe_size}_lr{peak_lr}/learning_rates", train_job.out_learning_rates)
            tk.register_output(f"output/ctc/finetune/{batch_size}_{bpe_size}_lr{peak_lr}/out_model_dir", train_job.out_model_dir)


def conformer_ctc_noreturnn_finetune_corpus(
    bpe_size: int,
    batch_size: int = 120,
    gpu_mem: int = 11,
    recog_mem: int = 20,
    vocab_size: int = 4989,
    learning_rates = [1e-4, 5e-5, 1e-5],
    lexicon_path: str = None,
    finetune_set: str = "switchlingua-tts",
    test_set: str = "voxpopuli",
    finetune_epochs: int = 100,
    eval_epoch: int = 600,
    vocab_name: str = None,
    checkpoint_path: str = None,
    ignore_params_prefixes: Optional[list] = None,
):
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config_noreturnn import get_training_config, get_search_config
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.i6modelsV1_VGG4LayerActFrontendV1_v6_onnx_exportable import get_model_config
    from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pipeline import training
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pipeline_flashlight import search, prepare_asr_model

    if lexicon_path is None:
        lexicon_path = f"/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr_lexicon"
    if not vocab_name:
        vocab_name = f"bpe_{vocab_size}.vocab"
    if ignore_params_prefixes is None:
        ignore_params_prefixes = ["final_linear"]

    prefix_name = "experiments/finetune/corpus/"

    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    # Build test dataset tuples
    test_dataset_tuples = {}
    if test_set == "voxpopuli":
        lang_list = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"]
        for lang in lang_list:
            test_dataset_tuples[lang] = (
                get_voxpopuli_data_per_lang(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    lexicon_path,
                    split="test",
                    lang_list=[lang],
                    partition_epoch=1,
                ),
                tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz"),
            )
    elif test_set == "csfleurs":
        datasets = ["mms", "read", "xtts"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = (
                get_csfleurs_data_per_set(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/csfleurs_asr",
                    split="test",
                    set_list=[dataset],
                    partition_epoch=1,
                ),
                tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/csfleurs/corpus/{dataset}.test.corpus.xml.gz"),
            )
    elif test_set == "miami":
        datasets = ["full", "spa", "eng"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = (
                get_miami_data_per_set(
                    "/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr",
                    split="test",
                    set_list=[dataset],
                    partition_epoch=1,
                ),
                tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz"),
            )
    elif test_set == "switchlingua":
        test_dataset_tuples["dev"] = (
            get_switchlingua_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
                split="dev",
                partition_epoch=1,
            ),
            tk.Path("/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz"),
        )

    def run_exp(ft_name, datasets, train_args, search_args=None, recog_args=None, extra_config=None, num_epochs=600,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", with_prior=False, finetune_epochs=0, evaluate_epoch=None, recog_mem:int = 10, vocab_name:str=None):
        evaluate_epoch = 0
        num_epochs = 0
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args, extra_config=extra_config)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs + finetune_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs

        returnn_search_config = get_search_config(**recog_args, decoder_args=search_args,
                                                  decoder=decoder)
        from ..ctc_rnnt_standalone_2024.pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig

        default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(ft_name + "/text_lex", bpe_size, add_prefix=False, lexicon_path=lexicon_path),
        returnn_vocab=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        beam_size=1024,
        beam_size_token=12,
        beam_threshold=14,
        )
        asr_model = prepare_asr_model(
        ft_name, train_job, train_args, with_prior=True, datasets=datasets, get_specific_checkpoint=evaluate_epoch + finetune_epochs
        )

        search_jobs, wers = search(
                           prefix_name=ft_name + f"/{finetune_set}/test_{test_set}/default_{evaluate_epoch}", 
                           forward_config={},
                           asr_model=asr_model,
                           decoder_module=decoder, 
                           decoder_args={"config": asdict(default_decoder_config)}, 
                           test_dataset_tuples=test_dataset_tuples, 
                           returnn_exe=RETURNN_EXE,
                           returnn_root=MINI_RETURNN_ROOT, use_gpu=search_args.get("use_gpu", False),
                           mem_rqmt=recog_mem)

        return train_job, search_jobs

    model_config = get_model_config(vocab_size_without_blank=vocab_size, network_args={})

    for peak_lr in learning_rates:
        train_args_adamw03_accum2_jjlr = {
            "config": {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(peak_lr/100, peak_lr, 270)) + list(
        np.linspace(peak_lr, peak_lr/100, 270)) + list(np.linspace(peak_lr/100, 1e-8, 60)),
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
                "batch_size": 180 * 16000,
                "accum_grad_multiple_step": 2,
            },
            "debug": True,
        }

        if checkpoint_path is None:
            checkpoint_path = (
                f"/u/kaloyan.nikolov/experiments/multilang_0325/output/output/ctc/"
                f"bpe_{vocab_size}.vocab/{batch_size}_{vocab_size}_lr{peak_lr}/"
                f"out_model_dir/epoch.{eval_epoch}.pt"
            )

        extra_config = {
            "preload_from_files": {
                "codebook": {
                    "filename": checkpoint_path,
                    "init_for_train": True,
                    "checkpoint_key": "model",
                    "ignore_missing": True,
                    "ignore_params_prefixes": ignore_params_prefixes,
                }
            }
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
            "net_args": {"model_config_dict": asdict(model_config)},
        }
        recog_args["config"]["batch_size"] = batch_size * 16000
        recog_args["config"]["accum_grad_multiple_step"] = 1
        search_args = {
        "beam_size": 12,
        "returnn_vocab": tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        }

        train_data = CTCTrainingDatasets(
            train=get_voxpopuli_data_per_lang(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
                lexicon_path,
                split="train",
                partition_epoch=20,
            ),
            cv=get_voxpopuli_data_per_lang(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
                lexicon_path,
                split="dev",
                partition_epoch=1,
            ),
            prior=get_voxpopuli_data_per_lang(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
                lexicon_path,
                split="train",
                partition_epoch=1,
            ),
        )

        exp_name = (
            prefix_name
            + f"conformer_new/i6modelsV1_VGG4LayerActFrontendV1_v6/ctc/"
            f"bpe_{vocab_size}/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}"
        )

        train_job, search_job = run_exp(
            exp_name,
            datasets=train_data,
            train_args=train_args,
            search_args=search_args,
            recog_args=recog_args,
            extra_config=extra_config,
            with_prior=False,
            finetune_epochs=finetune_epochs,
            evaluate_epoch=eval_epoch,
            recog_mem=recog_mem,
            vocab_name=vocab_name,
        )

        train_job.rqmt["gpu_mem"] = gpu_mem
        tk.register_output(
            f"output/ctc/finetune/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates",
            train_job.out_learning_rates,
        )
        tk.register_output(
            f"output/ctc/finetune/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir",
            train_job.out_model_dir,
        )

    return train_job, search_job


def lid_aware_sc_ctc_finetune(
    bpe_size: int | dict[str, int],
    lexicon_path: str = None,
    batch_size: int = 120,
    gpu_mem: int = 48,
    recog_mem: int = 10,
    vocab_size: int = 10396,
    learning_rates = [8e-5, 1e-4, 2e-4],
    lang_list = ["cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"],
    finetune_set: str = "switchlingua",
    test_set: str = "voxpopuli",
    eval_epoch: int = 600,
    finetune_epochs: int = 20,
    vocab_name: str = None,
    specaug_start_epoch: int = 1,
    checkpoint_path: str = None,
    checkpoint_lr: float = 2e-4,
    ignore_params_prefixes: Optional[list[str]] = None,
    language_data_dir: str = None,
    add_prefix: bool = False,
):
    from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.configs.config_noreturnn import get_training_config, get_search_config, get_prior_config
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.jxu_language_aware_sc_ctc_onnx import get_model_config
    from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pipeline import training
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pipeline_flashlight import search, prepare_asr_model

    base_bpe_size = bpe_size["base"] if isinstance(bpe_size, dict) else bpe_size
    if lexicon_path is None:
        lexicon_path = f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{base_bpe_size}"
    if not vocab_name:
        vocab_name = f"bpe_{vocab_size}.vocab"
    if ignore_params_prefixes is None:
        ignore_params_prefixes = ["final_linear", "sc_softmax_linear", "sc_linear"]

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_ctc/finetune/"
    network_module = "ctc.conformer_new.jxu_language_aware_sc_ctc"
    vocab_stem = vocab_name.split(".")[0] + ("_prefixed" if vocab_name.split(".")[-1] == "prefixed" else "")

    test_dataset_tuples = {}
    if test_set == "voxpopuli":
        for lang in lang_list:
            test_dataset_tuples[lang] = get_voxpopuli_data_per_lang(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                lexicon_path,
                split="test",
                lang_list=[lang],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")
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
    elif test_set == "miami":
        datasets = ["full", "spa", "eng"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_miami_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr",
                split="test",
                set_list=[dataset],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz")
    elif test_set == "switchlingua":
        test_dataset_tuples["dev"] = get_switchlingua_data_per_set(
            "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
            split="dev",
            partition_epoch=1,
        ), tk.Path("/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")
    else:
        raise ValueError(f"Unsupported test_set for lid_aware_sc_ctc_finetune: {test_set}")

    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    if checkpoint_path is None:
        if len(lang_list) < 16:
            checkpoint_path = (
                f"/u/kaloyan.nikolov/experiments/multilang_0325/output/output/{vocab_name}/"
                f"{'_'.join(lang_list)}/ctc/{batch_size}_{vocab_size}_lr{checkpoint_lr}/"
                f"out_model_dir/epoch.{eval_epoch}.pt"
            )
        else:
            checkpoint_path = (
                f"/u/kaloyan.nikolov/experiments/multilang_0325/output/output/ctc/{vocab_name}/"
                f"{batch_size}_{vocab_size}_lr{checkpoint_lr}/out_model_dir/epoch.{eval_epoch}.pt"
            )

    def build_single_corpus_dataset(audio_files, target_files, language_files, partition_epoch):
        raw_audio_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": audio_files,
            "partition_epoch": partition_epoch,
            "seq_ordering": "laplace:.1000",
        }
        target_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": target_files,
        }
        language_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": language_files,
        }
        return MetaDataset(
            data_map={"data": ("features", "data"), "targets": ("targets", "data"), "language": ("language", "data")},
            datasets={
                "features": raw_audio_dataset_dict,
                "targets": target_dataset_dict,
                "language": language_dataset_dict,
            },
            seq_order_control_dataset="features",
        )

    def get_finetune_dataset_with_language(split: str, partition_epoch: int):
        if finetune_set == "voxpopuli":
            return get_voxpopuli_data_per_lang(
                split=split,
                lang_list=lang_list,
                partition_epoch=partition_epoch,
                separate_heads=True,
                audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                target_base_dir=lexicon_path,
            )
        if language_data_dir is None:
            raise ValueError(
                "lid_aware_sc_ctc_finetune requires language_data_dir for non-VoxPopuli finetune sets "
                f"(got finetune_set={finetune_set!r})."
            )
        if finetune_set == "switchlingua":
            return build_single_corpus_dataset(
                audio_files=[f"/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr/{split}.hdf"],
                target_files=[f"/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_lex_hdf/{split}.hdf"],
                language_files=[f"{language_data_dir}/{split}.hdf"],
                partition_epoch=partition_epoch,
            )
        raise NotImplementedError(
            "lid_aware_sc_ctc_finetune currently supports finetune_set='voxpopuli' out of the box "
            "and finetune_set='switchlingua' when language_data_dir is provided."
        )

    def run_exp(ft_name, datasets, train_args, search_args=None, recog_args=None, extra_config=None, num_epochs=600, lexicon_path: str = None,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", with_prior=False, finetune_epochs=0, evaluate_epoch=None, recog_mem:int = 10, vocab_name:str=None):
        evaluate_epoch = 0
        num_epochs = 0
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args, extra_config=extra_config)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs + finetune_epochs)

        if not evaluate_epoch:
            evaluate_epoch = num_epochs

        returnn_search_config = get_search_config(**recog_args, decoder_args=search_args,
                                                  decoder=decoder)
        from ..ctc_rnnt_standalone_2024.pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig

        default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(ft_name + "/text_lex", bpe_size, add_prefix, lexicon_path),
        returnn_vocab=tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/vocab/{vocab_name}"),
        beam_size=1024,
        beam_size_token=12,
        beam_threshold=14,
        )
        asr_model = prepare_asr_model(
        ft_name, train_job, train_args, with_prior=True, datasets=datasets, get_specific_checkpoint=evaluate_epoch + finetune_epochs
        )

        search_jobs, wers = search(
                           prefix_name=ft_name + "/" + finetune_set + f"/test_{test_set}" + "/default_%i" % (evaluate_epoch + finetune_epochs),
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
        network_args={"specaug_start_epoch": specaug_start_epoch, "lid_sc_layer": 3, "sc_layer": [6, 9]},
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

        extra_config = {
            "preload_from_files": {
                "codebook": {
                    "filename": checkpoint_path,
                    "init_for_train": True,
                    "checkpoint_key": "model",
                    "ignore_missing": True,
                    "ignore_params_prefixes": ignore_params_prefixes,
                }
            }
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
            "net_args": {"model_config_dict": asdict(model_config)},
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
                    audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    target_base_dir=lexicon_path,
                ),
                cv=get_voxpopuli_data_per_lang(
                    split="dev",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=True,
                    audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    target_base_dir=lexicon_path,
                ),
                prior=get_voxpopuli_data_per_lang(
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=True,
                    audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    target_base_dir=lexicon_path,
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
                    audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    target_base_dir=lexicon_path,
                ),
                cv=get_voxpopuli_data_per_lang(
                    split="dev",
                    partition_epoch=1,
                    separate_heads=True,
                    audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    target_base_dir=lexicon_path,
                ),
                prior=get_voxpopuli_data_per_lang(
                    split="train",
                    lang_list=lang_list,
                    partition_epoch=1,
                    separate_heads=True,
                    audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                    target_base_dir=lexicon_path,
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
def engram_v2_ctc_finetune(
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
    finetune_epochs: int = 20,
    vocab_name: str = None,
    specaug_start_epoch: int = 1,
    checkpoint_path: str = None,
    checkpoint_lr: float = 2e-4,
    ignore_params_prefixes: Optional[list[str]] = None,
    language_data_dir: str = None,
    finetune_set: str = "switchlingua",
    test_set: str = "voxpopuli",
    # Engram-v2 specific args
    engram_table_size: int = 2**12,
    engram_num_heads: int = 8,
    engram_ngram_orders: list = [[2, 3], [2, 3], [3, 4]],
    engram_mem_dim: int = 1280,
    engram_layers: list = [2, 6, 10],
    acoustic_num_bins: int = 32,
    sample_rate: int = 16000,
    frame_ms: int = 40,
    conformer_size: int = 512,
    num_layers: int = 12,
):
    """
    Finetune an Engram-v2 model from a pre-trained checkpoint.

    Typically starts from a standard CTC or Engram-v1 checkpoint and
    initializes the Engram-v2 parameters (audio_embedder, engrams) fresh.

    Args:
        checkpoint_path: Path to pre-trained .pt checkpoint
        ignore_params_prefixes: Parameter prefixes to ignore when loading
            (defaults to Engram-v2 specific modules so they initialize fresh)
        engram_*: Engram hyperparameters
        acoustic_*: Acoustic quantizer hyperparameters
    """
    from i6_experiments.common.setups.returnn.datasets.base import MetaDataset

    base_bpe_size = bpe_size["base"] if isinstance(bpe_size, dict) else bpe_size
    if lexicon_path is None:
        lexicon_path = f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{base_bpe_size}"
    if not vocab_name:
        vocab_name = f"bpe_{vocab_size}.vocab"
    if ignore_params_prefixes is None:
        # Ignore Engram-v2 specific modules so they initialize fresh
        ignore_params_prefixes = ["engrams", "audio_embedder", "amplitude_quantizer",
                                  "intermediate_ctc", "final_linear"]

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_ctc/engram_v2_finetune/"
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
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")
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
    elif test_set == "miami":
        datasets = ["full", "spa", "eng"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_miami_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr",
                split="test",
                set_list=[dataset],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz")
    elif test_set == "switchlingua":
        test_dataset_tuples["dev"] = get_switchlingua_data_per_set(
            "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
            split="dev",
            partition_epoch=1,
        ), tk.Path("/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")
    else:
        raise ValueError(f"Unsupported test_set: {test_set}")

    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    def build_single_corpus_dataset(audio_files, target_files, language_files, partition_epoch):
        raw_audio_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": audio_files,
            "partition_epoch": partition_epoch,
            "seq_ordering": "laplace:.1000",
        }
        target_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": target_files,
        }
        language_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": language_files,
        }
        return MetaDataset(
            data_map={"data": ("features", "data"), "targets": ("targets", "data"), "language": ("language", "data")},
            datasets={
                "features": raw_audio_dataset_dict,
                "targets": target_dataset_dict,
                "language": language_dataset_dict,
            },
            seq_order_control_dataset="features",
        )

    def get_finetune_dataset_with_language(split: str, partition_epoch: int):
        if finetune_set == "voxpopuli":
            return get_voxpopuli_data_per_lang(
                split=split,
                lang_list=lang_list,
                partition_epoch=partition_epoch,
                separate_heads=True,
                audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                target_base_dir=lexicon_path,
            )
        if language_data_dir is None:
            raise ValueError(
                "engram_v2_ctc_finetune requires language_data_dir for non-VoxPopuli finetune sets "
                f"(got finetune_set={finetune_set!r})."
            )
        if finetune_set == "switchlingua":
            return build_single_corpus_dataset(
                audio_files=[f"/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr/{split}.hdf"],
                target_files=[f"/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_lex_hdf/{split}.hdf"],
                language_files=[f"{language_data_dir}/{split}.hdf"],
                partition_epoch=partition_epoch,
            )
        raise NotImplementedError(
            f"engram_v2_ctc_finetune does not support finetune_set={finetune_set!r} yet."
        )

    def run_exp(ft_name, datasets, train_args, search_args=None, recog_args=None, extra_config=None, num_epochs=600,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", with_prior=False, finetune_epochs=0, evaluate_epoch=None,
                recog_mem: int = 10, vocab_name: str = None):
        evaluate_epoch = 0
        num_epochs = 0
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args, extra_config=extra_config)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs + finetune_epochs)

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
            ft_name, train_job, train_args, with_prior=True, datasets=datasets, get_specific_checkpoint=evaluate_epoch + finetune_epochs
        )

        search_jobs, wers = search(
            prefix_name=ft_name + "/" + finetune_set + f"/test_{test_set}" + "/default_%i" % (evaluate_epoch + finetune_epochs),
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

    if checkpoint_path is None:
        checkpoint_path = (
            f"/u/kaloyan.nikolov/experiments/multilang_0325/output/output/ctc/"
            f"bpe_{vocab_size}.vocab/{batch_size}_{vocab_size}_lr{checkpoint_lr}/"
            f"out_model_dir/epoch.{eval_epoch}.pt"
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

        extra_config = {
            "preload_from_files": {
                "codebook": {
                    "filename": checkpoint_path,
                    "init_for_train": True,
                    "checkpoint_key": "model",
                    "ignore_missing": True,
                    "ignore_params_prefixes": ignore_params_prefixes,
                }
            }
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

        train_data = CTCTrainingDatasets(
            train=get_finetune_dataset_with_language(split="train", partition_epoch=20),
            cv=get_finetune_dataset_with_language(split="dev", partition_epoch=1),
            prior=get_finetune_dataset_with_language(split="train", partition_epoch=1),
        )

        exp_name = (
            prefix_name
            + f"conformer_new/engram_v2_ctc/ctc/{vocab_stem}/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}"
        )

        train_job, search_jobs = run_exp(
            exp_name,
            datasets=train_data,
            train_args=train_args,
            search_args=search_args,
            recog_args=recog_args,
            extra_config=extra_config,
            with_prior=False,
            finetune_epochs=finetune_epochs,
            evaluate_epoch=eval_epoch,
            recog_mem=recog_mem,
            vocab_name=vocab_name,
        )

        train_job.rqmt["gpu_mem"] = gpu_mem
        tk.register_output(
            f"output/ctc/engram_v2_finetune/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates",
            train_job.out_learning_rates,
        )
        tk.register_output(
            f"output/ctc/engram_v2_finetune/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir",
            train_job.out_model_dir,
        )

    return train_job, search_jobs
def gpt2_ngram_ctc_finetune(
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
    finetune_epochs: int = 20,
    vocab_name: str = None,
    checkpoint_path: str = None,
    checkpoint_lr: float = 2e-4,
    ignore_params_prefixes: Optional[list[str]] = None,
    language_data_dir: str = None,
    finetune_set: str = "switchlingua",
    test_set: str = "voxpopuli",
    # GPT-2 config
    gpt2_n_layer: int = 12,
    gpt2_n_head: int = 12,
    gpt2_n_embd: int = 768,
    gpt2_ff_dim: int = 3072,
    # Engram config
    engram_layers: list = [2, 6],
    engram_lid_layer: int = 6,
    engram_audio_bins: int = 32,
    engram_lid_classes: int = 17,
    engram_ngram_orders: list = [2, 3],
    engram_num_heads: int = 8,
    engram_mem_dim: int = 1280,
    engram_table_size: int = 2**12,
    engram_dropout: float = 0.0,
    # Audio config
    sample_rate: int = 16000,
    frame_ms: int = 40,
):
    """
    Finetune a GPT-2 Ngram CTC model from a pre-trained checkpoint.

    Typically started from a standard CTC Conformer checkpoint. The GPT-2
    blocks and Engram parameters are initialized randomly (ignored via
    ignore_params_prefixes), while the audio projection and CTC head can
    be carried over from the checkpoint.

    Args:
        checkpoint_path: Path to pre-trained .pt checkpoint
        ignore_params_prefixes: Parameter prefixes to ignore when loading
            (defaults to GPT-2-specific modules so they initialize fresh)
        finetune_set: Dataset to finetune on (switchlingua, switchlingua-tts, csfleurs, miami)
        test_set: Dataset to evaluate on
        finetune_epochs: Number of finetuning epochs
        gpt2_*: GPT-2 architecture hyperparameters
        engram_*: Engram hyperparameters
    """
    from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
    from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.gpt2_ngram_ctc_onnx import (
        get_model_config as get_recog_config,
    )

    base_bpe_size = bpe_size["base"] if isinstance(bpe_size, dict) else bpe_size
    if lexicon_path is None:
        lexicon_path = f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{base_bpe_size}"
    if not vocab_name:
        vocab_name = f"bpe_{vocab_size}.vocab"
    if ignore_params_prefixes is None:
        # Ignore GPT-2-specific modules so they initialize fresh
        ignore_params_prefixes = ["blocks", "audio_proj", "position_embed", "ln_f", "ctc_head", "lid_head", "lid_sc_linear", "engrams"]

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_ctc/gpt2_ngram_finetune/"
    network_module = "ctc.conformer_new.gpt2_ngram_ctc"
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
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")
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
    elif test_set == "miami":
        datasets = ["full", "spa", "eng"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_miami_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr",
                split="test",
                set_list=[dataset],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz")
    elif test_set == "switchlingua":
        test_dataset_tuples["dev"] = get_switchlingua_data_per_set(
            "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
            split="dev",
            partition_epoch=1,
        ), tk.Path("/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")
    else:
        raise ValueError(f"Unsupported test_set: {test_set}")

    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    def build_single_corpus_dataset(audio_files, target_files, partition_epoch):
        """Build a dataset from explicit HDF file lists (no language data)."""
        raw_audio_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": audio_files,
            "partition_epoch": partition_epoch,
            "seq_ordering": "laplace:.1000",
        }
        target_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": target_files,
        }
        return MetaDataset(
            data_map={"data": ("features", "data"), "targets": ("targets", "data")},
            datasets={
                "features": raw_audio_dataset_dict,
                "targets": target_dataset_dict,
            },
            seq_order_control_dataset="features",
        )

    def get_finetune_dataset(split: str, partition_epoch: int):
        """Get finetune dataset for various corpora."""
        if finetune_set == "voxpopuli":
            return get_voxpopuli_data_per_lang(
                split=split,
                lang_list=lang_list,
                partition_epoch=partition_epoch,
                separate_heads=False,
                audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                target_base_dir=lexicon_path,
            )
        if finetune_set == "switchlingua":
            return get_switchlingua_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
                target_base_dir=lexicon_path,
                split=split,
                partition_epoch=partition_epoch,
            )
        if finetune_set == "switchlingua-tts":
            return get_switchlingua_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr_tts",
                target_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_tts_asr_lexicon",
                split=split,
                partition_epoch=partition_epoch,
            )
        if finetune_set == "csfleurs":
            datasets = ["mms", "read", "xtts"]
            return build_single_corpus_dataset(
                audio_files=[f"/u/kaloyan.nikolov/experiments/multilang_0325/output/csfleurs_asr/{d}/{split}.hdf" for d in datasets],
                target_files=[f"/u/kaloyan.nikolov/experiments/multilang_0325/output/csfleurs_lex_hdf/{d}/{split}.hdf" for d in datasets],
                partition_epoch=partition_epoch,
            )
        if finetune_set == "miami":
            datasets = ["full", "spa", "eng"]
            return build_single_corpus_dataset(
                audio_files=[f"/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr/{d}/{split}.hdf" for d in datasets],
                target_files=[f"/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_lex_hdf/{d}/{split}.hdf" for d in datasets],
                partition_epoch=partition_epoch,
            )
        raise NotImplementedError(f"gpt2_ngram_ctc_finetune does not support finetune_set={finetune_set!r}")

    def run_exp(ft_name, datasets, train_args, search_args=None, recog_args=None, extra_config=None,
                num_epochs=600, decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", with_prior=False,
                finetune_epochs=0, evaluate_epoch=None, recog_mem: int = 10, vocab_name: str = None):
        evaluate_epoch = 0
        num_epochs = 0
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args, extra_config=extra_config)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs + finetune_epochs)

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
            get_specific_checkpoint=evaluate_epoch + finetune_epochs,
        )

        search_jobs, wers = search(
            prefix_name=ft_name + "/" + finetune_set + f"/test_{test_set}" + "/default_%i" % (evaluate_epoch + finetune_epochs),
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

    if checkpoint_path is None:
        checkpoint_path = (
            f"/u/kaloyan.nikolov/experiments/multilang_0325/output/output/ctc/"
            f"bpe_{vocab_size}.vocab/{batch_size}_{vocab_size}_lr{checkpoint_lr}/"
            f"out_model_dir/epoch.{eval_epoch}.pt"
        )

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

        extra_config = {
            "preload_from_files": {
                "codebook": {
                    "filename": checkpoint_path,
                    "init_for_train": True,
                    "checkpoint_key": "model",
                    "ignore_missing": True,
                    "ignore_params_prefixes": ignore_params_prefixes,
                }
            }
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

        train_data = CTCTrainingDatasets(
            train=get_finetune_dataset(split="train", partition_epoch=20),
            cv=get_finetune_dataset(split="dev", partition_epoch=1),
            prior=get_finetune_dataset(split="train", partition_epoch=1),
        )

        exp_name = (
            prefix_name
            + f"gpt2_ngram_ctc/ctc/{vocab_stem}/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}"
        )

        train_job, search_jobs = run_exp(
            exp_name,
            datasets=train_data,
            train_args=train_args,
            search_args=search_args,
            recog_args=recog_args,
            extra_config=extra_config,
            with_prior=False,
            finetune_epochs=finetune_epochs,
            evaluate_epoch=eval_epoch,
            recog_mem=recog_mem,
            vocab_name=vocab_name,
        )

        train_job.rqmt["gpu_mem"] = gpu_mem
        tk.register_output(
            f"output/ctc/gpt2_ngram_finetune/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates",
            train_job.out_learning_rates,
        )
        tk.register_output(
            f"output/ctc/gpt2_ngram_finetune/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir",
            train_job.out_model_dir,
        )

    return train_job, search_jobs
def engram_acoustic_ctc_finetune(
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
    finetune_epochs: int = 20,
    vocab_name: str = None,
    specaug_start_epoch: int = 1,
    checkpoint_path: str = None,
    checkpoint_lr: float = 2e-4,
    ignore_params_prefixes: Optional[list[str]] = None,
    language_data_dir: str = None,
    finetune_set: str = "switchlingua",
    test_set: str = "voxpopuli",
    # Engram-specific args
    engram_table_size: int = 2**12,
    engram_num_heads: int = 8,
    engram_ngram_orders: list = [2, 3],
    engram_mem_dim: int = 1280,
    acoustic_codebook_size: int = 256,
    acoustic_feat_dim: int = 80,
    acoustic_codebook_dim: int = 64,
):
    """
    Finetune an Engram-augmented CTC Conformer from a pre-trained checkpoint.

    Typically starts from a standard CTC or LID+SC CTC checkpoint and
    initializes the Engram parameters randomly.

    Args:
        checkpoint_path: Path to pre-trained .pt checkpoint
        ignore_params_prefixes: Parameter prefixes to ignore when loading
            (defaults to Engram-specific modules so they initialize fresh)
        engram_*: Engram hyperparameters
        acoustic_*: Acoustic quantizer hyperparameters
    """
    from i6_experiments.common.setups.returnn.datasets.base import MetaDataset

    base_bpe_size = bpe_size["base"] if isinstance(bpe_size, dict) else bpe_size
    if lexicon_path is None:
        lexicon_path = f"/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr_lexicon_{base_bpe_size}"
    if not vocab_name:
        vocab_name = f"bpe_{vocab_size}.vocab"
    if ignore_params_prefixes is None:
        # Ignore Engram-specific modules so they initialize fresh
        ignore_params_prefixes = ["engrams", "acoustic_quantizer", "final_linear"]

    prefix_name = "experiments/rescale/tedliumv2/torchaudio_bpe_ctc/engram_finetune/"
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
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/corpus/asr_{lang}.test.corpus.xml.gz")
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
    elif test_set == "miami":
        datasets = ["full", "spa", "eng"]
        for dataset in datasets:
            test_dataset_tuples[dataset] = get_miami_data_per_set(
                "/u/kaloyan.nikolov/experiments/multilang_0325/output/miami_asr",
                split="test",
                set_list=[dataset],
                partition_epoch=1,
            ), tk.Path(f"/work/asr3/jxu/hiwis/nikolov/multilang_0325/miami/text/Miami/tests/miami.{dataset}.corpus.xml.gz")
    elif test_set == "switchlingua":
        test_dataset_tuples["dev"] = get_switchlingua_data_per_set(
            "/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr",
            split="dev",
            partition_epoch=1,
        ), tk.Path("/work/asr3/jxu/hiwis/nikolov/multilang_0325/switchlingua/corpus/dev.corpus.xml.gz")
    else:
        raise ValueError(f"Unsupported test_set: {test_set}")

    RETURNN_EXE = tk.Path("/usr/bin/python3")
    MINI_RETURNN_ROOT = tk.Path("/u/kaloyan.nikolov/src/NoReturnn", hash_overwrite="TEDLIUM2_DEFAULT_RETURNN_ROOT")

    # Build finetune dataset
    def build_single_corpus_dataset(audio_files, target_files, language_files, partition_epoch):
        raw_audio_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": audio_files,
            "partition_epoch": partition_epoch,
            "seq_ordering": "laplace:.1000",
        }
        target_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": target_files,
        }
        language_dataset_dict = {
            "class": "HDFDataset",
            "use_cache_manager": True,
            "files": language_files,
        }
        return MetaDataset(
            data_map={"data": ("features", "data"), "targets": ("targets", "data"), "language": ("language", "data")},
            datasets={
                "features": raw_audio_dataset_dict,
                "targets": target_dataset_dict,
                "language": language_dataset_dict,
            },
            seq_order_control_dataset="features",
        )

    def get_finetune_dataset_with_language(split: str, partition_epoch: int):
        if finetune_set == "voxpopuli":
            return get_voxpopuli_data_per_lang(
                split=split,
                lang_list=lang_list,
                partition_epoch=partition_epoch,
                separate_heads=True,
                audio_base_dir="/u/kaloyan.nikolov/experiments/multilang_0325/output/voxpopuli_asr",
                target_base_dir=lexicon_path,
            )
        if language_data_dir is None:
            raise ValueError(
                "engram_acoustic_ctc_finetune requires language_data_dir for non-VoxPopuli finetune sets "
                f"(got finetune_set={finetune_set!r})."
            )
        if finetune_set == "switchlingua":
            return build_single_corpus_dataset(
                audio_files=[f"/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_asr/{split}.hdf"],
                target_files=[f"/u/kaloyan.nikolov/experiments/multilang_0325/output/switchlingua_lex_hdf/{split}.hdf"],
                language_files=[f"{language_data_dir}/{split}.hdf"],
                partition_epoch=partition_epoch,
            )
        raise NotImplementedError(
            f"engram_acoustic_ctc_finetune does not support finetune_set={finetune_set!r} yet."
        )

    def run_exp(ft_name, datasets, train_args, search_args=None, recog_args=None, extra_config=None, num_epochs=600,
                decoder="ctc.decoder.flashlight_ctc_v1_onnx_v2", with_prior=False, finetune_epochs=0, evaluate_epoch=None,
                recog_mem: int = 10, vocab_name: str = None):
        evaluate_epoch = 0
        num_epochs = 0
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args, extra_config=extra_config)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs + finetune_epochs)

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
            ft_name, train_job, train_args, with_prior=True, datasets=datasets, get_specific_checkpoint=evaluate_epoch + finetune_epochs
        )

        search_jobs, wers = search(
            prefix_name=ft_name + "/" + finetune_set + f"/test_{test_set}" + "/default_%i" % (evaluate_epoch + finetune_epochs),
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

    # Build model config
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

    if checkpoint_path is None:
        checkpoint_path = (
            f"/u/kaloyan.nikolov/experiments/multilang_0325/output/output/ctc/"
            f"bpe_{vocab_size}.vocab/{batch_size}_{vocab_size}_lr{checkpoint_lr}/"
            f"out_model_dir/epoch.{eval_epoch}.pt"
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

        extra_config = {
            "preload_from_files": {
                "codebook": {
                    "filename": checkpoint_path,
                    "init_for_train": True,
                    "checkpoint_key": "model",
                    "ignore_missing": True,
                    "ignore_params_prefixes": ignore_params_prefixes,
                }
            }
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

        train_data = CTCTrainingDatasets(
            train=get_finetune_dataset_with_language(split="train", partition_epoch=20),
            cv=get_finetune_dataset_with_language(split="dev", partition_epoch=1),
            prior=get_finetune_dataset_with_language(split="train", partition_epoch=1),
        )

        exp_name = (
            prefix_name
            + f"conformer_new/engram_acoustic_ctc/ctc/{vocab_stem}/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}"
        )

        train_job, search_jobs = run_exp(
            exp_name,
            datasets=train_data,
            train_args=train_args,
            search_args=search_args,
            recog_args=recog_args,
            extra_config=extra_config,
            with_prior=False,
            finetune_epochs=finetune_epochs,
            evaluate_epoch=eval_epoch,
            recog_mem=recog_mem,
            vocab_name=vocab_name,
        )

        train_job.rqmt["gpu_mem"] = gpu_mem
        tk.register_output(
            f"output/ctc/engram_finetune/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}/learning_rates",
            train_job.out_learning_rates,
        )
        tk.register_output(
            f"output/ctc/engram_finetune/{finetune_set}/{batch_size}_{vocab_size}_lr{peak_lr}/out_model_dir",
            train_job.out_model_dir,
        )

    return train_job, search_jobs
