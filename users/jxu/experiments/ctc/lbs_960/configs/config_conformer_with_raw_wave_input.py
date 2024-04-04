import copy
import os
from typing import cast
from dataclasses import asdict
import numpy as np

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
# from i6_experiments.users.berger.pytorch.models import conformer_ctc
# from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks import conformer_ctc_downsample_4 as conformer_ctc
from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks import \
    conformer_ctc_d_model_512_num_layers_12_raw_wave as conformer_ctc
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.jxu.experiments.ctc.lbs_960.ctc_data import get_librispeech_data_hdf
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
import i6_experiments.users.jxu.experiments.ctc.lbs_960.utils.helper as helper
from i6_experiments.users.rossenbach.experiments.rescale.tedlium2_standalone_2023.pipeline import search
import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.users.berger.recipe.lexicon.modification import (
    EnsureSilenceFirstJob,
    DeleteEmptyOrthJob,
)
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 800

tools = copy.deepcopy(default_tools_v2)
# tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/onnx/arch/linux-x86_64-standard")
# tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def returnn_config_generator(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, lr: dict,
                             batch_size: int) -> ReturnnConfig:
    model_config = conformer_ctc.get_default_config_v1(num_inputs=50, num_outputs=num_outputs)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {"classes": {"dim": num_outputs}}

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="targets",
        extra_python=[conformer_ctc.get_serializer(model_config, variant=variant)],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip=0.0,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.OCLR,
        max_seqs=60,
        initial_lr=lr["initial_lr"],
        peak_lr=lr["peak_lr"],
        final_lr=1e-08,
        batch_size=batch_size,
        use_chunking=False,
        extra_config=extra_config,
    )


def get_returnn_config_collection(
        train_data_config: dict,
        dev_data_config: dict,
        lr: dict,
        batch_size: int = 36000 * 160
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config, "lr": lr,
                        "batch_size": batch_size}
    return ReturnnConfigs(
        train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **generator_kwargs),
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **generator_kwargs)},
    )


def run_lbs_960_torch_conformer_with_raw_wave() -> SummaryReport:
    prefix = "experiments/ctc/conformer_with_raw_wave_input"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )

    data = get_librispeech_data_hdf(
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
        blank_index_last=False,
    )
    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=24)
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        # epochs=[100, 200, 300, 400, 500, 600, 700, num_subepochs],
        epochs = [num_subepochs],
        prior_scales=[0.5, 0.6, 0.7, 0.9],
        lm_scales=[0.9, 1.0, 1.1],
        feature_type=FeatureType.SAMPLES,
        flow_args={"scale_input": 1}
    )

    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    tools.rasr_binary_path = tk.Path(
        "/u/berger/repositories/rasr_versions/gen_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    )
    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring(score_kwargs={"sctk_binary_path": SCTK_BINARY_PATH})

    # ********** Returnn Configs **********

    peak_lr_7e_04 = {
        "initial_lr": 7e-06,
        "peak_lr": 7e-04,
    }
    system.add_experiment_configs(
        "Conformer_CTC_d_model_512_num_layers_12_epochs_40_peark_lr_7e_04_batch_36000_raw_wave_input",
        get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_7e_04)
    )

    system.run_train_step(**train_args)

    # ********** Do recognition with rasr **********

    system.run_dev_recog_step(**recog_args)
    #system.run_test_recog_step(**recog_args)

    # ********** Do recognition with flash-light **********

    # get lexicon without blank

    # bliss_lexicon = lbs_dataset.get_g2p_augmented_bliss_lexicon_dict(
    #     use_stress_marker=False,
    #     add_unknown_phoneme_and_mapping=True,
    # )["train-other-960"]

    # bliss_lexicon = EnsureSilenceFirstJob(bliss_lexicon).out_lexicon
    # bliss_lexicon = DeleteEmptyOrthJob(bliss_lexicon).out_lexicon

    # bliss_lexicon = tk.Path("/u/jxu/setups/librispeech-960/2023-10-17-torch-conformer-ctc/work/i6_core/g2p/convert/G2POutputToBlissLexiconJob.SIIDsOAhK3bA/output/oov.lexicon.gz")
    # bliss_lexicon = DeleteEmptyOrthJob(bliss_lexicon).out_lexicon
    # eow_bliss_lexicon = AddEowPhonemesToLexiconJob(bliss_lexicon).out_lexicon
    # # eow_bliss_lexicon = tk.Path("/u/jxu/setups/librispeech-960/2023-10-17-torch-conformer-ctc/work/i6_core/lexicon/modification/AddEowPhonemesToLexiconJob.hZvb3GYxsP1C/output/lexicon.xml")
    # arpa_lm = data.data_inputs["dev-other"].lm.filename
    # 
    # word_lexicon = BlissLexiconToWordLexicon(eow_bliss_lexicon).out_lexicon
    # train_data_label_datastream = helper.get_eow_vocab_datastream(eow_bliss_lexicon)
    # label_datastream = cast(LabelDatastream, train_data_label_datastream)
    # 
    # default_search_args = {
    #     "lexicon": word_lexicon,
    #     "returnn_vocab": label_datastream.vocab,
    #     "beam_size": 64,
    #     "arpa_lm": arpa_lm,
    #     "beam_threshold": 50,
    # }
    # 
    # train_job_name = "Conformer_CTC_d_model_512_num_layers_12_epochs_40_peark_lr_6e_04_batch_36000_raw_wave_input"
    # recog_epoch = 80
    # # for lm_weight in [1.5, 2.0, 2.5]:
    # #     for prior_scale in [0.3, 0.5, 0.75, 1.0]:
    # for lm_weight in [1.5]:
    #     for prior_scale in [0.3]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #         }
    #         search_args["prior_file"] = helper.get_prior_file(
    #             system._train_jobs[train_job_name].out_checkpoints[recog_epoch],
    #             system._returnn_configs[train_job_name].prior_config)
    # 
    #         # ! TODO: chang it later!!!
    #         train_args_adamw03_accum2 = {
    #             "config": {
    #                 "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
    #                 "learning_rates": list(np.linspace(1e-5, 1e-3, 125)) + list(np.linspace(1e-3, 1e-6, 125)),
    #                 "batch_size": 300 * 16000,
    #                 "max_seq_length": {"audio_features": 35 * 16000},
    #                 "accum_grad_multiple_step": 2,
    #                 "extern_data": {"data": {"dim": 50}, "targets": {"dim": 79, "sparse": True}},
    #             },
    #         }
    # 
    #         train_args = {
    #             **train_args_adamw03_accum2,
    #             "network_module": "conformer_ctc_d_model_512_num_layers_12_raw_wave",
    #             "debug": True,
    #             "net_args": {
    #                 "model_config_dict": asdict(
    #                     conformer_ctc.get_default_config_v1(num_inputs=50, num_outputs=num_outputs)),
    #             },
    #         }
    # 
    #         returnn_search_config = helper.get_search_config(**train_args, decoder_args=search_args,
    #                                                          decoder="ctc.decoder.flashlight_phoneme_ctc")
    # 
    #         # build testing datasets
    #         test_dataset_tuples = {}
    #         for testset in ["dev-other"]:
    #             test_dataset_tuples[testset] = helper.build_test_dataset(testset)
    # 
    #         nickt_returnn = tk.Path("/u/rossenbach/src/NoReturnn")
    #         # nickt_returnn = tk.Path("/u/jxu/setups/librispeech-960/2023-10-17-torch-conformer-ctc/tools/MiniReturnn")
    #         _, _, search_jobs = search(prefix + "/default_%i" % recog_epoch, returnn_search_config,
    #                                    system._train_jobs[train_job_name].out_checkpoints[recog_epoch],
    #                                    test_dataset_tuples, default_tools_v2.returnn_python_exe,
    #                                    nickt_returnn, )

    assert system.summary_report
    return system.summary_report
