__all__ = ["run", "run_single"]

import copy
import dataclasses
from dataclasses import dataclass
import itertools

import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk

# -------------------- Recipes --------------------

from i6_core import meta, mm, recognition
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_experiments.common.datasets.librispeech import durations
from i6_experiments.common.setups.rasr.config.am_config import Tdp
from i6_experiments.common.setups.rasr.hybrid_decoder import HybridDecoder
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.common.setups.rasr.util.decode import (
    AdvTreeSearchJobArgs,
    DevRecognitionParameters,
    Lattice2CtmArgs,
    OptimizeJobArgs,
    PriorPath,
)

from ...setups.common.nn import oclr, returnn_time_tag
from ...setups.common.decoder.rtf import ExtractSearchStatisticsJob
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.decoder.config import PriorInfo
from ...setups.fh.network import conformer
from ...setups.fh.factored import PhoneticContext
from ...setups.fh.network import aux_loss, extern_data
from ...setups.fh.network.augment import (
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CONF_CHUNKING,
    CONF_FH_DECODING_TENSOR_CONFIG,
    CONF_FOCAL_LOSS,
    CONF_LABEL_SMOOTHING,
    CONF_SA_CONFIG,
    FROM_SCRATCH_CV_INFO,
    L2,
    RAISSI_ALIGNMENT,
    RASR_ROOT_FH_GUNZ,
    RASR_ROOT_RS_RASR_GUNZ,
    RETURNN_PYTHON_TF15,
    SCRATCH_ALIGNMENT,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_FH_GUNZ, "arch", gs.RASR_ARCH))
RASR_BINARY_PATH.hash_override = "FH_RASR_PATH"
RASR_BINARY_PATH.hash_override = "RS_RASR_PATH"

RS_RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_RS_RASR_GUNZ, "arch", gs.RASR_ARCH))

RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON_TF15)
RETURNN_PYTHON_EXE.hash_override = "FH_RETURNN_PYTHON_EXE"

train_key = "train-other-960"


@dataclass(frozen=True)
class Experiment:
    alignment: tk.Path
    alignment_name: str
    lr: str
    multitask: bool
    dc_detection: bool
    run_performance_study: bool
    tune_decoding: bool

    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    scratch_align = tk.Path(SCRATCH_ALIGNMENT, cached=True)
    tri_gmm_align = tk.Path(RAISSI_ALIGNMENT, cached=True)

    configs = [
        Experiment(
            alignment=tri_gmm_align,
            alignment_name="GMMtri",
            dc_detection=False,
            lr="v7",
            multitask=True,
            run_performance_study=True,
            tune_decoding=True,
        ),
    ]
    for exp in configs:
        run_single(
            alignment=exp.alignment,
            alignment_name=exp.alignment_name,
            dc_detection=exp.dc_detection,
            focal_loss=exp.focal_loss,
            lr=exp.lr,
            multitask=exp.multitask,
            returnn_root=returnn_root,
            run_performance_study=exp.run_performance_study,
            tune_decoding=exp.tune_decoding,
        )


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    dc_detection: bool,
    focal_loss: float,
    lr: str,
    multitask: bool,
    returnn_root: tk.Path,
    run_performance_study: bool,
    tune_decoding: bool,
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"conf-ep:{num_epochs}-lr:{lr}-fl:{focal_loss}-mt:{int(multitask)}"
    print(f"fh {name}")

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()
    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=True, dc_detection=dc_detection)
    data_preparation_args = gmm_setups.get_final_output(name="data_preparation")
    # *********** System Instantiation *****************
    steps = rasr_util.RasrSteps()
    steps.add_step("init", None)  # you can create the label_info and pass here
    s = fh_system.FactoredHybridSystem(
        rasr_binary_path=RASR_BINARY_PATH,
        rasr_init_args=rasr_init_args,
        returnn_root=returnn_root,
        returnn_python_exe=RETURNN_PYTHON_EXE,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    s.do_not_set_returnn_python_exe_for_graph_compiles = True
    s.train_key = train_key
    if alignment_name == "scratch":
        s.cv_info = FROM_SCRATCH_CV_INFO
    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
    s.alignments[train_key] = alignment
    steps_input = rasr_util.RasrSteps()
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    steps_input.add_step("input", data_preparation_args)
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(
        is_cv_separate_from_train=alignment_name == "scratch",
        input_key="data_preparation",
        chunk_size=CONF_CHUNKING,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
    )

    # ---------------------- returnn config---------------
    partition_epochs = {"train": 40, "dev": 1}

    time_prolog, time_tag_name = returnn_time_tag.get_shared_time_tag()
    network_builder = conformer.get_best_model_config(
        conf_model_dim,
        chunking=CONF_CHUNKING,
        focal_loss_factor=CONF_FOCAL_LOSS,
        label_smoothing=CONF_LABEL_SMOOTHING,
        num_classes=s.label_info.get_n_of_dense_classes(),
        time_tag_name=time_tag_name,
    )
    network = network_builder.network
    network = augment_net_with_label_pops(network, label_info=s.label_info)
    network = augment_net_with_monophone_outputs(
        network,
        add_mlps=True,
        encoder_output_len=conf_model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        use_multi_task=multitask,
    )
    network = aux_loss.add_intermediate_loss(
        network,
        center_state_only=True,
        context=PhoneticContext.monophone,
        encoder_output_len=conf_model_dim,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        time_tag_name=time_tag_name,
    )

    base_config = {
        **s.initial_nn_args,
        **oclr.get_oclr_config(num_epochs=num_epochs, schedule=lr),
        **CONF_SA_CONFIG,
        "batch_size": 11000 if lr == "v7" else 6144,
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "chunking": CONF_CHUNKING,
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.0,
        "network": network,
        "extern_data": {
            "data": {
                "dim": 50,
                "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)},
            },
            **extern_data.get_extern_data_config(label_info=s.label_info, time_tag_name=time_tag_name),
        },
    }
    keep_epochs = [550, num_epochs]
    base_post_config = {
        "cleanup_old_models": {
            "keep_best_n": 3,
            "keep": keep_epochs,
        },
    }
    returnn_config = returnn.ReturnnConfig(
        config=base_config,
        post_config=base_post_config,
        hash_full_python_code=True,
        python_prolog={
            "numpy": "import numpy as np",
            "time": time_prolog,
        },
        python_epilog={
            "functions": [
                sa_mask,
                sa_random_mask,
                sa_summary,
                sa_transform,
            ],
        },
    )

    s.set_experiment_dict("fh", alignment_name, "mono", postfix_name=name)
    s.set_returnn_config_for_experiment("fh", copy.deepcopy(returnn_config))

    train_args = {
        **s.initial_train_args,
        "returnn_config": returnn_config,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
    }

    s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
        on_2080=False,
    )
    s.set_mono_priors_returnn_rasr(
        key="fh",
        epoch=keep_epochs[-2],
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
    )

    for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RS_RASR_BINARY_PATH)

        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.monophone,
            crp_corpus=crp_k,
            epoch=ep,
            gpu=False,
            tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
            recompile_graph_for_feature_scorer=True,
        )

        if run_performance_study:
            for altas, beam in itertools.product([2, 4, 8], [22, 20, 18]):
                previous_alias = gs.ALIAS_AND_OUTPUT_SUBDIR
                gs.ALIAS_AND_OUTPUT_SUBDIR = f"{previous_alias}_beam{beam}"

                log_softmax_returnn_config = copy.deepcopy(returnn_config)
                for layer in log_softmax_returnn_config.config["network"].values():
                    layer.pop("target", None)
                    layer.pop("loss", None)
                    layer.pop("loss_scale", None)
                    layer.pop("loss_opts", None)
                    layer.pop("register_as_extern_data", None)

                log_softmax_returnn_config.config["network"]["center-output"] = {
                    **log_softmax_returnn_config.config["network"]["center-output"],
                    "class": "linear",
                    "activation": "log_softmax",
                    "register_as_extern_data": "output",
                }

                for k in ["centerState", "classes", "futureLabel", "pastLabel"]:
                    log_softmax_returnn_config.config["extern_data"][k].pop("same_dim_tags_as", None)

                if altas == 4 and beam == 22:
                    params = (
                        dataclasses.replace(recog_args, altas=altas if altas > 0 else None, beam=beam)
                        .with_prior_scale(center=0.3)
                        .with_tdp_scale(0.4)
                    )
                else:
                    params = dataclasses.replace(recog_args, altas=altas if altas > 0 else None, beam=beam)
                crp_corpus = crp_k
                key = "fh"
                epoch = ep

                p_info: PriorInfo = s.experiments[key].get("priors", None)
                assert p_info is not None, "set priors first"

                p_mixtures = mm.CreateDummyMixturesJob(
                    s.label_info.get_n_state_classes(), s.initial_nn_args["num_input"]
                ).out_mixtures

                crp = copy.deepcopy(s.crp[crp_corpus])
                crp.acoustic_model_config.state_tying.type = "monophone-no-tying-dense"

                adv_tree_search_job: recognition.AdvancedTreeSearchJob

                def SearchJob(*args, **kwargs):
                    nonlocal adv_tree_search_job
                    adv_tree_search_job = recognition.AdvancedTreeSearchJob(*args, **kwargs)
                    return adv_tree_search_job

                decoder = HybridDecoder(
                    rasr_binary_path=s.rasr_binary_path,
                    returnn_root=s.returnn_root,
                    returnn_python_exe=s.returnn_python_exe,
                    required_native_ops=None,
                    search_job_class=SearchJob,
                )
                decoder.set_crp("init", crp)

                corpus = meta.CorpusObject()
                corpus.corpus_file = crp.corpus_config.file
                corpus.audio_format = crp.audio_format
                corpus.duration = crp.corpus_duration

                decoder.init_eval_datasets(
                    eval_datasets={crp_corpus: corpus},
                    concurrency={crp_corpus: crp.concurrent},
                    corpus_durations=durations,
                    feature_flows=s.feature_flows,
                    stm_paths={crp_corpus: s.scorer_args[crp_corpus]["ref"]},
                )

                @dataclasses.dataclass
                class RasrConfigWrapper:
                    obj: rasr.RasrConfig

                    def get(self) -> rasr.RasrConfig:
                        return self.obj

                if params.altas is not None:
                    adv_search_extra_config = rasr.RasrConfig()
                    adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = (
                        params.altas
                    )
                else:
                    adv_search_extra_config = None
                lat2ctm_extra_config = rasr.RasrConfig()
                lat2ctm_extra_config.flf_lattice_tool.network.to_lemma.links = "best"

                decoder.recognition(
                    name=s.experiments[key]["name"],
                    checkpoints={epoch: s._get_model_checkpoint(s.experiments[key]["train_job"], epoch)},
                    epochs=[epoch],
                    forward_output_layer="center__output",
                    prior_paths={
                        "rp": PriorPath(
                            acoustic_mixture_path=p_mixtures,
                            prior_xml_path=p_info.center_state_prior.file,
                        )
                    },
                    recognition_parameters={
                        crp_corpus: [
                            DevRecognitionParameters(
                                altas=[params.altas] if params.altas is not None else None,
                                am_scales=[1],
                                lm_scales=[params.lm_scale],
                                prior_scales=[params.prior_info.center_state_prior.scale],
                                pronunciation_scales=[params.pron_scale],
                                speech_tdps=[
                                    Tdp(
                                        loop=params.tdp_speech[0],
                                        forward=params.tdp_speech[1],
                                        skip=params.tdp_speech[2],
                                        exit=params.tdp_speech[3],
                                    )
                                ],
                                silence_tdps=[
                                    Tdp(
                                        loop=params.tdp_silence[0],
                                        forward=params.tdp_silence[1],
                                        skip=params.tdp_silence[2],
                                        exit=params.tdp_silence[3],
                                    )
                                ],
                                nonspeech_tdps=[
                                    Tdp(
                                        loop=params.tdp_non_word[0],
                                        forward=params.tdp_non_word[1],
                                        skip=params.tdp_non_word[2],
                                        exit=params.tdp_non_word[3],
                                    )
                                ],
                                tdp_scales=[params.tdp_scale],
                            )
                        ]
                    },
                    returnn_config=log_softmax_returnn_config,
                    lm_configs={crp_corpus: RasrConfigWrapper(obj=crp.language_model_config)},
                    search_job_args=AdvTreeSearchJobArgs(
                        search_parameters={
                            "beam-pruning": beam,
                            "beam-pruning-limit": params.beam_limit,
                            "word-end-pruning": params.we_pruning,
                            "word-end-pruning-limit": params.we_pruning_limit,
                        },
                        use_gpu=True,
                        mem=8,
                        cpu=4,
                        lm_lookahead=True,
                        lmgc_mem=12,
                        lookahead_options=None,
                        create_lattice=True,
                        eval_best_in_lattice=True,
                        eval_single_best=True,
                        extra_config=adv_search_extra_config,
                        extra_post_config=None,
                        rtf=0.75,
                    ),
                    lat_2_ctm_args=Lattice2CtmArgs(
                        parallelize=True,
                        best_path_algo="bellman-ford",
                        encoding="utf-8",
                        extra_config=lat2ctm_extra_config,
                        extra_post_config=None,
                        fill_empty_segments=True,
                    ),
                    scorer_args=s.scorer_args[crp_corpus],
                    optimize_parameters=OptimizeJobArgs(
                        opt_only_lm_scale=True,
                        maxiter=100,
                        precision=2,
                        extra_config=None,
                        extra_post_config=None,
                    ),
                    optimize_pron_lm_scales=False,
                )

                assert adv_tree_search_job is not None
                stats_job = ExtractSearchStatisticsJob(
                    search_logs=list(adv_tree_search_job.out_log_file.values()),
                    corpus_duration_hours=durations[crp_corpus],
                )
                stats_alias = f"statistics/{s.experiments[key]['name']}/Pron{params.pron_scale}Lm{params.lm_scale}Pr{params.prior_info.center_state_prior.scale}Altas{params.altas or 0}"

                stats_job.add_alias(stats_alias)
                tk.register_output(f"{stats_alias}/avg_states", stats_job.avg_states)
                tk.register_output(f"{stats_alias}/avg_trees", stats_job.avg_trees)
                tk.register_output(f"{stats_alias}/rtf", stats_job.decoding_rtf)

                gs.ALIAS_AND_OUTPUT_SUBDIR = previous_alias

    return s
