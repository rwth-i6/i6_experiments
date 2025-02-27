from typing import Dict, List, Optional, Tuple, Union, Any

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFunction
import i6_core.rasr as rasr
from i6_core.am.config import acoustic_model_config
from i6_core.returnn import CodeWrapper
from i6_experiments.users.berger.network.helpers.conformer import add_conformer_stack as add_conformer_stack_simon
from .network_helpers.specaug import add_specaug_layer, add_specaug_layer_v2
from .network_helpers.specaug_configurable import add_specaug_layer as add_specaug_layer_configurable
from .network_helpers.specaug_sort_layer2 import add_specaug_layer as add_specaug_layer_sort_layer2
from .network_helpers.specaug_stft import add_specaug_layer as add_specaug_layer_stft
from .network_helpers.conformer_wei import add_conformer_stack as add_conformer_stack_wei
from .network_helpers.conformer_wei import add_vgg_stack as add_vgg_stack_wei


def make_ctc_rasr_loss_config_v2(
    loss_corpus_path: str,
    loss_lexicon_path: str,
    am_args: Dict,
    loss_corpus_segments: Optional[str] = None,
    loss_corpus_prefix: Optional[str] = None,
    allow_label_loop: bool = True,
    min_duration: int = 1,
    extra_config: Optional[rasr.RasrConfig] = None,
    extra_post_config: Optional[rasr.RasrConfig] = None,
):

    # Make crp and set loss_corpus and lexicon
    loss_crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(loss_crp)

    loss_crp.corpus_config = rasr.RasrConfig()
    loss_crp.corpus_config.file = loss_corpus_path
    if loss_corpus_segments is not None:
        loss_crp.corpus_config.segments.file = loss_corpus_segments
    if loss_corpus_prefix is not None:
        loss_crp.corpus_config.remove_corpus_name_prefix = loss_corpus_prefix

    loss_crp.lexicon_config = rasr.RasrConfig()
    loss_crp.lexicon_config.file = loss_lexicon_path

    allophone_file = am_args.pop("allophone_file", None)
    state_tying_file = am_args.pop("state_tying_file", None)
    loss_crp.acoustic_model_config = acoustic_model_config(**am_args)
    if allophone_file:
        loss_crp.acoustic_model_config.allophones.add_from_lexicon = False
        loss_crp.acoustic_model_config.allophones.add_all = True
        loss_crp.acoustic_model_config.allophones.add_from_file = allophone_file
    if state_tying_file:
        loss_crp.acoustic_model_config.state_tying.type = "lookup"
        loss_crp.acoustic_model_config.state_tying.file = state_tying_file

    # Make config from crp
    mapping = {
        "acoustic_model": "*.model-combination.acoustic-model",
        "corpus": "*.corpus",
        "lexicon": "*.model-combination.lexicon",
    }
    config, post_config = rasr.build_config_from_mapping(
        loss_crp,
        mapping,
        parallelize=False,
    )
    config.action = "python-control"
    config.python_control_loop_type = "python-control-loop"
    config.extract_features = False

    # Allophone state transducer
    config["*"].transducer_builder_filter_out_invalid_allophones = True  # type: ignore
    config["*"].fix_allophone_context_at_word_boundaries = True  # type: ignore

    # Automaton manipulation
    if allow_label_loop:
        topology = "ctc"
    else:
        topology = "rna"
    config["*"].allophone_state_graph_builder.topology = topology  # type: ignore

    if min_duration > 1:
        config["*"].allophone_state_graph_builder.label_min_duration = min_duration  # type: ignore

    # maybe not needed
    config["*"].allow_for_silence_repetitions = False  # type: ignore

    config._update(extra_config)
    post_config._update(extra_post_config)

    return config, post_config


def format_func(s, *args, **kwargs):
    return s % args


def make_rasr_ctc_loss_opts(
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    v2: bool = True,
    num_instances: int = 2,
    **kwargs,
):
    trainer_exe = rasr_binary_path.join_right(f"nn-trainer.{rasr_arch}")

    if v2:
        config, post_config = make_ctc_rasr_loss_config_v2(**kwargs)
    else:
        assert False, "only v2 is supported for rasr ctc loss"

    loss_opts = {
        "sprint_opts": {
            "sprintExecPath": trainer_exe,
            "sprintConfigStr": DelayedFunction(
                "%s %s --*.LOGFILE=nn-trainer.loss.log --*.TASK=1", format_func, config, post_config
            ),
            "minPythonControlVersion": 4,
            "numInstances": num_instances,
            "usePythonSegmentOrder": False,
        },
        "tdp_scale": 0.0,
    }
    return loss_opts


def add_rasr_fastbw_output_layer(
    network: Dict,
    from_list: Union[str, List[str]],
    num_outputs: int,
    name: str = "output",
    l2: Optional[float] = None,
    **kwargs,
):
    network[name] = {
        "class": "softmax",
        "from": from_list,
        "loss": "fast_bw",
        "loss_opts": make_rasr_ctc_loss_opts(**kwargs),
        "target": None,
        "n_out": num_outputs,
    }
    if l2:
        network[name]["L2"] = l2

    return name


def make_conformer_fullsum_ctc_model(
    num_outputs: int,
    conformer_args: Optional[Dict] = None,
    output_args: Optional[Dict] = None,
    conformer_type: str = "wei",
    specaug_old: Optional[Dict[str, Any]] = None,
    specaug_config: Optional[Dict[str, Any]] = None,
    specaug_stft: Optional[Dict[str, Any]] = None,
    recognition: bool = False,
    num_epochs: Optional[int] = None,
) -> Tuple[Dict, Union[str, List[str]]]:
    network = {}
    from_list = ["data"]

    if recognition:
        python_code = []
    else:
        if specaug_stft is not None:
            frame_size = specaug_stft.pop("frame_size", 200)
            frame_shift = specaug_stft.pop("frame_shift", 80)
            fft_size = specaug_stft.pop("fft_size", 256)

            specaug_stft_args = {
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 4,
                **specaug_stft,
            }

            # Add STFT layer
            network["stft"] = {
                "class": "stft",
                "from": ["data"],
                "frame_size": frame_size,
                "frame_shift": frame_shift,
                "fft_size": fft_size,
            }
            from_list = ["stft"]

            specaug_func = add_specaug_layer_stft
            from_list, python_code = specaug_func(network, from_list=from_list, **specaug_stft_args)

            # Add iSTFT layer
            network["istft"] = {
                "class": "istft",
                "from": from_list,
                "frame_size": frame_size,
                "frame_shift": frame_shift,
                "fft_size": fft_size,
            }

        elif specaug_old is not None:
            assert specaug_config is None
            sort_layer2 = specaug_old.pop("sort_layer2", False)
            specaug_func = add_specaug_layer_sort_layer2 if sort_layer2 else add_specaug_layer
            specaug_old_args = {
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 4,
                **specaug_old,
            }
            from_list, python_code = specaug_func(network, from_list=from_list, **specaug_old_args)

        elif specaug_config is not None:
            assert specaug_old is None
            from_list, python_code = add_specaug_layer_configurable(
                network, from_list=from_list, num_epochs=num_epochs, config=specaug_config
            )
        else:
            from_list, python_code = add_specaug_layer_v2(network, from_list=from_list)

    if conformer_type == "wei":
        network, from_list = add_vgg_stack_wei(network, from_list)
        conformer_args_full = {
            "pos_enc_clip": 32,
            "batch_norm_fix": True,
            "switch_conv_mhsa_module": True,
            **(conformer_args or {}),
        }
        network, from_list = add_conformer_stack_wei(network, from_list, **conformer_args_full)
    elif conformer_type == "simon":
        transducer_vgg_defaults = {
            "out_dims": [32, 64, 64],
            "filter_sizes": [3, 3, 3],
            "strides": [1, (2, 1), (2, 1)],
            "pool_sizes": [(1, 2)],
            "activation": CodeWrapper("nn.swish"),
        }
        network, from_list = add_vgg_stack_wei(network, from_list)  # TODO: check for Simon's latest VGG variant
        transducer_conformer_defaults = {
            "size": 512,
            "conv_filter_size": 32,
            "use_batch_norm": True,
            "num_att_heads": 8,
        }
        conformer_args_full = {
            **transducer_conformer_defaults,
            **(conformer_args or {}),
        }
        from_list, _ = add_conformer_stack_simon(network, from_list, "conformer", **conformer_args_full)

    network["encoder"] = {"class": "copy", "from": from_list}
    if recognition:
        network["output"] = {
            "class": "linear",
            "from": "encoder",
            "activation": "log_softmax",
            "n_out": num_outputs,
        }
    else:
        add_rasr_fastbw_output_layer(network, from_list="encoder", num_outputs=num_outputs, **(output_args or {}))

    return network, python_code
