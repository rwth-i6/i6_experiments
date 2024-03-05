import copy
import os
from typing import Dict, Tuple

import i6_core.rasr as rasr
from i6_core.recognition import Hub5ScoreJob
from i6_core.returnn import Checkpoint
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
import i6_experiments.users.berger.network.models.fullsum_ctc as ctc_model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.util import default_tools_v2 as tools
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from i6_experiments.users.berger.corpus.switchboard.ctc_data import (
    get_switchboard_data,
)
from i6_experiments.users.berger.args.jobs.recognition_args import (
    get_seq2seq_search_parameters,
)
from sisyphus import gs, tk
from i6_experiments.users.berger.recipe.mm.alignment import ComputeTSEJob

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}


num_classes = 88


# ********** Return Config generators **********


def generate_returnn_config(
    train: bool,
    *,
    loss_corpus: tk.Path,
    loss_lexicon: tk.Path,
    am_args: dict,
    train_data_config: dict,
    dev_data_config: dict,
    num_epochs: int = 300,
    peak_lr: float = 8e-04,
    am_scale: float = 1.0,
) -> ReturnnConfig:
    if train:
        network_dict, extra_python = ctc_model.make_blstm_fullsum_ctc_model(
            num_outputs=num_classes,
            specaug_args={
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 4,
            },
            blstm_args={
                "num_layers": 6,
                "max_pool": [1, 2, 2],
                "size": 512,
                "dropout": 0.1,
                "l2": 1e-04,
            },
            mlp_args={"num_layers": 0},
            output_args={
                "rasr_binary_path": tools.rasr_binary_path,
                "loss_corpus_path": loss_corpus,
                "loss_lexicon_path": loss_lexicon,
                "am_args": am_args,
            },
        )
    else:
        network_dict, extra_python = ctc_model.make_blstm_ctc_recog_model(
            num_outputs=num_classes,
            blstm_args={
                "num_layers": 6,
                "max_pool": [1, 2, 2],
                "size": 512,
            },
            mlp_args={"num_layers": 0},
        )

    if am_scale != 1.0:
        sprint_opts = copy.deepcopy(network_dict["output"]["loss_opts"]["sprint_opts"])
        network_dict["output"] = {
            "class": "softmax",
            "from": "encoder",
            "n_out": num_classes,
        }
        network_dict["scaled_log_probs"] = {
            "class": "eval",
            "eval": f"{am_scale} * safe_log(source(0))",
            "from": "output",
        }
        network_dict["fast_bw"] = {
            "class": "fast_bw",
            "from": "scaled_log_probs",
            "sprint_opts": sprint_opts,
            "align_target": "sprint",
        }
        network_dict["output_loss"] = {
            "class": "copy",
            "from": "output",
            "loss": "via_layer",
            "loss_opts": {
                "align_layer": "fast_bw",
                "loss_wrt_to_act_in": "softmax",
            },
            "loss_scale": 1.0,
        }

    returnn_config = get_returnn_config(
        network=network_dict,
        target=None,
        num_epochs=num_epochs,
        num_inputs=40,
        extra_python=extra_python,
        extern_data_config=True,
        grad_noise=0.0,
        grad_clip=0.0,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=1e-05,
        peak_lr=peak_lr,
        final_lr=1e-05,
        cycle_epoch=4 * num_epochs // 10,
        batch_size=10000,
        use_chunking=False,
        extra_config={
            "train": train_data_config,
            "dev": dev_data_config,
        },
    )
    returnn_config = serialize_dim_tags(returnn_config)

    return returnn_config


def run_exp() -> Tuple[SummaryReport, Dict[str, Dict[str, AlignmentData]]]:
    assert tools.returnn_root is not None
    assert tools.returnn_python_exe is not None
    assert tools.rasr_binary_path is not None

    data = get_switchboard_data(
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        feature_type=FeatureType.GAMMATONE_8K,
        test_keys=["hub5e01"],
        use_wei_data=True,
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=300,
        gpu_mem_rqmt=11,
    )

    recog_args = exp_args.get_ctc_recog_step_args(num_classes)
    align_args = exp_args.get_ctc_align_step_args(num_classes)
    # recog_args["epochs"] = [160, 226, 236, 266, 273, 284, 287, 292, 293, 299, 300]
    # recog_args["epochs"] = [266, 284, 299]
    recog_args["epochs"] = [300]
    recog_args["feature_type"] = FeatureType.GAMMATONE_8K
    recog_args["prior_scales"] = [0.3]
    recog_args["lm_scales"] = [0.8]
    recog_args["flow_args"] = {"dc_detection": True}
    recog_args["search_parameters"] = get_seq2seq_search_parameters(
        lp=14.4,
        lpl=100000,
        wp=0.5,
        wpl=10000,
        allow_blank=True,
        allow_loop=True,
    )
    align_args["epoch"] = 300
    align_args["feature_type"] = FeatureType.GAMMATONE_8K
    align_args["flow_args"] = {"dc_detection": True}
    align_args["silence_phone"] = "[SILENCE]"

    recog_am_args = copy.deepcopy(exp_args.ctc_recog_am_args)
    recog_am_args.update(
        {
            "tying_type": "global-and-nonword",
            "nonword_phones": ["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"],
        }
    )
    loss_am_args = copy.deepcopy(exp_args.ctc_loss_am_args)
    loss_am_args.update(
        {
            "state_tying": "lookup",
            "state_tying_file": tk.Path("/work/asr4/berger/dependencies/switchboard/state_tying/wei_mono-eow"),
            "tying_type": "global-and-nonword",
            "nonword_phones": ["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"],
            "phon_history_length": 0,
            "phon_future_length": 0,
        }
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
        am_args=recog_am_args,
    )
    system.setup_scoring(scorer_type=Hub5ScoreJob)

    # ********** Returnn Configs **********

    config_generator_kwargs = {
        "loss_corpus": data.loss_corpus,
        "loss_lexicon": data.loss_lexicon,
        "am_args": loss_am_args,
        "dev_data_config": data.cv_data_config,
    }

    for am_scale in [0.1, 0.3, 0.5, 0.7, 1.0]:
        train_config = generate_returnn_config(
            train=True,
            train_data_config=data.train_data_config,
            am_scale=am_scale,
            **config_generator_kwargs,
        )
        recog_config = generate_returnn_config(
            train=False, train_data_config=data.train_data_config, **config_generator_kwargs
        )

        returnn_configs = ReturnnConfigs(
            train_config=train_config,
            recog_configs={"recog": recog_config},
        )

        system.add_experiment_configs(f"BLSTM_CTC_am-{am_scale}", returnn_configs)

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    # system.run_test_recog_step(**recog_args)

    alignments = system.run_align_step(**align_args)

    assert system.summary_report
    return system.summary_report, alignments


def py() -> Tuple[SummaryReport, Dict[str, Dict[str, AlignmentData]]]:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report, alignments = run_exp()
    for am_scale, align in alignments.items():
        train_alignment = align["train_align"]
        compute_tse_job = ComputeTSEJob(
            alignment_cache=train_alignment.alignment_cache_bundle,
            allophone_file=train_alignment.allophone_file,
            silence_phone=train_alignment.silence_phone,
            upsample_factor=4,
            ref_alignment_cache=tk.Path("/work/asr4/berger/20220711_alignment_plotting/bundles/ref_gmm.bundle"),
            ref_allophone_file=tk.Path(
                "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/2022-03--fullsum/work/allophones/StoreAllophones.wNiR4cF7cdOE/output/allophones"
            ),
            ref_silence_phone="[SILENCE]",
            ref_upsample_factor=1,
            remove_outlier_limit=100,
        )
        tk.register_output(f"tse/train_am-{am_scale}", compute_tse_job.out_tse_frames)

    compute_tse_job = ComputeTSEJob(
        alignment_cache=tk.Path(
            "/u/luescher/setups/librispeech/2024-01-24--is-paper/work/i6_core/mm/alignment/AlignmentJob.q36gcJyIijLZ/output/alignment.cache.bundle"
        ),
        allophone_file=tk.Path(
            "/u/luescher/setups/librispeech/2024-01-24--is-paper/work/i6_core/lexicon/allophones/StoreAllophonesJob.1kNrzYYG2BFu/output/allophones"
        ),
        silence_phone="[SILENCE]",
        upsample_factor=4,
        ref_alignment_cache=tk.Path(
            "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"
        ),
        ref_allophone_file=tk.Path(
            "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones"
        ),
        ref_silence_phone="[SILENCE]",
        ref_upsample_factor=1,
        # remove_outlier_limit=100,
    )
    tk.register_output("tse/chris", compute_tse_job.out_tse_frames)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report, alignments
