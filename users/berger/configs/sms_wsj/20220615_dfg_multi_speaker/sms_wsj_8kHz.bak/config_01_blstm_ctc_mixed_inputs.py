import copy
import os
from i6_core.corpus.convert import CorpusToStmJob

import i6_core.rasr as rasr
from i6_core.recognition.scoring import ScliteJob
from i6_core.returnn.config import CodeWrapper
from i6_core.returnn.extract_prior import ReturnnComputePriorJob
import i6_core.text as text
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_core.returnn.search import (
    ReturnnSearchJobV2,
    SearchBPEtoWordsJob,
    SearchWordsToCTMJob,
)
from i6_core.tools import CloneGitRepositoryJob
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack
from i6_experiments.users.berger.network.helpers.specaug import add_specaug_layer
from i6_experiments.users.berger.network.models.fullsum_ctc import (
    make_blstm_fullsum_ctc_model,
)
from i6_experiments.users.berger.network.models.lstm_lm import make_lstm_lm_recog_model
from i6_experiments.users.berger.util import change_source_name
from sisyphus import gs, tk
from sisyphus.delayed_ops import DelayedFormat


# ********** Settings **********

dir_handle = os.path.dirname(__file__).split("config/")[1]
filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

train_key = "train_si284"
dev_key = "cv_dev93"
test_key = "test_eval92"

frequency = 8

f_name = "gt"

num_inputs = 40

lm_models = {
    True: {
        100: tk.Path(
            "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/training/ReturnnTrainingJob.d43AUmy7n8xx/output/models/epoch.150.meta"
        ),
        1000: tk.Path(
            "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/training/ReturnnTrainingJob.N257e26OXI7G/output/models/epoch.150.meta"
        ),
    },
    False: {
        100: tk.Path(
            "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/training/ReturnnTrainingJob.OXKudR5Jxo0p/output/models/epoch.150.meta"
        ),
        1000: tk.Path(
            "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/training/ReturnnTrainingJob.Ol9oGphLUIx5/output/models/epoch.150.meta"
        ),
    },
}


def run_exp(**kwargs) -> None:

    text_cleaning = kwargs.get("text_cleaning", False)

    # ********** Transducer System **********

    subword_nmt_repo = CloneGitRepositoryJob("https://github.com/albertz/subword-nmt.git").out_repository

    bpe_size = kwargs.get("bpe_size", 100)
    if text_cleaning:
        train_bpe_job = ReturnnTrainBpeJob(
            tk.Path(
                "/work/asr4/berger/dependencies/sms_wsj/text/NAB-training-corpus-clean.gz",
                cached=True,
            ),
            bpe_size,
            subword_nmt_repo=subword_nmt_repo,
        )
    else:
        bpe_txt = text.PipelineJob(
            tk.Path("/u/corpora/language/wsj/NAB-training-corpus.gz", cached=True),
            ['sed "s/ \S*$//"', 'sed "s/^\S* //"'],
            mini_task=True,
        ).out  # Remove <s> and </s> tokens
        train_bpe_job = ReturnnTrainBpeJob(
            bpe_txt,
            bpe_size,
            subword_nmt_repo=subword_nmt_repo,
        )
    bpe_codes = train_bpe_job.out_bpe_codes
    bpe_vocab = train_bpe_job.out_bpe_vocab

    num_classes = train_bpe_job.out_vocab_size  # bpe count
    num_classes_b = num_classes + 1  # bpe count + blank

    name = "_".join(filter(None, ["BLSTM_CTC", kwargs.get("name_suffix", "")]))
    max_pool = kwargs.get("max_pool", [1, 2, 2])
    red_fact = 1
    for fact in max_pool:
        red_fact *= fact
    specaug_args = {
        "max_time_num": kwargs.get("max_time_num", 1),
        "max_time": kwargs.get("max_time", 15),
        "max_feature_num": 4,
        "max_feature": 5,
    }

    l2 = kwargs.get("l2", 5e-06)
    dropout = kwargs.get("dropout", 0.1)
    specaug = kwargs.get("specaug", True)

    _, train_python_code = make_blstm_fullsum_ctc_model(num_outputs=num_classes_b)
    train_blstm_net = {}

    clean_data = kwargs.get("clean_data", True)
    from_0 = "data:data_clean_0" if clean_data else "data:data_separated_0"
    if specaug:
        from_0 = add_specaug_layer(train_blstm_net, name="specaug_0", from_list=from_0, **specaug_args)
    from_0, _ = add_blstm_stack(
        train_blstm_net,
        from_list=from_0,
        name="lstm_0",
        num_layers=kwargs.get("num_separate_layers", 6),
        max_pool=max_pool,
        size=400,
        l2=l2,
        dropout=dropout,
    )

    from_1 = "data:data_clean_1" if clean_data else "data:data_separated_1"
    if specaug:
        from_1 = add_specaug_layer(train_blstm_net, name="specaug_1", from_list=from_1, **specaug_args)
    from_1, _ = add_blstm_stack(
        train_blstm_net,
        from_list=from_1,
        name="lstm_1",
        num_layers=kwargs.get("num_separate_layers", 6),
        max_pool=max_pool,
        size=400,
        l2=l2,
        dropout=dropout,
    )
    for layer, attrib in train_blstm_net.items():
        if not (layer.startswith("fwd_lstm_1_") or layer.startswith("bwd_lstm_1_")):
            continue
        attrib["reuse_params"] = layer.replace("lstm_1", "lstm_0")

    if kwargs.get("mixed_input", False):
        from_mix = "data"
        if specaug:
            from_mix = add_specaug_layer(train_blstm_net, name="specaug_mix", from_list=from_mix, **specaug_args)
        from_mix, _ = add_blstm_stack(
            train_blstm_net,
            from_list=from_mix,
            name="lstm_mix",
            num_layers=kwargs.get("num_shared_layers", 6),
            max_pool=max_pool,
            size=400,
            l2=l2,
            dropout=dropout,
        )
        from_0 = from_0 + from_mix
        from_1 = from_1 + from_mix

    train_blstm_net["encoder_0"] = {"class": "copy", "from": from_0}
    train_blstm_net["encoder_1"] = {"class": "copy", "from": from_1}

    train_blstm_net["output_0"] = {
        "class": "softmax",
        "from": ["encoder_0"],
        "n_out": num_classes_b,
    }
    train_blstm_net["output_1"] = {
        "class": "softmax",
        "from": ["encoder_1"],
        "n_out": num_classes_b,
        "reuse_params": "output_0",
    }
    train_blstm_net["ctc_loss_0"] = {
        "class": "fast_bw",
        "from": "output_0",
        "align_target_key": "bpe_0",
        "align_target": "ctc",
        "input_type": "prob",
        "tdp_scale": 0.0,
        "ctc_opts": {"blank_idx": num_classes},
    }
    train_blstm_net["output_loss_0"] = {
        "class": "copy",
        "from": "output_0",
        "loss": "via_layer",
        "loss_opts": {
            "loss_wrt_to_act_in": "softmax",
            "align_layer": "ctc_loss_0",
        },
    }
    train_blstm_net["ctc_loss_1"] = {
        "class": "fast_bw",
        "from": "output_1",
        "align_target_key": "bpe_1",
        "align_target": "ctc",
        "input_type": "prob",
        "tdp_scale": 0.0,
        "ctc_opts": {"blank_idx": num_classes},
    }
    train_blstm_net["output_loss_1"] = {
        "class": "copy",
        "from": "output_1",
        "loss": "via_layer",
        "loss_opts": {
            "loss_wrt_to_act_in": "softmax",
            "align_layer": "ctc_loss_1",
        },
    }

    num_subepochs = kwargs.get("num_subepochs", 150)

    train_config = get_returnn_config(
        train_blstm_net,
        target=None,
        num_inputs=num_inputs,
        num_outputs=num_classes_b,
        num_epochs=num_subepochs,
        extra_python=train_python_code,
        grad_noise=kwargs.get("grad_noise", 0.0),
        grad_clip=kwargs.get("grad_clip", 100.0),
        batch_size=kwargs.get("batch_size", 15000),
        schedule=kwargs.get("schedule", LearningRateSchedules.Newbob),
        peak_lr=kwargs.get("peak_lr", 2e-04),
        learning_rate=kwargs.get("learning_rate", 4e-04),
        min_learning_rate=1e-06,
        n_steps_per_epoch=1100,
        use_chunking=False,
        python_prolog=["from returnn.tf.util.data import Dim"],
        extra_config={
            "train": {
                "class": "HDFDataset",
                "files": [
                    f"/u/berger/asr-exps/sms_wsj/20220615_dfg_multi_speaker/dependencies/hdf/8kHz/sms_train_si284_complete.gt40.bpe-100.{'updated.' if text_cleaning else ''}hdf"
                ],
                "use_cache_manager": False,
                "seq_ordering": "random",
                "partition_epoch": 3,
            },
            "dev": {
                "class": "HDFDataset",
                "files": [
                    f"/u/berger/asr-exps/sms_wsj/20220615_dfg_multi_speaker/dependencies/hdf/8kHz/sms_cv_dev93_complete.gt40.bpe-100.{'updated.' if text_cleaning else ''}hdf"
                ],
                "use_cache_manager": False,
                "seq_ordering": "sorted",
                "partition_epoch": 1,
            },
            "data_time_tag": CodeWrapper('Dim(kind=Dim.Types.Time, description="time")'),
            "extern_data": {
                "data": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                },
                "data_separated_0": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                },
                "data_separated_1": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                },
                "data_clean_0": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                },
                "data_clean_1": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                },
                "bpe_0": {"dim": num_classes, "sparse": True},
                "bpe_1": {"dim": num_classes, "sparse": True},
            },
            "num_outputs": {
                "bpe_0": num_classes_b,
                "bpe_1": num_classes_b,
            },
        },
    )

    train_job = ReturnnTrainingJob(
        train_config,
        log_verbosity=5,
        num_epochs=num_subepochs,
        save_interval=1,
        keep_epochs=None,
        time_rqmt=168,
        mem_rqmt=8,
    )

    train_job.set_vis_name(f"Train {name}")
    train_job.add_alias(f"train_{name}")

    tk.register_output(f"train_nn/{name}", train_job.out_learning_rates)

    prior_net = copy.deepcopy(train_blstm_net)
    prior_net["output_0"]["class"] = "linear"
    prior_net["output_0"]["activation"] = "softmax"
    prior_net["output_1"]["class"] = "linear"
    prior_net["output_1"]["activation"] = "softmax"
    prior_net["output"] = {
        "class": "combine",
        "from": ["output_0", "output_1"],
        "kind": "average",
    }
    prior_net.pop("output_loss_0", None)
    prior_net.pop("output_loss_1", None)
    prior_net.pop("ctc_loss_0", None)
    prior_net.pop("ctc_loss_1", None)

    prior_config = get_returnn_config(
        prior_net,
        target=None,
        num_inputs=num_inputs,
        num_outputs=num_classes_b,
        num_epochs=num_subepochs,
        extra_python=train_python_code,
        batch_size=kwargs.get("batch_size", 15000),
        use_chunking=False,
        python_prolog=["from returnn.tf.util.data import Dim"],
        extra_config={
            "train": {
                "class": "HDFDataset",
                "files": [
                    f"/u/berger/asr-exps/sms_wsj/20220615_dfg_multi_speaker/dependencies/hdf/8kHz/sms_train_si284_complete.gt40.bpe-100.{'updated.' if text_cleaning else ''}hdf"
                ],
                "use_cache_manager": False,
                "seq_ordering": "random",
                "partition_epoch": 3,
            },
            "dev": {
                "class": "HDFDataset",
                "files": [
                    f"/u/berger/asr-exps/sms_wsj/20220615_dfg_multi_speaker/dependencies/hdf/8kHz/sms_cv_dev93_complete.gt40.bpe-100.{'updated.' if text_cleaning else ''}hdf"
                ],
                "use_cache_manager": False,
                "seq_ordering": "sorted",
                "partition_epoch": 1,
            },
            "data_time_tag": CodeWrapper('Dim(kind=Dim.Types.Time, description="time")'),
            "extern_data": {
                "data": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                },
                "data_separated_0": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                },
                "data_separated_1": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                },
                "data_clean_0": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                },
                "data_clean_1": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                },
                "bpe_0": {"dim": num_classes, "sparse": True},
                "bpe_1": {"dim": num_classes, "sparse": True},
            },
            "num_outputs": {
                "bpe_0": num_classes_b,
                "bpe_1": num_classes_b,
            },
            "forward_output_layer": "output",
        },
    )

    prior_job = ReturnnComputePriorJob(
        model_checkpoint=train_job.out_checkpoints[num_subepochs],
        returnn_config=prior_config,
        log_verbosity=5,
        mem_rqmt=8,
    )

    # "        "/u/berger/asr-exps/sms_wsj/20220615_dfg_multi_speaker/work/i6_core/returnn/extract_prior/ReturnnRasrComputePriorJob.OXQGJYEuN1tV/output/prior.txt",'
    recog_python_code = [
        DelayedFormat(
            'def get_prior_vector():\n    return np.loadtxt("{}", dtype=np.float32)\n',
            prior_job.out_prior_txt_file,
        )
    ]

    lm_scale = kwargs.get("lm_scale", 1.1)
    prior_scale = kwargs.get("prior_scale", 0.3)
    blank_penalty = kwargs.get("blank_penalty", 0.6)

    recog_blstm_net = copy.deepcopy(train_blstm_net)
    recog_blstm_net.pop("specaug_0", None)
    recog_blstm_net.pop("specaug_1", None)
    recog_blstm_net.pop("specaug_mix", None)
    recog_blstm_net.pop("ctc_loss_0", None)
    recog_blstm_net.pop("ctc_loss_1", None)
    recog_blstm_net.pop("output_loss_0", None)
    recog_blstm_net.pop("output_loss_1", None)
    change_source_name(
        recog_blstm_net,
        "specaug_0",
        "data:data_clean_0" if clean_data else "data:data_separated_0",
    )
    change_source_name(
        recog_blstm_net,
        "specaug_1",
        "data:data_clean_1" if clean_data else "data:data_separated_1",
    )
    change_source_name(recog_blstm_net, "specaug_mix", "data")

    recog_blstm_net["output_0"].update(
        {
            "class": "linear",
            "activation": "log_softmax",
        }
    )
    recog_blstm_net["output_1"].update(
        {
            "class": "linear",
            "activation": "log_softmax",
        }
    )

    recog_blstm_net["output_bp_0"] = {
        "class": "eval",
        "from": "output_0",
        "eval": DelayedFormat(
            "source(0) - tf.expand_dims(tf.one_hot([{}], {}, on_value={}, dtype=tf.float32), axis=0)",
            num_classes,
            num_classes_b,
            blank_penalty,
        ),
    }
    recog_blstm_net["output_bp_1"] = {
        "class": "eval",
        "from": "output_1",
        "eval": DelayedFormat(
            "source(0) - tf.expand_dims(tf.one_hot([{}], {}, on_value={}, dtype=tf.float32), axis=0)",
            num_classes,
            num_classes_b,
            blank_penalty,
        ),
    }
    recog_blstm_net["output_prior_0"] = {
        "class": "eval",
        "from": "output_bp_0",
        "eval": f'source(0) - {prior_scale} * self.network.get_config().typed_value("get_prior_vector")()',
    }
    recog_blstm_net["output_prior_1"] = {
        "class": "eval",
        "from": "output_bp_1",
        "eval": f'source(0) - {prior_scale} * self.network.get_config().typed_value("get_prior_vector")()',
    }

    recog_blstm_net.update(
        {
            "beam_search_0": {
                "class": "rec",
                "from": "output_prior_0",
                "unit": {
                    "output": {
                        "class": "choice",
                        "from": "combined_scores",
                        "input_type": "log_prob",
                        "target": "bpe_b",
                        "beam_size": kwargs.get("beam_size", 16),
                        "explicit_search_source": "prev:output",
                        "initial_output": num_classes,
                    },
                    "mask_non_blank": {
                        "class": "compare",
                        "from": "output",
                        "value": num_classes,
                        "kind": "not_equal",
                        "initial_output": True,
                    },
                    "prev_output_reinterpret": {
                        "class": "reinterpret_data",
                        "from": "prev:output",
                        "increase_sparse_dim": -1,
                    },
                    "lm_masked": {
                        "class": "masked_computation",
                        "from": "prev_output_reinterpret",
                        "mask": "prev:mask_non_blank",
                        "unit": {
                            "class": "subnetwork",
                            "load_on_init": lm_models[text_cleaning][bpe_size],
                            "subnetwork": make_lstm_lm_recog_model(
                                num_outputs=num_classes,
                                embedding_args={
                                    "size": 256,
                                },
                                lstm_args={
                                    "num_layers": 2,
                                    "size": 2048,
                                },
                            ),
                        },
                    },
                    "lm_padded": {
                        "class": "pad",
                        "from": "lm_masked",
                        "axes": "f",
                        "padding": (0, 1),
                        "value": 0,
                        "mode": "constant",
                    },
                    "combined_scores": {
                        "class": "eval",
                        "from": ["data:source", "lm_padded"],
                        "eval": f"source(0) + {lm_scale} * source(1)",
                    },
                },
            },
            "ctc_decode_0": {
                "class": "subnetwork",
                "is_output_layer": True,
                "target": "bpe_0",
                "subnetwork": {
                    "decision": {
                        "class": "decide",
                        "from": "base:beam_search_0",
                    },
                    "decision_shifted": {
                        "class": "shift_axis",
                        "from": "decision",
                        "axis": "T",
                        "amount": 1,
                        "pad_value": -1,
                        "adjust_size_info": False,
                    },
                    "mask_unique": {
                        "class": "compare",
                        "from": ["decision", "decision_shifted"],
                        "kind": "not_equal",
                    },
                    "mask_non_blank": {
                        "class": "compare",
                        "from": "decision",
                        "kind": "not_equal",
                        "value": num_classes,
                    },
                    "mask_label": {
                        "class": "combine",
                        "from": ["mask_unique", "mask_non_blank"],
                        "kind": "logical_and",
                    },
                    "decision_unique_labels": {
                        "class": "masked_computation",
                        "from": "decision",
                        "mask": "mask_label",
                        "unit": {"class": "copy"},
                    },
                    "output": {
                        "class": "reinterpret_data",
                        "from": "decision_unique_labels",
                        "increase_sparse_dim": -1,
                        "target": "bpe_0",
                        "loss": "edit_distance",
                    },
                },
            },
            "beam_search_1": {
                "class": "rec",
                "from": "output_prior_1",
                "unit": {
                    "output": {
                        "class": "choice",
                        "from": "combined_scores",
                        "input_type": "log_prob",
                        "target": "bpe_b",
                        "beam_size": kwargs.get("beam_size", 16),
                        "explicit_search_source": "prev:output",
                        "initial_output": num_classes,
                    },
                    "mask_non_blank": {
                        "class": "compare",
                        "from": "output",
                        "value": num_classes,
                        "kind": "not_equal",
                        "initial_output": True,
                    },
                    "prev_output_reinterpret": {
                        "class": "reinterpret_data",
                        "from": "prev:output",
                        "increase_sparse_dim": -1,
                    },
                    "lm_masked": {
                        "class": "masked_computation",
                        "from": "prev_output_reinterpret",
                        "mask": "prev:mask_non_blank",
                        "unit": {
                            "class": "subnetwork",
                            "load_on_init": lm_models[text_cleaning][bpe_size],
                            "subnetwork": make_lstm_lm_recog_model(
                                num_outputs=num_classes,
                                embedding_args={
                                    "size": 256,
                                },
                                lstm_args={
                                    "num_layers": 2,
                                    "size": 2048,
                                },
                            ),
                        },
                    },
                    "lm_padded": {
                        "class": "pad",
                        "from": "lm_masked",
                        "axes": "f",
                        "padding": (0, 1),
                        "value": 0,
                        "mode": "constant",
                    },
                    "combined_scores": {
                        "class": "eval",
                        "from": ["data:source", "lm_padded"],
                        "eval": f"source(0) + {lm_scale} * source(1)",
                    },
                },
            },
            "ctc_decode_1": {
                "class": "subnetwork",
                "is_output_layer": True,
                "target": "bpe_1",
                "subnetwork": {
                    "decision": {
                        "class": "decide",
                        "from": "base:beam_search_1",
                    },
                    "decision_shifted": {
                        "class": "shift_axis",
                        "from": "decision",
                        "axis": "T",
                        "amount": 1,
                        "pad_value": -1,
                        "adjust_size_info": False,
                    },
                    "mask_unique": {
                        "class": "compare",
                        "from": ["decision", "decision_shifted"],
                        "kind": "not_equal",
                    },
                    "mask_non_blank": {
                        "class": "compare",
                        "from": "decision",
                        "kind": "not_equal",
                        "value": num_classes,
                    },
                    "mask_label": {
                        "class": "combine",
                        "from": ["mask_unique", "mask_non_blank"],
                        "kind": "logical_and",
                    },
                    "decision_unique_labels": {
                        "class": "masked_computation",
                        "from": "decision",
                        "mask": "mask_label",
                        "unit": {"class": "copy"},
                    },
                    "output": {
                        "class": "reinterpret_data",
                        "from": "decision_unique_labels",
                        "increase_sparse_dim": -1,
                        "target": "bpe_1",
                        "loss": "edit_distance",
                    },
                },
            },
        }
    )

    recog_config = get_returnn_config(
        recog_blstm_net,
        target=None,
        num_inputs=num_inputs,
        num_outputs=num_classes_b,
        num_epochs=num_subepochs,
        use_chunking=False,
        extra_python=recog_python_code,
        python_prolog=["from returnn.tf.util.data import Dim"],
        hash_full_python_code=False,
        extra_config={
            "data_time_tag": CodeWrapper('Dim(kind=Dim.Types.Time, description="time")'),
            "extern_data": {
                "data": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                    "available_for_inference": True,
                },
                "data_separated_0": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                    "available_for_inference": True,
                },
                "data_separated_1": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                    "available_for_inference": True,
                },
                "data_clean_0": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                    "available_for_inference": True,
                },
                "data_clean_1": {
                    "dim": num_inputs,
                    "same_dim_tags_as": {"t": CodeWrapper("data_time_tag")},
                    "available_for_inference": True,
                },
                "bpe_0": {"dim": num_classes, "sparse": True},
                "bpe_1": {"dim": num_classes, "sparse": True},
                "bpe_b": {"dim": num_classes_b, "sparse": True},
            },
            "num_outputs": {
                "bpe_0": num_classes_b,
                "bpe_1": num_classes_b,
            },
            "search_output_layer": ["ctc_decode_0", "ctc_decode_1"],
        },
    )

    for recog_epoch in [num_subepochs]:
        search_job = ReturnnSearchJobV2(
            search_data={
                "class": "HDFDataset",
                "files": [
                    f"/u/berger/asr-exps/sms_wsj/20220615_dfg_multi_speaker/dependencies/hdf/8kHz/sms_test_eval92_complete.gt40.bpe-100.{'updated.' if text_cleaning else ''}hdf"
                ],
                "use_cache_manager": False,
                "seq_ordering": "sorted",
                "partition_epoch": 1,
            },
            model_checkpoint=train_job.out_checkpoints[recog_epoch],
            returnn_config=recog_config,
            output_mode="py",
            log_verbosity=5,
            returnn_python_exe=tk.Path(gs.RETURNN_PYTHON_EXE),
            returnn_root=tk.Path(gs.RETURNN_ROOT),
        )

        out_path = f"nn_recog/{name}/test_eval92_lm-{lm_scale:01.02f}_prior-{prior_scale:01.02f}_bp-{blank_penalty:01.02f}_ep-{recog_epoch:03d}"

        search_job.add_alias(out_path)

        words_job = SearchBPEtoWordsJob(search_job.out_search_file)

        words_job_processed = text.PipelineJob(
            words_job.out_word_search_results,
            [
                'sed "s/\/0000_ctc_decode_0/_0\/0000/"',
                'sed "s/\/0000_ctc_decode_1/_1\/0000/"',
            ],
            mini_task=True,
        )

        recog_bliss_corpus = tk.Path("/work/asr4/berger/dependencies/sms_wsj/corpus/8kHz/sms_test_eval92.corpus.gz")

        word2ctm_job = SearchWordsToCTMJob(words_job_processed.out, recog_bliss_corpus)
        scorer_job = ScliteJob(
            CorpusToStmJob(recog_bliss_corpus, non_speech_tokens=["<NOISE>"]).out_stm_path,
            word2ctm_job.out_ctm_file,
        )

        tk.register_output(f"{out_path}.wer", scorer_job.out_report_dir)


def py() -> None:
    # Clean:
    # Only sep inputs: 8.5, LR: 0.0002
    # Mixed inputs: 8.9, LR: 0.0004

    # Noisy:
    # Only sep inputs: 31.1, LR: 0.0004
    # Mixed inputs: 32.6, LR: 0.0008
    if False:
        for lr in [4e-04]:
            for batch_size in [5000, 10000, 15000]:
                for pool in [[1, 2, 2]]:
                    for mixed_input in [True, False]:
                        for clean_data in [True, False]:
                            name = f"newbob_lr-{lr}_pool-{'-'.join([str(p) for p in pool])}_bs-{batch_size}"
                            if mixed_input:
                                name += "_mixed-input"
                            if clean_data:
                                name += "_clean-data"
                            run_exp(
                                name_suffix=name,
                                learning_rate=lr,
                                max_pool=pool,
                                specaug=True,
                                batch_size=batch_size,
                                mixed_input=mixed_input,
                                clean_data=clean_data,
                                num_subepochs=150,
                            )

    if False:
        for mixed_input in [True]:
            for clean_data in [False]:
                for num_shared_layers in [4, 6]:
                    for num_separate_layers in [4, 6]:
                        for time_num in [1, 2, 3]:
                            for max_time in [10, 15]:
                                name = f"tn-{time_num}_mt-{max_time}_{num_separate_layers}l"
                                if mixed_input:
                                    name += f"_mixed-input-{num_shared_layers}l"
                                if clean_data:
                                    name += "_clean-data"
                                run_exp(
                                    name_suffix=name,
                                    learning_rate=4e-04,
                                    max_pool=[1, 2, 2],
                                    specaug=True,
                                    batch_size=batch_size,
                                    mixed_input=mixed_input,
                                    clean_data=clean_data,
                                    num_shared_layers=num_shared_layers,
                                    num_separate_layers=num_separate_layers,
                                    max_time_num=time_num,
                                    max_time=max_time,
                                    num_subepochs=150,
                                )

    if True:
        for clean_text in [True, False]:
            for mixed_input in [True, False]:
                for clean_data in [True, False]:
                    name = f"tn-2_mt-15_6l"
                    if clean_text:
                        name += "_clean-text"
                    if mixed_input:
                        name += "_mixed-input-4l"
                    if clean_data:
                        name += "_clean-data"
                    run_exp(
                        name_suffix=name,
                        learning_rate=4e-04,
                        max_pool=[1, 2, 2],
                        specaug=True,
                        batch_size=15000,
                        mixed_input=mixed_input,
                        clean_data=clean_data,
                        num_shared_layers=4,
                        num_separate_layers=6,
                        max_time_num=2,
                        max_time=15,
                        num_subepochs=150,
                    )
