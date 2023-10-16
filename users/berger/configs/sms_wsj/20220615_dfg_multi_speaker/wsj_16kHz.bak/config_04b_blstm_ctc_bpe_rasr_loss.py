import os
import copy

from sisyphus import gs, tk

import i6_core.corpus as corpus_recipe
import i6_core.rasr as rasr
import i6_core.text as text
from i6_core.lib import corpus
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_core.tools import CloneGitRepositoryJob

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.corpus.sms_wsj.data import (
    get_data_inputs,
    PreprocessWSJTranscriptionsJob,
    PreprocessWSJLexiconJob,
)
from i6_experiments.users.berger.systems.transducer_system import TransducerSystem
from i6_experiments.users.berger.args.jobs.hybrid_args import get_nn_args
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.network.models.fullsum_ctc import (
    make_blstm_fullsum_ctc_model,
    make_blstm_ctc_recog_model,
)
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args
from i6_experiments.users.berger.args.jobs.data import get_returnn_rasr_data_input
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_experiments.users.berger.recipe.lexicon.bpe_lexicon import CreateBPELexiconJob


# ********** Settings **********

dir_handle = os.path.dirname(__file__).split("config/")[1]
filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

train_key = "train_si284"
dev_key = "cv_dev93"
test_key = "test_eval92"

cv_segments = tk.Path("/work/asr4/berger/dependencies/sms_wsj/segments/cv_dev93_16kHz.reduced")

frequency = 16

f_name = "gt"

num_inputs = 50


def run_exp(**kwargs):
    am_args = {
        "state_tying": "monophone",
        "states_per_phone": 1,
        "phon_history_length": 0,
        "phon_future_length": 0,
        "tdp_transition": (0.0, 0.0, "infinity", 0.0),
        "tdp_silence": (0.0, 0.0, "infinity", 0.0),
        "tdp_nonword": (0.0, 0.0, "infinity", 0.0),
        "tdp_scale": 1.0,
    }

    # ********** Init args **********

    train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs(
        train_keys=[train_key],
        dev_keys=[dev_key],
        test_keys=[test_key],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        delete_empty_orth=True,
    )

    # initialize feature system and extract features
    init_args = get_init_args(sample_rate_kHz=frequency)
    init_args.feature_extraction_args = {
        f_name: init_args.feature_extraction_args[f_name]
    }  # only keep fname and discard other features

    feature_system = gmm_system.GmmSystem(rasr_binary_path=None)
    feature_system.init_system(
        rasr_init_args=init_args,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    feature_system.run(["extract"])

    train_corpus_object = train_data_inputs[train_key].corpus_object
    train_corpus_object.corpus_file = PreprocessWSJTranscriptionsJob(train_corpus_object.corpus_file).out_corpus_file
    dev_corpus_object = dev_data_inputs[dev_key].corpus_object
    dev_corpus_object.corpus_file = PreprocessWSJTranscriptionsJob(dev_corpus_object.corpus_file).out_corpus_file

    # ********** Data inputs **********

    nn_data_inputs = {}

    nn_data_inputs["train"] = {
        f"{train_key}.train": get_returnn_rasr_data_input(
            train_data_inputs[train_key],
            feature_flow=feature_system.feature_flows[train_key][f_name],
            features=feature_system.feature_caches[train_key][f_name],
            am_args=am_args,
            shuffle_data=True,
            concurrent=1,
        )
    }

    nn_data_inputs["train"][f"{train_key}.train"].crp.lexicon_config.file = PreprocessWSJLexiconJob(
        nn_data_inputs["train"][f"{train_key}.train"].crp.lexicon_config.file
    ).out_lexicon_file

    nn_data_inputs["cv"] = {
        f"{train_key}.cv": get_returnn_rasr_data_input(
            copy.deepcopy(dev_data_inputs[dev_key]),
            feature_flow=feature_system.feature_flows[dev_key][f_name],
            features=feature_system.feature_caches[dev_key][f_name],
            segment_path=cv_segments,
            am_args=am_args,
            shuffle_data=False,
            concurrent=1,
        )
    }

    nn_data_inputs["cv"][f"{train_key}.cv"].crp.lexicon_config.file = nn_data_inputs["train"][
        f"{train_key}.train"
    ].crp.lexicon_config.file

    nn_data_inputs["dev"] = {
        dev_key: get_returnn_rasr_data_input(
            dev_data_inputs[dev_key],
            feature_flow=feature_system.feature_flows[dev_key][f_name],
            features=feature_system.feature_caches[dev_key][f_name],
            am_args=am_args,
            shuffle_data=False,
        )
    }
    nn_data_inputs["test"] = {
        test_key: get_returnn_rasr_data_input(
            test_data_inputs[test_key],
            feature_flow=feature_system.feature_flows[test_key][f_name],
            features=feature_system.feature_caches[test_key][f_name],
            am_args=am_args,
            shuffle_data=False,
        )
    }

    # ********** Transducer System **********

    bpe_txt = text.PipelineJob(
        tk.Path("/u/corpora/language/wsj/NAB-training-corpus.gz", cached=True),
        ['sed "s/ \S*$//"', 'sed "s/^\S* //"'],
        mini_task=True,
    ).out  # Remove <s> and </s> tokens
    subword_nmt_repo = CloneGitRepositoryJob("https://github.com/albertz/subword-nmt.git").out_repository

    train_bpe_job = ReturnnTrainBpeJob(bpe_txt, kwargs.get("bpe_size", 1000), subword_nmt_repo=subword_nmt_repo)
    bpe_codes = train_bpe_job.out_bpe_codes
    bpe_vocab = train_bpe_job.out_bpe_vocab

    num_classes = train_bpe_job.out_vocab_size  # bpe count
    num_classes_b = num_classes + 1  # bpe count + blank

    for data_input in [
        nn_data_inputs["train"][f"{train_key}.train"],
        nn_data_inputs["cv"][f"{train_key}.cv"],
        nn_data_inputs["dev"][dev_key],
        nn_data_inputs["test"][test_key],
    ]:
        data_input.crp.lexicon_config.file = CreateBPELexiconJob(
            base_lexicon_path=data_input.crp.lexicon_config.file,
            bpe_codes=bpe_codes,
            bpe_vocab=bpe_vocab,
            subword_nmt_repo=subword_nmt_repo,
        ).out_lexicon

    system = TransducerSystem(rasr_binary_path=None)
    system.init_system(
        rasr_init_args=init_args,
        train_data=nn_data_inputs["train"],
        cv_data=nn_data_inputs["cv"],
        dev_data=nn_data_inputs["dev"],
        test_data=nn_data_inputs["test"],
        train_cv_pairing=[(f"{train_key}.train", f"{train_key}.cv")],
    )

    train_networks = {}
    recog_networks = {}

    name = "_".join(filter(None, ["BLSTM_CTC", kwargs.get("name_suffix", "")]))

    max_pool = kwargs.get("max_pool", [1, 2, 2])
    red_fact = 1
    for fact in max_pool:
        red_fact *= fact
    train_blstm_net, train_python_code = make_blstm_fullsum_ctc_model(
        num_outputs=num_classes_b,
        specaug_args={
            "max_time_num": 1,
            "max_time": 15,
            "max_feature_num": 5,
            "max_feature": 5,
        },
        blstm_args={
            "max_pool": max_pool,
            "num_layers": 6,
            "size": 400,
            "dropout": kwargs.get("dropout", 0.1),
            "l2": kwargs.get("l2", 5e-06),
        },
        mlp_args={
            "num_layers": kwargs.get("num_mlp_layers", 0),
            "size": 600,
            "dropout": kwargs.get("dropout", 0.1),
            "l2": kwargs.get("l2", 5e-06),
        },
    )
    # train_blstm_net["output_print"] = {
    #     "class": "print",
    #     "from": "output",
    #     "is_output_layer": True,
    #     "summarize": 50,
    # }
    train_networks[name] = train_blstm_net

    recog_blstm_net, recog_python_code = make_blstm_ctc_recog_model(
        num_outputs=num_classes_b,
        blstm_args={
            "max_pool": max_pool,
            "num_layers": 6,
            "size": 400,
        },
        mlp_args={
            "num_layers": kwargs.get("num_mlp_layers", 0),
            "size": 600,
        },
    )
    recog_networks[name] = recog_blstm_net

    num_subepochs = kwargs.get("num_subepochs", 300)

    nn_args = get_nn_args(
        train_networks=train_networks,
        recog_networks=recog_networks,
        num_inputs=num_inputs,
        num_outputs=num_classes_b,
        num_epochs=num_subepochs,
        search_type=SearchTypes.LabelSyncSearch,
        returnn_train_config_args={
            "extra_python": train_python_code,
            "grad_noise": kwargs.get("grad_noise", 0.0),
            "grad_clip": kwargs.get("grad_clip", 100.0),
            "batch_size": kwargs.get("batch_size", 15000),
            "schedule": kwargs.get("schedule", LearningRateSchedules.OCLR),
            "peak_lr": kwargs.get("peak_lr", 1e-3),
            "learning_rate": kwargs.get("learning_rate", 1e-03),
            "n_steps_per_epoch": 1100,
            "use_chunking": False,
        },
        recog_name=kwargs.get("recog_name", None),
        returnn_recog_config_args={"extra_python": recog_python_code},
        train_args={
            "partition_epochs": 3,
            "log_verbosity": 4,
            "use_rasr_ctc_loss": True,
            "rasr_ctc_loss_args": {"allow_label_loop": True},
        },
        prior_args={
            "num_classes": num_classes_b,
            "mem_rqmt": 6.0,
        },
        recog_args={
            "epochs": [num_subepochs] if kwargs.get("recog_final_epoch_only", False) else None,
            "lm_scales": kwargs.get("lm_scales", [0.7]),
            "prior_scales": kwargs.get("prior_scales", [0.0]),
            "blank_penalty": kwargs.get("blank_penalty", 0.0),
            "use_gpu": False,
            "label_unit": "phoneme",
            "add_eow": False,
            "allow_blank": True,
            "allow_loop": True,
            "label_scorer_type": "precomputed-log-posterior",
            "lp": 15.0,
            "label_scorer_args": {
                "use_prior": False,
                "num_classes": num_classes_b,
                "extra_args": {
                    "blank_label_index": 0,
                    "reduction_factors": red_fact,
                },
            },
            "label_tree_args": {
                "use_transition_penalty": False,
                "skip_silence": True,
            },
        },
    )

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("nn", nn_args)
    nn_steps.add_step("nn_recog", nn_args)

    system.run(nn_steps)


def py():
    for bp in [0.0, 0.6]:
        name_suffix = f"newbob_lr-0.0004_bpe-100"
        recog_name = f"bp-{bp}"
        run_exp(
            name_suffix=name_suffix,
            recog_name=recog_name,
            schedule=LearningRateSchedules.Newbob,
            learning_rate=0.0004,
            bpe_size=100,
            blank_penalty=bp,
            lm_scales=[0.6, 0.8, 1.0, 1.2, 1.4],
            recog_final_epoch_only=True,
        )
