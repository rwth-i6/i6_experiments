import copy
from functools import partial
import os
from i6_core.corpus.convert import CorpusToStmJob

import i6_core.rasr as rasr
from i6_core.recognition.scoring import ScliteJob
from i6_core.returnn.config import CodeWrapper
from i6_core.returnn.extract_prior import ReturnnComputePriorJob
import i6_core.text as text
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.search import (
    ReturnnSearchJobV2,
    SearchBPEtoWordsJob,
    SearchWordsToCTMJob,
)
from i6_experiments.users.berger.args.returnn.config import (
    get_network_config,
    get_returnn_config,
)
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_experiments.users.berger.corpus.sms_wsj.data import get_bpe, get_data_inputs
from i6_experiments.users.berger.network.models.fullsum_ctc_dual_output import (
    make_blstm_fullsum_ctc_dual_output_model,
    make_blstm_fullsum_ctc_dual_output_recog_model,
)
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.transducer_system import SummaryKey
from sisyphus import gs, tk
from ..lm.config_01_lstm_bpe import py as run_lm


# ********** Settings **********

dir_handle = os.path.dirname(__file__).split("config/")[1]
filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

train_key = "sms_train_si284"
dev_key = "sms_cv_dev93"
test_key = "sms_test_eval92"

frequency = 8

f_name = "gt"

num_inputs = 40

bpe_size = 100


def run_exp(lm_model: tk.Path, **kwargs) -> SummaryReport:

    lm_cleaning = kwargs.get("lm_cleaning", False)

    # ********** Summary Report **********

    summary_report = SummaryReport(
        [
            key.value
            for key in [
                SummaryKey.NAME,
                SummaryKey.CORPUS,
                SummaryKey.EPOCH,
                SummaryKey.PRIOR,
                SummaryKey.LM,
                SummaryKey.WER,
                SummaryKey.SUB,
                SummaryKey.DEL,
                SummaryKey.INS,
                SummaryKey.ERR,
            ]
        ],
        col_sort_key=SummaryKey.ERR.value,
    )

    # ********** Get BPEs **********

    bpe_job = get_bpe(size=bpe_size, lm_cleaning=lm_cleaning)

    num_classes = bpe_job.out_vocab_size  # bpe count
    num_classes_b = num_classes + 1  # bpe count + blank

    # ********** Extern data **********

    train_data_inputs, dev_data_inputs, test_data_inputs, _ = get_data_inputs(
        train_keys=[train_key],
        dev_keys=[dev_key],
        test_keys=[test_key],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        delete_empty_orth=True,
        lm_cleaning=lm_cleaning,
    )

    datasets = {
        train_key: {
            "class": "HDFDataset",
            "files": [
                f"/u/berger/asr-exps/sms_wsj/20220615_dfg_multi_speaker/dependencies/hdf/8kHz/sms_train_si284_complete.gt40.bpe-100.{'updated.' if lm_cleaning else ''}hdf"
            ],
            "use_cache_manager": False,
            "seq_ordering": "random",
            "partition_epoch": 3,
        },
        dev_key: {
            "class": "HDFDataset",
            "files": [
                f"/u/berger/asr-exps/sms_wsj/20220615_dfg_multi_speaker/dependencies/hdf/8kHz/sms_cv_dev93_complete.gt40.bpe-100.{'updated.' if lm_cleaning else ''}hdf"
            ],
            "use_cache_manager": False,
            "seq_ordering": "sorted",
            "partition_epoch": 1,
        },
        test_key: {
            "class": "HDFDataset",
            "files": [
                f"/u/berger/asr-exps/sms_wsj/20220615_dfg_multi_speaker/dependencies/hdf/8kHz/sms_test_eval92_complete.gt40.bpe-100.{'updated.' if lm_cleaning else ''}hdf"
            ],
            "use_cache_manager": False,
            "seq_ordering": "sorted",
            "partition_epoch": 1,
        },
    }

    extern_data_config = {
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
            "bpe_b": {"dim": num_classes_b, "sparse": True},
            "bpe_0": {"dim": num_classes, "sparse": True},
            "bpe_1": {"dim": num_classes, "sparse": True},
        },
        "num_outputs": {
            "bpe_0": num_classes_b,
            "bpe_1": num_classes_b,
        },
    }

    extern_data_config_recog = copy.deepcopy(extern_data_config)
    for key in [
        "data",
        "data_separated_0",
        "data_separated_1",
        "data_clean_0",
        "data_clean_1",
    ]:
        extern_data_config_recog["extern_data"][key]["available_for_inference"] = True

    # ********** Training setup **********

    name = "_".join(filter(None, ["BLSTM_CTC_dual", kwargs.get("name_suffix", "")]))
    max_pool_pre = kwargs.get("max_pool_pre", [1, 1, 2])
    max_pool_post = kwargs.get("max_pool_post", [2])

    train_blstm_net = {}

    if kwargs.get("clean_data", False):
        from_0 = "data:data_clean_0"
        from_1 = "data:data_clean_1"
    else:
        from_0 = "data:data_separated_0"
        from_1 = "data:data_separated_1"

    l2 = kwargs.get("l2", 5e-06)
    dropout = kwargs.get("dropout", 0.1)

    train_blstm_net, train_python_code = make_blstm_fullsum_ctc_dual_output_model(
        num_outputs=num_classes_b,
        from_0=from_0,
        from_1=from_1,
        target_key_0="bpe_0",
        target_key_1="bpe_1",
        from_mix="data",
        specaug_01_args={
            "max_time_num": kwargs.get("max_time_num", 2),
            "max_time": kwargs.get("max_time", 15),
            "max_feature_num": 4,
            "max_feature": 5,
        },
        blstm_01_args={
            "num_layers": kwargs.get("enc_01_layers", 4),
            "size": 400,
            "max_pool": max_pool_pre,
            "dropout": dropout,
            "l2": l2,
        },
        blstm_mix_args={
            "num_layers": kwargs.get("enc_mix_layers", 4),
            "size": 400,
            "max_pool": max_pool_pre,
            "dropout": dropout,
            "l2": l2,
        },
        blstm_01_mix_args={
            "num_layers": kwargs.get("enc_01_mix_layers", 2),
            "size": 400,
            "max_pool": max_pool_post,
            "dropout": dropout,
            "l2": l2,
        },
    )

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
            "train": datasets[train_key],
            "dev": datasets[dev_key],
            **extern_data_config,
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

    # ********** Prior computation **********

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

    prior_config = copy.deepcopy(train_config)
    prior_config.config.update(get_network_config(prior_net))
    prior_config.config.update({"forward_output_layer": "output"})

    prior_job = ReturnnComputePriorJob(
        model_checkpoint=train_job.out_checkpoints[num_subepochs],
        returnn_config=prior_config,
        log_verbosity=4,
        mem_rqmt=8,
    )

    # ********** Recognition **********

    lm_scale = kwargs.get("lm_scale", 1.1)
    prior_scale = kwargs.get("prior_scale", 0.3)

    recog_blstm_net, recog_python_code = make_blstm_fullsum_ctc_dual_output_recog_model(
        num_outputs=num_classes_b,
        from_0=from_0,
        from_1=from_1,
        target_key_0="bpe_0",
        target_key_1="bpe_1",
        from_mix="data",
        blstm_01_args={
            "num_layers": kwargs.get("enc_01_layers", 4),
            "size": 400,
            "max_pool": max_pool_pre,
        },
        blstm_mix_args={
            "num_layers": kwargs.get("enc_mix_layers", 4),
            "size": 400,
            "max_pool": max_pool_pre,
        },
        blstm_01_mix_args={
            "num_layers": kwargs.get("enc_01_mix_layers", 2),
            "size": 400,
            "max_pool": max_pool_post,
        },
        lm_path=lm_model,
        lm_scale=lm_scale,
        lm_args={
            "embedding_args": {
                "size": 256,
            },
            "lstm_args": {
                "num_layers": 2,
                "size": 2048,
            },
        },
        prior_path=prior_job.out_prior_txt_file,
        prior_scale=prior_scale,
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
            "search_output_layer": ["ctc_decode_0", "ctc_decode_1"],
            **extern_data_config_recog,
        },
    )

    for recog_key in [dev_key, test_key]:

        search_job = ReturnnSearchJobV2(
            search_data=datasets[recog_key],
            model_checkpoint=train_job.out_checkpoints[num_subepochs],
            returnn_config=recog_config,
            output_mode="py",
            log_verbosity=5,
            returnn_python_exe=tk.Path(gs.RETURNN_PYTHON_EXE),
            returnn_root=tk.Path(gs.RETURNN_ROOT),
        )

        out_path = f"nn_recog/{name}/{recog_key}_lm-{lm_scale:01.02f}_prior-{prior_scale:01.02f}_ep-{num_subepochs:03d}"
        search_job.add_alias(out_path)

        # ********** Scoring **********

        words_job = SearchBPEtoWordsJob(search_job.out_search_file)

        words_job_processed = text.PipelineJob(
            words_job.out_word_search_results,
            [
                'sed "s/\/0000_ctc_decode_0/_0\/0000/"',
                'sed "s/\/0000_ctc_decode_1/_1\/0000/"',
            ],
            mini_task=True,
        )

        if recog_key == dev_key:
            bliss_corpus = dev_data_inputs[recog_key].corpus_object.corpus_file
        else:
            bliss_corpus = test_data_inputs[recog_key].corpus_object.corpus_file

        word2ctm_job = SearchWordsToCTMJob(words_job_processed.out, bliss_corpus)
        scorer_job = ScliteJob(
            CorpusToStmJob(bliss_corpus, non_speech_tokens=["<NOISE>"]).out_stm_path,
            word2ctm_job.out_ctm_file,
        )

        tk.register_output(f"{out_path}.wer", scorer_job.out_report_dir)

        summary_report.add_row(
            {
                SummaryKey.NAME.value: name,
                SummaryKey.CORPUS.value: recog_key,
                SummaryKey.EPOCH.value: num_subepochs,
                SummaryKey.PRIOR.value: prior_scale,
                SummaryKey.LM.value: lm_scale,
                SummaryKey.WER.value: scorer_job.out_wer,
                SummaryKey.SUB.value: scorer_job.out_percent_substitution,
                SummaryKey.DEL.value: scorer_job.out_percent_deletions,
                SummaryKey.INS.value: scorer_job.out_percent_insertions,
                SummaryKey.ERR.value: scorer_job.out_num_errors,
            }
        )

    return summary_report


def py() -> SummaryReport:

    cleaned_text_lm = run_lm(lm_cleaning=True)
    lm = run_lm(lm_cleaning=False)

    run_exp_clean_partial = partial(run_exp, cleaned_text_lm, lm_cleaning=True)
    run_exp_partial = partial(run_exp, lm, lm_cleaning=False)

    dir_handle = os.path.dirname(__file__).split("config/")[1]
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"

    summary_report = SummaryReport()

    # All encoder types
    summary_report.merge_report(
        run_exp_clean_partial(),
        update_structure=True,
    )

    for enc01, encmix, enc01mix in [(0, 0, 6), (6, 4, 0)]:
        summary_report.merge_report(
            run_exp_clean_partial(
                name_suffix=f"enc01-{enc01}_encmix-{encmix}_enc01mix-{enc01mix}",
                enc_01_layers=enc01,
                enc_mix_layers=encmix,
                enc_01_mix_layers=enc01mix,
                max_pool_pre=[1, 2, 2],
                max_pool_post=[1, 2, 2],
            ),
        )
