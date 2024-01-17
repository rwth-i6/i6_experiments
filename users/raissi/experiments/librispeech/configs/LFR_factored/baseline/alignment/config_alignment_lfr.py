import os, copy
import dataclasses
import itertools
from IPython import embed

# -------------------- Sisyphus --------------------

from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.corpus as corpus_recipe
import i6_core.rasr as rasr
import i6_core.returnn as returnn
import i6_core.lexicon as lexicon


import i6_experiments.common.setups.rasr.util as rasr_util
import i6_experiments.users.raissi.experiments.librispeech.data_preparation.common.base_args as lbs_data_setups
import i6_experiments.users.raissi.experiments.librispeech.data_preparation.other_960h.pipeline_base_args as lbs_960_setups
import i6_experiments.users.raissi.utils.default_tools as run_tools
import i6_experiments.users.raissi.setups.common.helpers.train as train_helpers

from i6_experiments.users.raissi.setups.common.analysis import PlotViterbiAlignmentsJob, ComputeWordLevelTimestampErrorJob

from i6_experiments.users.raissi.setups.common.data.factored_label import RasrStateTying

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    PhoneticContext,
)
from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import (
    TrainingCriterion,
)
from i6_experiments.users.raissi.setups.common.TF_factored_hybrid_system import (
    TFFactoredHybridBaseSystem,
)

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    RasrFeatureScorer,
)
from i6_experiments.users.raissi.utils.general_helpers import load_pickle

from i6_experiments.users.raissi.setups.common.helpers.network.augment import (
    add_fast_bw_layer_to_returnn_config,
    augment_net_with_monophone_outputs,
    LogLinearScales
)

from i6_experiments.users.raissi.setups.librispeech.train.parameters import (
    get_initial_nn_args_fullsum,
)

from i6_experiments.users.raissi.experiments.librispeech.configs.LFR_factored.baseline.config import (
    BLSTM_FH_DECODING_TENSOR_CONFIG_TF2,
    ALIGN_GMM_TRI_ALLOPHONES,
    ALIGN_GMM_TRI_10MS,
    ZHOU_SUBSAMPLED_ALIGNMENT,
    ZHOU_ALLOPHONES
)


def get_system(key, lr=4e-4, num_epochs=None, am_scale=1.0, tdp_scale=0.1):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ***********Initial arguments and steps ********************
    # pipeline related
    corpus_data = lbs_data_setups.get_corpus_data_inputs(corpus_key="train-other-960")
    rasr_init_args = lbs_data_setups.get_init_args()
    label_info_init_args = {
        "ph_emb_size": 0,
        "st_emb_size": 0,
        "state_tying": RasrStateTying.monophone,
        "n_states_per_phone": 1
    }
    init_args_system = {
        "label_info": label_info_init_args,
        "frr_info": {"factor": 4},
    }
    data_preparation_args = lbs_960_setups.get_final_output(name="data_preparation")

    # NN training related
    initial_nn_args = get_initial_nn_args_fullsum()
    num_epochs = num_epochs or initial_nn_args["num_epochs"]
    if "num_epochs" in initial_nn_args:
        initial_nn_args.pop("num_epochs")
    initial_nn_args["keep_epochs"] = [450] + list(
        range(num_epochs - 10, num_epochs + 1)
    )
    initial_nn_args["keep_best_n"] = 2

    partition_epochs = initial_nn_args.pop("partition_epochs")
    hyper_params = train_helpers.default_blstm_fullsum

    steps = rasr_util.RasrSteps()
    steps.add_step("init", init_args_system)
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("input", data_preparation_args)

    # *********** System Instantiation *****************
    s = TFFactoredHybridBaseSystem(
        returnn_root=run_tools.u16_default_tools.returnn_root,
        returnn_python_exe=run_tools.u16_default_tools.returnn_python_exe,
        rasr_binary_path=run_tools.u16_default_tools.rasr_binary_path,
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
        initial_nn_args=initial_nn_args,
    )
    # setting up parameters for full-sum
    s.training_criterion = TrainingCriterion.fullsum
    s.partition_epochs = partition_epochs

    #specific to full-sum
    s.lexicon_args["norm_pronunciation"] = False
    s.shuffling_params["segment_order_sort_by_time_length_chunk_size"] = 1044

    train_args = copy.deepcopy(s.initial_train_args)
    train_args["num_epochs"] = num_epochs

    s.run(steps)
    s.set_crp_pairings(dev_key="dev-other", test_key="test-other")
    s.set_rasr_returnn_input_datas(
        input_key="data_preparation",
        is_cv_separate_from_train=True,
        cv_corpus_key="dev-other"
    )
    s.update_am_setting_for_all_crps(
        train_tdp_type="heuristic", eval_tdp_type="heuristic"
    )
    exp_name = f'am{am_scale}-t{tdp_scale}'
    s.set_experiment_dict(key=key, alignment="scratch", context="mono", postfix_name=exp_name)
    # ----------------------------- train -----------------------------------------------------
    blstm_args = {"spec_aug_as_data": True, "l2": hyper_params.l2}
    network = s.get_blstm_network(**blstm_args)
    mono_network = augment_net_with_monophone_outputs(
        shared_network=network,
        frame_rate_reduction_ratio_info=s.frame_rate_reduction_ratio_info,
        encoder_output_len=hyper_params.encoder_out_len,
        l2=hyper_params.l2,
        label_info=s.label_info,
        add_mlps=hyper_params.add_mlps,
        use_multi_task=hyper_params.use_multi_task,
    )
    mono_network["center-output"]["n_out"] = s.label_info.get_n_state_classes()

    returnn_config_params = train_helpers.get_base_returnn_dict_v2()
    base_config = {
        **s.initial_nn_args,
        **train_helpers.oclr.get_oclr_config(num_epochs=num_epochs, lrate=lr),
        **returnn_config_params,
        "batch_size": 10000,
        "network": mono_network,
    }

    return_config_dict = s.get_config_with_legacy_prolog_and_epilog(
        config=base_config,
        epilog_additional_str=train_helpers.specaugment.get_legacy_specaugment_epilog_blstm(
            t_num=1, t=15, f_num=5, f=5
        ),
    )

    s.set_returnn_config_for_experiment(key=key, config_dict=return_config_dict)

    bw_crp = s.crp[s.crp_names["bw"]]
    log_linear_scales = LogLinearScales.default()
    log_linear_scales = dataclasses.replace(log_linear_scales, label_posterior_scale=am_scale, transition_scale=tdp_scale)

    bw_augmented_returnn_config = add_fast_bw_layer_to_returnn_config(
        crp=bw_crp,
        returnn_config=s.experiments[key]["returnn_config"],
        log_linear_scales=log_linear_scales
    )
    s.experiments[key]["returnn_config"] = bw_augmented_returnn_config

    s.returnn_rasr_training_fullsum(
        experiment_key=key,
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
    )

    return_config_dict_infer = s.get_config_with_legacy_prolog_and_epilog(
        config=s.experiments[key]["returnn_config"].config,
        epilog_additional_str=train_helpers.specaugment.get_legacy_specaugment_epilog_blstm(
            t_num=1, t=15, f_num=5, f=5
        ),
        add_extern_data_for_fullsum=True)
    s.set_returnn_config_for_experiment(key=key, config_dict=return_config_dict_infer)


    s.set_single_prior_returnn_rasr(
        key=key,
        epoch=450,
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        data_share=0.3,
        context_type=PhoneticContext.monophone,
        smoothen=True,
        output_layer_name="center-output"
    )


    aligner, cfg = s.get_aligner_and_args(
        key=key,
        context_type=PhoneticContext.monophone,
        feature_scorer_type=RasrFeatureScorer.nn_precomputed,
        crp_corpus=s.crp_names['align.train'],
        epoch=500,
        tf_library=s.native_lstm2_path,
        tensor_map=BLSTM_FH_DECODING_TENSOR_CONFIG_TF2,
        set_batch_major_for_feature_scorer=False
    )
    #the tdps are adjusted for factor 4
    align_cfg = (
        cfg.with_prior_scale(center=0.4)
        .with_tdp_scale(1.0)
    )
    alignment_j = aligner.get_alignment_job(
        label_info=s.label_info,
        alignment_parameters=align_cfg,
        num_encoder_output=hyper_params.encoder_out_len,

    )
    s.experiments[key]["align_job"] = alignment_j

    return s




get_system(key='exp2', am_scale=1.0, tdp_scale=0.1)
#WER 9.0, TSE 57ms

#get_system(key='exp3', am_scale=0.5, tdp_scale=0.1)
#WER 9.2, TSE 129ms

#get_system(key='exp4', am_scale=0.5, tdp_scale=0.3)
#WER 9.5, TSE 121ms



