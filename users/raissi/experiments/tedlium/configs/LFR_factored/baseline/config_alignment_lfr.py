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
from i6_experiments.users.raissi.setups.tedlium.TED_TF_factored_hybrid_system import (
    TEDTFFactoredHybridSystem,
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

from i6_experiments.users.raissi.setups.tedlium.train.parameters import (
    get_initial_nn_args_fullsum,
)

from i6_experiments.users.raissi.experiments.librispeech.configs.LFR_factored.baseline.config import (
    BLSTM_FH_DECODING_TENSOR_CONFIG_TF2,
)

import i6_experiments.users.raissi.experiments.tedlium.data_preparation.pipeline_base_args as ted_setups


#ALIGNMENT
from i6_experiments.users.berger.recipe.mm.alignment import ComputeTSEJob
from i6_experiments.users.raissi.setups.tedlium.config import (
    TRI_GMM_ALIGNMENT,
    GMM_ALLOPHONES
)

def run(key, lr=4e-4, am_scale=0.7, tdp_scale=0.1, align=False, tune=False,
        decode=False, decode_corpus="dev"):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ***********Initial arguments and steps ********************
    # pipeline related
    rasr_init_args = ted_setups.get_init_args()
    corpus_data = ted_setups.get_corpus_data_inputs()
    label_info_init_args = {
        "ph_emb_size": 0,
        "st_emb_size": 0,
        "state_tying": RasrStateTying.monophone,
        "n_states_per_phone": 1,
    }
    init_args_system = {
        "label_info": label_info_init_args,
        "frr_info": {"factor": 4},
    }
    data_preparation_args = ted_setups.get_final_output(name="data_preparation")

    # NN training related
    initial_nn_args = get_initial_nn_args_fullsum()
    num_epochs = initial_nn_args.pop("num_epochs")
    initial_nn_args["keep_epochs"] = [50, 100, 150, 200]
    hyper_params = train_helpers.default_blstm_fullsum

    steps = rasr_util.RasrSteps()
    steps.add_step("init", init_args_system)
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("input", data_preparation_args)

    # *********** System Instantiation *****************
    s = TEDTFFactoredHybridSystem(
        returnn_root=run_tools.RETURNN_ROOT_TORCH,
        returnn_python_exe=run_tools.u16_default_tools.returnn_python_exe,
        rasr_binary_path=run_tools.u16_default_tools.rasr_binary_path,
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
        initial_nn_args=initial_nn_args,
    )
    # setting up parameters for full-sum
    s.training_criterion = TrainingCriterion.FULLSUM
    s.lexicon_args["norm_pronunciation"] = False
    s.shuffling_params["segment_order_sort_by_time_length_chunk_size"] = 1000

    train_args = copy.deepcopy(s.initial_train_args)
    train_args["num_epochs"] = num_epochs

    s.run(steps)
    s.set_crp_pairings(dev_key="dev", test_key="test")
    s.set_rasr_returnn_input_datas(
        input_key="data_preparation",
        is_cv_separate_from_train=True,
        cv_corpus_key="dev"
    )
    s.update_am_setting_for_all_crps(
        train_tdp_type="heuristic-40ms", eval_tdp_type="heuristic-40ms"
    )
    exp_name = f'lr{lr}-am{am_scale}-t{tdp_scale}'
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
            t_num=1, t=15, f_num=5, f=10
        ),
    )
    s.set_returnn_config_for_experiment(key=key, config_dict=return_config_dict)

    bw_crp = s.crp[s.crp_names["bw"]]
    log_linear_scales = LogLinearScales.default()
    log_linear_scales = dataclasses.replace(
        log_linear_scales, label_posterior_scale=am_scale, transition_scale=tdp_scale
    )

    bw_augmented_returnn_config = add_fast_bw_layer_to_returnn_config(
        crp=bw_crp,
        returnn_config=s.experiments[key]["returnn_config"],
        log_linear_scales=log_linear_scales,
    )
    s.experiments[key]["returnn_config"] = bw_augmented_returnn_config

    s.returnn_rasr_training_fullsum(
        experiment_key=key,
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
    )

    if decode:
        return_config_dict_infer = s.get_config_with_legacy_prolog_and_epilog(
            config=s.experiments[key]["returnn_config"].config,
            epilog_additional_str=train_helpers.specaugment.get_legacy_specaugment_epilog_blstm(
                t_num=1, t=15, f_num=5, f=5
            ),
            add_extern_data_for_fullsum=True,
        )
        s.set_returnn_config_for_experiment(
            key=key, config_dict=return_config_dict_infer
        )

        s.set_single_prior_returnn_rasr(
            key=key,
            epoch=s.initial_nn_args["keep_epochs"][0] if tune else 100, 
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            data_share=0.3 if tune else 0.1,
            context_type=PhoneticContext.monophone,
            smoothen=True,
            output_layer_name="center-output",
        )
        for prior, epoch in itertools.product([0.4], [100, 150, 200, 249]):
            recognizer, recog_args = s.get_recognizer_and_args(
                key=key,
                context_type=PhoneticContext.monophone,
                feature_scorer_type=RasrFeatureScorer.nn_precomputed,
                crp_corpus=decode_corpus,
                epoch=epoch,
                gpu=False,
                tensor_map=BLSTM_FH_DECODING_TENSOR_CONFIG_TF2,
                lm_gc_simple_hash=True,
                tf_library=s.native_lstm2_path,
            )
            cfg = (
                recog_args.with_prior_scale(center=prior)
                .with_tdp_scale(0.1)
                .with_lm_scale(1.14)
            )

            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=cfg,
                num_encoder_output=hyper_params.encoder_out_len,
                rerun_after_opt_lm=False if tune else True,
                calculate_stats=False,
            )
            if tune and epoch > 230:
                tune_args = cfg.with_beam_size(18.0)
                best_config = recognizer.recognize_optimize_scales(
                    label_info=s.label_info,
                    search_parameters=tune_args,
                    num_encoder_output=hyper_params.encoder_out_len,
                    altas_value=2.0,
                    tdp_sil=[(s_t, 0.0, "infinity", 20.0) for s_t in [12.0]],
                    tdp_speech=[(s_t, 0.0, "infinity", 0.0) for s_t in [8.0, 10.0]],
                    prior_scales=[[0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]],
                    tdp_scales=[0.1],
                )
                recognizer.recognize_count_lm(
                    label_info=s.label_info,
                    search_parameters=dataclasses.replace(
                        best_config, we_pruning=0.6, beam=22.0, altas=None
                    ),
                    num_encoder_output=hyper_params.encoder_out_len,
                    rerun_after_opt_lm=True,
                    calculate_stats=False,
                    name_override="best/4gram",
                    pre_path="decoding/tuned",
                )

                if align:
                    aligner, cfg = s.get_aligner_and_args(
                        key=key,
                        context_type=PhoneticContext.monophone,
                        feature_scorer_type=RasrFeatureScorer.nn_precomputed,
                        crp_corpus=s.crp_names["align.train"],
                        epoch=250,
                        tf_library=s.native_lstm2_path,
                        tensor_map=BLSTM_FH_DECODING_TENSOR_CONFIG_TF2,
                        set_batch_major_for_feature_scorer=False,
                    )

                    prior_scale = best_config.prior_info.center_state_prior.scale
                    # the exit penalty is corrected in the get_alignment_job below
                    sp_tdp = (0.0, 3.0, "infinity", 0.0)
                    sil_tdp = (best_config.tdp_silence[0]+3.0, 0.0, "infinity", 0.0)

                    align_cfg = (
                        cfg.with_prior_scale(center=prior_scale)
                        .with_tdp_speech(sp_tdp)
                        .with_tdp_silence(sil_tdp)
                    )
                    assert (
                        align_cfg.tdp_scale == 1.0
                    ), "Do not scale the tdp values during alignment"

                    alignment_j = aligner.get_alignment_job(
                        label_info=s.label_info,
                        alignment_parameters=align_cfg,
                        num_encoder_output=hyper_params.encoder_out_len,
                    )

                    allophones = lexicon.StoreAllophonesJob(
                        s.crp[s.crp_names["align.train"]]
                    )

                    tse_job =  ComputeTSEJob(
                        alignment_cache=alignment_j.out_alignment_bundle ,
                        ref_alignment_cache=TRI_GMM_ALIGNMENT,
                        allophone_file=allophones.out_allophone_file,
                        ref_allophone_file=GMM_ALLOPHONES,
                        upsample_factor = 4,
                    )
                    tk.register_output("statistics/alignment/tse_out", tse_job.out_tse_frames)

run(key="exp1", lr=4e-4, decode=True, align=True, tune=True)
