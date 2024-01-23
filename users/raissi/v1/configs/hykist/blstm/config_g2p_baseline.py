import os, copy, sys
from IPython import embed

# -------------------- Sisyphus --------------------

from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.corpus as corpus_recipe
import i6_core.rasr as rasr
import i6_core.returnn as returnn
import i6_core.mm as mm

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.users.raissi.setups.hykist.ukrainian_system import UkrainianHybridSystem

import i6_private.users.raissi.datasets.ukrainian_8khz as uk_8khz_data
import i6_private.users.raissi.setups.hykist.ukrainian_8khz.pipeline_base_args as uk_8khz_gmm_setup

from i6_experiments.users.raissi.setups.common.helpers.train_helpers import warmup_lrates, get_monophone_returnn_config

from i6_experiments.users.raissi.setups.common.helpers.specaugment_returnn_epilog import (
    get_specaugment_epilog,
)

from i6_experiments.users.raissi.experiments.hykist.train.blstm_args import (
    get_train_params_ce,
)

returnn_python_exe = tk.Path(
    "/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER"
)
# gs.SIS_COMMAND = ['/work/asr4/rossenbach/env/python38_sisyphus/bin/python3', sys.argv[0]] #g2p error

# -----------------------------------------------------------------------------------------------
def get_tf_flow(model_path, tf_graph, native_lstm_path):
    tf_flow = rasr.FlowNetwork()
    tf_flow.add_input("input-features")
    tf_flow.add_output("features")
    tf_flow.add_param("id")

    tf_fwd = tf_flow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
    tf_flow.link("network:input-features", tf_fwd + ":features")
    tf_flow.link(tf_fwd + ":posteriors", "network:features")

    tf_flow.config = rasr.RasrConfig()

    tf_flow.config[tf_fwd].input_map.info_0.param_name = "features"
    tf_flow.config[tf_fwd].input_map.info_0.tensor_name = "extern_data/placeholders/data/data"
    tf_flow.config[tf_fwd].input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/data/data_dim0_size"

    tf_flow.config[tf_fwd].output_map.info_0.param_name = "posteriors"
    tf_flow.config[tf_fwd].output_map.info_0.tensor_name = "center-output/output_batch_major"

    tf_flow.config[tf_fwd].loader.type = "meta"
    tf_flow.config[tf_fwd].loader.meta_graph_file = tf_graph
    tf_flow.config[tf_fwd].loader.saved_model_file = rasr.StringWrapper(model_path, Path(model_path + ".meta"))

    tf_flow.config[tf_fwd].loader.required_libraries = native_lstm_path

    return tf_flow


def get_feature_flow(base_flow, tf_flow):
    feature_flow = rasr.FlowNetwork()
    base_mapping = feature_flow.add_net(base_flow)
    tf_mapping = feature_flow.add_net(tf_flow)
    feature_flow.interconnect_inputs(base_flow, base_mapping)
    feature_flow.interconnect(base_flow, base_mapping, tf_flow, tf_mapping, {"features": "input-features"})
    feature_flow.interconnect_outputs(tf_flow, tf_mapping)

    return feature_flow


def run_blstm_from_gmm_alignment(key, decode_corpus="dev-baseline"):
    gs.ALIAS_AND_OUTPUT_SUBDIR = "baseline/gmm_it2_g2p"
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    # ******************** GMM Init ********************
    data_input = uk_8khz_data.get_data_inputs(use_g2p=True)

    rasr_init_args = uk_8khz_gmm_setup.get_init_args(
        dc_detection=True,
        mfcc_extra_args={
            "mfcc_options": {
                "warping_function": "mel",
                "filter_width": 268.258,
            }
        },
        scorer="sclite",
        scorer_args={"sort_files": False},
    )
    rasr_init_args.costa_args["eval_lm"] = False

    # ******************** GMM Pipeline Args ********************

    mono_args = uk_8khz_gmm_setup.get_monophone_args()

    cart_args = uk_8khz_data.get_cart_args()
    tri_args = uk_8khz_gmm_setup.get_triphone_args()
    vtln_args = uk_8khz_gmm_setup.get_vtln_args()
    sat_args = uk_8khz_gmm_setup.get_sat_args()
    vtln_sat_args = uk_8khz_gmm_setup.get_vtln_sat_args()
    final_output_args = uk_8khz_gmm_setup.get_final_output()
    """
    final_output_args={'name': 'final',
                       'corpus_type_mapping': {'train-baseline': 'train',
                       'dev-baseline': 'dev',
                       'test-baseline': 'test'},
                       'extract_features': []}

    
    """

    steps = rasr_util.RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("mono", mono_args)

    uk_gmm_system = gmm_system.GmmSystem(
        rasr_binary_path=tk.Path("/work/tools/asr/rasr/20191102_generic/arch/linux-x86_64-standard/"),
    )

    uk_gmm_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=data_input["train"],
        dev_data=data_input["dev"],
        test_data=data_input["test"],
    )

    uk_gmm_system.run(steps)
    for k in uk_gmm_system.corpora.keys():
        c2tJob = corpus_recipe.CorpusToTxtJob(bliss_corpus=uk_gmm_system.corpora[k].corpus_file)
        tk.register_output(f"corpora/texts/{k}.txt", c2tJob.out_txt)

    for trn_c in uk_gmm_system.train_corpora:
        uk_gmm_system.crp[trn_c].acoustic_model_trainer_exe = Path(
            "/u/raissi/dev/rasr_tf14py38_fh/src/Tools/AcousticModelTrainer/acoustic-model-trainer.linux-x86_64-standard"
        )
    cart_step = rasr_util.RasrSteps()
    cart_step.add_step("cart", cart_args)
    uk_gmm_system.run(cart_step)

    for trn_c in uk_gmm_system.train_corpora:
        uk_gmm_system.crp[trn_c].acoustic_model_trainer_exe = Path(
            "/work/tools/asr/rasr/20191102_generic/arch/linux-x86_64-standard/acoustic-model-trainer.linux-x86_64-standard"
        )

    steps_after_cart = rasr_util.RasrSteps()
    steps_after_cart.add_step("tri", tri_args)
    steps_after_cart.add_step("vtln", vtln_args)
    steps_after_cart.add_step("sat", sat_args)
    steps_after_cart.add_step("vtln+sat", vtln_sat_args)
    steps_after_cart.add_step("output", final_output_args)
    uk_gmm_system.run(steps_after_cart)

    # ******************** NN Args ********************
    config_path = __file__.split("config/")[1].split(".py")[0]
    gs.ALIAS_AND_OUTPUT_SUBDIR = ("").join(config_path.split("config_"))

    steps = rasr_util.RasrSteps()
    label_info_args = {
        "state_tying": "cart",
        "state_tying_file": uk_gmm_system.crp["train-baseline"].acoustic_model_config.state_tying.file,
        "n_cart_labels": int(cart_args.cart_questions.max_leaves),
    }
    steps.add_step("init", label_info_args)  # you can create the label_info and pass here
    s = UkrainianHybridSystem(
        returnn_python_exe=returnn_python_exe,
        rasr_init_args=rasr_init_args,
        train_data=data_input["train"],
        dev_data=data_input["dev"],
        test_data=data_input["test"],
    )
    s.run(steps)
    # *********** Preparation of data input for rasr-returnn training *****************
    steps_input = rasr_util.RasrSteps()
    #
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    #
    alignment_args = {
        "train-baseline": uk_gmm_system.alignments["train-baseline"]["train_vtln+sat"][0].alternatives["bundle"]
    }
    steps_input.add_step("alignment", alignment_args)
    #
    input_args = copy.deepcopy(final_output_args)
    input_args.name = "data_preparation"
    input_args.add_feature_to_extract("gt")
    steps_input.add_step("input", input_args)
    #
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(input_key="data_preparation")
    s._update_am_setting_for_all_crps(train_tdp_type="default", eval_tdp_type="default")
    s.create_hdf(with_alignment=True)
    # ---------------------- returnn config---------------
    return_config_arguments = copy.deepcopy(s.initial_nn_args)
    return_config_arguments.update(get_train_params_ce())
    num_epochs = return_config_arguments.pop("num_epochs")
    partition_epochs = return_config_arguments["partition_epochs"]
    additional_args = {
        "python_epilog": get_specaugment_epilog(t_num=3, t=10, f_num=4, f=5),
        "learning_rates": warmup_lrates(initial=0.0001, final=return_config_arguments["lr"], epochs=20),
    }
    return_config_arguments.update(additional_args)
    cart_config = get_monophone_returnn_config(
        num_classes=s.label_info.get_n_state_classes(),
        ph_emb_size=0,
        st_emb_size=0,
        add_mlps=False,
        use_multi_task=False,
        **return_config_arguments,
    )
    # setup the experiment dict
    prefix_name = "swb-params"
    s.set_experiment_dict(key, "GMMtri", "cart", postfix_name=prefix_name)

    s.experiments[key]["returnn_config"] = cart_config
    train_args = copy.deepcopy(s.initial_train_args)

    extra_config = rasr.RasrConfig()
    extra_config[
        "*"
    ].segments_to_skip = "corpus/Axxon.MixedMedia-batch.1/R20191120032450_I_8002749166_7037141011_IVR1_1/R20191120032450_I_8002749166_7037141011_IVR1_1_00056_803.28-804.51"
    additional_train_args = {
        "returnn_config": cart_config,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
        "keep_epochs": list(range(num_epochs - (partition_epochs["train"] * 2), num_epochs + 1)),
        "extra_rasr_config": extra_config,
    }
    train_args.update(additional_train_args)

    s.returnn_rasr_training(
        experiment_key=key,
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
    )

    s.set_mono_priors(key=key, epoch=180, hdf_key=s.crp_names["train"])
    # -------CART recognition with standard feature scorer ----------------------------------
    def run_decoding(p_scale=0.6, tdp_scale=1.0, exit_sil=20.0, beam=18.0, altas=None):
        feature_scorer = rasr.PrecomputedHybridFeatureScorer(
            prior_mixtures=mm.CreateDummyMixturesJob(9001, 40).out_mixtures,
            prior_file=s.experiments[key]["priors"][0],
            scale=1.0,
            priori_scale=p_scale,
        )

        new_config = copy.deepcopy(s.experiments[key]["returnn_config"])
        new_config.config["network"]["center-output"]["class"] = "linear"
        new_config.config["network"]["center-output"]["activation"] = "log_softmax"

        compiled_graph = returnn.CompileTFGraphJob(new_config).out_graph

        modelPath = s._get_model_path(s.experiments[key]["train_job"], 180)
        tf_flow = get_tf_flow(modelPath, compiled_graph, tk.Path(s.tf_library))

        feature_flow = get_feature_flow(s.feature_flows[decode_corpus], tf_flow)
        s.run_decoding_for_cart(
            name=f'{s.experiments[key]["name"]}_pscale-{p_scale}',
            corpus=decode_corpus,
            feature_flow=feature_flow,
            feature_scorer=feature_scorer,
            tdp_scale=tdp_scale,
            exit_sil=exit_sil,
            beam=beam,
            altas=altas,
        )

    run_decoding(p_scale=0.6, tdp_scale=0.1)


run_blstm_from_gmm_alignment(key="exp-base")
run_blstm_from_gmm_alignment(key="exp-base", decode_corpus="test-baseline")
