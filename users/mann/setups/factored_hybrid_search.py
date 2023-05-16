__all__ = ["FHDecoder"]

import copy
from IPython import embed
from dataclasses import dataclass

from sisyphus import *

Path = tk.Path

import i6_core.recognition as recog
import i6_core.rasr as rasr
import i6_core.am as am
import i6_core.lm as lm
import i6_core.mm as mm
import i6_core.corpus as corpus_recipes
from i6_core import returnn

from i6_experiments.users.raissi.setups.common.helpers.pipeline_data import (
    ContextEnum,
    ContextMapper
)

from i6_experiments.users.raissi.setups.common.decoder.rtf import (
    ExtractSearchStatisticsJob
)
from i6_experiments.users.raissi.setups.common.decoder.factored_hybrid_feature_scorer import (
    FactoredHybridFeatureScorer
)




def get_feature_scorer(context_type, context_mapper, featureScorerConfig, mixtures,
    prior_info, silence_id, posterior_scales=None, num_label_contexts=47, num_states_per_phone=3, num_encoder_output=1024,
    loop_scale=1.0, forward_scale=1.0, silence_loop_penalty=0.0, silence_forward_penalty=0.0,
    use_estimated_tdps=False, state_dependent_tdp_file=None,
    is_min_duration=False, use_word_end_classes=False, use_boundary_classes=False, is_multi_encoder_output=False):

    if context_type.value in [context_mapper.get_enum(i) for i in [1, 7]]:
        assert prior_info['center-state-prior']['file'] is not None
        if not prior_info['center-state-prior']['scale']:
            print('You are setting prior scale equale to zero, are you sure?')

        return FactoredHybridFeatureScorer(
            featureScorerConfig,
            prior_mixtures=mixtures,
            context_type=context_type.value,
            prior_info=prior_info,
            num_states_per_phone=num_states_per_phone,
            num_label_contexts=num_label_contexts,
            silence_id=silence_id,
            num_encoder_output=num_encoder_output,
            posterior_scales=posterior_scales,
            is_multi_encoder_output=is_multi_encoder_output,
            loop_scale=loop_scale,
            forward_scale=forward_scale,
            silence_loop_penalty=silence_loop_penalty,
            silence_forward_penalty=silence_forward_penalty,
            use_estimated_tdps=use_estimated_tdps,
            state_dependent_tdp_file=state_dependent_tdp_file,
            is_min_duration=is_min_duration,
            use_word_end_classes=use_word_end_classes,
            use_boundary_classes=use_boundary_classes
        )
    else:
        print("Not Implemented")
        assert(False)


class FHDecoder:
    default_tm = {"right": "right", "center": "center", "left": "left"}

    @dataclass
    class TensorMapping:
        encoder_output: str="concat_fwd_6_bwd_6/concat_sources/concat"
        encoder_posteriors: str="encoder-output"
        center_state_posteriors: str="center-output"
        delta_posteriors: str="delta_ce"
        delta_encoder_output: str="concat_fwd_delta_bwd_delta/concat_sources/concat"
        delta_encoder_posteriors: str="deltaEncoder-output"

        def __post_init__(self):
            attrs = self.__dict__
            for k in attrs:
                if k.endswith("posteriors") and "/output" not in attrs[k]:
                    attrs[k] += "/output_batch_major"
            

    def __init__(
        self,
        name,
        search_crp,
        context_type,
        context_mapper,
        feature_path,
        model_path,
        graph,
        mixtures,
        eval_files,
        tf_library=None,
        gpu=True,
        tm=default_tm,
        output_string="output/output_batch_major",
        tensor_mapping=TensorMapping(),
        is_multi_encoder_output=False,
        silence_id=40,
    ):

        self.name = name
        self.search_crp = copy.deepcopy(search_crp)  # s.crp["dev_magic"]
        self.context_type = context_type  # contextEnum.value
        self.context_mapper = context_mapper
        self.model_path = model_path
        self.graph = graph
        self.mixtures = (
            mixtures  # s.mixtures["train_magic"]["estimate_mixtures_monophoneContext"]
        )
        self.is_multi_encoder_output = is_multi_encoder_output
        self.tdp = {}
        self.tm = tm
        self.output_string = output_string
        self.silence_id = silence_id

        self.eval_files = (
            eval_files  # ctm file as ref
        )

        self.bellman_post_config = False
        self.gpu = gpu
        self.library_path = (
            tf_library
            if tf_library is not None
            else  "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/binaries/recognition/NativeLstm2.so"
        )

        # LM attributes
        self.tfrnn_lms = {}

        self.tensor_mapping = tensor_mapping

        # setting other attributes
        self.set_tf_fs_flow(feature_path, model_path, graph)
        self.set_fs_tf_config()

    def get_search_params(
        self,
        beam,
        beamLimit,
        wePruning=0.5,
        wePruningLimit=10000,
        lmStatePruning=None,
        isCount=False,
    ):

        sp = {
            "beam-pruning": beam,
            "beam-pruning-limit": beamLimit,
            "word-end-pruning": wePruning,
            "word-end-pruning-limit": wePruningLimit,
        }
        if isCount:
            return sp
        if lmStatePruning is not None:
            sp["lm-state-pruning"] = lmStatePruning
        return sp

    def get_requirements(self, beam, isLstm=False):
        # under 27 is short queue
        rtf = 15
        if (self.context_type.value != self.context_mapper.get_enum(1) and
            self.context_type.value != self.context_mapper.get_enum(2)):
            rtf += 5
        if beam > 17:
            rtf += 10

        if isLstm:
            rtf += 20
            mem = 16.0
            if 'eval' in self.name:
                rtf *= 2
        else:
            mem = 8

        return {"rtf": rtf, "mem": mem}

    def get_lookahead_options(scale=1.0, hlimit=-1, clow=0, chigh=500):
        lmla_options = {'scale': scale,
                        'history_limit': hlimit,
                        'cache_low': clow,
                        'cache_high': chigh
                        }
        return lmla_options

    def set_tf_fs_flow(self, feature_path, model_path, graph):

        tfFeatureFlow = rasr.FlowNetwork()
        baseMapping = tfFeatureFlow.add_net(feature_path)
        if self.is_multi_encoder_output:
            tfFlow = self.get_tf_flow_delta(model_path, graph)
        else:
            tfFlow = self.get_tf_flow(model_path, graph)

        tfMapping = tfFeatureFlow.add_net(tfFlow)

        tfFeatureFlow.interconnect_inputs(feature_path, baseMapping)
        tfFeatureFlow.interconnect(
            feature_path, baseMapping, tfFlow, tfMapping, {"features": "input-features"}
        )
        tfFeatureFlow.interconnect_outputs(tfFlow, tfMapping)

        self.featureScorerFlow = tfFeatureFlow

    def get_tf_flow(self, model_path, graph):
        stringModelPath = model_path
        if not isinstance(model_path, returnn.Checkpoint):
            stringModelPath = rasr.StringWrapper(model_path, Path(model_path + ".meta"))

        tfFlow = rasr.FlowNetwork()
        tfFlow.add_input("input-features")
        tfFlow.add_output("features")
        tfFlow.add_param("id")
        tfFwd = tfFlow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
        tfFlow.link("network:input-features", tfFwd + ":features")
        tfFlow.link(tfFwd + ":encoder-output", "network:features")

        tfFlow.config = rasr.RasrConfig()

        tfFlow.config[tfFwd].input_map.info_0.param_name = "features"
        tfFlow.config[
            tfFwd
        ].input_map.info_0.tensor_name = "extern_data/placeholders/data/data"
        tfFlow.config[
            tfFwd
        ].input_map.info_0.seq_length_tensor_name = (
            "extern_data/placeholders/data/data_dim0_size"
        )

        tfFlow.config[tfFwd].output_map.info_0.param_name = "encoder-output"
        tfFlow.config[
            tfFwd
        ].output_map.info_0.tensor_name = self.tensor_mapping.encoder_posteriors # "encoder-output/output_batch_major"

        tfFlow.config[tfFwd].loader.type = "meta"
        tfFlow.config[tfFwd].loader.meta_graph_file = graph
        tfFlow.config[tfFwd].loader.saved_model_file = stringModelPath
        tfFlow.config[tfFwd].loader.required_libraries = self.library_path

        return tfFlow

    def get_tf_flow_delta(self, model_path, graph):
        stringModelPath = model_path
        if not isinstance(model_path, returnn.Checkpoint):
            stringModelPath = rasr.StringWrapper(model_path, Path(model_path + ".meta"))

        tfFlow = rasr.FlowNetwork()
        tfFlow.add_input("input-features")
        tfFlow.add_output("features")
        tfFlow.add_param("id")
        tfFwd = tfFlow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
        tfFlow.link("network:input-features", tfFwd + ":features")

        concat = tfFlow.add_node('generic-vector-f32-concat',
                                 'concatenation',
                                 {'check-same-length': True,
                                  'timestamp-port': 'feature-1'})

        tfFlow.link(tfFwd + ":encoder-output", '%s:%s' % (concat, 'feature-1'))
        tfFlow.link(tfFwd + ":deltaEncoder-output", '%s:%s' % (concat, 'feature-2'))

        tfFlow.link(concat, "network:features")

        tfFlow.config = rasr.RasrConfig()

        tfFlow.config[tfFwd].input_map.info_0.param_name = "features"
        tfFlow.config[
            tfFwd
        ].input_map.info_0.tensor_name = "extern_data/placeholders/data/data"
        tfFlow.config[
            tfFwd
        ].input_map.info_0.seq_length_tensor_name = (
            "extern_data/placeholders/data/data_dim0_size"
        )

        tfFlow.config[tfFwd].output_map.info_0.param_name = "encoder-output"
        tfFlow.config[
            tfFwd
        ].output_map.info_0.tensor_name = self.tensor_mapping.encoder_posteriors # "encoder-output/output_batch_major"

        tfFlow.config[tfFwd].output_map.info_1.param_name = "deltaEncoder-output"
        tfFlow.config[
            tfFwd
        ].output_map.info_1.tensor_name = self.tensor_mapping.delta_encoder_posteriors # "deltaEncoder-output/output_batch_major"

        tfFlow.config[tfFwd].loader.type = "meta"
        tfFlow.config[tfFwd].loader.meta_graph_file = graph
        tfFlow.config[tfFwd].loader.saved_model_file = stringModelPath
        tfFlow.config[tfFwd].loader.required_libraries = self.library_path

        return tfFlow

    def set_fs_tf_config(self):
        fsTfConfig = rasr.RasrConfig()
        fsTfConfig.loader = self.featureScorerFlow.config["tf-fwd"]["loader"]
        del fsTfConfig.input_map
        #input is the same for each model, since the label embeddings are calculated from the dense label identity
        fsTfConfig.input_map.info_0.param_name = "encoder-output"
        fsTfConfig.input_map.info_0.tensor_name = self.tensor_mapping.encoder_output
        #monophone does not have any context
        if self.context_type.value not in [self.context_mapper.get_enum(i) for i in [1, 7]]:
            fsTfConfig.input_map.info_1.param_name = "dense-classes"
            fsTfConfig.input_map.info_1.tensor_name = (
                "extern_data/placeholders/classes/classes"
            )

        if self.context_type.value in [self.context_mapper.get_enum(i) for i in [1,7]] :
            fsTfConfig.output_map.info_0.param_name = "center-state-posteriors"
            fsTfConfig.output_map.info_0.tensor_name = self.tensor_mapping.center_state_posteriors
            if self.context_type.value == self.context_mapper.get_enum(7):
                #add the delta outputs
                fsTfConfig.output_map.info_1.param_name = "delta-posteriors"
                fsTfConfig.output_map.info_1.tensor_name = self.tensor_mapping.delta_posteriors

        if self.context_type.value in [self.context_mapper.get_enum(i) for i in [2, 8]]:
            fsTfConfig.output_map.info_0.param_name = "center-state-posteriors"
            fsTfConfig.output_map.info_0.tensor_name = ("-").join(
                [self.tm["center"], self.output_string]
            )
            fsTfConfig.output_map.info_1.param_name = "context-posteriors"
            fsTfConfig.output_map.info_1.tensor_name = ("-").join(
                [self.tm["context"], self.output_string]
            )
            if self.context_type.value == self.context_mapper.get_enum(8):
                fsTfConfig.output_map.info_2.param_name = "delta-posteriors"
                fsTfConfig.output_map.info_2.tensor_name = "delta_ce/output_batch_major"

        if self.context_type.value == self.context_mapper.get_enum(3):
            fsTfConfig.output_map.info_0.param_name = "center-state-posteriors"
            fsTfConfig.output_map.info_0.tensor_name = ("-").join(
                [self.tm["center"], self.output_string]
            )
            fsTfConfig.output_map.info_1.param_name = "left-context-posteriors"
            fsTfConfig.output_map.info_1.tensor_name = ("-").join(
                [self.tm["left"], self.output_string]
            )
            fsTfConfig.output_map.info_2.param_name = "right-context-posteriors"
            fsTfConfig.output_map.info_2.tensor_name = ("-").join(
                [self.tm["right"], self.output_string]
            )

        if self.context_type.value in [self.context_mapper.get_enum(i) for i in [4, 6]]:
            # outputs
            fsTfConfig.output_map.info_0.param_name = "right-context-posteriors"
            fsTfConfig.output_map.info_0.tensor_name = ("-").join(
                [self.tm["right"], self.output_string]
            )
            fsTfConfig.output_map.info_1.param_name = "center-state-posteriors"
            fsTfConfig.output_map.info_1.tensor_name = ("-").join(
                [self.tm["center"], self.output_string]
            )
            fsTfConfig.output_map.info_2.param_name = "left-context-posteriors"
            fsTfConfig.output_map.info_2.tensor_name = ("-").join(
                [self.tm["left"], self.output_string]
            )

            if self.context_type.value == self.context_mapper.get_enum(6):
                fsTfConfig.output_map.info_3.param_name = "delta-posteriors"
                fsTfConfig.output_map.info_3.tensor_name = "delta_ce/output_batch_major"

        elif self.context_type.value == self.context_mapper.get_enum(5):
            # outputs
            fsTfConfig.output_map.info_0.param_name = "left-context-posteriors"
            fsTfConfig.output_map.info_0.tensor_name = ("-").join(
                [self.tm["left"], self.output_string]
            )
            fsTfConfig.output_map.info_1.param_name = "right-context-posteriors"
            fsTfConfig.output_map.info_1.tensor_name = ("-").join(
                [self.tm["right"], self.output_string]
            )
            fsTfConfig.output_map.info_2.param_name = "center-state-posteriors"
            fsTfConfig.output_map.info_2.tensor_name = ("-").join(
                [self.tm["center"], self.output_string]
            )

        if self.is_multi_encoder_output:
            if self.context_type.value == self.context_mapper.get_enum(7) \
                or self.context_type.value == self.context_mapper.get_enum(1):
                fsTfConfig.input_map.info_1.param_name = "deltaEncoder-output"
                fsTfConfig.input_map.info_1.tensor_name = (
                    self.tensor_mapping.delta_encoder_output # "concat_fwd_delta_bwd_delta/concat_sources/concat"
                )
            else:
                fsTfConfig.input_map.info_2.param_name = "deltaEncoder-output"
                fsTfConfig.input_map.info_2.tensor_name = (
                    self.tensor_mapping.encoder_output # "concat_fwd_delta_bwd_delta/concat_sources/concat"
                )

        self.featureScorerConfig = fsTfConfig

    def getFeatureFlow(self, feature_path, tfFlow):
        tf_feature_flow = rasr.FlowNetwork()
        base_mapping = tf_feature_flow.add_net(feature_path)
        tf_mapping = tf_feature_flow.add_net(tfFlow)
        tf_feature_flow.interconnect_inputs(feature_path, base_mapping)
        tf_feature_flow.interconnect(
            feature_path,
            base_mapping,
            tfFlow,
            tf_mapping,
            {"features": "input-features"},
        )
        tf_feature_flow.interconnect_outputs(tfFlow, tf_mapping)

        return tf_feature_flow

    def get_tfrnn_lm_config(
        self,
        name,
        scale,
        min_batch_size=1024,
        opt_batch_size=1024,
        max_batch_size=1024,
        allow_reduced_hist=None,
        async_lm=False,
        single_step_only=False,
    ):
        res = copy.deepcopy(self.tfrnn_lms[name])
        if allow_reduced_hist is not None:
            res.allow_reduced_history = allow_reduced_hist
        res.scale = scale
        res.min_batch_size = min_batch_size
        res.opt_batch_size = opt_batch_size
        res.max_batch_size = max_batch_size
        if async_lm:
            res["async"] = async_lm
        if async_lm or single_step_only:
            res.single_step_only = True

        return res

    def get_nce_lm(self, **kwargs):
        lstm_lm_config = self.get_tfrnn_lm_config(**kwargs)
        lstm_lm_config.output_map.info_1.param_name = "weights"
        lstm_lm_config.output_map.info_1.tensor_name = "output/W/read"
        lstm_lm_config.output_map.info_2.param_name = "bias"
        lstm_lm_config.output_map.info_2.tensor_name = "output/b/read"
        lstm_lm_config.softmax_adapter.type = "blas_nce"

        return lstm_lm_config

    def get_rnn_config(self, scale=1.0):
        lmConfigParams = {
            "name": "kazuki_real_nce",
            "min_batch_size": 1024,
            "opt_batch_size": 1024,
            "max_batch_size": 1024,
            "scale": scale,
            "allow_reduced_hist": True,
        }

        return self.get_nce_lm(**lmConfigParams)

    def get_rnn_full_config(self, scale=13.0):
        lmConfigParams = {
            "name": "kazuki_full",
            "min_batch_size": 4,
            "opt_batch_size": 64,
            "max_batch_size": 128,
            "scale": scale,
            "allow_reduced_hist": True,
        }
        return self.get_tfrnn_lm_config(**lmConfigParams)

    def add_tfrnn_lms(self):
        tfrnn_dir = "/u/beck/setups/swb1/2018-06-08_nnlm_decoding/dependencies/tfrnn"
        # backup:  '/work/asr4/raissi/ms-thesis-setups/dependencies/tfrnn'

        rnn_lm_config = rasr.RasrConfig()
        rnn_lm_config.type = "tfrnn"
        rnn_lm_config.vocab_file = Path("%s/vocabulary" % tfrnn_dir)
        rnn_lm_config.transform_output_negate = True
        rnn_lm_config.vocab_unknown_word = "<unk>"

        rnn_lm_config.loader.type = "meta"
        rnn_lm_config.loader.meta_graph_file = Path("%s/network.019.meta" % tfrnn_dir)
        rnn_lm_config.loader.saved_model_file = rasr.StringWrapper(
            "%s/network.019" % tfrnn_dir, Path("%s/network.019.index" % tfrnn_dir)
        )
        rnn_lm_config.loader.required_libraries = Path(self.native_lstm_path)

        rnn_lm_config.input_map.info_0.param_name = "word"
        rnn_lm_config.input_map.info_0.tensor_name = (
            "extern_data/placeholders/delayed/delayed"
        )
        rnn_lm_config.input_map.info_0.seq_length_tensor_name = (
            "extern_data/placeholders/delayed/delayed_dim0_size"
        )

        rnn_lm_config.output_map.info_0.param_name = "softmax"
        rnn_lm_config.output_map.info_0.tensor_name = "output/output_batch_major"

        kazuki_fake_nce_lm = copy.deepcopy(rnn_lm_config)
        tfrnn_dir = "/work/asr3/beck/setups/swb1/2018-06-08_nnlm_decoding/dependencies/tfrnn_nce"
        kazuki_fake_nce_lm.vocab_file = Path("%s/vocabmap.freq_sorted.txt" % tfrnn_dir)
        kazuki_fake_nce_lm.loader.meta_graph_file = Path(
            "%s/inference.meta" % tfrnn_dir
        )
        kazuki_fake_nce_lm.loader.saved_model_file = rasr.StringWrapper(
            "%s/network.018" % tfrnn_dir, Path("%s/network.018.index" % tfrnn_dir)
        )

        kazuki_real_nce_lm = copy.deepcopy(kazuki_fake_nce_lm)
        kazuki_real_nce_lm.output_map.info_0.tensor_name = "sbn/output_batch_major"

        self.tfrnn_lms["kazuki_real_nce"] = kazuki_real_nce_lm

    def add_lstm_full(self):
        tfrnn_dir = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/lstm-lm-kazuki"

        rnn_lm_config = rasr.RasrConfig()
        rnn_lm_config.type = "tfrnn"
        rnn_lm_config.vocab_file = Path(("/").join([tfrnn_dir, "vocabulary"]))
        rnn_lm_config.transform_output_negate = True
        rnn_lm_config.vocab_unknown_word = "<unk>"

        rnn_lm_config.loader.type = "meta"
        rnn_lm_config.loader.meta_graph_file = Path("%s/network.019.meta" % tfrnn_dir)
        rnn_lm_config.loader.saved_model_file = rasr.StringWrapper(
            "%s/network.019" % tfrnn_dir, Path("%s/network.019.index" % tfrnn_dir)
        )
        rnn_lm_config.loader.required_libraries = Path(self.native_lstm_path)

        rnn_lm_config.input_map.info_0.param_name = "word"
        rnn_lm_config.input_map.info_0.tensor_name = (
            "extern_data/placeholders/delayed/delayed"
        )
        rnn_lm_config.input_map.info_0.seq_length_tensor_name = (
            "extern_data/placeholders/delayed/delayed_dim0_size"
        )

        rnn_lm_config.output_map.info_0.param_name = "softmax"
        rnn_lm_config.output_map.info_0.tensor_name = "output/output_batch_major"

        self.tfrnn_lms["kazuki_full"] = rnn_lm_config

    def recognize_count_lm(self, priorInfo, lmScale, posteriorScales=None, n_contexts=42, n_states_per_phone=3,
                           num_encoder_output=1024, is_min_duration=False, use_word_end_classes=False, transitionScales=None, silencePenalties=None,
                           useEstimatedTdps=False, forwardProbfile=None,
                           addAllAllos=True,
                           tdpScale=1.0, tdpExit=0.0, tdpNonword=20.0, silExit=20.0, tdpSkip=30.0, spLoop=3.0, spFwd=0.0, silLoop=0.0, silFwd=3.0,
                           beam=20.0, beamLimit=400000, wePruning=0.5, wePruningLimit=10000, pronScale=3.0, altas=None,
                           runOptJob=False, onlyLmOpt=True, calculateStat=False, keep_value=12,
    ):

        if posteriorScales is None:
            posteriorScales = dict(zip([f'{k}-scale' for k in ['left-context', 'center-state', 'right-context' ]], [1.0] * 3))
        loopScale = forwardScale = 1.0
        silFwdPenalty = silLoopPenalty = 0.0

        name = ('-').join([self.name, f"Beam{beam}", f"Lm{lmScale}", "pronScale-"])
        for k in ['left-context', 'center-state', 'right-context']:
            if priorInfo[f'{k}-prior']['file'] is not None:
                k_ = k.split("-")[0]
                name = f"{name}-{k_}-priorScale-{priorInfo[f'{k}-prior']['scale']}"

        if tdpScale > 0:
            name += f'tdpScale-{tdpScale}tdpEx-{tdpExit}silExitEx-{silExit}tdpNEx-{tdpNonword}'
            if transitionScales is not None:
                loopScale = transitionScales[0]
                forwardScale = transitionScales[1]
                name += f"loopScale-{loop_scale}"
                name += f"fwdScale-{forward_scale}"
            if silencePenalties is not None:
                silLoopPenalty = silencePenalties[0]
                silFwdPenalty = silencePenalties[1]
                name += "fsilLoopP-{silLoopPenalty}"
                name += f"silFwdP-{silFwdPenalty}"
            if tdpSkip == 'infinity':
                name += "noSkip"
        else:
            name += 'noTdp'
            spLoop = spFwd = tdpExit = silLoop = silFwd = silExit = 0.0
        if wePruning > 0.5:
            name += "wep" + str(wePruning)
        if altas is not None:
            name += f"-ALTAS{altas}"
        if addAllAllos:
            name += "addAll-allos"

        # am config update
        searchCrp = copy.deepcopy(self.search_crp)
        state_tying = searchCrp.acoustic_model_config.state_tying.type

        searchCrp.acoustic_model_config = am.acoustic_model_config(
            state_tying=state_tying,
            states_per_phone=n_states_per_phone,
            state_repetitions=1,
            across_word_model=True,
            early_recombination=False,
            tdp_scale=tdpScale,
            tdp_transition=(spLoop, spFwd, 'infinity', tdpExit),
            tdp_silence=(silLoop, silFwd, 'infinity', silExit),
        )

        searchCrp.acoustic_model_config.allophones["add-all"] = addAllAllos
        searchCrp.acoustic_model_config.allophones["add-from-lexicon"] = not addAllAllos

        if 'tying-dense' in state_tying:
            use_boundary_classes = self.search_crp.acoustic_model_config["state-tying"][
                "use-boundary-classes"
            ]
            searchCrp.acoustic_model_config["state-tying"][
                "use-boundary-classes"
            ] = use_boundary_classes

        if use_word_end_classes:
            searchCrp.acoustic_model_config["state-tying"]["use-word-end-classes"] = True

        # lm config update
        searchCrp.language_model_config.scale = lmScale
        # additional config
        if searchCrp.lexicon_config.normalize_pronunciation:
            modelCombinationConfig = rasr.RasrConfig()
            modelCombinationConfig.pronunciation_scale = pronScale
        else:
            modelCombinationConfig = None
            pronScale = 1.0
        # additional search parameters
        name += f'pronScale{pronScale}'
        rqms = self.get_requirements(beam)
        sp = self.get_search_params(beam, beamLimit, wePruning, wePruningLimit, isCount=True)

        if altas is not None:
            adv_search_extra_config = rasr.RasrConfig()
            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = (
                altas
            )
        else:
            adv_search_extra_config = None

        self.feature_scorer = featureScorer = get_feature_scorer(context_type=self.context_type, context_mapper=self.context_mapper,
                                           featureScorerConfig=self.featureScorerConfig,
                                           mixtures=self.mixtures, silence_id=self.silence_id,
                                           prior_info=priorInfo,
                                           posterior_scales=posteriorScales, num_label_contexts=n_contexts, num_states_per_phone=n_states_per_phone,
                                           num_encoder_output=num_encoder_output,
                                           loop_scale=loopScale, forward_scale=forwardScale,
                                           silence_loop_penalty=silLoopPenalty, silence_forward_penalty=silFwdPenalty,
                                           use_estimated_tdps=useEstimatedTdps, state_dependent_tdp_file=forwardProbfile,
                                           is_min_duration=is_min_duration, use_word_end_classes=use_word_end_classes,
                                           use_boundary_classes=use_boundary_classes, is_multi_encoder_output=self.is_multi_encoder_output)

        if altas is not None:
            prepath = 'decoding-gridsearch/'
        else: prepath = 'decoding/'

        search = recog.AdvancedTreeSearchJob(
            crp=searchCrp,
            feature_flow=self.featureScorerFlow,
            feature_scorer=featureScorer,
            search_parameters=sp,
            lm_lookahead=True,
            eval_best_in_lattice=True,
            use_gpu=self.gpu,
            rtf=rqms["rtf"],
            mem=rqms["mem"],
            model_combination_config=modelCombinationConfig,
            model_combination_post_config=None,
            extra_config=adv_search_extra_config,
            extra_post_config=None,
        )
        search.rqmt["cpu"] = 2
        search.add_alias(f"{prepath}recog_%s" % name)

        if calculateStat:
            stat = ExtractSearchStatisticsJob(list(search.log_file.values()), 3.61)
            stat.add_alias(f"statistics/stats_{name}")
            tk.register_output(f"rtf_decode_{name}", stat.decoding_rtf)

        if keep_value is not None:
            search.keep_value(keep_value)



        lat2ctm_extra_config = rasr.RasrConfig()
        lat2ctm_extra_config.flf_lattice_tool.network.to_lemma.links = "best"
        lat2ctm = recog.LatticeToCtmJob(
            crp=searchCrp,
            lattice_cache=search.out_lattice_bundle,
            parallelize=True,
            best_path_algo="bellman-ford",
            extra_config=lat2ctm_extra_config,
        )

        sKwrgs = copy.copy(self.eval_files)
        sKwrgs["hyp"] = lat2ctm.out_ctm_file
        scorer = recog.ScliteJob(**sKwrgs)
        tk.register_output(f"{prepath}{name}.wer", scorer.out_report_dir)

        if beam > 15.0:
            opt = recog.OptimizeAMandLMScaleJob(
                crp=searchCrp,
                lattice_cache=search.out_lattice_bundle,
                initial_am_scale=pronScale,
                initial_lm_scale=lmScale,
                scorer_cls=recog.ScliteJob,
                scorer_kwargs=sKwrgs,
                opt_only_lm_scale=onlyLmOpt,
            )
            tk.register_output(f"{prepath}{name}.onlyLmOpt{onlyLmOpt}.optlm.txt", opt.out_log_file)


    def align(
        self,
        name,
        crp,
        rtf=10,
        mem=8,
        am_trainer_exe_path=None,
        default_tdp=True,
    ):

        alignCrp = copy.deepcopy(crp)
        if am_trainer_exe_path is not None:
            alignCrp.acoustic_model_trainer_exe = am_trainer_exe_path


        if default_tdp:
            v = (3.0, 0.0, 'infinity', 0.0)
            sv = (0.0, 3.0, 'infinity', 0.0)
            keys = ["loop", "forward", "skip", "exit"]
            for i, k in enumerate(keys):
                alignCrp.acoustic_model_config.tdp["*"][k] = v[i]
                alignCrp.acoustic_model_config.tdp["silence"][k] = sv[i]

        #make sure it is correct for the fh feature scorer scorer
        alignCrp.acoustic_model_config.state_tying.type = 'no-tying-dense'

        # make sure the FSA is not buggy
        alignCrp.acoustic_model_config["*"]["fix-allophone-context-at-word-boundaries"] = True
        alignCrp.acoustic_model_config["*"]["transducer-builder-filter-out-invalid-allophones"] = True
        alignCrp.acoustic_model_config["*"]["allow-for-silence-repetitions"] = False
        alignCrp.acoustic_model_config["*"]["fix-tdp-leaving-epsilon-arc"] = True


        alignment = mm.AlignmentJob(
            crp=alignCrp,
            feature_flow=self.featureScorerFlow,
            feature_scorer=self.feature_scorer,
            use_gpu=self.gpu,
            rtf=rtf,
        )
        alignment.rqmt["cpu"] = 2
        alignment.rqmt["mem"] = 8
        alignment.add_alias(f'alignments/align_{name}')
        tk.register_output("alignments/realignment-{}".format(name), alignment.out_alignment_bundle)
        return alignment


