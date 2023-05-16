from sisyphus import tk, gs, delayed_ops

from collections import ChainMap, defaultdict
from functools import cached_property

from enum import Enum
import copy
import contextlib

from ..factored_hybrid_search import FHDecoder, ContextEnum, ContextMapper, get_feature_scorer
from i6_core.mm import CreateDummyMixturesJob
from i6_core import rasr
from i6_core.lib import lexicon

class DelayedPhonemeIndex(delayed_ops.DelayedBase):

    def __init__(self, lexicon_path, phoneme):
        super().__init__(lexicon_path)
        self.lexicon_path = lexicon_path
        self.phoneme = phoneme
    
    def get(self):
        lex = lexicon.Lexicon()
        lex.load(self.lexicon_path.get())
        return list(lex.phonemes).index(self.phoneme) + 1 # counting also "#" symbol
    
    def __sis_state__(self):
        return self.lexicon_path, self.phoneme

class DelayedPhonemeInventorySize(delayed_ops.DelayedBase):

    def __init__(self, lexicon_path):
        super().__init__(lexicon_path)
        self.lexicon_path = lexicon_path
    
    def get(self):
        lex = lexicon.Lexicon()
        lex.load(self.lexicon_path.get())
        return len(list(lex.phonemes)) + 1
    
    def __sis_state__(self):
        return self.lexicon_path

class TransitionType(Enum):
    label = "center"
    feature = "feature-center"


class FactoredHybridDecoder:
    """Wrapper around Tina's implementation to connect to BaseSystem."""
    default_recognition_args = {
        "rtf": 20,
    }

    def __init__(
        self, system=None,
        default_decoding_args=None,
        use_native_ops=False,
    ):
        self.system = system

        self.default_decoding_args = {
            # "context_type": ContextEnum.monophone,
            **(default_decoding_args or {}),
        }
        self.use_native_ops = use_native_ops
        self.flows = defaultdict(lambda: defaultdict(dict))
        self.decoders = defaultdict(dict)
        self.init_env()

    def init_env(self):
        RETURNN_PYTHON_HOME = tk.Path("/work/tools/asr/python/3.8.0")
        lib_subdir = "lib/python3.8/site-packages"
        libs = ["numpy/.libs"]
        path_buffer = ""
        for lib in libs:
            path_buffer += ":" + RETURNN_PYTHON_HOME.join_right(lib_subdir).join_right(lib) 
        gs.DEFAULT_ENVIRONMENT_SET["LD_LIBRARY_PATH"] += path_buffer
    
    def set_system(self, system):
        self.system = system

    @cached_property
    def num_all_classes(self):
        from i6_core.lexicon.allophones import DumpStateTyingJob
        from i6_experiments.users.mann.experimental.extractors import ExtractStateTyingStats

        extra_config = rasr.RasrConfig()
        extra_config["allophone-tool.acoustic-model.state-tying"].type = "no-tying-dense"
        state_tying = DumpStateTyingJob(self.system.crp["train"], extra_config=extra_config).out_state_tying
        return ExtractStateTyingStats(state_tying).out_num_states
    
    def num_phonemes(self):
        return DelayedPhonemeInventorySize(self.system.crp["dev"].lexicon_config.file)
    
    def num_mixtures(self):
        assert "no-tying-dense" in self.system.get_state_tying()
        we = self.system.crp["base"].acoustic_model_config.state_tying.use_word_end_classes
        hmm_partition = self.system.crp["base"].acoustic_model_config.hmm.states_per_phone
        n_phonemes = DelayedPhonemeInventorySize(self.system.crp["dev"].lexicon_config.file)
        return n_phonemes**3 * hmm_partition * (2 if we else 1)
    
    def get_tensor_mapping(self, context_type, transition_type):
        res = {
            ContextEnum.monophone: FHDecoder.TensorMapping(
                center_state_posteriors="output",
                encoder_posteriors="encoder_output",
            ),
            ContextEnum.mono_state_transition: FHDecoder.TensorMapping(
                center_state_posteriors="output",
                encoder_posteriors="encoder_output",
                delta_posteriors="tdps/output/stack",
            )
        }.get(context_type)
        if transition_type == TransitionType.feature:
            res = FHDecoder.TensorMapping(
                center_state_posteriors="output",
                encoder_posteriors="encoder_output",
                delta_posteriors="tdps/fwd_prob/output_batch_major",
                delta_encoder_posteriors = "tdps/delta_encoder_output/output_batch_major",
                delta_encoder_output = "tdps/concat_lstm_fwd_lstm_bwd/concat_sources/concat"
            )
        return res
    
    def make_compile_config(self, train_config, arch):
        assert arch in {"label", "blstm"}
        compile_config = copy.deepcopy(train_config)
        compile_config.config["network"]["encoder_output"] = {"class": "copy", "from": ["fwd_6", "bwd_6"], "is_output_layer": True}
        del compile_config.config["network"]["fast_bw"], compile_config.config["network"]["output_bw"]
        try:
            del compile_config.config["network"]["combine_prior"]
            del compile_config.config["network"]["accumulate_prior"]
        except KeyError:
            pass
        for key in list(compile_config.config.keys()):
            if key not in ["network", "num_outputs", "extern_data", "target"]:
                del compile_config.config[key]
        if "blstm" == arch:
            compile_config.config["network"]["tdps"]["subnetwork"]["delta_encoder_output"] = {"class": "copy", "from": ["lstm_fwd", "lstm_bwd"], "is_output_layer": True}
            compile_config.config["network"]["tdps"]["subnetwork"]["fwd_prob"]["activation"] = "sigmoid"
        return compile_config
    
    @contextlib.contextmanager
    def use_flf_tool(self, exe):
        if exe is None:
            yield self.system
            return
        old_exe = self.system.crp["dev"].flf_tool_exe
        self.system.crp["dev"].flf_tool_exe = exe
        yield self.system
        self.system.crp["dev"].flf_tool_exe = old_exe
    
    def recognize_arch(self, config):
        net = config.config["network"]
        if "tdps" not in net:
            return None
        tdps = net["tdps"]["subnetwork"]
        if "lstm_fwd" in tdps and "lstm_bwd" in tdps:
            return "blstm"
        return "label"
    
    def get_context_type(self, arch):
        return {
            "blstm": ContextEnum.mono_state_transition,
            "label": ContextEnum.mono_state_transition,
            None: ContextEnum.monophone
        }[arch]
    
    def get_transition_type(self, arch):
        return {
            "blstm": TransitionType.feature,
            "label": TransitionType.label,
            None: None
        }[arch]
    
    def get_feature_flow(self, corpus, name, arch):
        decoder = FHDecoder(
            name=name,
            search_crp=system.crp[corpus],
            context_type=context_type,
            context_mapper=ContextMapper(),
            feature_path=system.feature_flows[corpus][feature_flow_key],
            model_path=system.nn_checkpoints[train_corpus][name][epoch],
            graph=compile_graph,
            mixtures=None,
            tf_library=system.native_ops_path,
            eval_files=system.scorer_args[dev_corpus],
            is_multi_encoder_output=is_multi_encoder_output,
            tensor_mapping=self.get_tensor_mapping(context_type, transition_type)
        )
        self.flows[dev_corpus][name][epoch] = flow = decoder.featureScorerFlow
        return flow

    def decode(
        self,
        name,
        epoch,
        crnn_config,
        training_args,
        scorer_args,
        recognition_args,
        extra_suffix="-fh",
        recog_name=None,
        compile_args=None,
        compile_crnn_config=None,
        reestimate_prior=False,
        optimize=True,
        decoding_args=None,
        _adjust_train_args=True,
        clean=False,
        extra_compile_args={},
        **_ignored
    ):
        system = self.system
        if _adjust_train_args:
            if compile_args is None:
                compile_args = {}
            training_args = {**system.default_nn_training_args, **training_args}
        
        feature_flow_key = training_args["feature_flow"]
        train_corpus = training_args["feature_corpus"]
        decoding_args = ChainMap({}, decoding_args or {}, self.default_decoding_args)
        recog_args = ChainMap(
            recognition_args.copy(),
            self.default_recognition_args,
            system.default_recognition_args
        )

        dev_corpus = recog_args["corpus"]

        train_config = self.system.nn_config_dicts["train"][name]
        recognized_arch = self.recognize_arch(train_config)

        # context_type = decoding_args["context_type"]
        context_type = decoding_args.get("context_type", self.get_context_type(recognized_arch))
        # prior_info = decoding_args["prior_info"]

        transition_type = decoding_args.get("transition_type", self.get_transition_type(recognized_arch))
        if context_type == ContextEnum.mono_state_transition:
            if transition_type is None:
                transition_type = self.recognize_arch(train_config)
        is_multi_encoder_output=decoding_args.get("is_multi_encoder_output", recognized_arch == "blstm")

        n_mixtures = decoding_args.get(
            "n_mixtures",
            self.num_mixtures()
        )

        override_compile_config = decoding_args.get("compile", False)
        if override_compile_config:
            compile_crnn_config = None

        # get tf graph
        compile_args = ChainMap(
            dict(adjust_output_layer=False),
            decoding_args.get("extra_compile_args", {}),
            compile_args)
        if isinstance(compile_crnn_config, str):
            compile_graph, _ = system.compile_graph[train_corpus][compile_crnn_config]
        else:
            if compile_crnn_config is None:
                compile_crnn_config = self.make_compile_config(train_config, recognized_arch)
            compile_graph, _ = system.compile_model(
                returnn_config = compile_crnn_config,
                # alias = recog_name or name,
                alias = (recog_name or name) + extra_suffix,
                **compile_args,
            )

        def decoder_fn(corpus):
            return FHDecoder(
                name=name,
                search_crp=system.crp[corpus],
                context_type=context_type,
                context_mapper=ContextMapper(),
                feature_path=system.feature_flows[corpus][feature_flow_key],
                model_path=system.nn_checkpoints[train_corpus][name][epoch],
                graph=compile_graph,
                mixtures=None,
                tf_library=system.native_ops_path if self.use_native_ops else None,
                # eval_files=system.scorer_args[corpus],
                eval_files=None,
                is_multi_encoder_output=is_multi_encoder_output,
                tensor_mapping=self.get_tensor_mapping(context_type, transition_type)
            )
        self.decoders[name][epoch] = decoder_fn
        decoder = FHDecoder(
            name=name,
            search_crp=system.crp[dev_corpus],
            context_type=context_type,
            context_mapper=ContextMapper(),
            feature_path=system.feature_flows[dev_corpus][feature_flow_key],
            model_path=system.nn_checkpoints[train_corpus][name][epoch],
            graph=compile_graph,
            mixtures=None,
            tf_library=system.native_ops_path if self.use_native_ops else None,
            # eval_files=system.scorer_args[dev_corpus],
            eval_files=None,
            is_multi_encoder_output=is_multi_encoder_output,
            tensor_mapping=self.get_tensor_mapping(context_type, transition_type)
        )
        # self.flows[name][epoch] = flow_fn

        def fh_feature_scorer(name, prior_mixtures, scale, prior_file, priori_scale, **kwargs):
            prior_info={
                "center-state-prior": {"file": prior_file, "scale": priori_scale},
                "left-context-prior": {"file": None},
                "right-context-prior": {"file": None},
            }
            tying = system.crp["base"].acoustic_model_config
            states_per_phone = tying.hmm.states_per_phone
            feature_scorer = get_feature_scorer(
                context_type=context_type,
                context_mapper=ContextMapper(),
                featureScorerConfig=decoder.featureScorerConfig,
                mixtures=CreateDummyMixturesJob(n_mixtures, system.num_input).out_mixtures,
                silence_id=DelayedPhonemeIndex(self.system.crp["dev"].lexicon_config.file, "[SILENCE]"),
                is_multi_encoder_output=is_multi_encoder_output,
                prior_info=prior_info,
                forward_scale=scorer_args.get("fwd_loop_scale", 0.7),
                loop_scale=scorer_args.get("fwd_loop_scale", 0.7),
                use_word_end_classes=system.crp["base"].acoustic_model_config.state_tying.use_word_end_classes,
                use_boundary_classes=system.crp["base"].acoustic_model_config.state_tying.use_boundary_classes,
                num_states_per_phone=states_per_phone,
                num_label_contexts=scorer_args.get("num_label_contexts", self.num_phonemes()),
            )
            if context_type == ContextEnum.mono_state_transition:
                try:
                    assert isinstance(transition_type, TransitionType), "transition_type must be of type TransitionType"
                    feature_scorer.config.transition_type = transition_type.value
                except KeyError:
                    raise KeyError("transition_type must be set for context type mono_state_transition")
            self.system.feature_scorers[dev_corpus][name] = feature_scorer
            self.system.feature_scorers[train_corpus][name] = feature_scorer
            return feature_scorer

        extra_config = recog_args.get("extra_config", rasr.RasrConfig())
        extra_config["*"].python_home = "/work/tools/asr/python/3.8.0"
        extra_config["*"].python_program_name = "/work/tools/asr/python/3.8.0/bin/python3.8"
        extra_config["flf-lattice-tool.network.recognizer.acoustic-model.state-tying"].type = "no-tying-dense"

        recog_args.update(
            extra_config=extra_config,
            flow=decoder.featureScorerFlow,
        )

        with self.use_flf_tool(decoding_args.get("flf_tool_exe", None)) as system:
            system.decode(
                name=name,
                epoch=epoch,
                crnn_config=crnn_config,
                training_args=training_args,
                scorer_args=scorer_args,
                recognition_args=recog_args.maps[0],
                extra_suffix=extra_suffix,
                recog_name=recog_name,
                compile_args=compile_args,
                compile_crnn_config=compile_crnn_config,
                reestimate_prior=reestimate_prior,
                optimize=optimize,
                clean=clean,
                _feature_scorer=fh_feature_scorer,
                _adjust_train_args=False,
            )
    
    def nn_align(
		self,
        nn_name,
        returnn_config,
        epoch,
        scorer_suffix='',
        mem_rqmt=8,
		compile_crnn_config=None,
        name=None,
        feature_flow='gt',
        dump=False,
		graph=None,
        time_rqmt=4,
		compile_args=None,
        evaluate=False,
        feature_scorer=None,
        **kwargs
    ):
        pass