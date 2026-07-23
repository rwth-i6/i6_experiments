import torch
import numpy as np

from dataclasses import fields

SHOULD_LOG = True

def get_config_value(config, key, default=None, dtype=None):
    tobetype = dtype if dtype is not None else type(default) if default is not None else None
    val = config[key]
    if val is None:
        return default
    if SHOULD_LOG:
        print(f"config[{key}] = {val}", tobetype)
    if tobetype is not None:
        return tobetype(val)
    else:
        return val

class FixedContextTransducerPy:

    def __init__(self, config):
        if SHOULD_LOG:
            config.enable_logging()
        base_selection = config.get_selection()
        self._history_length = get_config_value(config, "history-length", 1)
        self._start_label_index = get_config_value(config, "start-label-index", 0)
        self._blank_updates_history = get_config_value(config, "blank-updates-history", False)
        self._loop_updates_history = get_config_value(config, "loop-updates-history", False)
        self._vertical_label_transition = get_config_value(config,"vertical-label-transition", False)

        config.set_selection(f"{base_selection}.recognition")
        experiment = get_config_value(config, "experiment", None)
        if experiment is None:
            raise ValueError("recognition.experiment must be specified in the config")
        
        if "qat" in experiment:
            prev_selection = config.get_selection()
            config.set_selection(f"{base_selection}.qat")
            # TODO: QAT params are semi-hardcoded in the label scoring config
            qat_params = dict(
                weight_bit_prec=get_config_value(config, "weight-bit-prec", dtype=int),
                activation_bit_prec=get_config_value(config, "activation-bit-prec", dtype=int),
                weight_dropout=get_config_value(config, "weight-dropout", dtype=float),
                weight_pruning_config=get_config_value(config, "weight-pruning-config"),
            )

            # TODO: weight_pruning_config cannot be anything but none atm
            assert qat_params["weight_pruning_config"] is None, "weight_pruning_config non None configuration is not supported"
            config.set_selection(prev_selection)

        if experiment == "ffnn_transducer_qat_encoder":
            from ..ffnn_transducer_qat_encoder.pytorch_modules import FFNNTransducerQATEncoderScorer, FFNNTransducerQATEncoderRecogConfig
            from ...experiments.librispeech.training.ffnn_transducer_qat_encoder_bpe import get_model_config

            ilm_scale = get_config_value(config, "ilm-scale", 0.0)
            blank_penalty = get_config_value(config, "blank-penalty", 0.0)
            model_config = get_model_config(**qat_params)
            recog_model_config = FFNNTransducerQATEncoderRecogConfig(
                **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
                ilm_scale=ilm_scale,
                blank_penalty=blank_penalty,
            )
            checkpoint = get_config_value(config, "model-path", None)
            if checkpoint is None:
                raise ValueError("recognition.model-path must be specified in the config")
            
            scorer = FFNNTransducerQATEncoderScorer(cfg=recog_model_config)
            ckpt = torch.load(checkpoint, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = scorer.load_state_dict(state_dict, strict=False)
            if len(missing) > 0:
                print(f"missing keys in state_dict: {missing}")
            scorer.eval()
            
            self._model = scorer
        
        elif experiment == "qat_ffnn_transducer":
            from ..qat_ffnn_transducer.pytorch_modules import QATFFNNTransducerScorer, QATFFNNTransducerRecogConfig
            from ...experiments.librispeech.training.qat_ffnn_transducer_bpe import get_model_config

            ilm_scale = get_config_value(config, "ilm_scale", 0.0)
            blank_penalty = get_config_value(config, "blank-penalty", 0.0)
            model_config = get_model_config(**qat_params)
            recog_model_config = QATFFNNTransducerRecogConfig(
                **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
                ilm_scale=ilm_scale,
                blank_penalty=blank_penalty,
            )
            checkpoint = get_config_value(config, "model-path", None)
            if checkpoint is None:
                raise ValueError("recognition.model-path must be specified in the config")
            
            scorer = QATFFNNTransducerScorer(cfg=recog_model_config)
            ckpt = torch.load(checkpoint, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = scorer.load_state_dict(state_dict, strict=False)
            if len(missing) > 0:
                print(f"missing keys in state_dict: {missing}")
            scorer.eval()
            
            self._model = scorer
        config.set_selection(base_selection)
        
        self._encoder_states = None

        self._score_cache = {}

        self._inputs = []
        self._expect_more_features = True

    def allowed_transition_types(self):
        from librasr import TransitionType
        transition_types = [
            TransitionType.BLANK_TO_LABEL,
            TransitionType.LABEL_TO_LABEL,
            TransitionType.LABEL_TO_BLANK,
            TransitionType.BLANK_LOOP,
            TransitionType.LABEL_LOOP,
            TransitionType.INITIAL_LABEL,
            TransitionType.INITIAL_BLANK,
            TransitionType.SENTENCE_END,
        ]
        return transition_types

    def reset(self):
        self._encoder_states = None
        self._score_cache.clear()
        self._inputs.clear()
        self._expect_more_features = True

    def signal_no_more_features(self):
        self._expect_more_features = False

    def get_initial_scoring_context(self):
        return (0, tuple([self._start_label_index] * self._history_length))

    def extended_scoring_context(self, context, next_token, transition_type):
        from librasr import TransitionType

        update_history = False
        increment_time = False

        def push_token(history, next_token):
            return history[1:] + (next_token,)
        
        step, history = context
        
        if transition_type in [TransitionType.BLANK_LOOP]:
            update_history = self._blank_updates_history and self._loop_updates_history
            increment_time = True

        elif transition_type in [TransitionType.LABEL_TO_BLANK, TransitionType.INITIAL_BLANK]:
            update_history = self._blank_updates_history
            increment_time = True

        elif transition_type in [TransitionType.LABEL_LOOP]:
            update_history = self._loop_updates_history
            increment_time = not self._vertical_label_transition

        elif transition_type in [TransitionType.BLANK_TO_LABEL, TransitionType.LABEL_TO_LABEL, TransitionType.INITIAL_LABEL, TransitionType.SENTENCE_END]:
            update_history = True
            increment_time = not self._vertical_label_transition
        else:
            raise ValueError(f"Unsupported transition type {transition_type}")
        
        if not update_history and not increment_time:
            return context
    
        if not update_history and increment_time:
            return (step + 1, history)

        return (step + increment_time, push_token(history, next_token))
        
    def add_inputs(self, inputs):
        for t in range(inputs.shape[0]):
            self._inputs.append(inputs[t])   # each is [D]

        # Rebuild flat array for fast [step] indexing
        if len(self._inputs) > 0:
            self._encoder_states = np.stack(self._inputs, axis=0)  # [total_T, D]

    def compute_scores_with_times(self, contexts):
        results = []
        for ctx in contexts:
            step, history = ctx

            if step >= len(self._inputs):
                results.append(None)
                continue

            if ctx in self._score_cache:
                scores = self._score_cache[ctx]
                results.append((list(scores), step))
                continue

            enc = self._inputs[step]  # [D]

            enc_tensor = torch.from_numpy(enc).unsqueeze(0).float()  # [1, D]
            hist_tensor = torch.tensor([history], dtype=torch.long)  # [1, H]

            with torch.no_grad():
                scores_tensor = self._model(enc_tensor, hist_tensor)      # [1, V]
                scores = scores_tensor.squeeze(0).cpu().numpy()            # [V]

            self._score_cache[ctx] = scores
            results.append((list(scores), step))
        return results

# class StatefulTransducerPy:
#         def __init__(self, config):
#             if SHOULD_LOG:
#                 config.enable_logging()
#             base_selection = config.get_selection()
#             self._history_length = get_config_value(config, "history-length", 1)
#             self._start_label_index = get_config_value(config, "start-label-index", 0)
#             self._blank_updates_history = get_config_value(config, "blank-updates-history", False)
#             self._loop_updates_history = get_config_value(config, "loop-updates-history", False)
#             self._vertical_label_transition = get_config_value(config,"vertical-label-transition", False)

#             config.set_selection(f"{base_selection}.recognition")
#             experiment = get_config_value(config, "experiment", None)
#             if experiment is None:
#                 raise ValueError("recognition.experiment must be specified in the config")
            
#             if "qat" in experiment:
#                 prev_selection = config.get_selection()
#                 config.set_selection(f"{base_selection}.qat")
#                 # TODO: QAT params are semi-hardcoded in the label scoring config
#                 qat_params = dict(
#                     weight_bit_prec=get_config_value(config, "weight-bit-prec", dtype=int),
#                     activation_bit_prec=get_config_value(config, "activation-bit-prec", dtype=int),
#                     weight_dropout=get_config_value(config, "weight-dropout", dtype=float),
#                     weight_pruning_config=get_config_value(config, "weight-pruning-config"),
#                 )

#                 # TODO: weight_pruning_config cannot be anything but none atm
#                 assert qat_params["weight_pruning_config"] is None, "weight_pruning_config non None configuration is not supported"
#                 config.set_selection(prev_selection)

#             if experiment == "full_ctx_transducer_qat_encoder":
#                 from ..ffnn_transducer_qat_encoder.pytorch_modules import FFNNTransducerQATEncoderScorer, FFNNTransducerQATEncoderRecogConfig
#                 from ...experiments.librispeech.training.ffnn_transducer_qat_encoder_bpe import get_model_config

#                 ilm_scale = get_config_value(config, "ilm-scale", 0.0)
#                 blank_penalty = get_config_value(config, "blank-penalty", 0.0)
#                 model_config = get_model_config(**qat_params)
#                 recog_model_config = FFNNTransducerQATEncoderRecogConfig(
#                     **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
#                     ilm_scale=ilm_scale,
#                     blank_penalty=blank_penalty,
#                 )
#                 checkpoint = get_config_value(config, "model-path", None)
#                 if checkpoint is None:
#                     raise ValueError("recognition.model-path must be specified in the config")
                
#                 scorer = FFNNTransducerQATEncoderScorer(cfg=recog_model_config)
#                 ckpt = torch.load(checkpoint, map_location="cpu")
#                 state_dict = ckpt.get("model", ckpt)
#                 missing, unexpected = scorer.load_state_dict(state_dict, strict=False)
#                 if len(missing) > 0:
#                     print(f"missing keys in state_dict: {missing}")
#                 scorer.eval()
                
#                 self._model = scorer

#     def allowed_transition_types(self):
#         return []
    
#     def reset(self):
#         self._state = None
    
#     def signal_no_more_features(self):
#         pass

#     def get_initial_scoring_context(self):
#         return []
    
#     def extended_scoring_context(self, context, next_token, transition_type):
#         return []
    
#     def add_inputs(self, inputs):
#         pass

#     def compute_scores_with_times(self, contexts):
#         return []

#     def set_state(self, state):
#         self._state = state



def register_pyscorers():
    
    from librasr import LabelScorer, register_label_scorer_type
    class PyFixedContextTransducerPy(FixedContextTransducerPy, LabelScorer):
        def __init__(self, config):
            LabelScorer.__init__(self, config)
            FixedContextTransducerPy.__init__(self, config)
    register_label_scorer_type("fixed-context-py", PyFixedContextTransducerPy)
