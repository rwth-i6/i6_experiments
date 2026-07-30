import torch
import numpy as np

from dataclasses import fields

SHOULD_LOG = True

_TransitionType = None


def _get_transition_types():
    global _TransitionType
    if _TransitionType is None:
        from librasr import TransitionType as _TT

        _TransitionType = _TT
    return _TransitionType


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


_MODEL_CACHE = {}


def _get_eval_scorer(scorer_cls, recog_cfg_cls, get_model_config, qat_params, ilm_scale, blank_penalty, checkpoint, experiment):
    cache_key = ("?", experiment, checkpoint, ilm_scale, blank_penalty) 
    print(f"get_eval_scorer: cache_key={cache_key}")
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    model_config = get_model_config(**qat_params)
    recog_model_config = recog_cfg_cls(
        **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
        ilm_scale=ilm_scale,
        blank_penalty=blank_penalty,
    )
    scorer = scorer_cls(cfg=recog_model_config)
    ckpt = torch.load(checkpoint, map_location="cpu")  # TODO: can be changed to 'device'
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = scorer.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"missing keys in state_dict: {missing}")
    scorer.eval()
    _MODEL_CACHE[cache_key] = scorer
    return scorer


class FixedContextTransducerPy:

    def __init__(self, config):

        if SHOULD_LOG:
            config.enable_logging()
        base_selection = config.get_selection()
        self._history_length = get_config_value(config, "history-length", 1)
        self._start_label_index = get_config_value(config, "start-label-index", 0)
        self._blank_updates_history = get_config_value(config, "blank-updates-history", False)
        self._loop_updates_history = get_config_value(config, "loop-updates-history", False)
        self._vertical_label_transition = get_config_value(config, "vertical-label-transition", False)

        config.set_selection(f"{base_selection}.recognition")
        experiment = get_config_value(config, "experiment", None)
        if experiment is None:
            raise ValueError("recognition.experiment must be specified in the config")
        print("read experiment", experiment)
        self._model = self.get_scorer(config, experiment)

        self._score_cache = {}

        self._inputs = []
        self._expect_more_features = True

    def get_scorer(self, config, experiment):
        base_selection = config.get_selection()
        print(f"get_scorer: base_selection={base_selection}, experiment={experiment}")
        if "qat" in experiment:
            config.set_selection(f"{base_selection}.qat")
            # TODO: QAT params are semi-hardcoded in the label scoring config
            qat_params = dict(
                weight_bit_prec=get_config_value(config, "weight-bit-prec", dtype=int),
                activation_bit_prec=get_config_value(config, "activation-bit-prec", dtype=int),
                weight_dropout=get_config_value(config, "weight-dropout", dtype=float),
                weight_pruning_config=get_config_value(config, "weight-pruning-config"),
            )

            # TODO: weight_pruning_config cannot be anything but none atm
            assert (
                qat_params["weight_pruning_config"] is None
            ), "weight_pruning_config non None configuration is not supported"
            config.set_selection(base_selection)

        if experiment == "ffnn_transducer_qat_encoder":
            from ..ffnn_transducer_qat_encoder.pytorch_modules import (
                FFNNTransducerQATEncoderScorer,
                FFNNTransducerQATEncoderRecogConfig,
            )
            from ...experiments.librispeech.training.ffnn_transducer_qat_encoder_bpe import get_model_config

            ilm_scale = get_config_value(config, "ilm_scale", 0.0)
            blank_penalty = get_config_value(config, "blank-penalty", 0.0)
            checkpoint = get_config_value(config, "model-path", None)
            if checkpoint is None:
                raise ValueError("recognition.model-path must be specified in the config")

            print(f"get_scorer: experiment={experiment}, checkpoint={checkpoint}")
            scorer = _get_eval_scorer(
                FFNNTransducerQATEncoderScorer,
                FFNNTransducerQATEncoderRecogConfig,
                get_model_config,
                qat_params,
                ilm_scale,
                blank_penalty,
                checkpoint,
                experiment,
            )

        # elif experiment == "qat_ffnn_transducer":
        #     from ..qat_ffnn_transducer.pytorch_modules import QATFFNNTransducerScorer, QATFFNNTransducerRecogConfig
        #     from ...experiments.librispeech.training.qat_ffnn_transducer_bpe import get_model_config

        #     ilm_scale = get_config_value(config, "ilm-scale", 0.0)
        #     blank_penalty = get_config_value(config, "blank-penalty", 0.0)
        #     model_config = get_model_config(**qat_params)
        #     recog_model_config = QATFFNNTransducerRecogConfig(
        #         **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
        #         ilm_scale=ilm_scale,
        #         blank_penalty=blank_penalty,
        #     )
        #     checkpoint = get_config_value(config, "model-path", None)
        #     if checkpoint is None:
        #         raise ValueError("recognition.model-path must be specified in the config")

        #     scorer = QATFFNNTransducerScorer(cfg=recog_model_config)
        #     ckpt = torch.load(checkpoint, map_location="cpu")
        #     state_dict = ckpt.get("model", ckpt)
        #     missing, unexpected = scorer.load_state_dict(state_dict, strict=False)
        #     if len(missing) > 0:
        #         print(f"missing keys in state_dict: {missing}")

        elif experiment == "ffnn_transducer":
            from ..ffnn_transducer.pytorch_modules import FFNNTransducerScorer, FFNNTransducerRecogConfig
            from ...experiments.librispeech.training.ffnn_transducer_bpe import get_model_config

            ilm_scale = get_config_value(config, "ilm-scale", 0.0)
            blank_penalty = get_config_value(config, "blank-penalty", 0.0)
            model_config = get_model_config()
            recog_config = FFNNTransducerRecogConfig(
                **model_config.__dict__,
                ilm_scale=ilm_scale,
                blank_penalty=blank_penalty,
            )
            scorer = FFNNTransducerScorer(cfg=recog_config)

            checkpoint = get_config_value(config, "model-path", None)
            if checkpoint is None:
                raise ValueError("recognition.model-path must be specified in the config")

            missing, unexpected = scorer.load_state_dict(
                torch.load(str(checkpoint), map_location="cpu")["model"], strict=False
            )
            if len(missing) > 0:
                print(f"missing keys in state_dict: {missing}")
        else:
            raise ValueError(f"Unsupported experiment type {experiment}")
        scorer.eval()
        config.set_selection(base_selection)
        return scorer

    def allowed_transition_types(self):
        TT = _get_transition_types()
        return [
            TT.BLANK_TO_LABEL,
            TT.LABEL_TO_LABEL,
            TT.LABEL_TO_BLANK,
            TT.BLANK_LOOP,
            TT.LABEL_LOOP,
            TT.INITIAL_LABEL,
            TT.INITIAL_BLANK,
            TT.SENTENCE_END,
        ]

    def reset(self):
        self._score_cache.clear()
        self._inputs.clear()
        self._expect_more_features = True

    def signal_no_more_features(self):
        self._expect_more_features = False

    def get_initial_scoring_context(self):
        return (0, tuple([self._start_label_index] * self._history_length))

    def extended_scoring_context(self, context, next_token, transition_type):
        TT = _get_transition_types()
        blank_updates = self._blank_updates_history
        loop_updates = self._loop_updates_history
        vertical = self._vertical_label_transition

        update_history = False
        increment_time = 0

        if transition_type == TT.BLANK_LOOP:
            update_history = blank_updates and loop_updates
            increment_time = 1
        elif transition_type == TT.LABEL_TO_BLANK or transition_type == TT.INITIAL_BLANK:
            update_history = blank_updates
            increment_time = 1
        elif transition_type == TT.LABEL_LOOP:
            update_history = loop_updates
            increment_time = 0 if vertical else 1
        elif (
            transition_type == TT.BLANK_TO_LABEL
            or transition_type == TT.LABEL_TO_LABEL
            or transition_type == TT.INITIAL_LABEL
            or transition_type == TT.SENTENCE_END
        ):
            update_history = True
            increment_time = 0 if vertical else 1
        else:
            raise ValueError(f"Unsupported transition type {transition_type}")

        step, history = context

        if not update_history:
            return context if increment_time == 0 else (step + 1, history)

        new_history = history[1:] + (next_token,) if update_history else history
        return (step + increment_time, new_history)

    def add_inputs(self, inputs):
        self._inputs.extend(inputs[t] for t in range(inputs.shape[0]))  # per-step [D] views

    def compute_scores_with_times(self, contexts):
        cache = self._score_cache
        inputs = self._inputs
        n_inputs = len(inputs)
        results = [None] * len(contexts)
        to_score = {}

        for idx, ctx in enumerate(contexts):
            step, history = ctx

            if step >= n_inputs:
                continue

            cached = cache.get(ctx)
            if cached is not None:
                results[idx] = (cached, step)
                continue

            to_score.setdefault(step, []).append((idx, ctx, history))

        if not to_score:
            return results

        with torch.no_grad():
            for step, items in to_score.items():
                enc = self._inputs[step]  # [D]
                B = len(items)
                enc_tensor = torch.from_numpy(enc).unsqueeze(0).float()  # [1, D] (model expands internally)
                hist_flat = [tok for _, _, h in items for tok in h]  # [B * H]
                hist_tensor = torch.tensor(hist_flat, dtype=torch.long).view(B, -1)  # [B, H]

                scores_tensor = self._model(enc_tensor, hist_tensor)  # [B, V]
                scores_lists = scores_tensor.tolist()  # [B] -> list[float] of len V

                for b, (idx, ctx, _) in enumerate(items):
                    s = scores_lists[b]
                    cache[ctx] = s
                    results[idx] = (s, step)

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
