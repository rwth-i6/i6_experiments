from dataclasses import fields
import numpy as np
import torch

import synaptogen_ml
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

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
    if tobetype is bool and isinstance(val, str):
        return val.lower() in ("true", "yes", "1", "on")
    if tobetype is not None:
        return tobetype(val)
    else:
        return val


def _get_device(config, base_selection):
    device_val = get_config_value(config, "device", None)
    if device_val is None:
        config.set_selection(f"{base_selection}.recognition")
        device_val = get_config_value(config, "device", None)
        config.set_selection(base_selection)
    return torch.device(device_val if device_val is not None else "cpu")


_MODEL_CACHE = {}


def _parse_dac_adc_settings(value):
    if isinstance(value, (list, tuple)):
        vals = [str(v) for v in value]
    else:
        vals = str(value).split()
    assert len(vals) == 5, f"expected 5 converter hardware settings values, got {vals!r}"
    return DacAdcHardwareSettings(
        input_bits=int(vals[0]),
        output_precision_bits=int(vals[1]),
        output_range_bits=int(vals[2]),
        hardware_input_vmax=float(vals[3]),
        hardware_output_current_scaling=float(vals[4]),
    )


def _parse_correction_settings(value):
    if isinstance(value, (list, tuple)):
        vals = [str(v) for v in value]
    else:
        vals = str(value).split()
    assert len(vals) == 4, f"expected 4 cycle correction settings values, got {vals!r}"

    def _opt_float(x):
        return None if x.lower() in ("none", "null") else float(x)

    return CycleCorrectionSettings(
        num_cycles=None if vals[0].lower() in ("none", "null") else int(vals[0]),
        test_input_value=_opt_float(vals[1]),
        relative_deviation=_opt_float(vals[2]),
        ideal_programming=vals[3].lower() in ("yes", "true", "1", "on"),
    )


def _get_memristor_params(config, base_selection):
    memristor_params = {}
    try:
        config.set_selection(f"{base_selection}.memristor")
        converter = get_config_value(config, "converter-hardware-settings", None)
        if converter is not None:
            memristor_params["converter_hardware_settings"] = _parse_dac_adc_settings(converter)
        pos_enc = get_config_value(config, "pos-enc-converter-hardware-settings", None)
        if pos_enc is not None:
            memristor_params["pos_enc_converter_hardware_settings"] = _parse_dac_adc_settings(pos_enc)
        num_cycles = get_config_value(config, "num-cycles", None, dtype=int)
        if num_cycles is not None:
            memristor_params["num_cycles"] = num_cycles
        correction = get_config_value(config, "correction-settings", None)
        if correction is not None:
            memristor_params["correction_settings"] = _parse_correction_settings(correction)
    except Exception:
        pass
    finally:
        config.set_selection(base_selection)
    return memristor_params


def _get_eval_scorer(
    scorer_cls,
    recog_cfg_cls,
    get_model_config,
    qat_params,
    ilm_scale,
    blank_penalty,
    checkpoint,
    device,
    memristor_params=None,
):
    memristor_params = memristor_params or {}
    cache_key = (checkpoint, ilm_scale, blank_penalty, str(device), repr(memristor_params))
    if SHOULD_LOG:
        print(f"get_eval_scorer: cache_key={cache_key}")
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    model_config = get_model_config(**qat_params) if qat_params else get_model_config()
    if memristor_params:
        model_config = model_config.with_replaced(**memristor_params)
    recog_model_config = recog_cfg_cls(
        **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
        ilm_scale=ilm_scale,
        blank_penalty=blank_penalty,
    )
    scorer = scorer_cls(cfg=recog_model_config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
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
        self._silence_updates_history = get_config_value(config, "silence-updates-history", False)
        self._loop_updates_history = get_config_value(config, "loop-updates-history", False)
        self._vertical_label_transition = get_config_value(config, "vertical-label-transition", False)
        self._max_batch_size = get_config_value(config, "max-batch-size", 2147483647, dtype=int)

        self._device = _get_device(config, base_selection)

        config.set_selection(f"{base_selection}.recognition")

        self._model = self.get_scorer(config)

        self._score_cache = {}
        self._enc_dev_cache = {}
        self._inputs = []
        self._expect_more_features = True
        synaptogen_ml.set_fast_inference(True)

    def get_scorer(self, config):
        base_selection = config.get_selection()
        qat_params = {}

        try:
            config.set_selection(f"{base_selection}.qat")
            w_prec = get_config_value(config, "weight-bit-prec", None, dtype=int)
            if w_prec is not None:
                qat_params = dict(
                    weight_bit_prec=w_prec,
                    activation_bit_prec=get_config_value(config, "activation-bit-prec", dtype=int),
                    weight_dropout=get_config_value(config, "weight-dropout", dtype=float),
                    weight_pruning_config=get_config_value(config, "weight-pruning-config"),
                )
                assert (
                    qat_params["weight_pruning_config"] is None
                ), "weight_pruning_config non None configuration is not supported"
        except Exception:
            pass
        finally:
            config.set_selection(base_selection)

        memristor_params = _get_memristor_params(config, base_selection)

        ilm_scale = get_config_value(config, "ilm-scale", dtype=float)
        blank_penalty = get_config_value(config, "blank-penalty", dtype=float)
        checkpoint = get_config_value(config, "model-path", None)

        import_val = get_config_value(config, "imports", None)

        if isinstance(import_val, (list, tuple)):
            code_str = "\n".join(
                item.get().strip() if hasattr(item, "get") else str(item).strip() for item in import_val
            )
        else:
            code_str = str(import_val)

        lines = [
            line.strip()
            for line in code_str.replace(";", "\n").replace(" from ", "\nfrom ").split("\n")
            if line.strip()
        ]
        ns = {}
        exec("\n".join(lines), globals(), ns)

        scorer_cls = ns.get("ScorerModel")
        recog_cfg_cls = ns.get("RecogConfig")
        get_model_config = ns.get("get_model_config")

        scorer = _get_eval_scorer(
            scorer_cls,
            recog_cfg_cls,
            get_model_config,
            qat_params,
            ilm_scale,
            blank_penalty,
            checkpoint,
            self._device,
            memristor_params=memristor_params,
        )
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
            TT.SILENCE_LOOP,
            TT.LABEL_TO_SILENCE,
            TT.INITIAL_SILENCE,
            TT.SILENCE_TO_LABEL,
        ]

    def reset(self):
        self._score_cache.clear()
        self._enc_dev_cache.clear()
        self._inputs.clear()
        self._expect_more_features = True

    def signal_no_more_features(self):
        self._expect_more_features = False

    def get_initial_scoring_context(self):
        return (0, tuple([self._start_label_index] * self._history_length))

    def extended_scoring_context(self, context, next_token, transition_type):
        TT = _get_transition_types()
        blank_updates = self._blank_updates_history
        silence_updates = self._silence_updates_history
        loop_updates = self._loop_updates_history
        vertical = self._vertical_label_transition

        update_history = False
        increment_time = 0

        if transition_type == TT.BLANK_LOOP:
            update_history = blank_updates and loop_updates
            increment_time = 1
        elif transition_type == TT.SILENCE_LOOP:
            update_history = silence_updates and loop_updates
            increment_time = 1
        elif transition_type in (TT.LABEL_TO_BLANK, TT.INITIAL_BLANK):
            update_history = blank_updates
            increment_time = 1
        elif transition_type in (TT.LABEL_TO_SILENCE, TT.INITIAL_SILENCE):
            update_history = silence_updates
            increment_time = 1
        elif transition_type == TT.LABEL_LOOP:
            update_history = loop_updates
            increment_time = 0 if vertical else 1
        elif transition_type in (
            TT.BLANK_TO_LABEL,
            TT.SILENCE_TO_LABEL,
            TT.LABEL_TO_LABEL,
            TT.INITIAL_LABEL,
            TT.SENTENCE_END,
        ):
            update_history = True
            increment_time = 0 if vertical else 1
        else:
            raise ValueError(f"Unsupported transition type {transition_type}")

        step, history = context

        if not update_history:
            return context if increment_time == 0 else (step + 1, history)

        new_history = history[1:] + (next_token,)
        return (step + increment_time, new_history)

    def add_inputs(self, inputs):
        self._inputs.extend(inputs[t] for t in range(inputs.shape[0]))  # per-step [D] views

    def compute_scores_with_times(self, contexts):
        synaptogen_ml.set_fast_inference(True)
        cache = self._score_cache
        enc_cache = self._enc_dev_cache
        inputs = self._inputs
        n_inputs = len(inputs)
        results = [None] * len(contexts)

        if contexts:
            min_step = min(ctx[0] for ctx in contexts)
            if min_step > 0:
                stale_keys = [k for k in cache if k[0] < min_step]
                for k in stale_keys:
                    del cache[k]
                stale_enc = [k for k in enc_cache if k < min_step]
                for k in stale_enc:
                    del enc_cache[k]

        by_step = {}
        for idx, ctx in enumerate(contexts):
            step, history = ctx
            if step >= n_inputs:
                continue
            cached_scores = cache.get(ctx)
            if cached_scores is not None:
                results[idx] = (cached_scores, step)
            else:
                by_step.setdefault(step, []).append((idx, ctx, history))

        if not by_step:
            return results

        device = self._device

        with torch.no_grad():
            max_bs = self._max_batch_size
            for step, entries in by_step.items():
                enc_tensor = enc_cache.get(step)
                if enc_tensor is None:
                    enc_tensor = torch.from_numpy(inputs[step]).float().unsqueeze(0).to(device)
                    enc_cache[step] = enc_tensor

                unique_ctxs = {}
                for _, ctx, history in entries:
                    if ctx not in unique_ctxs:
                        unique_ctxs[ctx] = history

                to_forward = list(unique_ctxs.items())
                for offset in range(0, len(to_forward), max_bs):
                    chunk = to_forward[offset : offset + max_bs]
                    B = len(chunk)

                    hist_flat = [tok for (_, h) in chunk for tok in h]
                    hist_tensor = torch.tensor(hist_flat, dtype=torch.long, device=device).view(B, -1)

                    scores_tensor = self._model(enc_tensor, hist_tensor)
                    scores_np = scores_tensor.detach().cpu().numpy()

                    for b, (ctx, _) in enumerate(chunk):
                        cache[ctx] = scores_np[b]

                for idx, ctx, _ in entries:
                    results[idx] = (cache[ctx], step)

        return results


class StatefulTransducerPy:
    """Stateful Transducer Scorer in Python.
    Features:
    - Score and state cache eviction for stale timesteps
    - Configurable PyTorch device parameter
    - Dynamic code execution of imports via config 'import' parameter
    - Batched state updater & scorer execution
    """

    def __init__(self, config):
        if SHOULD_LOG:
            config.enable_logging()
        base_selection = config.get_selection()
        self._history_length = get_config_value(config, "history-length", 1)
        self._start_label_index = get_config_value(config, "start-label-index", 0)
        self._blank_updates_history = get_config_value(config, "blank-updates-history", False)
        self._silence_updates_history = get_config_value(config, "silence-updates-history", False)
        self._loop_updates_history = get_config_value(config, "loop-updates-history", False)
        self._vertical_label_transition = get_config_value(config, "vertical-label-transition", False)
        self._max_batch_size = get_config_value(config, "max-batch-size", 2147483647, dtype=int)

        self._device = _get_device(config, base_selection)

        config.set_selection(f"{base_selection}.recognition")

        self._scorer, self._state_initializer, self._state_updater = self.get_scorer(config)

        self._score_cache = {}
        self._enc_dev_cache = {}
        self._state_cache = {}
        self._initial_hidden_state = None

        self._inputs = []
        self._expect_more_features = True
        synaptogen_ml.set_fast_inference(True)

    def get_scorer(self, config):
        rec_selection = config.get_selection()
        qat_params = {}
        try:
            config.set_selection(f"{rec_selection}.qat")
            w_prec = get_config_value(config, "weight-bit-prec", None, dtype=int)
            if w_prec is not None:
                qat_params = dict(
                    weight_bit_prec=w_prec,
                    activation_bit_prec=get_config_value(config, "activation-bit-prec", dtype=int),
                    weight_dropout=get_config_value(config, "weight-dropout", dtype=float),
                    weight_pruning_config=get_config_value(config, "weight-pruning-config"),
                )
                assert (
                    qat_params["weight_pruning_config"] is None
                ), "weight_pruning_config non None configuration is not supported"
        except Exception:
            pass
        finally:
            config.set_selection(rec_selection)

        ilm_scale = get_config_value(config, "ilm-scale", dtype=float)
        if ilm_scale is None:
            ilm_scale = get_config_value(config, "ilm_scale", 0.0, dtype=float)
        blank_penalty = get_config_value(config, "blank-penalty", dtype=float)
        if blank_penalty is None:
            blank_penalty = get_config_value(config, "blank_penalty", 0.0, dtype=float)

        checkpoint = get_config_value(config, "model-path", None)
        if checkpoint is None:
            raise ValueError("recognition.model-path must be specified in the config")

        import_val = (
            get_config_value(config, "imports", None)
            or get_config_value(config, "import", None)
            or get_config_value(config, "import-path", None)
        )
        if import_val is None:
            raise ValueError("Config must specify an 'imports' (or 'import') parameter.")

        if isinstance(import_val, (list, tuple)):
            code_str = "\n".join(
                item.get().strip() if hasattr(item, "get") else str(item).strip() for item in import_val
            )
        else:
            code_str = str(import_val)

        lines = [
            line.strip()
            for line in code_str.replace(";", "\n").replace(" from ", "\nfrom ").split("\n")
            if line.strip()
        ]
        ns = {}
        exec("\n".join(lines), globals(), ns)

        scorer_cls = ns.get("ScorerModel") or ns.get("Scorer")
        state_init_cls = ns.get("StateInitializerModel") or ns.get("StateInitializer")
        state_upd_cls = ns.get("StateUpdaterModel") or ns.get("StateUpdater")
        recog_cfg_cls = ns.get("RecogConfig")
        get_model_config = ns.get("get_model_config")

        if not (scorer_cls and state_init_cls and state_upd_cls and recog_cfg_cls and get_model_config):
            raise ValueError("Could not resolve required model components from imports.")

        model_config = get_model_config(**qat_params) if qat_params else get_model_config()
        recog_model_config = recog_cfg_cls(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            ilm_scale=ilm_scale,
            blank_penalty=blank_penalty,
        )

        ckpt = torch.load(checkpoint, map_location=self._device)
        state_dict = ckpt.get("model", ckpt)

        scorer = scorer_cls(cfg=recog_model_config).to(self._device)
        missing, unexpected = scorer.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"scorer missing keys in state_dict: {missing}")
        scorer.eval()

        state_initializer = state_init_cls(cfg=recog_model_config).to(self._device)
        missing, unexpected = state_initializer.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"state_initializer missing keys in state_dict: {missing}")
        state_initializer.eval()

        state_updater = state_upd_cls(cfg=recog_model_config).to(self._device)
        missing, unexpected = state_updater.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"state_updater missing keys in state_dict: {missing}")
        state_updater.eval()

        config.set_selection(rec_selection)
        return scorer, state_initializer, state_updater

    def allowed_transition_types(self):
        TT = _get_transition_types()
        return [
            TT.BLANK_LOOP,
            TT.SILENCE_LOOP,
            TT.LABEL_TO_BLANK,
            TT.INITIAL_BLANK,
            TT.LABEL_TO_SILENCE,
            TT.INITIAL_SILENCE,
            TT.LABEL_LOOP,
            TT.BLANK_TO_LABEL,
            TT.SILENCE_TO_LABEL,
            TT.LABEL_TO_LABEL,
            TT.INITIAL_LABEL,
            TT.SENTENCE_END,
        ]

    def reset(self):
        self._score_cache.clear()
        self._enc_dev_cache.clear()
        self._state_cache.clear()
        self._initial_hidden_state = None
        self._inputs.clear()
        self._expect_more_features = True

    def signal_no_more_features(self):
        self._expect_more_features = False

    def get_initial_scoring_context(self):
        return (0, ())

    def extended_scoring_context(self, context, next_token, transition_type):
        TT = _get_transition_types()
        blank_updates = self._blank_updates_history
        silence_updates = self._silence_updates_history
        loop_updates = self._loop_updates_history
        vertical = self._vertical_label_transition

        update_state = False
        increment_time = 0

        if transition_type == TT.BLANK_LOOP:
            update_state = blank_updates and loop_updates
            increment_time = 1
        elif transition_type == TT.SILENCE_LOOP:
            update_state = silence_updates and loop_updates
            increment_time = 1
        elif transition_type in (TT.LABEL_TO_BLANK, TT.INITIAL_BLANK):
            update_state = blank_updates
            increment_time = 1
        elif transition_type in (TT.LABEL_TO_SILENCE, TT.INITIAL_SILENCE):
            update_state = silence_updates
            increment_time = 1
        elif transition_type == TT.LABEL_LOOP:
            update_state = loop_updates
            increment_time = 0 if vertical else 1
        elif transition_type in (
            TT.BLANK_TO_LABEL,
            TT.SILENCE_TO_LABEL,
            TT.LABEL_TO_LABEL,
            TT.INITIAL_LABEL,
            TT.SENTENCE_END,
        ):
            update_state = True
            increment_time = 0 if vertical else 1
        else:
            raise ValueError(f"Unsupported transition type {transition_type}")

        step, label_seq = context

        if not update_state:
            return context if increment_time == 0 else (step + 1, label_seq)

        new_label_seq = label_seq + (next_token,)
        return (step + increment_time, new_label_seq)

    def add_inputs(self, inputs):
        self._inputs.extend(inputs[t] for t in range(inputs.shape[0]))  # per-step [D] views
        self._initial_hidden_state = None

    def _get_initial_hidden_state(self):
        if self._initial_hidden_state is None:
            with torch.no_grad():
                lstm_out, lstm_h, lstm_c = self._state_initializer()
                self._initial_hidden_state = (lstm_out, lstm_h, lstm_c)
        return self._initial_hidden_state

    def _get_state(self, label_seq):
        if not label_seq:
            return self._get_initial_hidden_state()
        return self._state_cache[label_seq]

    def _ensure_states_cached(self, label_seqs):
        missing_by_len = {}
        for seq in label_seqs:
            curr = seq
            while curr and curr not in self._state_cache:
                missing_by_len.setdefault(len(curr), set()).add(curr)
                curr = curr[:-1]

        if not missing_by_len:
            return

        device = self._device

        for length in sorted(missing_by_len.keys()):
            seqs_at_length = list(missing_by_len[length])
            parent_states = [self._get_state(seq[:-1]) for seq in seqs_at_length]
            tokens = [seq[-1] for seq in seqs_at_length]

            tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

            h_list = [ps[1] if ps[1].dim() == 3 else ps[1].unsqueeze(0) for ps in parent_states]
            c_list = [ps[2] if ps[2].dim() == 3 else ps[2].unsqueeze(0) for ps in parent_states]

            h_batch = torch.cat(h_list, dim=0)  # [B, L, P]
            c_batch = torch.cat(c_list, dim=0)  # [B, L, P]

            lstm_out_b, lstm_h_b, lstm_c_b = self._state_updater(tokens_tensor, h_batch, c_batch)

            for i, seq in enumerate(seqs_at_length):
                new_state = (
                    lstm_out_b[i : i + 1],  # [1, P]
                    lstm_h_b[i : i + 1],  # [1, L, P]
                    lstm_c_b[i : i + 1],  # [1, L, P]
                )
                self._state_cache[seq] = new_state

    def compute_scores_with_times(self, contexts):
        cache = self._score_cache
        enc_cache = self._enc_dev_cache
        inputs = self._inputs
        n_inputs = len(inputs)
        results = [None] * len(contexts)
        synaptogen_ml.set_fast_inference(True)

        if contexts:
            min_step = min(ctx[0] for ctx in contexts)
            if min_step > 0:
                stale_keys = [k for k in cache if k[0] < min_step]
                for k in stale_keys:
                    del cache[k]
                stale_enc = [k for k in enc_cache if k < min_step]
                for k in stale_enc:
                    del enc_cache[k]

        to_score = {}

        for idx, ctx in enumerate(contexts):
            step, label_seq = ctx

            if step >= n_inputs:
                continue

            cached = cache.get(ctx)
            if cached is not None:
                results[idx] = (cached, step)
                continue

            to_score.setdefault(step, []).append((idx, ctx, label_seq))

        if not to_score:
            return results

        device = self._device

        with torch.no_grad():
            max_bs = self._max_batch_size
            all_seqs_to_ensure = set()
            for step, items in to_score.items():
                for _, _, label_seq in items:
                    all_seqs_to_ensure.add(label_seq)

            self._ensure_states_cached(all_seqs_to_ensure)

            for step, items in to_score.items():
                enc_tensor = enc_cache.get(step)
                if enc_tensor is None:
                    enc_tensor = torch.from_numpy(inputs[step]).unsqueeze(0).float().to(device)
                    enc_cache[step] = enc_tensor

                unique_ctxs = {}
                for _, ctx, label_seq in items:
                    if ctx not in unique_ctxs:
                        unique_ctxs[ctx] = label_seq

                to_forward = list(unique_ctxs.items())

                for offset in range(0, len(to_forward), max_bs):
                    chunk = to_forward[offset : offset + max_bs]
                    B = len(chunk)

                    lstm_outs = []
                    for _, label_seq in chunk:
                        state = self._get_state(label_seq)
                        l_out = state[0]
                        if l_out.dim() == 1:
                            l_out = l_out.unsqueeze(0)
                        lstm_outs.append(l_out)

                    lstm_out_batch = torch.cat(lstm_outs, dim=0)  # [B, P]

                    scores_tensor = self._scorer(enc_tensor, lstm_out_batch)  # [B, V]
                    scores_np = scores_tensor.detach().cpu().numpy()

                    for b, (ctx, _) in enumerate(chunk):
                        cache[ctx] = scores_np[b]

                for idx, ctx, _ in items:
                    results[idx] = (cache[ctx], step)

        return results


def register_pyscorers():

    from librasr import LabelScorer, register_label_scorer_type

    class PyFixedContextTransducerPy(FixedContextTransducerPy, LabelScorer):
        def __init__(self, config):
            LabelScorer.__init__(self, config)
            FixedContextTransducerPy.__init__(self, config)

    register_label_scorer_type("fixed-context-py", PyFixedContextTransducerPy)
    register_label_scorer_type("fixed-context-py-optim", PyFixedContextTransducerPy)

    class PyStatefulTransducerPy(StatefulTransducerPy, LabelScorer):
        def __init__(self, config):
            LabelScorer.__init__(self, config)
            StatefulTransducerPy.__init__(self, config)

    register_label_scorer_type("stateful-py", PyStatefulTransducerPy)
    register_label_scorer_type("stateful-transducer-py", PyStatefulTransducerPy)
