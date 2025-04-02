import sys
import copy
import warnings
import torch
import numpy as np

from typing import Optional, Tuple, List, Dict, Any

import returnn.frontend as rf
import returnn.torch.frontend as rtf
from returnn.tensor import Tensor, Dim, batch_dim
from returnn.frontend.tensor_array import TensorArray

from sisyphus import tk

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_flashlight_neural_lm import _format_align_label_seq
from i6_experiments.users.mueller.experiments.language_models.ffnn import FeedForwardLm
from i6_experiments.users.mueller.experiments.ctc_baseline.model import Model, OUT_BLANK_LABEL
from i6_experiments.users.mueller.experiments.ctc_baseline import recombination

CHECK_DECODER_CONSISTENCY = False

def recog_flashlight_ngram(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    arpa_4gram_lm: Optional[str],
    lexicon: str,
    hyperparameters: dict,
    prior_file: tk.Path = None
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.
    
    Uses a 4gram LM and beam search.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from torchaudio.models.decoder import ctc_decoder
    import torch
    import json
    from returnn.util.basic import cf
    from i6_experiments.users.mueller.experiments.language_models.ffnn import FFNN_LM_flashlight
    
    # Get the logits from the model
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    
    hyp_params = copy.copy(hyperparameters)
    lm_name = hyp_params.pop("lm_order", None)
    greedy = hyp_params.pop("greedy", False)
    prior_weight = hyp_params.pop("prior_weight", 0.0)
    prior_weight_tune = hyp_params.pop("prior_weight_tune", None)
    lm_weight_tune = hyp_params.pop("lm_weight_tune", None)
    use_logsoftmax = hyp_params.pop("use_logsoftmax", False)
    
    if prior_weight_tune:
        prior_weight_tune = json.load(open(prior_weight_tune))
        prior_weight_tune = prior_weight_tune["best_tune"]
        assert type(prior_weight_tune) == float, "Prior weight tune is not a float!"
        print(f"Prior weight with tune: {prior_weight} + {prior_weight_tune} = {prior_weight + prior_weight_tune}")
        prior_weight += prior_weight_tune
    if lm_weight_tune:
        lm_weight_tune = json.load(open(lm_weight_tune))
        lm_weight_tune = lm_weight_tune["best_tune"]
        assert type(lm_weight_tune) == float, "LM weight tune is not a float!"
        old_lm_weight = hyp_params.get("lm_weight", 0.0)
        print(f"LM weight with tune: {old_lm_weight} + {lm_weight_tune} = {old_lm_weight + lm_weight_tune}")
        hyp_params["lm_weight"] = old_lm_weight + lm_weight_tune
    
    if greedy:
        use_logsoftmax = True
    
    if use_logsoftmax:
        label_log_prob = model.log_probs_wb_from_logits(logits)
        label_log_prob = label_log_prob.raw_tensor.cpu()
    
        # Subtract prior of labels if available
        if prior_file and prior_weight > 0.0:
            prior = np.loadtxt(prior_file, dtype="float32")
            label_log_prob -= prior_weight * prior
            print("We subtracted the prior!")
    elif prior_file and prior_weight > 0.0:
        print("Cannot subtract prior without running log softmax")
        return None
    
    if greedy:
        probs, greedy_res = torch.max(label_log_prob, dim=-1)
        greedy_res = greedy_res.unsqueeze(1)
        
        scores = torch.sum(probs, dim=-1)
        scores = scores.unsqueeze(1)
        
        beam_dim = rtf.TorchBackend.get_new_dim_raw(greedy_res, 1, name="beam_dim")
        dims = [batch_dim, beam_dim, enc_spatial_dim]
        hyps = rtf.TorchBackend.convert_to_tensor(greedy_res, dims = dims, sparse_dim=model.wb_target_dim, dtype = "int64", name="hyps")
        
        dims = [batch_dim, beam_dim]
        scores = Tensor("scores", dims = dims, dtype = "float32", raw_tensor = scores)
        
        return hyps, scores, enc_spatial_dim, beam_dim
    
    if arpa_4gram_lm:
        arpa_4gram_lm = str(cf(arpa_4gram_lm))
    else:
        assert lm_name.startswith("ffnn")
        assert model.recog_language_model
        assert model.recog_language_model.vocab_dim == model.target_dim
        context_size = int(lm_name[len("ffnn"):])
        arpa_4gram_lm = FFNN_LM_flashlight(model.recog_language_model, model.recog_language_model.vocab_dim, context_size)
    
    use_lm = hyp_params.pop("use_lm", True)
    use_lexicon = hyp_params.pop("use_lexicon", True)
    
    configs = {
        "tokens": list(model.wb_target_dim.vocab.labels),
        "blank_token": OUT_BLANK_LABEL,
        "sil_token": OUT_BLANK_LABEL,
        "unk_word": "<unk>",
        "beam_size_token": None, # 16
        "beam_threshold": 1000000, # 14
    }
    configs["lexicon"] = lexicon if use_lexicon else None
    configs["lm"] = arpa_4gram_lm if use_lm else None
    
    configs.update(hyp_params)
    
    assert "ps_nbest" not in configs, "We only support nbest == 1"
    
    decoder = ctc_decoder(**configs)
    enc_spatial_dim_torch = enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()
    if use_logsoftmax:
        decoder_results = decoder(label_log_prob, enc_spatial_dim_torch)
    else:
        decoder_results = decoder(logits.raw_tensor.cpu(), enc_spatial_dim_torch)
    
    if use_lexicon:
        print("Use words directly!")
        if CHECK_DECODER_CONSISTENCY:
            for l1 in decoder_results:
                for l2 in l1:
                    lexicon_words = " ".join(l2.words)
                    token_words = " ".join([configs["tokens"][t] for t in l2.tokens])
                    assert not token_words.endswith("@@"), f"Token words ends with @@: {token_words}, Lexicon words: {lexicon_words}"
                    token_words = token_words.replace("@@ ", "")
                    assert lexicon_words == token_words, f"Words don't match: Lexicon words: {lexicon_words}, Token words: {token_words}"
        
        words = [[" ".join(l2.words) for l2 in l1] for l1 in decoder_results]
        words = np.array(words)
        words = np.expand_dims(words, axis=2)
        scores = [[l2.score for l2 in l1] for l1 in decoder_results]
        scores = torch.tensor(scores)
        
        beam_dim = Dim(words.shape[1], name="beam_dim")
        enc_spatial_dim = Dim(1, name="spatial_dim")
        words = rf._numpy_backend.NumpyBackend.convert_to_tensor(words, dims = [batch_dim, beam_dim, enc_spatial_dim], dtype = "string", name="hyps")
        scores = Tensor("scores", dims = [batch_dim, beam_dim], dtype = "float32", raw_tensor = scores)
        
        return words, scores, enc_spatial_dim, beam_dim
    else:
        def _pad_blanks(tokens, max_len):
            if len(tokens) < max_len:
                # print("We had to pad blanks")
                tokens = torch.cat([tokens, torch.tensor([model.blank_idx] * (max_len - len(tokens)))])
            return tokens
        
        def _pad_lists(t, max_len, max_len2):
            if t.shape[0] < max_len2:
                print("We had to pad the list")
                t = torch.cat([t, torch.tensor([[model.blank_idx] * max_len] * (max_len2 - t.shape[0]))])
            return t
        
        def _pad_scores(l, max_len):
            l = torch.tensor(l)
            if len(l) < max_len:
                print("We had to pad scores")
                l = torch.cat([l, torch.tensor([-1000000.0] * (max_len - len(l)))])
            return l
            
        max_length = int(enc_spatial_dim_torch.max())
        hyps = [torch.stack([_pad_blanks(l2.tokens, max_length) for l2 in l1]) for l1 in decoder_results]
        max_length_2 = max([l.shape[0] for l in hyps])
        hyps = [_pad_lists(t, max_length, max_length_2) for t in hyps]
        hyps = torch.stack(hyps)
        beam_dim = rtf.TorchBackend.get_new_dim_raw(hyps, 1, name="beam_dim")
        dims = [batch_dim, beam_dim, enc_spatial_dim]
        hyps = rtf.TorchBackend.convert_to_tensor(hyps, dims = dims, sparse_dim=model.wb_target_dim, dtype = "int64", name="hyps")
        
        scores = [[l2.score for l2 in l1] for l1 in decoder_results]
        max_length_3 = max([len(l) for l in scores])
        scores = torch.stack([_pad_scores(l, max_length_3) for l in scores])
        dims = [batch_dim, beam_dim]
        scores = Tensor("scores", dims = dims, dtype = "float32", raw_tensor = scores)
        
        # print(f"CUSTOM seq_targets: {hyps} \n{hyps.raw_tensor.cpu()},\nscores: {scores} \n{scores.raw_tensor.cpu()}n {scores.raw_tensor.cpu()[0][0]},\nspatial_dim: {enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()},\n beam_size: {beam_dim}")
        
        return hyps, scores, enc_spatial_dim, beam_dim
    
def recog_no_lm(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, Vocab
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
        label_log_prob, k_dim=Dim(beam_size, name=f"pre-filter-beam"), axis=[model.wb_target_dim]
    )  # seq_log_prob, backrefs_global: Batch, Spatial, PreFilterBeam. backrefs_pre_filter -> Vocab
    label_log_prob_pre_filter_ta = TensorArray.unstack(
        label_log_prob_pre_filter, axis=enc_spatial_dim
    )  # t -> Batch, PreFilterBeam
    backrefs_pre_filter_ta = TensorArray.unstack(backrefs_pre_filter, axis=enc_spatial_dim)  # t -> Batch, PreFilterBeam

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets = []
    seq_backrefs = []
    for t in range(max_seq_len):
        # Filter out finished beams
        seq_log_prob = seq_log_prob + label_log_prob_pre_filter_ta[t]  # Batch, InBeam, PreFilterBeam
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, pre_filter_beam_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> PreFilterBeam.
        target = rf.gather(backrefs_pre_filter_ta[t], indices=target)  # Batch, Beam -> Vocab
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = enc_spatial_dim
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim

def recog_flashlight_ffnn(
    *,
    model: Model,
    label_log_prob: Tensor,
    enc_spatial_dim: Dim,
    hyperparameters: dict,
    prior_file: tk.Path = None,
    train_lm = False,
    print_idx = []
) -> Tuple[Tensor, Tensor, Dim, Dim] | list:
    from dataclasses import dataclass
    import torch
    import json
    from flashlight.lib.text.decoder import LM, LMState
    from i6_experiments.users.zeyer.utils.lru_cache import lru_cache
    from returnn.util import basic as util
    
    def _output_hyps(hyp: list, model: Model) -> str:
        prev = None
        ls = []
        for h in hyp:
            if h != prev:
                ls.append(h)
                prev = h
        ls = [model.target_dim.vocab.id_to_label(h) for h in ls if h != model.blank_idx]
        s = " ".join(ls).replace("@@ ", "")
        if s.endswith("@@"):
            s = s[:-2]
        return s

    hyp_params = copy.copy(hyperparameters)
    lm_name = hyp_params.pop("lm_order", None)
    prior_weight = hyp_params.pop("prior_weight", 0.0)
    prior_weight_tune = hyp_params.pop("prior_weight_tune", None)
    lm_weight_tune = hyp_params.pop("lm_weight_tune", None)
    
    if prior_weight_tune:
        prior_weight_tune = json.load(open(prior_weight_tune))
        prior_weight_tune = prior_weight_tune["best_tune"]
        assert type(prior_weight_tune) == float, "Prior weight tune is not a float!"
        print(f"Prior weight with tune: {prior_weight} + {prior_weight_tune} = {prior_weight + prior_weight_tune}")
        prior_weight += prior_weight_tune
    if lm_weight_tune:
        lm_weight_tune = json.load(open(lm_weight_tune))
        lm_weight_tune = lm_weight_tune["best_tune"]
        assert type(lm_weight_tune) == float, "LM weight tune is not a float!"
        old_lm_weight = hyp_params.get("lm_weight", 0.0)
        print(f"LM weight with tune: {old_lm_weight} + {lm_weight_tune} = {old_lm_weight + lm_weight_tune}")
        hyp_params["lm_weight"] = old_lm_weight + lm_weight_tune

    n_best = hyp_params.pop("ps_nbest", 1)
    beam_size = hyp_params.pop("beam_size", 1)
    beam_size_token = hyp_params.pop("beam_size_token", model.wb_target_dim.vocab.num_labels)
    beam_threshold = hyp_params.pop("beam_threshold", 1000000)
    log_add = hyp_params.pop("log_add", False)

    # Eager-mode implementation of beam search using Flashlight.

    # noinspection PyUnresolvedReferences
    assert lm_name.startswith("ffnn")
    context_size = int(lm_name[len("ffnn"):])
    if train_lm:
        assert model.train_language_model
        assert model.train_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.train_language_model
    else:
        assert model.recog_language_model
        assert model.recog_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.recog_language_model
    # noinspection PyUnresolvedReferences
    lm_scale: float = hyp_params["lm_weight"]

    dev_s = rf.get_default_device()
    dev = torch.device(dev_s)

    total_mem = None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        _, total_mem = torch.cuda.mem_get_info(dev if dev.index is not None else None)

    def _collect_mem_stats():
        if dev.type == "cuda":
            return [
                f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
            ]
        return ["(unknown)"]

    # print(
    #     f"Memory usage {dev_s} before encoder forward:",
    #     " ".join(_collect_mem_stats()),
    #     "total:",
    #     util.human_bytes_size(total_mem) if total_mem else "(unknown)",
    # )

    lm_initial_state = lm.default_initial_state(batch_dims=[])

    # https://github.com/flashlight/text/tree/main/bindings/python#decoding-with-your-own-language-model
    # https://github.com/facebookresearch/fairseq/blob/main/examples/speech_recognition/new/decoders/flashlight_decoder.py
    # https://github.com/pytorch/audio/blob/main/src/torchaudio/models/decoder/_ctc_decoder.py

    # The current implementation of FlashlightLM below assumes we can just use the token_idx as-is for the LM.
    assert model.blank_idx == model.target_dim.dimension

    @dataclass
    class FlashlightLMState:
        def __init__(self, label_seq: List[int], prev_state: LMState):
            if len(label_seq) > context_size:
                self.label_seq = label_seq[-context_size:]
            else:
                self.label_seq = label_seq
            assert len(self.label_seq) == context_size
            self.prev_state = prev_state

    # Use LRU cache for the LM states (on GPU) and log probs.
    # Note that additionally to the cache size limit here,
    # we free more when we run out of CUDA memory.
    start_lru_cache_size = 1024
    max_used_mem_fraction = 0.9

    class FlashlightLM(LM):
        def __init__(self):
            super().__init__()
            # Cannot use weakrefs because the LMState object will always be recreated on-the-fly,
            # i.e. the Python object does not persist.
            self.mapping_states: Dict[LMState, FlashlightLMState] = {}
            self._count_recalc_whole_seq = 0
            self._recent_debug_log_time = -sys.maxsize
            self._max_used_mem_fraction = max_used_mem_fraction

        def reset(self):
            self.mapping_states.clear()
            self._count_recalc_whole_seq = 0
            self._recent_debug_log_time = -sys.maxsize
            self._max_used_mem_fraction = max_used_mem_fraction
            self._calc_next_lm_state.cache_clear()
            self._calc_next_lm_state.cache_set_maxsize(start_lru_cache_size)

        @lru_cache(maxsize=start_lru_cache_size)
        def _calc_next_lm_state(self, state: LMState) -> Tuple[Any, torch.Tensor]:
            """
            :return: LM state, log probs [Vocab]
            """
            state_ = self.mapping_states[state]
            
            lm_logits, lm_state = None, None
            while True:
                self._cache_maybe_free_memory()
                try:
                    self._count_recalc_whole_seq += 1
                    spatial_dim = Dim(len(state_.label_seq), name="seq")
                    out_spatial_dim = Dim(context_size + 1, name="seq_out")
                    lm_logits, lm_state = lm(
                        rf.convert_to_tensor(state_.label_seq, dims=[spatial_dim], sparse_dim=model.target_dim),
                        spatial_dim=spatial_dim,
                        out_spatial_dim=out_spatial_dim,
                        state=lm_initial_state,
                    )  # Vocab / ...
                    lm_logits = rf.gather(lm_logits, axis=out_spatial_dim, indices=rf.last_frame_position_of_dim(out_spatial_dim))
                except torch.cuda.OutOfMemoryError as exc:
                    if self._calc_next_lm_state.cache_len() == 0:
                        raise  # cannot free more
                    print(f"{type(exc).__name__}: {exc}")
                    new_max_used_mem_fraction = max(0.2, self._max_used_mem_fraction - 0.1)
                    if new_max_used_mem_fraction != self._max_used_mem_fraction:
                        print(f"Reduce max used mem fraction to {new_max_used_mem_fraction:.0%}")
                    continue  # try again
                break
            assert lm_logits.dims == (model.target_dim,)
            lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Vocab
            log_probs_raw = lm_log_probs.raw_tensor.cpu()
            return lm_state, log_probs_raw

        def _cache_maybe_free_memory(self):
            if dev.type == "cuda":
                # Maybe check if we should free some more memory.
                count_pop = 0
                used_mem = 0
                while self._calc_next_lm_state.cache_len() > 0:
                    used_mem = torch.cuda.memory_reserved(dev)
                    if used_mem / total_mem < self._max_used_mem_fraction:
                        break
                    # Check again after trying to empty the cache.
                    # Note: gc.collect() is problematic here because of how Flashlight handles the states:
                    # We have millions of Python objects in the mapping_states dict,
                    # which takes a very long time to go through.
                    torch.cuda.empty_cache()
                    used_mem = torch.cuda.memory_reserved(dev)
                    if used_mem / total_mem < self._max_used_mem_fraction:
                        break
                    self._calc_next_lm_state.cache_pop_oldest()
                    count_pop += 1
                if count_pop > 0:
                    print(
                        f"Pop {count_pop} states from cache,"
                        f" cache size {self._calc_next_lm_state.cache_len()},"
                        f" reached {used_mem / total_mem:.1%} of total mem,"
                        f" mem usage {dev_s}: {' '.join(_collect_mem_stats())}"
                    )
                    self._calc_next_lm_state.cache_set_maxsize(self._calc_next_lm_state.cache_len())

        def start(self, start_with_nothing: bool):
            """
            Parameters:
                start_with_nothing (bool): whether or not to start sentence with sil token.
            """
            start_with_nothing  # noqa  # not sure how to handle this?
            self.reset()
            state = LMState()
            self.mapping_states[state] = FlashlightLMState(label_seq=[model.bos_idx]*context_size, prev_state=state)
            return state

        def score(self, state: LMState, token_index: int):
            """
            Evaluate language model based on the current lm state and new word

            Parameters:
                state: current lm state
                token_index: index of the word
                            (can be lexicon index then you should store inside LM the
                            mapping between indices of lexicon and lm, or lm index of a word)

            Returns:
                (LMState, float): pair of (new state, score for the current word)
            """
            state_ = self.mapping_states[state]
            # if time.monotonic() - self._recent_debug_log_time > 1:
            #     print(
            #         "LM prefix",
            #         [model.target_dim.vocab.id_to_label(label_idx) for label_idx in state_.label_seq],
            #         f"score {model.target_dim.vocab.id_to_label(token_index)!r}",
            #         f"({len(self.mapping_states)} states seen)",
            #         f"(cache info {self._calc_next_lm_state.cache_info()})",
            #         f"(mem usage {dev_s}: {' '.join(_collect_mem_stats())})",
            #     )
            #     self._recent_debug_log_time = time.monotonic()
            outstate = state.child(token_index)
            if outstate not in self.mapping_states:
                self.mapping_states[outstate] = FlashlightLMState(
                    label_seq=state_.label_seq + [token_index], prev_state=state
                )

            _, log_probs_raw = self._calc_next_lm_state(state)
            return outstate, log_probs_raw[token_index]

        def finish(self, state: LMState):
            """
            Evaluate eos for language model based on the current lm state

            Returns:
                (LMState, float): pair of (new state, score for the current word)
            """
            return self.score(state, model.eos_idx)

    fl_lm = FlashlightLM()

    from flashlight.lib.text.decoder import LexiconFreeDecoderOptions, LexiconFreeDecoder, CriterionType

    fl_decoder_opts = LexiconFreeDecoderOptions(
        beam_size=beam_size,
        beam_size_token=beam_size_token,
        beam_threshold=beam_threshold,
        lm_weight=lm_scale,
        sil_score=0.0,
        log_add=log_add,
        criterion_type=CriterionType.CTC,
    )
    fl_decoder = LexiconFreeDecoder(fl_decoder_opts, fl_lm, -1, model.blank_idx, [])

    # Subtract prior of labels if available
    if prior_file and prior_weight > 0.0:
        prior = np.loadtxt(prior_file, dtype="float32")
        prior *= prior_weight
        prior = torch.tensor(prior, dtype=torch.float32, device=dev)
        prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.wb_target_dim], dtype="float32")
        label_log_prob = label_log_prob - prior
        # print("We subtracted the prior!")
    
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob = label_log_prob.copy_transpose((batch_dim, enc_spatial_dim, model.wb_target_dim))
    batch_size, max_seq_len = label_log_prob.raw_tensor.shape[:2]
    assert enc_spatial_dim.dyn_size_ext.dims == (batch_dim,)

    label_log_prob = rf.cast(label_log_prob, "float32")
    label_log_prob = rf.copy_to_device(label_log_prob, "cpu")
    label_log_prob_raw = label_log_prob.raw_tensor.contiguous()
    float_bytes = 4

    # print(f"Memory usage {dev_s} after encoder forward:", " ".join(_collect_mem_stats()))

    hyps = []
    scores = []
    for batch_idx in range(batch_size):
        emissions_ptr = label_log_prob_raw.data_ptr() + float_bytes * batch_idx * label_log_prob_raw.stride(0)
        seq_len = enc_spatial_dim.dyn_size[batch_idx]
        assert seq_len <= max_seq_len
        results = fl_decoder.decode(emissions_ptr, seq_len, model.wb_target_dim.dimension)
        # I get -1 (silence label?) at the beginning and end in the tokens? Filter those away.
        # These are also additional frames which don't correspond to the input frames?
        # When removing those two frames, the len of tokens (align labels) matches the emission frames
        # (as it should be).
        hyps_per_batch = [[label for label in result.tokens if label >= 0] for result in results]
        scores_per_batch = [result.score for result in results]
        # print(
        #     f"batch {batch_idx + 1}/{batch_size}: {len(results)} hyps,"
        #     f" best score: {scores_per_batch[0]},"
        #     f" best seq {_format_align_label_seq(results[0].tokens, model.wb_target_dim)},"
        #     f" worst score: {scores_per_batch[-1]},"
        #     f" LM cache info {fl_lm._calc_next_lm_state.cache_info()},"
        #     f" LM recalc whole seq count {fl_lm._count_recalc_whole_seq},"
        #     f" mem usage {dev_s}: {' '.join(_collect_mem_stats())}"
        # )
        assert all(
            len(hyp) == seq_len for hyp in hyps_per_batch
        ), f"seq_len {seq_len}, hyps lens {[len(hyp) for hyp in hyps_per_batch]}"
        if print_idx:
            for idx in print_idx:
                if idx == batch_idx:
                    print(f"RES: {hyps_per_batch}, {scores_per_batch}")
        
        if len(results) >= n_best:
            if n_best > 1:
                # We have to select the n_best on output level
                hyps_shortened = [_output_hyps(hyp, model) for hyp in hyps_per_batch]
                nbest_hyps = []
                nbest_hyps_ids = []
                k = 0
                i = 0
                while k < n_best:
                    if i >= len(hyps_shortened):
                        break
                    if hyps_shortened[i] not in nbest_hyps:
                        nbest_hyps.append(hyps_shortened[i])
                        nbest_hyps_ids.append(i)
                        k += 1
                    i += 1
                hyps_per_batch = [hyps_per_batch[id] for id in nbest_hyps_ids]
                scores_per_batch = [scores_per_batch[id] for id in nbest_hyps_ids]
                
                if len(hyps_per_batch) < n_best:
                    print("Not enough n-best")
                    hyps_per_batch += [[]] * (n_best - len(hyps_per_batch))
                    scores_per_batch += [-1e30] * (n_best - len(scores_per_batch))
            else:
                hyps_per_batch = hyps_per_batch[:n_best]
                scores_per_batch = scores_per_batch[:n_best]
        else:
            hyps_per_batch += [[]] * (n_best - len(results))
            scores_per_batch += [-1e30] * (n_best - len(results))
        assert len(hyps_per_batch) == len(scores_per_batch) == n_best
        hyps_per_batch = [hyp + [model.blank_idx] * (max_seq_len - len(hyp)) for hyp in hyps_per_batch]
        assert all(len(hyp) == max_seq_len for hyp in hyps_per_batch)
        hyps.append(hyps_per_batch)
        scores.append(scores_per_batch)
    fl_lm.reset()
    hyps_pt = torch.tensor(hyps, dtype=torch.int32)
    assert hyps_pt.shape == (batch_size, n_best, max_seq_len)
    scores_pt = torch.tensor(scores, dtype=torch.float32)
    assert scores_pt.shape == (batch_size, n_best)

    beam_dim = Dim(n_best, name="beam")
    out_spatial_dim = enc_spatial_dim
    hyps_r = rf.convert_to_tensor(hyps_pt, dims=(batch_dim, beam_dim, out_spatial_dim), sparse_dim=model.wb_target_dim)
    scores_r = rf.convert_to_tensor(scores_pt, dims=(batch_dim, beam_dim))
    print(f"Memory usage ({dev_s}) after batch:", " ".join(_collect_mem_stats()))
    if train_lm:
        return hyps
    else:
        return hyps_r, scores_r, out_spatial_dim, beam_dim
    
def recog_ffnn(
    *,
    model: Model,
    label_log_prob: Tensor,
    enc_spatial_dim: Dim,
    hyperparameters: dict,
    batch_dims: List[Dim],
    prior_file: tk.Path = None,
    train_lm: bool = False,
    version: int = 1,
    print_idx: list = []
):
    import json
    
    def _update_context(context: Tensor, new_label: Tensor, context_dim: Dim) -> Tensor:
        new_dim = Dim(1, name="new_label")
        new_label = rf.expand_dim(new_label, dim=new_dim)
        old_context, old_context_dim = rf.slice(context, axis=context_dim, start=1)
        new_context, new_context_dim = rf.concat((old_context, old_context_dim), (new_label, new_dim), out_dim=context_dim)
        assert new_context_dim == context_dim
        return new_context
    
    def _target_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
        assert target.sparse_dim == wb_target_dim
        assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
        return rf.set_sparse_dim(target, target_dim)


    def _target_dense_extend_blank(
        target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int, value: float
    ) -> Tensor:
        assert target_dim in target.dims
        assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
        res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
        return res

    hyp_params = copy.copy(hyperparameters)
    lm_name = hyp_params.pop("lm_order", None)
    prior_weight = hyp_params.pop("prior_weight", 0.0)
    prior_weight_tune = hyp_params.pop("prior_weight_tune", None)
    lm_weight_tune = hyp_params.pop("lm_weight_tune", None)
    
    if prior_weight_tune:
        prior_weight_tune = json.load(open(prior_weight_tune))
        prior_weight_tune = prior_weight_tune["best_tune"]
        assert type(prior_weight_tune) == float, "Prior weight tune is not a float!"
        print(f"Prior weight with tune: {prior_weight} + {prior_weight_tune} = {prior_weight + prior_weight_tune}")
        prior_weight += prior_weight_tune
    if lm_weight_tune:
        lm_weight_tune = json.load(open(lm_weight_tune))
        lm_weight_tune = lm_weight_tune["best_tune"]
        assert type(lm_weight_tune) == float, "LM weight tune is not a float!"
        old_lm_weight = hyp_params.get("lm_weight", 0.0)
        print(f"LM weight with tune: {old_lm_weight} + {lm_weight_tune} = {old_lm_weight + lm_weight_tune}")
        hyp_params["lm_weight"] = old_lm_weight + lm_weight_tune

    n_best = hyp_params.pop("ps_nbest", 1)
    beam_size = hyp_params.pop("beam_size", 1)
    use_recombination = hyp_params.pop("use_recombination", False)
    assert n_best == 1 or use_recombination, "n-best only implemented with recombination"
    recomb_blank = hyp_params.pop("recomb_blank", False)
    recomb_after_topk = hyp_params.pop("recomb_after_topk", False)
    
    dev_s = rf.get_default_device()
    dev = torch.device(dev_s)

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    import returnn
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__
    
    # Subtract prior of labels if available
    if prior_file and prior_weight > 0.0:
        prior = np.loadtxt(prior_file, dtype="float32")
        prior *= prior_weight
        prior = torch.tensor(prior, dtype=torch.float32, device=dev)
        prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.wb_target_dim], dtype="float32")
        label_log_prob = label_log_prob - prior
        # print("We subtracted the prior!")
        
    assert lm_name.startswith("ffnn")
    context_size = int(lm_name[len("ffnn"):])
    if train_lm:
        assert model.train_language_model
        assert model.train_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.train_language_model
    else:
        assert model.recog_language_model
        assert model.recog_language_model.vocab_dim == model.target_dim
        lm: FeedForwardLm = model.recog_language_model
    # noinspection PyUnresolvedReferences
    lm_scale: float = hyp_params["lm_weight"]

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    context_dim = Dim(context_size, name="context")
    lm_out_dim = Dim(context_size + 1, name="context+1")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_ta = TensorArray.unstack(label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    target = rf.constant(model.bos_idx, dims=batch_dims_ + [context_dim], sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB

    lm_state = lm.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
    lm_logits, lm_state = lm(
        target,
        spatial_dim=context_dim,
        out_spatial_dim=lm_out_dim,
        state=lm_state,
    )  # Batch, InBeam, Vocab / ...
    lm_logits = rf.gather(lm_logits, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
    assert lm_logits.dims == (*batch_dims_, model.target_dim)
    lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
    lm_log_probs *= lm_scale

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    backrefs = None
    if use_recombination:
        assert len(batch_dims) == 1
        if recomb_after_topk:
            seq_hash = rf.constant(0, dims=batch_dims_, dtype="int64")
        else:
            seq_hash = rf.constant(0, dims=batch_dims_ + [model.wb_target_dim], dtype="int64")
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        seq_log_prob = seq_log_prob + label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seq_log_prob += rf.where(
                (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
                _target_dense_extend_blank(
                    lm_log_probs,
                    target_dim=model.target_dim,
                    wb_target_dim=model.wb_target_dim,
                    blank_idx=model.blank_idx,
                    value=0.0,
                ),
                0.0,
            )  # Batch, InBeam, VocabWB
            
        if use_recombination and not recomb_after_topk:
            seq_hash = recombination.update_seq_hash(seq_hash, rf.range_over_dim(model.wb_target_dim), backrefs, target_wb, model.blank_idx)
            if t > 0:
                seq_log_prob = recombination.recombine_seqs(
                    seq_log_prob,
                    seq_hash,
                    beam_dim,
                    batch_dims[0],
                    model.wb_target_dim,
                    model.blank_idx,
                    recomb_blank=recomb_blank,
                    use_sum=False,
                )
            
        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        target_wb = rf.cast(target_wb, "int32")
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, Vocab
        lm_state = rf.nested.gather_nested(lm_state, indices=backrefs)
        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB
        got_new_label = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _update_context(
                prev_target,
                _target_remove_blank(
                    target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
                ),
                context_dim
            ),
            prev_target,
        )  # Batch, Beam -> Vocab
        
        if use_recombination and recomb_after_topk:
            seq_hash = recombination.update_seq_hash(seq_hash, target_wb, backrefs, prev_target_wb, model.blank_idx, gather_old_target=False)
            if t > 0:
                seq_log_prob = recombination.recombine_seqs(
                    seq_log_prob,
                    seq_hash,
                    beam_dim,
                    batch_dims[0],
                    None,
                    model.blank_idx,
                    recomb_blank=recomb_blank,
                    use_sum=False,
                    is_blank=(target_wb == model.blank_idx),
                )

        got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
        if got_new_label_cpu.raw_tensor.sum().item() > 0:
            (target_, lm_state_), packed_new_label_dim, packed_new_label_dim_map = rf.nested.masked_select_nested(
                (target, lm_state),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                dims=batch_dims + [beam_dim],
            )
            # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
            assert packed_new_label_dim.get_dim_value() > 0
            
            lm_logits_, lm_state_ = lm(
                target_,
                spatial_dim=context_dim,
                out_spatial_dim=lm_out_dim,
                state=lm_state_,
            )  # Flat_Batch_Beam, Vocab / ...
            lm_logits_ = rf.gather(lm_logits_, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
            assert lm_logits_.dims == (packed_new_label_dim, model.target_dim)
            lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
            lm_log_probs_ *= lm_scale

            lm_log_probs, lm_state = rf.nested.masked_scatter_nested(
                (lm_log_probs_, lm_state_),
                (lm_log_probs, lm_state),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                dims=batch_dims + [beam_dim],
                in_dim=packed_new_label_dim,
                masked_select_dim_map=packed_new_label_dim_map,
            )  # Batch, Beam, Vocab / ...
                
    # seq_log_prob, lm_log_probs: Batch, Beam
    # Add LM EOS score at the end.
    lm_eos_score = rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)
    seq_log_prob += lm_eos_score  # Batch, Beam -> VocabWB
        

    # Backtrack via backrefs, resolve beams.
    seq_targets_wb_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target_wb in zip(seq_backrefs[::-1], seq_targets_wb[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_wb_.insert(0, rf.gather(target_wb, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets_wb__ = TensorArray(seq_targets_wb_[0])
    for target_wb in seq_targets_wb_:
        seq_targets_wb__ = seq_targets_wb__.push_back(target_wb)
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_wb__.stack(axis=out_spatial_dim)
    
    if int(beam_dim.get_dim_value()) >= n_best:
        if n_best > 1:
            assert recomb_after_topk and recomb_blank # TODO update hash also in the other cases
            
            seq_log_prob, indices, beam_dim_new = rf.top_k(
                seq_log_prob, k_dim=Dim(n_best, name=f"nbest-beam"), axis=beam_dim
            )
            seq_targets_wb = rf.gather(seq_targets_wb, axis=beam_dim, indices=indices)
            # Filter out duplicated seqs which are still appearing even though we recombine
            seq_targets_wb = rf.where(
                (seq_log_prob <= -1.0e30),
                rf.constant(model.blank_idx, dims=batch_dims + [beam_dim_new], sparse_dim=model.wb_target_dim),
                seq_targets_wb
            )
            beam_dim = beam_dim_new
        else:
            seq_log_prob, indices, beam_dim_new = rf.top_k(
                seq_log_prob, k_dim=Dim(n_best, name=f"nbest-beam"), axis=beam_dim
            )
            seq_targets_wb = rf.gather(seq_targets_wb, axis=beam_dim, indices=indices)
            beam_dim = beam_dim_new

    if train_lm:
        return seq_targets_wb.raw_tensor.transpose(0,1).transpose(1,2).tolist()
    else:
        return seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim