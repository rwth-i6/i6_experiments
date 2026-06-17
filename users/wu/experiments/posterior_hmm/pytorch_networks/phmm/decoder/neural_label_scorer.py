"""
A pure-Python **neural Transformer** LM label scorer for LibRASR's
``lexiconfree-timesync-beam-search`` that runs the torch LM DIRECTLY on the GPU,
batched over the whole beam, instead of going through RASR's ``stateful-onnx``
scorer (which in the current image is pinned to the CPU execution provider --
``librasr.so`` is built without ``MODULE_CUDA`` and the bundled onnxruntime 1.15.1
has no CUDA provider). That CPU-ONNX path is the measured bottleneck of the
lexicon-free neural recog (a single variant did not finish in 26 min, ~0.8% CPU,
i.e. the search stalls on serialized per-frame ONNX round-trips).

Why a Python scorer can exploit the GPU here: RASR's beam search calls
``compute_scores_with_times(contexts)`` ONCE per step with the *list* of active
hypotheses' scoring contexts (Search/LexiconfreeTimesyncBeamSearch.cc:404), so we
can score the whole beam with a SINGLE batched ``_single_step`` forward on the GPU.

This mirrors ``ngram_label_scorer.register_ngram_label_scorer`` /
``_NGramLabelScorer`` and the semantics of ``StatefulOnnxLabelScorer``:

  * The scoring context is the emitted-label history. Blank/loop transitions do
    NOT advance it (matching ``blank_updates_history=False`` /
    ``loop_updates_history=False``): ``extended_scoring_context`` returns the
    SAME context object on a non-emitting transition.
  * Unlike the n-gram scorer (whose context collapses to a fixed (order-1)-token
    window), a neural LM has UNBOUNDED context, so we thread a real per-hypothesis
    KV cache through the context object -- exactly as the C++ stateful-ONNX scorer
    threads its CACHE_i / POS state. The KV state is computed lazily and BATCHED:
    ``extended_scoring_context`` only records (parent_state, token); the forward
    that actually advances the state is deferred to the next
    ``compute_scores_with_times`` so all beam entries advance in one GPU call.
  * The returned per-context vector is ``-log_softmax`` over all label indices,
    i.e. the natural-log cost RASR expects (``_single_step`` already returns
    ``neg_log_probs``). This is the IDENTICAL quantity the ONNX ``Scorer`` read out
    (an identity readout of ``LAST_LOGITS``), so scores match the ONNX path. RASR
    multiplies by ``scorer-2.scale`` (= lm_scale), so we return the unscaled cost.

The LM vocab is index-for-index identical to the lexicon-free label inventory
(both come from the same ``get_phmm_eow_lm_vocab_datastream`` job), so the LM
output dimension lines up with the RASR label indices, exactly as for the ONNX
scorer's ``LAST_LOGITS``.
"""

import os
import sys
from typing import List, Optional, Sequence

# Guard against double-registration within one process (init() runs once per forward
# job, but be defensive against re-entry / multiple segments).
_REGISTERED = set()


def register_neural_label_scorer(
    *,
    scorer_name: str,
    lm_checkpoint,
    lm_net_args: dict,
    bos_index: int,
    num_labels: int,
    device: Optional[str] = None,
    max_batch_size: Optional[int] = None,
):
    """
    Build and register a torch-Transformer-backed ``librasr.LabelScorer`` subclass.

    Must be called (in the forward process) BEFORE the ``SearchAlgorithm`` is
    constructed, so RASR can instantiate ``scorer_name`` from its label-scorer
    factory.

    :param scorer_name: type name referenced by ``label-scorer-2.type`` in the RASR config
    :param lm_checkpoint: tk.Path / PtCheckpoint to the trained LM checkpoint (``__fspath__``)
    :param lm_net_args: the LM ``net_args`` dict (``{"model_config_dict": {...}}``); may embed
        Sisyphus delayed values, resolved here via ``instanciate_delayed``
    :param bos_index: label index of ``<s>`` (the initial scoring context, never re-emitted)
    :param num_labels: size of the RASR label inventory; asserted == LM vocab dim
    :param device: torch device for the LM; defaults to cuda if available else cpu
    :param max_batch_size: optional cap on the per-step forward batch (contexts beyond it are
        scored in chunks); None = no cap (the whole beam in one forward)
    """
    import copy

    import torch

    import i6_core.util as util
    import librasr

    if scorer_name in _REGISTERED:
        return

    # Sisyphus runs the forward task with CWD inside the job's work dir while its
    # RecipeFinder resolves recipe modules CWD-relative; top-level recipe pkgs pulled in
    # transitively by the model (e.g. i6_models) can then be unresolvable. i6_experiments is
    # already imported here, so derive the absolute recipe root and put it on sys.path
    # (same guard as ExportStatefulOnnxLMJob.run).
    import i6_experiments

    recipe_root = os.path.dirname(os.path.dirname(os.path.abspath(i6_experiments.__file__)))
    if recipe_root not in sys.path:
        sys.path.insert(0, recipe_root)

    # Decoding variant: ``_single_step`` relies on the cache-aware positional encoding
    # (``positional_encoding(..., cache_length=pos)``) that only the *_decoding model exposes;
    # the training model adds pe[0] every step (correct only at pos==0). Same params/shapes,
    # so the trained checkpoint loads as-is (see ExportStatefulOnnxLMJob).
    from i6_experiments.users.wu.experiments.posterior_hmm.pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v1_decoding import (
        Model as DecodingModel,
    )
    from i6_experiments.users.wu.experiments.posterior_hmm.pytorch_networks.lm.trafo.stateful_onnx_v1 import (
        _single_step,
    )

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config_dict = util.instanciate_delayed(copy.deepcopy(lm_net_args["model_config_dict"]))
    base = DecodingModel(model_config_dict=model_config_dict)
    ckpt = torch.load(os.fspath(lm_checkpoint), map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    base.load_state_dict(state_dict)
    base.to(dev).eval()

    hidden_dim = base.cfg.hidden_dim
    vocab_dim = base.cfg.vocab_dim
    num_layers = base.cfg.num_layers
    assert vocab_dim == num_labels, (
        f"LM vocab dim {vocab_dim} != RASR label inventory size {num_labels}; the neural LM "
        f"distribution must be index-for-index aligned with the lexicon-free labels."
    )

    # Transitions on which the LM history does NOT advance (no new phoneme emitted).
    _NON_EMITTING = {
        librasr.TransitionType.LABEL_TO_BLANK,
        librasr.TransitionType.BLANK_LOOP,
        librasr.TransitionType.INITIAL_BLANK,
        librasr.TransitionType.LABEL_LOOP,
    }

    class _Ctx:
        """Per-hypothesis scoring context.

        Hashable/eq by the full emitted-token ``history`` (so RASR recombination
        merges identical-history hypotheses and threads ONE state). Carries the KV
        state AFTER consuming ``history``'s last token, computed lazily/batched:

          * computed:  ``caches`` (list of L tensors [1, len(history), hidden]),
            ``pos`` (= len(history)), ``scores`` (= -log p(. | history)).
          * deferred:  ``parent_caches`` / ``parent_pos`` / ``token`` -- the single
            step that produces this context's state from its parent's, run in the
            next ``compute_scores_with_times`` batch.
        """

        __slots__ = ("history", "caches", "pos", "scores", "parent_caches", "parent_pos", "token")

        def __init__(self, history):
            self.history = history
            self.caches: Optional[List["torch.Tensor"]] = None
            self.pos: Optional[int] = None
            self.scores = None
            self.parent_caches: Optional[List["torch.Tensor"]] = None
            self.parent_pos: Optional[int] = None
            self.token: Optional[int] = None

        def __hash__(self):
            return hash(self.history)

        def __eq__(self, other):
            return isinstance(other, _Ctx) and self.history == other.history

    class _NeuralLabelScorer(librasr.LabelScorer):
        """Torch Transformer LM label scorer; batches the beam into one GPU forward per step."""

        def __init__(self, config):
            super().__init__(config)
            self._initial: Optional[_Ctx] = None

        def reset(self):
            # Per-segment reset: drop the cached initial context and free its GPU state.
            self._initial = None
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        def signal_no_more_features(self):
            pass

        def add_inputs(self, inputs):
            # Pure LM: acoustic features are irrelevant.
            pass

        @torch.no_grad()
        def _make_initial(self) -> _Ctx:
            # State "after consuming <s>": consume bos at pos 0 into an empty (len-1) cache.
            token = torch.tensor([bos_index], dtype=torch.int64, device=dev)
            empty = [torch.zeros((1, 1, hidden_dim), dtype=torch.float32, device=dev) for _ in range(num_layers)]
            pos = torch.zeros((1,), dtype=torch.int32, device=dev)
            caches_out, neg_log_probs = _single_step(base, token, empty, pos, t_max=1)
            ctx = _Ctx((bos_index,))
            ctx.caches = [c.detach() for c in caches_out]  # each [1, 1, hidden]
            ctx.pos = 1
            ctx.scores = neg_log_probs[0].detach().to("cpu").tolist()
            return ctx

        def get_initial_scoring_context(self):
            if self._initial is None:
                self._initial = self._make_initial()
            return self._initial

        def extended_scoring_context(self, context, next_token, transition_type):
            if transition_type in _NON_EMITTING:
                return context  # blank/loop: LM history unchanged -> same state object
            child = _Ctx(context.history + (int(next_token),))
            # Defer the forward: just record the parent state + the token to consume.
            child.parent_caches = context.caches
            child.parent_pos = context.pos
            child.token = int(next_token)
            return child

        @torch.no_grad()
        def _advance_batch(self, pending: List[_Ctx]):
            """Run ONE batched ``_single_step`` for all deferred contexts in ``pending``."""
            # Each parent cache is [1, parent_pos, hidden]; pad to the batch's max parent length
            # so a single step can consume the new token at index parent_pos for every entry.
            parent_pos = [c.parent_pos for c in pending]
            max_par = max(parent_pos)
            t_step = max_par + 1  # room to write the new token at index parent_pos (<= max_par)

            b = len(pending)
            caches_in = []
            for layer in range(num_layers):
                buf = torch.zeros((b, t_step, hidden_dim), dtype=torch.float32, device=dev)
                for i, c in enumerate(pending):
                    pc = c.parent_caches[layer]  # [1, parent_pos_i, hidden]
                    buf[i, : parent_pos[i], :] = pc[0]
                caches_in.append(buf)
            tokens = torch.tensor([c.token for c in pending], dtype=torch.int64, device=dev)
            pos = torch.tensor(parent_pos, dtype=torch.int32, device=dev)

            caches_out, neg_log_probs = _single_step(base, tokens, caches_in, pos, t_max=t_step)

            scores_cpu = neg_log_probs.detach().to("cpu")
            for i, c in enumerate(pending):
                # Trim this context's cache to its own valid length (parent_pos_i + 1).
                keep = parent_pos[i] + 1
                c.caches = [caches_out[layer][i : i + 1, :keep, :].detach() for layer in range(num_layers)]
                c.pos = keep
                c.scores = scores_cpu[i].tolist()
                # Release the (now consumed) deferred refs so parents can be GC'd.
                c.parent_caches = None
                c.parent_pos = None
                c.token = None

        @torch.no_grad()
        def compute_scores_with_times(self, contexts):
            pending = [c for c in contexts if c.scores is None]
            if pending:
                if max_batch_size is None or len(pending) <= max_batch_size:
                    self._advance_batch(pending)
                else:
                    for start in range(0, len(pending), max_batch_size):
                        self._advance_batch(pending[start : start + max_batch_size])
            # timeframe = history length (monotonic; the AM frame index dominates the cascade).
            return [(c.scores, len(c.history)) for c in contexts]

    librasr.register_label_scorer_type(scorer_name, _NeuralLabelScorer)
    _REGISTERED.add(scorer_name)
