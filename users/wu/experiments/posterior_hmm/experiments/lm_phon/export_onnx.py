"""
Sisyphus job that exports a trained kazuki_trafo_zijian_variant_v1 Transformer
LM checkpoint to the (state-initializer, state-updater, scorer) ONNX triplet
expected by RASR's `StatefulOnnxLabelScorer`.
"""

import os
import pickle
from typing import Any, Dict

from sisyphus import Job, Task, tk


class ExportStatefulOnnxLMJob(Job):
    """
    Build the three ONNX files needed by `StatefulOnnxLabelScorer`:

      * state-initializer.onnx
      * state-updater.onnx
      * scorer.onnx

    Custom ONNX metadata is added to each model to map its input/output tensor
    names to the shared state names {CACHE_i, POS, LAST_LOGITS} that RASR uses
    to thread state between calls.
    """

    __sis_hash_exclude__ = {"opset": 17}

    def __init__(
        self,
        checkpoint: tk.Path,
        net_args: Dict[str, Any],
        vocab: tk.Path,
        bos_token: str = "<s>",
        t_max: int = 1024,
        opset: int = 17,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.net_args = net_args
        self.vocab = vocab
        self.bos_token = bos_token
        self.t_max = t_max
        self.opset = opset

        self.out_initializer = self.output_path("state_initializer.onnx")
        self.out_updater = self.output_path("state_updater.onnx")
        self.out_scorer = self.output_path("scorer.onnx")
        self.out_bos_index = self.output_var("bos_index")
        self.out_num_layers = self.output_var("num_layers")
        self.out_t_max = self.output_var("t_max")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _set_metadata(self, onnx_path: str, mapping: Dict[str, str]):
        import onnx

        model = onnx.load(onnx_path)
        # remove pre-existing entries that we are about to overwrite
        keep = [p for p in model.metadata_props if p.key not in mapping]
        del model.metadata_props[:]
        model.metadata_props.extend(keep)
        for k, v in mapping.items():
            entry = model.metadata_props.add()
            entry.key = k
            entry.value = v
        onnx.save(model, onnx_path)

    def run(self):
        import sys

        import torch

        # Sisyphus runs the task with the CWD changed into the job's ``work`` dir, but its
        # ``RecipeFinder`` resolves recipe modules via a CWD-relative ``recipe`` path. Top-level
        # recipe packages imported lazily here (e.g. ``i6_models``, pulled in transitively by the
        # model below) are therefore not found. ``i6_experiments`` is already imported (and cached)
        # at this point, so derive the absolute recipe root from it and put it on ``sys.path``.
        import i6_experiments

        recipe_root = os.path.dirname(os.path.dirname(os.path.abspath(i6_experiments.__file__)))
        if recipe_root not in sys.path:
            sys.path.insert(0, recipe_root)

        # Use the decoding variant of the model: ``stateful_onnx_v1._single_step`` reimplements the
        # incremental single-step KV-cache decode and relies on the cache-aware positional encoding
        # (``positional_encoding(..., cache_length=pos)``), which only the *_decoding model exposes.
        # The base training model adds ``pe[0]`` at every step (correct only for pos==0). The two
        # variants share identical parameter names/shapes, so the trained checkpoint loads as-is.
        from i6_experiments.users.wu.experiments.posterior_hmm.pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v1_decoding import (
            Model as BaseModel,
        )
        from i6_experiments.users.wu.experiments.posterior_hmm.pytorch_networks.lm.trafo.stateful_onnx_v1 import (
            Scorer,
            StateInitializer,
            StateUpdater,
            state_name_map_initializer,
            state_name_map_scorer,
            state_name_map_updater,
        )

        with open(self.vocab.get_path(), "rb") as f:
            vocab: Dict[str, int] = pickle.load(f)
        assert self.bos_token in vocab, f"BOS token {self.bos_token!r} missing from vocab {list(vocab)[:5]}..."
        bos_index = int(vocab[self.bos_token])

        # ``checkpoint`` may be a ``tk.Path`` or an i6_core ``PtCheckpoint``; both implement
        # ``__fspath__``, so normalize via ``os.fspath`` instead of assuming ``get_path``.
        ckpt = torch.load(os.fspath(self.checkpoint), map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # ``net_args`` may embed unresolved Sisyphus delayed values (e.g. ``vocab_dim`` is a
        # ``Variable``). Sisyphus tracks these as dependencies but does not substitute their
        # concrete values into the stored dict, so resolve them here. Work on a deep copy to
        # leave ``self.net_args`` untouched and to avoid ``instanciate_delayed_copy``'s ``tree`` dep.
        import copy

        import i6_core.util as util

        model_config_dict = util.instanciate_delayed(copy.deepcopy(self.net_args["model_config_dict"]))
        base = BaseModel(model_config_dict=model_config_dict)
        base.load_state_dict(state_dict)
        base.eval()
        num_layers = base.cfg.num_layers
        hidden_dim = base.cfg.hidden_dim
        vocab_dim = base.cfg.vocab_dim

        initializer = StateInitializer(base, t_max=self.t_max, bos_index=bos_index).eval()
        updater = StateUpdater(base, t_max=self.t_max).eval()
        scorer = Scorer().eval()

        # --- state-initializer ---
        init_dummy = torch.zeros((1,), dtype=torch.int32)
        init_output_names = [f"cache_{i}_out" for i in range(num_layers)] + ["pos_out", "last_logits_out"]
        init_dynamic_axes = {name: {0: "B"} for name in init_output_names}
        torch.onnx.export(
            initializer,
            (init_dummy,),
            self.out_initializer.get_path(),
            input_names=["dummy_es_size"],
            output_names=init_output_names,
            dynamic_axes={"dummy_es_size": {0: "B"}, **init_dynamic_axes},
            opset_version=self.opset,
            do_constant_folding=True,
        )
        self._set_metadata(self.out_initializer.get_path(), state_name_map_initializer(num_layers))

        # --- state-updater ---
        upd_token = torch.zeros((1,), dtype=torch.int32)
        upd_pos = torch.zeros((1,), dtype=torch.int32)
        upd_caches = tuple(
            torch.zeros((1, self.t_max, hidden_dim), dtype=torch.float32) for _ in range(num_layers)
        )
        upd_input_names = ["token", "pos_in"] + [f"cache_{i}_in" for i in range(num_layers)]
        upd_output_names = [f"cache_{i}_out" for i in range(num_layers)] + ["pos_out", "last_logits_out"]
        upd_dynamic_axes = {name: {0: "B"} for name in upd_input_names + upd_output_names}
        torch.onnx.export(
            updater,
            (upd_token, upd_pos, *upd_caches),
            self.out_updater.get_path(),
            input_names=upd_input_names,
            output_names=upd_output_names,
            dynamic_axes=upd_dynamic_axes,
            opset_version=self.opset,
            do_constant_folding=True,
        )
        self._set_metadata(self.out_updater.get_path(), state_name_map_updater(num_layers))

        # --- scorer ---
        scr_last_logits = torch.zeros((1, vocab_dim), dtype=torch.float32)
        torch.onnx.export(
            scorer,
            (scr_last_logits,),
            self.out_scorer.get_path(),
            input_names=["last_logits_in"],
            output_names=["scores"],
            dynamic_axes={"last_logits_in": {0: "B"}, "scores": {0: "B"}},
            opset_version=self.opset,
            do_constant_folding=True,
        )
        self._set_metadata(self.out_scorer.get_path(), state_name_map_scorer())

        self.out_bos_index.set(bos_index)
        self.out_num_layers.set(num_layers)
        self.out_t_max.set(self.t_max)
