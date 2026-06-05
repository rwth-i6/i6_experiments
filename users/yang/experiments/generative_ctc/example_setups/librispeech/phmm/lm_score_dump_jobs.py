from __future__ import annotations

import subprocess
import textwrap
from typing import Dict, Any

from sisyphus import Job, Task, tk


class DumpNgramConvLmScoreTableJob(Job):
    """
    Dump a dense n-gram LM log-probability table.

    The output tensor has shape [vocab_size ** context_length, vocab_size].
    Row ids are base-vocab encoded contexts. For context [a, b, c]:

        context_id = ((a * vocab_size) + b) * vocab_size + c

    The job is intended for the v2 n-gram conv LM where no zero-vector padding is
    used and every context is represented by explicit token ids.
    """

    def __init__(
        self,
        *,
        lm_checkpoint: tk.Path,
        python_exe: tk.Path,
        recipe_root: tk.Path,
        context_length: int,
        vocab_size: int,
        model_config_dict: Dict[str, Any] | None = None,
        chunk_size: int = 8192,
        output_filename: str = "ngram_lm_log_probs.pt",
        mem_rqmt: int = 8,
        time_rqmt: int = 2,
    ):
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.lm_checkpoint = lm_checkpoint
        self.python_exe = python_exe
        self.recipe_root = recipe_root
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.model_config_dict = model_config_dict or {
            "vocab_size": vocab_size,
            "embedding_dim": 128,
            "conv_channels": 256,
            "conv_kernel_size": context_length,
            "projection_dim": 256,
            "dropout": 0.0,
            "pad_token_id": 0,
            "bos_token_id": vocab_size - 1,
        }
        self.chunk_size = chunk_size
        self.mem_rqmt = mem_rqmt
        self.time_rqmt = time_rqmt
        self.out_scores = self.output_path(output_filename)
        self.out_stats = self.output_path("ngram_lm_score_table_stats.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt})

    @staticmethod
    def _to_python_expr(value):
        if isinstance(value, dict):
            items = ", ".join(
                f"{DumpNgramConvLmScoreTableJob._to_python_expr(k)}: "
                f"{DumpNgramConvLmScoreTableJob._to_python_expr(v)}"
                for k, v in value.items()
            )
            return "{" + items + "}"
        if isinstance(value, (list, tuple)):
            items = ", ".join(DumpNgramConvLmScoreTableJob._to_python_expr(v) for v in value)
            if isinstance(value, tuple):
                if len(value) == 1:
                    items += ","
                return "(" + items + ")"
            return "[" + items + "]"
        if hasattr(value, "get_path"):
            return repr(tk.uncached_path(value))
        if hasattr(value, "get"):
            return DumpNgramConvLmScoreTableJob._to_python_expr(value.get())
        return repr(value)

    def run(self):
        script = self.output_path("dump_ngram_lm_scores.py")
        model_config_repr = self._to_python_expr(self.model_config_dict)
        script_text = textwrap.dedent(
            f"""
            import sys
            import torch

            sys.path.insert(0, {repr(tk.uncached_path(self.recipe_root))})

            from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.pytorch_networks.phmm.ngram_conv_lm_v2 import Model

            checkpoint_path = {repr(tk.uncached_path(self.lm_checkpoint))}
            output_path = {repr(self.out_scores.get_path())}
            stats_path = {repr(self.out_stats.get_path())}
            vocab_size = {int(self.vocab_size)}
            context_length = {int(self.context_length)}
            chunk_size = {int(self.chunk_size)}
            model_config_dict = {model_config_repr}

            model = Model(model_config_dict=model_config_dict)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            if any(key.startswith("module.") for key in state_dict):
                state_dict = {{key.removeprefix("module."): value for key, value in state_dict.items()}}
            model.load_state_dict(state_dict, strict=True)
            model.eval()

            num_contexts = vocab_size ** context_length
            out = torch.empty((num_contexts, vocab_size), dtype=torch.float32)
            powers = vocab_size ** torch.arange(context_length - 1, -1, -1, dtype=torch.long)

            with torch.no_grad():
                for start in range(0, num_contexts, chunk_size):
                    end = min(start + chunk_size, num_contexts)
                    ids = torch.arange(start, end, dtype=torch.long)
                    ctx = torch.div(ids[:, None], powers[None, :], rounding_mode="floor") % vocab_size
                    logits = model(ctx)
                    out[start:end] = torch.log_softmax(logits[:, -1], dim=-1).cpu()
                    if start == 0 or end == num_contexts or (start // chunk_size) % 20 == 0:
                        print(f"dumped {{end}}/{{num_contexts}} contexts", flush=True)

            torch.save(out, output_path)
            with open(stats_path, "w") as f:
                f.write(f"checkpoint: {{checkpoint_path}}\\n")
                f.write(f"vocab_size: {{vocab_size}}\\n")
                f.write(f"context_length: {{context_length}}\\n")
                f.write(f"num_contexts: {{num_contexts}}\\n")
                f.write(f"chunk_size: {{chunk_size}}\\n")
                f.write(f"output: {{output_path}}\\n")
                f.write(f"shape: {{tuple(out.shape)}}\\n")
            """
        ).strip() + "\n"

        with open(script.get_path(), "w", encoding="utf-8") as f:
            f.write(script_text)

        subprocess.check_call([tk.uncached_path(self.python_exe), script.get_path()])
