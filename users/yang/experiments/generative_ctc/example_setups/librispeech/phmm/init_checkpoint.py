import subprocess
import textwrap
from typing import Dict, Any

from sisyphus import Job, Task, tk


class InitializeGaussianCheckpointJob(Job):
    """
    Create a RETURNN-style PyTorch checkpoint by instantiating the model once
    with the provided model config and saving its initialized state_dict.
    """

    def __init__(
        self,
        *,
        python_exe: tk.Path,
        recipe_root: tk.Path,
        network_module: str,
        model_config_dict: Dict[str, Any],
        format_version: int = 1,
    ):
        self.format_version = format_version
        self.python_exe = python_exe
        self.recipe_root = recipe_root
        self.network_module = network_module
        self.model_config_dict = model_config_dict

        self.out_checkpoint = self.output_path("init_checkpoint.pt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        script = self.output_path("init_checkpoint.py")
        config_repr = self._to_python_expr(self.model_config_dict)

        script_text = textwrap.dedent(
            f"""
            import sys
            import torch

            sys.path.insert(0, {repr(tk.uncached_path(self.recipe_root))})

            from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.pytorch_networks.{self.network_module} import Model

            model_config_dict = {config_repr}

            model = Model(model_config_dict=model_config_dict)
            torch.save(
                {{
                    "model": model.state_dict(),
                    "epoch": 1,
                    "step": 0,
                    "effective_learning_rate": None,
                    "returnn_version": None,
                }},
                {repr(self.out_checkpoint.get_path())},
            )
            """
        ).strip() + "\n"

        with open(script.get_path(), "w", encoding="utf-8") as f:
            f.write(script_text)

        subprocess.check_call(
            [
                tk.uncached_path(self.python_exe),
                script.get_path(),
            ]
        )

    def _to_python_expr(self, value):
        if isinstance(value, dict):
            items = ", ".join(f"{self._to_python_expr(k)}: {self._to_python_expr(v)}" for k, v in value.items())
            return "{" + items + "}"
        if isinstance(value, (list, tuple)):
            items = ", ".join(self._to_python_expr(v) for v in value)
            if isinstance(value, tuple):
                if len(value) == 1:
                    items += ","
                return "(" + items + ")"
            return "[" + items + "]"
        if hasattr(value, "get_path"):
            return repr(tk.uncached_path(value))
        if hasattr(value, "get"):
            return self._to_python_expr(value.get())
        if isinstance(value, str):
            value = value.strip()
            try:
                return repr(int(value))
            except ValueError:
                try:
                    return repr(float(value))
                except ValueError:
                    return repr(value)
        return repr(value)
