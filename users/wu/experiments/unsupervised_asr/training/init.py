"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/init_jobs.py (``FlatRecognizerInitJob``).

theta's flat init: a recognizer checkpoint whose output logits are exactly zero.  Every phase-4a arm
starts from ``FlatRecognizerInitJob(net_args=config.NET_ARGS, seed=0)`` (the blank-free, 40-output
recognizer); the second-seed replicate uses ``seed=1``.
"""

from typing import Any, Dict, Optional

from sisyphus import Job, Task

__all__ = ["FlatRecognizerInitJob"]


class FlatRecognizerInitJob(Job):
    """A checkpoint whose OUTPUT LOGITS are exactly zero.

    Only the final (logit-producing) convolution is zeroed; the BN / residual / intermediate layers
    keep their default initialisation, because "zero output logits" is a statement about the
    emission distribution -- every frame's posterior is uniform over the outputs -- and not about
    the whole parameter vector.  Written in RETURNN's own checkpoint layout
    (``{"model": state_dict, "epoch": ..., "step": ...}``) so it loads through the same path as a
    trained one.

    ``net_args = None`` is the recognizer-only (CTC, 41-output) net
    (``model.recognizer_only.RECOGNIZER_NET_ARGS``), as in the source; the blank-free arms pass
    :data:`~.config.NET_ARGS`.
    """

    def __init__(self, *, net_args: Optional[Dict[str, Any]] = None, seed: int = 0):
        super().__init__()
        if not net_args:
            from ..model.recognizer_only import RECOGNIZER_NET_ARGS

            net_args = RECOGNIZER_NET_ARGS
        self.net_args = dict(net_args)
        self.seed = seed
        self.out_checkpoint = self.output_path("flat_init.pt")
        self.out_stats = self.output_path("flat_init.stats.txt")
        self.rqmt = {"cpu": 2, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import math

        import torch

        from ..model.recognizer import ConvRecognizer

        torch.manual_seed(self.seed)
        model = ConvRecognizer(**self.net_args)
        with torch.no_grad():
            model.out_logit_layer.weight.zero_()
            if model.out_logit_layer.bias is not None:
                model.out_logit_layer.bias.zero_()
            x = torch.zeros(1, 7, self.net_args["in_dim"])
            lp = model.eval()(x, torch.tensor([7]))
            spread = float(lp.max() - lp.min())
        assert spread < 1e-6, f"flat init is not flat: logit spread {spread}"

        torch.save(
            {"model": model.state_dict(), "epoch": 0, "step": 0, "effective_learning_rate": None},
            self.out_checkpoint.get_path(),
        )
        lines = [
            "SAE 4a init (ii) flat: output logits zeroed, uniform per-frame posterior",
            f"net_args = {self.net_args}",
            f"parameters = {model.num_parameters()}",
            f"log-prob spread on a zero input = {spread:.3e} "
            f"(uniform log-prob = -log {model.n_out} = {-math.log(model.n_out):.6f})",
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
