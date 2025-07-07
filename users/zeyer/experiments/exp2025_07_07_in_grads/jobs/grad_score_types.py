from typing import TYPE_CHECKING, Union, Protocol

if TYPE_CHECKING:
    import torch


class GradScoreTypeInterface(Protocol):
    def __call__(self, act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient score for the given activation and gradient tensors.
        This is supposed to be for one specific output label, over a sequence of inputs.

        :param act: activation (forward pass), shape [time, features]
        :param grad: grad, same shape [time, features]
        :return: gradient score, shape [time]
        """


def get_grad_score_func(s: Union[str, GradScoreTypeInterface]) -> GradScoreTypeInterface:
    """
    Factory function to create an instance of the class specified by the string `s`.
    The class must implement the Interface protocol.
    """
    import re

    if callable(s):
        return s  # Already a callable, return it directly

    if not isinstance(s, str):
        raise TypeError(f"Expected a string or callable, got {type(s).__name__}")

    if s == "dot_e_grad":
        return DotActGrad()
    elif m := re.match(r"^L(\d+(\.\d+)?)_(e_)?grad$", s):
        p_s = m.group(1)
        if p_s[0:1] == "0" and len(p_s) > 1 and "." not in p_s:
            p_s = "0." + p_s[1:]
        p = float(p_s)
        if m.group(2) is None:
            return LpNormGrad(p)
        else:
            return LpNormActGrad(p)
    else:
        raise ValueError(f"Unknown gradient score type: {s!r}")


class DotActGrad:
    def __call__(self, act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return (act * grad).sum(dim=-1)


class LpNormGrad:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return torch.norm(grad, p=self.p, dim=-1)


class LpNormActGrad:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return torch.norm(act * grad, p=self.p, dim=-1)
