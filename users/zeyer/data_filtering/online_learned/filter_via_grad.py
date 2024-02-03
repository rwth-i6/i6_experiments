"""
Learned filter
"""

from typing import Optional, Any, Tuple, Dict
import torch
from torch import nn
from .filter_base import LearnedDataFilterBase


class LearnedDataFilterViaGrad(LearnedDataFilterBase):
    """
    learned filter
    """

    def __init__(self, *args, **kwargs):
        """
        :param in_features: for forward, what input dimension to expect
        """
        super().__init__(*args, **kwargs)

        self._score_estimator_last_loss: Optional[torch.Tensor] = None  # scalar
        self._fwd_device: Optional[torch.device] = None

    def reset_epoch(self):
        """
        Call this at the beginning or end of an epoch.
        """
        self._score_estimator_last_loss = None

    def forward(
        self, x: torch.Tensor, *, seq_lens: torch.Tensor, btd_axes: Tuple[int, int, int] = (0, 1, 2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward

        :param x: [B,T,D]
        :param seq_lens: [B]
        :param btd_axes: B axis, T axis, D axis
        """
        x, seq_lens = super().forward(x, seq_lens=seq_lens, btd_axes=btd_axes)
        self._fwd_device = x.device
        est_scores, est_scores_seq_lens = self._get_filtered_estimated_scores()
        x = _ForwardWithEstimatedScores.apply(x, btd_axes, est_scores, est_scores_seq_lens, self)
        return x, seq_lens

    def score_estimator_loss(self) -> torch.Tensor:
        """
        Returns the last loss, as scalar.
        (Currently, it is actually from the last step, but consider this an implementation detail.)
        """
        if self._score_estimator_last_loss is not None:
            return self._score_estimator_last_loss
        assert self._fwd_device is not None, "forward was not called?"
        return torch.zeros((), device=self._fwd_device)

    def _set_score_estimator_loss(self, loss: torch.Tensor):
        self._score_estimator_last_loss = loss

    def _real_scores_from_input_grad(self, grad: torch.Tensor, *, btd_axes: Tuple[int, int, int]) -> torch.Tensor:
        grad = grad.permute([btd_axes[0], btd_axes[2], btd_axes[1]])  # [B',D,T_]
        scores = grad.abs().mean(dim=1, keepdim=True)  # [B',1,T_]
        stride = self.score_estimator.total_stride
        scores = torch.nn.functional.avg_pool1d(scores, stride, padding=(stride - 1) // 2)  # [B',1,T'_]
        scores = scores.squeeze(dim=1)  # [B',T'_]
        return scores


class _ForwardWithEstimatedScores(torch.autograd.Function):
    """
    Similar as synthetic gradients, see e.g. https://github.com/koz4k/dni-pytorch/blob/master/dni.py.
    """

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        x_btd_axes: Tuple[int, int, int],
        est_scores: torch.Tensor,
        est_scores_seq_len: torch.Tensor,
        filter_mod: LearnedDataFilterViaGrad,
    ) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert isinstance(x_btd_axes, tuple) and len(x_btd_axes) == 3
        assert isinstance(est_scores, torch.Tensor) and isinstance(est_scores_seq_len, torch.Tensor)
        assert x.shape[x_btd_axes[0]] == est_scores.shape[0] == est_scores_seq_len.shape[0]
        assert isinstance(filter_mod, LearnedDataFilterViaGrad)
        ctx.save_for_backward(est_scores, est_scores_seq_len)
        ctx.x_btd_axes = x_btd_axes
        ctx.filter_mod = filter_mod
        return x.clone()

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, input_grad: torch.Tensor):
        # noinspection PyUnresolvedReferences
        filter_mod: LearnedDataFilterViaGrad = ctx.filter_mod
        ctx.filter_mod = None  # https://discuss.pytorch.org/t/68732, mem leak otherwise?
        x_btd_axes: Tuple[int, int, int] = ctx.x_btd_axes
        ctx.x_btd_axes = None
        # noinspection PyProtectedMember
        real_scores = filter_mod._real_scores_from_input_grad(input_grad, btd_axes=x_btd_axes)  # [B',T'_]
        # noinspection PyProtectedMember
        est_scores, est_scores_seq_lens = ctx.saved_tensors  # [B',T'_]
        est_scores_time_size = est_scores.shape[1]
        scores_diff = est_scores - real_scores
        # noinspection PyTypeChecker
        mask = torch.range(0, est_scores_time_size)[None, :] < est_scores_seq_lens[:, None]  # [B',T'_]
        mask: torch.Tensor
        scores_diff = torch.where(mask.to(scores_diff.device), scores_diff, 0.0)
        loss = scores_diff.square().mean() * (est_scores_time_size / est_scores_seq_lens.sum())
        # noinspection PyProtectedMember
        filter_mod._set_score_estimator_loss(loss.detach())
        # compute MSE gradient manually to avoid dependency on PyTorch internals
        estimated_scores_grad = 2 / est_scores_seq_lens.sum() * scores_diff
        return input_grad, None, estimated_scores_grad, None, None


def demo():
    """demo"""

    class Model(nn.Module):
        def __init__(self, in_features: int, out_features: int, *, hidden_size: int = 10):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.layer1 = nn.Linear(in_features, hidden_size)
            self.data_filter = LearnedDataFilterViaGrad(hidden_size, hidden_size // 2)
            self.layer2 = nn.Linear(hidden_size, hidden_size)
            self.layer3 = nn.Linear(hidden_size, out_features)

        # noinspection PyShadowingNames
        def forward(
            self, x: torch.Tensor, seq_lens: torch.Tensor, *, targets: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            """
            :param x: [B,T,D]
            :param seq_lens: [B]
            :param targets: [B,T]
            :return: y [B',T,D], loss [B']
            """
            x = self.layer1(x).relu()  # [B,T,D]
            x, seq_lens = self.data_filter(x, seq_lens=seq_lens, btd_axes=(0, 1, 2))  # [B',T_,D]
            targets = self.data_filter.filter_batch(targets, time_axis=1)  # [B',T_]
            x = self.layer2(x).relu()  # [B',T,D]
            x = self.layer3(x)  # [B',T_,D]
            model_loss = torch.nn.functional.cross_entropy(x.permute(0, 2, 1), targets, reduction="none")  # [B',T_]
            return x, {"model": model_loss.mean(), "data_filter": self.data_filter.score_estimator_loss()}

    model = Model(5, 5)
    opt = torch.optim.Adam(model.parameters())
    while True:
        opt.zero_grad()
        b, t = 5, 7
        x = torch.randn((b, t, model.in_features))
        targets = torch.randint(0, model.out_features, (b, t))
        seq_lens = torch.tensor([t, t - 1, t - 2, t - 3, t - 4])
        assert seq_lens.shape == x.shape[:1]
        _, loss = model(x, seq_lens=seq_lens, targets=targets)
        print("loss:", loss)
        total_loss: torch.Tensor = sum(v for v in loss.values())
        total_loss.backward()
        opt.step()


if __name__ == "__main__":
    demo()
