from typing import Optional, Any, Tuple, Dict
import torch
from torch import nn


class ScoreEstimator(nn.Module):
    """
    Estimate scores [B,T'], where T' is downsampled from input [B,T,D].
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int = None,
        *,
        kernel_size: int = 3,
        stride: int = 3,
    ):
        super().__init__()

        self.in_features = in_features
        if hidden_size is None:
            hidden_size = max(in_features // 8, 10)
        self.hidden_size = hidden_size

        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.score_estimator = nn.Conv1d(
            in_features, hidden_size * 2, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.score_estimator_out = nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.total_stride = stride**2

    def forward(
        self, x: torch.Tensor, *, seq_lens: torch.Tensor, btd_axes: Tuple[int, int, int] = (0, 1, 2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: [B,T,D]
        :param seq_lens: [B]
        :param btd_axes: B axis, T axis, D axis
        :return: scores [B,T'], scores seq lens [B]
        """
        assert len(btd_axes) == 3
        assert x.shape[btd_axes[2]] == self.in_features, f"input shape {x.shape} unexpected"
        x = x.permute(btd_axes[0], btd_axes[2], btd_axes[1])  # [B,D_in,T]
        x = self.score_estimator(x)  # [B,D_hidden*2,T'']
        x, _ = x.reshape(x.shape[0], self.hidden_size, 2, x.shape[-1]).max(dim=2)  # maxout, [B,D_hidden*2,T'']
        x = self.score_estimator_out(x)  # [B,1,T']
        x = x.squeeze(dim=1)  # [B,T']
        seq_lens = _calc_seq_lens_after_conv(seq_lens, self.score_estimator)
        seq_lens = _calc_seq_lens_after_conv(seq_lens, self.score_estimator_out)
        return x, seq_lens


class LearnedDataFilterBase(nn.Module):
    """
    Learned filter base class
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        batch_factor: float = 0.5,
        score_estimator_opts: Optional[Dict[str, Any]] = None,
    ):
        """
        :param in_features: for forward, what input dimension to expect
        """
        super().__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.batch_factor = batch_factor

        self.score_estimator = ScoreEstimator(in_features, hidden_size, **(score_estimator_opts or {}))

        self._selected_batch_mask: Optional[Dict[torch.device, torch.Tensor]] = None  # [B]
        self._selected_batch_num_entries: Optional[int] = None  # sum(mask) == B'
        self._selected_seq_lens: Optional[torch.Tensor] = None  # [B']
        self._selected_max_time: Optional[int] = None  # scalar, max(filter(seq_len))
        self._estimated_scores: Optional[torch.Tensor] = None  # [B,T']
        self._estimated_scores_seq_lens: Optional[torch.Tensor] = None  # [B]

    def reset_step(self):
        """
        Call this at the beginning or at the end of a step.
        """
        self._selected_batch_mask = None
        self._selected_batch_num_entries = None
        self._selected_seq_lens = None
        self._selected_max_time = None
        self._estimated_scores = None
        self._estimated_scores_seq_lens = None

    def forward(
        self, x: torch.Tensor, *, seq_lens: torch.Tensor, btd_axes: Tuple[int, int, int] = (0, 1, 2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward

        :param x: [B,T,D]
        :param seq_lens: [B]
        :param btd_axes: B axis, T axis, D axis
        :return: filter(x) [B',T_,D], filter(seq_lens) [B'].
            B' is filtered, T_ the max seq len of the filtered seqs.
        """
        self._estimate_scores(x, seq_lens=seq_lens, btd_axes=btd_axes)
        self._select_mask()
        self._selected_seq_lens = self.filter_batch(seq_lens)
        self._selected_max_time = max(self._selected_seq_lens)
        x = self.filter_batch(x, batch_axis=btd_axes[0], time_axis=btd_axes[1])
        return x, self._selected_seq_lens

    def _estimate_scores(self, x: torch.Tensor, *, seq_lens: torch.Tensor, btd_axes: Tuple[int, int, int] = (0, 1, 2)):
        """
        :param x: [B,T,D]
        :param btd_axes: B axis, T axis, D axis
        """
        scores, scores_seq_lens = self.score_estimator(x, seq_lens=seq_lens, btd_axes=btd_axes)
        self._estimated_scores = scores
        self._estimated_scores_seq_lens = scores_seq_lens

    def _select_mask(self):
        est_scores = self._estimated_scores  # [B,T']
        est_scores = est_scores.mean(dim=1)  # [B]
        est_scores = est_scores * (self._estimated_scores.shape[1] / self._estimated_scores_seq_lens)
        batch_size = est_scores.shape[0]
        self._selected_batch_num_entries = int(batch_size * self.batch_factor)
        _, indices = torch.topk(est_scores, self._selected_batch_num_entries)
        mask = torch.full(est_scores.shape, False, dtype=torch.bool, device=est_scores.device)
        mask.index_fill_(0, indices, True)
        mask_cpu = mask.to(torch.device("cpu"))
        self._selected_batch_mask = {mask.device: mask, mask_cpu.device: mask_cpu}

    def filter_batch(self, x: torch.Tensor, *, batch_axis: int = 0, time_axis: Optional[int] = None) -> torch.Tensor:
        """
        filter after current forward

        :param x: shape [B, ...]
        :param batch_axis:
        :param time_axis: if specified, will shorten time by selected_max_time
        :return: shape [B', ...]
        """
        assert self._selected_batch_mask is not None, "forward not called yet?"
        mask = self._selected_batch_mask[x.device]
        batch_size = mask.shape[0]
        assert (
            batch_size == x.shape[batch_axis]
        ), f"shape mismatch, mask {mask.shape} vs input {x.shape}, batch axis {batch_axis}"
        bc_shape = [1] * x.ndim
        bc_shape[batch_axis] = batch_size
        mask_bc = mask.reshape(bc_shape)
        out_shape = list(x.shape)
        out_shape[batch_axis] = self._selected_batch_num_entries
        y = torch.masked_select(x, mask_bc).reshape(out_shape)
        if time_axis is not None:
            y = y[(slice(None),) * time_axis + (slice(None, self._selected_max_time),)]
        return y

    def _get_filtered_estimated_scores(self) -> Tuple[torch.Tensor, torch.Tensor]:
        est_scores = self.filter_batch(self._estimated_scores)  # [B',T']
        est_scores_seq_lens = self.filter_batch(self._estimated_scores_seq_lens)  # [B']
        est_scores_time_size = max(est_scores_seq_lens)
        est_scores = est_scores[:, :est_scores_time_size]  # [B',T'_]
        return est_scores, est_scores_seq_lens


def _calc_seq_lens_after_conv(seq_lens: torch.Tensor, conv: torch.nn.Conv1d):
    return _ceil_div(seq_lens - conv.kernel_size[0] + 1 + conv.padding[0] * 2, conv.stride[0])


def _ceil_div(a, b):
    return -(-a // b)


def make_learned_data_filter(opts: Dict[str, Any], *, in_features: int) -> LearnedDataFilterBase:
    """
    Some helper
    """
    from . import filter_via_loss, filter_via_grad

    classes = {}
    for mod in [filter_via_loss, filter_via_grad]:
        for k, v in vars(mod).items():
            if isinstance(v, type) and issubclass(v, LearnedDataFilterBase):
                classes[k] = v

    opts_ = dict(opts)
    cls_name = opts_.pop("class", "LearnedDataFilterViaLoss")
    cls = classes[cls_name]
    return cls(in_features, **opts_)
