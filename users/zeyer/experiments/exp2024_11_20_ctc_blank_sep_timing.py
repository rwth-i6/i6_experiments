"""
Do some timings on CTC with separated blank.

Actually currently not really for Sisyphus, but standalone script...

To run many of the things here:

Fish: set -x PYTHONPATH tools/espnet:tools/returnn:tools/sisyphus:recipe
Bash: export PYTHONPATH="tools/espnet:tools/returnn:tools/sisyphus:recipe"

Then: python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_11_20_ctc_blank_sep_timing import ... as f; f()"
For example:
  python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_11_20_ctc_blank_sep_timing import plot_all as f; f()"

Similar as :func:`i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align.visualize_grad_scores`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, List
import sys


if TYPE_CHECKING:
    from returnn.tensor import Tensor


def timings(
    *, n_batch: int = 100, n_seq_len: int = 1000, n_model: int = 512, n_vocab: int = 10_000, n_trials: int = 10
):
    import torch
    import torch.utils.benchmark as benchmark

    model_std = _Model(n_batch=n_batch, n_seq_len=n_seq_len, n_model=n_model, n_vocab=n_vocab)
    model_sep = _Model(n_batch=n_batch, n_seq_len=n_seq_len, n_model=n_model, n_vocab=n_vocab, out_blank_separated=True)

    for case in ["greedy_decode_only_labels", "greedy_decode_with_probs", "train"]:
        print("* Profiling case:", case)
        out1_ = model_sep(case=case, out_blank_separated="simple")
        out2_ = model_sep(case=case, out_blank_separated="efficient")
        assert len(out1_) == len(out2_)
        for out1, out2 in zip(out1_, out2_):
            assert out1.dims == out2.dims and out1.sparse_dim == out2.sparse_dim and out1.dtype == out2.dtype
            torch.testing.assert_allclose(out1.raw_tensor, out2.raw_tensor, rtol=1e-3, atol=1e-4)

        baseline = None
        for out_blank_separated in [False, "simple", "efficient"]:
            print(f"** Profiling case {case} with out_blank_separated={out_blank_separated}")
            timer = benchmark.Timer(
                stmt=f"model(case={case!r}, out_blank_separated={out_blank_separated!r})",
                globals={"model": model_sep if out_blank_separated else model_std},
            )
            measurement = timer.timeit(n_trials)
            print(measurement)
            if not baseline:
                baseline = measurement
            else:
                print(f"Speedup: {(baseline.median / measurement.median - 1.) * 100.:.2f}%")


class _Model:
    def __init__(
        self,
        *,
        n_batch: int = 100,
        n_seq_len: int = 1000,
        n_model: int = 512,
        n_vocab: int = 10_000,
        amount_blank_frac: float = 0.9,
        out_blank_separated: bool = False,
    ):
        import torch
        import returnn.frontend as rf
        from returnn.tensor import Dim

        self.batch_dim = Dim(n_batch, name="batch")
        self.seq_len_dim = Dim(n_seq_len, name="seq_len")
        self.model_dim = Dim(n_model, name="model")
        self.target_dim = Dim(n_vocab, name="vocab")
        self.dummy_blank_feat_dim = Dim(1, name="blank_feat")
        self.wb_target_dim = self.target_dim + self.dummy_blank_feat_dim  # with blank
        self.blank_idx = self.target_dim.dimension
        self.out_blank_separated = out_blank_separated

        self.x = rf.random_uniform(
            (self.batch_dim, self.seq_len_dim, self.model_dim), minval=-1, maxval=1, dtype="float32"
        )
        self.weights = rf.random_uniform((self.wb_target_dim, self.model_dim), minval=-0.1, maxval=0.1, dtype="float32")
        self.bias = rf.random_uniform((self.wb_target_dim,), minval=-0.1, maxval=0.1, dtype="float32")

        # First fill target labels with all blank, then below set some to non-blank.
        self.target_labels = rf.fill(
            dims=[self.batch_dim, self.seq_len_dim], sparse_dim=self.wb_target_dim, fill_value=self.blank_idx
        )

        # Set X% (1-amount_blank_frac) to non-blank of target labels.
        for i in range(round(self.batch_dim.dimension * self.seq_len_dim.dimension * (1 - amount_blank_frac))):
            while True:
                b = torch.randint(0, self.batch_dim.dimension, (1,)).item()
                t = torch.randint(0, self.seq_len_dim.dimension, (1,)).item()
                if self.target_labels.raw_tensor[b, t] == self.blank_idx:
                    break
            self.target_labels.raw_tensor[b, t] = torch.randint(0, self.target_dim.dimension, (1,)).item()

        self._tune_params_for_amount_blank(amount_blank_frac, max_iters=100)

    def _tune_params_for_amount_blank(self, amount_blank_frac_wanted: float, *, max_iters: int = 100):
        import torch

        num_frames = self.batch_dim.dimension * self.seq_len_dim.dimension
        amount_blank_wanted = round(num_frames * amount_blank_frac_wanted)
        print(
            f"Wanted amount of blank: {amount_blank_frac_wanted * 100.:.1f}%,"
            f" {amount_blank_wanted}/{num_frames} frames"
        )
        opt = torch.optim.Adam([self.weights.raw_tensor, self.bias.raw_tensor], lr=0.1)
        for i in range(max_iters):
            (log_probs, *_grads) = self(case="train")
            loss = -log_probs.raw_tensor.sum()
            print(f"Loss {i}: {loss.item() / num_frames:.3f}")
            # loss.backward already called, as the func already calculated the grads.
            opt.step()
            opt.zero_grad()
            amount_blank = self._get_amount_blank()
            print(f"Amount blank: {amount_blank}")
            self._maybe_report_non_blank_mask_count()
            # Do not break early. We want that the distrib over the other labels is realistic (peaky).
        # Now do the final fine-tuning.
        self._tune_bias_for_amount_blank(amount_blank_frac_wanted)

    def _maybe_report_non_blank_mask_count(self):
        import returnn.frontend as rf

        if self.out_blank_separated:
            weights_blank = rf.convert_to_tensor(self.weights.raw_tensor[self.blank_idx], dims=[self.model_dim])
            bias_blank = rf.convert_to_tensor(self.bias.raw_tensor[self.blank_idx], dims=[])
            logits_blank = rf.dot(self.x, weights_blank, reduce=self.model_dim)  # [B, T]
            logits_blank += bias_blank  # [B, T]
            non_blank_mask = logits_blank < 0.0  # [B, T]
            print(
                "Potential non blank count (potential speedup in efficient sep-blank calc):",
                non_blank_mask.raw_tensor.sum().cpu().item(),
                "/",
                non_blank_mask.raw_tensor.numel(),
            )

    def _tune_bias_for_amount_blank(self, amount_blank_frac_wanted: float):
        """
        This just changes the blank bias to get the wanted amount of blank.
        Note that this does not change anything else.
        The distribution over the other labels will be quite random (uniform).
        Thus, the amount of potential non-blank frames via the ``non_blank_mask = logits_blank < 0.0`` mask
        used for the efficient blank separated case, will be very high, often 100%,
        so then we will not get any speedup!
        However, this is an unrealistic case anyway.
        Usually the distribution over the other labels is very peaky.
        Thus, see :func:`_tune_params_for_amount_blank` for a better tuning.
        """
        num_frames = self.batch_dim.dimension * self.seq_len_dim.dimension
        amount_blank_wanted = round(num_frames * amount_blank_frac_wanted)
        bias_blank = 0.0
        self.bias.raw_tensor[self.blank_idx] = bias_blank
        print(
            f"Wanted amount of blank: {amount_blank_frac_wanted * 100.:.1f}%,"
            f" {amount_blank_wanted}/{num_frames} frames"
        )
        while True:
            amount_blank = self._get_amount_blank()
            print(f"Have amount of blank: {amount_blank}/{num_frames}, bias {bias_blank}")
            if amount_blank == amount_blank_wanted:
                self._maybe_report_non_blank_mask_count()
                print(f"Reached wanted amount of blank. (First loop, bias {bias_blank})")
                return
            if amount_blank < amount_blank_wanted:
                break
            bias_blank -= 1.0
            self.bias.raw_tensor[self.blank_idx] = bias_blank
        # Now amount blank frac < amount blank frac wanted.
        while True:
            lower_bound = bias_blank
            bias_blank += 1.0
            self.bias.raw_tensor[self.blank_idx] = bias_blank
            amount_blank = self._get_amount_blank()
            print(f"Have amount of blank: {amount_blank}/{num_frames}, bias {bias_blank}")
            if amount_blank == amount_blank_wanted:
                self._maybe_report_non_blank_mask_count()
                print(f"Reached wanted amount of blank. (Second loop, bias {bias_blank})")
                return
            if amount_blank >= amount_blank_wanted:
                break
        # Now amount blank frac for lower_bound <= amount blank frac wanted
        # and amount blank frac for lower_bound + 1 > amount blank frac wanted.
        upper_bound = bias_blank
        # Now do binary search.
        # No early break needed: It's an integer, we should reach it exactly.
        while True:
            bias_blank = (lower_bound + upper_bound) / 2
            self.bias.raw_tensor[self.blank_idx] = bias_blank
            amount_blank = self._get_amount_blank()
            print(f"Have amount of blank: {amount_blank}/{num_frames}, bias {bias_blank}")
            self._maybe_report_non_blank_mask_count()
            if amount_blank == amount_blank_wanted:
                print(f"Reached wanted amount of blank. (Binary search, bias {bias_blank})")
                return
            if amount_blank < amount_blank_wanted:
                lower_bound = bias_blank
            else:
                upper_bound = bias_blank

    def _get_amount_blank(self) -> int:
        (labels,) = self(case="greedy_decode_only_labels")
        labels: Tensor
        c = (labels.raw_tensor == self.blank_idx).sum().cpu().item()
        return c

    def __call__(self, *, case: str, out_blank_separated: Optional[Union[bool, str]] = None) -> List[Tensor]:
        import returnn.frontend as rf

        for x in [self.x, self.weights]:
            x.raw_tensor.grad = None
            x.raw_tensor.requires_grad = case == "train"

        if out_blank_separated is None and self.out_blank_separated:
            out_blank_separated = "efficient"

        if not self.out_blank_separated:  # standard case, joint distrib incl blank
            assert not out_blank_separated
            logits = rf.dot(self.x, self.weights, reduce=self.model_dim)  # [B, T, V+1]
            logits += self.bias

            if case == "greedy_decode_only_labels":
                return [rf.reduce_argmax(logits, axis=self.wb_target_dim)]  # [B, T]

            denom = rf.reduce_logsumexp(logits, axis=self.wb_target_dim)  # [B, T]

            if case == "greedy_decode_with_probs":
                labels = rf.reduce_argmax(logits, axis=self.wb_target_dim)  # [B, T]
            elif case == "train":
                labels = self.target_labels
            else:
                assert False, f"invalid case {case!r}"

            log_probs = rf.gather(logits, indices=labels) - denom  # [B, T]

        elif out_blank_separated == "simple":
            assert self.out_blank_separated
            logits = rf.dot(self.x, self.weights, reduce=self.model_dim)
            logits += self.bias

            logits_wo_blank, logits_blank = rf.split(
                logits, axis=self.wb_target_dim, out_dims=[self.target_dim, self.dummy_blank_feat_dim]
            )
            log_probs_wo_blank = rf.log_softmax(logits_wo_blank, axis=self.target_dim)
            log_probs_blank = rf.log_sigmoid(logits_blank)
            log_probs_emit = rf.squeeze(rf.log_sigmoid(-logits_blank), axis=self.dummy_blank_feat_dim)
            log_probs, _ = rf.concat(
                (log_probs_wo_blank + log_probs_emit, self.target_dim),
                (log_probs_blank, self.dummy_blank_feat_dim),
                out_dim=self.wb_target_dim,
            )  # [B, T, V+1]
            log_probs.feature_dim = self.wb_target_dim

            if case == "greedy_decode_only_labels":
                return [rf.reduce_argmax(log_probs, axis=self.wb_target_dim)]  # [B, T]

            if case == "greedy_decode_with_probs":
                labels = rf.reduce_argmax(log_probs, axis=self.wb_target_dim)  # [B, T]
            elif case == "train":
                labels = self.target_labels
            else:
                assert False, f"invalid case {case!r}"

            log_probs = rf.gather(log_probs, indices=labels)  # [B, T]

        elif out_blank_separated == "efficient":
            assert self.out_blank_separated

            weights_blank = rf.convert_to_tensor(self.weights.raw_tensor[self.blank_idx], dims=[self.model_dim])
            weights_non_blank = rf.convert_to_tensor(
                self.weights.raw_tensor[: self.blank_idx], dims=[self.target_dim, self.model_dim]
            )
            bias_blank = rf.convert_to_tensor(self.bias.raw_tensor[self.blank_idx], dims=[])
            bias_non_blank = rf.convert_to_tensor(self.bias.raw_tensor[: self.blank_idx], dims=[self.target_dim])

            logits_blank = rf.dot(self.x, weights_blank, reduce=self.model_dim)  # [B, T]
            logits_blank += bias_blank  # [B, T]

            if case.startswith("greedy_decode_"):
                # It's not necessarily blank, but anyway we need to check it.
                non_blank_mask = logits_blank < 0.0  # [B, T]
            elif case == "train":
                non_blank_mask = self.target_labels != self.blank_idx  # [B, T]
            else:
                assert False, f"invalid case {case!r}"

            non_blank_frames_x, non_blank_dim = rf.masked_select(
                self.x, mask=non_blank_mask, dims=[self.batch_dim, self.seq_len_dim]
            )  # [B_T', D]
            non_blank_frames_logits = rf.dot(non_blank_frames_x, weights_non_blank, reduce=self.model_dim)  # [B_T', V]
            non_blank_frames_logits += bias_non_blank  # [B_T', V]
            nb_denom = rf.reduce_logsumexp(non_blank_frames_logits, axis=self.target_dim)  # [B_T']

            nb_logits_blank, _ = rf.masked_select(
                logits_blank,
                mask=non_blank_mask,
                dims=[self.batch_dim, self.seq_len_dim],
                out_dim=non_blank_dim,
            )  # [B_T']

            nb_log_probs_emit = rf.log_sigmoid(-nb_logits_blank)  # [B_T']

            if case == "train":
                # Here we know what labels to look at. This makes it simper.
                non_blank_labels, _ = rf.masked_select(
                    self.target_labels,
                    mask=non_blank_mask,
                    dims=[self.batch_dim, self.seq_len_dim],
                    out_dim=non_blank_dim,
                )  # [B_T']
                non_blank_labels.sparse_dim = self.target_dim
                labels = self.target_labels  # [B, T]
                nb_log_probs = rf.gather(non_blank_frames_logits, indices=non_blank_labels) - nb_denom  # [B_T']
                nb_log_probs = nb_log_probs + nb_log_probs_emit  # [B_T']

            else:  # not train, i.e. greedy decoding
                # We don't know the labels.
                # Non-blank is only potentially non-blank but might turn out to be blank.
                non_blank_frames_log_probs = non_blank_frames_logits - nb_denom + nb_log_probs_emit  # [B_T', V]
                nb_log_probs_blank = rf.log_sigmoid(nb_logits_blank)  # [B_T']
                nb_log_probs, _ = rf.concat(
                    (non_blank_frames_log_probs, self.target_dim),
                    (rf.expand_dim(nb_log_probs_blank, self.dummy_blank_feat_dim), self.dummy_blank_feat_dim),
                    out_dim=self.wb_target_dim,
                )  # [B_T', V+1]
                non_blank_labels = rf.reduce_argmax(nb_log_probs, axis=self.wb_target_dim)  # [B_T']->V+1
                labels = rf.masked_scatter(
                    non_blank_labels,
                    mask=non_blank_mask,
                    dims=[self.batch_dim, self.seq_len_dim],
                    in_dim=non_blank_dim,
                )  # [B, T]
                labels = rf.where(non_blank_mask, labels, self.blank_idx)  # [B, T]
                if case == "greedy_decode_only_labels":
                    return [labels]
                nb_log_probs = rf.gather(nb_log_probs, indices=non_blank_labels)  # [B_T']

            log_probs = rf.masked_scatter(
                nb_log_probs,
                mask=non_blank_mask,
                dims=[self.batch_dim, self.seq_len_dim],
                in_dim=non_blank_dim,
            )  # [B, T]
            log_probs_blank = rf.log_sigmoid(logits_blank)  # [B, T]
            log_probs = rf.where(non_blank_mask, log_probs, log_probs_blank)  # [B, T]

        else:
            assert False, f"invalid out_blank_separated {out_blank_separated!r}, {self.out_blank_separated!r}"

        if case == "greedy_decode_with_probs":
            return [labels, log_probs]  # [B, T]

        (-log_probs.raw_tensor.sum()).backward()
        x_grad = rf.convert_to_tensor(self.x.raw_tensor.grad, dims=[self.batch_dim, self.seq_len_dim, self.model_dim])
        weights_grad = rf.convert_to_tensor(self.weights.raw_tensor.grad, dims=[self.wb_target_dim, self.model_dim])
        return [log_probs, x_grad, weights_grad]


def _setup():
    import i6_core.util as util

    returnn_root = util.get_returnn_root(None)

    sys.path.insert(0, returnn_root.get_path())

    from returnn.util import better_exchook

    better_exchook.setup_all()

    import returnn.frontend as rf

    rf.select_backend_torch()
    rf.set_random_seed(42)


_setup()
