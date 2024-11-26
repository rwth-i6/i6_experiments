"""
Testing different CTC (full sum) implementations which use non-normalized log probs (e.g. with prior).

PyTorch CTC does it wrong (at least the grad is wrong).
https://github.com/pytorch/pytorch/issues/52241
"""

import torch
import sys


def neg_log_prob_pure_torch(
    *,
    log_probs: torch.Tensor,
    blank_idx: int,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Like :func:`neg_log_prob_torch_ctc`.

    :param log_probs: (T, N, C)
        where T is the length of the sequence, N is the batch size, C is the number of classes (including the blank).
        The user is responsible for normalization.
    :param blank_idx: the index of the blank symbol (in C)
    :param targets: (N, S), where S is the target length
    :param input_lengths: (N,)
    :param target_lengths: (N,)
    :return: (N,), the forward score (CTC loss)
    """
    from alignments.loss import ctc_loss, SizedTensor

    return ctc_loss(
        log_probs=SizedTensor(log_probs.permute(1, 0, 2), seq_lens=input_lengths),
        targets=SizedTensor(targets, seq_lens=target_lengths),
        blank_idx=blank_idx,
    )


def neg_log_prob_torch_ctc(
    *,
    log_probs: torch.Tensor,
    blank_idx: int,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Forward score with CTC loss using torch.nn.functional.ctc_loss.

    :param log_probs: (T, N, C)
        where T is the length of the sequence, N is the batch size, C is the number of classes (including the blank).
        The user is responsible for normalization.
    :param blank_idx: the index of the blank symbol (in C)
    :param targets: (N, S), where S is the target length
    :param input_lengths: (N,)
    :param target_lengths: (N,)
    :return: (N,), the forward score (CTC loss)
    """
    # Note: the CTC loss calcs the correct scores, but the wrong grad.
    # This is actually the grad w.r.t. the logits, i.e. it calculates sm(x) - y,
    # where y are the soft targets.
    return torch.nn.functional.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=blank_idx, reduction="none", zero_infinity=True
    )


def neg_log_prob_torch_ctc_fixed_grad(
    *,
    log_probs: torch.Tensor,
    blank_idx: int,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Forward score with CTC loss using torch.nn.functional.ctc_loss
    and fixing up the gradient.

    :param log_probs: (T, N, C)
        where T is the length of the sequence, N is the batch size, C is the number of classes (including the blank).
        The user is responsible for normalization.
    :param blank_idx: the index of the blank symbol (in C)
    :param targets: (N, S), where S is the target length
    :param input_lengths: (N,)
    :param target_lengths: (N,)
    :return: (N,), the forward score (CTC loss)
    """
    return torch_ctc_fixed_grad(
        log_probs, targets, input_lengths, target_lengths, blank=blank_idx, reduction="none", zero_infinity=True
    )


def torch_ctc_fixed_grad(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    log_probs = _FixCTCGradFunc.apply(log_probs, input_lengths)
    return torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, *args, **kwargs)


class _FixCTCGradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, input_lengths):
        ctx.save_for_backward(log_probs, input_lengths)
        return log_probs

    @staticmethod
    def backward(ctx, grad_output):
        (log_probs, input_lengths) = ctx.saved_tensors

        # The ctc_loss calculates exp(log_probs) - y, where y are the soft targets.
        # We want to return -y instead.
        # Thus, subtract the exp(log_probs) from the grad_output.
        grad_input = grad_output - log_probs.exp()
        input_lengths = input_lengths.to(grad_input.device)
        max_time = grad_input.shape[0]
        mask = torch.arange(max_time, device=input_lengths.device)[:, None] < input_lengths[None, :]  # [T, N]
        grad_input = torch.where(mask[:, :, None], grad_input, torch.zeros_like(grad_input))

        return grad_input, None


def test():
    print("PyTorch:", torch.__version__)
    torch.random.manual_seed(42)

    num_labels = 11
    batch_size = 3
    max_target_len = 5
    max_input_len = 17
    batch_idx = num_labels - 1
    devices = [torch.device("cpu"), torch.device("cuda")]
    # devices = [torch.device("cpu")]
    funcs = [neg_log_prob_pure_torch, neg_log_prob_torch_ctc, neg_log_prob_torch_ctc_fixed_grad]
    eps = 1e-4
    do_gradcheck = False
    with_log_prob_grad = True

    _log_probs: torch.Tensor = torch.randn(max_input_len, batch_size, num_labels) * 3.0
    targets = torch.randint(0, num_labels - 1, (batch_size, max_target_len))
    input_lengths = torch.randint(1, max_input_len + 1, (batch_size,))
    target_lengths = torch.randint(1, max_target_len + 1, (batch_size,))
    input_lengths = torch.maximum(target_lengths * 2, input_lengths)  # make sure there is a valid path

    if do_gradcheck:
        for func in funcs:
            print("* Checking grad for", func.__name__)
            logits = _log_probs.detach().clone().double()
            logits.requires_grad_()
            torch.autograd.gradcheck(
                lambda x: func(
                    log_probs=torch.nn.functional.log_softmax(x, -1),
                    targets=targets,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                    blank_idx=batch_idx,
                ),
                logits,
            )

    for case in ["normalized", "non_normalized"]:
        ref_scores = None
        ref_logits_grads = None
        ref_log_prob_grads = None

        for dev in devices:
            print("***", dev, case)

            for func in funcs:
                print("  *", func.__name__)

                logits: torch.Tensor = _log_probs.detach().clone().to(dev)
                logits.requires_grad_()
                if case == "normalized":
                    log_probs = logits.log_softmax(dim=-1)
                    if with_log_prob_grad:
                        log_probs.requires_grad_()
                        log_probs.retain_grad()
                else:
                    log_probs = logits
                targets: torch.Tensor = targets.to(dev)

                loss = func(
                    log_probs=log_probs,
                    blank_idx=batch_idx,
                    targets=targets,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                )
                assert loss.shape == (batch_size,)
                print("    loss:", loss)
                if ref_scores is None:
                    ref_scores = loss
                else:
                    torch.testing.assert_close(ref_scores, loss.to(ref_scores.device))
                    print("    loss matches to reference")

                loss.sum().backward()
                assert logits.grad is not None
                if with_log_prob_grad:
                    assert log_probs.grad is not None
                logits_grad_cpu = logits.grad.cpu()  # [T, N, C]
                log_probs_grad_cpu = log_probs.grad.cpu() if with_log_prob_grad else None  # [T, N, C]

                # print("logits grads (0,0):")
                # print(logits_grad_cpu[0, 0])
                # if with_log_prob_grad:
                #     print("log probs grads (0,0):")
                #     print(log_probs_grad_cpu[0, 0])

                if case == "normalized":
                    # The grads of logits should be like probs - y.
                    y = log_probs.exp().cpu() - logits_grad_cpu
                    y_sum = y.sum(dim=-1)  # [T, N]
                    got_exc = False
                    for b in range(batch_size):
                        for t in range(input_lengths[b]):
                            if not torch.isclose(y_sum[t, b], torch.tensor(1.0), atol=eps):
                                print(f"    logits grad error at t={t} b={b}: {y_sum[t, b]} vs 1.")
                                got_exc = True
                                break
                            if not (y[t, b] >= -eps).all():
                                print(f"    logits grad error at t={t} b={b}: got negative: {y[t, b]}.")
                                got_exc = True
                                break
                        if got_exc:
                            break
                    if not got_exc:
                        print("    logits grad + probs is prob distrib")

                if with_log_prob_grad:
                    # The negative grads should be a probability distrib (even when not normalized before).
                    # Check this.
                    y = -log_probs_grad_cpu
                    y_sum = y.sum(dim=-1)  # [T, N]
                    got_exc = False
                    for b in range(batch_size):
                        for t in range(input_lengths[b]):
                            if not torch.isclose(y_sum[t, b], torch.tensor(1.0), atol=eps):
                                print(f"    log probs grad error at t={t} b={b}: {y_sum[t, b]} vs 1.")
                                got_exc = True
                                break
                            if not (y[t, b] >= -eps).all():
                                print(f"    log probs grad error at t={t} b={b}: got negative: {y[t, b]}.")
                                got_exc = True
                                break
                        if got_exc:
                            break
                    if not got_exc:
                        print("    log prob grad is neg prob distrib")

                # All the padded frames should have zero grad.
                got_exc = False
                for b in range(batch_size):
                    for t in range(input_lengths[b], max_input_len):
                        if not (logits_grad_cpu[t, b] == 0.0).all():
                            print(f"    Logits grad zero check: Error at t={t} b={b}: {logits_grad_cpu[t, b]}")
                            got_exc = True
                            break
                    if got_exc:
                        break
                if not got_exc:
                    print("    logits grad is zero for padded frames")

                if with_log_prob_grad:
                    # All the padded frames should have zero grad.
                    got_exc = False
                    for b in range(batch_size):
                        for t in range(input_lengths[b], max_input_len):
                            if not (log_probs_grad_cpu[t, b] == 0.0).all():
                                print(f"    Log prob grad zero check: Error at t={t} b={b}: {log_probs_grad_cpu[t, b]}")
                                got_exc = True
                                break
                        if got_exc:
                            break
                    if not got_exc:
                        print("    log prob grad is zero for padded frames")

                if ref_logits_grads is None:
                    ref_logits_grads = logits_grad_cpu
                else:
                    try:
                        torch.testing.assert_close(ref_logits_grads, logits_grad_cpu, rtol=eps, atol=eps)
                    except Exception as exc:
                        print("    Error comparing grads to ref grads:")
                        print(exc)
                    else:
                        print("    Logits grad matches to reference")

                if with_log_prob_grad:
                    if ref_log_prob_grads is None:
                        ref_log_prob_grads = log_probs_grad_cpu
                    else:
                        try:
                            torch.testing.assert_close(ref_log_prob_grads, log_probs_grad_cpu, rtol=eps, atol=eps)
                        except Exception as exc:
                            print("    Error comparing log prob grads to ref log prob grads:")
                            print(exc)
                        else:
                            print("    Log prob grad matches to reference")


def _setup():
    sys.path.insert(0, "projects/2024-alignment-analysis")


if __name__ == "__main__":
    _setup()
    test()
    print("Done.")
