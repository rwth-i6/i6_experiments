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
    return torch.nn.functional.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=blank_idx, reduction="none", zero_infinity=True
    )


def test():
    print("PyTorch:", torch.__version__)
    torch.random.manual_seed(42)

    num_labels = 11
    batch_size = 3
    max_target_len = 5
    max_input_len = 17
    batch_idx = num_labels - 1
    devices = [torch.device("cpu"), torch.device("cuda")]
    funcs = [neg_log_prob_pure_torch, neg_log_prob_torch_ctc]

    _log_probs = torch.randn(max_input_len, batch_size, num_labels)
    log_probs_cases = {
        "normalized": _log_probs.log_softmax(dim=-1),
        "non_normalized": _log_probs,
    }
    targets = torch.randint(0, num_labels - 1, (batch_size, max_target_len))
    input_lengths = torch.randint(1, max_input_len + 1, (batch_size,))
    target_lengths = torch.randint(1, max_target_len + 1, (batch_size,))
    input_lengths = torch.minimum(target_lengths * 2, input_lengths)  # make sure there is a valid path

    for case, log_probs in log_probs_cases.items():
        ref_scores = None
        ref_grads = None

        for dev in devices:
            print("***", dev, case)

            for func in funcs:
                print("  *", func.__name__)

                log_probs: torch.Tensor = log_probs.detach().clone().to(dev)
                log_probs.requires_grad_()
                log_probs.grad = None

                targets: torch.Tensor = targets.clone().to(dev)
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
                assert log_probs.grad is not None
                log_probs_grad_cpu = log_probs.grad.cpu()  # [T, N, C]

                if case == "normalized":
                    # The negative grads should be a probability distrib. Check this.
                    y_sum = -log_probs_grad_cpu.sum(dim=-1)  # [T, N]
                    got_exc = False
                    for b in range(batch_size):
                        for t in range(input_lengths[b]):
                            try:
                                torch.testing.assert_close(y_sum[t, b], torch.tensor(1.0))
                            except Exception as exc:
                                print(f"    Error at t={t} b={b}:")
                                print(exc)
                                got_exc = True
                                break
                        if got_exc:
                            break
                    if not got_exc:
                        print("    grad is neg prob distrib")

                # All the padded frames should have zero grad.
                got_exc = False
                for b in range(batch_size):
                    for t in range(input_lengths[b], max_input_len):
                        if not (log_probs_grad_cpu[t, b] == 0.0).all():
                            print(f"    Grad-zero check: Error at t={t} b={b}: {log_probs_grad_cpu[t, b]}")
                            got_exc = True
                            break
                    if got_exc:
                        break
                if not got_exc:
                    print("    grad is zero for padded frames")

                if ref_grads is None:
                    ref_grads = log_probs_grad_cpu
                else:
                    try:
                        torch.testing.assert_close(ref_grads, log_probs_grad_cpu)
                    except Exception as exc:
                        print("    Error comparing grads to ref grads:")
                        print(exc)
                    else:
                        print("    Grad matches to reference")


def _setup():
    sys.path.insert(0, "projects/2024-alignment-analysis")


if __name__ == "__main__":
    _setup()
    test()
    print("Done.")
