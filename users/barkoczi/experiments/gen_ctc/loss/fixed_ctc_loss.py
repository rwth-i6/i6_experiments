import torch


def torch_ctc_fixed_grad(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    global _FixCTCGradFunc, _StoreGradScaleFunc
    if not _FixCTCGradFunc or not _StoreGradScaleFunc:

        class _FixCTCGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, log_probs, input_lengths):
                loss_scale_buffer = {}
                ctx.loss_scale_buffer = loss_scale_buffer
                ctx.save_for_backward(log_probs, input_lengths)
                return log_probs, loss_scale_buffer

            @staticmethod
            def backward(ctx, grad_output, _grad_scale):
                loss_scale_buffer = ctx.loss_scale_buffer.pop("scale")
                log_probs, input_lengths = ctx.saved_tensors
                grad_input = grad_output - log_probs.exp()
                if loss_scale_buffer.ndim == 1:
                    grad_input.multiply_(loss_scale_buffer[None, :, None])
                else:
                    grad_input.multiply_(loss_scale_buffer)
                input_lengths = input_lengths.to(grad_input.device)
                mask = torch.arange(grad_input.shape[0], device=input_lengths.device)[:, None] < input_lengths[None, :]
                return torch.where(mask[:, :, None], grad_input, torch.zeros_like(grad_input)), None

        class _StoreGradScaleFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, loss, loss_scale_buffer):
                ctx.loss_scale_buffer = loss_scale_buffer
                return loss.clone()

            @staticmethod
            def backward(ctx, grad_output):
                ctx.loss_scale_buffer["scale"] = grad_output
                return torch.ones_like(grad_output), None

        _FixCTCGradFunc = _FixCTCGradFunc
        _StoreGradScaleFunc = _StoreGradScaleFunc

    log_probs, loss_scale_buffer = _FixCTCGradFunc.apply(log_probs, input_lengths)
    loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, *args, **kwargs)
    return _StoreGradScaleFunc.apply(loss, loss_scale_buffer)


_FixCTCGradFunc = None
_StoreGradScaleFunc = None
