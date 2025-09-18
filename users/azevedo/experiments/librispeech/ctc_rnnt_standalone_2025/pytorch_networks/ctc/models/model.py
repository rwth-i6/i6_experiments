from .._base_streamable_ctc import StreamableCTC as Model
from ...trainers import train_handler
from ..train_step_mode import CTCTrainStepMode, prior_init_hook, prior_finish_hook, prior_step


def train_step(*, model: Model, data, run_ctx, **kwargs):
    # NOTE: only need to change `train_step_mode` for different model
    train_strat: train_handler.TrainingStrategy = None
    train_step_mode = CTCTrainStepMode()
    match model.cfg.train_mode:
        case train_handler.TrainMode.UNIFIED:
            train_strat = train_handler.TrainUnified(model, train_step_mode, streaming_scale=model.cfg.streaming_scale)
        case train_handler.TrainMode.SWITCHING:
            train_strat = train_handler.TrainSwitching(model, train_step_mode, run_ctx=run_ctx)
        case train_handler.TrainMode.STREAMING:
            train_strat = train_handler.TrainStreaming(model, train_step_mode)
        case train_handler.TrainMode.OFFLINE:
            train_strat = train_handler.TrainOffline(model, train_step_mode)
        case _:
            raise NotImplementedError("Training Strategy not available.")

    loss_dict, num_phonemes = train_strat.step(data)

    for loss_key in loss_dict:
        run_ctx.mark_as_loss(
            name=loss_key,
            loss=loss_dict[loss_key]["loss"],
            inv_norm_factor=num_phonemes,
            scale=loss_dict[loss_key]["scale"]
        )