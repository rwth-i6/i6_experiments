from returnn.tensor.tensor_dict import TensorDict

from ...pytorch_modules import LstmLmScorerModel, LstmLmStateInitializerModel, LstmLmStateUpdaterModel


def scorer_forward_step(*, model: LstmLmScorerModel, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    lstm_out = extern_data["lstm_out"].raw_tensor
    assert lstm_out is not None

    scores = model.forward(lstm_out=lstm_out)

    run_ctx.mark_as_output(name="scores", tensor=scores)


def state_initializer_forward_step(*, model: LstmLmStateInitializerModel, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    lstm_out, lstm_h, lstm_c = model.forward()

    run_ctx.mark_as_output(name="lstm_out", tensor=lstm_out)
    run_ctx.mark_as_output(name="lstm_h", tensor=lstm_h)
    run_ctx.mark_as_output(name="lstm_c", tensor=lstm_c)


def state_updater_forward_step(*, model: LstmLmStateUpdaterModel, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    token = extern_data["token"].raw_tensor
    assert token is not None
    lstm_c_in = extern_data["lstm_c_in"].raw_tensor
    assert lstm_c_in is not None
    lstm_h_in = extern_data["lstm_h_in"].raw_tensor
    assert lstm_h_in is not None

    lstm_out, lstm_h_out, lstm_c_out = model.forward(token=token, lstm_h=lstm_h_in, lstm_c=lstm_c_in)

    run_ctx.mark_as_output(name="lstm_out", tensor=lstm_out)
    run_ctx.mark_as_output(name="lstm_h_out", tensor=lstm_h_out)
    run_ctx.mark_as_output(name="lstm_c_out", tensor=lstm_c_out)
