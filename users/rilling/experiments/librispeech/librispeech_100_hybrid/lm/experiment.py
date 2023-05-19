from sisyphus import tk

from i6_core.returnn.training import ReturnnTrainingJob

from .default_tools import RETURNN_EXE, RETURNN_ROOT
from .data import build_training_data
from .config import get_training_config

def train(config, num_epochs=50):
    default_rqmt = {
        'mem_rqmt': 15,
        'time_rqmt': 168,
        'log_verbosity': 5,
        'returnn_python_exe': RETURNN_EXE,
        'returnn_root': RETURNN_ROOT,
    }

    train_job = ReturnnTrainingJob(
        returnn_config=config,
        num_epochs=num_epochs,
        **default_rqmt
    )
    return train_job


def test_train_lm():

    exp_prefix = "experiments/librispeech/librispeech_100_hybrid/lm/kazuki_lstm_2x4k_samp16k_v2"
    training_data = build_training_data(output_prefix=exp_prefix)

    config = get_training_config(training_data)
    config.config["batch_size"] = 950
    train_job = train(config)
    train_job.add_alias(exp_prefix + "/training")
    tk.register_output(exp_prefix + "/learning_rates", train_job.out_learning_rates)

