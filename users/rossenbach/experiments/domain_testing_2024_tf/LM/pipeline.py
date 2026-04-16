
from i6_core.returnn.training import ReturnnTrainingJob

from ..default_tools import RETURNN_EXE, RETURNN_ROOT

def train(prefix, config, num_epochs=20):
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
    train_job.add_alias(prefix + "/training")
    return train_job