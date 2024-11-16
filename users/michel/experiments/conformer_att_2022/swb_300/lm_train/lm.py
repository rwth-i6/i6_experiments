from sisyphus import tk

from dataclasses import dataclass
from typing import Any, Dict

from i6_core.returnn.training import ReturnnTrainingJob

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.swb_300.default_tools import (
    RETURNN_EXE,
    RETURNN_ROOT,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.swb_300.lm_train.lm_data import build_training_data
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.swb_300.lm_train.lm_config import (
    get_training_config,
)


def train(config, num_epochs=20):
    default_rqmt = {
        "mem_rqmt": 15,
        "time_rqmt": 168,
        "log_verbosity": 5,
        "returnn_python_exe": RETURNN_EXE,
        "returnn_root": RETURNN_ROOT,
    }

    train_job = ReturnnTrainingJob(returnn_config=config, num_epochs=num_epochs, **default_rqmt)
    return train_job


@dataclass()
class ZeineldeenLM:
    """
    Contains BPE-LMs compatible with the Zeineldeen-setup
    """

    combination_network: Dict[str, Any]
    train_job: ReturnnTrainingJob


_lm_stash = {}
_generated_lms = False

EXP_PREFIX = "lm_training/swb300/kazuki_lm/"


def get_lm(name: str) -> ZeineldeenLM:
    global _lm_stash
    global _generated_lms
    if _generated_lms is False:
        test_train_lm()
    _lm_stash[name].train_job.add_alias(EXP_PREFIX + name + "/training")
    tk.register_output(EXP_PREFIX + name + "/learning_rates", _lm_stash[name].train_job.out_learning_rates)
    return _lm_stash[name]


def test_train_lm():
    global _lm_stash

    name = "swb300_trafo6_bs1350_500bpe"
    training_data = build_training_data(output_prefix=EXP_PREFIX + name, partition_epoch=4)
    config, ext_lm_net = get_training_config(training_data)
    train_job = train(config, num_epochs=50)
    _lm_stash[name] = ZeineldeenLM(combination_network=ext_lm_net, train_job=train_job)

    _generated_lms = True


def train_all_lms():
    global _lm_stash
    test_train_lm()
    for name in _lm_stash.keys():
        get_lm(name)
