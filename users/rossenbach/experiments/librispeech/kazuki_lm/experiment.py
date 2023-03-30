from sisyphus import tk

from dataclasses import dataclass
from typing import Any, Dict

from i6_core.returnn.training import ReturnnTrainingJob

from .default_tools import RETURNN_EXE, RETURNN_ROOT
from .data import build_training_data
from .config import get_training_config

def train(config, num_epochs=20):
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


@dataclass()
class ZeineldeenLM:
    """
    Contains BPE-LMs compatible with the Zeineldeen-setup
    """
    combination_network: Dict[str, Any]
    train_job: ReturnnTrainingJob


_lm_stash = {}
_generated_lms = False


EXP_PREFIX = "experiments/librispeech/kazuki_lm/"


def get_lm(name: str) -> ZeineldeenLM:
    global _lm_stash
    global _generated_lms
    if _generated_lms is False:
        test_train_lm()
    _lm_stash[name].train_job.add_alias(EXP_PREFIX + name + "/training")
    tk.register_output(EXP_PREFIX + name +  "/learning_rates", _lm_stash[name].train_job.out_learning_rates)
    return _lm_stash[name]



def test_train_lm():
    global _lm_stash

    name = "ls100_trafo24_bs3000_5ep_2kbpe"
    training_data = build_training_data(output_prefix=EXP_PREFIX+name, partition_epoch=20)
    
    config, ext_lm_net = get_training_config(training_data)
    config.config["batch_size"] = 3000
    config.config["max_seqs"] = 96
    train_job = train(config, num_epochs=100)
    _lm_stash[name] = ZeineldeenLM(combination_network=ext_lm_net, train_job=train_job)



    name = "ls960_trafo24_bs3000_5ep_10kbpe"
    training_data = build_training_data(corpus_key="train-other-960", bpe_size=10000, output_prefix=name, partition_epoch=20)

    config, ext_lm_net = get_training_config(training_data)
    config.config["batch_size"] = 3000
    config.config["max_seqs"] = 96
    train_job = train(config, num_epochs=100)
    _lm_stash[name]  = ZeineldeenLM(combination_network=ext_lm_net, train_job=train_job)


    name = "ls960_trafo24_bs3000_5ep_5kbpe"
    training_data = build_training_data(corpus_key="train-other-960", bpe_size=5000, output_prefix=name, partition_epoch=20)

    config, ext_lm_net = get_training_config(training_data)
    config.config["batch_size"] = 3000
    config.config["max_seqs"] = 96
    train_job = train(config, num_epochs=100)
    _lm_stash[name]  = ZeineldeenLM(combination_network=ext_lm_net, train_job=train_job)



    name = "ls960_trafo24_bs3000_5ep_1kbpe"
    training_data = build_training_data(corpus_key="train-other-960", bpe_size=1000, output_prefix=name, partition_epoch=20)

    config, ext_lm_net = get_training_config(training_data)
    config.config["batch_size"] = 3000
    config.config["max_seqs"] = 96
    train_job = train(config, num_epochs=100)
    _lm_stash[name]  = ZeineldeenLM(combination_network=ext_lm_net, train_job=train_job)


    _generated_lms = True


def train_all_lms():
    global _lm_stash
    test_train_lm()
    for name in _lm_stash.keys():
        get_lm(name)
