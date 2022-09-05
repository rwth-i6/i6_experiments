"""
Replicating the pipeline of my 2020 transducer work:
https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer

Note: this file is loaded in different contexts:

- as a sisyphus config file. In this case, the function `py` is called.
- via the generated RETURNN configs. In this case, all the Sisyphus stuff is ignored,
  and only selected functions will run.

The reason for this is that we want to have all the relevant code for the experiment
to be in one place, such that we can make a copy of the file as a base for another separate experiment.

Note on the hash of the model definition:
This is explicit, via the version object below,
and via the module name (__name__; this includes the package name),
and via the model def function name.

Note on the motivation for the interface:
- should be flexible to different tasks (datasets)
- should be simple (obviously!)
- for training and recognition (alignment)
- need to use dynamic get_network because we don't want to run the net code in the root config

"""

from sisyphus import tk
from .task import get_switchboard_task
from .train import train
from .recog import recog
from .align import align
from returnn_common import nn
from returnn_common.nn.encoder.blstm_cnn_specaug import BlstmCnnSpecAugEncoder


# version is used for the hash of the model definition,
# together with the model def function name together with the module name (__name__).
assert __name__.startswith("i6_experiments.")  # just a sanity check
version = 1
extra_hash = (version,)


def sis_config_main():
    """sis config function"""
    task = get_switchboard_task()

    step1_model = train(
        task=task, model_def=from_scratch_model_def, train_def=from_scratch_training, extra_hash=extra_hash)
    step2_alignment = align(task=task, model=step1_model)
    # use step1 model params; different to the paper
    step3_model = train(
        task=task, model_def=extended_model_def, train_def=extended_model_training, extra_hash=extra_hash,
        alignment=step2_alignment, init_params=step1_model.checkpoint)
    step4_model = train(
        task=task, model_def=extended_model_def, train_def=extended_model_training, extra_hash=extra_hash,
        alignment=step2_alignment, init_params=step3_model.checkpoint)

    tk.register_output('step1', recog(task, step1_model).main_measure_value)
    tk.register_output('step3', recog(task, step3_model).main_measure_value)
    tk.register_output('step4', recog(task, step4_model).main_measure_value)


py = sis_config_main  # `py` is the default sis config function name


class Model(nn.Module):
    """Model definition"""

    def __init__(self, *, num_enc_layers=6):
        super(Model, self).__init__()
        self.encoder = BlstmCnnSpecAugEncoder(num_layers=num_enc_layers)
        # TODO pretrain...
        # TODO decoder...


def from_scratch_model_def(*, epoch: int) -> Model:
    """Function is run within RETURNN."""
    pass  # TODO


def from_scratch_training(*,
                          model: Model,
                          data: nn.Data, data_spatial_dim: nn.Dim,
                          targets: nn.Data, targets_spatial_dim: nn.Dim
                          ):
    """Function is run within RETURNN."""
    # TODO...
    # TODO feed through model, define full sum loss, mark_as_loss
    # TODO pretrain epoch dependent...
    return model


def extended_model_def(*, epoch: int) -> Model:
    """Function is run within RETURNN."""
    pass  # TODO


def extended_model_training(*,
                            model: Model,
                            data: nn.Data, data_spatial_dim: nn.Dim,
                            align_targets: nn.Data, align_targets_spatial_dim: nn.Dim
                            ):
    """Function is run within RETURNN."""
    pass  # TODO


def model_recog(*,
                model: Model,
                data: nn.Data, data_spatial_dim: nn.Dim,
                target_vocab: nn.Dim,
                ):
    """Function is run within RETURNN."""
    pass  # TODO
