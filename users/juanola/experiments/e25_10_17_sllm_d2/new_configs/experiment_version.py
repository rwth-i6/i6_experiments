from enum import Enum

from .experiment_config import get_exp_config_v1, \
    get_exp_config_v2


class ExperimentVersion(Enum):
    V1 = "v1"
    V2 = "v2"
    # And so on

    def get_experiment_config(self):
        """
        Get experiment config of experiment version.
        """
        if self == ExperimentVersion.V1:
            return get_exp_config_v1()
        if self == ExperimentVersion.V2:
            return get_exp_config_v2()
        else:
            raise NotImplementedError("Experiment version not implemented")
