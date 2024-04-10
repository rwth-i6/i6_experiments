__all__ = ["NnSystem", "returnn_training"]

import copy
from dataclasses import asdict
from typing import Dict, List, Optional, Union

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

# -------------------- Recipes --------------------

import i6_core.returnn as returnn


from .rasr_system import RasrSystem

from .util import ReturnnTrainingJobArgs, AllowedReturnnTrainingDataInput

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------


class NnSystem(RasrSystem):
    """
    Neural Network Common System Class
    """

    def __init__(
        self,
        rasr_binary_path: tk.Path,
        rasr_arch: str = "linux-x86_64-standard",
        returnn_root: Optional[tk.Path] = None,
        returnn_python_home: Optional[tk.Path] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        blas_lib: Optional[tk.Path] = None,
    ):
        super().__init__(rasr_binary_path=rasr_binary_path, rasr_arch=rasr_arch)

        self.returnn_root = returnn_root or (gs.RETURNN_ROOT if hasattr(gs, "RETURNN_ROOT") else None)
        self.returnn_python_home = returnn_python_home or (
            gs.RETURNN_PYTHON_HOME if hasattr(gs, "RETURNN_PYTHON_HOME") else None
        )
        self.returnn_python_exe = returnn_python_exe or (
            gs.RETURNN_PYTHON_EXE if hasattr(gs, "RETURNN_PYTHON_EXE") else None
        )

        self.blas_lib = blas_lib or (gs.BLAS_LIB if hasattr(gs, "BLAS_LIB") else None)

        self.native_ops = {}  # type: Dict[str, tk.Path]

    def compile_native_op(self, op_name: str):
        """
        Compiles and stores a native op for RETURNN

        :param op_name: op name, e.g. "NativeLstm2"
        """
        native_op_job = returnn.CompileNativeOpJob(
            op_name,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            blas_lib=self.blas_lib,
        )
        native_op_job.add_alias("wei_native_ops/compile_native_%s" % op_name)
        self.native_ops[op_name] = native_op_job.out_op

    def get_native_ops(self, op_names: Optional[List[str]]) -> Optional[List[tk.Path]]:
        """
        Access self.native_ops and compile if not existing

        :param op_names: list of RETURNN op names, can be None for convenience
        :return list of native op paths
        """
        if op_names is None:
            return None
        for op_name in op_names:
            if op_name not in self.native_ops.keys():
                self.compile_native_op(op_name)
        return [self.native_ops[op_name] for op_name in op_names]


def returnn_training(
    name: str,
    returnn_config: returnn.ReturnnConfig,
    training_args: Union[Dict, ReturnnTrainingJobArgs],
    train_data: AllowedReturnnTrainingDataInput,
    *,
    cv_data: Optional[AllowedReturnnTrainingDataInput] = None,
    additional_data: Optional[Dict[str, AllowedReturnnTrainingDataInput]] = None,
    register_output: bool = True,
) -> returnn.ReturnnTrainingJob:
    assert isinstance(returnn_config, returnn.ReturnnConfig)

    config = copy.deepcopy(returnn_config)

    config.config["train"] = train_data if isinstance(train_data, Dict) else train_data.get_data_dict()
    if "split_10" in name:
        config.config["train"]["datasets"]["align"]["partition_epoch"] = 10
    if cv_data is not None:
        config.config["dev"] = cv_data if isinstance(cv_data, Dict) else cv_data.get_data_dict()
    if additional_data is not None:
        config.config["eval_datasets"] = {}
        for data_name, data in additional_data.items():
            config.config["eval_datasets"][data_name] = data if isinstance(data, Dict) else data.get_data_dict()
    returnn_training_job = returnn.ReturnnTrainingJob(
        returnn_config=config,
        **asdict(training_args) if isinstance(training_args, ReturnnTrainingJobArgs) else training_args,
    )
    #if any(sub in name for sub in ["larger", "whisper_medium", "whisper_large"]):
    #print(name, any(f"keepuntil_{x}_" in name for x in range(9)))
    if any(sub in name for sub in ["larger", "medium", "large", "parakeet"]) and not any(f"keepuntil_{x}_" in name for x in range(9)):
        returnn_training_job.rqmt["gpu_mem"] = 24
    if any(sub in name for sub in ["whisper_large", "whisper_v2_large", "parakeet_1.1"]):
        returnn_training_job.rqmt["mem"] = 12
    if register_output:
        returnn_training_job.add_alias(f"nn_train/{name}")
        tk.register_output(f"nn_train/{name}_learning_rates.png", returnn_training_job.out_plot_lr)

    return returnn_training_job
