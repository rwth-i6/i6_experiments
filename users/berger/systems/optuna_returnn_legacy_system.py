from i6_core import returnn
from i6_experiments.users.berger.systems import functors
from i6_experiments.users.berger.recipe import returnn as returnn_custom

from .base_system import BaseSystem

from sisyphus import tk

assert __package__ is not None
Path = tk.setup_path(__package__)


class OptunaReturnnLegacySystem(
    BaseSystem[returnn_custom.OptunaReturnnTrainingJob, returnn_custom.OptunaReturnnConfig]
):
    def _initialize_functors(
        self,
    ) -> functors.Functors[returnn_custom.OptunaReturnnTrainingJob, returnn_custom.OptunaReturnnConfig]:
        assert self._tool_paths.returnn_root is not None
        assert self._tool_paths.returnn_python_exe is not None
        assert self._tool_paths.rasr_binary_path is not None
        assert self._tool_paths.rasr_python_exe is not None

        train_functor = functors.OptunaReturnnTrainFunctor(
            self._tool_paths.returnn_root, self._tool_paths.returnn_python_exe
        )

        recog_functor = functors.OptunaAdvancedTreeSearchFunctor(
            returnn_root=self._tool_paths.returnn_root,
            returnn_python_exe=self._tool_paths.returnn_python_exe,
            rasr_binary_path=self._tool_paths.rasr_binary_path,
            rasr_python_exe=self._tool_paths.rasr_python_exe,
            blas_lib=self._tool_paths.blas_lib,
        )

        align_functor = functors.OptunaLegacyAlignmentFunctor(
            returnn_root=self._tool_paths.returnn_root,
            returnn_python_exe=self._tool_paths.returnn_python_exe,
            rasr_binary_path=self._tool_paths.rasr_binary_path,
            rasr_python_exe=self._tool_paths.rasr_python_exe,
            blas_lib=self._tool_paths.blas_lib,
        )

        return functors.Functors(train_functor, recog_functor, align_functor)
