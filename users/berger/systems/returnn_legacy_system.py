from i6_core import returnn
from i6_experiments.users.berger.systems import functors
from i6_experiments.users.berger.recipe import returnn as returnn_custom

from .base_system import BaseSystem

from sisyphus import tk

Path = tk.setup_path(__package__)


class ReturnnLegacySystem(BaseSystem[returnn.ReturnnTrainingJob, returnn.ReturnnConfig]):
    def _initialize_functors(
        self,
    ) -> functors.Functors[returnn.ReturnnTrainingJob, returnn.ReturnnConfig]:
        train_functor = functors.ReturnnTrainFunctor(self._tool_paths.returnn_root, self._tool_paths.returnn_python_exe)

        recog_functor = functors.AdvancedTreeSearchFunctor(
            self._tool_paths.returnn_root,
            self._tool_paths.returnn_python_exe,
            self._tool_paths.blas_lib,
        )

        align_functor = functors.LegacyAlignmentFunctor(
            self._tool_paths.returnn_root,
            self._tool_paths.returnn_python_exe,
            self._tool_paths.blas_lib,
        )

        return functors.Functors(train_functor, recog_functor, align_functor)
