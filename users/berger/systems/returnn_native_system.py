from i6_core import returnn
from i6_experiments.users.berger.systems import functors

from .base_system import BaseSystem

from sisyphus import tk

Path = tk.setup_path(__package__)


class ReturnnNativeSystem(BaseSystem[returnn.ReturnnTrainingJob, returnn.ReturnnConfig]):
    def _initialize_functors(
        self,
    ) -> functors.Functors[returnn.ReturnnTrainingJob, returnn.ReturnnConfig]:
        assert self._tool_paths.returnn_root is not None
        assert self._tool_paths.returnn_python_exe is not None
        train_functor = functors.ReturnnTrainFunctor(self._tool_paths.returnn_root, self._tool_paths.returnn_python_exe)

        recog_functor = functors.ReturnnBpeSearchFunctor(
            self._tool_paths.returnn_root,
            self._tool_paths.returnn_python_exe,
        )

        align_functor = functors.ReturnnAlignmentFunctor()

        return functors.Functors(train_functor, recog_functor, align_functor)
