from sisyphus import tk

from i6_experiments.users.berger.recipe import returnn
from i6_experiments.users.berger.systems import functors

from .base_system import BaseSystem

assert __package__ is not None
Path = tk.setup_path(__package__)


class OptunaReturnnNativeSystem(BaseSystem[returnn.OptunaReturnnTrainingJob, returnn.OptunaReturnnConfig]):
    def _initialize_functors(
        self,
    ) -> functors.Functors[returnn.OptunaReturnnTrainingJob, returnn.OptunaReturnnConfig]:
        assert self._tool_paths.returnn_root is not None
        assert self._tool_paths.returnn_python_exe is not None
        assert self._tool_paths.rasr_binary_path is not None
        train_functor = functors.OptunaReturnnTrainFunctor(
            self._tool_paths.returnn_root, self._tool_paths.returnn_python_exe
        )

        recog_functor = functors.OptunaReturnnSearchFunctor(
            self._tool_paths.returnn_root,
            self._tool_paths.returnn_python_exe,
            self._tool_paths.rasr_binary_path,
        )

        align_functor = functors.OptunaReturnnAlignmentFunctor()

        return functors.Functors(train_functor, recog_functor, align_functor)
