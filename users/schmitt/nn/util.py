from i6_core.returnn import CodeWrapper, ReturnnConfig
from sisyphus.delayed_ops import DelayedFormat
from sisyphus.tools import try_get

from i6_core import util

class CustomDelayedFormat(DelayedFormat):
    def get(self):
        # args = (try_get(i) for i in self.args)
        # kwargs = {k: try_get(v) for k, v in self.kwargs.items()}
        args = util.instanciate_delayed(self.args)
        kwargs = util.instanciate_delayed(self.kwargs)
        return try_get(self.string).format(*args, **kwargs)

class DelayedCodeWrapper(CustomDelayedFormat):
    def get(self):
        return CodeWrapper(super().get())
    
def maybe_add_dependencies(returnn_config: ReturnnConfig, *dependencies):
    if len(dependencies) > 1:
        for dep in dependencies:
            maybe_add_dependencies(returnn_config, dep)
        return
    # single dependency case
    dep_code = dependencies[0]
    if dep_code is None:
        return
    if returnn_config.python_prolog is None:
        returnn_config.python_prolog = (dep_code,)
    elif not dep_code in returnn_config.python_prolog:
        returnn_config.python_prolog += (dep_code,)
    # already in prolog