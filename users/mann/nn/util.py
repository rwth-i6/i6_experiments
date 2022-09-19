from i6_core.returnn import CodeWrapper, ReturnnConfig
from sisyphus.delayed_ops import DelayedFormat

class DelayedCodeWrapper(DelayedFormat):
    def get(self):
        return CodeWrapper(super().get())
    
def maybe_add_dependencies(returnn_config: ReturnnConfig, *dependencies):
    if len(dependencies) > 1:
        for dep in dependencies:
            maybe_add_dependencies(returnn_config, dep)
        return
    # single dependency case
    dep_code = dependencies[0]
    if returnn_config.python_prolog is None:
        returnn_config.python_prolog = (dep_code,)
    elif not dep_code in returnn_config.python_prolog:
        returnn_config.python_prolog += (dep_code,)
    # already in prolog