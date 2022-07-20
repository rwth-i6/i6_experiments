from sisyphus import tk
Path = tk.Path

import numpy as np

import i6_core.returnn as returnn

def prepare_static_prior(config=None, network=None, prob=False):
    assert config or network
    if network is None:
        network = config.config["network"]
    network["accumulate_prior"] = {
        "class": "constant", "dtype": "float32",
        "value": returnn.CodeWrapper("array_priors"),
    }
    if not prob:
        network["combine_prior"].update(
            eval="safe_log(source(0)) * am_scale - source(1) * prior_scale"
        )


class SimpleEqualityArray(np.ndarray):
    """Helper numpy array subclass only overriding equality operator to work with RETURNN."""
    def __eq__(self, other):
        return super().__eq__(other).all()

def make_loader_code(path):
    return returnn.CodeWrapper('np.loadtxt("{}").view({})'.format(
        path, SimpleEqualityArray.__name__
    ))

def add_static_prior(config, prior_txt):
    assert isinstance(config, returnn.ReturnnConfig)
    code = prior_txt.function(make_loader_code) \
        if isinstance(prior_txt, Path) \
            else make_loader_code(prior_txt)
    config.config["array_priors"] = code
    import_code = (
        "import numpy as np",
        SimpleEqualityArray,
    )
    try:
        if import_code not in config.python_prolog:
            config.python_prolog += import_code
    except TypeError:
        config.python_prolog = import_code


def add_static_prior_from_var(config, prior_var, numpy=False):
    assert isinstance(config, returnn.ReturnnConfig)
    # code = prior_txt.function(make_loader_code) \
    #     if isinstance(prior_txt, Path) \
    #         else crnn.CodeWrapper(make_loader_code(prior_txt))
    # print(code)
    import_code = ("import numpy as np",)
    if numpy:
        config["array_priors"] = returnn.CodeWrapper(
            "np.array(" + str(prior_var) + ").view({})".format(SimpleEqualityArray.__name__)
        )
        import_code += (SimpleEqualityArray,)
    else:
        config["array_priors"] = prior_var
    try:
        if import_code not in config.python_prolog:
            config.python_prolog += import_code
    except AttributeError:
        config.python_prolog = import_code