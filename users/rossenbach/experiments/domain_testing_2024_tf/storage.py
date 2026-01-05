from sisyphus import tk

from dataclasses import dataclass

@dataclass
class LmModel:
    network: dict[str, any]
    model: tk.Path


# Neural LM Models -------------------------------------------------------------------------------------------------------

_lm_models: dict[str, tk.Path] = {}

def add_lm(name: str, lm_model: LmModel):
    global _lm_models
    assert name not in _lm_models.keys()
    _lm_models[name] = lm_model


def get_lm_model(name: str) -> LmModel:
    global _lm_models
    return _lm_models[name]

def get_lm_model_as_opts(name: str) -> dict[str, any]:
    global _lm_models
    lm_model = _lm_models[name]
    return {
        "lm_subnet": lm_model.network,
        "lm_model": lm_model.model,
        "name": name,
    }