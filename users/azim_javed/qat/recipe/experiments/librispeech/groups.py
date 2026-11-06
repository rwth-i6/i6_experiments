from collections.abc import Mapping
from pathlib import Path


class LazyModels(Mapping):
    def __init__(self, factories):
        self.factories = factories
        self.built = {}

    def __getitem__(self, key):
        if key not in self.built:
            self.built[key] = self.factories[key]()
        return self.built[key]

    def __iter__(self):
        return iter(self.factories)

    def __len__(self):
        return len(self.factories)


def model_group(key):
    target = key.split("___", 1)[0]
    if "full_ctx_transducer" in target:
        family = "full_ctx_transducer"
    elif "ffnn_transducer" in target:
        family = "ffnn_transducer"
    elif "ctc" in target:
        family = "ctc"
    else:
        raise ValueError(key)

    if "qat" not in target:
        return family, "baseline", None
    if "qat_encoder_prediction" in target or "qat_encoder_pred" in target:
        level = "qat_encoder_pred"
    elif "qat_encoder" in target:
        level = "qat_encoder"
    else:
        level = "qat_full_model"
    return family, level, 4 if "w4_a8" in target else 8


def run_digital(family, level, bits):
    from . import run_all

    return run_all(report_path("digital", family, level, bits), selection(family, level, bits))


async def run_memristor(family, level, bits):
    from . import run_mem

    return await run_mem(report_path("memristor", family, level, bits), selection(family, level, bits))


def selection(family, level, bits):
    return lambda key: model_group(key) == (family, level, bits)


def report_path(mode, family, level, bits):
    directory = Path(__file__).resolve().parents[3] / "reports" / mode / family / level
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / ("baseline.txt" if bits is None else f"{bits}bit.txt"))
