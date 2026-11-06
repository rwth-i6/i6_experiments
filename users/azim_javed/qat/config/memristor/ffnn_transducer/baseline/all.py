from recipe.experiments.librispeech import run_all
from recipe.experiments.librispeech.groups import report_path, selection


def py():
    return run_all(
        report_path("memristor", "ffnn_transducer", "baseline", None),
        selection("ffnn_transducer", "baseline", None),
    )
