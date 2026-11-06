from recipe.experiments.librispeech import run_adapter


def py():
    models, results = run_adapter(filename="reports/score_adapter.txt")
