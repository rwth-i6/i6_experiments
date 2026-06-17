from recipe.baseline_rep.experiments.librispeech import run_baselines, run_ctc_bpe_experimental
from sisyphus import tk

def py():
    baseline_models, baseline_results = run_baselines(filename="/u/azim.javed/experiments/training/baselines/baselines_report.txt")
    # ctc_bpe_experimental_models, ctc_bpe_experimental_results = run_ctc_bpe_experimental(filename="/u/azim.javed/experiments/training/baselines/ctc_bpe_experimental_report.txt")
    # print(models)
    # print(results)
