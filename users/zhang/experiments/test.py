from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob, GnuPlotJob
from sisyphus import tk
from i6_experiments.users.zhang.experiments.exp_wer_ppl import EVAL_DATASET_KEYS

def py():
    summary_path = "work/i6_experiments/users/zhang/experiments/WER_PPL/util/WER_ppl_PlotAndSummaryJob.KGt43qy5JGZO/output/summary.csv"
    gnuplotjob = GnuPlotJob(tk.Path(summary_path), EVAL_DATASET_KEYS)
    for i, key in enumerate(EVAL_DATASET_KEYS):
        tk.register_output(f"test/gnuplot/{key}.pdf", gnuplotjob.out_plots[key])