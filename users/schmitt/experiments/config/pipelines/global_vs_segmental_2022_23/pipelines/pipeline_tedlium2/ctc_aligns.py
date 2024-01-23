from i6_experiments.users.schmitt.alignment.alignment import AlignmentStatisticsJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.tedlium2.label_singletons import TEDLIUM2BPE1058_CTC_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_ROOT, RETURNN_EXE_NEW

from sisyphus import tk

def calc_align_stats():
  statistics_job = AlignmentStatisticsJob(
    alignment=TEDLIUM2BPE1058_CTC_ALIGNMENT.alignment_paths["train"],
    blank_idx=TEDLIUM2BPE1058_CTC_ALIGNMENT.model_hyperparameters.blank_idx,
    silence_idx=20000,  # dummy idx which is larger than the vocab size
    returnn_root=RETURNN_ROOT, returnn_python_exe=RETURNN_EXE_NEW)
  statistics_job.add_alias("datasets/TedLium2/statistics/train")
  tk.register_output(statistics_job.get_one_alias(), statistics_job.out_statistics)
