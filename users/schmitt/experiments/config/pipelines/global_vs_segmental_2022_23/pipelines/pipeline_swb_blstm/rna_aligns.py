from sisyphus import tk

from i6_experiments.users.schmitt.alignment.alignment import AlignmentStatisticsJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.label_singletons import SWB_BPE_1030_RNA_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_ROOT, RETURNN_EXE_NEW


def calc_align_stats():
  for corpus_key in ("train",):
    statistics_job = AlignmentStatisticsJob(
      alignment=SWB_BPE_1030_RNA_ALIGNMENT.alignment_paths[corpus_key],
      json_vocab=SWB_BPE_1030_RNA_ALIGNMENT.vocab_path,
      blank_idx=SWB_BPE_1030_RNA_ALIGNMENT.model_hyperparameters.blank_idx,
      silence_idx=20000,  # dummy idx which is larger than the vocab size
      returnn_root=RETURNN_ROOT, returnn_python_exe=RETURNN_EXE_NEW)
    statistics_job.add_alias("datasets/swb/statistics/%s" % corpus_key)
    tk.register_output(statistics_job.get_one_alias(), statistics_job.out_statistics)
