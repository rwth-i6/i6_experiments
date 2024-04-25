from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_CURRENT_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.swb import SWBCorpus
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.label_singletons import SWBBPE534_LABELS, SWB_BPE_534_CTC_ALIGNMENT

SWB_CORPUS = SWBCorpus()

models = {
  "swb_conformer_glob_att": {
    "glob.conformer.best": {
      "dependencies": SWBBPE534_LABELS,
      "dataset": {
        "feature_type": "raw",
        "corpus": SWB_CORPUS
      },
      "network": {},
      "config": {
        "train_seq_ordering": "laplace:6000"
      },
      "returnn_python_exe": RETURNN_EXE,
      "returnn_root": RETURNN_CURRENT_ROOT
    },
  },
  "swb_conformer_seg_att": {
    "seg.conformer.best": {
      "dependencies": SWB_BPE_534_CTC_ALIGNMENT,
      "dataset": {
        "feature_type": "raw",
        "corpus": SWB_CORPUS
      },
      "network": {
        "segment_center_window_size": None
      },
      "config": {
        "train_seq_ordering": "laplace:.1000"
      },
      "returnn_python_exe": RETURNN_EXE,
      "returnn_root": RETURNN_CURRENT_ROOT
    },
  },
}
