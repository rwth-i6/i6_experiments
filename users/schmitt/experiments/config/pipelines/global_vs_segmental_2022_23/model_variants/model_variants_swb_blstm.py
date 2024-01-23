from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.swb import SWBCorpus
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.label_singletons import SWBBPE1030_LABELS, SWB_BPE_1030_RNA_ALIGNMENT

SWB_CORPUS = SWBCorpus()

models = {
  "swb_blstm_glob_att": {
    "glob.blstm.best": {
      "dependencies": SWBBPE1030_LABELS,
      "dataset": {
        "feature_type": "gammatone",
        "corpus": SWB_CORPUS
      },
      "network": {},
      "config": {},
      "returnn_python_exe": RETURNN_EXE,
      "returnn_root": RETURNN_ROOT
    },
  },
  "swb_blstm_seg_att": {
    "seg.blstm.best": {
      "dependencies": SWB_BPE_1030_RNA_ALIGNMENT,
      "dataset": {
        "feature_type": "gammatone",
        "corpus": SWB_CORPUS
      },
      "network": {
        "segment_center_window_size": None
      },
      "config": {},
      "returnn_python_exe": RETURNN_EXE,
      "returnn_root": RETURNN_ROOT
    },
  },
}
