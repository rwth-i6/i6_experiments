import copy

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_ROOT, RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_LABELS, LibrispeechBPE10025_CTC_ALIGNMENT

LIBRISPEECH_CORPUS = LibrispeechCorpora()

models = {
  "librispeech_conformer_glob_att": {
    "glob.conformer.mohammad.5.6": {
      "dependencies": LibrispeechBPE10025_LABELS,
      "dataset": {
        "feature_type": "raw",
        "corpus": LIBRISPEECH_CORPUS
      },
      "network": {},
      "config": {
        "train_seq_ordering": "laplace:.1000"
      },
      "returnn_python_exe": RETURNN_EXE_NEW,
      "returnn_root": RETURNN_CURRENT_ROOT
    },
  }, "librispeech_conformer_seg_att": {
    "seg.conformer.like-global": {
      "dependencies": LibrispeechBPE10025_CTC_ALIGNMENT,
      "dataset": {
        "feature_type": "raw",
        "corpus": LIBRISPEECH_CORPUS
      },
      "network": {
        "segment_center_window_size": None,
        "length_model_opts": {
          "use_embedding": True,
          "embedding_size": 128,
          "use_alignment_ctx": True,
          "layer_class": "lstm"
        }
      },
      "config": {
        "train_seq_ordering": "laplace:.1000"
      },
      "returnn_python_exe": RETURNN_EXE_NEW,
      "returnn_root": RETURNN_CURRENT_ROOT
    },
  }, "librispeech_conformer_center-window_att": {}}

# for center_win_size in (0.5, 1, 2, 4, 8, 16, 32, 64):
#   variant_name = "center-window-%d.conformer" % (center_win_size * 2)
#   models["librispeech_conformer_center-window_att"][variant_name] = copy.deepcopy(models["librispeech_conformer_seg_att"]["seg.conformer.best"])
#   models["librispeech_conformer_center-window_att"][variant_name]["network"]["segment_center_window_size"] = center_win_size
