from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_ROOT

default_variant_segmental_attention = dict(
  config=dict(),
  network=dict(),
  returnn_python_exe=RETURNN_EXE,
  returnn_root=RETURNN_ROOT
)

default_variant_global_attention = dict(
  config=dict(),
  network=dict(),
  returnn_python_exe=RETURNN_EXE,
  returnn_root=RETURNN_ROOT
)

models = {
  "global_conformer": {
    "glob.conformer-mohammad-best": {
      "label_type": "bpe",
      **default_variant_global_attention
    },
  }
}

global_model_variants = {}
for model_type in models:
  if model_type in ["global_conformer"]:
    global_model_variants[model_type] = models[model_type]

segmental_model_variants = {}
for model_type in models:
  if model_type in ["segmental"]:
    segmental_model_variants[model_type] = models[model_type]
