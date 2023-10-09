import copy

from sisyphus import tk

from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.returnn.training import ReturnnTrainingJob

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_CURRENT_ROOT, RETURNN_EXE, RETURNN_EXE_NEW, RETURNN_ROOT
# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder import LibrispeechConformerGlobalAttentionConfigBuilder, LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import LibrispeechConformerGlobalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints

from i6_experiments.users.schmitt.alignment.alignment import AlignmentStatisticsJob

directory_name = "models/ls_conformer/ctc_aligns"


class GlobalAttCtcAlignment:
  def __init__(self):
    self._ctc_alignments = {}

  @property
  def ctc_alignments(self):
    if self._ctc_alignments == {}:
      raise ValueError("You first need to run get_global_attention_ctc_align()!")
    else:
      return self._ctc_alignments

  @ctc_alignments.setter
  def ctc_alignments(self, value):
    assert type(value) == dict
    assert "train" in value and "cv" in value and "devtrain" in value
    self._ctc_alignments = value


global_att_ctc_align = GlobalAttCtcAlignment()


def get_global_attention_ctc_align():
  """

  :return:
  """
  model_type = "librispeech_conformer_glob_att"
  variant_name = "glob.conformer.mohammad.5.6"
  variant_params = copy.deepcopy(models[model_type][variant_name])
  base_alias = "%s/%s/%s" % (directory_name, model_type, variant_name)

  ctc_aligns_global_att = {}

  config_builder = LibrispeechConformerGlobalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  for corpus_key in ("cv", "train", "dev-other"):
    eval_config = config_builder.get_ctc_align_config(
      corpus_key=corpus_key,
      opts={
        "align_target": "data:targets",
        "hdf_filename": "alignments.hdf",
        "dataset_opts": {"seq_postfix": None}
      }
    )

    forward_job = ReturnnForwardJob(
      model_checkpoint=external_checkpoints["glob.conformer.mohammad.5.6"],
      returnn_config=eval_config,
      returnn_root=RETURNN_CURRENT_ROOT,
      returnn_python_exe=RETURNN_EXE_NEW if corpus_key == "dev-other" else RETURNN_EXE,
      hdf_outputs=["alignments.hdf"],
      eval_mode=True
    )
    forward_job.add_alias("%s/dump_ctc_%s" % (base_alias, corpus_key))
    tk.register_output(forward_job.get_one_alias(), forward_job.out_hdf_files["alignments.hdf"])

    ctc_aligns_global_att[corpus_key] = forward_job.out_hdf_files["alignments.hdf"]

    if corpus_key in ["cv", "train"]:
      statistics_job = AlignmentStatisticsJob(
        alignment=ctc_aligns_global_att[corpus_key],
        blank_idx=10025,
        silence_idx=20000,  # dummy idx which is larger than the vocab size
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE_NEW
      )
      statistics_job.add_alias("datasets/LibriSpeech/statistics/%s" % corpus_key)
      tk.register_output(statistics_job.get_one_alias(), statistics_job.out_statistics)

  ctc_aligns_global_att["devtrain"] = ctc_aligns_global_att["train"]

  global_att_ctc_align.ctc_alignments = ctc_aligns_global_att


def get_seg_attention_ctc_align():
  model_type = "librispeech_conformer_seg_att"
  variant_name = "seg.conformer.like-global"
  variant_params = copy.deepcopy(models[model_type][variant_name])
  base_alias = "%s/%s/%s" % (directory_name, model_type, variant_name)

  ctc_aligns_seg_att = {}

  config_builder = LibrispeechConformerSegmentalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  num_epochs = 1

  train_job = ReturnnTrainingJob(
    config_builder.get_train_config(
      opts={
        "cleanup_old_models": {"keep_best_n": 0, "keep_last_n": 1},
        "lr_opts": {
          "type": "const",
          "const_lr": 1e-4,
        },
        "import_model_train_epoch1": external_checkpoints["glob.conformer.mohammad.5.6"],
        "dataset_opts": {
          "hdf_targets": get_global_attention_ctc_align()
        }
      }),
    num_epochs=num_epochs,
    keep_epochs=[num_epochs],
    log_verbosity=5,
    returnn_python_exe=variant_params["returnn_python_exe"],
    returnn_root=variant_params["returnn_root"],
    mem_rqmt=24,
    time_rqmt=12)
  train_job.add_alias(base_alias + "/import-from-global-train-%d-epochs" % num_epochs)
  alias = train_job.get_one_alias()
  tk.register_output(alias + "/models", train_job.out_model_dir)
  tk.register_output(alias + "/plot_lr", train_job.out_plot_lr)

  train_1_epoch_after_loading_global_checkpoint = train_job.out_checkpoints[1]

  for corpus_key in ("cv", "train"):
    config_builder = LibrispeechConformerSegmentalAttentionConfigBuilder(
      dependencies=variant_params["dependencies"],
      variant_params=variant_params,
    )
    eval_config = config_builder.get_eval_config(
      eval_corpus_key=corpus_key,
      opts={
        "align_target": "data:label_ground_truth",
        "hdf_filename": "alignments.hdf",
        "use_train_net": True,
        "dataset_opts": {
          "hdf_targets": get_global_attention_ctc_align()
        }
      }
    )

    forward_job = ReturnnForwardJob(
      model_checkpoint=train_1_epoch_after_loading_global_checkpoint,
      returnn_config=eval_config,
      returnn_root=variant_params["returnn_root"],
      returnn_python_exe=variant_params["returnn_python_exe"],
      hdf_outputs=["alignments.hdf"],
      eval_mode=True
    )
    forward_job.add_alias("%s/dump_ctc_%s" % (base_alias, corpus_key))

    ctc_aligns_seg_att[corpus_key] = forward_job.out_hdf_files["alignments.hdf"]

  ctc_aligns_seg_att["devtrain"] = ctc_aligns_seg_att["train"]

  return ctc_aligns_seg_att
