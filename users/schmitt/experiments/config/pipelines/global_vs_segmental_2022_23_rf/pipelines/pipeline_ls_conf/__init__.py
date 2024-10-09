from sisyphus import Path, tk

from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.gmm_alignments import LIBRISPEECH_GMM_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.gmm_alignments import LIBRISPEECH_GMM_ALIGNMENT_CONVERTED, LIBRISPEECH_GMM_WORD_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LIBRISPEECH_CORPUS
from i6_experiments.users.schmitt.alignment.alignment import ConvertGmmAlignmentJob, GmmAlignmentToWordBoundariesJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v1 as center_window_baseline_v1
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v3 as center_window_baseline_v3
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v4 as center_window_baseline_v4
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v5 as center_window_baseline_v5
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v5_small as center_window_baseline_v5_small
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v6 as center_window_baseline_v6
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v7 as center_window_baseline_v7
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import data as center_window_att_data
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import plot_gradient_wrt_enc11, plot_diff_models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.alias import alias as center_window_base_alias

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import baseline_v1 as global_att_baseline_v1
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import baseline_v2 as global_att_baseline_v2

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.ctc import baseline_v1 as ctc_baseline_v1

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att import baseline_v1 as global_att_baseline_v1_no_rf

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf import trafo_lm


def run_exps():
  # run these two first since they set alignment which are later needed by the other experiments
  global_att_baseline_v1_no_rf.register_ctc_alignments()
  ctc_baseline_v1.bpe1k.register_ctc_alignments()
  ctc_baseline_v1.run_exps()
  setup_gmm_alignment()

  trafo_lm.run_exps()

  global_att_baseline_v1.run_exps()
  global_att_baseline_v2.run_exps()

  # center_window_baseline_v1.rune_exps()
  center_window_baseline_v3.run_exps()
  center_window_baseline_v4.run_exps()
  center_window_baseline_v5.run_exps()
  center_window_baseline_v5_small.run_exps()
  center_window_baseline_v6.run_exps()
  # center_window_baseline_v7.run_exps()

  for folder_name in [
    "log-prob-grads_wrt_enc-11_log-space",
    "log-probs-wo-h_t-grads_wrt_enc-11_log-space",
  ]:
    analyze_gradients_jobs = [
      center_window_att_data.analyze_gradients_jobs["baseline_v3_full-sum"]["v2"][f"epoch-360"],
      center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v2"],
      center_window_att_data.analyze_gradients_jobs["baseline_v5_full-sum"]["v1"][f"epoch-720"],
      center_window_att_data.analyze_gradients_jobs["baseline_v5_two-stage"]["fixed-path"]["v2"],
      center_window_att_data.analyze_gradients_jobs["baseline_v5_two-stage"]["fixed-path"]["v4"],
    ]
    titles = [
      "Full ctx transducer w/ global RNN attention full-sum (epoch 450/900)",
      "Full ctx transducer w/ global RNN attention fixed-path (epoch 600/600)",
      "Full ctx transducer w/ global Trafo attention full-sum (epoch 720/900)",
      "Full ctx transducer w/ global Trafo attention fixed-path (epoch 600/600)",
      "Ctx-1 transducer w/ global 'Linear' attention fixed-path (epoch 600/600)",
    ]

    if folder_name == "log-prob-grads_wrt_enc-11_log-space":
      analyze_gradients_jobs.insert(0, center_window_att_data.analyze_gradients_jobs["baseline_v3_full-sum"]["v1"][f"epoch-450"])
      titles.insert(0, "Full ctx transducer w/o attention full-sum (epoch 450/900)")

    plot_diff_models(
      analyze_gradients_jobs,
      alias=f"{center_window_base_alias}/{folder_name}_different_models",
      titles=titles,
      folder_name=folder_name,
    )


def setup_gmm_alignment():
  gmm_alignment = hdf.build_hdf_from_alignment(
    alignment_cache=LIBRISPEECH_GMM_ALIGNMENT.alignment_caches["train"],
    allophone_file=LIBRISPEECH_GMM_ALIGNMENT.allophone_file,
    state_tying_file=Path("/u/schmitt/experiments/segmental_models_2022_23_rf/state-tying.lut"),
    returnn_python_exe=RETURNN_EXE_NEW,
    returnn_root=RETURNN_CURRENT_ROOT,
  )

  convert_gmm_alignment_job = ConvertGmmAlignmentJob(
    gmm_alignment_hdf=gmm_alignment,
    allophone_path=LIBRISPEECH_GMM_ALIGNMENT.allophone_file,
    state_tying_path=Path("/u/schmitt/experiments/segmental_models_2022_23_rf/state-tying.lut"),
  )
  convert_gmm_alignment_job.add_alias("datasets/gmm-alignments/train")
  tk.register_output(convert_gmm_alignment_job.get_one_alias(), convert_gmm_alignment_job.out_hdf_align)

  LIBRISPEECH_GMM_ALIGNMENT_CONVERTED.alignment_paths = {"train": convert_gmm_alignment_job.out_hdf_align}
  LIBRISPEECH_GMM_ALIGNMENT_CONVERTED.vocab_path = convert_gmm_alignment_job.out_vocab
  
  gmm_alignment = hdf.build_hdf_from_alignment(
    alignment_cache=LIBRISPEECH_GMM_ALIGNMENT.alignment_caches["train"],
    allophone_file=LIBRISPEECH_GMM_ALIGNMENT.allophone_file,
    state_tying_file=Path("/u/schmitt/experiments/segmental_models_2022_23_rf/state-tying"),
    returnn_python_exe=RETURNN_EXE_NEW,
    returnn_root=RETURNN_CURRENT_ROOT,
  )
  
  gmm_to_word_boundaries_job = GmmAlignmentToWordBoundariesJob(
    gmm_alignment_hdf=gmm_alignment,
    bliss_corpus=LIBRISPEECH_CORPUS.corpus_paths["train-other-960"],
    allophone_path=LIBRISPEECH_GMM_ALIGNMENT.allophone_file,
  )
  gmm_to_word_boundaries_job.add_alias("datasets/gmm-word-boundaries/train")
  tk.register_output(gmm_to_word_boundaries_job.get_one_alias(), gmm_to_word_boundaries_job.out_hdf_align)

  LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths = {"train": gmm_to_word_boundaries_job.out_hdf_align}
  LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path = gmm_to_word_boundaries_job.out_vocab
