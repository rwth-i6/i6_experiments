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
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import baseline_v8 as center_window_baseline_v8
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
  # center_window_baseline_v5_small.run_exps()
  center_window_baseline_v6.run_exps()
  # center_window_baseline_v7.run_exps()
  center_window_baseline_v8.run_exps()

  for alias in [
    "v2_long_two-stage",
    "v3_long_two-stage",
  ]:
    plot_gradient_wrt_enc11(
      analyze_gradients_job=center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"][alias]["24gb-gpu"],
      alias=f"stacked-enc-11-grads/center_window_baseline_v3_two-stage_fixed-path_{alias}_24gb-gpu",
    )

  # for folder_name in [
  #   "log-prob-grads_wrt_enc-11_log-space",
  #   # "log-probs-wo-h_t-grads_wrt_enc-11_log-space",
  # ]:
  #   analyze_gradients_jobs = [
  #     center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v1_long_two-stage"][
  #       "24gb-gpu"],
  #     center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v2_long_two-stage"][
  #       "24gb-gpu"],
  #     center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v3_long_two-stage"][
  #       "24gb-gpu"],
  #     center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v9_long_two-stage"][
  #       "24gb-gpu"],
  #   ]
  #   titles = [
  #     "Standard full output label ctx. Transducer",
  #     "Simple concat.; W/o att. ctx. in state; w/o att. weight feedback",
  #     "Simple concat.; W/ att. ctx. in state; w/o att. weight feedback",
  #     "Simple concat.; W/ att. ctx. in state; w/ att. weight feedback",
  #   ]
  #
  #   plot_diff_models(
  #     analyze_gradients_jobs,
  #     alias=f"stacked-enc-11-grads/center_window_baseline_v3_two-stage_fixed-path_v1-vs-v2-vs-v3-vs-v9_long_two-stage_24gb-gpu",
  #     titles=titles,
  #     folder_name=folder_name,
  #     scale=1.8,
  #   )

  for folder_name in [
    "log-prob-grads_wrt_enc-11_log-space",
    # "log-probs-wo-h_t-grads_wrt_enc-11_log-space",
  ]:
    analyze_gradients_jobs = [
      center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v1_long_two-stage"][
        "24gb-gpu"],
      center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v12_long_two-stage"][
        "24gb-gpu"],
      center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v2_long_two-stage"][
        "24gb-gpu"],
      center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v4_long_two-stage"][
        "24gb-gpu"],
      center_window_att_data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v6_long_two-stage"][
        "24gb-gpu"],
    ]
    aliases = [
      "v1", "v12", "v2", "v4", "v6", "v3", "v5", "v8", "v9", "v11", "v13", "v14"

    ]
    titles = [
      "Only $h_t$",
      "Only $c_{s_t}$",
      "Simple Concat.",
      "Double Gate",
      "Single Gate",
    ]

    plot_diff_models(
      analyze_gradients_jobs,
      alias=f"stacked-enc-11-grads/center_window_baseline_v3_two-stage_fixed-path_v1-vs-v12-vs-v2-vs-v4-vs-v6_long_two-stage_24gb-gpu",
      titles=titles,
      folder_name=folder_name,
      scale=1.8,
    )

    for baseline, alias in [
      ("baseline_v3_two-stage", "v1"),
      ("baseline_v3_two-stage", "v12"),
      ("baseline_v3_two-stage", "v2"),
      ("baseline_v3_two-stage", "v4"),
      ("baseline_v3_two-stage", "v6"),
      ("baseline_v3_two-stage", "v3"),
      ("baseline_v3_two-stage", "v5"),
      ("baseline_v3_two-stage", "v8"),
      ("baseline_v3_two-stage", "v9"),
      ("baseline_v3_two-stage", "v11"),
      ("baseline_v3_two-stage", "v13"),
      ("baseline_v3_two-stage", "v14"),
      ("baseline_v5_two-stage", "v2"),
      ("baseline_v5_two-stage", "v3"),
    ]:
      if baseline == "baseline_v3_two-stage":
        analyze_gradients_job = center_window_att_data.analyze_gradients_jobs[baseline]["fixed-path"][f"{alias}_long_two-stage"]["24gb-gpu"]
      else:
        analyze_gradients_job = center_window_att_data.analyze_gradients_jobs[baseline]["fixed-path"][f"{alias}_long_two-stage"]
      plot_diff_models(
        [analyze_gradients_job],
        alias=f"enc-11-grads/fixed-path/{baseline}_{alias}_long_two-stage_24gb-gpu",
        titles=None,  # titles,
        folder_name=folder_name,
        scale=1.0,
        vmin=-20.0,
        vmax=4.0,
      )
      if baseline == "baseline_v3_two-stage":
        plot_diff_models(
          [analyze_gradients_job],
          alias=f"att-weights/fixed-path/{baseline}_{alias}_long_two-stage_24gb-gpu",
          titles=None,  # titles,
          folder_name="enc-layer-12/att_weights",
          folder_prefix="cross-att",
          scale=1.0,
          vmin=0.0,
          vmax=1.0,
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
