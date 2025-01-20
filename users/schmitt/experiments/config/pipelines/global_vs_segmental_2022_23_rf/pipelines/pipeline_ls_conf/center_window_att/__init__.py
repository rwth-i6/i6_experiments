from i6_experiments.users.schmitt.visualization.visualization import PlotAttentionWeightsJobV2
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.gmm_alignments import LIBRISPEECH_GMM_WORD_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE1056_CTC_ALIGNMENT

from sisyphus.delayed_ops import DelayedFormat
from sisyphus import Path, tk


def plot_gradient_wrt_enc11(analyze_gradients_job, alias):
  gradient_hdfs = [
    Path(DelayedFormat(
      f"{{}}/{alias}/att_weights.hdf",
      analyze_gradients_job.out_files["enc-11"]
    ).get()) for alias in [
      "log-prob-grads_wrt_enc-11_log-space",
      "log-probs-wo-att-grads_wrt_enc-11_log-space",
      "log-probs-wo-h_t-grads_wrt_enc-11_log-space",
    ]
  ]
  targets_hdf = analyze_gradients_job.out_files["targets.hdf"]

  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=gradient_hdfs,
    targets_hdf=targets_hdf,
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
    ref_alignment_hdf=LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"],
    json_vocab_path=LibrispeechBPE1056_CTC_ALIGNMENT.vocab_path,
    ctc_alignment_hdf=None,
    ref_alignment_json_vocab_path=LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
    plot_w_cog=False,
    titles=[
      "$G_{11}$",
      "$G_{11}$ w/o contribution of $c_{s_t}$",
      "$G_{11}$ w/o contribution of $h_t$",
    ],
    vmin=-20,
    vmax=0,
  )
  plot_att_weights_job.add_alias(alias)
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)


def plot_diff_models(
        analyze_gradients_jobs, alias, titles, folder_name, folder_prefix: str = "enc-11", scale: float = 1.0, vmin: float = -20, vmax: float = 0):
  gradient_hdfs = [
    Path(DelayedFormat(
      f"{{}}/{folder_name}/att_weights.hdf",
      analyze_gradients_job.out_files[folder_prefix]
    ).get()) for analyze_gradients_job in analyze_gradients_jobs
  ]
  targets_hdf = analyze_gradients_jobs[0].out_files["targets.hdf"]

  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=gradient_hdfs,
    targets_hdf=targets_hdf,
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
    ref_alignment_hdf=LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"],
    json_vocab_path=LibrispeechBPE1056_CTC_ALIGNMENT.vocab_path,
    ctc_alignment_hdf=None,
    ref_alignment_json_vocab_path=LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
    plot_w_cog=False,
    titles=titles,
    vmin=vmin,
    vmax=vmax,
    scale=scale,
  )
  plot_att_weights_job.add_alias(alias)
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)
