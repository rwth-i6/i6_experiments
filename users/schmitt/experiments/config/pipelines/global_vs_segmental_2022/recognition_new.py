from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.sub_pipelines_new import \
  run_rasr_decoding, run_eval, calc_rasr_search_errors
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.miscellaneous_new import \
  calc_align_stats
from recipe.i6_experiments.users.schmitt.alignment.alignment import CompareAlignmentsJob
from recipe.i6_experiments.users.schmitt.returnn.tools import DumpAttentionWeightsJob
from recipe.i6_experiments.users.schmitt.visualization.visualization import PlotAttentionWeightsJob

from sisyphus import *


def start_rasr_recog_pipeline(
  ref_stm_path, model_name, recog_epoch, rasr_decoding_opts, rasr_search_error_decoding_opts,
  blank_idx, cv_segments, recog_corpus_name,
  alias_addon):
  ctm_results = run_rasr_decoding(**rasr_decoding_opts)
  run_eval(ctm_file=ctm_results,
           reference=ref_stm_path, name=model_name,
           dataset_key=recog_corpus_name, num_epochs=recog_epoch, alias_addon=alias_addon)

  search_align, ctm_results = calc_rasr_search_errors(**rasr_search_error_decoding_opts)
  # run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=model_name, dataset_key="cv", num_epochs=recog_epoch,
  #          alias_addon=alias_addon)

  calc_align_stats(alignment=search_align, blank_idx=blank_idx, seq_filter_file=cv_segments,
    alias=model_name + "/" + alias_addon + "/cv_search_align_stats_epoch-%s" % recog_epoch)

  return search_align


def start_analysis_pipeline(
  group_alias, feed_config, vocab_file, cv_align, search_align, rasr_config, blank_idx, model_type,
  rasr_nn_trainer_exe, seq_tag, epoch):

  for align_alias, align in zip(["ground-truth", "search", "realign"], [cv_align, search_align]):
    dump_att_weights_job = DumpAttentionWeightsJob(
      returnn_config=feed_config, model_type=model_type,
      rasr_config=rasr_config, blank_idx=blank_idx, label_name="alignment",
      rasr_nn_trainer_exe=rasr_nn_trainer_exe, hdf_targets=align, seq_tag=seq_tag)
    dump_att_weights_job.add_alias(
      group_alias + "/" + seq_tag.replace("/", "_") + "/att_weights_%s_%s" % (align_alias, epoch))
    tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

    plot_weights_job = PlotAttentionWeightsJob(data_path=dump_att_weights_job.out_data, blank_idx=blank_idx,
                                               json_vocab_path=vocab_file, time_red=6, seq_tag=seq_tag)
    plot_weights_job.add_alias(
      group_alias + "/" + seq_tag.replace("/", "_") + "/plot_att_weights_%s_%s" % (align_alias, epoch))
    tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)

  compare_aligns_job = CompareAlignmentsJob(
    hdf_align1=cv_align, hdf_align2=search_align, seq_tag=seq_tag,
    blank_idx1=blank_idx, blank_idx2=blank_idx, vocab1=vocab_file, vocab2=vocab_file,
    name1="ground_truth", name2="search_alignment")
  compare_aligns_job.add_alias(group_alias + "/" + seq_tag.replace("/", "_") + "/search-align-compare")
  tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)
