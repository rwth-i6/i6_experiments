from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.sub_pipelines import \
  run_rasr_decoding, run_eval, calc_rasr_search_errors
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.miscellaneous import \
  calc_align_stats
from i6_private.users.schmitt.returnn.tools import \
  CompareAlignmentsJob, DumpAttentionWeightsJob, PlotAttentionWeightsJob

import copy

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

  calc_align_stats(alignment=search_align, blank_idx=blank_idx, seq_filter_file=cv_segments,
    alias=model_name + "/" + alias_addon + "/cv_search_align_stats_epoch-%s" % recog_epoch)

  # if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs":
  #   if alias_addon == "rasr_limit12_pruning12.0_no-recomb_label-dep-length-glob-var-None_max-seg-len-25_length-scale-0.0_length-norm_beam_search_all-segments_global_import":
  #     dump_non_blanks_job = DumpNonBlanksFromAlignmentJob(search_align, blank_idx=targetb_blank_idx)
  #     dump_non_blanks_job.add_alias("dump_non_blanks_" + alias_addon)
  #     search_aligns["global_import_segmental"] = search_align
  #     search_labels["global_import_segmental"] = dump_non_blanks_job.out_labels
  # if name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs":
  #   if alias_addon == "rasr_limit12_pruning12.0_vit-recomb_label-dep-length-glob-var-None_max-seg-len-20_length-scale-0.0_length-norm_beam_search_all-segments_global_import_split-sil":
  #     dump_non_blanks_job = DumpNonBlanksFromAlignmentJob(search_align, blank_idx=targetb_blank_idx)
  #     dump_non_blanks_job.add_alias("dump_non_blanks_" + alias_addon)
  #     search_aligns["global_import_segmental_w_split_sil"] = search_align
  #     search_labels["global_import_segmental_w_split_sil"] = dump_non_blanks_job.out_labels

  return search_align, ctm_results


def start_analysis_pipeline(
  group_alias, feed_config, vocab_file, cv_align, search_align, rasr_config, blank_idx, model_type,
  rasr_nn_trainer_exe, seq_tag, epoch, cv_realign=None):
  # cv_realignment = run_rasr_realignment(
  #   compile_config=compile_config,
  #   alias_addon="", segment_path=Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/cv_test_segments1"),
  #   loop_update_history=True, blank_update_history=True if not vit_recomb else False, name=group_alias,
  #   corpus_path=corpus_files["train"], lexicon_path=bpe_phon_lexicon_path if params["config"]["label_type"] == "bpe" else bpe_sil_phon_lexicon_path,
  #   allophone_path=total_data[params["config"]["label_type"]]["allophones"],
  #   state_tying_path=total_data[params["config"]["label_type"]]["state_tying"],
  #   feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
  #   label_file=total_data[params["config"]["label_type"]]["rasr_label_file"], label_pruning=12.0,
  #   label_pruning_limit=5000, label_recombination_limit=-1, blank_label_index=targetb_blank_idx,
  #   model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
  #   rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
  #   rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1, time_rqmt=2,
  #   blank_allophone_state_idx=4119 if params["config"]["label_type"] == "bpe" else 4123,
  #   max_segment_len=max_seg_len, mem_rqmt=16)

  aligns = [cv_align, search_align]
  if cv_realign is not None:
    aligns += [cv_realign]

  for align_alias, align in zip(["ground-truth", "search", "realign"], aligns):
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

  if cv_realign is not None:
    compare_aligns_job = CompareAlignmentsJob(
      hdf_align1=cv_align, hdf_align2=cv_realign, seq_tag=seq_tag,
      blank_idx1=blank_idx, blank_idx2=blank_idx, vocab1=vocab_file, vocab2=vocab_file,
      name1="ground_truth", name2="realignment")
    compare_aligns_job.add_alias(group_alias + "/" + seq_tag.replace("/", "_") + "/realign-compare")
    tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)
