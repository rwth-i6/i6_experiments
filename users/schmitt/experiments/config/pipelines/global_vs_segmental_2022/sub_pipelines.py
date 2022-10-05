# from sisyphus import *
from recipe.i6_experiments.users.schmitt.experiments.swb.transducer.config import *
from recipe.i6_core.returnn.config import ReturnnConfig
from recipe.i6_core.returnn.training import ReturnnTrainingJob, ReturnnTrainingFromFileJob
from recipe.i6_core.returnn.search import ReturnnSearchJob, SearchWordsToCTMJob, SearchBPEtoWordsJob, \
  ReturnnSearchFromFileJob
from recipe.i6_core.recognition.scoring import Hub5ScoreJob
from recipe.i6_private.users.schmitt.returnn.tools import RASRLatticeToCTMJob, RASRDecodingJob, CompileTFGraphJob, \
  ConvertCTMBPEToWordsJob, CalcSearchErrorJob, RASRRealignmentJob, DumpRASRAlignJob, DumpAlignmentFromTxtJob, \
  WordsToCTMJob
from recipe.i6_private.users.schmitt.returnn.search import ReturnnDumpSearchJob
from sisyphus import *
import copy
from recipe.i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.build_rasr_configs import build_lattice_to_ctm_config, build_decoding_config, build_realignment_config, \
  build_phon_align_extraction_config, build_iterate_corpus_config


def run_training(returnn_config: ReturnnConfig, name, num_epochs, alias_suffix, mem_rqmt, time_rqmt, add_input=None, qsub_args=None):
  train_job = ReturnnTrainingJob(
    copy.deepcopy(returnn_config),
    num_epochs=num_epochs[-1],
    log_verbosity=5,
    returnn_python_exe="/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
    returnn_root="/u/schmitt/src/returnn",
    mem_rqmt=mem_rqmt,
    time_rqmt=time_rqmt,
    qsub_args=qsub_args
    # keep_epochs=num_epochs
  )
  if add_input:
    train_job.add_input(add_input)
  train_job.add_alias(name + "/" + alias_suffix)
  alias = train_job.get_one_alias()
  tk.register_output(alias + "/config", train_job.out_returnn_config_file)
  tk.register_output(alias + "/models", train_job.out_model_dir)
  tk.register_output(alias + "/learning_rates", train_job.out_learning_rates)
  tk.register_output(alias + "/plot_se", train_job.out_plot_se)
  tk.register_output(alias + "/plot_lr", train_job.out_plot_lr)

  return train_job.out_checkpoints, train_job.returnn_config, train_job


def run_training_from_file(config_file_path, parameter_dict, name, alias_suffix, time_rqmt, mem_rqmt):
  assert type(parameter_dict) == dict
  parameter_dict.update(dict(ext_task="train"))
  train_job = ReturnnTrainingFromFileJob(
    returnn_config_file=config_file_path, parameter_dict=parameter_dict, time_rqmt=time_rqmt, mem_rqmt=mem_rqmt)
  train_job.add_alias(name + "/" + alias_suffix)
  alias = train_job.get_one_alias()
  tk.register_output(alias + "/config", train_job.returnn_config_file)
  tk.register_output(alias + "/models", train_job.model_dir)
  tk.register_output(alias + "/learning_rates", train_job.learning_rates)

  return train_job.model_dir


def run_search_from_file(
        config_file_path, parameter_dict, model_dir, load_epoch, default_model_name, name, stm_job, alias_suffix, time_rqmt=1, mem_rqmt=4):
  assert type(parameter_dict) == dict
  parameter_dict.update(
    dict(ext_task="search", ext_model=model_dir, ext_load_epoch=load_epoch))
  search_job = ReturnnSearchFromFileJob(
    returnn_config_file=config_file_path, parameter_dict=parameter_dict, default_model_name=default_model_name,
    time_rqmt=time_rqmt, mem_rqmt=mem_rqmt)
  search_job.add_alias(name + ("/search_dev_%d" % load_epoch) + alias_suffix)
  alias = search_job.get_one_alias()
  tk.register_output(alias + "/config", search_job.returnn_config_file)
  tk.register_output(alias + "/search-results", search_job.out_search_file)

  bpe_to_words_job = SearchBPEtoWordsJob(search_job.out_search_file)
  bpe_to_words_job.add_alias(name + ("/words_dev_%d" % load_epoch) + alias_suffix)
  alias = bpe_to_words_job.get_one_alias()
  tk.register_output(alias + "/word_search_results", bpe_to_words_job.out_word_search_results)

  ctm_job = SearchWordsToCTMJob(bpe_to_words_job.out_word_search_results, stm_job.bliss_corpus)
  ctm_job.add_alias(name + ("/ctm_dev_%d" % load_epoch) + alias_suffix)
  alias = ctm_job.get_one_alias()
  tk.register_output(alias + "/ctm_search_results", ctm_job.out_ctm_file)

  return ctm_job.out_ctm_file


def run_bpe_returnn_decoding(
  returnn_config: ReturnnConfig, checkpoint, stm_job, name, dataset_key, num_epochs, device="gpu", alias_addon="",
  concat_seqs=False, stm_path=None,
  mem_rqmt=4, time_rqmt=2):
  assert not concat_seqs or stm_path

  search_job = ReturnnSearchJob(
    search_data={},
    model_checkpoint=checkpoint,
    returnn_config=copy.deepcopy(returnn_config),
    returnn_python_exe="/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
    returnn_root="/u/schmitt/src/returnn",
    device=device,
    mem_rqmt=mem_rqmt,
    time_rqmt=time_rqmt)
  search_job.add_alias(name + "/%s/search_%s_%d" % (alias_addon, dataset_key, num_epochs))
  alias = search_job.get_one_alias()
  tk.register_output(alias + "/bpe_search_results", search_job.out_search_file)
  tk.register_output(alias + "/config", search_job.out_returnn_config_file)

  bpe_to_words_job = SearchBPEtoWordsJob(search_job.out_search_file)
  # bpe_to_words_job.add_alias(name + ("/words_%s_%d" % (dataset_key, num_epochs)) + alias_addon)
  alias = name + ("/words_%s_%d" % (dataset_key, num_epochs)) + alias_addon
  tk.register_output(alias + "/word_search_results", bpe_to_words_job.out_word_search_results)

  if concat_seqs:
    ctm_job = WordsToCTMJob(
      words_path=bpe_to_words_job.out_word_search_results, stm_path=stm_path, dataset_name=dataset_key)
    alias = name + ("/ctm_%s_%d" % (dataset_key, num_epochs)) + alias_addon
    tk.register_output(alias + "/ctm_search_results", ctm_job.out_ctm_file)
  else:
    ctm_job = SearchWordsToCTMJob(
      bpe_to_words_job.out_word_search_results,
      stm_job.bliss_corpus)
    # ctm_job.add_alias(name + ("/ctm_%s_%d" % (dataset_key, num_epochs)) + alias_addon)
    alias = name + ("/ctm_%s_%d" % (dataset_key, num_epochs)) + alias_addon
    tk.register_output(alias + "/ctm_search_results", ctm_job.out_ctm_file)

  return ctm_job.out_ctm_file


def run_rasr_decoding(
  corpus_path, segment_path, lexicon_path, feature_cache_path, label_file_path,
  simple_beam_search, blank_label_index, reduction_factors, start_label_index, blank_update_history,
  allow_label_recombination, allow_word_end_recombination, full_sum_decoding, label_pruning, label_pruning_limit,
  use_lm_score, word_end_pruning, word_end_pruning_limit, label_recombination_limit, label_unit,
  skip_silence, lm_type, lm_scale, lm_file, lm_image, lm_lookahead, max_seg_len,
  compile_config, rasr_exe_path, loop_update_history, length_norm,
  model_checkpoint, name, num_epochs, lm_lookahead_cache_size_high=None, label_scorer_type="tf-rnn-transducer",
  lm_lookahead_cache_size_low=None, lm_lookahead_history_limit=None, lm_lookahead_scale=None,
  time_rqmt=10, mem_rqmt=4, gpu_rqmt=1, alias_addon="", debug=False, max_batch_size=256):

  compile_graph_job = CompileTFGraphJob(compile_config, "output")
  compile_graph_job.add_alias(name + "/tf-graph" + alias_addon)
  alias = name + "/tf-graph" + alias_addon
  tk.register_output(alias + "/tf-graph", compile_graph_job.out_graph)
  tk.register_output(alias + "/tf-rec-info", compile_graph_job.out_rec_info)
  meta_graph_path = compile_graph_job.out_graph


  decoding_crp, decoding_config = build_decoding_config(
    corpus_path=corpus_path, length_norm=length_norm,
    segment_path=segment_path, lexicon_path=lexicon_path, feature_cache_path=feature_cache_path,
    label_pruning=label_pruning, label_unit=label_unit, label_pruning_limit=label_pruning_limit,
    word_end_pruning_limit=word_end_pruning_limit, loop_update_history=loop_update_history,
    simple_beam_search=simple_beam_search, full_sum_decoding=full_sum_decoding,
    blank_update_history=blank_update_history, word_end_pruning=word_end_pruning,
    allow_word_end_recombination=allow_word_end_recombination,
    allow_label_recombination=allow_label_recombination, lm_type=lm_type,
    lm_image=lm_image, lm_scale=lm_scale, use_lm_score=use_lm_score, lm_file=lm_file,
    label_file_path=label_file_path, start_label_index=start_label_index, skip_silence=skip_silence,
    blank_label_index=blank_label_index,
    label_recombination_limit=label_recombination_limit, reduction_factors=reduction_factors, debug=debug,
    meta_graph_path=meta_graph_path, lm_lookahead=lm_lookahead, max_seg_len=max_seg_len,
    lm_lookahead_cache_size_high=lm_lookahead_cache_size_high, lm_lookahead_cache_size_low=lm_lookahead_cache_size_low,
    lm_lookahead_scale=lm_lookahead_scale, lm_lookahead_history_limit=lm_lookahead_history_limit,
    max_batch_size=max_batch_size, label_scorer_type=label_scorer_type,
  )

  rasr_decoding_job = RASRDecodingJob(
    rasr_exe_path=rasr_exe_path, flf_lattice_tool_config=decoding_config, crp=decoding_crp,
    model_checkpoint=model_checkpoint, dump_best_trace=False,
    mem_rqmt=mem_rqmt, time_rqmt=time_rqmt, gpu_rqmt=gpu_rqmt)
  rasr_decoding_job.add_alias(name + ("/%s/rasr-decoding-epoch-%d" % (alias_addon, num_epochs)))
  alias = rasr_decoding_job.get_one_alias()
  tk.register_output(alias + "/results", rasr_decoding_job.out_lattice)

  lattice_to_ctm_crp, lattice_to_ctm_config = build_lattice_to_ctm_config(
    corpus_path=corpus_path, segment_path=segment_path, lexicon_path=lexicon_path,
    lattice_path=rasr_decoding_job.out_lattice
  )

  lattice_to_ctm_job = RASRLatticeToCTMJob(
    rasr_exe_path=rasr_exe_path, lattice_path=rasr_decoding_job.out_lattice,
    crp=lattice_to_ctm_crp, flf_lattice_tool_config=lattice_to_ctm_config)
  # lattice_to_ctm_job.add_alias(name + ("/lattice-to-ctm-epoch-%d" % num_epochs) + alias_addon)
  alias = name + ("/lattice-to-ctm-epoch-%d" % num_epochs) + alias_addon
  tk.register_output(alias, lattice_to_ctm_job.out_ctm)

  bpe_ctm_to_words_job = ConvertCTMBPEToWordsJob(bpe_ctm_file=lattice_to_ctm_job.out_ctm)
  # bpe_ctm_to_words_job.add_alias(name + ("/bpe-ctm-to-words-epoch-%d" % num_epochs) + alias_addon)
  alias = name + ("/bpe-ctm-to-words-epoch-%d" % num_epochs) + alias_addon
  tk.register_output(alias, bpe_ctm_to_words_job.out_ctm_file)

  return bpe_ctm_to_words_job.out_ctm_file


def calc_rasr_search_errors(
  corpus_path, segment_path, lexicon_path, feature_cache_path, label_file_path, length_norm,
  simple_beam_search, blank_label_index, reduction_factors, start_label_index, blank_update_history,
  allow_label_recombination, allow_word_end_recombination, full_sum_decoding, label_pruning, label_pruning_limit,
  use_lm_score, word_end_pruning, word_end_pruning_limit, label_recombination_limit, label_unit,
  skip_silence, lm_type, lm_scale, lm_file, lm_image, lm_lookahead, max_seg_len, num_classes,
  compile_config: ReturnnConfig, rasr_exe_path, train_config, extern_sprint_rasr_config,
  rasr_nn_trainer_exe, blank_idx, ref_align, loop_update_history, model_type, label_name,
  model_checkpoint, name, num_epochs, lm_lookahead_cache_size_high=None, label_scorer_type="tf-rnn-transducer",
  lm_lookahead_cache_size_low=None, lm_lookahead_history_limit=None, lm_lookahead_scale=None,
  time_rqmt=10, mem_rqmt=4, gpu_rqmt=1, alias_addon="", debug=False, max_batch_size=256):

  compile_graph_job = CompileTFGraphJob(compile_config, "output")
  compile_graph_job.add_alias(name + "/tf-graph" + alias_addon)
  alias = name + "/tf-graph" + alias_addon
  tk.register_output(alias + "/tf-graph", compile_graph_job.out_graph)
  tk.register_output(alias + "/tf-rec-info", compile_graph_job.out_rec_info)

  decoding_crp, decoding_config = build_decoding_config(
    corpus_path=corpus_path, length_norm=length_norm,
    segment_path=segment_path, lexicon_path=lexicon_path, feature_cache_path=feature_cache_path,
    label_pruning=label_pruning, label_unit=label_unit, label_pruning_limit=label_pruning_limit,
    word_end_pruning_limit=word_end_pruning_limit, loop_update_history=loop_update_history,
    simple_beam_search=simple_beam_search, full_sum_decoding=full_sum_decoding,
    blank_update_history=blank_update_history, word_end_pruning=word_end_pruning,
    allow_word_end_recombination=allow_word_end_recombination,
    allow_label_recombination=allow_label_recombination, lm_type=lm_type,
    lm_image=lm_image, lm_scale=lm_scale, use_lm_score=use_lm_score, lm_file=lm_file,
    label_file_path=label_file_path, start_label_index=start_label_index, skip_silence=skip_silence,
    blank_label_index=blank_label_index,
    label_recombination_limit=label_recombination_limit, reduction_factors=reduction_factors, debug=debug,
    meta_graph_path=compile_graph_job.out_graph, lm_lookahead=lm_lookahead, max_seg_len=max_seg_len,
    lm_lookahead_cache_size_high=lm_lookahead_cache_size_high, lm_lookahead_cache_size_low=lm_lookahead_cache_size_low,
    lm_lookahead_scale=lm_lookahead_scale, lm_lookahead_history_limit=lm_lookahead_history_limit,
    max_batch_size=max_batch_size, label_scorer_type=label_scorer_type
  )

  rasr_decoding_job = RASRDecodingJob(
    rasr_exe_path=rasr_exe_path, flf_lattice_tool_config=decoding_config, crp=decoding_crp,
    model_checkpoint=model_checkpoint, dump_best_trace=True,
    mem_rqmt=mem_rqmt, time_rqmt=time_rqmt, gpu_rqmt=gpu_rqmt)
  rasr_decoding_job.add_alias(name + ("/%s/rasr-decoding-dump-traces-epoch-%d" % (alias_addon, num_epochs)))
  alias = rasr_decoding_job.get_one_alias()
  tk.register_output(alias + "/results", rasr_decoding_job.out_lattice)

  dump_align_from_txt_job = DumpAlignmentFromTxtJob(
    alignment_txt=rasr_decoding_job.out_best_traces, segment_file=segment_path, num_classes=num_classes
  )
  dump_align_from_txt_job.add_alias(name + ("/%s/rasr-decoding-dump-traces-best-traces-epoch-%d" % (alias_addon, num_epochs)))

  calc_search_err_job = CalcSearchErrorJob(
    returnn_config=train_config, rasr_config=extern_sprint_rasr_config,
    rasr_nn_trainer_exe=rasr_nn_trainer_exe, segment_file=segment_path, blank_idx=blank_idx,
    search_targets=dump_align_from_txt_job.out_hdf_align,
    ref_targets=ref_align, label_name=label_name, model_type=model_type,
    max_seg_len=max_seg_len if max_seg_len is not None else -1, length_norm=length_norm)
  calc_search_err_job.add_alias(name + ("/%s/search_errors_%d" % (alias_addon, num_epochs)))
  alias = calc_search_err_job.get_one_alias()
  tk.register_output(alias + "search_errors", calc_search_err_job.out_search_errors)

  lattice_to_ctm_crp, lattice_to_ctm_config = build_lattice_to_ctm_config(corpus_path=corpus_path,
    segment_path=segment_path, lexicon_path=lexicon_path, lattice_path=rasr_decoding_job.out_lattice)

  lattice_to_ctm_job = RASRLatticeToCTMJob(rasr_exe_path=rasr_exe_path, lattice_path=rasr_decoding_job.out_lattice,
    crp=lattice_to_ctm_crp, flf_lattice_tool_config=lattice_to_ctm_config)
  # lattice_to_ctm_job.add_alias(name + ("/lattice-to-ctm-epoch-%d" % num_epochs) + alias_addon)
  alias = name + ("/lattice-to-ctm-epoch-%d" % num_epochs) + alias_addon
  tk.register_output(alias, lattice_to_ctm_job.out_ctm)

  bpe_ctm_to_words_job = ConvertCTMBPEToWordsJob(bpe_ctm_file=lattice_to_ctm_job.out_ctm)
  # bpe_ctm_to_words_job.add_alias(name + ("/bpe-ctm-to-words-epoch-%d" % num_epochs) + alias_addon)
  alias = name + ("/bpe-ctm-to-words-epoch-%d" % num_epochs) + alias_addon
  tk.register_output(alias, bpe_ctm_to_words_job.out_ctm_file)

  return dump_align_from_txt_job.out_hdf_align, bpe_ctm_to_words_job.out_ctm_file


def run_rasr_realignment(
  compile_config, name, corpus_path, segment_path, lexicon_path, feature_cache_path, allophone_path, label_pruning,
  label_pruning_limit, rasr_am_trainer_exe_path, rasr_nn_trainer_exe_path, model_checkpoint, num_epochs,
  label_recombination_limit, blank_label_index, context_size, label_file, reduction_factors, max_segment_len,
  start_label_index, state_tying_path, num_classes, time_rqmt, blank_allophone_state_idx, length_norm,
  blank_update_history, loop_update_history, mem_rqmt, data_key,
  alias_addon=""
):

  compile_graph_job = CompileTFGraphJob(compile_config, "output")
  compile_graph_job.add_alias(name + "/tf-graph" + alias_addon)
  alias = compile_graph_job.get_one_alias()
  tk.register_output(alias + "/tf-graph", compile_graph_job.out_graph)
  tk.register_output(alias + "/tf-rec-info", compile_graph_job.out_rec_info)

  realignment_crp, realignment_config = build_realignment_config(
    corpus_path=corpus_path, lexicon_path=lexicon_path, segment_path=segment_path, length_norm=length_norm,
    feature_cache_path=feature_cache_path, reduction_factors=reduction_factors, blank_label_index=blank_label_index,
    start_label_index=start_label_index, label_pruning=label_pruning, label_pruning_limit=label_pruning_limit,
    label_recombination_limit=label_recombination_limit, label_file=label_file, allophone_path=allophone_path,
    context_size=context_size, meta_graph_file=compile_graph_job.out_graph, state_tying_path=state_tying_path,
    max_segment_len=max_segment_len, blank_update_history=blank_update_history, loop_update_history=loop_update_history)

  realignment_job = RASRRealignmentJob(
    rasr_exe_path=rasr_am_trainer_exe_path, crp=realignment_crp, model_checkpoint=model_checkpoint,
    mem_rqmt=mem_rqmt, time_rqtm=time_rqmt, am_model_trainer_config=realignment_config,
    blank_allophone_state_idx=blank_allophone_state_idx)
  realignment_job.add_alias(name + ("/%s/rasr-realignment-epoch-%d-%s" % (alias_addon, num_epochs, data_key)))
  alias = realignment_job.get_one_alias()
  tk.register_output(alias + "/results", realignment_job.out_alignment)

  dump_align_from_txt_job = DumpAlignmentFromTxtJob(
    alignment_txt=realignment_job.out_alignment_txt,
    segment_file=segment_path, num_classes=num_classes)
  dump_align_from_txt_job.add_alias(name + ("/%s/rasr-realignment-epoch-%d-best-traces" % (alias_addon, num_epochs)))
  tk.register_output(dump_align_from_txt_job.get_one_alias(), dump_align_from_txt_job.out_hdf_align)

  return dump_align_from_txt_job.out_hdf_align




def run_eval(ctm_file, reference, name, dataset_key, num_epochs, alias_addon=""):

  score_job = Hub5ScoreJob(
    reference, "/work/asr2/oberdorfer/kaldi-stable/egs/swbd/s5/data/eval2000/glm", ctm_file
  )
  score_job.add_alias(name + ("/%s/scores_%s_%d" % (alias_addon, dataset_key, num_epochs)))
  alias = score_job.get_one_alias()
  tk.register_output(alias + "/score_reports", score_job.out_report_dir)


def calculate_search_errors(
  checkpoint, search_config, train_config, rasr_config, rasr_nn_trainer_exe, segment_path, model_type, label_name,
  name, epoch, ref_targets, blank_idx, dataset_key, stm_job, length_norm, concat_seq_tags_file=None, alias_addon=""):
  search_job = ReturnnDumpSearchJob(search_data={}, model_checkpoint=checkpoint,
                                    returnn_config=copy.deepcopy(search_config.get_config()),
                                    returnn_python_exe="/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
                                    returnn_root="/u/schmitt/src/returnn", mem_rqmt=6, time_rqmt=1)
  search_job.add_alias(name + "/search_%s_%d" % (dataset_key, epoch) + alias_addon)
  alias = name + "/search_%s_%d" % (dataset_key, epoch) + alias_addon
  # tk.register_output(alias + "/bpe_search_results", search_job.out_search_file)
  # tk.register_output(alias + "/config", search_job.out_returnn_config_file)
  tk.register_output(alias + "/search_out_seqs", search_job.out_search_seqs_file)

  search_targets = search_job.out_search_seqs_file

  calc_search_err_job = CalcSearchErrorJob(
    returnn_config=train_config, rasr_config=rasr_config, rasr_nn_trainer_exe=rasr_nn_trainer_exe,
    segment_file=segment_path, blank_idx=blank_idx, model_type=model_type, label_name=label_name,
    search_targets=search_targets, ref_targets=ref_targets, max_seg_len=-1, length_norm=length_norm,
    concat_seqs=False if concat_seq_tags_file is None else True, concat_seq_tags_file=concat_seq_tags_file)
  calc_search_err_job.add_alias(name + ("/%s/search_errors_%d" % (alias_addon, epoch)))
  alias = calc_search_err_job.get_one_alias()
  tk.register_output(alias + "search_errors", calc_search_err_job.out_search_errors)

  bpe_to_words_job = SearchBPEtoWordsJob(search_job.out_search_file)
  # bpe_to_words_job.add_alias(name + ("/words_%s_%d" % (dataset_key, num_epochs)) + alias_addon)
  alias = name + ("/words_%s_%d" % (dataset_key, epoch)) + alias_addon
  tk.register_output(alias + "/word_search_results", bpe_to_words_job.out_word_search_results)

  ctm_job = SearchWordsToCTMJob(bpe_to_words_job.out_word_search_results, stm_job.bliss_corpus)
  # ctm_job.add_alias(name + ("/ctm_%s_%d" % (dataset_key, num_epochs)) + alias_addon)
  alias = name + ("/ctm_%s_%d" % (dataset_key, epoch)) + alias_addon
  tk.register_output(alias + "/ctm_search_results", ctm_job.out_ctm_file)

  return search_targets, ctm_job.out_ctm_file

