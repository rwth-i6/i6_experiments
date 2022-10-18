from recipe.i6_experiments.users.schmitt.alignment.alignment import DumpPhonemeAlignJob, AugmentBPEAlignmentJob, \
  AlignmentStatisticsJob, AlignmentSplitSilenceJob, ReduceAlignmentJob
from recipe.i6_experiments.users.schmitt.util.util import ModifySeqFileJob
from recipe.i6_experiments.users.schmitt.rasr.convert import PhonJSONVocabToRasrFormatsJob, \
  BPEJSONVocabToRasrFormatsJob

from sisyphus import *


def update_seq_list_file(seq_list_file, seqs_to_skip, alias):
  mod_seq_file_job = ModifySeqFileJob(
    seq_file=seq_list_file,
    seqs_to_skip=seqs_to_skip)
  mod_seq_file_job.add_alias(alias)
  alias = mod_seq_file_job.get_one_alias()
  tk.register_output(alias, mod_seq_file_job.out_seqs_file)
  return mod_seq_file_job.out_seqs_file


def dump_phoneme_align(rasr_exe, rasr_config, time_red, alias, time_rqmt=4, mem_rqmt=4):
  dump_phon_align_job = DumpPhonemeAlignJob(
    rasr_config=rasr_config,
    time_red=time_red, time_rqtm=time_rqmt, rasr_exe=rasr_exe, mem_rqmt=mem_rqmt)
  dump_phon_align_job.add_alias(alias)
  alias = dump_phon_align_job.get_one_alias()
  tk.register_output(alias, dump_phon_align_job.out_align)
  return dump_phon_align_job.out_align, dump_phon_align_job.out_phoneme_vocab


def calc_align_stats(alignment, seq_filter_file, alias, blank_idx=89):
  stat_job = AlignmentStatisticsJob(alignment=alignment, returnn_root="/u/schmitt/src/returnn",
    returnn_python_exe="/u/rossenbach/bin/returnn_tf2.3_launcher.sh", seq_list_filter_file=seq_filter_file,
    blank_idx=blank_idx, silence_idx=0, time_rqmt=1)
  stat_job.add_alias(alias)
  alias = stat_job.get_one_alias()
  tk.register_output(alias, stat_job.out_statistics)

  return stat_job.out_label_dep_stats_var, stat_job.out_mean_non_sil_len_var, stat_job.out_95_percentile_var


def alignment_split_silence(alignment, seq_filter_file, alias, blank_idx, sil_idx, max_len):
  split_align_job = AlignmentSplitSilenceJob(hdf_align_path=alignment, segment_file=seq_filter_file,
    blank_idx=blank_idx, sil_idx=sil_idx, max_len=max_len)
  split_align_job.add_alias(alias)
  alias = split_align_job.get_one_alias()
  tk.register_output(alias, split_align_job.out_align)

  return split_align_job.out_align


def reduce_alignment(alignment, seq_filter_file, alias, blank_idx, sil_idx, reduction_factor):
  red_align_job = ReduceAlignmentJob(hdf_align_path=alignment, segment_file=seq_filter_file,
    blank_idx=blank_idx, sil_idx=sil_idx, reduction_factor=reduction_factor)
  red_align_job.add_alias(alias)
  alias = red_align_job.get_one_alias()
  tk.register_output(alias, red_align_job.out_align)

  return red_align_job.out_align, red_align_job.out_skipped_seqs_var


def augment_bpe_align_with_sil(phon_align, bpe_align, seq_filter_file, phon_time_red, phon_vocab, alias, time_rqmt, mem_rqmt):
  augment_bpe_align_job = AugmentBPEAlignmentJob(
    bpe_align_hdf=bpe_align,
    phoneme_align_hdf=phon_align, bpe_blank_idx=1030, phoneme_blank_idx=89,
    bpe_vocab="/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k",
    phoneme_vocab=phon_vocab,
    phoneme_lexicon="/u/zhou/asr-exps/swb1/dependencies/train.lex.v1_0_3.ci.gz",
    segment_file=seq_filter_file, time_red_phon_align=phon_time_red,
    time_red_bpe_align=6, time_rqtm=time_rqmt, mem_rqmt=mem_rqmt)
  augment_bpe_align_job.add_alias(alias)
  alias = augment_bpe_align_job.get_one_alias()
  tk.register_output(alias + "/align", augment_bpe_align_job.out_align)
  tk.register_output(alias + "/vocab", augment_bpe_align_job.out_vocab)

  return augment_bpe_align_job.out_align, augment_bpe_align_job.out_skipped_seqs_var, augment_bpe_align_job.out_vocab


def convert_phon_json_vocab_to_rasr_formats(json_vocab_path, blank_idx):
  json_to_rasr_job = PhonJSONVocabToRasrFormatsJob(json_vocab_path, blank_idx=blank_idx)
  json_to_rasr_job.add_alias("phon_rasr_formats")
  tk.register_output("phon_state_tying", json_to_rasr_job.out_state_tying)
  tk.register_output("phon_allophones", json_to_rasr_job.out_allophones)
  tk.register_output("phon_label_file", json_to_rasr_job.out_rasr_label_file)

  return json_to_rasr_job.out_state_tying, json_to_rasr_job.out_allophones, json_to_rasr_job.out_rasr_label_file


def convert_bpe_json_vocab_to_rasr_formats(json_vocab_path, blank_idx, alias):
  json_to_rasr_job = BPEJSONVocabToRasrFormatsJob(json_vocab_path, blank_idx=blank_idx)
  json_to_rasr_job.add_alias(alias)
  tk.register_output("bpe_state_tying", json_to_rasr_job.out_state_tying)
  tk.register_output("bpe_allophones", json_to_rasr_job.out_allophones)
  tk.register_output("bpe_label_file", json_to_rasr_job.out_rasr_label_file)

  return json_to_rasr_job.out_state_tying, json_to_rasr_job.out_allophones, json_to_rasr_job.out_rasr_label_file
