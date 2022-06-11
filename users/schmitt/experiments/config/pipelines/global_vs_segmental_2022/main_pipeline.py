import copy
import os

import numpy
import numpy as np

from i6_private.users.schmitt.returnn.tools import DumpForwardJob, CompileTFGraphJob, RASRDecodingJob, \
  CombineAttentionPlotsJob, DumpPhonemeAlignJob, AugmentBPEAlignmentJob, FindSegmentsToSkipJob, ModifySeqFileJob, \
  ConvertCTMBPEToWordsJob, RASRLatticeToCTMJob, CompareAlignmentsJob, DumpAttentionWeightsJob, PlotAttentionWeightsJob, \
  DumpNonBlanksFromAlignmentJob, CalcSearchErrorJob, RemoveLabelFromAlignmentJob, WordsToCTMJob

from recipe.i6_core.corpus import *
from recipe.i6_core.bpe.apply import ApplyBPEModelToLexiconJob
from recipe.i6_core.tools.git import CloneGitRepositoryJob
from recipe.i6_core.corpus.convert import CorpusToTxtJob
from recipe.i6_core.returnn.training import Checkpoint
from recipe.i6_experiments.users.schmitt.experiments.config.general.concat_seqs import run_concat_seqs

# from sisyphus import *
from sisyphus.delayed_ops import DelayedFormat, DelayedReplace
from i6_experiments.users.schmitt.experiments.swb.transducer.config import \
  TransducerSWBExtendedConfig, TransducerSWBAlignmentConfig
from i6_experiments.users.schmitt.experiments.swb.global_enc_dec.config import \
  GlobalEncoderDecoderConfig
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.sub_pipelines import run_eval, run_training, run_bpe_returnn_decoding, \
  run_rasr_decoding, calculate_search_errors, run_rasr_realignment, calc_rasr_search_errors, run_training_from_file, \
  run_search_from_file

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.model_variants import build_alias, model_variants
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.build_rasr_configs import build_returnn_train_config, build_returnn_train_feature_flow, \
  build_phon_align_extraction_config, write_config, build_decoding_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.miscellaneous import find_seqs_to_skip, update_seq_list_file, dump_phoneme_align, calc_align_stats, \
  augment_bpe_align_with_sil, alignment_split_silence, reduce_alignment, convert_phon_json_vocab_to_rasr_vocab, \
  convert_phon_json_vocab_to_allophones, convert_phon_json_vocab_to_state_tying, \
  convert_phon_json_vocab_to_rasr_formats, convert_bpe_json_vocab_to_rasr_formats


def run_pipeline():
  hub5e_00_stm_job = CorpusToStmJob(bliss_corpus=Path("/work/asr3/irie/data/switchboard/corpora/dev.corpus.gz"))
  hub5e_00_stm_job.add_alias("stm_files" + "/hub5e_00")
  alias = hub5e_00_stm_job.get_one_alias()
  tk.register_output(alias + "/stm_corpus", hub5e_00_stm_job.out_stm_path)

  hub5e_01_stm_job = CorpusToStmJob(bliss_corpus=Path("/work/asr3/irie/data/switchboard/corpora/hub5e_01.corpus.gz"))
  hub5e_01_stm_job.add_alias("stm_files" + "/hub5e_01")
  alias = hub5e_01_stm_job.get_one_alias()
  tk.register_output(alias + "/stm_corpus", hub5e_01_stm_job.out_stm_path)

  rt03s_stm_job = CorpusToStmJob(bliss_corpus=Path("/work/asr3/irie/data/switchboard/corpora/rt03s.corpus.gz"))
  rt03s_stm_job.add_alias("stm_files" + "/rt03s")
  alias = rt03s_stm_job.get_one_alias()
  tk.register_output(alias + "/stm_corpus", rt03s_stm_job.out_stm_path)

  eval_ref_files = {
    "dev": Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), "hub5e_01": Path("/u/tuske/bin/switchboard/hub5e_01.2.stm"),
    "rt03s": Path("/u/tuske/bin/switchboard/rt03s_ctsonly.stm"), }

  concat_jobs = run_concat_seqs(
    ref_stm_paths={
      "hub5e_00": eval_ref_files["dev"], "hub5e_01": eval_ref_files["hub5e_01"],
      "rt03s": eval_ref_files["rt03s"]},
    glm_path=Path("/work/asr2/oberdorfer/kaldi-stable/egs/swbd/s5/data/eval2000/glm"),
    concat_nums=[1, 2, 4, 10, 20, 30, 100])

  assert concat_jobs is not None

  train_stm_job = CorpusToStmJob(bliss_corpus=Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz"))
  train_stm_job.add_alias("stm_files" + "/train")
  alias = train_stm_job.get_one_alias()
  tk.register_output(alias + "/stm_corpus", train_stm_job.out_stm_path)

  filter_train_corpus_job = FilterCorpusBySegmentsJob(
    bliss_corpus=Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz"),
    segment_file=Path("/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000"))

  cv_stm_job = CorpusToStmJob(bliss_corpus=filter_train_corpus_job.out_corpus)
  cv_stm_job.add_alias("stm_files" + "/cv")
  alias = cv_stm_job.get_one_alias()
  tk.register_output(alias + "/stm_corpus", cv_stm_job.out_stm_path)

  allophone_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/allophones")

  phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/zhou-phoneme-transducer/lexicon/train.lex.v1_0_3.ci.gz")
  phon_lexicon_w_blank_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/phonemes/lex_with_blank")
  phon_lexicon_wei = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/phonemes/lexicon_wei")

  bpe_sil_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe-with-sil_lexicon")
  bpe_sil_phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_lexicon_phon")
  bpe_phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_lexicon_phon_wo_sil")

  phon_state_tying_mono_eow_3_states_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/state-tying_mono-eow_3-states")
  zoltan_phon_align_cache = Path("/work/asr4/zhou/asr-exps/swb1/dependencies/zoltan-alignment/train.alignment.cache.bundle", cached=True)

  rasr_nn_trainer = Path("/u/schmitt/src/rasr/arch/linux-x86_64-standard/nn-trainer.linux-x86_64-standard")
  rasr_flf_tool = Path("/u/schmitt/src/rasr/arch/linux-x86_64-standard/flf-tool.linux-x86_64-standard")
  rasr_am_trainer = Path("/u/schmitt/src/rasr/arch/linux-x86_64-standard/acoustic-model-trainer.linux-x86_64-standard")

  bpe_vocab = {
    "bpe_file": Path('/work/asr3/irie/data/switchboard/subword_clean/ready/swbd_clean.bpe_code_1k'),
    "vocab_file": Path('/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k')
  }

  corpus_files = {
    "train": Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz", cached=True),
    "dev": Path("/work/asr3/irie/data/switchboard/corpora/dev.corpus.gz", cached=True),
    "hub5e_01": Path("/work/asr3/irie/data/switchboard/corpora/hub5e_01.corpus.gz", cached=True),
    "rt03s": Path("/work/asr3/irie/data/switchboard/corpora/rt03s.corpus.gz", cached=True),
  }

  feature_cache_files = {
    "train": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle", cached=True),
    "dev": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.dev.bundle", cached=True),
    "hub5e_01": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.hub5e_01.bundle", cached=True),
    "rt03s": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.rt03s.bundle", cached=True)
  }

  bpe_standard_aligns = {
    "train": Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-train.hdf",
      cached=True),
    "cv": Path(
      "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-dev.hdf",
      cached=True)}

  seq_filter_files_standard = {
    "train": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_train", cached=True),
    "devtrain": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_train_head3000", cached=True),
    "cv": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000", cached=True)}

  dev_debug_segment_file = Path(
    "/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/segments.1"
  )

  zoltan_4gram_lm = {
    "image": Path("/u/zhou/asr-exps/swb1/dependencies/zoltan_4gram.image"),
    "image_wei": Path("/u/zhou/asr-exps/swb1/dependencies/monophone-eow/clean-all/zoltan_4gram.image"),
    "file": Path("/work/asr4/zhou/asr-exps/swb1/dependencies/zoltan_4gram.gz")
  }

  build_returnn_train_feature_flow(feature_cache_path=feature_cache_files["train"])

  phon_extraction_rasr_configs = {
    "train": write_config(*build_phon_align_extraction_config(
      corpus_path=corpus_files["train"], feature_cache_path=feature_cache_files["train"],
      segment_path=seq_filter_files_standard["train"],
      lexicon_path=phon_lexicon_path, allophone_path=allophone_path, alignment_cache_path=zoltan_phon_align_cache
    ), alias="phon_extract_train_rasr_config"),
    "cv": write_config(
      *build_phon_align_extraction_config(corpus_path=corpus_files["train"],
        feature_cache_path=feature_cache_files["train"], segment_path=seq_filter_files_standard["cv"],
        lexicon_path=phon_lexicon_path, allophone_path=allophone_path, alignment_cache_path=zoltan_phon_align_cache),
      alias="phon_extract_cv_rasr_config")
  }


  segment_corpus_job = SegmentCorpusJob(corpus_files["dev"], num_segments=1)
  segment_corpus_job.add_alias("segments_dev")
  tk.register_output("segments_dev", segment_corpus_job.out_single_segment_files[1])

  total_data = {
    "bpe": {}, "phonemes": {}, "phonemes-split-sil": {}, "bpe-with-sil": {}, "bpe-with-sil-split-sil": {},
    "bpe-sil-wo-sil": {}, "bpe-sil-wo-sil-in-middle": {}, "bpe-with-sil-split-silv2": {}}
  # phoneme_aligns = {}
  # phoneme_split_sil_aligns = {"time-red-1": {}}
  # seq_filter_files_phons = {"time-red-1": {}}
  # phoneme_label_dep_mean_lens = {}
  # phoneme_split_sil_label_dep_mean_lens = {"time-red-1": {}}
  # phoneme_aligns["time-red-1"] = {}
  # bpe_sil_aligns = {"time-red-1": {}}
  # bpe_sil_split_sil_aligns = {"time-red-1": {}}
  # seq_filter_files_bpe_sil = {"time-red-1": {}}
  # bpe_sil_label_dep_mean_lens = {"time-red-1": {}}
  # bpe_sil_split_sil_label_dep_mean_lens = {"time-red-1": {}}
  # bpe_label_dep_mean_lens = {}
  # bpe_sil_aligns["time-red-1"] = {}
  for corpus_key in [
    "cv",
    "train"
  ]:
    if corpus_key == "cv":
      time_rqmt = 2
      mem_rqmt = 4
    else:
      time_rqmt = 10
      mem_rqmt = 6

    # # prepare dict structure for time red 1
    # total_data["time-red-1"][corpus_key] = {}
    # if corpus_key == "train":
    #   total_data["time-red-1"]["devtrain"] = {}

    # total_data["bpe_align"] = {}
    # total_data["time-red-1"][corpus_key]["bpe_sil_align"] = {}
    # total_data["time-red-1"][corpus_key]["bpe_sil_split_sil_align"] = {}
    # total_data["time-red-1"][corpus_key]["phoneme_align"] = {}
    # total_data["time-red-1"][corpus_key]["phoneme_split_sil_align"] = {}
    # if corpus_key == "train":
    #   total_data["time-red-6"]["devtrain"]["bpe_align"] = {}
    #   total_data["time-red-1"]["devtrain"]["bpe_sil_align"] = {}
    #   total_data["time-red-1"]["devtrain"]["bpe_sil_split_sil_align"] = {}
    #   total_data["time-red-1"]["devtrain"]["phoneme_align"] = {}
    #   total_data["time-red-1"]["devtrain"]["phoneme_split_sil_align"] = {}

    bpe_seq_filter_file = seq_filter_files_standard[corpus_key]
    # ----------------------- BPE ALIGNMENTS -----------------------------------------
    bpe_label_dep_mean_lens, bpe_mean_non_sil_len, bpe_95_percentile = calc_align_stats(
      alignment=bpe_standard_aligns[corpus_key],
      seq_filter_file=bpe_seq_filter_file, alias="bpe_align_stats/stats_" + corpus_key,
      blank_idx=1030)

    bpe_labels_job = DumpNonBlanksFromAlignmentJob(
      alignment=bpe_standard_aligns[corpus_key], blank_idx=1030, time_rqmt=time_rqmt)
    bpe_labels_job.add_alias("bpe_labels/%s" % corpus_key)
    tk.register_output("bpe_labels/%s" % corpus_key, bpe_labels_job.out_labels)

    seq_filter_files_bpe = update_seq_list_file(seq_list_file=bpe_seq_filter_file,
      seqs_to_skip=bpe_labels_job.out_skipped_seqs_var,
      alias="seq_filter_files_bpe/time-red-%s/%s" % (6, corpus_key))

    bpe_state_tying, bpe_allophones, bpe_rasr_label_file = convert_bpe_json_vocab_to_rasr_formats(
      bpe_vocab["vocab_file"], blank_idx=1030, alias="bpe_rasr_formats")
    total_data["bpe"].update({
      "json_vocab": bpe_vocab["vocab_file"],
      "state_tying": bpe_state_tying, "allophones": bpe_allophones, "rasr_label_file": bpe_rasr_label_file})
    total_data["bpe"][corpus_key] = {
      "label_seqs": bpe_labels_job.out_labels,
      "time-red-6": {
        "align": bpe_standard_aligns[corpus_key],
        "seq_filter_file": seq_filter_files_bpe,
        "label_dep_mean_lens": bpe_label_dep_mean_lens, "mean_non_sil_len": bpe_mean_non_sil_len,
        "95_percentile": bpe_95_percentile}
    }
    if corpus_key == "train":
      total_data["bpe"]["devtrain"] = {
        "time-red-6": {
          "seq_filter_file": seq_filter_files_standard["devtrain"]}}

    # ----------------------- PHONEME ALIGNMENTS -----------------------------------------
    # extract phoneme alignments
    phoneme_align, phoneme_vocab_path = dump_phoneme_align(
      time_rqmt=time_rqmt, rasr_exe=rasr_nn_trainer, rasr_config=phon_extraction_rasr_configs[corpus_key],
      mem_rqmt=mem_rqmt, time_red=1, alias="phon_align/%s/%s" % ("time-red-1", corpus_key))

    phon_state_tying, phon_allophones, phon_rasr_label_file = convert_phon_json_vocab_to_rasr_formats(
      phoneme_vocab_path, blank_idx=89)

    # calculate alignment statistics for phoneme alignment without time reduction
    phoneme_label_dep_mean_lens, phoneme_mean_non_sil_len, phoneme_95_percentile = calc_align_stats(
      alignment=phoneme_align,
      seq_filter_file=bpe_seq_filter_file, alias="phon_align_stats/stats_" + corpus_key)

    phoneme_labels_job = DumpNonBlanksFromAlignmentJob(
      alignment=phoneme_align, blank_idx=89, time_rqmt=time_rqmt
    )
    phoneme_labels_job.add_alias("phoneme_labels/%s" % corpus_key)
    tk.register_output("phoneme_labels/%s" % corpus_key, phoneme_labels_job.out_labels)


    total_data["phonemes"].update({
      "json_vocab": phoneme_vocab_path,
      "state_tying": phon_state_tying, "allophones": phon_allophones, "rasr_label_file": phon_rasr_label_file})
    total_data["phonemes"][corpus_key] = {
      "label_seqs": phoneme_labels_job.out_labels,
      "time-red-1": {
        "align": phoneme_align,
        "seq_filter_file": bpe_seq_filter_file, "label_dep_mean_lens": phoneme_label_dep_mean_lens,
        "mean_non_sil_len": phoneme_mean_non_sil_len, "95_percentile": phoneme_95_percentile}}
    if corpus_key == "train":
      total_data["phonemes"]["devtrain"] = {
        "time-red-1": {
          "seq_filter_file": seq_filter_files_standard["devtrain"]}}

    # ----------------------- PHONEME SPLIT SILENCE ALIGNMENTS -----------------------------------------

    phoneme_split_sil_align = alignment_split_silence(
      sil_idx=0, blank_idx=89, alias="phon_split_sil_align/align_%s" % corpus_key, alignment=phoneme_align,
      seq_filter_file=bpe_seq_filter_file, max_len=phoneme_mean_non_sil_len)
    phoneme_split_sil_label_dep_mean_lens, phoneme_split_sil_mean_non_sil_len, phoneme_split_sil_95_percentile = calc_align_stats(
      alignment=phoneme_split_sil_align, seq_filter_file=bpe_seq_filter_file,
      alias="phon_split_sil_align_stats/stats_" + corpus_key)

    phoneme_split_sil_labels_job = DumpNonBlanksFromAlignmentJob(alignment=phoneme_split_sil_align, blank_idx=89, time_rqmt=time_rqmt)
    phoneme_split_sil_labels_job.add_alias("phoneme-split-sil_labels/%s" % corpus_key)
    tk.register_output(phoneme_split_sil_labels_job.get_one_alias(), phoneme_split_sil_labels_job.out_labels)

    total_data["phonemes-split-sil"][corpus_key] = {
      "label_seqs": phoneme_split_sil_labels_job.out_labels,
      "time-red-1": {
        "align": phoneme_split_sil_align, "seq_filter_file": bpe_seq_filter_file,
        "label_dep_mean_lens": phoneme_split_sil_label_dep_mean_lens,
        "mean_non_sil_len": phoneme_split_sil_mean_non_sil_len,
        "95_percentile": phoneme_split_sil_95_percentile}}
    if corpus_key == "train":
      total_data["phonemes-split-sil"]["devtrain"] = {
        "time-red-1": {
          "seq_filter_file": seq_filter_files_standard["devtrain"]}}

    # ----------------------- BPE + SILENCE ALIGNMENTS -----------------------------------------

    bpe_sil_align, bpe_sil_skipped_seqs, bpe_sil_vocab_path = augment_bpe_align_with_sil(
      phon_align=phoneme_align,
      bpe_align=bpe_standard_aligns[corpus_key],
      seq_filter_file=bpe_seq_filter_file,
      phon_vocab=phoneme_vocab_path,
      alias="bpe_sil_align/%s/%s" % ("time-red-1", corpus_key), phon_time_red=1,
      time_rqmt=2 if corpus_key == "dev" else 6, mem_rqmt=mem_rqmt)

    bpe_sil_vocab = {
      "bpe_file": bpe_vocab["bpe_file"],
      "vocab_file": bpe_sil_vocab_path
    }

    seq_filter_files_bpe_sil = update_seq_list_file(
      seq_list_file=bpe_seq_filter_file, seqs_to_skip=bpe_sil_skipped_seqs,
      alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (1, corpus_key))

    # calculate alignment statistics for bpe + silence alignment with time red factor 1
    bpe_sil_label_dep_mean_lens, bpe_sil_mean_non_sil_len, bpe_sil_95_percentile = calc_align_stats(
      alignment=bpe_sil_align,
      seq_filter_file=seq_filter_files_bpe_sil,
      alias="bpe-with-sil_align_stats/stats_" + corpus_key, blank_idx=1031)

    bpe_sil_labels_job = DumpNonBlanksFromAlignmentJob(alignment=bpe_sil_align, blank_idx=1031, time_rqmt=time_rqmt)
    bpe_sil_labels_job.add_alias("bpe-sil_labels/%s" % corpus_key)
    tk.register_output("bpe-sil_labels/%s" % corpus_key, bpe_sil_labels_job.out_labels)

    bpe_sil_state_tying, bpe_sil_allophones, bpe_sil_rasr_label_file = convert_bpe_json_vocab_to_rasr_formats(
      bpe_sil_vocab_path, blank_idx=1031, alias="bpe_sil_rasr_formats")

    total_data["bpe-with-sil"].update({
      "json_vocab": bpe_sil_vocab_path,
      "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones, "rasr_label_file": bpe_sil_rasr_label_file})
    total_data["bpe-with-sil"][corpus_key] = {
      "label_seqs": bpe_sil_labels_job.out_labels,
      "time-red-1": {
        "align": bpe_sil_align, "seq_filter_file": seq_filter_files_bpe_sil,
        "label_dep_mean_lens": bpe_sil_label_dep_mean_lens,
        "mean_non_sil_len": bpe_sil_mean_non_sil_len, "95_percentile": bpe_sil_95_percentile}}
    if corpus_key == "train":
      seq_filter_files_bpe_sil_devtrain = update_seq_list_file(
        seq_list_file=seq_filter_files_standard["devtrain"],
        seqs_to_skip=bpe_sil_skipped_seqs, alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (1, "devtrain"))
      total_data["bpe-with-sil"]["devtrain"] = {
        "time-red-1": {
          "seq_filter_file": seq_filter_files_bpe_sil_devtrain}}

    # ----------------------- BPE + SILENCE SPLIT SILENCE ALIGNMENTS -----------------------------------------

    bpe_sil_split_sil_align = alignment_split_silence(
      sil_idx=0, blank_idx=1031,
      alias="bpe_sil_split_sil_align/align_%s" % corpus_key, alignment=bpe_sil_align,
      seq_filter_file=seq_filter_files_bpe_sil, max_len=bpe_sil_mean_non_sil_len)
    bpe_sil_split_sil_label_dep_mean_lens, bpe_sil_split_sil_mean_non_sil_len, bpe_sil_split_sil_95_percentile = calc_align_stats(
      alignment=bpe_sil_split_sil_align, blank_idx=1031,
      seq_filter_file=seq_filter_files_bpe_sil,
      alias="bpe-with-sil-split-sil_align_stats/time-red-1/stats_" + corpus_key)

    bpe_sil_split_sil_labels_job = DumpNonBlanksFromAlignmentJob(alignment=bpe_sil_split_sil_align, blank_idx=1031, time_rqmt=time_rqmt)
    bpe_sil_split_sil_labels_job.add_alias("bpe-sil-split-sil_labels/%s" % corpus_key)
    tk.register_output(bpe_sil_split_sil_labels_job.get_one_alias(), bpe_sil_split_sil_labels_job.out_labels)

    total_data["bpe-with-sil-split-sil"].update({
      "json_vocab": bpe_sil_vocab_path,
      "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones, "rasr_label_file": bpe_sil_rasr_label_file})
    total_data["bpe-with-sil-split-sil"][corpus_key] = {
      "label_seqs": bpe_sil_split_sil_labels_job.out_labels,
      "time-red-1": {
        "align": bpe_sil_split_sil_align, "seq_filter_file": seq_filter_files_bpe_sil,
        "label_dep_mean_lens": bpe_sil_split_sil_label_dep_mean_lens,
        "mean_non_sil_len": bpe_sil_split_sil_mean_non_sil_len,
        "95_percentile": bpe_sil_split_sil_95_percentile}}
    if corpus_key == "train":
      seq_filter_files_bpe_sil_devtrain = update_seq_list_file(
        seq_list_file=seq_filter_files_standard["devtrain"],
        seqs_to_skip=bpe_sil_skipped_seqs, alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (1, "devtrain"))
      total_data["bpe-with-sil-split-sil"]["devtrain"] = {
        "time-red-1": {
          "seq_filter_file": seq_filter_files_bpe_sil_devtrain}}

  # ----------------------- BPE + SILENCE SPLIT SILENCE V2 label dependent means --------------------------------------

  bpe_sil_split_silv2_label_dep_mean_lens, bpe_sil_split_silv2_mean_non_sil_len, bpe_sil_split_silv2_95_percentile = calc_align_stats(
    alignment=Path("/work/asr3/zeyer/schmitt/old_models_and_analysis/old_bpe_sil_split_sil_aligns/train/AlignmentSplitSilenceJob.4dfiua41gqWb/output/out_align"),
    blank_idx=1031, seq_filter_file=seq_filter_files_bpe_sil,
    alias="bpe-with-sil-split-silv2_align_stats/time-red-6/stats_train")

  total_data["bpe-with-sil-split-silv2"].update({
    "json_vocab": bpe_sil_vocab_path, "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones,
    "rasr_label_file": bpe_sil_rasr_label_file})
  total_data["bpe-with-sil-split-silv2"]["train"] = {
    "time-red-6": {
      "seq_filter_file": seq_filter_files_bpe_sil,
      "label_dep_mean_lens": bpe_sil_split_silv2_label_dep_mean_lens,
      "mean_non_sil_len": bpe_sil_split_silv2_mean_non_sil_len, "95_percentile": bpe_sil_split_silv2_95_percentile}}

  for time_red in [2, 3, 6]:
    for label_type in ["bpe-with-sil", "phonemes"]:
      for corpus_key in ["train", "cv"]:
        # get reduce alignment
        align, red_skipped_seqs = reduce_alignment(
          alignment=total_data[label_type][corpus_key]["time-red-1"]["align"],
          sil_idx=0,
          blank_idx=1031 if label_type == "bpe-with-sil" else 89,
          alias="%s_align/%s/%s" % (label_type, "time-red-%d" % time_red, corpus_key),
          seq_filter_file=total_data[label_type][corpus_key]["time-red-1"]["seq_filter_file"],
          reduction_factor=time_red)
        # get seq filter file for reduced alignment
        seq_filter_file = update_seq_list_file(
          seq_list_file=total_data[label_type][corpus_key]["time-red-1"]["seq_filter_file"],
          seqs_to_skip=red_skipped_seqs,
          alias="seq_filter_files_%s/time-red-%s/%s" % (label_type, time_red, corpus_key))
        # get label dependent means and mean non sil len for reduced alignment
        label_dep_mean_lens, mean_non_sil_len, percentile_95 = calc_align_stats(
          alignment=align,
          seq_filter_file=seq_filter_file,
          alias="%s-align_stats/time-red-%d/stats_%s" % (label_type, time_red, corpus_key),
          blank_idx=1031 if label_type == "bpe-with-sil" else 89)

        total_data[label_type][corpus_key].update({
          "time-red-%s" % time_red: {
            "align": align, "seq_filter_file": seq_filter_file,
            "label_dep_mean_lens": label_dep_mean_lens,
            "mean_non_sil_len": mean_non_sil_len, "95_percentile": percentile_95}})
        if corpus_key == "train":
          seq_filter_file_devtrain = update_seq_list_file(
            seq_list_file=total_data[label_type]["devtrain"]["time-red-1"]["seq_filter_file"],
            seqs_to_skip=red_skipped_seqs,
            alias="seq_filter_files_%s/time-red-%s/%s" % (label_type, time_red, "devtrain"))
          total_data[label_type]["devtrain"].update({
            "time-red-%s" % time_red: {
              "seq_filter_file": seq_filter_file_devtrain}})

        # get reduced alignment with split silence
        split_sil_align = alignment_split_silence(
          sil_idx=0,
          blank_idx=1031 if label_type == "bpe-with-sil" else 89,
          alias="%s-split-sil_align/time-red-%d/align_%s" % (label_type, time_red, corpus_key),
          alignment=align,
          seq_filter_file=seq_filter_file,
          max_len=mean_non_sil_len)
        # get label dep means and mean non sil len for reduced split sil alignment
        split_sil_label_dep_mean_lens, split_sil_mean_non_sil_len, split_sil_percentile_95 = calc_align_stats(
          alignment=split_sil_align,
          blank_idx=1031 if label_type == "bpe-with-sil" else 89,
          seq_filter_file=seq_filter_file,
          alias="%s-split-sil_align_stats/time-red-%d/stats_%s" % (label_type, time_red, corpus_key))

        total_data[label_type + "-split-sil"][corpus_key].update({
          "time-red-%s" % time_red: {
            "align": split_sil_align, "seq_filter_file": seq_filter_file,
            "label_dep_mean_lens": split_sil_label_dep_mean_lens,
            "mean_non_sil_len": split_sil_mean_non_sil_len,
            "95_percentile": split_sil_percentile_95}})
        if corpus_key == "train":
          seq_filter_file_devtrain = update_seq_list_file(
            seq_list_file=total_data[label_type]["devtrain"]["time-red-1"]["seq_filter_file"],
            seqs_to_skip=red_skipped_seqs,
            alias="seq_filter_files_%s/time-red-%s/%s" % (label_type, time_red, "devtrain"))
          total_data[label_type + "-split-sil"]["devtrain"].update({
            "time-red-%s" % time_red: {
              "seq_filter_file": seq_filter_file_devtrain}})

  for remove_only_middle in [True, False]:
    for corpus_key in ["train", "cv"]:
      bpe_sil_wo_sil_align_job = RemoveLabelFromAlignmentJob(
        alignment=total_data["bpe-with-sil-split-sil"][corpus_key]["time-red-6"]["align"], blank_idx=1031,
        remove_idx=0, remove_only_middle=remove_only_middle)
      bpe_sil_wo_sil_align_job.add_alias("bpe-sil-wo-sil%s/time-red-6/%s" % ("-in-middle" if remove_only_middle else "", corpus_key))
      tk.register_output(bpe_sil_wo_sil_align_job.get_one_alias(), bpe_sil_wo_sil_align_job.out_alignment)

      bpe_sil_wo_sil_label_dep_mean_lens, bpe_sil_wo_sil_mean_non_sil_len, bpe_sil_wo_sil_95_percentile = calc_align_stats(
        alignment=bpe_sil_wo_sil_align_job.out_alignment, blank_idx=1031,
        seq_filter_file=total_data["bpe-with-sil-split-sil"][corpus_key]["time-red-6"]["seq_filter_file"],
        alias="bpe-sil-wo-sil%s_align_stats/time-red-6/stats_%s" % ("-in-middle" if remove_only_middle else "", corpus_key))

      total_data["bpe-sil-wo-sil%s" % ("-in-middle" if remove_only_middle else "")].update({
        "json_vocab": bpe_sil_vocab_path, "state_tying": bpe_sil_state_tying, "allophones": bpe_sil_allophones,
        "rasr_label_file": bpe_sil_rasr_label_file})
      total_data["bpe-sil-wo-sil%s" % ("-in-middle" if remove_only_middle else "")][corpus_key] = {
        "label_seqs": None,
        "time-red-6": {
          "align": bpe_sil_wo_sil_align_job.out_alignment,
          "seq_filter_file": total_data["bpe-with-sil-split-sil"][corpus_key]["time-red-6"]["seq_filter_file"],
          "label_dep_mean_lens": bpe_sil_wo_sil_label_dep_mean_lens,
          "mean_non_sil_len": bpe_sil_wo_sil_mean_non_sil_len, "95_percentile": bpe_sil_wo_sil_95_percentile}}
      if corpus_key == "train":
        seq_filter_files_bpe_sil_devtrain = update_seq_list_file(seq_list_file=seq_filter_files_standard["devtrain"],
          seqs_to_skip=bpe_sil_skipped_seqs, alias="seq_filter_files_bpe_sil/time-red-%s/%s" % (1, "devtrain"))
        total_data["bpe-sil-wo-sil%s" % ("-in-middle" if remove_only_middle else "")]["devtrain"] = {
          "time-red-6": {
            "seq_filter_file": seq_filter_files_bpe_sil_devtrain}}

  search_aligns = {}
  search_labels = {}
  for i in range(1):
    for variant_name, params in model_variants.items():
      name = "%s" % variant_name
      check_name = "" + build_alias(**params["config"])
      # check if name is according to my naming conventions
      assert name == check_name, "\n{} \t should be \n{}".format(name, check_name)

      num_epochs = [40, 80, 120, 150]

      # if name in [
      #   "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-like-labels.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.prev-target-in-readout.weight-drop0.0.new-pre.6pretrain-reps.bpe-sil-segs",
      #   "seg.bpe-with-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.bpe-sil-segs"
      # ]:
      #   num_epochs = [60, 80, 100, 120, 150]
      # if name in [
      #   "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.prev-target-in-readout.weight-drop0.0.new-pre.6pretrain-reps.bpe-sil-segs",
      #   "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-like-labels.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.prev-target-in-readout.weight-drop0.0.new-pre.6pretrain-reps.bpe-sil-segs",
      #   "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-like-labels.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.prev-target-in-readout.weight-drop0.0.new-pre.6pretrain-reps.bpe-sil-segs",
      #   "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-pooling.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.prev-target-in-readout.weight-drop0.0.new-pre.6pretrain-reps.bpe-sil-segs",
      #   "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-pooling.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.prev-target-in-readout.weight-drop0.0.new-pre.6pretrain-reps.bpe-sil-segs"
      # ]:
      #   num_epochs = [20, 40, 60, 80, 100, 120, 150]
      #
      # if "ctx-w-bias" in name:
      #   num_epochs = [80, 100, 120, 150]
      # if name in [
      #   "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.ctx-use-bias.all-segs",
      #   "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-l2.all-segs",
      #   "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-l2.ctx-use-bias.all-segs",
      # ]:
      #   num_epochs = [80, 100, 120, 150]


      # Currently different segments, depending on the label type
      segment_selection = params["config"].pop("segment_selection")
      time_red = int(np.prod(params["config"]["time_red"]))
      if segment_selection == "bpe-sil":
        train_segments = total_data["bpe-with-sil"]["train"]["time-red-%d" % time_red]["seq_filter_file"]
        cv_segments = total_data["bpe-with-sil"]["cv"]["time-red-%d" % time_red]["seq_filter_file"]
        devtrain_segments = total_data["bpe-with-sil"]["devtrain"]["time-red-%d" % time_red]["seq_filter_file"]
      elif segment_selection == "all":
        train_segments = total_data["bpe"]["train"]["time-red-%d" % time_red]["seq_filter_file"]
        cv_segments = total_data["bpe"]["cv"]["time-red-%d" % time_red]["seq_filter_file"]
        devtrain_segments = total_data["bpe"]["devtrain"]["time-red-%d" % time_red]["seq_filter_file"]
      elif segment_selection == "phonemes":
        train_segments = total_data["phonemes"]["train"]["time-red-%d" % time_red]["seq_filter_file"]
        cv_segments = total_data["phonemes"]["cv"]["time-red-%d" % time_red]["seq_filter_file"]
        devtrain_segments = total_data["phonemes"]["devtrain"]["time-red-%d" % time_red]["seq_filter_file"]
      else:
        raise NotImplementedError

      returnn_train_rasr_configs = {
        "train": write_config(*build_returnn_train_config(
          segment_file=train_segments, corpus_file=corpus_files["train"],
          feature_cache_path=feature_cache_files["train"]),
          alias="returnn_train_rasr_config"),
        "cv": write_config(
          *build_returnn_train_config(
            segment_file=cv_segments, corpus_file=corpus_files["train"],
            feature_cache_path=feature_cache_files["train"]), alias="returnn_cv_rasr_config"),
        "devtrain": write_config(
          *build_returnn_train_config(
            segment_file=devtrain_segments, corpus_file=corpus_files["train"],
            feature_cache_path=feature_cache_files["train"]),
          alias="returnn_devtrain_rasr_config"),
        "dev": write_config(
          *build_returnn_train_config(
            segment_file=None, corpus_file=corpus_files["dev"],
            feature_cache_path=feature_cache_files["dev"]), alias="returnn_dev_rasr_config"),
        "hub5e_01": write_config(
          *build_returnn_train_config(segment_file=None, corpus_file=corpus_files["hub5e_01"],
            feature_cache_path=feature_cache_files["hub5e_01"]), alias="returnn_hub5e_01_rasr_config"),
        "rt03s": write_config(
          *build_returnn_train_config(segment_file=None, corpus_file=corpus_files["rt03s"],
            feature_cache_path=feature_cache_files["rt03s"]), alias="returnn_rt03s_rasr_config")
      }

      # General data opts, which apply for all models
      train_data_opts = {
        "data": "train", "rasr_config_path": returnn_train_rasr_configs["train"],
        "rasr_nn_trainer_exe": rasr_nn_trainer}
      cv_data_opts = {
        "data": "cv", "rasr_config_path": returnn_train_rasr_configs["cv"],
        "rasr_nn_trainer_exe": rasr_nn_trainer}
      devtrain_data_opts = {
        "data": "devtrain", "rasr_config_path": returnn_train_rasr_configs["devtrain"],
        "rasr_nn_trainer_exe": rasr_nn_trainer}
      dev_data_opts = {
        "data": "dev", "rasr_config_path": returnn_train_rasr_configs["dev"],
        "rasr_nn_trainer_exe": rasr_nn_trainer}
      hub5e_01_data_opts = {
        "data": "hub5e_01", "rasr_config_path": returnn_train_rasr_configs["hub5e_01"], "rasr_nn_trainer_exe": rasr_nn_trainer}
      rt03s_data_opts = {
        "data": "rt03s", "rasr_config_path": returnn_train_rasr_configs["rt03s"], "rasr_nn_trainer_exe": rasr_nn_trainer}

      rasr_decoding_opts = dict(
        corpus_path=corpus_files["dev"],
        reduction_factors=int(np.prod(params["config"]["time_red"])),
        feature_cache_path=feature_cache_files["dev"], skip_silence=False, name=name)

      # Set more specific data opts for the individual model and label types
      if params["config"]["model_type"] == "glob":
        if params["config"]["label_type"] == "bpe":
          sos_idx = 0
          sil_idx = None
          target_num_labels = 1030
          vocab = bpe_vocab
          vocab["seq_postfix"] = [sos_idx]
          # train_data_opts["vocab"] = vocab
          # cv_data_opts["vocab"] = vocab
          # devtrain_data_opts["vocab"] = vocab
          dev_data_opts["vocab"] = vocab
          hub5e_01_data_opts["vocab"] = vocab
          rt03s_data_opts["vocab"] = vocab
          train_data_opts.update({
            "label_hdf": total_data["bpe"]["train"]["label_seqs"], "label_name": "bpe",
            "segment_file": train_segments})
          cv_data_opts.update({
            "label_hdf": total_data["bpe"]["cv"]["label_seqs"], "label_name": "bpe",
            "segment_file": cv_segments})
          devtrain_data_opts.update({
            "label_hdf": total_data["bpe"]["train"]["label_seqs"], "label_name": "bpe",
            "segment_file": devtrain_segments})
          params["config"]["label_name"] = "bpe"
          # sos_idx = 0
          # target_num_labels = 1030
        elif params["config"]["label_type"] == "bpe-with-sil":
          dev_data_opts["vocab"] = vocab
          train_data_opts.update({
            "label_hdf": total_data["bpe-with-sil"]["train"]["label_seqs"], "label_name": "bpe",
            "segment_file": train_segments})
          cv_data_opts.update({
            "label_hdf": total_data["bpe-with-sil"]["cv"]["label_seqs"], "label_name": "bpe",
            "segment_file": cv_segments})
          devtrain_data_opts.update({
            "label_hdf": total_data["bpe-with-sil"]["train"]["label_seqs"], "label_name": "bpe",
            "segment_file": devtrain_segments})
          params["config"]["label_name"] = "bpe"
          sos_idx = 1030
          sil_idx = 0
          target_num_labels = 1031
        elif params["config"]["label_type"] == "bpe-with-sil-split-sil":
          dev_data_opts["vocab"] = vocab
          train_data_opts.update({
            "label_hdf": total_data["bpe-with-sil-split-sil"]["train"]["label_seqs"], "label_name": "bpe",
            "segment_file": train_segments})
          cv_data_opts.update({
            "label_hdf": total_data["bpe-with-sil-split-sil"]["cv"]["label_seqs"], "label_name": "bpe",
            "segment_file": cv_segments})
          devtrain_data_opts.update({
            "label_hdf": total_data["bpe-with-sil-split-sil"]["train"]["label_seqs"], "label_name": "bpe",
            "segment_file": devtrain_segments})
          params["config"]["label_name"] = "bpe"
          sil_idx = 0
          sos_idx = 1030
          target_num_labels = 1031
        elif params["config"]["label_type"] == "phonemes-split-sil":
          train_data_opts.update({
            "label_hdf": total_data["phonemes-split-sil"]["train"]["label_seqs"], "label_name": "phonemes",
            "segment_file": train_segments})
          cv_data_opts.update({
            "label_hdf": total_data["phonemes-split-sil"]["cv"]["label_seqs"], "label_name": "phonemes",
            "segment_file": cv_segments})
          devtrain_data_opts.update({
            "label_hdf": total_data["phonemes-split-sil"]["train"]["label_seqs"], "label_name": "phonemes",
            "segment_file": devtrain_segments})
          params["config"]["label_name"] = "phonemes"
          sos_idx = 88
          sil_idx = 0
          target_num_labels = 89
        else:
          assert params["config"]["label_type"] == "phonemes"
          train_data_opts.update({
            "label_hdf": total_data["phonemes"]["train"]["label_seqs"],
            "label_name": "phonemes",
            "segment_file": train_segments
          })
          cv_data_opts.update({
            "label_hdf": total_data["phonemes"]["cv"]["label_seqs"], "label_name": "phonemes",
            "segment_file": cv_segments})
          devtrain_data_opts.update({
            "label_hdf": total_data["phonemes"]["train"]["label_seqs"], "label_name": "phonemes",
            "segment_file": devtrain_segments})
          params["config"]["label_name"] = "phonemes"
          sos_idx = 88
          sil_idx = 0
          target_num_labels = 89
        rasr_decoding_opts.update(
          dict(
            lexicon_path=bpe_sil_lexicon_path, label_unit="word", label_scorer_type="tf-attention",
            label_file_path=total_data["bpe"]["rasr_label_file"], lm_type="simple-history", use_lm_score=False,
            lm_scale=None, lm_file=None, lm_image=None, label_pruning=10.0, label_pruning_limit=12,
            word_end_pruning_limit=12, word_end_pruning=10.0, lm_lookahead_cache_size_high=None,
            lm_lookahead_cache_size_low=None, lm_lookahead_history_limit=None, lm_lookahead_scale=None, lm_lookahead=False,
            blank_label_index=1031, label_recombination_limit=-1))
      else:
        assert params["config"]["model_type"] == "seg"
        rasr_decoding_opts["label_recombination_limit"] = params["config"]["ctx_size"] if params["config"]["ctx_size"] != "inf" else -1
        if params["config"]["label_type"] == "bpe":
          sos_idx = 0
          sil_idx = None
          target_num_labels = 1030
          targetb_blank_idx = 1030
          vocab = bpe_vocab
          dev_data_opts["vocab"] = vocab
          hub5e_01_data_opts["vocab"] = vocab
          rt03s_data_opts["vocab"] = vocab
          train_align = total_data["bpe"]["train"]["time-red-%d" % time_red]["align"]
          cv_align = total_data["bpe"]["cv"]["time-red-%d" % time_red]["align"]
          train_data_opts.update({
            "segment_file": train_segments, "alignment": train_align})
          cv_data_opts.update({
            "segment_file": cv_segments, "alignment": cv_align})
          devtrain_data_opts.update({
            "segment_file": devtrain_segments, "alignment": train_align})
          rasr_decoding_opts.update(
            dict(
              lexicon_path=bpe_sil_lexicon_path, label_unit="word",
              label_file_path=total_data["bpe"]["rasr_label_file"],
              lm_type="simple-history", use_lm_score=False, lm_scale=None, lm_file=None, lm_image=None,
              label_pruning=10.0, label_pruning_limit=128, word_end_pruning_limit=128, word_end_pruning=10.0,
              lm_lookahead_cache_size_high=None, lm_lookahead_cache_size_low=None, lm_lookahead_history_limit=None,
              lm_lookahead_scale=None, lm_lookahead=False))
        elif params["config"]["label_type"].startswith("bpe"):
          if params["config"]["label_type"] == "bpe-with-sil-split-sil":
            train_align = total_data["bpe-with-sil-split-sil"]["train"]["time-red-%d" % time_red]["align"]
            cv_align = total_data["bpe-with-sil-split-sil"]["cv"]["time-red-%d" % time_red]["align"]
          elif params["config"]["label_type"] == "bpe-with-sil-split-silv2":
            train_align = Path("/work/asr3/zeyer/schmitt/old_models_and_analysis/old_bpe_sil_split_sil_aligns/train/AlignmentSplitSilenceJob.4dfiua41gqWb/output/out_align")
            cv_align = Path("/work/asr3/zeyer/schmitt/old_models_and_analysis/old_bpe_sil_split_sil_aligns/cv/AlignmentSplitSilenceJob.p0VY0atlAdkq/output/out_align")
          elif params["config"]["label_type"] == "bpe-sil-wo-sil":
            train_align = total_data["bpe-sil-wo-sil"]["train"]["time-red-%d" % time_red]["align"]
            cv_align = total_data["bpe-sil-wo-sil"]["cv"]["time-red-%d" % time_red]["align"]
          elif params["config"]["label_type"] == "bpe-sil-wo-sil-in-middle":
            train_align = total_data["bpe-sil-wo-sil-in-middle"]["train"]["time-red-%d" % time_red]["align"]
            cv_align = total_data["bpe-sil-wo-sil-in-middle"]["cv"]["time-red-%d" % time_red]["align"]
          else:
            assert params["config"]["label_type"] == "bpe-with-sil"
            train_align = total_data["bpe-with-sil"]["train"]["time-red-%d" % time_red]["align"]
            cv_align = total_data["bpe-with-sil"]["cv"]["time-red-%d" % time_red]["align"]
          sos_idx = 1030
          sil_idx = 0
          target_num_labels = 1031
          targetb_blank_idx = 1031
          vocab = bpe_vocab
          dev_data_opts["vocab"] = bpe_sil_vocab
          hub5e_01_data_opts["vocab"] = vocab
          rt03s_data_opts["vocab"] = vocab
          train_data_opts.update({
            "segment_file": train_segments, "alignment": train_align})
          cv_data_opts.update({
            "segment_file": cv_segments, "alignment": cv_align})
          devtrain_data_opts.update({
            "segment_file": devtrain_segments, "alignment": train_align})
          rasr_decoding_opts.update(dict(
            lexicon_path=bpe_sil_lexicon_path, label_unit="word",
            label_file_path=total_data["bpe-with-sil"]["rasr_label_file"],
            lm_type="simple-history", use_lm_score=False, lm_scale=None, lm_file=None, lm_image=None,
            label_pruning=10.0, label_pruning_limit=128, word_end_pruning_limit=128, word_end_pruning=10.0,
            lm_lookahead_cache_size_high=None, lm_lookahead_cache_size_low=None, lm_lookahead_history_limit=None,
            lm_lookahead_scale=None, lm_lookahead=False
          ))
        else:
          assert params["config"]["label_type"].startswith("phonemes")
          if params["config"]["label_type"] == "phonemes-split-sil":
            train_align = total_data["phonemes-split-sil"]["train"]["time-red-%d" % time_red]["align"]
            cv_align = total_data["phonemes-split-sil"]["cv"]["time-red-%d" % time_red]["align"]
          else:
            assert params["config"]["label_type"] == "phonemes"
            train_align = total_data["phonemes"]["train"]["time-red-%d" % time_red]["align"]
            cv_align = total_data["phonemes"]["cv"]["time-red-%d" % time_red]["align"]
          sos_idx = 88
          sil_idx = 0
          target_num_labels = 89
          targetb_blank_idx = 89
          vocab = bpe_vocab
          train_data_opts.update({
            "segment_file": train_segments, "alignment": train_align})
          cv_data_opts.update({
            "segment_file": cv_segments, "alignment": cv_align})
          devtrain_data_opts.update({
            "segment_file": devtrain_segments, "alignment": train_align})
          rasr_decoding_opts.update(dict(
            lexicon_path=phon_lexicon_wei, label_unit="phoneme",
            label_file_path=total_data["phonemes"]["rasr_label_file"], lm_type="ARPA",
            lm_file=zoltan_4gram_lm["file"], lm_image=zoltan_4gram_lm["image_wei"], lm_scale=0.8, use_lm_score=True,
            label_pruning=12.0, label_pruning_limit=50000, word_end_pruning_limit=5000, word_end_pruning=0.5,
            lm_lookahead_cache_size_high=None, lm_lookahead_cache_size_low=None,
            lm_lookahead_history_limit=None,
            lm_lookahead_scale=None, lm_lookahead=False))

      # update the config params with the specific info
      params["config"].update({
        "sos_idx": sos_idx, "target_num_labels": target_num_labels, "vocab": vocab, "sil_idx": sil_idx
      })
      rasr_decoding_opts.update(dict(start_label_index=sos_idx))
      # in case of segmental/transducer model, we need to set the blank index
      if params["config"]["model_type"] == "seg":
        params["config"].update({
          "targetb_blank_idx": targetb_blank_idx,
        })
        rasr_decoding_opts.update(dict(blank_label_index=targetb_blank_idx))
      # choose the config class depending on the model type
      config_class = {
        "seg": TransducerSWBExtendedConfig, "glob": GlobalEncoderDecoderConfig}.get(params["config"]["model_type"])
      config_params = copy.deepcopy(params["config"])

      # these parameters are not needed for the config class
      del config_params["label_type"]
      del config_params["model_type"]

      # initialize returnn config
      train_config_obj = config_class(
        task="train",
        post_config={"cleanup_old_models": {"keep_last_n": 1, "keep_best_n": 1, "keep": num_epochs}},
        train_data_opts=train_data_opts,
        cv_data_opts=cv_data_opts,
        devtrain_data_opts=devtrain_data_opts,
        **config_params).get_config()

      # start standard returnn training
      checkpoints, train_config, train_job = run_training(train_config_obj, mem_rqmt=24, time_rqmt=30, num_epochs=num_epochs,
                                               name=name, alias_suffix="train")

      # for each previously specified epoch, run decoding
      # if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.all-segs":
      #   num_epochs = [150]
      #   checkpoints = {150: Checkpoint(index_path=Path("/u/schmitt/experiments/transducer/alias/glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-weight-feedback.no-l2.ctx-use-bias.all-segs/train/output/models/epoch.150.index"))}
      for epoch in num_epochs:
        if epoch in checkpoints and (epoch == 150 or (name in [
          "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.seg-neural-length-model-in_label+mean-pool.bpe-sil-segs",
          "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-pooling.no-sil-in-ctx.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs",
          "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-pooling.no-sil-in-ctx.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs",
          "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-pooling.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs",
          "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-pooling.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs",
          "seg.bpe-sil-wo-sil-in-middle.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs",
          "seg.bpe-sil-wo-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs",
          "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs",
          "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs"] and epoch == 40)):
          checkpoint = checkpoints[epoch]
          if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs":
            checkpoint = Checkpoint(
              index_path=Path(
                "/work/asr3/zeyer/schmitt/old_models_and_analysis/seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs/train/output/models/epoch.150.index",
                hash_overwrite=(train_job, "models/epoch.150.index"), creator=train_job
              ))

          if params["config"]["model_type"] == "seg":
            # for bpe + sil model use additional RETURNN decoding as sanity check
            if params["config"]["label_type"].startswith("bpe"):
              if params["config"]["label_type"].startswith("bpe-with-sil") or params["config"]["label_type"].startswith("bpe-sil-wo-sil"):
                config_params["vocab"]["vocab_file"] = total_data["bpe-with-sil"]["json_vocab"]

              if name in [
                "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs"]:
                for concat_num in concat_jobs:
                  for corpus_name in concat_jobs[concat_num]:
                    if corpus_name == "hub5e_00":
                      data_opts = copy.deepcopy(dev_data_opts)
                      stm_job = hub5e_00_stm_job
                    elif corpus_name == "hub5e_01":
                      data_opts = copy.deepcopy(hub5e_01_data_opts)
                      stm_job = hub5e_01_stm_job
                    else:
                      assert corpus_name == "rt03s"
                      data_opts = copy.deepcopy(rt03s_data_opts)
                      stm_job = rt03s_stm_job
                    data_opts.update({
                      "concat_seqs": True, "concat_seq_tags": concat_jobs[concat_num][corpus_name].out_concat_seq_tags,
                      "concat_seq_lens": concat_jobs[concat_num][corpus_name].out_orig_seq_lens_py})
                    search_config = config_class(
                      task="search", beam_size=12, search_data_opts=data_opts, search_use_recomb=False,
                      target="bpe", length_scale=1., **config_params)

                    ctm_results = run_bpe_returnn_decoding(
                      returnn_config=search_config.get_config(),
                      checkpoint=checkpoint, stm_job=stm_job, num_epochs=epoch,
                      name=name, dataset_key=corpus_name, concat_seqs=True,
                      alias_addon="concat-%s_beam-%s" % (concat_num, 12), mem_rqmt=4,
                      stm_path=concat_jobs[concat_num][corpus_name].out_stm)
                    run_eval(
                      ctm_file=ctm_results, reference=concat_jobs[concat_num][corpus_name].out_stm, name=name,
                      dataset_key="%s_concat-%s" % (corpus_name, concat_num), num_epochs=epoch, alias_addon="_beam-%s" % 12)

              for beam_size in [12]:
                for use_recomb in [True, False]:
                  for length_scale in [1., .5]:
                    alias_addon = "returnn_%srecomb_length-scale-%s_beam-%s" % ("" if use_recomb else "no-", length_scale, beam_size)
                    # standard returnn decoding
                    search_config = config_class(
                      task="search", search_data_opts=dev_data_opts, target="bpe", search_use_recomb=use_recomb,
                      beam_size=beam_size, length_scale=length_scale, **config_params)
                    ctm_results = run_bpe_returnn_decoding(
                      returnn_config=search_config.get_config(), checkpoint=checkpoint,
                      stm_job=hub5e_00_stm_job, num_epochs=epoch, name=name,
                      dataset_key="dev", alias_addon=alias_addon)
                    run_eval(ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name=name,
                             dataset_key="dev", num_epochs=epoch, alias_addon=alias_addon)

                    search_error_data_opts = copy.deepcopy(cv_data_opts)
                    alignment_hdf = search_error_data_opts.pop("alignment")
                    segment_file = search_error_data_opts.pop("segment_file")
                    search_error_data_opts["vocab"] = dev_data_opts["vocab"]
                    dump_search_config = config_class(search_use_recomb=True if use_recomb else False, task="search", target="bpe", beam_size=beam_size,
                      search_data_opts=search_error_data_opts, dump_output=True, length_scale=length_scale, **config_params)
                    feed_config_load = config_class(
                      task="train",
                      length_scale=length_scale,
                      post_config={"cleanup_old_models": {"keep_last_n": 1, "keep_best_n": 1, "keep": num_epochs}},
                      train_data_opts=train_data_opts,
                      cv_data_opts=cv_data_opts,
                      devtrain_data_opts=devtrain_data_opts,
                      **config_params).get_config()
                    feed_config_load.config["load"] = checkpoint
                    # alias_addon = "_returnn_search_errors_%srecomb_length-scale-%s_beam-%s" % ("" if use_recomb else "no-", length_scale, beam_size)
                    calculate_search_errors(checkpoint=checkpoint, search_config=dump_search_config, stm_job=cv_stm_job,
                      train_config=feed_config_load, name=name, segment_path=segment_file, ref_targets=alignment_hdf,
                      label_name="alignment", model_type="seg", blank_idx=targetb_blank_idx, rasr_nn_trainer_exe=rasr_nn_trainer,
                      rasr_config=returnn_train_rasr_configs["cv"], alias_addon=alias_addon, epoch=epoch, dataset_key="cv", length_norm=False)

              if epoch in [150]:
                length_scales = [1.]
                if name in [
                  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs",
                  "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs",
                  "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs"]:
                  length_scales += [.01, .1, .3, .5, .7, .9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.]
                for length_scale in length_scales:
                  max_seg_lens = [20]
                  if name in [
                    "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs",
                    "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs"] and length_scale == 1.:
                    max_seg_lens += [10, 15, 25, 30]
                  for max_seg_len in max_seg_lens:
                    vit_recombs = [True, False]
                    if max_seg_len != 20 or length_scale != 1.:
                      vit_recombs = [True]
                    for vit_recomb in vit_recombs:
                      if length_scale == 1.:
                        length_norms = [False]
                      else:
                        length_norms = [False]
                      for length_norm in length_norms:
                        if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs":
                          eval_corpus_names = ["dev", "rt03s", "hub5e_01"] if length_scale in [1., 0.7] and max_seg_len == 20 else ["dev"]

                        else:
                          eval_corpus_names = ["dev"]
                        for eval_corpus_name in eval_corpus_names:
                          # Config for compiling model for RASR
                          compile_config = config_class(task="eval", feature_stddev=3., length_scale=length_scale, **config_params)

                          if length_scale == 0.0:
                            mem_rqmt = 24
                          else:
                            mem_rqmt = 12

                          if vit_recomb:
                            blank_update_history = True
                            allow_label_recombination = True
                            allow_word_end_recombination = True
                          else:
                            blank_update_history = True
                            allow_label_recombination = False
                            allow_word_end_recombination = False

                          # RASR NEURAL LENGTH DECODING

                          alias_addon = "rasr_limit12_pruning12.0_%s-recomb_neural-length_max-seg-len-%s_length-scale-%s" % ("vit" if vit_recomb else "no", max_seg_len, length_scale)
                          new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                          new_rasr_decoding_opts.update(
                            dict(
                              word_end_pruning_limit=12, word_end_pruning=12.0, label_pruning_limit=12, label_pruning=12.0,
                              corpus_path=corpus_files[eval_corpus_name], feature_cache_path=feature_cache_files[eval_corpus_name]))
                          ctm_results = run_rasr_decoding(
                            segment_path=None, mem_rqmt=mem_rqmt, simple_beam_search=True, length_norm=length_norm,
                            full_sum_decoding=False, blank_update_history=blank_update_history,
                            allow_word_end_recombination=allow_word_end_recombination, loop_update_history=True,
                            allow_label_recombination=allow_label_recombination, max_seg_len=max_seg_len, debug=False,
                            compile_config=compile_config.get_config(), alias_addon=alias_addon,
                            rasr_exe_path=rasr_flf_tool, model_checkpoint=checkpoint, num_epochs=epoch,
                            time_rqmt=24, gpu_rqmt=1, **new_rasr_decoding_opts)
                          run_eval(ctm_file=ctm_results, reference=eval_ref_files[eval_corpus_name], name=name,
                                   dataset_key=eval_corpus_name, num_epochs=epoch, alias_addon=alias_addon)

                          # cv_realignment = run_rasr_realignment(
                          #   compile_config=compile_config.get_config(), alias_addon=alias_addon,
                          #   segment_path=cv_segments, loop_update_history=True, blank_update_history=True, name=name,
                          #   corpus_path=corpus_files["train"], lexicon_path=bpe_phon_lexicon_path if params["config"]["label_type"] == "bpe" else bpe_sil_phon_lexicon_path,
                          #   allophone_path=total_data[params["config"]["label_type"]]["allophones"],
                          #   state_tying_path=total_data[params["config"]["label_type"]]["state_tying"],
                          #   feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                          #   label_file=total_data[params["config"]["label_type"]]["rasr_label_file"], label_pruning=12.0,
                          #   label_pruning_limit=5000, label_recombination_limit=-1, blank_label_index=targetb_blank_idx,
                          #   model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
                          #   rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
                          #   rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1, time_rqmt=48,
                          #   blank_allophone_state_idx=4119 if params["config"]["label_type"] == "bpe" else 4123,
                          #   max_segment_len=90, mem_rqmt=12, length_norm=False, data_key="cv")

                          if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs" and length_scale == 1.:
                            train_realignment = run_rasr_realignment(
                              compile_config=compile_config.get_config(),
                              alias_addon=alias_addon, segment_path=train_segments, loop_update_history=True,
                              blank_update_history=True, name=name, corpus_path=corpus_files["train"],
                              lexicon_path=bpe_phon_lexicon_path if params["config"]["label_type"] == "bpe" else bpe_sil_phon_lexicon_path,
                              allophone_path=total_data[params["config"]["label_type"]]["allophones"],
                              state_tying_path=total_data[params["config"]["label_type"]]["state_tying"],
                              feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                              label_file=total_data[params["config"]["label_type"]]["rasr_label_file"], label_pruning=12.0,
                              label_pruning_limit=5000, label_recombination_limit=-1, blank_label_index=targetb_blank_idx,
                              model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
                              rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
                              rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1, time_rqmt=96,
                              blank_allophone_state_idx=4119 if params["config"]["label_type"] == "bpe" else 4123,
                              max_segment_len=20, mem_rqmt=12, length_norm=False, data_key="train_max-seg-len-20")
                            train_realignment = run_rasr_realignment(compile_config=compile_config.get_config(),
                              alias_addon=alias_addon, segment_path=train_segments, loop_update_history=True,
                              blank_update_history=True, name=name, corpus_path=corpus_files["train"],
                              lexicon_path=bpe_phon_lexicon_path if params["config"][
                                                                      "label_type"] == "bpe" else bpe_sil_phon_lexicon_path,
                              allophone_path=total_data[params["config"]["label_type"]]["allophones"],
                              state_tying_path=total_data[params["config"]["label_type"]]["state_tying"],
                              feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                              label_file=total_data[params["config"]["label_type"]]["rasr_label_file"], label_pruning=12.0,
                              label_pruning_limit=5000, label_recombination_limit=-1, blank_label_index=targetb_blank_idx,
                              model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
                              rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
                              rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1, time_rqmt=96,
                              blank_allophone_state_idx=4119 if params["config"]["label_type"] == "bpe" else 4123,
                              max_segment_len=40, mem_rqmt=12, length_norm=False, data_key="train_max-seg-len-40")
                            train_realignment = run_rasr_realignment(compile_config=compile_config.get_config(),
                              alias_addon=alias_addon, segment_path=train_segments, loop_update_history=True,
                              blank_update_history=True, name=name, corpus_path=corpus_files["train"],
                              lexicon_path=bpe_phon_lexicon_path if params["config"][
                                                                      "label_type"] == "bpe" else bpe_sil_phon_lexicon_path,
                              allophone_path=total_data[params["config"]["label_type"]]["allophones"],
                              state_tying_path=total_data[params["config"]["label_type"]]["state_tying"],
                              feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                              label_file=total_data[params["config"]["label_type"]]["rasr_label_file"], label_pruning=12.0,
                              label_pruning_limit=5000, label_recombination_limit=-1, blank_label_index=targetb_blank_idx,
                              model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
                              rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
                              rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1, time_rqmt=96,
                              blank_allophone_state_idx=4119 if params["config"]["label_type"] == "bpe" else 4123,
                              max_segment_len=80, mem_rqmt=12, length_norm=False, data_key="train_max-seg-len-80")
                            train_realignment = run_rasr_realignment(compile_config=compile_config.get_config(),
                              alias_addon=alias_addon, segment_path=train_segments, loop_update_history=True,
                              blank_update_history=True, name=name, corpus_path=corpus_files["train"],
                              lexicon_path=bpe_phon_lexicon_path if params["config"][
                                                                      "label_type"] == "bpe" else bpe_sil_phon_lexicon_path,
                              allophone_path=total_data[params["config"]["label_type"]]["allophones"],
                              state_tying_path=total_data[params["config"]["label_type"]]["state_tying"],
                              feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                              label_file=total_data[params["config"]["label_type"]]["rasr_label_file"], label_pruning=12.0,
                              label_pruning_limit=5000, label_recombination_limit=-1, blank_label_index=targetb_blank_idx,
                              model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red,
                              rasr_nn_trainer_exe_path=rasr_nn_trainer, start_label_index=sos_idx,
                              rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=targetb_blank_idx + 1, time_rqmt=96,
                              blank_allophone_state_idx=4119 if params["config"]["label_type"] == "bpe" else 4123,
                              max_segment_len=100, mem_rqmt=12, length_norm=False, data_key="train_max-seg-len-100")

                          search_error_opts = copy.deepcopy(cv_data_opts)
                          cv_align = search_error_opts.pop("alignment")
                          cv_segments = search_error_opts.pop("segment_file")
                          feed_config_load = config_class(task="train", length_scale=length_scale,
                            post_config={"cleanup_old_models": {"keep_last_n": 1, "keep_best_n": 1, "keep": num_epochs}},
                            train_data_opts=train_data_opts, cv_data_opts=cv_data_opts, devtrain_data_opts=devtrain_data_opts,
                            **config_params).get_config()
                          feed_config_load.config["load"] = checkpoint
                          if name == "seg.bpe-with-sil-split-silv2.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-pooling.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs":
                            feed_config_load.config["network"]["label_model"]["unit"].update({
                              "label_log_prob": {
                                "class": "copy", "from": ["sil_log_prob", "label_log_prob1"], },
                              "label_log_prob1": {
                                "class": "combine", "from": ["label_log_prob0", "non_sil_log_prob"], "kind": "add", },
                              "label_prob": {
                                "activation": "exp", "class": "activation", "from": ["label_log_prob"],
                                "is_output_layer": True, "loss": "ce",
                                "loss_opts": {"focal_loss_factor": 0.0, "label_smoothing": 0.1},
                                "target": "label_ground_truth", },
                            })
                          new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                          new_rasr_decoding_opts.update(
                            dict(word_end_pruning_limit=12, word_end_pruning=12.0, label_pruning_limit=12, label_pruning=12.0,
                                 corpus_path=corpus_files["train"], feature_cache_path=feature_cache_files["train"]))
                          search_align, ctm_results = calc_rasr_search_errors(
                            segment_path=cv_segments, mem_rqmt=mem_rqmt, simple_beam_search=True, length_norm=length_norm,
                            ref_align=cv_align, num_classes=targetb_blank_idx+1, num_epochs=epoch,
                            blank_idx=targetb_blank_idx, rasr_nn_trainer_exe=rasr_nn_trainer,
                            extern_sprint_rasr_config=returnn_train_rasr_configs["cv"],
                            train_config=feed_config_load, loop_update_history=True,
                            full_sum_decoding=False, blank_update_history=blank_update_history,
                            allow_word_end_recombination=allow_word_end_recombination, allow_label_recombination=allow_label_recombination,
                            max_seg_len=max_seg_len, debug=False,
                            compile_config=compile_config.get_config(),
                            alias_addon=alias_addon, rasr_exe_path=rasr_flf_tool,
                            model_checkpoint=checkpoint, time_rqmt=48, gpu_rqmt=1,
                            model_type="seg", label_name="alignment", **new_rasr_decoding_opts)
                          run_eval(ctm_file=ctm_results, reference=cv_stm_job.out_stm_path, name=name,
                                   dataset_key="cv", num_epochs=epoch, alias_addon=alias_addon)

                          calc_align_stats(alignment=search_align, blank_idx=targetb_blank_idx, seq_filter_file=cv_segments,
                            alias=name + "/" + alias_addon + "/cv_search_align_stats_epoch-%s" % epoch)

                          if "pooling-att" not in name:
                            for seq_tag in [
                              "switchboard-1/sw02102A/sw2102A-ms98-a-0092",
                              "switchboard-1/sw02102A/sw2102A-ms98-a-0090", "switchboard-1/sw02025A/sw2025A-ms98-a-0035",
                              "switchboard-1/sw02102A/sw2102A-ms98-a-0002", "switchboard-1/sw02038B/sw2038B-ms98-a-0069"
                            ]:

                              group_alias = name + "/neural-length_analysis_epoch-%s_length-scale-%s_%s-recomb_max-seg-len-%s" % (epoch, length_scale, "no" if not vit_recomb else "vit", max_seg_len)

                              for align_alias, align in zip(["ground-truth", "search", "realign"],
                                                            [cv_align, search_align]):
                                vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else \
                                bpe_sil_vocab["vocab_file"]
                                dump_att_weights_job = DumpAttentionWeightsJob(returnn_config=feed_config_load,
                                  model_type="seg", rasr_config=returnn_train_rasr_configs["cv"],
                                  blank_idx=targetb_blank_idx, label_name="alignment", rasr_nn_trainer_exe=rasr_nn_trainer,
                                  hdf_targets=align, seq_tag=seq_tag, )
                                dump_att_weights_job.add_alias(
                                  group_alias + "/" + seq_tag.replace("/", "_") + "/att_weights_%s_%s" % (
                                  align_alias, epoch))
                                tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

                                plot_weights_job = PlotAttentionWeightsJob(data_path=dump_att_weights_job.out_data,
                                                                           blank_idx=targetb_blank_idx,
                                                                           json_vocab_path=vocab_file, time_red=6,
                                                                           seq_tag=seq_tag)
                                plot_weights_job.add_alias(
                                  group_alias + "/" + seq_tag.replace("/", "_") + "/plot_att_weights_%s_%s" % (
                                  align_alias, epoch))
                                tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)

                              compare_aligns_job = CompareAlignmentsJob(hdf_align1=cv_align, hdf_align2=search_align,
                                seq_tag=seq_tag, blank_idx1=targetb_blank_idx, blank_idx2=targetb_blank_idx,
                                vocab1=vocab_file, vocab2=vocab_file, name1="ground_truth", name2="search_alignment")
                              compare_aligns_job.add_alias(
                                group_alias + "/" + seq_tag.replace("/", "_") + "/search-align-compare")
                              tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)

              if epoch in [150] and name in [
                "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs",
                "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs",
                "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs",
                # "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.pooling-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.bpe-sil-segs"
              ] and params["config"]["length_model_type"] == "frame":
                length_scales = [1., 0.0]
                if name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs":
                  length_scales += [.01, .1, .3, .5, .7, .9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.]
                for length_scale in length_scales:
                  if params["config"]["label_type"] in ["bpe", "bpe_sil_wo_sil"]:
                    max_seg_lens = [25]
                  else:
                    max_seg_lens = [20]
                  for max_seg_len in max_seg_lens:
                    limits = [12]
                    if name in [
                      "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs",
                      "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs"] and length_scale in [1., 0.0]:
                      limits += [5000]
                    for limit in limits:
                      vit_recombs = [True, False]
                      if max_seg_len in [10, 30, 40]:
                        vit_recombs = [True]
                      if limit == 5000:
                        vit_recombs = [True]
                      for vit_recomb in vit_recombs:
                        if length_scale == 0.0:
                          length_norms = [True]
                          beam_searches = [True]
                        elif max_seg_len in [10, 30, 40]:
                          length_norms = [False]
                          beam_searches = [True]
                        else:
                          length_norms = [True, False]
                          beam_searches = [True]
                        for length_norm in length_norms:
                          for beam_search in beam_searches:
                            seg_selections = ["all"]
                            for seg_selection in seg_selections:
                              global_length_vars = [None]
                              silence_splits = [False]
                              net_types = ["default"]
                              if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs" and length_scale == 0.0:
                                net_types += ["global_import"]
                              elif name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs" and length_scale == 0.0:
                                net_types += ["global_import"]
                                silence_splits += [True]
                              for net_type in net_types:
                                if name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs" and net_type != "global_import" and limit == 5000:
                                  continue
                                for glob_length_var in global_length_vars:
                                  for silence_split in silence_splits:
                                    # parameters to use label dep length model
                                    label_dep_params = copy.deepcopy(config_params)
                                    label_dep_params.pop("length_model_type")
                                    label_dep_params.pop("max_seg_len")
                                    label_dep_params.update(dict(
                                      length_scale=length_scale, label_dep_length_model=True, length_model_type="seg-static",
                                      label_dep_means=total_data[params["config"]["label_type"]]["train"]["time-red-%s" % time_red]["label_dep_mean_lens"]))
                                    # compile config for RASR
                                    compile_config = config_class(
                                      task="eval", feature_stddev=3., max_seg_len=max_seg_len,
                                      network_type=net_type,
                                      global_length_var=glob_length_var,
                                      **label_dep_params).get_config()
                                    feed_config_load = config_class(
                                      task="eval", pretrain=False, max_seg_len=max_seg_len,
                                      network_type=net_type,
                                      global_length_var=glob_length_var,
                                      **label_dep_params).get_config()

                                    # print(name)

                                    if net_type == "global_import":
                                      if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs":
                                        checkpoint = Checkpoint(index_path=Path(
                                          "/u/schmitt/experiments/transducer/alias/glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs/train/output/models/epoch.150.index"))
                                      else:
                                        if silence_split:
                                          checkpoint = Checkpoint(index_path=Path(
                                            "/u/schmitt/experiments/transducer/alias/glob.best-model.bpe-with-sil-split-sil.time-red6.am2048.1pretrain-reps.ctx-use-bias.pretrain-like-seg.bpe-sil-segs/train/output/models/epoch.150.index"))
                                        else:
                                          checkpoint = Checkpoint(index_path=Path(
                                            "/u/schmitt/experiments/transducer/alias/glob.best-model.bpe-with-sil.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.bpe-sil-segs/train/output/models/epoch.150.index"))
                                    else:
                                      checkpoint = checkpoints[epoch]

                                    feed_config_load.config["load"] = checkpoint

                                    # RASR LABEL DEP LENGTH DECODING

                                    if vit_recomb:
                                      blank_update_history = True
                                      allow_label_recombination = True
                                      allow_word_end_recombination = True
                                      mem_rqmt = 64
                                    else:
                                      blank_update_history = True
                                      allow_label_recombination = False
                                      allow_word_end_recombination = False
                                      mem_rqmt = 64

                                    if not vit_recomb and length_scale in [0.0, 0.01, 0.1, 0.5] and max_seg_len == 20:
                                      if not length_norm:
                                        label_pruning = 1.0
                                      else:
                                        label_pruning = 0.1
                                      mem_rqmt = 64
                                    else:
                                      label_pruning = 12.0

                                    alias_addon = "rasr_limit%s_pruning%s_%s-recomb_label-dep-length-%s_max-seg-len-%s_length-scale-%s%s%s_%s-segments%s%s" % (limit, label_pruning, "vit" if vit_recomb else "no", "glob-var-%s" % glob_length_var, max_seg_len, length_scale, "" if not length_norm else "_length-norm", "" if not beam_search else "_beam_search", "all" if seg_selection == "all" else "red", "" if net_type == "default" else "_" + net_type, "" if not silence_split else "_split-sil")
                                    new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                                    new_rasr_decoding_opts.update(
                                      dict(
                                        word_end_pruning_limit=limit,
                                        word_end_pruning=label_pruning,
                                        label_pruning_limit=limit,
                                        label_pruning=label_pruning))
                                    if net_type != "default":
                                      if "sil" not in params["config"]["label_type"]:
                                        new_rasr_decoding_opts.update(dict(
                                          label_file_path=Path(
                                            "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos")
                                        ))
                                      else:
                                        new_rasr_decoding_opts.update(dict(label_file_path=Path(
                                          "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_sil_label_file_w_add_eos")))
                                    ctm_results = run_rasr_decoding(
                                      segment_path=None if seg_selection == "all" else Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/hub5_00_10div"),
                                      mem_rqmt=128 if limit == 5000 else mem_rqmt, simple_beam_search=False if not beam_search else True, length_norm=length_norm,
                                      full_sum_decoding=False, blank_update_history=blank_update_history,
                                      allow_word_end_recombination=allow_word_end_recombination, loop_update_history=True,
                                      allow_label_recombination=allow_label_recombination, max_seg_len=max_seg_len, debug=False,
                                      compile_config=compile_config, alias_addon=alias_addon,
                                      rasr_exe_path=rasr_flf_tool, model_checkpoint=checkpoint,
                                      num_epochs=epoch, time_rqmt=72 if limit == 5000 else 24, gpu_rqmt=1, **new_rasr_decoding_opts)
                                    run_eval(ctm_file=ctm_results,
                                             reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm") if seg_selection == "all" else Path("/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/hub5_00_stm_10div"),
                                             name=name,
                                             dataset_key="dev", num_epochs=epoch, alias_addon=alias_addon)

                                    # cv_realignment = run_rasr_realignment(
                                    #   compile_config=compile_config,
                                    #   alias_addon=alias_addon, segment_path=cv_segments, loop_update_history=True,
                                    #   blank_update_history=True, name=name, corpus_path=corpus_files["train"],
                                    #   lexicon_path=bpe_phon_lexicon_path if params["config"]["label_type"] == "bpe" else bpe_sil_phon_lexicon_path,
                                    #   allophone_path=total_data[params["config"]["label_type"]]["allophones"],
                                    #   state_tying_path=total_data[params["config"]["label_type"]]["state_tying"],
                                    #   feature_cache_path=feature_cache_files["train"], num_epochs=epoch,
                                    #   label_file=total_data[params["config"]["label_type"]]["rasr_label_file"],
                                    #   label_pruning=12.0, label_pruning_limit=5000, label_recombination_limit=-1,
                                    #   blank_label_index=targetb_blank_idx, model_checkpoint=checkpoint, context_size=-1,
                                    #   reduction_factors=time_red, rasr_nn_trainer_exe_path=rasr_nn_trainer,
                                    #   start_label_index=sos_idx, rasr_am_trainer_exe_path=rasr_am_trainer,
                                    #   num_classes=targetb_blank_idx + 1, time_rqmt=48,
                                    #   blank_allophone_state_idx=4119 if params["config"]["label_type"] == "bpe" else 4123,
                                    #   max_segment_len=max_seg_len, mem_rqmt=12, length_norm=length_norm, data_key="cv")

                                    search_error_opts = copy.deepcopy(cv_data_opts)
                                    cv_align = search_error_opts.pop("alignment")
                                    cv_segments = search_error_opts.pop("segment_file")
                                    new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
                                    new_rasr_decoding_opts.update(
                                      dict(word_end_pruning_limit=limit, word_end_pruning=label_pruning, label_pruning_limit=limit, label_pruning=label_pruning,
                                        corpus_path=corpus_files["train"], feature_cache_path=feature_cache_files["train"]))
                                    if net_type != "default":
                                      if "sil" not in params["config"]["label_type"]:
                                        new_rasr_decoding_opts.update(dict(label_file_path=Path(
                                          "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_label_file_w_add_eos")))
                                      else:
                                        new_rasr_decoding_opts.update(dict(label_file_path=Path(
                                          "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_sil_label_file_w_add_eos")))

                                    # alias_addon = "_rasr_search_errors_limit12_pruning12.0_%s-recomb_label-dep-length_max-seg-len-%s_length-scale-%s" % ("vit" if vit_recomb else "no", max_seg_len, length_scale)
                                    search_align, ctm_results = calc_rasr_search_errors(
                                      segment_path=cv_segments, mem_rqmt=128 if limit == 5000 else mem_rqmt,
                                      simple_beam_search=False if not beam_search else True, length_norm=length_norm,
                                      ref_align=cv_align, num_classes=targetb_blank_idx+1, num_epochs=epoch, blank_idx=targetb_blank_idx,
                                      rasr_nn_trainer_exe=rasr_nn_trainer,
                                      extern_sprint_rasr_config=returnn_train_rasr_configs["cv"],
                                      train_config=feed_config_load, loop_update_history=True, full_sum_decoding=False,
                                      blank_update_history=blank_update_history, allow_word_end_recombination=allow_word_end_recombination,
                                      allow_label_recombination=allow_label_recombination, max_seg_len=max_seg_len, debug=False,
                                      compile_config=compile_config, alias_addon=alias_addon,
                                      rasr_exe_path=rasr_flf_tool, model_checkpoint=checkpoint, time_rqmt=72 if limit == 5000 else 24,
                                      gpu_rqmt=1, model_type="seg_lab_dep" if net_type == "default" else "global-import", label_name="alignment", **new_rasr_decoding_opts)
                                    run_eval(ctm_file=ctm_results, reference=cv_stm_job.out_stm_path, name=name,
                                             dataset_key="cv", num_epochs=epoch, alias_addon=alias_addon)

                                    calc_align_stats(
                                      alignment=search_align, blank_idx=targetb_blank_idx,
                                      seq_filter_file=cv_segments, alias=name + "/" + alias_addon + "/cv_search_align_stats_epoch-%s" % epoch)

                                    if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.all-segs":
                                      if alias_addon == "rasr_limit12_pruning12.0_no-recomb_label-dep-length-glob-var-None_max-seg-len-25_length-scale-0.0_length-norm_beam_search_all-segments_global_import":
                                        dump_non_blanks_job = DumpNonBlanksFromAlignmentJob(search_align,
                                          blank_idx=targetb_blank_idx)
                                        dump_non_blanks_job.add_alias("dump_non_blanks_" + alias_addon)
                                        search_aligns["global_import_segmental"] = search_align
                                        search_labels["global_import_segmental"] = dump_non_blanks_job.out_labels
                                    if name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs":
                                      if alias_addon == "rasr_limit12_pruning12.0_vit-recomb_label-dep-length-glob-var-None_max-seg-len-20_length-scale-0.0_length-norm_beam_search_all-segments_global_import_split-sil":
                                        dump_non_blanks_job = DumpNonBlanksFromAlignmentJob(search_align,
                                          blank_idx=targetb_blank_idx)
                                        dump_non_blanks_job.add_alias("dump_non_blanks_" + alias_addon)
                                        search_aligns["global_import_segmental_w_split_sil"] = search_align
                                        search_labels["global_import_segmental_w_split_sil"] = dump_non_blanks_job.out_labels

                                    # if alias_addon == "rasr_limit12_pruning0.1_no-recomb_label-dep-length-glob-var-None_max-seg-len-20_length-scale-0.0_length-norm_beam_search_all-segments_global_import_w_feedback":
                                    #   for search_align_alias, other_search_align in search_aligns.items():
                                    #     calc_search_err_job = CalcSearchErrorJob(
                                    #       returnn_config=train_config,
                                    #       rasr_config=returnn_train_rasr_configs["cv"],
                                    #       rasr_nn_trainer_exe=rasr_nn_trainer,
                                    #       segment_file=cv_segments,
                                    #       blank_idx=targetb_blank_idx,
                                    #       search_targets=other_search_align,
                                    #       ref_targets=cv_align, label_name="alignment",
                                    #       model_type="seg_lab_dep",
                                    #       max_seg_len=max_seg_len if max_seg_len is not None else -1,
                                    #       length_norm=length_norm)
                                    #     calc_search_err_job.add_alias(
                                    #       name + ("/%s/search_errors_%s_%d" % (alias_addon, search_align_alias, epoch)))
                                    #     alias = calc_search_err_job.get_one_alias()
                                    #     tk.register_output(alias + "search_errors", calc_search_err_job.out_search_errors)

                                    if limit == 12 and seg_selection == "all" and "pooling-att" not in name:
                                      for seq_tag in [
                                        "switchboard-1/sw02102A/sw2102A-ms98-a-0092",
                                        "switchboard-1/sw02022A/sw2022A-ms98-a-0002",
                                        "switchboard-1/sw02102A/sw2102A-ms98-a-0090",
                                        "switchboard-1/sw02025A/sw2025A-ms98-a-0035",
                                        "switchboard-1/sw02102A/sw2102A-ms98-a-0002",
                                        "switchboard-1/sw02038B/sw2038B-ms98-a-0069"
                                      ]:
                                        group_alias = name + "/analysis_epoch-%s_length-scale-%s-%s_%s-recomb_max-seg-len-%s%s%s%s%s" % (epoch, length_scale, "glob-var-%s" % glob_length_var, "no" if not vit_recomb else "vit", max_seg_len,"" if not length_norm else "_length-norm", "" if not beam_search else "_beam_search", "" if net_type == "default" else "_" + net_type, "" if not silence_split else "_split-sil")

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
                                        feed_config_load = config_class(task="train",
                                                                        post_config={
                                                                          "cleanup_old_models": {"keep_last_n": 1, "keep_best_n": 1,
                                                                                                 "keep": num_epochs}},
                                                                        network_type=net_type,
                                                                        train_data_opts=train_data_opts, cv_data_opts=cv_data_opts,
                                                                        devtrain_data_opts=devtrain_data_opts,
                                                                        max_seg_len=max_seg_len, **label_dep_params).get_config()
                                        feed_config_load.config["load"] = checkpoint
                                        feed_config_load.config["network"]["label_model"]["unit"]["label_prob"]["loss"] = None
                                        feed_config_load.config["network"]["label_model"]["unit"]["label_prob"]["is_output_layer"] = False
                                        feed_config_load.config["network"]["output"]["unit"]["emit_blank_prob"]["loss"] = None

                                        for align_alias, align in zip(["ground-truth", "search", "realign"], [cv_align, search_align]):
                                          vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else bpe_sil_vocab["vocab_file"]
                                          dump_att_weights_job = DumpAttentionWeightsJob(
                                            returnn_config=feed_config_load, model_type="seg_lab_dep" if net_type == "default" else "global-import",
                                            rasr_config=returnn_train_rasr_configs["cv"], blank_idx=targetb_blank_idx,
                                            label_name="alignment", rasr_nn_trainer_exe=rasr_nn_trainer, hdf_targets=align,
                                            seq_tag=seq_tag, )
                                          dump_att_weights_job.add_alias(group_alias + "/" + seq_tag.replace("/", "_") + "/att_weights_%s_%s" % (align_alias, epoch))
                                          tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

                                          plot_weights_job = PlotAttentionWeightsJob(data_path=dump_att_weights_job.out_data,
                                            blank_idx=targetb_blank_idx,
                                            json_vocab_path=vocab_file, time_red=6,
                                            seq_tag=seq_tag)
                                          plot_weights_job.add_alias(group_alias + "/" + seq_tag.replace("/", "_") + "/plot_att_weights_%s_%s" % (align_alias, epoch))
                                          tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)

                                        compare_aligns_job = CompareAlignmentsJob(
                                          hdf_align1=cv_align, hdf_align2=search_align, seq_tag=seq_tag,
                                          blank_idx1=targetb_blank_idx, blank_idx2=targetb_blank_idx, vocab1=vocab_file, vocab2=vocab_file,
                                          name1="ground_truth", name2="search_alignment"
                                        )
                                        compare_aligns_job.add_alias(group_alias + "/" + seq_tag.replace("/", "_") + "/search-align-compare")
                                        tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)

                                        # compare_aligns_job = CompareAlignmentsJob(hdf_align1=cv_align, hdf_align2=cv_realignment,
                                        #   seq_tag=seq_tag, blank_idx1=targetb_blank_idx, blank_idx2=targetb_blank_idx,
                                        #   vocab1=vocab_file, vocab2=vocab_file, name1="ground_truth",
                                        #   name2="search_alignment")
                                        # compare_aligns_job.add_alias(group_alias + "/" + seq_tag + "/realignment-compare")
                                        # tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)



              # # example for calculating RASR search errors
              # cv_align = cv_data_opts.pop("alignment")
              # cv_segments = cv_data_opts.pop("segment_file")
              # new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
              # new_rasr_decoding_opts.update(
              #   dict(
              #     word_end_pruning_limit=12, word_end_pruning=10.0, label_pruning_limit=12, label_pruning=10.0,
              #     corpus_path=corpus_files["train"], feature_cache_path=feature_cache_files["train"]))
              # alias_addon = "_rasr_beam_search12_no-recomb"
              # calc_rasr_search_errors(
              #   segment_path=cv_segments, mem_rqmt=8, simple_beam_search=True, ref_align=cv_align,
              #   num_classes=1032, num_epochs=epoch, blank_idx=1031, rasr_nn_trainer_exe=rasr_nn_trainer,
              #   extern_sprint_rasr_config=returnn_train_rasr_configs["cv"],
              #   train_config=train_config, loop_update_history=True,
              #   full_sum_decoding=False, blank_update_history=True,
              #   allow_word_end_recombination=False, allow_label_recombination=False,
              #   max_seg_len=None, debug=False, compile_config=compile_config.get_config(),
              #   alias_addon=alias_addon, rasr_exe_path=rasr_flf_tool,
              #   model_checkpoint=checkpoint, time_rqmt=time_rqmt,
              #   gpu_rqmt=1, **new_rasr_decoding_opts)

              # # example for realignment + retraining
              # cv_realignment = run_rasr_realignment(
              #   compile_config=compile_config.get_config(), alias_addon=alias_addon + "_cv",
              #   segment_path=cv_segments, loop_update_history=True, blank_update_history=True,
              #   name=name, corpus_path=corpus_files["train"], lexicon_path=bpe_sil_phon_lexicon_path,
              #   allophone_path=total_data["bpe-with-sil"]["allophones"],
              #   state_tying_path=total_data["bpe-with-sil"]["state_tying"], feature_cache_path=feature_cache_files["train"],
              #   num_epochs=epoch, label_file=total_data["bpe-with-sil"]["rasr_label_file"],
              #   label_pruning=50.0, label_pruning_limit=1000, label_recombination_limit=-1, blank_label_index=1031,
              #   model_checkpoint=checkpoint, context_size=-1, reduction_factors=time_red, rasr_nn_trainer_exe_path=rasr_nn_trainer,
              #   start_label_index=1030, rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=1032, time_rqmt=3,
              #   blank_allophone_state_idx=4123,
              #   max_segment_len=total_data[params["config"]["label_type"]]["train"]["time-red-%s" % time_red]["95_percentile"]
              # )
              #
              # train_realignment = run_rasr_realignment(compile_config=compile_config.get_config(),
              #   alias_addon=alias_addon + "_train", segment_path=train_data_opts["segment_file"], name=name,
              #   corpus_path=corpus_files["train"], lexicon_path=bpe_sil_phon_lexicon_path,
              #   allophone_path=total_data["bpe-with-sil"]["allophones"], loop_update_history=True,
              #   blank_update_history=True,
              #   state_tying_path=total_data["bpe-with-sil"]["state_tying"], feature_cache_path=feature_cache_files["train"],
              #   num_epochs=epoch, label_file=total_data["bpe-with-sil"]["rasr_label_file"], label_pruning=50.0,
              #   label_pruning_limit=1000, label_recombination_limit=-1, blank_label_index=1031, model_checkpoint=checkpoint,
              #   context_size=-1, reduction_factors=time_red, rasr_nn_trainer_exe_path=rasr_nn_trainer,
              #   start_label_index=1030, rasr_am_trainer_exe_path=rasr_am_trainer, num_classes=1032, time_rqmt=80,
              #   blank_allophone_state_idx=4123,
              #   max_segment_len=total_data[params["config"]["label_type"]]["train"]["time-red-%s" % time_red][
              #     "95_percentile"])
              #
              # checkpoint_path = DelayedReplace(checkpoint.index_path, ".index", "")
              #
              # retrain_config_obj = config_class(
              #   task="train",
              #   post_config={"cleanup_old_models": {"keep_last_n": 1, "keep_best_n": 1, "keep": num_epochs}},
              #   train_data_opts=train_data_opts,
              #   cv_data_opts=cv_data_opts,
              #   devtrain_data_opts=devtrain_data_opts,
              #   import_model=checkpoint_path,
              #   pretrain=False,
              #   learning_rates=list(numpy.linspace(0.001 * 0.1, 0.001, num=10))  ,# lr warmup
              #   **config_params).get_config()
              #
              # retrain_checkpoints, retrain_config = run_training(retrain_config_obj, mem_rqmt=24, time_rqmt=30, num_epochs=num_epochs,
              #                                          name=name, alias_suffix="retrain_neural-length")
              #
              # for epoch in num_epochs:
              #   retrain_checkpoint = retrain_checkpoints[epoch]
              #   # standard returnn decoding
              #   search_config = config_class(task="search", search_data_opts=dev_data_opts, target="bpe",
              #     search_use_recomb=True, **config_params)
              #   ctm_results = run_bpe_returnn_decoding(
              #     returnn_config=search_config.get_config(), checkpoint=retrain_checkpoint,
              #     stm_job=hub5e_00_stm_job, num_epochs=epoch, name=name,
              #     dataset_key="dev", alias_addon="_retrain_returnn_recomb")
              #   run_eval(ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name=name,
              #            dataset_key="dev", num_epochs=epoch, alias_addon="_retrain_returnn_recomb")



          # for global attention models with BPE labels use RETURNN decoding
          elif params["config"]["label_type"].startswith("bpe"):
            if name == "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.ctx-use-bias.all-segs":
              beam_sizes = [12]
            else:
              beam_sizes = [12]
            for beam_size in beam_sizes:
              if name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
                eval_corpus_names = ["dev", "hub5e_01", "rt03s"]
              else:
                eval_corpus_names = ["dev"]
              for eval_corpus_name in eval_corpus_names:
                if eval_corpus_name == "dev":
                  data_opts = copy.deepcopy(dev_data_opts)
                  stm_job = hub5e_00_stm_job
                elif eval_corpus_name == "hub5e_01":
                  data_opts = copy.deepcopy(hub5e_01_data_opts)
                  stm_job = hub5e_01_stm_job
                else:
                  assert eval_corpus_name == "rt03s"
                  data_opts = copy.deepcopy(rt03s_data_opts)
                  stm_job = rt03s_stm_job
                search_config = config_class(
                  task="search",
                  beam_size=beam_size,
                  search_data_opts=data_opts,
                  **config_params)
                ctm_results = run_bpe_returnn_decoding(
                  returnn_config=search_config.get_config(), checkpoint=checkpoint, stm_job=stm_job,
                  num_epochs=epoch, name=name, dataset_key=eval_corpus_name, alias_addon="_beam-%s" % beam_size,
                  mem_rqmt=4 if beam_size == 12 else 16)
                run_eval(
                  ctm_file=ctm_results, reference=eval_ref_files[eval_corpus_name], name=name,
                  dataset_key=eval_corpus_name, num_epochs=epoch, alias_addon="_beam-%s" % beam_size)

            search_error_data_opts = copy.deepcopy(cv_data_opts)
            label_hdf = search_error_data_opts.pop("label_hdf")
            label_name = search_error_data_opts.pop("label_name")
            segment_file = search_error_data_opts.pop("segment_file")
            search_error_data_opts["vocab"] = dev_data_opts["vocab"]
            dump_search_config = config_class(
              task="search", search_data_opts=search_error_data_opts, dump_output=True, import_model=checkpoint,
              **config_params)
            train_config_load = copy.deepcopy(train_config_obj)
            train_config_load.config["load"] = checkpoint
            search_targets_hdf, ctm_results = calculate_search_errors(
              checkpoint=checkpoint, search_config=dump_search_config, train_config=train_config_load,
              name=name, segment_path=segment_file, ref_targets=label_hdf, label_name=label_name, model_type="glob",
              blank_idx=0, rasr_nn_trainer_exe=rasr_nn_trainer, rasr_config=returnn_train_rasr_configs["cv"],
              alias_addon="_debug", epoch=epoch, dataset_key="cv", stm_job=cv_stm_job, length_norm=True)
            run_eval(ctm_file=ctm_results, reference=cv_stm_job.out_stm_path, name=name,
              dataset_key="cv", num_epochs=epoch, alias_addon="_beam-%s" % beam_size)

            if epoch == 150:
              feed_config_load = copy.deepcopy(train_config_obj)
              feed_config_load.config["load"] = checkpoint
              vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else bpe_sil_vocab[
                "vocab_file"]
              if name.startswith("glob.best-model.bpe."):
                hdf_aliases = ["ground-truth", "search"]
                hdf_targetss = [label_hdf, search_targets_hdf]
                if name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
                  hdf_aliases += ["global_import_segmental"]
                  hdf_targetss += [search_labels["global_import_segmental"]]
              else:
                assert name.startswith("glob.best-model.bpe-with-sil")
                hdf_aliases = ["ground-truth", "search", "global_import_segmental_w_split_sil"]
                hdf_targetss = [label_hdf, search_targets_hdf, search_labels["global_import_segmental_w_split_sil"]]
              for hdf_alias, hdf_targets in zip(hdf_aliases, hdf_targetss):
                for seq_tag in [
                  "switchboard-1/sw02102A/sw2102A-ms98-a-0092",
                  "switchboard-1/sw02022A/sw2022A-ms98-a-0002",
                  "switchboard-1/sw02102A/sw2102A-ms98-a-0090",
                  "switchboard-1/sw02025A/sw2025A-ms98-a-0035",
                  "switchboard-1/sw02102A/sw2102A-ms98-a-0002",
                  "switchboard-1/sw02023A/sw2023A-ms98-a-0001"
                ]:
                  dump_att_weights_job = DumpAttentionWeightsJob(returnn_config=feed_config_load, model_type="glob",
                    rasr_config=returnn_train_rasr_configs["cv"], blank_idx=0, label_name=label_name,
                    rasr_nn_trainer_exe=rasr_nn_trainer, hdf_targets=hdf_targets,
                    seq_tag=seq_tag, )
                  dump_att_weights_job.add_alias(name + "/" + seq_tag.replace("/", "_") + "/att_weights_%s-labels" % (hdf_alias,))
                  tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

                  plot_weights_job = PlotAttentionWeightsJob(
                    data_path=dump_att_weights_job.out_data,
                    blank_idx=None, json_vocab_path=vocab_file,
                    time_red=6, seq_tag=seq_tag)
                  plot_weights_job.add_alias(name + "/" + seq_tag.replace("/", "_") + "/plot_att_weights_%s-labels" % (hdf_alias,))
                  tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)

              for hdf_alias, hdf_targets in zip(hdf_aliases, hdf_targetss):
                calc_search_err_job = CalcSearchErrorJob(returnn_config=train_config, rasr_config=returnn_train_rasr_configs["cv"],
                  rasr_nn_trainer_exe=rasr_nn_trainer, segment_file=segment_file, blank_idx=0,
                  model_type="glob", label_name=label_name, search_targets=hdf_targets, ref_targets=label_hdf,
                  max_seg_len=-1, length_norm=True)
                calc_search_err_job.add_alias(name + ("/search_errors_%d_%s" % (epoch, hdf_alias)))
                alias = calc_search_err_job.get_one_alias()
                tk.register_output(alias + "search_errors", calc_search_err_job.out_search_errors)

            if name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
              compile_config = config_class(
                task="eval", feature_stddev=3.0, beam_size=beam_size, search_data_opts=dev_data_opts,
                **config_params)
              alias_addon = "rasr_limit12"
              new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
              new_rasr_decoding_opts.update(
                dict(word_end_pruning_limit=12, word_end_pruning=12.0, label_pruning_limit=12, label_pruning=12.0))
              ctm_results = run_rasr_decoding(
                segment_path=None, mem_rqmt=mem_rqmt, simple_beam_search=True,
                length_norm=True, full_sum_decoding=False, blank_update_history=True,
                allow_word_end_recombination=False, loop_update_history=True,
                allow_label_recombination=False, max_seg_len=None, debug=False,
                compile_config=compile_config.get_config(), alias_addon=alias_addon, rasr_exe_path=rasr_flf_tool,
                model_checkpoint=checkpoint, num_epochs=epoch, time_rqmt=20, gpu_rqmt=1, **new_rasr_decoding_opts)
              run_eval(ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name=name,
                       dataset_key="dev", num_epochs=epoch, alias_addon=alias_addon)

            if name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
              for concat_num in concat_jobs:
                for corpus_name in concat_jobs[concat_num]:
                  if corpus_name == "hub5e_00":
                    data_opts = copy.deepcopy(dev_data_opts)
                    stm_job = hub5e_00_stm_job
                  elif corpus_name == "hub5e_01":
                    data_opts = copy.deepcopy(hub5e_01_data_opts)
                    stm_job = hub5e_01_stm_job
                  else:
                    assert corpus_name == "rt03s"
                    data_opts = copy.deepcopy(rt03s_data_opts)
                    stm_job = rt03s_stm_job
                  data_opts.update({
                    "concat_seqs": True, "concat_seq_tags": concat_jobs[concat_num][corpus_name].out_concat_seq_tags,
                    "concat_seq_lens": concat_jobs[concat_num][corpus_name].out_orig_seq_lens_py
                  })
                  search_config = config_class(
                    task="search", beam_size=12, search_data_opts=data_opts,
                    **config_params)

                  from recipe.i6_experiments.users.schmitt.experiments.config.concat_seqs.scoring import ScliteHubScoreJob
                  ctm_results = run_bpe_returnn_decoding(returnn_config=search_config.get_config(), checkpoint=checkpoint,
                    stm_job=stm_job, num_epochs=epoch, name=name, dataset_key=corpus_name, concat_seqs=True,
                    alias_addon="concat-%s_beam-%s" % (concat_num, 12), mem_rqmt=4,
                    stm_path=concat_jobs[concat_num][corpus_name].out_stm)
                  run_eval(ctm_file=ctm_results, reference=concat_jobs[concat_num][corpus_name].out_stm, name=name,
                    dataset_key="%s_concat-%s" % (corpus_name, concat_num), num_epochs=epoch, alias_addon="_beam-%s" % 12)


  # for config_file, name in zip([
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss.config",
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss.config",
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss.config",
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss.config",
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss_sep-loops.config",
  #   "/u/schmitt/experiments/transducer/recipe/i6_experiments/users/schmitt/experiments/swb/transducer/returnn_config_files/seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss_sep-loops.config"
  # ], [
  #   "clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss",
  #   "seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss",
  #   "seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss",
  #   "clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss",
  #   "seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss_sep-loops",
  #   "seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed+prev-att.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss_sep-loops"
  # ]):
  #
  #   config_path = Path(config_file)
  #
  #   model_dir = run_training_from_file(
  #     config_file_path=config_path, mem_rqmt=24, time_rqmt=30, parameter_dict={},
  #     name="old_" + name, alias_suffix="_train"
  #   )

    # model_dir = Path("/u/schmitt/experiments/transducer/alias/glob.best-model.bpe.time-red6.am2048.6pretrain-reps.ctx-use-bias.all-segs/train/output/models")

    # for epoch in [33, 150]:
    #   ctm_results = run_search_from_file(
    #     config_file_path=config_path, parameter_dict={}, time_rqmt=1, mem_rqmt=4, name="old_" + name,
    #     alias_suffix="search", model_dir=model_dir, load_epoch=epoch, default_model_name="epoch",
    #     stm_job=hub5e_00_stm_job
    #   )
    #   run_eval(ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name="old_" + name,
    #     dataset_key="dev", num_epochs=epoch, alias_addon="returnn")




