import copy
import os

import numpy
import numpy as np

from i6_private.users.schmitt.returnn.tools import DumpForwardJob, CompileTFGraphJob, RASRDecodingJob, \
  CombineAttentionPlotsJob, DumpPhonemeAlignJob, AugmentBPEAlignmentJob, FindSegmentsToSkipJob, ModifySeqFileJob, \
  ConvertCTMBPEToWordsJob, RASRLatticeToCTMJob, CompareAlignmentsJob, DumpAttentionWeightsJob, \
  PlotAttentionWeightsJob, DumpNonBlanksFromAlignmentJob
from recipe.i6_core.corpus import *
from recipe.i6_core.bpe.apply import ApplyBPEModelToLexiconJob
from recipe.i6_core.tools.git import CloneGitRepositoryJob
from recipe.i6_core.corpus.convert import CorpusToTxtJob
from recipe.i6_core.returnn.training import Checkpoint

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

  allophone_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/allophones")

  phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/zhou-phoneme-transducer/lexicon/train.lex.v1_0_3.ci.gz")
  phon_lexicon_w_blank_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/phonemes/lex_with_blank")
  phon_lexicon_wei = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/phonemes/lexicon_wei")

  bpe_sil_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe-with-sil_lexicon")
  bpe_sil_phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_lexicon_phon")

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
  }

  feature_cache_files = {
    "train": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle", cached=True),
    "dev": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.dev.bundle", cached=True)
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
    "bpe": {}, "phonemes": {}, "phonemes-split-sil": {}, "bpe-with-sil": {}, "bpe-with-sil-split-sil": {}}
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

  for variant_name, params in model_variants.items():
    name = "%s" % variant_name
    check_name = "" + build_alias(**params["config"])
    # check if name is according to my naming conventions
    assert name == check_name, "\n{} \t should be \n{}".format(name, check_name)

    num_epochs = [33, 150]

    if name == "glob.new-model.bpe.time-red6.am2048.6pretrain-reps.all-segs":
      num_epochs = [150]

    if name == "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.sep-sil-model-mean.seg.mlp-att.am2048.length-model-in_am+prev-out-embed.length-focal-0.0.label-focal-0.0.bpe-sil-segs":
      num_epochs = [150]

    if name == "glob.new-model.bpe.time-red6.am2048.all-segs":
      num_epochs.append(200)

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
          feature_cache_path=feature_cache_files["dev"]), alias="returnn_dev_rasr_config")}

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

    rasr_decoding_opts = dict(
      corpus_path=corpus_files["dev"],
      reduction_factors=int(np.prod(params["config"]["time_red"])),
      feature_cache_path=feature_cache_files["dev"], skip_silence=False, name=name)

    # Set more specific data opts for the individual model and label types
    if params["config"]["model_type"] == "glob":
      if params["config"]["label_type"] == "bpe":
        sos_idx = 0
        target_num_labels = 1030
        vocab = bpe_vocab
        vocab["seq_postfix"] = [sos_idx]
        # train_data_opts["vocab"] = vocab
        # cv_data_opts["vocab"] = vocab
        # devtrain_data_opts["vocab"] = vocab
        dev_data_opts["vocab"] = vocab
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
        target_num_labels = 1031
      elif params["config"]["label_type"] == "bpe-with-sil-split-sil":
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
      elif params["config"]["label_type"].startswith("bpe-with-sil"):
        if params["config"]["label_type"] == "bpe-with-sil-split-sil":
          train_align = total_data["bpe-with-sil-split-sil"]["train"]["time-red-%d" % time_red]["align"]
          cv_align = total_data["bpe-with-sil-split-sil"]["cv"]["time-red-%d" % time_red]["align"]
        else:
          train_align = total_data["bpe-with-sil"]["train"]["time-red-%d" % time_red]["align"]
          cv_align = total_data["bpe-with-sil"]["cv"]["time-red-%d" % time_red]["align"]
        sos_idx = 1030
        sil_idx = 0
        target_num_labels = 1031
        targetb_blank_idx = 1031
        vocab = bpe_vocab
        dev_data_opts["vocab"] = bpe_sil_vocab
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
      "sos_idx": sos_idx, "target_num_labels": target_num_labels, "vocab": vocab
    })
    rasr_decoding_opts.update(dict(start_label_index=sos_idx))
    # in case of segmental/transducer model, we need to set the blank index
    if params["config"]["model_type"] == "seg":
      params["config"].update({
        "targetb_blank_idx": targetb_blank_idx,
        "sil_idx": sil_idx})
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
    checkpoints, train_config = run_training(train_config_obj, mem_rqmt=24, time_rqmt=30, num_epochs=num_epochs,
                                             name=name, alias_suffix="train")

    # for each previously specified epoch, run decoding
    for epoch in num_epochs:
      checkpoint = checkpoints[epoch]

      if params["config"]["model_type"] == "seg":
        if epoch in [33, 150]:
          # for bpe + sil model use additional RETURNN decoding as sanity check
          if params["config"]["label_type"].startswith("bpe"):
            if params["config"]["label_type"].startswith("bpe-with-sil"):
              config_params["vocab"]["vocab_file"] = total_data["bpe-with-sil"]["json_vocab"]
            # standard returnn decoding
            search_config = config_class(
              task="search", search_data_opts=dev_data_opts, target="bpe", search_use_recomb=True,
              **config_params)
            ctm_results = run_bpe_returnn_decoding(returnn_config=search_config.get_config(), checkpoint=checkpoint,
                                                   stm_job=hub5e_00_stm_job, num_epochs=epoch, name=name,
                                                   dataset_key="dev", alias_addon="_returnn_recomb")
            run_eval(ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name=name,
                     dataset_key="dev", num_epochs=epoch, alias_addon="_returnn_recomb")

            # # Config for label dep length model
            # search_config = config_class(
            #   task="search", search_data_opts=dev_data_opts, target="bpe",
            #   label_dep_length_model=True,
            #   label_dep_means=total_data[params["config"]["label_type"]]["train"]["time-red-%s" % time_red]["label_dep_mean_lens"],
            #   max_seg_len=total_data[params["config"]["label_type"]]["train"]["time-red-%s" % time_red]["95_percentile"],
            #   **config_params)

            # # Config for compiling model for RASR
            # compile_config = config_class(task="eval", feature_stddev=3., **config_params)

            # # example for RASR decoding
            # new_rasr_decoding_opts = copy.deepcopy(rasr_decoding_opts)
            # new_rasr_decoding_opts.update(
            #   dict(word_end_pruning_limit=12, word_end_pruning=10.0, label_pruning_limit=12, label_pruning=10.0))
            # alias_addon = "_rasr_pruning10.0_limit12_fullsum-recomb_loop-update-hist"
            # ctm_results = run_rasr_decoding(segment_path=None, mem_rqmt=8, simple_beam_search=False,
            #   full_sum_decoding=True, blank_update_history=False, allow_word_end_recombination=True,
            #   loop_update_history=True,
            #   allow_label_recombination=True, max_seg_len=None, debug=False, compile_config=compile_config.get_config(),
            #   alias_addon=alias_addon, rasr_exe_path=rasr_flf_tool, model_checkpoint=checkpoint, num_epochs=epoch,
            #   time_rqmt=time_rqmt, gpu_rqmt=1, **new_rasr_decoding_opts)
            # run_eval(
            #   ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name=name,
            #   dataset_key="dev", num_epochs=epoch, alias_addon=alias_addon)

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
      elif params["config"]["label_type"] == "bpe":
        checkpoint_index_path = checkpoint.index_path
        pretrain_checkpoint_index_path = DelayedReplace(str(checkpoint_index_path), "epoch", "epoch.pretrain")
        if os.path.exists(str(pretrain_checkpoint_index_path)):
          # checkpoint_index_path = checkpoint.index_path
          # pretrain_checkpoint_index_path = DelayedReplace(str(checkpoint_index_path), "epoch", "epoch.pretrain")
          checkpoint = Checkpoint(pretrain_checkpoint_index_path)
          print(checkpoint)
        search_config = config_class(
          task="search",
          search_data_opts=dev_data_opts,
          **config_params)
        ctm_results = run_bpe_returnn_decoding(
          returnn_config=search_config.get_config(), checkpoint=checkpoint, stm_job=hub5e_00_stm_job,
          num_epochs=epoch, name=name, dataset_key="dev")
        run_eval(
          ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name=name,
          dataset_key="dev", num_epochs=epoch, alias_addon="returnn")

        # if epoch == 33:
        #   label_hdf = cv_data_opts.pop("label_hdf")
        #   label_name = cv_data_opts.pop("label_name")
        #   segment_file = cv_data_opts.pop("segment_file")
        #   cv_data_opts["vocab"] = dev_data_opts["vocab"]
        #   dump_search_config = config_class(
        #     task="search", search_data_opts=cv_data_opts, dump_output=True, **config_params)
        #   calculate_search_errors(
        #     checkpoint=checkpoint, search_config=dump_search_config, train_config=train_config_obj,
        #     name=name, segment_path=segment_file, ref_targets=label_hdf, label_name=label_name, model_type="glob",
        #     blank_idx=0, rasr_nn_trainer_exe=rasr_nn_trainer, rasr_config=returnn_train_rasr_configs["cv"],
        #     alias_addon="_debug", epoch=epoch)

        # some problem with tf graph compile
        # if name == "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.all-segs" and epoch == 33:
        #   compile_config = config_class(task="eval", feature_stddev=3., **config_params)
        #
        #   # standard beam search (12)
        #   # alias_addon = "_debug"
        #   # compile_graph_job = CompileTFGraphJob(compile_config, "output")
        #   # compile_graph_job.add_alias(name + "/tf-graph" + alias_addon)
        #   # alias = compile_graph_job.get_one_alias()
        #   # tk.register_output(alias + "/tf-graph", compile_graph_job.out_graph)
        #   # tk.register_output(alias + "/tf-rec-info", compile_graph_job.out_rec_info)
        #   # ctm_results = run_rasr_decoding(segment_path=cv_segments, mem_rqmt=8, simple_beam_search=True,
        #   #                                 full_sum_decoding=False, blank_update_history=True,
        #   #                                 allow_word_end_recombination=False, allow_label_recombination=False,
        #   #                                 max_seg_len=None, debug=False, compile_config=compile_config.get_config(),
        #   #                                 alias_addon=alias_addon, rasr_exe_path=rasr_flf_tool,
        #   #                                 model_checkpoint=checkpoint, num_epochs=epoch, time_rqmt=time_rqmt,
        #   #                                 loop_update_history=True,
        #   #                                 **rasr_decoding_opts)

  for config_file, name in zip([
    "/u/schmitt/experiments/transducer/config/returnn_config/config_files/switchboard/old_configs/clamped6.seg.mlp-att.am2048.key1024.query-am.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed+lm.readout-in_lm+att.emit-prob-in_s.config",
    "/u/schmitt/experiments/transducer/config/returnn_config/config_files/switchboard/old_configs/clamped6.seg.mlp-att.am2048.key1024.query-am.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s.config",
    "/u/schmitt/experiments/transducer/config/returnn_config/config_files/switchboard/old_configs/clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s.config",
    "/u/schmitt/experiments/transducer/config/returnn_config/config_files/switchboard/old_configs/clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.config",
    "/u/schmitt/experiments/transducer/config/returnn_config/config_files/switchboard/old_configs/clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss.config"
  ], [
    "clamped6.seg.mlp-att.am2048.key1024.query-am.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed+lm.readout-in_lm+att.emit-prob-in_s",
    "clamped6.seg.mlp-att.am2048.key1024.query-am.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s",
    "clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s",
    "clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout",
    "clamped6.seg.mlp-att.am2048.key1024.query-lm.slow-rnn-in_input-embed.fast-rnn-in_am+prev-out-embed.readout-in_lm+att.emit-prob-in_s_no-switchout.no-focal-loss"

  ]):

    config_path = Path(config_file)

    model_dir = run_training_from_file(
      config_file_path=config_path, mem_rqmt=24, time_rqmt=30, parameter_dict={},
      name="old_" + name, alias_suffix="_train"
    )

    for epoch in [33, 150]:
      ctm_results = run_search_from_file(
        config_file_path=config_path, parameter_dict={}, time_rqmt=1, mem_rqmt=4, name="old_" + name,
        alias_suffix="search", model_dir=model_dir, load_epoch=epoch, default_model_name="epoch.pretrain",
        stm_job=hub5e_00_stm_job
      )
      run_eval(ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name="old_" + name,
        dataset_key="dev", num_epochs=epoch, alias_addon="returnn")



