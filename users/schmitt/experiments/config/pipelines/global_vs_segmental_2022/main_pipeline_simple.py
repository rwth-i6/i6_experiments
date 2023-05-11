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
from recipe.i6_experiments.users.schmitt.experiments.config.concat_seqs.concat_seqs import MergeSeqTagFiles

# from sisyphus import *
from sisyphus.delayed_ops import DelayedFormat, DelayedReplace
from i6_experiments.users.schmitt.experiments.swb.transducer.config import \
  SegmentalSWBExtendedConfig, SegmentalSWBAlignmentConfig
from i6_experiments.users.schmitt.experiments.swb.global_enc_dec.config import \
  GlobalEncoderDecoderConfig
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.sub_pipelines import run_eval, run_training, run_bpe_returnn_decoding, \
  run_rasr_decoding, calculate_search_errors, run_rasr_realignment, calc_rasr_search_errors, run_training_from_file, \
  run_search_from_file

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.model_variants_simple import build_alias, model_variants
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.build_rasr_configs import build_returnn_train_config, build_returnn_train_feature_flow, \
  build_phon_align_extraction_config, write_config, build_decoding_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.miscellaneous import find_seqs_to_skip, update_seq_list_file, dump_phoneme_align, calc_align_stats, \
  augment_bpe_align_with_sil, alignment_split_silence, reduce_alignment, convert_phon_json_vocab_to_rasr_vocab, \
  convert_phon_json_vocab_to_allophones, convert_phon_json_vocab_to_state_tying, \
  convert_phon_json_vocab_to_rasr_formats, convert_bpe_json_vocab_to_rasr_formats

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.alignments import \
  create_swb_alignments, create_librispeech_alignments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022.recognition import \
  start_rasr_recog_pipeline, start_analysis_pipeline


def run_pipeline():
  stm_jobs = {}
  cv_corpus_job = FilterCorpusBySegmentsJob(
    bliss_corpus=Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz"),
    segment_file=Path("/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000"))
  train_corpus_job = FilterCorpusBySegmentsJob(
    bliss_corpus=Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz"),
    segment_file=Path("/u/schmitt/experiments/transducer/config/dependencies/seg_train"))
  bliss_corpora = [
    Path("/work/asr3/irie/data/switchboard/corpora/dev.corpus.gz"),
    Path("/work/asr3/irie/data/switchboard/corpora/hub5e_01.corpus.gz"),
    Path("/work/asr3/irie/data/switchboard/corpora/rt03s.corpus.gz"),
    cv_corpus_job.out_corpus, train_corpus_job.out_corpus]

  for corpus_path, corpus_alias in zip(bliss_corpora, ["hub5_00", "hub5_01", "rt03s", "cv", "train"]):
    stm_jobs[corpus_alias] = CorpusToStmJob(bliss_corpus=corpus_path)
    stm_jobs[corpus_alias].add_alias("stm_files" + "/" + corpus_alias)
    alias = stm_jobs[corpus_alias].get_one_alias()
    tk.register_output(alias + "/stm_corpus", stm_jobs[corpus_alias].out_stm_path)

  eval_ref_files = {
    "dev": Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), "hub5e_01": Path("/u/tuske/bin/switchboard/hub5e_01.2.stm"),
    "rt03s": Path("/u/tuske/bin/switchboard/rt03s_ctsonly.stm"), }

  concat_jobs = run_concat_seqs(
    ref_stm_paths={
      "hub5e_00": eval_ref_files["dev"], "hub5e_01": eval_ref_files["hub5e_01"],
      "rt03s": eval_ref_files["rt03s"], "train": stm_jobs["train"].out_stm_path, "cv": stm_jobs["cv"].out_stm_path},
    glm_path=Path("/work/asr2/oberdorfer/kaldi-stable/egs/swbd/s5/data/eval2000/glm"),
    concat_nums=[1, 2, 4, 10, 20])

  merge_train_concat_1_2_job = MergeSeqTagFiles([
    concat_jobs[1]["train"].out_concat_seq_tags, concat_jobs[2]["train"].out_concat_seq_tags])
  tk.register_output("merge_train_seqs_concat_1_2", merge_train_concat_1_2_job.out_seq_tags)

  allophone_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/allophones")
  allophone_path_librispeech = Path("/work/asr_archive/assis/luescher/best-models/librispeech/960h_2019-04-10/StoreAllophones.34VPSakJyy0U/output/allophones")

  phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/zhou-phoneme-transducer/lexicon/train.lex.v1_0_3.ci.gz")
  phon_lexicon_path_librispeech = Path("/u/corpora/speech/LibriSpeech/lexicon/original.lexicon.golik.xml.gz")
  phon_lexicon_w_blank_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/phonemes/lex_with_blank")
  phon_lexicon_wei = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/phonemes/lexicon_wei")

  bpe_sil_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe-with-sil_lexicon")
  bpe_sil_phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_lexicon_phon")
  bpe_phon_lexicon_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_lexicon_phon_wo_sil2")

  phon_state_tying_mono_eow_3_states_path = Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/state-tying_mono-eow_3-states")
  zoltan_phon_align_cache = Path("/work/asr4/zhou/asr-exps/swb1/dependencies/zoltan-alignment/train.alignment.cache.bundle", cached=True)
  chris_phon_align_librispeech = Path("/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/AlignmentJob.uPtxlMbFI4lx/output/alignment.cache.bundle", cached=True)

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

  corpus_files_librispeech = {
    "train-merged-960": Path("/work/speech/golik/setups/librispeech/resources/corpus/corpus.train-merged-960.corpus.gz", cached=True),
  }

  feature_cache_files = {
    "train": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle", cached=True),
    "dev": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.dev.bundle", cached=True),
    "hub5e_01": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.hub5e_01.bundle", cached=True),
    "rt03s": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.rt03s.bundle", cached=True)
  }

  feature_cache_files_librispeech = {
    "train-merged-960": Path("/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.bundle", cached=True),
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

  seq_filter_files_standard_librispeech = {
    "train": Path("/u/luescher/setups/librispeech/2019-07-23--data-augmentation/work/corpus/ShuffleAndSplitSegments.HwFoXgqL5gha/output/train.segments", cached=True),
  }

  dev_debug_segment_file = Path(
    "/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/segments.1"
  )
  cv_debug_segment_file = Path(
    "/work/asr3/zeyer/schmitt/tests/swb1/bpe-transducer_decoding-test/cv_test_segments1"
  )

  zoltan_4gram_lm = {
    "image": Path("/u/zhou/asr-exps/swb1/dependencies/zoltan_4gram.image"),
    "image_wei": Path("/u/zhou/asr-exps/swb1/dependencies/monophone-eow/clean-all/zoltan_4gram.image"),
    "file": Path("/work/asr4/zhou/asr-exps/swb1/dependencies/zoltan_4gram.gz")
  }

  concat_seq_tags_examples = {
    1: [
      "switchboard-1/sw02022A/sw2022A-ms98-a-0002"
    ],
    2: [
      "switchboard-1/sw02022A/sw2022A-ms98-a-0002;switchboard-1/sw02022A/sw2022A-ms98-a-0003"
    ],
    4: [
      "switchboard-1/sw02022A/sw2022A-ms98-a-0002;switchboard-1/sw02022A/sw2022A-ms98-a-0003;switchboard-1/sw02022A/sw2022A-ms98-a-0004;switchboard-1/sw02022A/sw2022A-ms98-a-0005"
    ],
    10: [
      "switchboard-1/sw02022A/sw2022A-ms98-a-0002;switchboard-1/sw02022A/sw2022A-ms98-a-0003;switchboard-1/sw02022A/sw2022A-ms98-a-0004;switchboard-1/sw02022A/sw2022A-ms98-a-0005;switchboard-1/sw02022A/sw2022A-ms98-a-0006;switchboard-1/sw02022A/sw2022A-ms98-a-0007;switchboard-1/sw02022A/sw2022A-ms98-a-0009;switchboard-1/sw02022A/sw2022A-ms98-a-0011;switchboard-1/sw02022A/sw2022A-ms98-a-0013;switchboard-1/sw02022A/sw2022A-ms98-a-0015"
    ],
    20: [
      "switchboard-1/sw02022A/sw2022A-ms98-a-0002;switchboard-1/sw02022A/sw2022A-ms98-a-0003;switchboard-1/sw02022A/sw2022A-ms98-a-0004;switchboard-1/sw02022A/sw2022A-ms98-a-0005;switchboard-1/sw02022A/sw2022A-ms98-a-0006;switchboard-1/sw02022A/sw2022A-ms98-a-0007;switchboard-1/sw02022A/sw2022A-ms98-a-0009;switchboard-1/sw02022A/sw2022A-ms98-a-0011;switchboard-1/sw02022A/sw2022A-ms98-a-0013;switchboard-1/sw02022A/sw2022A-ms98-a-0015;switchboard-1/sw02022A/sw2022A-ms98-a-0017;switchboard-1/sw02022A/sw2022A-ms98-a-0019;switchboard-1/sw02022A/sw2022A-ms98-a-0021;switchboard-1/sw02022A/sw2022A-ms98-a-0023;switchboard-1/sw02022A/sw2022A-ms98-a-0024;switchboard-1/sw02022A/sw2022A-ms98-a-0025;switchboard-1/sw02022A/sw2022A-ms98-a-0027;switchboard-1/sw02022A/sw2022A-ms98-a-0028;switchboard-1/sw02022A/sw2022A-ms98-a-0030;switchboard-1/sw02022A/sw2022A-ms98-a-0032"
    ]
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

  phon_extraction_rasr_configs_librispeech = {
    "train": write_config(*build_phon_align_extraction_config(
      corpus_path=corpus_files_librispeech["train-merged-960"],
      feature_cache_path=feature_cache_files_librispeech["train-merged-960"],
      segment_path=seq_filter_files_standard_librispeech["train"],
      lexicon_path=phon_lexicon_path_librispeech, allophone_path=allophone_path_librispeech,
      alignment_cache_path=chris_phon_align_librispeech
    ), alias="phon_extract_librispeech_train_rasr_config")
  }


  segment_corpus_job = SegmentCorpusJob(corpus_files["dev"], num_segments=1)
  segment_corpus_job.add_alias("segments_dev")
  tk.register_output("segments_dev", segment_corpus_job.out_single_segment_files[1])

  total_data = {
    "bpe": {}, "phonemes": {}, "phonemes-split-sil": {}, "bpe-with-sil": {}, "bpe-with-sil-split-sil": {},
    "bpe-sil-wo-sil": {}, "bpe-sil-wo-sil-in-middle": {}, "bpe-with-sil-split-silv2": {}, "bpe-center-positions": {},
    "bpe-sil-wo-sil-center-positions": {}, "bpe-sil-center-positions": {}, "bpe-sil-center-pos-wo-sil": {},
    "bpe-sil-center-pos-wo-sil-in-middle": {}}

  total_data_ls = {
    "phonemes"
  }

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

    bpe_seq_filter_file = seq_filter_files_standard[corpus_key]

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
        "seq_filter_file": seq_filter_files_bpe,}
    }
    if corpus_key == "train":
      total_data["bpe"]["devtrain"] = {
        "time-red-6": {
          "seq_filter_file": seq_filter_files_standard["devtrain"]}}

  search_aligns = {}
  search_labels = {}
  for i in range(1):
    for variant_name, params in model_variants.items():
      name = "%s" % variant_name
      check_name = "" + build_alias(**params["config"])
      # check if name is according to my naming conventions
      assert name == check_name, "\n{} \t should be \n{}".format(name, check_name)

      if name in [
        "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.pooling-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.bpe-sil-segs",
        "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
        "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.prev-target-in-readout.all-segs",
        "glob.best-model.bpe-with-sil.time-red6.am2048.1pretrain-reps.ctx-use-bias.pretrain-like-seg.bpe-sil-segs",
        "glob.best-model.bpe-with-sil.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.bpe-sil-segs",
        "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs/search_cv_120_debug",
        "glob.best-model.bpe-with-sil-split-sil.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.bpe-sil-segs",
        "glob.best-model.bpe-with-sil-split-sil.time-red6.am2048.1pretrain-reps.ctx-use-bias.pretrain-like-seg.bpe-sil-segs",
        "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-state-vector.no-weight-feedback.no-l2.ctx-use-bias.all-segs",
        "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-weight-feedback.no-l2.ctx-use-bias.all-segs",
        # "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
        "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.all-segs",
        "glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-state-vector.no-l2.ctx-use-bias.all-segs",
        "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs"
      ]:
        num_epochs = [150]
      else:
        num_epochs = [40, 80, 120, 150]

      # if name in [
      #   "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
      #   "seg.bpe.full-ctx.time-red6.conf.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
      #   "seg.bpe.full-ctx.time-red6.conf-tim.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
      #   "seg.bpe.full-ctx.time-red6.conf-wei.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
      #   "seg.bpe.mlr-1e-06.gc-20.0.gn-0.1.nadam.bs-15000.sa-wei.d-lr.nb-lrd-0.9.full-ctx.time-red6.conf-wei.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-pt.no-ctx-reg.chunk-size-64.all-segs",
      #   "seg.bpe.mlr-1e-06.gc-20.0.gn-0.1.nadam.bs-15000.sa-wei.d-lr.nb-lrd-0.9.no-ctc.full-ctx.time-red6.conf-wei.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-pt.no-ctx-reg.chunk-size-64.all-segs"
      # ]:
      #   num_epochs.append(300)

      # if name == "seg.bpe.concat.ep-split-12.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs":
      #   num_epochs.append(59)

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

      returnn_train_rasr_configs = {}
      for config_alias, corpus_name, corpus_segments in zip(
        ["train", "cv", "devtrain", "dev", "hub5e_01", "rt03s"],
        ["train", "train", "train", "dev", "hub5e_01", "rt03s"],
        [train_segments, cv_segments, devtrain_segments, None, None, None]):

        returnn_train_rasr_configs[config_alias] = write_config(*build_returnn_train_config(
          segment_file=corpus_segments, corpus_file=corpus_files[corpus_name],
          feature_cache_path=feature_cache_files[corpus_name]),
          alias="returnn_%s_rasr_config" % config_alias)

      # General data opts, which apply for all models
      train_data_opts = {
        "data": "train", "rasr_config_path": returnn_train_rasr_configs["train"],
        "rasr_nn_trainer_exe": rasr_nn_trainer, "epoch_split": params["config"]["epoch_split"]}
      if params["config"]["model_type"] == "seg":
        train_data_opts["correct_concat_ep_split"] = params["config"].pop("correct_concat_ep_split")
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
        if params["config"]["label_type"].startswith("bpe"):
          sos_idx = 0 if params["config"]["label_type"] == "bpe" else 1030
          sil_idx = None if params["config"]["label_type"] == "bpe" else 0
          target_num_labels = 1030 if params["config"]["label_type"] == "bpe" else 1031
          vocab = bpe_vocab
          vocab["seq_postfix"] = [0]
          dev_data_opts["vocab"] = bpe_vocab
          hub5e_01_data_opts["vocab"] = bpe_vocab
          rt03s_data_opts["vocab"] = bpe_vocab
          train_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["train"]["label_seqs"],
            "label_name": "bpe", "segment_file": train_segments, "concat_seqs": params["config"].pop("concat_seqs"),
            "concat_seq_tags": merge_train_concat_1_2_job.out_seq_tags})
          cv_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["cv"]["label_seqs"],
            "label_name": "bpe", "segment_file": cv_segments})
          devtrain_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["train"]["label_seqs"],
            "label_name": "bpe", "segment_file": devtrain_segments})
          params["config"]["label_name"] = "bpe"
        else:
          assert params["config"]["label_type"] in ["phonemes", "phonemes-split-sil"]
          train_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["train"]["label_seqs"],
            "label_name": "phonemes",
            "segment_file": train_segments})
          cv_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["cv"]["label_seqs"],
            "label_name": "phonemes",
            "segment_file": cv_segments})
          devtrain_data_opts.update({
            "label_hdf": total_data[params["config"]["label_type"]]["train"]["label_seqs"],
            "label_name": "phonemes",
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
        if params["config"]["label_type"].startswith("bpe"):
          sos_idx = 0 if params["config"]["label_type"] == "bpe" else 1030
          sil_idx = None if params["config"]["label_type"] == "bpe" else 0
          target_num_labels = 1030 if params["config"]["label_type"] == "bpe" else 1031
          targetb_blank_idx = 1030 if params["config"]["label_type"] == "bpe" else 1031
          vocab = bpe_vocab
          dev_data_opts["vocab"] = vocab if params["config"]["label_type"] == "bpe" else {
            "bpe_file": bpe_vocab["bpe_file"], "vocab_file": total_data["bpe-with-sil"]["json_vocab"]}
          hub5e_01_data_opts["vocab"] = vocab
          rt03s_data_opts["vocab"] = vocab
          train_align = total_data[params["config"]["label_type"]]["train"]["time-red-%d" % time_red]["align"] if params["config"]["label_type"] != "bpe-with-sil-split-silv2" else Path("/work/asr3/zeyer/schmitt/old_models_and_analysis/old_bpe_sil_split_sil_aligns/train/AlignmentSplitSilenceJob.4dfiua41gqWb/output/out_align")
          cv_align = total_data[params["config"]["label_type"]]["cv"]["time-red-%d" % time_red]["align"] if params["config"]["label_type"] != "bpe-with-sil-split-silv2" else Path("/work/asr3/zeyer/schmitt/old_models_and_analysis/old_bpe_sil_split_sil_aligns/cv/AlignmentSplitSilenceJob.p0VY0atlAdkq/output/out_align")

          train_data_opts.update({
            "segment_file": train_segments, "alignment": train_align,
            "concat_seqs": params["config"].pop("concat_seqs"),
            "concat_seq_tags": merge_train_concat_1_2_job.out_seq_tags})
          cv_data_opts.update({
            "segment_file": cv_segments, "alignment": cv_align})
          devtrain_data_opts.update({
            "segment_file": devtrain_segments, "alignment": train_align})
          rasr_decoding_opts.update(dict(lexicon_path=bpe_sil_lexicon_path, label_unit="word",
            label_file_path=total_data[params["config"]["label_type"]]["rasr_label_file"],
            lm_type="simple-history", use_lm_score=False,
            lm_scale=None, lm_file=None, lm_image=None, label_pruning=10.0, label_pruning_limit=128,
            word_end_pruning_limit=128, word_end_pruning=10.0, lm_lookahead_cache_size_high=None,
            lm_lookahead_cache_size_low=None, lm_lookahead_history_limit=None, lm_lookahead_scale=None,
            lm_lookahead=False))
        else:
          assert params["config"]["label_type"] in ["phonemes", "phonemes-split-sil"]
          train_align = total_data[params["config"]["label_type"]]["train"]["time-red-%d" % time_red]["align"]
          cv_align = total_data[params["config"]["label_type"]]["cv"]["time-red-%d" % time_red]["align"]
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
        "seg": SegmentalSWBExtendedConfig, "glob": GlobalEncoderDecoderConfig}.get(params["config"]["model_type"])
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
      if "conf-tim" in name:
        train_qsub_args = "-l qname=*2080*"
      else:
        train_qsub_args = None
      checkpoints, train_config, train_job = run_training(train_config_obj, mem_rqmt=24, time_rqmt=30, num_epochs=num_epochs,
                                               name=name, alias_suffix="train", qsub_args=train_qsub_args)

      # for each previously specified epoch, run decoding
      # if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.all-segs":
      #   num_epochs = [150]
      #   checkpoints = {150: Checkpoint(index_path=Path("/u/schmitt/experiments/transducer/alias/glob.best-model.bpe.time-red6.am2048.6pretrain-reps.no-weight-feedback.no-l2.ctx-use-bias.all-segs/train/output/models/epoch.150.index"))}
      for epoch in num_epochs:
        if epoch in checkpoints:
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
                "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.frame-length-model-in_am+prev-out-embed.all-segs",
                "seg.bpe.concat.ep-split-12.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs",
                "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.bpe-sil-segs"] and epoch == 150:
                for concat_num in concat_jobs:
                  for corpus_name in concat_jobs[concat_num]:
                    if corpus_name == "hub5e_00":
                      data_opts = copy.deepcopy(dev_data_opts)
                      stm_job = stm_jobs["hub5_00"]
                    elif corpus_name == "hub5e_01":
                      data_opts = copy.deepcopy(hub5e_01_data_opts)
                      stm_job = stm_jobs["hub5_01"]
                    elif corpus_name == "rt03s":
                      data_opts = copy.deepcopy(rt03s_data_opts)
                      stm_job = stm_jobs["rt03s"]
                    else:
                      continue
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
                for use_recomb in [False]:
                  length_scales = [1.]
                  if name == "seg.win-size520.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and epoch == 150:
                    length_scales += [0.]
                  for length_scale in length_scales:
                    alias_addon = "returnn_%srecomb_length-scale-%s_beam-%s" % ("" if use_recomb else "no-", length_scale, beam_size)
                    # standard returnn decoding
                    search_config = config_class(
                      task="search", search_data_opts=dev_data_opts, target="bpe", search_use_recomb=use_recomb,
                      beam_size=beam_size, length_scale=length_scale, **config_params)
                    ctm_results = run_bpe_returnn_decoding(
                      returnn_config=search_config.get_config(), checkpoint=checkpoint,
                      stm_job=stm_jobs["hub5_00"], num_epochs=epoch, name=name,
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
                    calculate_search_errors(checkpoint=checkpoint, search_config=dump_search_config, stm_job=stm_jobs["cv"],
                      train_config=feed_config_load, name=name, segment_path=segment_file, ref_targets=alignment_hdf,
                      label_name="alignment", model_type="seg", blank_idx=targetb_blank_idx, rasr_nn_trainer_exe=rasr_nn_trainer,
                      rasr_config=returnn_train_rasr_configs["cv"], alias_addon=alias_addon, epoch=epoch, dataset_key="cv", length_norm=False)

                    if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs" and length_scale == 1. and epoch == 150 and use_recomb:
                      for concat_num in concat_seq_tags_examples:
                        search_error_data_opts.update({
                          "concat_seqs": True, "concat_seq_tags": concat_jobs[concat_num]["cv"].out_concat_seq_tags,
                          "concat_seq_lens": concat_jobs[concat_num]["cv"].out_orig_seq_lens_py})
                        dump_search_concat_config = config_class(
                          search_use_recomb=True if use_recomb else False, task="search",
                          search_data_opts=search_error_data_opts, target="bpe", beam_size=beam_size,
                          length_scale=length_scale, dump_output=True, import_model=checkpoint, **config_params)
                        search_targets_concat_hdf, ctm_results = calculate_search_errors(
                          checkpoint=checkpoint, search_config=dump_search_concat_config, train_config=feed_config_load,
                          name=name, segment_path=segment_file, ref_targets=alignment_hdf, label_name="alignment",
                          model_type="seg", blank_idx=targetb_blank_idx,  rasr_nn_trainer_exe=rasr_nn_trainer,
                          rasr_config=returnn_train_rasr_configs["cv"], alias_addon="_cv_concat_%d" % concat_num,
                          epoch=epoch, dataset_key="cv", stm_job=stm_jobs["cv"], length_norm=False,
                          concat_seqs=True, stm_path=concat_jobs[concat_num]["cv"].out_stm)
                        # run_eval(
                        #   ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name,
                        #   dataset_key="cv_concat_%d" % concat_num,
                        #   num_epochs=epoch, alias_addon="_beam-%s" % beam_size)
                        vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else \
                          total_data["bpe-with-sil"]["json_vocab"]
                        hdf_aliases = ["ground-truth-concat", "search-concat"]
                        hdf_targetss = [alignment_hdf, search_targets_concat_hdf]
                        for hdf_alias, hdf_targets in zip(hdf_aliases, hdf_targetss):
                          for seq_tag in concat_seq_tags_examples[concat_num]:
                            dump_att_weights_job = DumpAttentionWeightsJob(
                              returnn_config=feed_config_load,  model_type="seg",
                              rasr_config=returnn_train_rasr_configs["cv"], blank_idx=targetb_blank_idx,
                              label_name="alignment", rasr_nn_trainer_exe=rasr_nn_trainer, hdf_targets=hdf_targets,
                              concat_seqs=True, seq_tag=seq_tag,
                              concat_seq_tags_file=concat_jobs[concat_num]["cv"].out_concat_seq_tags,
                              concat_hdf=False if hdf_alias == "search-concat" else True)
                            seq_tags = seq_tag.split(";")
                            seq_tag_alias = seq_tags[0].replace("/", "_") + "-" + seq_tags[-1].replace("/", "_")
                            dump_att_weights_job.add_alias(
                              name + "/" + seq_tag_alias + "/att_weights_%s-labels" % (hdf_alias,))
                            tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

                            plot_weights_job = PlotAttentionWeightsJob(
                              data_path=dump_att_weights_job.out_data, blank_idx=targetb_blank_idx,
                              json_vocab_path=vocab_file, time_red=6, seq_tag=seq_tag, mem_rqmt=12)
                            plot_weights_job.add_alias(
                              name + "/" + seq_tag_alias + "/plot_att_weights_%s-labels" % (hdf_alias,))
                            tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)

              if epoch in [150, 300]:
                length_scales = [1.]
                if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs":
                  length_scales += [.7]
                for length_scale in length_scales:
                  max_seg_lens = [20]
                  for max_seg_len in max_seg_lens:
                    vit_recombs = [True, False]
                    if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs":
                      vit_recombs = [True]
                    if max_seg_len != 20 or length_scale != 1.:
                      vit_recombs = [True]
                    for vit_recomb in vit_recombs:
                      length_norms = [False]
                      if length_scale in [1., 0.7] and \
                        name in [
                          "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs",
                          "seg.bpe-with-sil-split-sil.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.ctx-w-bias.am2048.frame-length-model-in_am+prev-out-embed.no-ctx-reg.bpe-sil-segs"] and \
                        vit_recomb and max_seg_len == 20:
                        length_norms += [True]
                      if name == "seg.bpe.full-ctx.time-red6.fast-rec.fast-rec-full.seg.mlp-att.am2048.prev-att-in-state.frame-length-model-in_am+prev-out-embed.no-ctx-reg.all-segs":
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

                          alias_addon = "rasr_limit12_pruning12.0_%s-recomb_neural-length_max-seg-len-%s_length-scale-%s%s" % ("vit" if vit_recomb else "no", max_seg_len, length_scale, "length-norm" if length_norm else "")
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
                          # run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name,
                          #          dataset_key="cv", num_epochs=epoch, alias_addon=alias_addon)

                          calc_align_stats(alignment=search_align, blank_idx=targetb_blank_idx, seq_filter_file=cv_segments,
                            alias=name + "/" + alias_addon + "/cv_search_align_stats_epoch-%s" % epoch)

                          if "pooling-att" not in name:
                            for seq_tag in [
                              "switchboard-1/sw02102A/sw2102A-ms98-a-0092",
                              "switchboard-1/sw02102A/sw2102A-ms98-a-0090", "switchboard-1/sw02025A/sw2025A-ms98-a-0035",
                              "switchboard-1/sw02102A/sw2102A-ms98-a-0002", "switchboard-1/sw02038B/sw2038B-ms98-a-0069"
                            ]:

                              group_alias = name + "/neural-length_analysis_epoch-%s_length-scale-%s_%s-recomb_max-seg-len-%s%s" % (epoch, length_scale, "no" if not vit_recomb else "vit", max_seg_len, "_length-norm" if length_norm else "")

                              for align_alias, align in zip(["ground-truth", "search", "realign"],
                                                            [cv_align, search_align]):
                                vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else \
                                total_data["bpe-with-sil"]["json_vocab"]
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
                  stm_job = stm_jobs["hub5_00"]
                elif eval_corpus_name == "hub5e_01":
                  data_opts = copy.deepcopy(hub5e_01_data_opts)
                  stm_job = stm_jobs["hub5_01"]
                else:
                  assert eval_corpus_name == "rt03s"
                  data_opts = copy.deepcopy(rt03s_data_opts)
                  stm_job = stm_jobs["rt03s"]
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
              alias_addon="_debug", epoch=epoch, dataset_key="cv", stm_job=stm_jobs["cv"], length_norm=True)
            run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name,
              dataset_key="cv", num_epochs=epoch, alias_addon="_beam-%s" % beam_size)

            if name in [
              "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs",
              "glob.best-model.bpe.concat.ep-split-12.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs"] and epoch == 150:
              for concat_num in concat_seq_tags_examples:
                search_error_data_opts.update({
                  "concat_seqs": True, "concat_seq_tags": concat_jobs[concat_num]["cv"].out_concat_seq_tags,
                  "concat_seq_lens": concat_jobs[concat_num]["cv"].out_orig_seq_lens_py})
                dump_search_concat_config = config_class(task="search", search_data_opts=search_error_data_opts, dump_output=True,
                  import_model=checkpoint, **config_params)
                search_targets_concat_hdf, ctm_results = calculate_search_errors(checkpoint=checkpoint,
                  search_config=dump_search_concat_config, train_config=train_config_load, name=name, segment_path=segment_file,
                  ref_targets=label_hdf, label_name=label_name, model_type="glob", blank_idx=0,
                  rasr_nn_trainer_exe=rasr_nn_trainer, rasr_config=returnn_train_rasr_configs["cv"], alias_addon="_cv_concat%d" % concat_num,
                  epoch=epoch, dataset_key="cv", stm_job=stm_jobs["cv"], length_norm=True,
                  concat_seq_tags_file=concat_jobs[concat_num]["cv"].out_concat_seq_tags)
                # run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name, dataset_key="cv_concat%d" % concat_num,
                #          num_epochs=epoch, alias_addon="_beam-%s" % beam_size)

                feed_config_load = copy.deepcopy(train_config_obj)
                feed_config_load.config["load"] = checkpoint
                vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else total_data["bpe-with-sil"]["json_vocab"]
                hdf_aliases = ["ground-truth-concat", "search-concat"]
                hdf_targetss = [label_hdf, search_targets_concat_hdf]
                for hdf_alias, hdf_targets in zip(hdf_aliases, hdf_targetss):
                  for seq_tag in concat_seq_tags_examples[concat_num]:
                    dump_att_weights_job = DumpAttentionWeightsJob(
                      returnn_config=feed_config_load, model_type="glob",
                      rasr_config=returnn_train_rasr_configs["cv"], blank_idx=0, label_name=label_name,
                      rasr_nn_trainer_exe=rasr_nn_trainer, hdf_targets=hdf_targets, concat_seqs=True,
                      seq_tag=seq_tag, concat_hdf=False if hdf_alias == "search-concat" else True,
                      concat_seq_tags_file=concat_jobs[concat_num]["cv"].out_concat_seq_tags)
                    seq_tags = seq_tag.split(";")
                    seq_tag_alias = seq_tags[0].replace("/", "_") + "-" + seq_tags[-1].replace("/", "_")
                    dump_att_weights_job.add_alias(name + "/" + seq_tag_alias + "/att_weights_%s-labels" % (hdf_alias,))
                    tk.register_output(dump_att_weights_job.get_one_alias(), dump_att_weights_job.out_data)

                    plot_weights_job = PlotAttentionWeightsJob(
                      data_path=dump_att_weights_job.out_data,
                      blank_idx=None, json_vocab_path=vocab_file,
                      time_red=6, seq_tag=seq_tag, mem_rqmt=12)
                    plot_weights_job.add_alias(name + "/" + seq_tag_alias + "/plot_att_weights_%s-labels" % (hdf_alias,))
                    tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)


            if epoch == 150 and name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
              feed_config_load = copy.deepcopy(train_config_obj)
              feed_config_load.config["load"] = checkpoint
              vocab_file = bpe_vocab["vocab_file"] if params["config"]["label_type"] == "bpe" else total_data["bpe-with-sil"]["json_vocab"]
              if name.startswith("glob.best-model.bpe."):
                hdf_aliases = ["ground-truth", "search"]
                hdf_targetss = [label_hdf, search_targets_hdf]
                # if name == "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs":
                #   hdf_aliases += ["global_import_segmental"]
                #   hdf_targetss += [search_labels["global_import_segmental"]]
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
              # ctm_results = run_rasr_decoding(
              #   segment_path=None, mem_rqmt=mem_rqmt, simple_beam_search=True,
              #   length_norm=True, full_sum_decoding=False, blank_update_history=True,
              #   allow_word_end_recombination=False, loop_update_history=True,
              #   allow_label_recombination=False, max_seg_len=None, debug=False,
              #   compile_config=compile_config.get_config(), alias_addon=alias_addon, rasr_exe_path=rasr_flf_tool,
              #   model_checkpoint=checkpoint, num_epochs=epoch, time_rqmt=20, gpu_rqmt=1, **new_rasr_decoding_opts)
              # run_eval(ctm_file=ctm_results, reference=Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"), name=name,
              #          dataset_key="dev", num_epochs=epoch, alias_addon=alias_addon)

            if name in [
              "glob.best-model.bpe.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs",
              "glob.best-model.bpe.concat.ep-split-12.time-red6.am2048.1pretrain-reps.no-weight-feedback.ctx-use-bias.pretrain-like-seg.all-segs"] and epoch == 150:
              for concat_num in concat_jobs:
                for corpus_name in concat_jobs[concat_num]:
                  if corpus_name == "hub5e_00":
                    data_opts = copy.deepcopy(dev_data_opts)
                    stm_job = stm_jobs["hub5_00"]
                  # elif corpus_name == "cv":
                  #   data_opts = copy.deepcopy(cv_data_opts)
                  #   stm_job = stm_jobs["cv"]
                  elif corpus_name == "hub5e_01":
                    continue
                    data_opts = copy.deepcopy(hub5e_01_data_opts)
                    stm_job = stm_jobs["hub5_01"]
                  else:
                    continue
                    data_opts = copy.deepcopy(rt03s_data_opts)
                    stm_job = stm_jobs["rt03s"]
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
                    name=name, segment_path=segment_file, ref_targets=label_hdf, label_name=label_name,
                    model_type="glob",
                    blank_idx=0, rasr_nn_trainer_exe=rasr_nn_trainer, rasr_config=returnn_train_rasr_configs["cv"],
                    alias_addon="concat-%s_beam-%s_search-errors" % (concat_num, 12), epoch=epoch, dataset_key="cv", stm_job=stm_jobs["cv"], length_norm=True)
                  run_eval(ctm_file=ctm_results, reference=stm_jobs["cv"].out_stm_path, name=name,
                           dataset_key="cv", num_epochs=epoch, alias_addon="_beam-%s" % beam_size)





