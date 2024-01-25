from i6_core.rasr.config import RasrConfig, build_config_from_mapping, WriteRasrConfigJob
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output

from sisyphus import *

from typing import Tuple, Optional, Dict, Any, Union


class RasrConfigBuilder:
  @staticmethod
  def _write_config(config: RasrConfig, post_config: RasrConfig) -> Path:
    write_config_job = WriteRasrConfigJob(config=config, post_config=post_config)

    return write_config_job.out_config

  @staticmethod
  def _get_corpus_config(corpus_path, segment_path):
    corpus_config = RasrConfig()
    corpus_config.audio_dir = "/"
    corpus_config.capitalize_transcriptions = False
    if corpus_path is not None:
      corpus_config.file = corpus_path
    if segment_path is not None:
      corpus_config.segments.file = segment_path
    corpus_config.progress_indication = "global"
    corpus_config.warn_about_unexpected_elements = True

    return corpus_config

  @staticmethod
  def get_lm_image_crp(
          lexicon_path: Path,
          lm_type: str,
          lm_file: Path,

  ) -> CommonRasrParameters:
    assert lm_type == "ARPA", "Only ARPA LMs are supported."

    crp = CommonRasrParameters()
    crp_add_default_output(crp, unbuffered=True)

    lexicon_config = RasrConfig()
    lexicon_config.file = lexicon_path
    lexicon_config.normalize_pronunciation = False
    crp.lexicon_config = lexicon_config

    lm_config = RasrConfig()
    lm_config.file = lm_file
    lm_config.type = lm_type
    crp.language_model_config = lm_config

    return crp

  @staticmethod
  def get_phon_align_extraction_config(
          corpus_path: Path,
          segment_path: Path,
          allophone_path: Path,
          lexicon_path: Path,
          feature_cache_path: Path,
          alignment_cache_path: Path
  ):
    crp = CommonRasrParameters()
    crp_add_default_output(crp, unbuffered=True)

    corpus_config = RasrConfigBuilder._get_corpus_config(corpus_path, segment_path)
    crp.corpus_config = corpus_config

    am_config = RasrConfig()
    am_config.allophones.add_all = True
    am_config.allophones.add_from_file = allophone_path
    am_config.allophones.add_from_lexicon = False
    am_config.hmm.across_word_model = True
    am_config.hmm.early_recombination = False
    am_config.hmm.state_repetitions = 1
    am_config.hmm.states_per_phone = 3
    am_config.state_tying.type = "monophone-eow"
    am_config.tdp.entry_m1.loop = "infinity"
    am_config.tdp.entry_m2.loop = "infinity"
    am_config.tdp.scale = 1.0
    am_config.tdp["*"].exit = 0
    am_config.tdp["*"].forward = 0
    am_config.tdp["*"].loop = 0
    am_config.tdp["*"].skip = 0
    am_config.tdp.silence.exit = 0
    am_config.tdp.silence.forward = 0
    am_config.tdp.silence.loop = 0
    am_config.tdp.silence.skip = 0
    crp.acoustic_model_config = am_config

    lexicon_config = RasrConfig()
    lexicon_config.file = lexicon_path
    lexicon_config.normalize_pronunciation = False
    crp.lexicon_config = lexicon_config

    config, post_config = build_config_from_mapping(crp, {
      "corpus": "neural-network-trainer.corpus",
      "acoustic_model": "neural-network-trainer.model-combination.acoustic-model",
      "lexicon": "neural-network-trainer.model-combination.lexicon"})

    nn_trainer_config = RasrConfig()
    nn_trainer_config.action = "supervised-training"
    flow_path = "/u/schmitt/experiments/transducer/config/rasr-configs/feature_alignment.flow"
    nn_trainer_config.aligning_feature_extractor.feature_extraction.file = flow_path
    nn_trainer_config.aligning_feature_extractor.feature_extraction.feature_cache.path = feature_cache_path
    nn_trainer_config.aligning_feature_extractor.feature_extraction.alignment_cache.path = alignment_cache_path
    nn_trainer_config.buffer_size = 204800
    nn_trainer_config.buffer_type = "utterance"
    nn_trainer_config.class_labels.save_to_file = "class.labels"
    nn_trainer_config.estimator = "steepest-descent"
    nn_trainer_config.feature_extraction.file = "/u/schmitt/experiments/transducer/config/rasr-configs/dummy.flow"
    nn_trainer_config.regression_window_size = 5
    nn_trainer_config.shuffle = False
    nn_trainer_config.silence_weight = 1.0
    nn_trainer_config.single_precision = True
    nn_trainer_config.trainer_output_dimension = 88
    nn_trainer_config.training_criterion = "cross-entropy"
    nn_trainer_config.weighted_alignment = False
    nn_trainer_config.window_size = 1
    nn_trainer_config.window_size_derivatives = 0
    nn_trainer_config["*"].peak_position = 1.0
    nn_trainer_config["*"].peaky_alignment = False
    nn_trainer_config["*"].force_single_state = False
    nn_trainer_config["*"].reduce_alignment_factor = 1

    config.neural_network_trainer._update(nn_trainer_config)

    return RasrConfigBuilder._write_config(config=config, post_config=post_config)

  @staticmethod
  def get_feature_extraction_config(segment_path: Optional[Path], feature_cache_path: Path, corpus_path: Path) -> Path:
    crp = CommonRasrParameters()

    misc_config = RasrConfig()
    misc_config["*"]["window-size"] = 1
    misc_config["*"]["window-size-derivatives"] = 0
    misc_config["*"]["shuffle"] = "true"
    misc_config["*"]["job-name"] = "train"
    misc_config["*"]["use-cuda"] = "false"
    misc_config["*"]["action"] = "python-control"
    misc_config["*"]["python-control-enabled"] = "true"
    misc_config["*"]["python-control-loop-type"] = "iterate-corpus"
    misc_config["*"]["extract-alignments"] = "false"
    misc_config["*"]["python-segment-order"] = "true"
    misc_config["*"]["python-segment-order-pymod-path"] = "."
    misc_config["*"]["python-segment-order-pymod-name"] = "crnn.SprintInterface"
    misc_config["*"]["use-data-source"] = "false"
    misc_config["*"]["pymod-path"] = "."
    misc_config["*"]["pymod-name"] = "crnn.SprintInterface"
    misc_config["*"]["progress-indication"] = "global"
    misc_config["*"]["use-data-source"] = "false"

    misc_config["*"].configuration.channel = "log-channel"
    misc_config["*"].real_time_factor.channel = "log-channel"
    misc_config["*"].system_info.channel = "log-channel"
    misc_config["*"].time.channel = "log-channel"
    misc_config["*"].version.channel = "log-channel"

    misc_config["*"].log.channel = "log-channel"
    misc_config["*"].warning.channel = "log-channel, stderr"
    misc_config["*"].error.channel = "log-channel, stderr"

    misc_config["*"].statistics.channel = "log-channel"
    misc_config["*"].progress.channel = "log-channel"

    misc_config["*"].dot.channel = "nil"
    misc_config["*"].encoding = "UTF-8"
    misc_config["*"].log_channel.file = "sprint.log"

    corpus_config = RasrConfig()
    corpus_config.file = corpus_path
    if segment_path is not None:
      corpus_config.segments.file = segment_path
    corpus_config["segment-order-shuffle"] = True
    crp.corpus_config = corpus_config

    config, post_config = build_config_from_mapping(
      crp, {
        "corpus": "neural-network-trainer.corpus"
      }
    )

    feature_flow_path = Path("/u/schmitt/experiments/transducer/config/rasr-configs/feature.flow")
    config.neural_network_trainer.feature_extraction.file = feature_flow_path
    config.neural_network_trainer.feature_extraction.feature_cache.path = feature_cache_path
    config.neural_network_trainer.feature_extraction.feature_cache.read_only = True

    config._update(misc_config)

    return RasrConfigBuilder._write_config(config=config, post_config=post_config)

  @staticmethod
  def get_decoding_config(
          corpus_path: Path,
          segment_path: Path,
          lexicon_path: Path,
          feature_cache_path: Path,
          feature_extraction_file: Union[Path, str],
          label_file_path: Path,
          meta_graph_path: Path,
          simple_beam_search: bool,
          blank_label_index: int,
          reduction_factors: int,
          reduction_subtrahend: int,
          start_label_index: int,
          blank_update_history: bool,
          loop_update_history: bool,
          length_norm: bool,
          allow_label_recombination: bool,
          allow_word_end_recombination: bool,
          full_sum_decoding: bool,
          label_pruning: float,
          label_pruning_limit: int,
          word_end_pruning: float,
          word_end_pruning_limit: int,
          label_recombination_limit: int,
          max_seg_len: int,
          skip_silence: bool,
          native_lstm2_so_path: Path,
          max_batch_size: int = 256,
          label_scorer_type: str = "tf-rnn-transducer",
          lm_opts: Optional[Dict[str, Any]] = None,
          lm_lookahead_opts: Optional[Dict[str, Any]] = None,
          open_vocab: bool = True,
          debug: bool = False
  ) -> Tuple[CommonRasrParameters, RasrConfig]:
    crp = CommonRasrParameters()
    crp_add_default_output(crp, unbuffered=True)

    corpus_config = RasrConfigBuilder._get_corpus_config(corpus_path, segment_path)
    crp.corpus_config = corpus_config

    lexicon_config = RasrConfig()
    lexicon_config.file = lexicon_path
    lexicon_config.normalize_pronunciation = False
    crp.lexicon_config = lexicon_config

    recognizer_config = RasrConfig()
    recognizer_config.add_confidence_score = False
    recognizer_config.apply_non_word_closure_filter = False
    recognizer_config.apply_posterior_pruning = False
    recognizer_config.feature_extraction.file = feature_extraction_file
    recognizer_config.feature_extraction.feature_cache.path = feature_cache_path
    recognizer_config.links = "evaluator archive-writer"
    recognizer_config.pronunciation_scale = 1.0
    # recognizer_config.search_type = "label-sync-search"
    recognizer_config.search_type = "generic-seq2seq-tree-search"
    recognizer_config.type = "recognizer"
    recognizer_config.use_acoustic_model = False
    recognizer_config.debug = debug

    # LM
    if lm_opts is None:
      recognizer_config.lm.type = "simple-history"
      recognizer_config.recognizer.use_lm_score = False
    else:
      assert not open_vocab, "Currently, closed vocab is assumed for LM decoding."

      # basic settings
      recognizer_config.lm.scale = lm_opts["scale"]
      recognizer_config.lm.type = lm_opts["type"]
      if "image" in lm_opts:
        recognizer_config.lm.image = lm_opts["image"]
      if "file" in lm_opts:
        recognizer_config.lm.file = lm_opts["file"]
      if "allow_reduced_history" in lm_opts:
        recognizer_config.lm.allow_reduced_history = lm_opts["allow_reduced_history"]
      if "max_batch_size" in lm_opts:
        recognizer_config.lm.max_batch_size = lm_opts["max_batch_size"]
      if "min_batch_size" in lm_opts:
        recognizer_config.lm.min_batch_size = lm_opts["min_batch_size"]
      if "opt_batch_size" in lm_opts:
        recognizer_config.lm.opt_batch_size = lm_opts["opt_batch_size"]
      if "sort_batch_request" in lm_opts:
        recognizer_config.lm.sort_batch_request = lm_opts["sort_batch_request"]
      if "transform_output_negate" in lm_opts:
        recognizer_config.lm.transform_output_negate = lm_opts["transform_output_negate"]
      if "vocab_file" in lm_opts:
        recognizer_config.lm.vocab_file = lm_opts["vocab_file"]
      if "vocab_unknown_word" in lm_opts:
        recognizer_config.lm.vocab_unknown_word = lm_opts["vocab_unknown_word"]

      # tf settings
      if lm_opts["type"] == "tfrnn":
        # input map
        recognizer_config.lm.input_map.info_0.param_name = "word"
        recognizer_config.lm.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"
        recognizer_config.lm.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
        # output map
        recognizer_config.lm.output_map.info_0.param_name = "softmax"
        recognizer_config.lm.output_map.info_0.tensor_name = "output/output_batch_major"
        # loader
        recognizer_config.lm.loader.meta_graph_file = lm_opts["meta_graph_file"]
        recognizer_config.lm.loader.required_libraries = native_lstm2_so_path
        recognizer_config.lm.loader.type = "meta"
        recognizer_config.lm.loader.saved_model_file = lm_opts["saved_model_file"]

      recognizer_config.recognizer.use_lm_score = True

    # LM Lookahead
    if lm_lookahead_opts is None:
      recognizer_config.recognizer.lm_lookahead = False
    else:
      recognizer_config.recognizer.lm_lookahead.cache_size_high = lm_lookahead_opts["cache_size_high"]
      recognizer_config.recognizer.lm_lookahead.cache_size_low = lm_lookahead_opts["cache_size_low"]
      recognizer_config.recognizer.lm_lookahead.history_limit = lm_lookahead_opts["history_limit"]
      recognizer_config.recognizer.lm_lookahead.scale = lm_lookahead_opts["scale"]
      recognizer_config.recognizer.lm_lookahead = True
      if lm_lookahead_opts.get("separate_lookahead_lm", False):
        recognizer_config.recognizer.separate_lookahead_lm = True
        recognizer_config.recognizer.lookahead_lm.scale = 1.0
        recognizer_config.recognizer.lookahead_lm.type = lm_lookahead_opts["type"]
        recognizer_config.recognizer.lookahead_lm.file = lm_lookahead_opts["file"]
        recognizer_config.recognizer.lookahead_lm.image = lm_lookahead_opts["image"]


    # Open/Closed Vocab
    if not open_vocab:
      recognizer_config.recognizer.label_tree.label_unit = "phoneme"
      recognizer_config.acoustic_model.state_tying.type = "monophone"
      recognizer_config.acoustic_model.allophones.add_all = True
      recognizer_config.acoustic_model.allophones.add_from_lexicon = False
      recognizer_config.acoustic_model.hmm.across_word_model = True
      recognizer_config.acoustic_model.hmm.early_recombination = False
      recognizer_config.acoustic_model.hmm.state_repetitions = 1
      recognizer_config.acoustic_model.hmm.states_per_phone = 1
      recognizer_config.acoustic_model.phonology.future_length = 0
      recognizer_config.acoustic_model.phonology.history_length = 0
    else:
      recognizer_config.recognizer.label_tree.label_unit = "word"

    recognizer_config.label_scorer.blank_label_index = blank_label_index
    recognizer_config.label_scorer.label_file = label_file_path
    recognizer_config.label_scorer.label_scorer_type = label_scorer_type
    recognizer_config.label_scorer.max_batch_size = max_batch_size
    recognizer_config.label_scorer.reduction_factors = reduction_factors
    recognizer_config.label_scorer.reduction_subtrahend = reduction_subtrahend
    recognizer_config.label_scorer.scale = 1.0
    recognizer_config.label_scorer.start_label_index = start_label_index
    recognizer_config.label_scorer.transform_output_negate = True
    recognizer_config.label_scorer.use_start_label = True
    recognizer_config.label_scorer.blank_update_history = blank_update_history
    recognizer_config.label_scorer.loop_update_history = loop_update_history
    recognizer_config.label_scorer.position_dependent = False
    recognizer_config.label_scorer.blank_label_index = blank_label_index

    recognizer_config.label_scorer.feature_input_map.info_0.param_name = "feature"
    recognizer_config.label_scorer.feature_input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/data/data_dim0_size"
    recognizer_config.label_scorer.feature_input_map.info_0.tensor_name = "extern_data/placeholders/data/data"

    recognizer_config.label_scorer.loader.required_libraries = native_lstm2_so_path
    recognizer_config.label_scorer.loader.meta_graph_file = meta_graph_path
    recognizer_config.label_scorer.loader.type = "meta"

    recognizer_config.recognizer.allow_blank_label = True
    recognizer_config.recognizer.allow_label_recombination = allow_label_recombination
    recognizer_config.recognizer.allow_word_end_recombination = allow_word_end_recombination
    recognizer_config.recognizer.create_lattice = True
    recognizer_config.recognizer.full_sum_decoding = full_sum_decoding
    recognizer_config.recognizer.label_pruning = label_pruning
    recognizer_config.recognizer.label_pruning_limit = label_pruning_limit
    recognizer_config.recognizer.optimize_lattice = False
    recognizer_config.recognizer.word_end_pruning = word_end_pruning
    recognizer_config.recognizer.word_end_pruning_limit = word_end_pruning_limit
    recognizer_config.recognizer.debug = debug
    recognizer_config.recognizer.label_recombination_limit = label_recombination_limit
    recognizer_config.recognizer.fixed_beam_search = simple_beam_search
    recognizer_config.recognizer.simple_beam_search = simple_beam_search
    if max_seg_len is not None:
      recognizer_config.recognizer.max_segment_len = max_seg_len
    if length_norm:
      recognizer_config.recognizer.length_normalization = True

    recognizer_config.recognizer.label_tree.skip_silence = skip_silence
    crp.recognizer_config = recognizer_config

    flf_lattice_tool_config = RasrConfig()
    flf_lattice_tool_config.global_cache.file = "global.cache"
    flf_lattice_tool_config.network.initial_nodes = "segment"

    flf_lattice_tool_config.network.archive_writer.format = "flf"
    flf_lattice_tool_config.network.archive_writer.info = True
    flf_lattice_tool_config.network.archive_writer.links = "sink:1"
    flf_lattice_tool_config.network.archive_writer.type = "archive-writer"

    flf_lattice_tool_config.network.evaluator.best_in_lattice = True
    flf_lattice_tool_config.network.evaluator.links = "sink:0"
    flf_lattice_tool_config.network.evaluator.single_best = True
    flf_lattice_tool_config.network.evaluator.type = "evaluator"
    flf_lattice_tool_config.network.evaluator.word_errors = True
    flf_lattice_tool_config.network.evaluator.edit_distance.allow_broken_words = False
    flf_lattice_tool_config.network.evaluator.edit_distance.format = "bliss"

    flf_lattice_tool_config.network.segment.links = "1->recognizer:1 0->archive-writer:1 0->evaluator:1"
    flf_lattice_tool_config.network.segment.type = "speech-segment"

    flf_lattice_tool_config.network.sink.error_on_empty_lattice = False
    flf_lattice_tool_config.network.sink.type = "sink"
    flf_lattice_tool_config.network.sink.warn_on_empty_lattice = True

    return crp, flf_lattice_tool_config

  @staticmethod
  def get_lattice_to_ctm_config(
      corpus_path: Path, segment_path: Path, lexicon_path: Path, lattice_path: Path
  ) -> Tuple[CommonRasrParameters, RasrConfig]:
    crp = CommonRasrParameters()
    crp_add_default_output(crp, unbuffered=True)

    corpus_config = RasrConfigBuilder._get_corpus_config(corpus_path, segment_path)
    crp.corpus_config = corpus_config

    lexicon_config = RasrConfig()
    lexicon_config.file = lexicon_path
    lexicon_config.normalize_pronunciation = False
    crp.lexicon_config = lexicon_config

    flf_lattice_tool_config = RasrConfig()
    flf_lattice_tool_config.network.initial_nodes = "segment"

    flf_lattice_tool_config.network.add_word_confidence.confidence_key = "confidence"
    flf_lattice_tool_config.network.add_word_confidence.links = "best"
    flf_lattice_tool_config.network.add_word_confidence.rescore_mode = "in-place-cached"
    flf_lattice_tool_config.network.add_word_confidence.type = "fCN-features"

    flf_lattice_tool_config.network.archive_reader.format = "flf"
    flf_lattice_tool_config.network.archive_reader.links = "to-lemma"
    flf_lattice_tool_config.network.archive_reader.path = lattice_path
    flf_lattice_tool_config.network.archive_reader.type = "archive-reader"

    flf_lattice_tool_config.network.archive_reader.flf.append.confidence.scale = 0.0
    flf_lattice_tool_config.network.archive_reader.flf.append.keys = "confidence"

    flf_lattice_tool_config.network.best.algorithm = "bellman-ford"
    flf_lattice_tool_config.network.best.links = "dump-ctm"
    flf_lattice_tool_config.network.best.type = "best"

    flf_lattice_tool_config.network.dump_ctm.format = "ctm"
    flf_lattice_tool_config.network.dump_ctm.links = "sink:1"
    flf_lattice_tool_config.network.dump_ctm.type = "dump-traceback"

    flf_lattice_tool_config.network.dump_ctm.ctm.fill_empty_segments = True
    flf_lattice_tool_config.network.dump_ctm.ctm.scores = "am"

    flf_lattice_tool_config.network.segment.links = "0->archive-reader:1 0->dump-ctm:1"
    flf_lattice_tool_config.network.segment.type = "speech-segment"

    flf_lattice_tool_config.network.sink.error_on_empty_lattice = False
    flf_lattice_tool_config.network.sink.type = "sink"
    flf_lattice_tool_config.network.sink.warn_on_empty_lattice = True

    flf_lattice_tool_config.network.to_lemma.links = "add-word-confidence"
    flf_lattice_tool_config.network.to_lemma.map_input = "to-lemma"
    flf_lattice_tool_config.network.to_lemma.project_input = True
    flf_lattice_tool_config.network.to_lemma.type = "map-alphabet"

    return crp, flf_lattice_tool_config

  @staticmethod
  def get_realignment_config(
          corpus_path: Path,
          segment_path: Path,
          lexicon_path: Path,
          feature_cache_path: Optional[Path],
          feature_extraction_file: Union[Path, str],
          allophone_path: Path,
          label_pruning: float,
          label_pruning_limit: int,
          label_recombination_limit: int,
          blank_label_index: int,
          context_size: int,
          label_file: Path,
          reduction_factors: int,
          reduction_subtrahend: int,
          start_label_index: int,
          meta_graph_path: Path,
          state_tying_path: Path,
          max_segment_len: int,
          length_norm: bool,
          native_lstm2_so_path: Path,
          blank_update_history: bool = False,
          loop_update_history: bool = False,
  ) -> Tuple[CommonRasrParameters, RasrConfig]:
    crp = CommonRasrParameters()
    crp_add_default_output(crp, unbuffered=True)

    corpus_config = RasrConfigBuilder._get_corpus_config(corpus_path, segment_path)
    crp.corpus_config = corpus_config

    lexicon_config = RasrConfig()
    lexicon_config.file = lexicon_path
    lexicon_config.normalize_pronunciation = False
    crp.lexicon_config = lexicon_config

    am_config = RasrConfig()

    am_config.allophones.add_all = "yes"
    am_config.allophones.add_from_file = allophone_path
    am_config.allophones.add_from_lexicon = False

    am_config.state_tying.file = state_tying_path
    am_config.state_tying.type = "lookup"

    am_config.hmm.across_word_model = True
    am_config.hmm.early_recombination = False
    am_config.hmm.state_repetitions = 1
    am_config.hmm.states_per_phone = 1

    am_config.phonology.future_length = 0
    am_config.phonology.history_length = 0

    am_config.tdp.entry_m1.loop = "infinity"
    am_config.tdp.entry_m2.loop = "infinity"
    am_config.tdp.scale = 1.0
    am_config.tdp["*"].exit = 0
    am_config.tdp["*"].forward = 0
    am_config.tdp["*"].loop = "infinity"
    am_config.tdp["*"].skip = "infinity"
    crp.acoustic_model_config = am_config

    am_model_trainer_config = RasrConfig()

    am_model_trainer_config.alignment.aligner.label_pruning = label_pruning
    am_model_trainer_config.alignment.aligner.label_pruning_limit = label_pruning_limit
    am_model_trainer_config.alignment.aligner.label_recombination_limit = label_recombination_limit
    am_model_trainer_config.alignment.aligner.max_segment_len = max_segment_len
    am_model_trainer_config.alignment.aligner.length_normalization = length_norm

    am_model_trainer_config.alignment.label_scorer.blank_label_index = blank_label_index
    am_model_trainer_config.alignment.label_scorer.context_size = context_size
    am_model_trainer_config.alignment.label_scorer.label_file = label_file
    am_model_trainer_config.alignment.label_scorer.label_scorer_type = "tf-rnn-transducer"
    am_model_trainer_config.alignment.label_scorer.max_batch_size = 256
    am_model_trainer_config.alignment.label_scorer.reduction_factors = reduction_factors
    am_model_trainer_config.alignment.label_scorer.reduction_subtrahend = reduction_subtrahend
    am_model_trainer_config.alignment.label_scorer.scale = 1.0
    am_model_trainer_config.alignment.label_scorer.start_label_index = start_label_index
    am_model_trainer_config.alignment.label_scorer.transform_output_negate = True
    am_model_trainer_config.alignment.label_scorer.use_start_label = True
    if blank_update_history:
      am_model_trainer_config.alignment.label_scorer.blank_update_history = True
    if loop_update_history:
      am_model_trainer_config.alignment.label_scorer.loop_update_history = True

    am_model_trainer_config.alignment.label_scorer.feature_input_map.info_0.param_name = "feature"
    am_model_trainer_config.alignment.label_scorer.feature_input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/data/data_dim0_size"
    am_model_trainer_config.alignment.label_scorer.feature_input_map.info_0.tensor_name = "extern_data/placeholders/data/data"
    am_model_trainer_config.alignment.label_scorer.loader.meta_graph_file = meta_graph_path
    # am_model_trainer_config.alignment.label_scorer.loader.required_libraries = Path("/u/rossenbach/temp/ctc_speedtest/tf_new_native_mkl/NativeLstm2.so")
    am_model_trainer_config.alignment.label_scorer.loader.required_libraries = native_lstm2_so_path
    am_model_trainer_config.alignment.label_scorer.loader.type = "meta"

    # flow_path = "/u/schmitt/experiments/transducer/config/rasr-configs/realignment.flow"
    am_model_trainer_config.file = feature_extraction_file
    if feature_cache_path is not None:
      am_model_trainer_config.feature_cache.path = feature_cache_path

    return crp, am_model_trainer_config

  @staticmethod
  def get_alignment_extraction_config(
          allophone_path: Path,
          state_tying_path: Path,
          lexicon_path: Path,
          corpus_path: Path,
          segment_path: Path,
          feature_cache_path: Path,
          alignment_cache_path: Path
  ) -> Tuple[RasrConfig, RasrConfig]:
    crp = CommonRasrParameters()
    crp_add_default_output(crp, unbuffered=True)

    corpus_config = RasrConfigBuilder._get_corpus_config(corpus_path=corpus_path, segment_path=segment_path)
    crp.corpus_config = corpus_config

    am_config = RasrConfig()
    am_config.allophones.add_all = True
    am_config.allophones.add_from_file = allophone_path
    am_config.allophones.add_from_lexicon = False
    am_config.hmm.across_word_model = True
    am_config.hmm.early_recombination = False
    am_config.hmm.state_repetitions = 1
    am_config.hmm.states_per_phone = 1
    am_config.phonology.future_length = 0
    am_config.phonology.history_length = 0
    am_config.state_tying.file = state_tying_path
    am_config.state_tying.type = "lookup"
    am_config.tdp.entry_m1.loop = "infinity"
    am_config.tdp.entry_m2.loop = "infinity"
    am_config.tdp.scale = 1.0
    am_config.tdp["*"].exit = 0
    am_config.tdp["*"].forward = 0
    am_config.tdp["*"].loop = "infinity"
    am_config.tdp["*"].skip = "infinity"
    crp.acoustic_model_config = am_config

    lexicon_config = RasrConfig()
    lexicon_config.file = lexicon_path
    lexicon_config.normalize_pronunciation = False
    crp.lexicon_config = lexicon_config

    config, post_config = build_config_from_mapping(crp, {
      "corpus": "neural-network-trainer.corpus",
      "acoustic_model": "neural-network-trainer.model-combination.acoustic-model",
      "lexicon": "neural-network-trainer.model-combination.lexicon"})

    nn_trainer_config = RasrConfig()
    nn_trainer_config.action = "supervised-training"
    flow_path = "/u/schmitt/experiments/transducer/config/rasr-configs/feature_alignment.flow"
    nn_trainer_config.aligning_feature_extractor.feature_extraction.file = flow_path
    nn_trainer_config.aligning_feature_extractor.feature_extraction.feature_cache.path = feature_cache_path
    nn_trainer_config.aligning_feature_extractor.feature_extraction.alignment_cache.path = alignment_cache_path
    nn_trainer_config.buffer_size = 204800
    nn_trainer_config.buffer_type = "utterance"
    nn_trainer_config.class_labels.save_to_file = "class.labels"
    nn_trainer_config.estimator = "steepest-descent"
    nn_trainer_config.feature_extraction.file = "/u/schmitt/experiments/transducer/config/rasr-configs/dummy.flow"
    nn_trainer_config.regression_window_size = 5
    nn_trainer_config.shuffle = False
    nn_trainer_config.silence_weight = 1.0
    nn_trainer_config.single_precision = True
    nn_trainer_config.trainer_output_dimension = 88
    nn_trainer_config.training_criterion = "cross-entropy"
    nn_trainer_config.weighted_alignment = False
    nn_trainer_config.window_size = 1
    nn_trainer_config.window_size_derivatives = 0
    nn_trainer_config["*"].reduce_alignment_factor = 6

    config.neural_network_trainer._update(nn_trainer_config)

    return config, post_config
