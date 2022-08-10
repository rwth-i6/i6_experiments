"""
Auto-generated code via dependency_boundary.
Do not modify by hand!
"""

import i6_experiments.common.setups.rasr.util.rasr
_rasr_init_args = object.__new__(i6_experiments.common.setups.rasr.util.rasr.RasrInitArgs)
_rasr_init_args.costa_args = {
    'eval_recordings': True,
    'eval_lm': False,
}
_rasr_init_args.scorer = None
_rasr_init_args.scorer_args = None
_rasr_init_args.am_args = {
    'state_tying': 'monophone',
    'states_per_phone': 3,
    'state_repetitions': 1,
    'across_word_model': True,
    'early_recombination': False,
    'tdp_scale': 1.0,
    'tdp_transition': (3.0, 0.0, 30.0, 0.0, ),
    'tdp_silence': (0.0, 3.0, 'infinity', 20.0, ),
    'tying_type': 'global',
    'nonword_phones': '',
    'tdp_nonword': (0.0, 3.0, 'infinity', 6.0, ),
}
_dict2 = {
    'audio_format': 'wav',
    'dc_detection': True,
}
_dict3 = {
    'normalize': False,
    'outputs': 16,
    'add_epsilon': False,
}
_dict1 = {
    'warping_function': 'mel',
    'filter_width': 268.258,
    'normalize': True,
    'normalization_options': None,
    'without_samples': False,
    'samples_options': _dict2,
    'cepstrum_options': _dict3,
    'fft_options': None,
}
_dict = {
    'num_deriv': 2,
    'num_features': None,
    'mfcc_options': _dict1,
}
_dict6 = {
    'audio_format': 'wav',
    'dc_detection': True,
}
_dict7 = {
}
_dict5 = {
    'minfreq': 100,
    'maxfreq': 7500,
    'channels': 50,
    'tempint_type': 'hanning',
    'tempint_shift': 0.01,
    'tempint_length': 0.025,
    'flush_before_gap': True,
    'do_specint': False,
    'specint_type': 'hanning',
    'specint_shift': 4,
    'specint_length': 9,
    'normalize': True,
    'preemphasis': True,
    'legacy_scaling': False,
    'without_samples': False,
    'samples_options': _dict6,
    'normalization_options': _dict7,
}
_dict4 = {
    'gt_options': _dict5,
}
_dict10 = {
    'audio_format': 'wav',
    'dc_detection': True,
}
_dict11 = {
}
_dict9 = {
    'without_samples': False,
    'samples_options': _dict10,
    'fft_options': _dict11,
}
_dict8 = {
    'energy_options': _dict9,
}
_rasr_init_args.feature_extraction_args = {
    'mfcc': _dict,
    'gt': _dict4,
    'energy': _dict8,
}
_rasr_init_args.stm_args = None
import i6_experiments.common.setups.rasr.util.nn
_returnn_rasr_data_input = object.__new__(i6_experiments.common.setups.rasr.util.nn.ReturnnRasrDataInput)
_returnn_rasr_data_input.name = 'init'
import i6_core.rasr as rasr
_returnn_rasr_data_input_crp_base = rasr.CommonRasrParameters()
_returnn_rasr_data_input_crp_base.acoustic_model_config = rasr.RasrConfig()
_returnn_rasr_data_input_crp_base.acoustic_model_config.state_tying.type = 'cart'
from i6_experiments.common.utils.dump_py_code import _make_fake_job as make_fake_job
_estimate_cart_job = make_fake_job(module='i6_core.cart.estimate', name='EstimateCartJob', sis_hash='GUv9i8tzV7DN')
from sisyphus import tk
_returnn_rasr_data_input_crp_base.acoustic_model_config.state_tying.file = tk.Path('cart.tree.xml.gz', creator=_estimate_cart_job)
_returnn_rasr_data_input_crp_base.acoustic_model_config.allophones.add_from_lexicon = True
_returnn_rasr_data_input_crp_base.acoustic_model_config.allophones.add_all = False
_returnn_rasr_data_input_crp_base.acoustic_model_config.hmm.states_per_phone = 3
_returnn_rasr_data_input_crp_base.acoustic_model_config.hmm.state_repetitions = 1
_returnn_rasr_data_input_crp_base.acoustic_model_config.hmm.across_word_model = True
_returnn_rasr_data_input_crp_base.acoustic_model_config.hmm.early_recombination = False
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp.scale = 1.0
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp['*'].loop = 3.0
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp['*'].forward = 0.0
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp['*'].skip = 30.0
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp['*'].exit = 0.0
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp.silence.loop = 0.0
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp.silence.forward = 3.0
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp.silence.skip = 'infinity'
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp.silence.exit = 20.0
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp.entry_m1.loop = 'infinity'
_returnn_rasr_data_input_crp_base.acoustic_model_config.tdp.entry_m2.loop = 'infinity'
_returnn_rasr_data_input_crp_base.acoustic_model_post_config = None
_returnn_rasr_data_input_crp_base.corpus_config = None
_returnn_rasr_data_input_crp_base.corpus_post_config = None
_returnn_rasr_data_input_crp_base.lexicon_config = None
_returnn_rasr_data_input_crp_base.lexicon_post_config = None
_returnn_rasr_data_input_crp_base.language_model_config = None
_returnn_rasr_data_input_crp_base.language_model_post_config = None
_returnn_rasr_data_input_crp_base.recognizer_config = None
_returnn_rasr_data_input_crp_base.recognizer_post_config = None
_returnn_rasr_data_input_crp_base.log_config = rasr.RasrConfig()
_returnn_rasr_data_input_crp_base.log_config['*'].configuration.channel = 'output-channel'
_returnn_rasr_data_input_crp_base.log_config['*'].real_time_factor.channel = 'output-channel'
_returnn_rasr_data_input_crp_base.log_config['*'].system_info.channel = 'output-channel'
_returnn_rasr_data_input_crp_base.log_config['*'].time.channel = 'output-channel'
_returnn_rasr_data_input_crp_base.log_config['*'].version.channel = 'output-channel'
_returnn_rasr_data_input_crp_base.log_config['*'].log.channel = 'output-channel'
_returnn_rasr_data_input_crp_base.log_config['*'].warning.channel = 'output-channel, stderr'
_returnn_rasr_data_input_crp_base.log_config['*'].error.channel = 'output-channel, stderr'
_returnn_rasr_data_input_crp_base.log_config['*'].statistics.channel = 'output-channel'
_returnn_rasr_data_input_crp_base.log_config['*'].progress.channel = 'output-channel'
_returnn_rasr_data_input_crp_base.log_config['*'].dot.channel = 'nil'
_returnn_rasr_data_input_crp_base.log_post_config = rasr.RasrConfig()
_returnn_rasr_data_input_crp_base.log_post_config['*'].encoding = 'UTF-8'
_returnn_rasr_data_input_crp_base.log_post_config['*'].output_channel.file = '$(LOGFILE)'
_returnn_rasr_data_input_crp_base.log_post_config['*'].output_channel.compressed = False
_returnn_rasr_data_input_crp_base.log_post_config['*'].output_channel.append = False
_returnn_rasr_data_input_crp_base.log_post_config['*'].output_channel.unbuffered = False
_returnn_rasr_data_input_crp_base.compress_log_file = True
_returnn_rasr_data_input_crp_base.default_log_channel = 'output-channel'
_returnn_rasr_data_input_crp_base.audio_format = 'wav'
_returnn_rasr_data_input_crp_base.corpus_duration = 1.0
_returnn_rasr_data_input_crp_base.concurrent = 1
_returnn_rasr_data_input_crp_base.segment_path = None
_returnn_rasr_data_input_crp_base.acoustic_model_trainer_exe = None
_returnn_rasr_data_input_crp_base.allophone_tool_exe = None
_returnn_rasr_data_input_crp_base.costa_exe = None
_returnn_rasr_data_input_crp_base.feature_extraction_exe = None
_returnn_rasr_data_input_crp_base.feature_statistics_exe = None
_returnn_rasr_data_input_crp_base.flf_tool_exe = None
_returnn_rasr_data_input_crp_base.kws_tool_exe = None
_returnn_rasr_data_input_crp_base.lattice_processor_exe = None
_returnn_rasr_data_input_crp_base.lm_util_exe = None
_returnn_rasr_data_input_crp_base.nn_trainer_exe = None
_returnn_rasr_data_input_crp_base.speech_recognizer_exe = None
_returnn_rasr_data_input_crp_base.python_home = None
_returnn_rasr_data_input_crp_base.python_program_name = None
_returnn_rasr_data_input.crp = rasr.CommonRasrParameters(_returnn_rasr_data_input_crp_base)
_returnn_rasr_data_input.crp.corpus_config = rasr.RasrConfig()
_merge_corpora_job = make_fake_job(module='i6_core.corpus.transform', name='MergeCorporaJob', sis_hash='MFQmNDQlxmAB')
_returnn_rasr_data_input.crp.corpus_config.file = tk.Path('merged.xml.gz', creator=_merge_corpora_job)
_returnn_rasr_data_input.crp.corpus_config.audio_dir = None
_returnn_rasr_data_input.crp.corpus_config.warn_about_unexpected_elements = True
_returnn_rasr_data_input.crp.corpus_config.capitalize_transcriptions = False
_returnn_rasr_data_input.crp.corpus_config.progress_indication = 'global'
_returnn_rasr_data_input.crp.corpus_config.segment_order_shuffle = True
_returnn_rasr_data_input.crp.corpus_config.segment_order_sort_by_time_length = True
_returnn_rasr_data_input.crp.corpus_config.segment_order_sort_by_time_length_chunk_size = 384
_returnn_rasr_data_input.crp.audio_format = 'wav'
_returnn_rasr_data_input.crp.corpus_duration = 960.9000000000001
_returnn_rasr_data_input.crp.concurrent = 1
_shuffle_and_split_segments_job = make_fake_job(module='i6_core.corpus.segments', name='ShuffleAndSplitSegmentsJob', sis_hash='hPMsdZr1PSjY')
_returnn_rasr_data_input.crp.segment_path = tk.Path('train.segments', creator=_shuffle_and_split_segments_job)
from sisyphus import gs
import os
_returnn_rasr_data_input.crp.acoustic_model_trainer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/acoustic-model-trainer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/acoustic-model-trainer.linux-x86_64-standard', ))
_returnn_rasr_data_input.crp.allophone_tool_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/allophone-tool.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/allophone-tool.linux-x86_64-standard', ))
_returnn_rasr_data_input.crp.costa_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/costa.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/costa.linux-x86_64-standard', ))
_returnn_rasr_data_input.crp.feature_extraction_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/feature-extraction.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/feature-extraction.linux-x86_64-standard', ))
_returnn_rasr_data_input.crp.feature_statistics_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/feature-statistics.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/feature-statistics.linux-x86_64-standard', ))
_returnn_rasr_data_input.crp.flf_tool_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/flf-tool.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/flf-tool.linux-x86_64-standard', ))
_returnn_rasr_data_input.crp.kws_tool_exe = None
_returnn_rasr_data_input.crp.lattice_processor_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/lattice-processor.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/lattice-processor.linux-x86_64-standard', ))
_returnn_rasr_data_input.crp.lm_util_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/lm-util.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/lm-util.linux-x86_64-standard', ))
_returnn_rasr_data_input.crp.nn_trainer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/nn-trainer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/nn-trainer.linux-x86_64-standard', ))
_returnn_rasr_data_input.crp.speech_recognizer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/speech-recognizer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/speech-recognizer.linux-x86_64-standard', ))
_returnn_rasr_data_input.crp.lexicon_config = rasr.RasrConfig()
_g2_p_output_to_bliss_lexicon_job = make_fake_job(module='i6_core.g2p.convert', name='G2POutputToBlissLexiconJob', sis_hash='SIIDsOAhK3bA')
_returnn_rasr_data_input.crp.lexicon_config.file = tk.Path('oov.lexicon.gz', creator=_g2_p_output_to_bliss_lexicon_job, cached=True)
_returnn_rasr_data_input.crp.lexicon_config.normalize_pronunciation = False
_returnn_rasr_data_input.crp.acoustic_model_post_config = rasr.RasrConfig()
_store_allophones_job = make_fake_job(module='i6_core.lexicon.allophones', name='StoreAllophonesJob', sis_hash='w3QrnaS2VbXx')
_returnn_rasr_data_input.crp.acoustic_model_post_config.allophones.add_from_file = tk.Path('allophones', creator=_store_allophones_job)
import i6_core.rasr.flow
_returnn_rasr_data_input.alignments = object.__new__(i6_core.rasr.flow.FlagDependentFlowAttribute)
_returnn_rasr_data_input.alignments.flag = 'cache_mode'
_alignment_job = make_fake_job(module='i6_core.mm.alignment', name='AlignmentJob', sis_hash='kJlXQfRnYiCX')
_dict13 = {
    1: tk.Path('alignment.cache.1', creator=_alignment_job, cached=True),
    2: tk.Path('alignment.cache.2', creator=_alignment_job, cached=True),
    3: tk.Path('alignment.cache.3', creator=_alignment_job, cached=True),
    4: tk.Path('alignment.cache.4', creator=_alignment_job, cached=True),
    5: tk.Path('alignment.cache.5', creator=_alignment_job, cached=True),
    6: tk.Path('alignment.cache.6', creator=_alignment_job, cached=True),
    7: tk.Path('alignment.cache.7', creator=_alignment_job, cached=True),
    8: tk.Path('alignment.cache.8', creator=_alignment_job, cached=True),
    9: tk.Path('alignment.cache.9', creator=_alignment_job, cached=True),
    10: tk.Path('alignment.cache.10', creator=_alignment_job, cached=True),
    11: tk.Path('alignment.cache.11', creator=_alignment_job, cached=True),
    12: tk.Path('alignment.cache.12', creator=_alignment_job, cached=True),
    13: tk.Path('alignment.cache.13', creator=_alignment_job, cached=True),
    14: tk.Path('alignment.cache.14', creator=_alignment_job, cached=True),
    15: tk.Path('alignment.cache.15', creator=_alignment_job, cached=True),
    16: tk.Path('alignment.cache.16', creator=_alignment_job, cached=True),
    17: tk.Path('alignment.cache.17', creator=_alignment_job, cached=True),
    18: tk.Path('alignment.cache.18', creator=_alignment_job, cached=True),
    19: tk.Path('alignment.cache.19', creator=_alignment_job, cached=True),
    20: tk.Path('alignment.cache.20', creator=_alignment_job, cached=True),
    21: tk.Path('alignment.cache.21', creator=_alignment_job, cached=True),
    22: tk.Path('alignment.cache.22', creator=_alignment_job, cached=True),
    23: tk.Path('alignment.cache.23', creator=_alignment_job, cached=True),
    24: tk.Path('alignment.cache.24', creator=_alignment_job, cached=True),
    25: tk.Path('alignment.cache.25', creator=_alignment_job, cached=True),
    26: tk.Path('alignment.cache.26', creator=_alignment_job, cached=True),
    27: tk.Path('alignment.cache.27', creator=_alignment_job, cached=True),
    28: tk.Path('alignment.cache.28', creator=_alignment_job, cached=True),
    29: tk.Path('alignment.cache.29', creator=_alignment_job, cached=True),
    30: tk.Path('alignment.cache.30', creator=_alignment_job, cached=True),
    31: tk.Path('alignment.cache.31', creator=_alignment_job, cached=True),
    32: tk.Path('alignment.cache.32', creator=_alignment_job, cached=True),
    33: tk.Path('alignment.cache.33', creator=_alignment_job, cached=True),
    34: tk.Path('alignment.cache.34', creator=_alignment_job, cached=True),
    35: tk.Path('alignment.cache.35', creator=_alignment_job, cached=True),
    36: tk.Path('alignment.cache.36', creator=_alignment_job, cached=True),
    37: tk.Path('alignment.cache.37', creator=_alignment_job, cached=True),
    38: tk.Path('alignment.cache.38', creator=_alignment_job, cached=True),
    39: tk.Path('alignment.cache.39', creator=_alignment_job, cached=True),
    40: tk.Path('alignment.cache.40', creator=_alignment_job, cached=True),
    41: tk.Path('alignment.cache.41', creator=_alignment_job, cached=True),
    42: tk.Path('alignment.cache.42', creator=_alignment_job, cached=True),
    43: tk.Path('alignment.cache.43', creator=_alignment_job, cached=True),
    44: tk.Path('alignment.cache.44', creator=_alignment_job, cached=True),
    45: tk.Path('alignment.cache.45', creator=_alignment_job, cached=True),
    46: tk.Path('alignment.cache.46', creator=_alignment_job, cached=True),
    47: tk.Path('alignment.cache.47', creator=_alignment_job, cached=True),
    48: tk.Path('alignment.cache.48', creator=_alignment_job, cached=True),
    49: tk.Path('alignment.cache.49', creator=_alignment_job, cached=True),
    50: tk.Path('alignment.cache.50', creator=_alignment_job, cached=True),
    51: tk.Path('alignment.cache.51', creator=_alignment_job, cached=True),
    52: tk.Path('alignment.cache.52', creator=_alignment_job, cached=True),
    53: tk.Path('alignment.cache.53', creator=_alignment_job, cached=True),
    54: tk.Path('alignment.cache.54', creator=_alignment_job, cached=True),
    55: tk.Path('alignment.cache.55', creator=_alignment_job, cached=True),
    56: tk.Path('alignment.cache.56', creator=_alignment_job, cached=True),
    57: tk.Path('alignment.cache.57', creator=_alignment_job, cached=True),
    58: tk.Path('alignment.cache.58', creator=_alignment_job, cached=True),
    59: tk.Path('alignment.cache.59', creator=_alignment_job, cached=True),
    60: tk.Path('alignment.cache.60', creator=_alignment_job, cached=True),
    61: tk.Path('alignment.cache.61', creator=_alignment_job, cached=True),
    62: tk.Path('alignment.cache.62', creator=_alignment_job, cached=True),
    63: tk.Path('alignment.cache.63', creator=_alignment_job, cached=True),
    64: tk.Path('alignment.cache.64', creator=_alignment_job, cached=True),
    65: tk.Path('alignment.cache.65', creator=_alignment_job, cached=True),
    66: tk.Path('alignment.cache.66', creator=_alignment_job, cached=True),
    67: tk.Path('alignment.cache.67', creator=_alignment_job, cached=True),
    68: tk.Path('alignment.cache.68', creator=_alignment_job, cached=True),
    69: tk.Path('alignment.cache.69', creator=_alignment_job, cached=True),
    70: tk.Path('alignment.cache.70', creator=_alignment_job, cached=True),
    71: tk.Path('alignment.cache.71', creator=_alignment_job, cached=True),
    72: tk.Path('alignment.cache.72', creator=_alignment_job, cached=True),
    73: tk.Path('alignment.cache.73', creator=_alignment_job, cached=True),
    74: tk.Path('alignment.cache.74', creator=_alignment_job, cached=True),
    75: tk.Path('alignment.cache.75', creator=_alignment_job, cached=True),
    76: tk.Path('alignment.cache.76', creator=_alignment_job, cached=True),
    77: tk.Path('alignment.cache.77', creator=_alignment_job, cached=True),
    78: tk.Path('alignment.cache.78', creator=_alignment_job, cached=True),
    79: tk.Path('alignment.cache.79', creator=_alignment_job, cached=True),
    80: tk.Path('alignment.cache.80', creator=_alignment_job, cached=True),
    81: tk.Path('alignment.cache.81', creator=_alignment_job, cached=True),
    82: tk.Path('alignment.cache.82', creator=_alignment_job, cached=True),
    83: tk.Path('alignment.cache.83', creator=_alignment_job, cached=True),
    84: tk.Path('alignment.cache.84', creator=_alignment_job, cached=True),
    85: tk.Path('alignment.cache.85', creator=_alignment_job, cached=True),
    86: tk.Path('alignment.cache.86', creator=_alignment_job, cached=True),
    87: tk.Path('alignment.cache.87', creator=_alignment_job, cached=True),
    88: tk.Path('alignment.cache.88', creator=_alignment_job, cached=True),
    89: tk.Path('alignment.cache.89', creator=_alignment_job, cached=True),
    90: tk.Path('alignment.cache.90', creator=_alignment_job, cached=True),
    91: tk.Path('alignment.cache.91', creator=_alignment_job, cached=True),
    92: tk.Path('alignment.cache.92', creator=_alignment_job, cached=True),
    93: tk.Path('alignment.cache.93', creator=_alignment_job, cached=True),
    94: tk.Path('alignment.cache.94', creator=_alignment_job, cached=True),
    95: tk.Path('alignment.cache.95', creator=_alignment_job, cached=True),
    96: tk.Path('alignment.cache.96', creator=_alignment_job, cached=True),
    97: tk.Path('alignment.cache.97', creator=_alignment_job, cached=True),
    98: tk.Path('alignment.cache.98', creator=_alignment_job, cached=True),
    99: tk.Path('alignment.cache.99', creator=_alignment_job, cached=True),
    100: tk.Path('alignment.cache.100', creator=_alignment_job, cached=True),
    101: tk.Path('alignment.cache.101', creator=_alignment_job, cached=True),
    102: tk.Path('alignment.cache.102', creator=_alignment_job, cached=True),
    103: tk.Path('alignment.cache.103', creator=_alignment_job, cached=True),
    104: tk.Path('alignment.cache.104', creator=_alignment_job, cached=True),
    105: tk.Path('alignment.cache.105', creator=_alignment_job, cached=True),
    106: tk.Path('alignment.cache.106', creator=_alignment_job, cached=True),
    107: tk.Path('alignment.cache.107', creator=_alignment_job, cached=True),
    108: tk.Path('alignment.cache.108', creator=_alignment_job, cached=True),
    109: tk.Path('alignment.cache.109', creator=_alignment_job, cached=True),
    110: tk.Path('alignment.cache.110', creator=_alignment_job, cached=True),
    111: tk.Path('alignment.cache.111', creator=_alignment_job, cached=True),
    112: tk.Path('alignment.cache.112', creator=_alignment_job, cached=True),
    113: tk.Path('alignment.cache.113', creator=_alignment_job, cached=True),
    114: tk.Path('alignment.cache.114', creator=_alignment_job, cached=True),
    115: tk.Path('alignment.cache.115', creator=_alignment_job, cached=True),
    116: tk.Path('alignment.cache.116', creator=_alignment_job, cached=True),
    117: tk.Path('alignment.cache.117', creator=_alignment_job, cached=True),
    118: tk.Path('alignment.cache.118', creator=_alignment_job, cached=True),
    119: tk.Path('alignment.cache.119', creator=_alignment_job, cached=True),
    120: tk.Path('alignment.cache.120', creator=_alignment_job, cached=True),
    121: tk.Path('alignment.cache.121', creator=_alignment_job, cached=True),
    122: tk.Path('alignment.cache.122', creator=_alignment_job, cached=True),
    123: tk.Path('alignment.cache.123', creator=_alignment_job, cached=True),
    124: tk.Path('alignment.cache.124', creator=_alignment_job, cached=True),
    125: tk.Path('alignment.cache.125', creator=_alignment_job, cached=True),
    126: tk.Path('alignment.cache.126', creator=_alignment_job, cached=True),
    127: tk.Path('alignment.cache.127', creator=_alignment_job, cached=True),
    128: tk.Path('alignment.cache.128', creator=_alignment_job, cached=True),
    129: tk.Path('alignment.cache.129', creator=_alignment_job, cached=True),
    130: tk.Path('alignment.cache.130', creator=_alignment_job, cached=True),
    131: tk.Path('alignment.cache.131', creator=_alignment_job, cached=True),
    132: tk.Path('alignment.cache.132', creator=_alignment_job, cached=True),
    133: tk.Path('alignment.cache.133', creator=_alignment_job, cached=True),
    134: tk.Path('alignment.cache.134', creator=_alignment_job, cached=True),
    135: tk.Path('alignment.cache.135', creator=_alignment_job, cached=True),
    136: tk.Path('alignment.cache.136', creator=_alignment_job, cached=True),
    137: tk.Path('alignment.cache.137', creator=_alignment_job, cached=True),
    138: tk.Path('alignment.cache.138', creator=_alignment_job, cached=True),
    139: tk.Path('alignment.cache.139', creator=_alignment_job, cached=True),
    140: tk.Path('alignment.cache.140', creator=_alignment_job, cached=True),
    141: tk.Path('alignment.cache.141', creator=_alignment_job, cached=True),
    142: tk.Path('alignment.cache.142', creator=_alignment_job, cached=True),
    143: tk.Path('alignment.cache.143', creator=_alignment_job, cached=True),
    144: tk.Path('alignment.cache.144', creator=_alignment_job, cached=True),
    145: tk.Path('alignment.cache.145', creator=_alignment_job, cached=True),
    146: tk.Path('alignment.cache.146', creator=_alignment_job, cached=True),
    147: tk.Path('alignment.cache.147', creator=_alignment_job, cached=True),
    148: tk.Path('alignment.cache.148', creator=_alignment_job, cached=True),
    149: tk.Path('alignment.cache.149', creator=_alignment_job, cached=True),
    150: tk.Path('alignment.cache.150', creator=_alignment_job, cached=True),
    151: tk.Path('alignment.cache.151', creator=_alignment_job, cached=True),
    152: tk.Path('alignment.cache.152', creator=_alignment_job, cached=True),
    153: tk.Path('alignment.cache.153', creator=_alignment_job, cached=True),
    154: tk.Path('alignment.cache.154', creator=_alignment_job, cached=True),
    155: tk.Path('alignment.cache.155', creator=_alignment_job, cached=True),
    156: tk.Path('alignment.cache.156', creator=_alignment_job, cached=True),
    157: tk.Path('alignment.cache.157', creator=_alignment_job, cached=True),
    158: tk.Path('alignment.cache.158', creator=_alignment_job, cached=True),
    159: tk.Path('alignment.cache.159', creator=_alignment_job, cached=True),
    160: tk.Path('alignment.cache.160', creator=_alignment_job, cached=True),
    161: tk.Path('alignment.cache.161', creator=_alignment_job, cached=True),
    162: tk.Path('alignment.cache.162', creator=_alignment_job, cached=True),
    163: tk.Path('alignment.cache.163', creator=_alignment_job, cached=True),
    164: tk.Path('alignment.cache.164', creator=_alignment_job, cached=True),
    165: tk.Path('alignment.cache.165', creator=_alignment_job, cached=True),
    166: tk.Path('alignment.cache.166', creator=_alignment_job, cached=True),
    167: tk.Path('alignment.cache.167', creator=_alignment_job, cached=True),
    168: tk.Path('alignment.cache.168', creator=_alignment_job, cached=True),
    169: tk.Path('alignment.cache.169', creator=_alignment_job, cached=True),
    170: tk.Path('alignment.cache.170', creator=_alignment_job, cached=True),
    171: tk.Path('alignment.cache.171', creator=_alignment_job, cached=True),
    172: tk.Path('alignment.cache.172', creator=_alignment_job, cached=True),
    173: tk.Path('alignment.cache.173', creator=_alignment_job, cached=True),
    174: tk.Path('alignment.cache.174', creator=_alignment_job, cached=True),
    175: tk.Path('alignment.cache.175', creator=_alignment_job, cached=True),
    176: tk.Path('alignment.cache.176', creator=_alignment_job, cached=True),
    177: tk.Path('alignment.cache.177', creator=_alignment_job, cached=True),
    178: tk.Path('alignment.cache.178', creator=_alignment_job, cached=True),
    179: tk.Path('alignment.cache.179', creator=_alignment_job, cached=True),
    180: tk.Path('alignment.cache.180', creator=_alignment_job, cached=True),
    181: tk.Path('alignment.cache.181', creator=_alignment_job, cached=True),
    182: tk.Path('alignment.cache.182', creator=_alignment_job, cached=True),
    183: tk.Path('alignment.cache.183', creator=_alignment_job, cached=True),
    184: tk.Path('alignment.cache.184', creator=_alignment_job, cached=True),
    185: tk.Path('alignment.cache.185', creator=_alignment_job, cached=True),
    186: tk.Path('alignment.cache.186', creator=_alignment_job, cached=True),
    187: tk.Path('alignment.cache.187', creator=_alignment_job, cached=True),
    188: tk.Path('alignment.cache.188', creator=_alignment_job, cached=True),
    189: tk.Path('alignment.cache.189', creator=_alignment_job, cached=True),
    190: tk.Path('alignment.cache.190', creator=_alignment_job, cached=True),
    191: tk.Path('alignment.cache.191', creator=_alignment_job, cached=True),
    192: tk.Path('alignment.cache.192', creator=_alignment_job, cached=True),
    193: tk.Path('alignment.cache.193', creator=_alignment_job, cached=True),
    194: tk.Path('alignment.cache.194', creator=_alignment_job, cached=True),
    195: tk.Path('alignment.cache.195', creator=_alignment_job, cached=True),
    196: tk.Path('alignment.cache.196', creator=_alignment_job, cached=True),
    197: tk.Path('alignment.cache.197', creator=_alignment_job, cached=True),
    198: tk.Path('alignment.cache.198', creator=_alignment_job, cached=True),
    199: tk.Path('alignment.cache.199', creator=_alignment_job, cached=True),
    200: tk.Path('alignment.cache.200', creator=_alignment_job, cached=True),
    201: tk.Path('alignment.cache.201', creator=_alignment_job, cached=True),
    202: tk.Path('alignment.cache.202', creator=_alignment_job, cached=True),
    203: tk.Path('alignment.cache.203', creator=_alignment_job, cached=True),
    204: tk.Path('alignment.cache.204', creator=_alignment_job, cached=True),
    205: tk.Path('alignment.cache.205', creator=_alignment_job, cached=True),
    206: tk.Path('alignment.cache.206', creator=_alignment_job, cached=True),
    207: tk.Path('alignment.cache.207', creator=_alignment_job, cached=True),
    208: tk.Path('alignment.cache.208', creator=_alignment_job, cached=True),
    209: tk.Path('alignment.cache.209', creator=_alignment_job, cached=True),
    210: tk.Path('alignment.cache.210', creator=_alignment_job, cached=True),
    211: tk.Path('alignment.cache.211', creator=_alignment_job, cached=True),
    212: tk.Path('alignment.cache.212', creator=_alignment_job, cached=True),
    213: tk.Path('alignment.cache.213', creator=_alignment_job, cached=True),
    214: tk.Path('alignment.cache.214', creator=_alignment_job, cached=True),
    215: tk.Path('alignment.cache.215', creator=_alignment_job, cached=True),
    216: tk.Path('alignment.cache.216', creator=_alignment_job, cached=True),
    217: tk.Path('alignment.cache.217', creator=_alignment_job, cached=True),
    218: tk.Path('alignment.cache.218', creator=_alignment_job, cached=True),
    219: tk.Path('alignment.cache.219', creator=_alignment_job, cached=True),
    220: tk.Path('alignment.cache.220', creator=_alignment_job, cached=True),
    221: tk.Path('alignment.cache.221', creator=_alignment_job, cached=True),
    222: tk.Path('alignment.cache.222', creator=_alignment_job, cached=True),
    223: tk.Path('alignment.cache.223', creator=_alignment_job, cached=True),
    224: tk.Path('alignment.cache.224', creator=_alignment_job, cached=True),
    225: tk.Path('alignment.cache.225', creator=_alignment_job, cached=True),
    226: tk.Path('alignment.cache.226', creator=_alignment_job, cached=True),
    227: tk.Path('alignment.cache.227', creator=_alignment_job, cached=True),
    228: tk.Path('alignment.cache.228', creator=_alignment_job, cached=True),
    229: tk.Path('alignment.cache.229', creator=_alignment_job, cached=True),
    230: tk.Path('alignment.cache.230', creator=_alignment_job, cached=True),
    231: tk.Path('alignment.cache.231', creator=_alignment_job, cached=True),
    232: tk.Path('alignment.cache.232', creator=_alignment_job, cached=True),
    233: tk.Path('alignment.cache.233', creator=_alignment_job, cached=True),
    234: tk.Path('alignment.cache.234', creator=_alignment_job, cached=True),
    235: tk.Path('alignment.cache.235', creator=_alignment_job, cached=True),
    236: tk.Path('alignment.cache.236', creator=_alignment_job, cached=True),
    237: tk.Path('alignment.cache.237', creator=_alignment_job, cached=True),
    238: tk.Path('alignment.cache.238', creator=_alignment_job, cached=True),
    239: tk.Path('alignment.cache.239', creator=_alignment_job, cached=True),
    240: tk.Path('alignment.cache.240', creator=_alignment_job, cached=True),
    241: tk.Path('alignment.cache.241', creator=_alignment_job, cached=True),
    242: tk.Path('alignment.cache.242', creator=_alignment_job, cached=True),
    243: tk.Path('alignment.cache.243', creator=_alignment_job, cached=True),
    244: tk.Path('alignment.cache.244', creator=_alignment_job, cached=True),
    245: tk.Path('alignment.cache.245', creator=_alignment_job, cached=True),
    246: tk.Path('alignment.cache.246', creator=_alignment_job, cached=True),
    247: tk.Path('alignment.cache.247', creator=_alignment_job, cached=True),
    248: tk.Path('alignment.cache.248', creator=_alignment_job, cached=True),
    249: tk.Path('alignment.cache.249', creator=_alignment_job, cached=True),
    250: tk.Path('alignment.cache.250', creator=_alignment_job, cached=True),
    251: tk.Path('alignment.cache.251', creator=_alignment_job, cached=True),
    252: tk.Path('alignment.cache.252', creator=_alignment_job, cached=True),
    253: tk.Path('alignment.cache.253', creator=_alignment_job, cached=True),
    254: tk.Path('alignment.cache.254', creator=_alignment_job, cached=True),
    255: tk.Path('alignment.cache.255', creator=_alignment_job, cached=True),
    256: tk.Path('alignment.cache.256', creator=_alignment_job, cached=True),
    257: tk.Path('alignment.cache.257', creator=_alignment_job, cached=True),
    258: tk.Path('alignment.cache.258', creator=_alignment_job, cached=True),
    259: tk.Path('alignment.cache.259', creator=_alignment_job, cached=True),
    260: tk.Path('alignment.cache.260', creator=_alignment_job, cached=True),
    261: tk.Path('alignment.cache.261', creator=_alignment_job, cached=True),
    262: tk.Path('alignment.cache.262', creator=_alignment_job, cached=True),
    263: tk.Path('alignment.cache.263', creator=_alignment_job, cached=True),
    264: tk.Path('alignment.cache.264', creator=_alignment_job, cached=True),
    265: tk.Path('alignment.cache.265', creator=_alignment_job, cached=True),
    266: tk.Path('alignment.cache.266', creator=_alignment_job, cached=True),
    267: tk.Path('alignment.cache.267', creator=_alignment_job, cached=True),
    268: tk.Path('alignment.cache.268', creator=_alignment_job, cached=True),
    269: tk.Path('alignment.cache.269', creator=_alignment_job, cached=True),
    270: tk.Path('alignment.cache.270', creator=_alignment_job, cached=True),
    271: tk.Path('alignment.cache.271', creator=_alignment_job, cached=True),
    272: tk.Path('alignment.cache.272', creator=_alignment_job, cached=True),
    273: tk.Path('alignment.cache.273', creator=_alignment_job, cached=True),
    274: tk.Path('alignment.cache.274', creator=_alignment_job, cached=True),
    275: tk.Path('alignment.cache.275', creator=_alignment_job, cached=True),
    276: tk.Path('alignment.cache.276', creator=_alignment_job, cached=True),
    277: tk.Path('alignment.cache.277', creator=_alignment_job, cached=True),
    278: tk.Path('alignment.cache.278', creator=_alignment_job, cached=True),
    279: tk.Path('alignment.cache.279', creator=_alignment_job, cached=True),
    280: tk.Path('alignment.cache.280', creator=_alignment_job, cached=True),
    281: tk.Path('alignment.cache.281', creator=_alignment_job, cached=True),
    282: tk.Path('alignment.cache.282', creator=_alignment_job, cached=True),
    283: tk.Path('alignment.cache.283', creator=_alignment_job, cached=True),
    284: tk.Path('alignment.cache.284', creator=_alignment_job, cached=True),
    285: tk.Path('alignment.cache.285', creator=_alignment_job, cached=True),
    286: tk.Path('alignment.cache.286', creator=_alignment_job, cached=True),
    287: tk.Path('alignment.cache.287', creator=_alignment_job, cached=True),
    288: tk.Path('alignment.cache.288', creator=_alignment_job, cached=True),
    289: tk.Path('alignment.cache.289', creator=_alignment_job, cached=True),
    290: tk.Path('alignment.cache.290', creator=_alignment_job, cached=True),
    291: tk.Path('alignment.cache.291', creator=_alignment_job, cached=True),
    292: tk.Path('alignment.cache.292', creator=_alignment_job, cached=True),
    293: tk.Path('alignment.cache.293', creator=_alignment_job, cached=True),
    294: tk.Path('alignment.cache.294', creator=_alignment_job, cached=True),
    295: tk.Path('alignment.cache.295', creator=_alignment_job, cached=True),
    296: tk.Path('alignment.cache.296', creator=_alignment_job, cached=True),
    297: tk.Path('alignment.cache.297', creator=_alignment_job, cached=True),
    298: tk.Path('alignment.cache.298', creator=_alignment_job, cached=True),
    299: tk.Path('alignment.cache.299', creator=_alignment_job, cached=True),
    300: tk.Path('alignment.cache.300', creator=_alignment_job, cached=True),
}
import i6_core.util
_multi_output_path = i6_core.util.MultiOutputPath(
    _alignment_job,
    'alignment.cache.$(TASK)',
    _dict13,
    cached=True
)
_returnn_rasr_data_input.alignments.alternatives = {
    'task_dependent': _multi_output_path,
    'bundle': tk.Path('alignment.cache.bundle', creator=_alignment_job, cached=True),
}
_returnn_rasr_data_input.alignments.hidden_paths = _dict13
_returnn_rasr_data_input.feature_flow = object.__new__(i6_core.rasr.flow.FlowNetwork)
_returnn_rasr_data_input.feature_flow.name = 'network'
_named_flow_attribute = object.__new__(i6_core.rasr.flow.NamedFlowAttribute)
_named_flow_attribute.name = 'cache'
_named_flow_attribute.value = object.__new__(i6_core.rasr.flow.FlagDependentFlowAttribute)
_named_flow_attribute.value.flag = 'cache_mode'
_feature_extraction_job = make_fake_job(module='i6_core.features.extraction', name='FeatureExtractionJob', sis_hash='Gammatone.v3q8Zc1deFVj')
_dict15 = {
    1: tk.Path('gt.cache.1', creator=_feature_extraction_job, cached=True),
    2: tk.Path('gt.cache.2', creator=_feature_extraction_job, cached=True),
    3: tk.Path('gt.cache.3', creator=_feature_extraction_job, cached=True),
    4: tk.Path('gt.cache.4', creator=_feature_extraction_job, cached=True),
    5: tk.Path('gt.cache.5', creator=_feature_extraction_job, cached=True),
    6: tk.Path('gt.cache.6', creator=_feature_extraction_job, cached=True),
    7: tk.Path('gt.cache.7', creator=_feature_extraction_job, cached=True),
    8: tk.Path('gt.cache.8', creator=_feature_extraction_job, cached=True),
    9: tk.Path('gt.cache.9', creator=_feature_extraction_job, cached=True),
    10: tk.Path('gt.cache.10', creator=_feature_extraction_job, cached=True),
    11: tk.Path('gt.cache.11', creator=_feature_extraction_job, cached=True),
    12: tk.Path('gt.cache.12', creator=_feature_extraction_job, cached=True),
    13: tk.Path('gt.cache.13', creator=_feature_extraction_job, cached=True),
    14: tk.Path('gt.cache.14', creator=_feature_extraction_job, cached=True),
    15: tk.Path('gt.cache.15', creator=_feature_extraction_job, cached=True),
    16: tk.Path('gt.cache.16', creator=_feature_extraction_job, cached=True),
    17: tk.Path('gt.cache.17', creator=_feature_extraction_job, cached=True),
    18: tk.Path('gt.cache.18', creator=_feature_extraction_job, cached=True),
    19: tk.Path('gt.cache.19', creator=_feature_extraction_job, cached=True),
    20: tk.Path('gt.cache.20', creator=_feature_extraction_job, cached=True),
    21: tk.Path('gt.cache.21', creator=_feature_extraction_job, cached=True),
    22: tk.Path('gt.cache.22', creator=_feature_extraction_job, cached=True),
    23: tk.Path('gt.cache.23', creator=_feature_extraction_job, cached=True),
    24: tk.Path('gt.cache.24', creator=_feature_extraction_job, cached=True),
    25: tk.Path('gt.cache.25', creator=_feature_extraction_job, cached=True),
    26: tk.Path('gt.cache.26', creator=_feature_extraction_job, cached=True),
    27: tk.Path('gt.cache.27', creator=_feature_extraction_job, cached=True),
    28: tk.Path('gt.cache.28', creator=_feature_extraction_job, cached=True),
    29: tk.Path('gt.cache.29', creator=_feature_extraction_job, cached=True),
    30: tk.Path('gt.cache.30', creator=_feature_extraction_job, cached=True),
    31: tk.Path('gt.cache.31', creator=_feature_extraction_job, cached=True),
    32: tk.Path('gt.cache.32', creator=_feature_extraction_job, cached=True),
    33: tk.Path('gt.cache.33', creator=_feature_extraction_job, cached=True),
    34: tk.Path('gt.cache.34', creator=_feature_extraction_job, cached=True),
    35: tk.Path('gt.cache.35', creator=_feature_extraction_job, cached=True),
    36: tk.Path('gt.cache.36', creator=_feature_extraction_job, cached=True),
    37: tk.Path('gt.cache.37', creator=_feature_extraction_job, cached=True),
    38: tk.Path('gt.cache.38', creator=_feature_extraction_job, cached=True),
    39: tk.Path('gt.cache.39', creator=_feature_extraction_job, cached=True),
    40: tk.Path('gt.cache.40', creator=_feature_extraction_job, cached=True),
    41: tk.Path('gt.cache.41', creator=_feature_extraction_job, cached=True),
    42: tk.Path('gt.cache.42', creator=_feature_extraction_job, cached=True),
    43: tk.Path('gt.cache.43', creator=_feature_extraction_job, cached=True),
    44: tk.Path('gt.cache.44', creator=_feature_extraction_job, cached=True),
    45: tk.Path('gt.cache.45', creator=_feature_extraction_job, cached=True),
    46: tk.Path('gt.cache.46', creator=_feature_extraction_job, cached=True),
    47: tk.Path('gt.cache.47', creator=_feature_extraction_job, cached=True),
    48: tk.Path('gt.cache.48', creator=_feature_extraction_job, cached=True),
    49: tk.Path('gt.cache.49', creator=_feature_extraction_job, cached=True),
    50: tk.Path('gt.cache.50', creator=_feature_extraction_job, cached=True),
    51: tk.Path('gt.cache.51', creator=_feature_extraction_job, cached=True),
    52: tk.Path('gt.cache.52', creator=_feature_extraction_job, cached=True),
    53: tk.Path('gt.cache.53', creator=_feature_extraction_job, cached=True),
    54: tk.Path('gt.cache.54', creator=_feature_extraction_job, cached=True),
    55: tk.Path('gt.cache.55', creator=_feature_extraction_job, cached=True),
    56: tk.Path('gt.cache.56', creator=_feature_extraction_job, cached=True),
    57: tk.Path('gt.cache.57', creator=_feature_extraction_job, cached=True),
    58: tk.Path('gt.cache.58', creator=_feature_extraction_job, cached=True),
    59: tk.Path('gt.cache.59', creator=_feature_extraction_job, cached=True),
    60: tk.Path('gt.cache.60', creator=_feature_extraction_job, cached=True),
    61: tk.Path('gt.cache.61', creator=_feature_extraction_job, cached=True),
    62: tk.Path('gt.cache.62', creator=_feature_extraction_job, cached=True),
    63: tk.Path('gt.cache.63', creator=_feature_extraction_job, cached=True),
    64: tk.Path('gt.cache.64', creator=_feature_extraction_job, cached=True),
    65: tk.Path('gt.cache.65', creator=_feature_extraction_job, cached=True),
    66: tk.Path('gt.cache.66', creator=_feature_extraction_job, cached=True),
    67: tk.Path('gt.cache.67', creator=_feature_extraction_job, cached=True),
    68: tk.Path('gt.cache.68', creator=_feature_extraction_job, cached=True),
    69: tk.Path('gt.cache.69', creator=_feature_extraction_job, cached=True),
    70: tk.Path('gt.cache.70', creator=_feature_extraction_job, cached=True),
    71: tk.Path('gt.cache.71', creator=_feature_extraction_job, cached=True),
    72: tk.Path('gt.cache.72', creator=_feature_extraction_job, cached=True),
    73: tk.Path('gt.cache.73', creator=_feature_extraction_job, cached=True),
    74: tk.Path('gt.cache.74', creator=_feature_extraction_job, cached=True),
    75: tk.Path('gt.cache.75', creator=_feature_extraction_job, cached=True),
    76: tk.Path('gt.cache.76', creator=_feature_extraction_job, cached=True),
    77: tk.Path('gt.cache.77', creator=_feature_extraction_job, cached=True),
    78: tk.Path('gt.cache.78', creator=_feature_extraction_job, cached=True),
    79: tk.Path('gt.cache.79', creator=_feature_extraction_job, cached=True),
    80: tk.Path('gt.cache.80', creator=_feature_extraction_job, cached=True),
    81: tk.Path('gt.cache.81', creator=_feature_extraction_job, cached=True),
    82: tk.Path('gt.cache.82', creator=_feature_extraction_job, cached=True),
    83: tk.Path('gt.cache.83', creator=_feature_extraction_job, cached=True),
    84: tk.Path('gt.cache.84', creator=_feature_extraction_job, cached=True),
    85: tk.Path('gt.cache.85', creator=_feature_extraction_job, cached=True),
    86: tk.Path('gt.cache.86', creator=_feature_extraction_job, cached=True),
    87: tk.Path('gt.cache.87', creator=_feature_extraction_job, cached=True),
    88: tk.Path('gt.cache.88', creator=_feature_extraction_job, cached=True),
    89: tk.Path('gt.cache.89', creator=_feature_extraction_job, cached=True),
    90: tk.Path('gt.cache.90', creator=_feature_extraction_job, cached=True),
    91: tk.Path('gt.cache.91', creator=_feature_extraction_job, cached=True),
    92: tk.Path('gt.cache.92', creator=_feature_extraction_job, cached=True),
    93: tk.Path('gt.cache.93', creator=_feature_extraction_job, cached=True),
    94: tk.Path('gt.cache.94', creator=_feature_extraction_job, cached=True),
    95: tk.Path('gt.cache.95', creator=_feature_extraction_job, cached=True),
    96: tk.Path('gt.cache.96', creator=_feature_extraction_job, cached=True),
    97: tk.Path('gt.cache.97', creator=_feature_extraction_job, cached=True),
    98: tk.Path('gt.cache.98', creator=_feature_extraction_job, cached=True),
    99: tk.Path('gt.cache.99', creator=_feature_extraction_job, cached=True),
    100: tk.Path('gt.cache.100', creator=_feature_extraction_job, cached=True),
    101: tk.Path('gt.cache.101', creator=_feature_extraction_job, cached=True),
    102: tk.Path('gt.cache.102', creator=_feature_extraction_job, cached=True),
    103: tk.Path('gt.cache.103', creator=_feature_extraction_job, cached=True),
    104: tk.Path('gt.cache.104', creator=_feature_extraction_job, cached=True),
    105: tk.Path('gt.cache.105', creator=_feature_extraction_job, cached=True),
    106: tk.Path('gt.cache.106', creator=_feature_extraction_job, cached=True),
    107: tk.Path('gt.cache.107', creator=_feature_extraction_job, cached=True),
    108: tk.Path('gt.cache.108', creator=_feature_extraction_job, cached=True),
    109: tk.Path('gt.cache.109', creator=_feature_extraction_job, cached=True),
    110: tk.Path('gt.cache.110', creator=_feature_extraction_job, cached=True),
    111: tk.Path('gt.cache.111', creator=_feature_extraction_job, cached=True),
    112: tk.Path('gt.cache.112', creator=_feature_extraction_job, cached=True),
    113: tk.Path('gt.cache.113', creator=_feature_extraction_job, cached=True),
    114: tk.Path('gt.cache.114', creator=_feature_extraction_job, cached=True),
    115: tk.Path('gt.cache.115', creator=_feature_extraction_job, cached=True),
    116: tk.Path('gt.cache.116', creator=_feature_extraction_job, cached=True),
    117: tk.Path('gt.cache.117', creator=_feature_extraction_job, cached=True),
    118: tk.Path('gt.cache.118', creator=_feature_extraction_job, cached=True),
    119: tk.Path('gt.cache.119', creator=_feature_extraction_job, cached=True),
    120: tk.Path('gt.cache.120', creator=_feature_extraction_job, cached=True),
    121: tk.Path('gt.cache.121', creator=_feature_extraction_job, cached=True),
    122: tk.Path('gt.cache.122', creator=_feature_extraction_job, cached=True),
    123: tk.Path('gt.cache.123', creator=_feature_extraction_job, cached=True),
    124: tk.Path('gt.cache.124', creator=_feature_extraction_job, cached=True),
    125: tk.Path('gt.cache.125', creator=_feature_extraction_job, cached=True),
    126: tk.Path('gt.cache.126', creator=_feature_extraction_job, cached=True),
    127: tk.Path('gt.cache.127', creator=_feature_extraction_job, cached=True),
    128: tk.Path('gt.cache.128', creator=_feature_extraction_job, cached=True),
    129: tk.Path('gt.cache.129', creator=_feature_extraction_job, cached=True),
    130: tk.Path('gt.cache.130', creator=_feature_extraction_job, cached=True),
    131: tk.Path('gt.cache.131', creator=_feature_extraction_job, cached=True),
    132: tk.Path('gt.cache.132', creator=_feature_extraction_job, cached=True),
    133: tk.Path('gt.cache.133', creator=_feature_extraction_job, cached=True),
    134: tk.Path('gt.cache.134', creator=_feature_extraction_job, cached=True),
    135: tk.Path('gt.cache.135', creator=_feature_extraction_job, cached=True),
    136: tk.Path('gt.cache.136', creator=_feature_extraction_job, cached=True),
    137: tk.Path('gt.cache.137', creator=_feature_extraction_job, cached=True),
    138: tk.Path('gt.cache.138', creator=_feature_extraction_job, cached=True),
    139: tk.Path('gt.cache.139', creator=_feature_extraction_job, cached=True),
    140: tk.Path('gt.cache.140', creator=_feature_extraction_job, cached=True),
    141: tk.Path('gt.cache.141', creator=_feature_extraction_job, cached=True),
    142: tk.Path('gt.cache.142', creator=_feature_extraction_job, cached=True),
    143: tk.Path('gt.cache.143', creator=_feature_extraction_job, cached=True),
    144: tk.Path('gt.cache.144', creator=_feature_extraction_job, cached=True),
    145: tk.Path('gt.cache.145', creator=_feature_extraction_job, cached=True),
    146: tk.Path('gt.cache.146', creator=_feature_extraction_job, cached=True),
    147: tk.Path('gt.cache.147', creator=_feature_extraction_job, cached=True),
    148: tk.Path('gt.cache.148', creator=_feature_extraction_job, cached=True),
    149: tk.Path('gt.cache.149', creator=_feature_extraction_job, cached=True),
    150: tk.Path('gt.cache.150', creator=_feature_extraction_job, cached=True),
    151: tk.Path('gt.cache.151', creator=_feature_extraction_job, cached=True),
    152: tk.Path('gt.cache.152', creator=_feature_extraction_job, cached=True),
    153: tk.Path('gt.cache.153', creator=_feature_extraction_job, cached=True),
    154: tk.Path('gt.cache.154', creator=_feature_extraction_job, cached=True),
    155: tk.Path('gt.cache.155', creator=_feature_extraction_job, cached=True),
    156: tk.Path('gt.cache.156', creator=_feature_extraction_job, cached=True),
    157: tk.Path('gt.cache.157', creator=_feature_extraction_job, cached=True),
    158: tk.Path('gt.cache.158', creator=_feature_extraction_job, cached=True),
    159: tk.Path('gt.cache.159', creator=_feature_extraction_job, cached=True),
    160: tk.Path('gt.cache.160', creator=_feature_extraction_job, cached=True),
    161: tk.Path('gt.cache.161', creator=_feature_extraction_job, cached=True),
    162: tk.Path('gt.cache.162', creator=_feature_extraction_job, cached=True),
    163: tk.Path('gt.cache.163', creator=_feature_extraction_job, cached=True),
    164: tk.Path('gt.cache.164', creator=_feature_extraction_job, cached=True),
    165: tk.Path('gt.cache.165', creator=_feature_extraction_job, cached=True),
    166: tk.Path('gt.cache.166', creator=_feature_extraction_job, cached=True),
    167: tk.Path('gt.cache.167', creator=_feature_extraction_job, cached=True),
    168: tk.Path('gt.cache.168', creator=_feature_extraction_job, cached=True),
    169: tk.Path('gt.cache.169', creator=_feature_extraction_job, cached=True),
    170: tk.Path('gt.cache.170', creator=_feature_extraction_job, cached=True),
    171: tk.Path('gt.cache.171', creator=_feature_extraction_job, cached=True),
    172: tk.Path('gt.cache.172', creator=_feature_extraction_job, cached=True),
    173: tk.Path('gt.cache.173', creator=_feature_extraction_job, cached=True),
    174: tk.Path('gt.cache.174', creator=_feature_extraction_job, cached=True),
    175: tk.Path('gt.cache.175', creator=_feature_extraction_job, cached=True),
    176: tk.Path('gt.cache.176', creator=_feature_extraction_job, cached=True),
    177: tk.Path('gt.cache.177', creator=_feature_extraction_job, cached=True),
    178: tk.Path('gt.cache.178', creator=_feature_extraction_job, cached=True),
    179: tk.Path('gt.cache.179', creator=_feature_extraction_job, cached=True),
    180: tk.Path('gt.cache.180', creator=_feature_extraction_job, cached=True),
    181: tk.Path('gt.cache.181', creator=_feature_extraction_job, cached=True),
    182: tk.Path('gt.cache.182', creator=_feature_extraction_job, cached=True),
    183: tk.Path('gt.cache.183', creator=_feature_extraction_job, cached=True),
    184: tk.Path('gt.cache.184', creator=_feature_extraction_job, cached=True),
    185: tk.Path('gt.cache.185', creator=_feature_extraction_job, cached=True),
    186: tk.Path('gt.cache.186', creator=_feature_extraction_job, cached=True),
    187: tk.Path('gt.cache.187', creator=_feature_extraction_job, cached=True),
    188: tk.Path('gt.cache.188', creator=_feature_extraction_job, cached=True),
    189: tk.Path('gt.cache.189', creator=_feature_extraction_job, cached=True),
    190: tk.Path('gt.cache.190', creator=_feature_extraction_job, cached=True),
    191: tk.Path('gt.cache.191', creator=_feature_extraction_job, cached=True),
    192: tk.Path('gt.cache.192', creator=_feature_extraction_job, cached=True),
    193: tk.Path('gt.cache.193', creator=_feature_extraction_job, cached=True),
    194: tk.Path('gt.cache.194', creator=_feature_extraction_job, cached=True),
    195: tk.Path('gt.cache.195', creator=_feature_extraction_job, cached=True),
    196: tk.Path('gt.cache.196', creator=_feature_extraction_job, cached=True),
    197: tk.Path('gt.cache.197', creator=_feature_extraction_job, cached=True),
    198: tk.Path('gt.cache.198', creator=_feature_extraction_job, cached=True),
    199: tk.Path('gt.cache.199', creator=_feature_extraction_job, cached=True),
    200: tk.Path('gt.cache.200', creator=_feature_extraction_job, cached=True),
    201: tk.Path('gt.cache.201', creator=_feature_extraction_job, cached=True),
    202: tk.Path('gt.cache.202', creator=_feature_extraction_job, cached=True),
    203: tk.Path('gt.cache.203', creator=_feature_extraction_job, cached=True),
    204: tk.Path('gt.cache.204', creator=_feature_extraction_job, cached=True),
    205: tk.Path('gt.cache.205', creator=_feature_extraction_job, cached=True),
    206: tk.Path('gt.cache.206', creator=_feature_extraction_job, cached=True),
    207: tk.Path('gt.cache.207', creator=_feature_extraction_job, cached=True),
    208: tk.Path('gt.cache.208', creator=_feature_extraction_job, cached=True),
    209: tk.Path('gt.cache.209', creator=_feature_extraction_job, cached=True),
    210: tk.Path('gt.cache.210', creator=_feature_extraction_job, cached=True),
    211: tk.Path('gt.cache.211', creator=_feature_extraction_job, cached=True),
    212: tk.Path('gt.cache.212', creator=_feature_extraction_job, cached=True),
    213: tk.Path('gt.cache.213', creator=_feature_extraction_job, cached=True),
    214: tk.Path('gt.cache.214', creator=_feature_extraction_job, cached=True),
    215: tk.Path('gt.cache.215', creator=_feature_extraction_job, cached=True),
    216: tk.Path('gt.cache.216', creator=_feature_extraction_job, cached=True),
    217: tk.Path('gt.cache.217', creator=_feature_extraction_job, cached=True),
    218: tk.Path('gt.cache.218', creator=_feature_extraction_job, cached=True),
    219: tk.Path('gt.cache.219', creator=_feature_extraction_job, cached=True),
    220: tk.Path('gt.cache.220', creator=_feature_extraction_job, cached=True),
    221: tk.Path('gt.cache.221', creator=_feature_extraction_job, cached=True),
    222: tk.Path('gt.cache.222', creator=_feature_extraction_job, cached=True),
    223: tk.Path('gt.cache.223', creator=_feature_extraction_job, cached=True),
    224: tk.Path('gt.cache.224', creator=_feature_extraction_job, cached=True),
    225: tk.Path('gt.cache.225', creator=_feature_extraction_job, cached=True),
    226: tk.Path('gt.cache.226', creator=_feature_extraction_job, cached=True),
    227: tk.Path('gt.cache.227', creator=_feature_extraction_job, cached=True),
    228: tk.Path('gt.cache.228', creator=_feature_extraction_job, cached=True),
    229: tk.Path('gt.cache.229', creator=_feature_extraction_job, cached=True),
    230: tk.Path('gt.cache.230', creator=_feature_extraction_job, cached=True),
    231: tk.Path('gt.cache.231', creator=_feature_extraction_job, cached=True),
    232: tk.Path('gt.cache.232', creator=_feature_extraction_job, cached=True),
    233: tk.Path('gt.cache.233', creator=_feature_extraction_job, cached=True),
    234: tk.Path('gt.cache.234', creator=_feature_extraction_job, cached=True),
    235: tk.Path('gt.cache.235', creator=_feature_extraction_job, cached=True),
    236: tk.Path('gt.cache.236', creator=_feature_extraction_job, cached=True),
    237: tk.Path('gt.cache.237', creator=_feature_extraction_job, cached=True),
    238: tk.Path('gt.cache.238', creator=_feature_extraction_job, cached=True),
    239: tk.Path('gt.cache.239', creator=_feature_extraction_job, cached=True),
    240: tk.Path('gt.cache.240', creator=_feature_extraction_job, cached=True),
    241: tk.Path('gt.cache.241', creator=_feature_extraction_job, cached=True),
    242: tk.Path('gt.cache.242', creator=_feature_extraction_job, cached=True),
    243: tk.Path('gt.cache.243', creator=_feature_extraction_job, cached=True),
    244: tk.Path('gt.cache.244', creator=_feature_extraction_job, cached=True),
    245: tk.Path('gt.cache.245', creator=_feature_extraction_job, cached=True),
    246: tk.Path('gt.cache.246', creator=_feature_extraction_job, cached=True),
    247: tk.Path('gt.cache.247', creator=_feature_extraction_job, cached=True),
    248: tk.Path('gt.cache.248', creator=_feature_extraction_job, cached=True),
    249: tk.Path('gt.cache.249', creator=_feature_extraction_job, cached=True),
    250: tk.Path('gt.cache.250', creator=_feature_extraction_job, cached=True),
    251: tk.Path('gt.cache.251', creator=_feature_extraction_job, cached=True),
    252: tk.Path('gt.cache.252', creator=_feature_extraction_job, cached=True),
    253: tk.Path('gt.cache.253', creator=_feature_extraction_job, cached=True),
    254: tk.Path('gt.cache.254', creator=_feature_extraction_job, cached=True),
    255: tk.Path('gt.cache.255', creator=_feature_extraction_job, cached=True),
    256: tk.Path('gt.cache.256', creator=_feature_extraction_job, cached=True),
    257: tk.Path('gt.cache.257', creator=_feature_extraction_job, cached=True),
    258: tk.Path('gt.cache.258', creator=_feature_extraction_job, cached=True),
    259: tk.Path('gt.cache.259', creator=_feature_extraction_job, cached=True),
    260: tk.Path('gt.cache.260', creator=_feature_extraction_job, cached=True),
    261: tk.Path('gt.cache.261', creator=_feature_extraction_job, cached=True),
    262: tk.Path('gt.cache.262', creator=_feature_extraction_job, cached=True),
    263: tk.Path('gt.cache.263', creator=_feature_extraction_job, cached=True),
    264: tk.Path('gt.cache.264', creator=_feature_extraction_job, cached=True),
    265: tk.Path('gt.cache.265', creator=_feature_extraction_job, cached=True),
    266: tk.Path('gt.cache.266', creator=_feature_extraction_job, cached=True),
    267: tk.Path('gt.cache.267', creator=_feature_extraction_job, cached=True),
    268: tk.Path('gt.cache.268', creator=_feature_extraction_job, cached=True),
    269: tk.Path('gt.cache.269', creator=_feature_extraction_job, cached=True),
    270: tk.Path('gt.cache.270', creator=_feature_extraction_job, cached=True),
    271: tk.Path('gt.cache.271', creator=_feature_extraction_job, cached=True),
    272: tk.Path('gt.cache.272', creator=_feature_extraction_job, cached=True),
    273: tk.Path('gt.cache.273', creator=_feature_extraction_job, cached=True),
    274: tk.Path('gt.cache.274', creator=_feature_extraction_job, cached=True),
    275: tk.Path('gt.cache.275', creator=_feature_extraction_job, cached=True),
    276: tk.Path('gt.cache.276', creator=_feature_extraction_job, cached=True),
    277: tk.Path('gt.cache.277', creator=_feature_extraction_job, cached=True),
    278: tk.Path('gt.cache.278', creator=_feature_extraction_job, cached=True),
    279: tk.Path('gt.cache.279', creator=_feature_extraction_job, cached=True),
    280: tk.Path('gt.cache.280', creator=_feature_extraction_job, cached=True),
    281: tk.Path('gt.cache.281', creator=_feature_extraction_job, cached=True),
    282: tk.Path('gt.cache.282', creator=_feature_extraction_job, cached=True),
    283: tk.Path('gt.cache.283', creator=_feature_extraction_job, cached=True),
    284: tk.Path('gt.cache.284', creator=_feature_extraction_job, cached=True),
    285: tk.Path('gt.cache.285', creator=_feature_extraction_job, cached=True),
    286: tk.Path('gt.cache.286', creator=_feature_extraction_job, cached=True),
    287: tk.Path('gt.cache.287', creator=_feature_extraction_job, cached=True),
    288: tk.Path('gt.cache.288', creator=_feature_extraction_job, cached=True),
    289: tk.Path('gt.cache.289', creator=_feature_extraction_job, cached=True),
    290: tk.Path('gt.cache.290', creator=_feature_extraction_job, cached=True),
    291: tk.Path('gt.cache.291', creator=_feature_extraction_job, cached=True),
    292: tk.Path('gt.cache.292', creator=_feature_extraction_job, cached=True),
    293: tk.Path('gt.cache.293', creator=_feature_extraction_job, cached=True),
    294: tk.Path('gt.cache.294', creator=_feature_extraction_job, cached=True),
    295: tk.Path('gt.cache.295', creator=_feature_extraction_job, cached=True),
    296: tk.Path('gt.cache.296', creator=_feature_extraction_job, cached=True),
    297: tk.Path('gt.cache.297', creator=_feature_extraction_job, cached=True),
    298: tk.Path('gt.cache.298', creator=_feature_extraction_job, cached=True),
    299: tk.Path('gt.cache.299', creator=_feature_extraction_job, cached=True),
    300: tk.Path('gt.cache.300', creator=_feature_extraction_job, cached=True),
}
_multi_output_path1 = i6_core.util.MultiOutputPath(
    _feature_extraction_job,
    'gt.cache.$(TASK)',
    _dict15,
    cached=True
)
_named_flow_attribute.value.alternatives = {
    'task_dependent': _multi_output_path1,
    'bundle': tk.Path('gt.cache.bundle', creator=_feature_extraction_job, cached=True),
}
_dict14 = {
    'id': '$(id)',
    'path': _named_flow_attribute,
    'filter': 'generic-cache',
}
_returnn_rasr_data_input.feature_flow.nodes = {
    'cache': _dict14,
}
_returnn_rasr_data_input.feature_flow.links = [('cache', 'network:features', )]
_returnn_rasr_data_input.feature_flow.inputs = set()
_returnn_rasr_data_input.feature_flow.outputs = {'features'}
_returnn_rasr_data_input.feature_flow.params = {'id'}
_returnn_rasr_data_input.feature_flow.named_attributes = {
    'cache': _named_flow_attribute,
}
_returnn_rasr_data_input.feature_flow.hidden_inputs = set()
_returnn_rasr_data_input.feature_flow.flags = {
    'cache_mode': 'task_dependent',
}
_returnn_rasr_data_input.feature_flow.config = None
_returnn_rasr_data_input.feature_flow.post_config = None
_returnn_rasr_data_input.features = _multi_output_path1
_estimate_mixtures_job = make_fake_job(module='i6_core.mm.mixtures', name='EstimateMixturesJob', sis_hash='accumulate.7lOMBxKEZsTk')
_returnn_rasr_data_input.acoustic_mixtures = tk.Path('am.mix', creator=_estimate_mixtures_job, cached=True)
_returnn_rasr_data_input.feature_scorers = {
}
_returnn_rasr_data_input.shuffle_data = True
_dict12 = {
    'train-other-960.train': _returnn_rasr_data_input,
}
_returnn_rasr_data_input1 = object.__new__(i6_experiments.common.setups.rasr.util.nn.ReturnnRasrDataInput)
_returnn_rasr_data_input1.name = 'init'
_returnn_rasr_data_input1_crp_base = rasr.CommonRasrParameters()
_returnn_rasr_data_input1_crp_base.acoustic_model_config = rasr.RasrConfig()
_returnn_rasr_data_input1_crp_base.acoustic_model_config.state_tying.type = 'cart'
_returnn_rasr_data_input1_crp_base.acoustic_model_config.state_tying.file = tk.Path('cart.tree.xml.gz', creator=_estimate_cart_job)
_returnn_rasr_data_input1_crp_base.acoustic_model_config.allophones.add_from_lexicon = True
_returnn_rasr_data_input1_crp_base.acoustic_model_config.allophones.add_all = False
_returnn_rasr_data_input1_crp_base.acoustic_model_config.hmm.states_per_phone = 3
_returnn_rasr_data_input1_crp_base.acoustic_model_config.hmm.state_repetitions = 1
_returnn_rasr_data_input1_crp_base.acoustic_model_config.hmm.across_word_model = True
_returnn_rasr_data_input1_crp_base.acoustic_model_config.hmm.early_recombination = False
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp.scale = 1.0
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp['*'].loop = 3.0
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp['*'].forward = 0.0
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp['*'].skip = 30.0
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp['*'].exit = 0.0
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp.silence.loop = 0.0
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp.silence.forward = 3.0
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp.silence.skip = 'infinity'
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp.silence.exit = 20.0
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp.entry_m1.loop = 'infinity'
_returnn_rasr_data_input1_crp_base.acoustic_model_config.tdp.entry_m2.loop = 'infinity'
_returnn_rasr_data_input1_crp_base.acoustic_model_post_config = None
_returnn_rasr_data_input1_crp_base.corpus_config = None
_returnn_rasr_data_input1_crp_base.corpus_post_config = None
_returnn_rasr_data_input1_crp_base.lexicon_config = None
_returnn_rasr_data_input1_crp_base.lexicon_post_config = None
_returnn_rasr_data_input1_crp_base.language_model_config = None
_returnn_rasr_data_input1_crp_base.language_model_post_config = None
_returnn_rasr_data_input1_crp_base.recognizer_config = None
_returnn_rasr_data_input1_crp_base.recognizer_post_config = None
_returnn_rasr_data_input1_crp_base.log_config = rasr.RasrConfig()
_returnn_rasr_data_input1_crp_base.log_config['*'].configuration.channel = 'output-channel'
_returnn_rasr_data_input1_crp_base.log_config['*'].real_time_factor.channel = 'output-channel'
_returnn_rasr_data_input1_crp_base.log_config['*'].system_info.channel = 'output-channel'
_returnn_rasr_data_input1_crp_base.log_config['*'].time.channel = 'output-channel'
_returnn_rasr_data_input1_crp_base.log_config['*'].version.channel = 'output-channel'
_returnn_rasr_data_input1_crp_base.log_config['*'].log.channel = 'output-channel'
_returnn_rasr_data_input1_crp_base.log_config['*'].warning.channel = 'output-channel, stderr'
_returnn_rasr_data_input1_crp_base.log_config['*'].error.channel = 'output-channel, stderr'
_returnn_rasr_data_input1_crp_base.log_config['*'].statistics.channel = 'output-channel'
_returnn_rasr_data_input1_crp_base.log_config['*'].progress.channel = 'output-channel'
_returnn_rasr_data_input1_crp_base.log_config['*'].dot.channel = 'nil'
_returnn_rasr_data_input1_crp_base.log_post_config = rasr.RasrConfig()
_returnn_rasr_data_input1_crp_base.log_post_config['*'].encoding = 'UTF-8'
_returnn_rasr_data_input1_crp_base.log_post_config['*'].output_channel.file = '$(LOGFILE)'
_returnn_rasr_data_input1_crp_base.log_post_config['*'].output_channel.compressed = False
_returnn_rasr_data_input1_crp_base.log_post_config['*'].output_channel.append = False
_returnn_rasr_data_input1_crp_base.log_post_config['*'].output_channel.unbuffered = False
_returnn_rasr_data_input1_crp_base.compress_log_file = True
_returnn_rasr_data_input1_crp_base.default_log_channel = 'output-channel'
_returnn_rasr_data_input1_crp_base.audio_format = 'wav'
_returnn_rasr_data_input1_crp_base.corpus_duration = 1.0
_returnn_rasr_data_input1_crp_base.concurrent = 1
_returnn_rasr_data_input1_crp_base.segment_path = None
_returnn_rasr_data_input1_crp_base.acoustic_model_trainer_exe = None
_returnn_rasr_data_input1_crp_base.allophone_tool_exe = None
_returnn_rasr_data_input1_crp_base.costa_exe = None
_returnn_rasr_data_input1_crp_base.feature_extraction_exe = None
_returnn_rasr_data_input1_crp_base.feature_statistics_exe = None
_returnn_rasr_data_input1_crp_base.flf_tool_exe = None
_returnn_rasr_data_input1_crp_base.kws_tool_exe = None
_returnn_rasr_data_input1_crp_base.lattice_processor_exe = None
_returnn_rasr_data_input1_crp_base.lm_util_exe = None
_returnn_rasr_data_input1_crp_base.nn_trainer_exe = None
_returnn_rasr_data_input1_crp_base.speech_recognizer_exe = None
_returnn_rasr_data_input1_crp_base.python_home = None
_returnn_rasr_data_input1_crp_base.python_program_name = None
_returnn_rasr_data_input1.crp = rasr.CommonRasrParameters(_returnn_rasr_data_input1_crp_base)
_returnn_rasr_data_input1.crp.corpus_config = rasr.RasrConfig()
_returnn_rasr_data_input1.crp.corpus_config.file = tk.Path('merged.xml.gz', creator=_merge_corpora_job)
_returnn_rasr_data_input1.crp.corpus_config.audio_dir = None
_returnn_rasr_data_input1.crp.corpus_config.warn_about_unexpected_elements = True
_returnn_rasr_data_input1.crp.corpus_config.capitalize_transcriptions = False
_returnn_rasr_data_input1.crp.corpus_config.progress_indication = 'global'
_returnn_rasr_data_input1.crp.corpus_config.segment_order_shuffle = True
_returnn_rasr_data_input1.crp.corpus_config.segment_order_sort_by_time_length = True
_returnn_rasr_data_input1.crp.corpus_config.segment_order_sort_by_time_length_chunk_size = 384
_returnn_rasr_data_input1.crp.audio_format = 'wav'
_returnn_rasr_data_input1.crp.corpus_duration = 960.9000000000001
_returnn_rasr_data_input1.crp.concurrent = 1
_returnn_rasr_data_input1.crp.segment_path = tk.Path('cv.segments', creator=_shuffle_and_split_segments_job)
_returnn_rasr_data_input1.crp.acoustic_model_trainer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/acoustic-model-trainer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/acoustic-model-trainer.linux-x86_64-standard', ))
_returnn_rasr_data_input1.crp.allophone_tool_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/allophone-tool.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/allophone-tool.linux-x86_64-standard', ))
_returnn_rasr_data_input1.crp.costa_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/costa.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/costa.linux-x86_64-standard', ))
_returnn_rasr_data_input1.crp.feature_extraction_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/feature-extraction.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/feature-extraction.linux-x86_64-standard', ))
_returnn_rasr_data_input1.crp.feature_statistics_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/feature-statistics.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/feature-statistics.linux-x86_64-standard', ))
_returnn_rasr_data_input1.crp.flf_tool_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/flf-tool.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/flf-tool.linux-x86_64-standard', ))
_returnn_rasr_data_input1.crp.kws_tool_exe = None
_returnn_rasr_data_input1.crp.lattice_processor_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/lattice-processor.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/lattice-processor.linux-x86_64-standard', ))
_returnn_rasr_data_input1.crp.lm_util_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/lm-util.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/lm-util.linux-x86_64-standard', ))
_returnn_rasr_data_input1.crp.nn_trainer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/nn-trainer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/nn-trainer.linux-x86_64-standard', ))
_returnn_rasr_data_input1.crp.speech_recognizer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/speech-recognizer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/speech-recognizer.linux-x86_64-standard', ))
_returnn_rasr_data_input1.crp.lexicon_config = rasr.RasrConfig()
_returnn_rasr_data_input1.crp.lexicon_config.file = tk.Path('oov.lexicon.gz', creator=_g2_p_output_to_bliss_lexicon_job, cached=True)
_returnn_rasr_data_input1.crp.lexicon_config.normalize_pronunciation = False
_returnn_rasr_data_input1.crp.acoustic_model_post_config = rasr.RasrConfig()
_returnn_rasr_data_input1.crp.acoustic_model_post_config.allophones.add_from_file = tk.Path('allophones', creator=_store_allophones_job)
_returnn_rasr_data_input1.alignments = _returnn_rasr_data_input.alignments
_returnn_rasr_data_input1.feature_flow = _returnn_rasr_data_input.feature_flow
_returnn_rasr_data_input1.features = _multi_output_path1
_returnn_rasr_data_input1.acoustic_mixtures = tk.Path('am.mix', creator=_estimate_mixtures_job, cached=True)
_returnn_rasr_data_input1.feature_scorers = _returnn_rasr_data_input.feature_scorers
_returnn_rasr_data_input1.shuffle_data = True
_dict16 = {
    'train-other-960.cv': _returnn_rasr_data_input1,
}
_returnn_rasr_data_input2 = object.__new__(i6_experiments.common.setups.rasr.util.nn.ReturnnRasrDataInput)
_returnn_rasr_data_input2.name = 'init'
_returnn_rasr_data_input2_crp_base = rasr.CommonRasrParameters()
_returnn_rasr_data_input2_crp_base.acoustic_model_config = rasr.RasrConfig()
_returnn_rasr_data_input2_crp_base.acoustic_model_config.state_tying.type = 'cart'
_returnn_rasr_data_input2_crp_base.acoustic_model_config.state_tying.file = tk.Path('cart.tree.xml.gz', creator=_estimate_cart_job)
_returnn_rasr_data_input2_crp_base.acoustic_model_config.allophones.add_from_lexicon = True
_returnn_rasr_data_input2_crp_base.acoustic_model_config.allophones.add_all = False
_returnn_rasr_data_input2_crp_base.acoustic_model_config.hmm.states_per_phone = 3
_returnn_rasr_data_input2_crp_base.acoustic_model_config.hmm.state_repetitions = 1
_returnn_rasr_data_input2_crp_base.acoustic_model_config.hmm.across_word_model = True
_returnn_rasr_data_input2_crp_base.acoustic_model_config.hmm.early_recombination = False
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp.scale = 1.0
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp['*'].loop = 3.0
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp['*'].forward = 0.0
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp['*'].skip = 30.0
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp['*'].exit = 0.0
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp.silence.loop = 0.0
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp.silence.forward = 3.0
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp.silence.skip = 'infinity'
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp.silence.exit = 20.0
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp.entry_m1.loop = 'infinity'
_returnn_rasr_data_input2_crp_base.acoustic_model_config.tdp.entry_m2.loop = 'infinity'
_returnn_rasr_data_input2_crp_base.acoustic_model_post_config = None
_returnn_rasr_data_input2_crp_base.corpus_config = None
_returnn_rasr_data_input2_crp_base.corpus_post_config = None
_returnn_rasr_data_input2_crp_base.lexicon_config = None
_returnn_rasr_data_input2_crp_base.lexicon_post_config = None
_returnn_rasr_data_input2_crp_base.language_model_config = None
_returnn_rasr_data_input2_crp_base.language_model_post_config = None
_returnn_rasr_data_input2_crp_base.recognizer_config = None
_returnn_rasr_data_input2_crp_base.recognizer_post_config = None
_returnn_rasr_data_input2_crp_base.log_config = rasr.RasrConfig()
_returnn_rasr_data_input2_crp_base.log_config['*'].configuration.channel = 'output-channel'
_returnn_rasr_data_input2_crp_base.log_config['*'].real_time_factor.channel = 'output-channel'
_returnn_rasr_data_input2_crp_base.log_config['*'].system_info.channel = 'output-channel'
_returnn_rasr_data_input2_crp_base.log_config['*'].time.channel = 'output-channel'
_returnn_rasr_data_input2_crp_base.log_config['*'].version.channel = 'output-channel'
_returnn_rasr_data_input2_crp_base.log_config['*'].log.channel = 'output-channel'
_returnn_rasr_data_input2_crp_base.log_config['*'].warning.channel = 'output-channel, stderr'
_returnn_rasr_data_input2_crp_base.log_config['*'].error.channel = 'output-channel, stderr'
_returnn_rasr_data_input2_crp_base.log_config['*'].statistics.channel = 'output-channel'
_returnn_rasr_data_input2_crp_base.log_config['*'].progress.channel = 'output-channel'
_returnn_rasr_data_input2_crp_base.log_config['*'].dot.channel = 'nil'
_returnn_rasr_data_input2_crp_base.log_post_config = rasr.RasrConfig()
_returnn_rasr_data_input2_crp_base.log_post_config['*'].encoding = 'UTF-8'
_returnn_rasr_data_input2_crp_base.log_post_config['*'].output_channel.file = '$(LOGFILE)'
_returnn_rasr_data_input2_crp_base.log_post_config['*'].output_channel.compressed = False
_returnn_rasr_data_input2_crp_base.log_post_config['*'].output_channel.append = False
_returnn_rasr_data_input2_crp_base.log_post_config['*'].output_channel.unbuffered = False
_returnn_rasr_data_input2_crp_base.compress_log_file = True
_returnn_rasr_data_input2_crp_base.default_log_channel = 'output-channel'
_returnn_rasr_data_input2_crp_base.audio_format = 'wav'
_returnn_rasr_data_input2_crp_base.corpus_duration = 1.0
_returnn_rasr_data_input2_crp_base.concurrent = 1
_returnn_rasr_data_input2_crp_base.segment_path = None
_returnn_rasr_data_input2_crp_base.acoustic_model_trainer_exe = None
_returnn_rasr_data_input2_crp_base.allophone_tool_exe = None
_returnn_rasr_data_input2_crp_base.costa_exe = None
_returnn_rasr_data_input2_crp_base.feature_extraction_exe = None
_returnn_rasr_data_input2_crp_base.feature_statistics_exe = None
_returnn_rasr_data_input2_crp_base.flf_tool_exe = None
_returnn_rasr_data_input2_crp_base.kws_tool_exe = None
_returnn_rasr_data_input2_crp_base.lattice_processor_exe = None
_returnn_rasr_data_input2_crp_base.lm_util_exe = None
_returnn_rasr_data_input2_crp_base.nn_trainer_exe = None
_returnn_rasr_data_input2_crp_base.speech_recognizer_exe = None
_returnn_rasr_data_input2_crp_base.python_home = None
_returnn_rasr_data_input2_crp_base.python_program_name = None
_returnn_rasr_data_input2.crp = rasr.CommonRasrParameters(_returnn_rasr_data_input2_crp_base)
_returnn_rasr_data_input2.crp.corpus_config = rasr.RasrConfig()
_returnn_rasr_data_input2.crp.corpus_config.file = tk.Path('merged.xml.gz', creator=_merge_corpora_job)
_returnn_rasr_data_input2.crp.corpus_config.audio_dir = None
_returnn_rasr_data_input2.crp.corpus_config.warn_about_unexpected_elements = True
_returnn_rasr_data_input2.crp.corpus_config.capitalize_transcriptions = False
_returnn_rasr_data_input2.crp.corpus_config.progress_indication = 'global'
_returnn_rasr_data_input2.crp.corpus_config.segment_order_shuffle = True
_returnn_rasr_data_input2.crp.corpus_config.segment_order_sort_by_time_length = True
_returnn_rasr_data_input2.crp.corpus_config.segment_order_sort_by_time_length_chunk_size = 384
_returnn_rasr_data_input2.crp.audio_format = 'wav'
_returnn_rasr_data_input2.crp.corpus_duration = 960.9000000000001
_returnn_rasr_data_input2.crp.concurrent = 1
_tail_job = make_fake_job(module='i6_core.text.processing', name='TailJob', sis_hash='zN9qqvtxote7')
_returnn_rasr_data_input2.crp.segment_path = tk.Path('out.gz', creator=_tail_job)
_returnn_rasr_data_input2.crp.acoustic_model_trainer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/acoustic-model-trainer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/acoustic-model-trainer.linux-x86_64-standard', ))
_returnn_rasr_data_input2.crp.allophone_tool_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/allophone-tool.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/allophone-tool.linux-x86_64-standard', ))
_returnn_rasr_data_input2.crp.costa_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/costa.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/costa.linux-x86_64-standard', ))
_returnn_rasr_data_input2.crp.feature_extraction_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/feature-extraction.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/feature-extraction.linux-x86_64-standard', ))
_returnn_rasr_data_input2.crp.feature_statistics_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/feature-statistics.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/feature-statistics.linux-x86_64-standard', ))
_returnn_rasr_data_input2.crp.flf_tool_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/flf-tool.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/flf-tool.linux-x86_64-standard', ))
_returnn_rasr_data_input2.crp.kws_tool_exe = None
_returnn_rasr_data_input2.crp.lattice_processor_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/lattice-processor.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/lattice-processor.linux-x86_64-standard', ))
_returnn_rasr_data_input2.crp.lm_util_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/lm-util.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/lm-util.linux-x86_64-standard', ))
_returnn_rasr_data_input2.crp.nn_trainer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/nn-trainer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/nn-trainer.linux-x86_64-standard', ))
_returnn_rasr_data_input2.crp.speech_recognizer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/speech-recognizer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/speech-recognizer.linux-x86_64-standard', ))
_returnn_rasr_data_input2.crp.lexicon_config = rasr.RasrConfig()
_returnn_rasr_data_input2.crp.lexicon_config.file = tk.Path('oov.lexicon.gz', creator=_g2_p_output_to_bliss_lexicon_job, cached=True)
_returnn_rasr_data_input2.crp.lexicon_config.normalize_pronunciation = False
_returnn_rasr_data_input2.crp.acoustic_model_post_config = rasr.RasrConfig()
_returnn_rasr_data_input2.crp.acoustic_model_post_config.allophones.add_from_file = tk.Path('allophones', creator=_store_allophones_job)
_returnn_rasr_data_input2.alignments = _returnn_rasr_data_input.alignments
_returnn_rasr_data_input2.feature_flow = _returnn_rasr_data_input.feature_flow
_returnn_rasr_data_input2.features = _multi_output_path1
_returnn_rasr_data_input2.acoustic_mixtures = tk.Path('am.mix', creator=_estimate_mixtures_job, cached=True)
_returnn_rasr_data_input2.feature_scorers = _returnn_rasr_data_input.feature_scorers
_returnn_rasr_data_input2.shuffle_data = True
_dict17 = {
    'train-other-960.devtrain': _returnn_rasr_data_input2,
}
_returnn_rasr_data_input3 = object.__new__(i6_experiments.common.setups.rasr.util.nn.ReturnnRasrDataInput)
_returnn_rasr_data_input3.name = 'init'
_returnn_rasr_data_input3_crp_base = rasr.CommonRasrParameters()
_returnn_rasr_data_input3_crp_base.acoustic_model_config = rasr.RasrConfig()
_returnn_rasr_data_input3_crp_base.acoustic_model_config.state_tying.type = 'cart'
_returnn_rasr_data_input3_crp_base.acoustic_model_config.state_tying.file = tk.Path('cart.tree.xml.gz', creator=_estimate_cart_job)
_returnn_rasr_data_input3_crp_base.acoustic_model_config.allophones.add_from_lexicon = True
_returnn_rasr_data_input3_crp_base.acoustic_model_config.allophones.add_all = False
_returnn_rasr_data_input3_crp_base.acoustic_model_config.hmm.states_per_phone = 3
_returnn_rasr_data_input3_crp_base.acoustic_model_config.hmm.state_repetitions = 1
_returnn_rasr_data_input3_crp_base.acoustic_model_config.hmm.across_word_model = True
_returnn_rasr_data_input3_crp_base.acoustic_model_config.hmm.early_recombination = False
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp.scale = 1.0
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp['*'].loop = 3.0
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp['*'].forward = 0.0
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp['*'].skip = 30.0
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp['*'].exit = 0.0
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp.silence.loop = 0.0
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp.silence.forward = 3.0
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp.silence.skip = 'infinity'
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp.silence.exit = 20.0
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp.entry_m1.loop = 'infinity'
_returnn_rasr_data_input3_crp_base.acoustic_model_config.tdp.entry_m2.loop = 'infinity'
_returnn_rasr_data_input3_crp_base.acoustic_model_post_config = None
_returnn_rasr_data_input3_crp_base.corpus_config = None
_returnn_rasr_data_input3_crp_base.corpus_post_config = None
_returnn_rasr_data_input3_crp_base.lexicon_config = None
_returnn_rasr_data_input3_crp_base.lexicon_post_config = None
_returnn_rasr_data_input3_crp_base.language_model_config = None
_returnn_rasr_data_input3_crp_base.language_model_post_config = None
_returnn_rasr_data_input3_crp_base.recognizer_config = None
_returnn_rasr_data_input3_crp_base.recognizer_post_config = None
_returnn_rasr_data_input3_crp_base.log_config = rasr.RasrConfig()
_returnn_rasr_data_input3_crp_base.log_config['*'].configuration.channel = 'output-channel'
_returnn_rasr_data_input3_crp_base.log_config['*'].real_time_factor.channel = 'output-channel'
_returnn_rasr_data_input3_crp_base.log_config['*'].system_info.channel = 'output-channel'
_returnn_rasr_data_input3_crp_base.log_config['*'].time.channel = 'output-channel'
_returnn_rasr_data_input3_crp_base.log_config['*'].version.channel = 'output-channel'
_returnn_rasr_data_input3_crp_base.log_config['*'].log.channel = 'output-channel'
_returnn_rasr_data_input3_crp_base.log_config['*'].warning.channel = 'output-channel, stderr'
_returnn_rasr_data_input3_crp_base.log_config['*'].error.channel = 'output-channel, stderr'
_returnn_rasr_data_input3_crp_base.log_config['*'].statistics.channel = 'output-channel'
_returnn_rasr_data_input3_crp_base.log_config['*'].progress.channel = 'output-channel'
_returnn_rasr_data_input3_crp_base.log_config['*'].dot.channel = 'nil'
_returnn_rasr_data_input3_crp_base.log_post_config = rasr.RasrConfig()
_returnn_rasr_data_input3_crp_base.log_post_config['*'].encoding = 'UTF-8'
_returnn_rasr_data_input3_crp_base.log_post_config['*'].output_channel.file = '$(LOGFILE)'
_returnn_rasr_data_input3_crp_base.log_post_config['*'].output_channel.compressed = False
_returnn_rasr_data_input3_crp_base.log_post_config['*'].output_channel.append = False
_returnn_rasr_data_input3_crp_base.log_post_config['*'].output_channel.unbuffered = False
_returnn_rasr_data_input3_crp_base.compress_log_file = True
_returnn_rasr_data_input3_crp_base.default_log_channel = 'output-channel'
_returnn_rasr_data_input3_crp_base.audio_format = 'wav'
_returnn_rasr_data_input3_crp_base.corpus_duration = 1.0
_returnn_rasr_data_input3_crp_base.concurrent = 1
_returnn_rasr_data_input3_crp_base.segment_path = None
_returnn_rasr_data_input3_crp_base.acoustic_model_trainer_exe = None
_returnn_rasr_data_input3_crp_base.allophone_tool_exe = None
_returnn_rasr_data_input3_crp_base.costa_exe = None
_returnn_rasr_data_input3_crp_base.feature_extraction_exe = None
_returnn_rasr_data_input3_crp_base.feature_statistics_exe = None
_returnn_rasr_data_input3_crp_base.flf_tool_exe = None
_returnn_rasr_data_input3_crp_base.kws_tool_exe = None
_returnn_rasr_data_input3_crp_base.lattice_processor_exe = None
_returnn_rasr_data_input3_crp_base.lm_util_exe = None
_returnn_rasr_data_input3_crp_base.nn_trainer_exe = None
_returnn_rasr_data_input3_crp_base.speech_recognizer_exe = None
_returnn_rasr_data_input3_crp_base.python_home = None
_returnn_rasr_data_input3_crp_base.python_program_name = None
_returnn_rasr_data_input3.crp = rasr.CommonRasrParameters(_returnn_rasr_data_input3_crp_base)
_returnn_rasr_data_input3.crp.corpus_config = rasr.RasrConfig()
_bliss_change_encoding_job = make_fake_job(module='i6_core.audio.encoding', name='BlissChangeEncodingJob', sis_hash='vUdgDkgc97ZK')
_returnn_rasr_data_input3.crp.corpus_config.file = tk.Path('corpus.xml.gz', creator=_bliss_change_encoding_job)
_returnn_rasr_data_input3.crp.corpus_config.audio_dir = None
_returnn_rasr_data_input3.crp.corpus_config.warn_about_unexpected_elements = True
_returnn_rasr_data_input3.crp.corpus_config.capitalize_transcriptions = False
_returnn_rasr_data_input3.crp.corpus_config.progress_indication = 'global'
_returnn_rasr_data_input3.crp.audio_format = 'wav'
_returnn_rasr_data_input3.crp.corpus_duration = 5.3
_returnn_rasr_data_input3.crp.concurrent = 20
_segment_corpus_job = make_fake_job(module='i6_core.corpus.segments', name='SegmentCorpusJob', sis_hash='OGYXX2IUHkLb')
_dict19 = {
    1: tk.Path('segments.1', creator=_segment_corpus_job),
    2: tk.Path('segments.2', creator=_segment_corpus_job),
    3: tk.Path('segments.3', creator=_segment_corpus_job),
    4: tk.Path('segments.4', creator=_segment_corpus_job),
    5: tk.Path('segments.5', creator=_segment_corpus_job),
    6: tk.Path('segments.6', creator=_segment_corpus_job),
    7: tk.Path('segments.7', creator=_segment_corpus_job),
    8: tk.Path('segments.8', creator=_segment_corpus_job),
    9: tk.Path('segments.9', creator=_segment_corpus_job),
    10: tk.Path('segments.10', creator=_segment_corpus_job),
    11: tk.Path('segments.11', creator=_segment_corpus_job),
    12: tk.Path('segments.12', creator=_segment_corpus_job),
    13: tk.Path('segments.13', creator=_segment_corpus_job),
    14: tk.Path('segments.14', creator=_segment_corpus_job),
    15: tk.Path('segments.15', creator=_segment_corpus_job),
    16: tk.Path('segments.16', creator=_segment_corpus_job),
    17: tk.Path('segments.17', creator=_segment_corpus_job),
    18: tk.Path('segments.18', creator=_segment_corpus_job),
    19: tk.Path('segments.19', creator=_segment_corpus_job),
    20: tk.Path('segments.20', creator=_segment_corpus_job),
}
_returnn_rasr_data_input3.crp.segment_path = i6_core.util.MultiOutputPath(
    _segment_corpus_job,
    'segments.$(TASK)',
    _dict19,
)
_returnn_rasr_data_input3.crp.acoustic_model_trainer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/acoustic-model-trainer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/acoustic-model-trainer.linux-x86_64-standard', ))
_returnn_rasr_data_input3.crp.allophone_tool_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/allophone-tool.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/allophone-tool.linux-x86_64-standard', ))
_returnn_rasr_data_input3.crp.costa_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/costa.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/costa.linux-x86_64-standard', ))
_returnn_rasr_data_input3.crp.feature_extraction_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/feature-extraction.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/feature-extraction.linux-x86_64-standard', ))
_returnn_rasr_data_input3.crp.feature_statistics_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/feature-statistics.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/feature-statistics.linux-x86_64-standard', ))
_returnn_rasr_data_input3.crp.flf_tool_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/flf-tool.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/flf-tool.linux-x86_64-standard', ))
_returnn_rasr_data_input3.crp.kws_tool_exe = None
_returnn_rasr_data_input3.crp.lattice_processor_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/lattice-processor.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/lattice-processor.linux-x86_64-standard', ))
_returnn_rasr_data_input3.crp.lm_util_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/lm-util.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/lm-util.linux-x86_64-standard', ))
_returnn_rasr_data_input3.crp.nn_trainer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/nn-trainer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/nn-trainer.linux-x86_64-standard', ))
_returnn_rasr_data_input3.crp.speech_recognizer_exe = tk.Path(os.path.join(gs.RASR_ROOT, '/speech-recognizer.linux-x86_64-standard'), hash_overwrite=(None, 'RASR_ROOT/speech-recognizer.linux-x86_64-standard', ))
_returnn_rasr_data_input3.crp.lexicon_config = rasr.RasrConfig()
_merge_lexicon_job = make_fake_job(module='i6_core.lexicon.modification', name='MergeLexiconJob', sis_hash='9BsYzLqAlJeq')
_returnn_rasr_data_input3.crp.lexicon_config.file = tk.Path('lexicon.xml.gz', creator=_merge_lexicon_job)
_returnn_rasr_data_input3.crp.lexicon_config.normalize_pronunciation = False
_returnn_rasr_data_input3.crp.language_model_config = rasr.RasrConfig()
_returnn_rasr_data_input3.crp.language_model_config.type = 'ARPA'
_download_job = make_fake_job(module='i6_core.tools.download', name='DownloadJob', sis_hash='6ij8dDC1z4zK')
_returnn_rasr_data_input3.crp.language_model_config.file = tk.Path('4-gram.arpa.gz', creator=_download_job)
import sisyphus.job_path
_returnn_rasr_data_input3.crp.language_model_config.scale = object.__new__(sisyphus.job_path.Variable)
_optimize_a_mand_l_m_scale_job = make_fake_job(module='i6_core.recognition.optimize_parameters', name='OptimizeAMandLMScaleJob', sis_hash='y9ETphimgLX0')
_dict20 = {
    'creator': _optimize_a_mand_l_m_scale_job,
    'path': 'bast_lm_score',
    'cached': False,
    '_hash_overwrite': None,
    '_tags': None,
    '_available': None,
    'pickle': False,
    'backup': None,
}
_returnn_rasr_data_input3.crp.language_model_config.scale.__setstate__(_dict20)
_returnn_rasr_data_input3.alignments = None
_returnn_rasr_data_input3.feature_flow = object.__new__(i6_core.rasr.flow.FlowNetwork)
_returnn_rasr_data_input3.feature_flow.name = 'network'
_named_flow_attribute1 = object.__new__(i6_core.rasr.flow.NamedFlowAttribute)
_named_flow_attribute1.name = 'cache'
_named_flow_attribute1.value = object.__new__(i6_core.rasr.flow.FlagDependentFlowAttribute)
_named_flow_attribute1.value.flag = 'cache_mode'
_feature_extraction_job1 = make_fake_job(module='i6_core.features.extraction', name='FeatureExtractionJob', sis_hash='Gammatone.BWIJ9k2uU6VK')
_dict22 = {
    1: tk.Path('gt.cache.1', creator=_feature_extraction_job1, cached=True),
    2: tk.Path('gt.cache.2', creator=_feature_extraction_job1, cached=True),
    3: tk.Path('gt.cache.3', creator=_feature_extraction_job1, cached=True),
    4: tk.Path('gt.cache.4', creator=_feature_extraction_job1, cached=True),
    5: tk.Path('gt.cache.5', creator=_feature_extraction_job1, cached=True),
    6: tk.Path('gt.cache.6', creator=_feature_extraction_job1, cached=True),
    7: tk.Path('gt.cache.7', creator=_feature_extraction_job1, cached=True),
    8: tk.Path('gt.cache.8', creator=_feature_extraction_job1, cached=True),
    9: tk.Path('gt.cache.9', creator=_feature_extraction_job1, cached=True),
    10: tk.Path('gt.cache.10', creator=_feature_extraction_job1, cached=True),
    11: tk.Path('gt.cache.11', creator=_feature_extraction_job1, cached=True),
    12: tk.Path('gt.cache.12', creator=_feature_extraction_job1, cached=True),
    13: tk.Path('gt.cache.13', creator=_feature_extraction_job1, cached=True),
    14: tk.Path('gt.cache.14', creator=_feature_extraction_job1, cached=True),
    15: tk.Path('gt.cache.15', creator=_feature_extraction_job1, cached=True),
    16: tk.Path('gt.cache.16', creator=_feature_extraction_job1, cached=True),
    17: tk.Path('gt.cache.17', creator=_feature_extraction_job1, cached=True),
    18: tk.Path('gt.cache.18', creator=_feature_extraction_job1, cached=True),
    19: tk.Path('gt.cache.19', creator=_feature_extraction_job1, cached=True),
    20: tk.Path('gt.cache.20', creator=_feature_extraction_job1, cached=True),
}
_multi_output_path2 = i6_core.util.MultiOutputPath(
    _feature_extraction_job1,
    'gt.cache.$(TASK)',
    _dict22,
    cached=True
)
_named_flow_attribute1.value.alternatives = {
    'task_dependent': _multi_output_path2,
    'bundle': tk.Path('gt.cache.bundle', creator=_feature_extraction_job1, cached=True),
}
_dict21 = {
    'id': '$(id)',
    'path': _named_flow_attribute1,
    'filter': 'generic-cache',
}
_returnn_rasr_data_input3.feature_flow.nodes = {
    'cache': _dict21,
}
_returnn_rasr_data_input3.feature_flow.links = [('cache', 'network:features', )]
_returnn_rasr_data_input3.feature_flow.inputs = set()
_returnn_rasr_data_input3.feature_flow.outputs = {'features'}
_returnn_rasr_data_input3.feature_flow.params = {'id'}
_returnn_rasr_data_input3.feature_flow.named_attributes = {
    'cache': _named_flow_attribute1,
}
_returnn_rasr_data_input3.feature_flow.hidden_inputs = set()
_returnn_rasr_data_input3.feature_flow.flags = {
    'cache_mode': 'task_dependent',
}
_returnn_rasr_data_input3.feature_flow.config = None
_returnn_rasr_data_input3.feature_flow.post_config = None
_returnn_rasr_data_input3.features = _multi_output_path2
_returnn_rasr_data_input3.acoustic_mixtures = None
_returnn_rasr_data_input3.feature_scorers = {
    'estimate_mixtures_sdm.vtln+sat': None,
    'train_vtln+sat': None,
}
_returnn_rasr_data_input3.shuffle_data = True
_dict18 = {
    'dev-other': _returnn_rasr_data_input3,
}
_dict23 = {
}
obj = {
    'hybrid_init_args': _rasr_init_args,
    'train_data': _dict12,
    'cv_data': _dict16,
    'devtrain_data': _dict17,
    'dev_data': _dict18,
    'test_data': _dict23,
    'train_cv_pairing': [('train-other-960.train', 'train-other-960.cv', )],
}
