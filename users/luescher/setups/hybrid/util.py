__all__ = ["GmmDataInput", "GmmPipelineArgs", "NnArgs"]

import i6_core.meta as meta


class GmmDataInput:
    def __init__(
        self,
        corpus_object: meta.CorpusObject = None,
        lexicon: dict = None,
        lm: dict = None,
        concurrent: int = 10,
    ):
        """
        :param corpus_object: corpus_file: Path, audio_dir: Path, audio_format: str, duration: float
        :param lexicon: file: Path, normalize_pronunciation: bool
        :param lm: file: Path, type: str, scale: float
        :param concurrent: concurrency for gmm hmm pipeline
        """
        self.corpus_object = corpus_object
        self.lexicon = lexicon
        self.lm = lm
        self.concurrent = concurrent


class GmmPipelineArgs:
    def __init__(
        self,
        costa_args: dict = None,
        am_args: dict = None,
        feature_extraction_args: dict = None,
        linear_alignment_args: dict = None,
        monophone_training_args: dict = None,
        monophone_recognition_args: dict = None,
        cart_questions=None,
        cart_lda_args: dict = None,
        triphone_training_args: dict = None,
        triphone_recognition_args: dict = None,
        sdm_tri_args: dict = None,
        vtln_training_args: dict = None,
        vtln_recognition_args: dict = None,
        sat_training_args: dict = None,
        sat_recognition_args: dict = None,
        sdm_vtln_args: dict = None,
        vtln_sat_training_args: dict = None,
        vtln_sat_recognition_args: dict = None,
    ):
        """
        ##################################################
        :param costa_args"
        {
          'eval_recordings': True,
          'eval_lm': True
        }
        ##################################################
        :param am_args:
        {
          state_tying="monophone",
          states_per_phone=3,
          state_repetitions=1,
          across_word_model=True,
          early_recombination=False,
          tdp_scale=1.0,
          tdp_transition=(3.0, 0.0, 30.0, 0.0),  # loop, forward, skip, exit
          tdp_silence=(0.0, 3.0, "infinity", 20.0),
          tying_type="global",
          nonword_phones="",
          tdp_nonword=(0.0, 3.0, "infinity", 6.0)  # only used when tying_type = global-and-nonword
        }
        ##################################################
        :param feature_extraction_args:
          'mfcc':
          {
            'num_deriv': 2,
            'num_features': None,  # confusing name: number of max features, above number -> clipped
            'mfcc_options':
            {
              'warping_function': "mel",
              'filter_width': 268.258,  # 80
              'normalize': True,
              'normalization_options': None,
              'without_samples': False,
              'samples_options': {'audio_format': "wav",
                                  'dc_detection': True,},
              'cepstrum_options': {'normalize': False,
                                   'outputs': 16,
                                   'add_epsilon': False,},
              'fft_options': None,
            }
          }
          'gt':
          {
            'minfreq': 100,
            'maxfreq': 7500,
            'channels': 50,
            'warp_freqbreak': None,  # 3700
            'tempint_type': 'hanning',
            'tempint_shift': .01,
            'tempint_length': .025,
            'flush_before_gap': True,
            'do_specint': False,
            'specint_type': 'hanning',
            'specint_shift': 4,
            'specint_length': 9,
            'normalize': True,
            'preemphasis': True,
            'legacy_scaling': False,
            'without_samples': False,
            'samples_options': {'audio_format': "wav",
                                'dc_detection': True},
            'normalization_options': {},
          }
          'fb':
          {
            'warping_function': "mel",
            'filter_width': 80,
            'normalize': True,
            'normalization_options': None,
            'without_samples': False,
            'samples_options': {'audio_format': "wav",
                                'dc_detection': True},
            'fft_options': None,
            'apply_log': True,
            'add_epsilon': False,
          }
          'energy':
          {
              'without_samples': False,
              'samples_options': {'audio_format': "wav",
                          'dc_detection': True},
              'fft_options': {},
          }
        ##################################################
        :param linear_alignment_args:
          'minimum_segment_length': 0,
          'maximum_segment_length': 6000,
          'iterations': 20,
          'penalty': 0,
          'minimum_speech_proportion': .7,
          'save_alignment': False,
          'keep_accumulators': False,
          'extra_merge_args': None,
          'extra_config': None,
          'extra_post_config': None,
        ##################################################
        :param monophone_training_args:
          'feature_flow': 'mfcc+deriv+norm',
          'feature_energy_flow': 'energy,mfcc+deriv+norm',
          'align_iter': 75,
          'splits': 10,
          'accs_per_split': 2,
        :param monophone_recognition_args:
          'eval_iter': [7, 8, 9, 10]
          'pronunciation_scales': [10.0, 11.0, 12.0, 13.0, 14.0]
          'lm_scales': [9.0, 9.25, 9.50, 9.75, 10.0, 10.25, 10.50]
          'recog_args': {
                          'feature_flow': dev_corpus_name,
                          'pronunciation_scale': pronunciation_scale,
                          'lm_scale': lm_scale,
                          'lm_lookahead': True,
                          'lookahead_options': None,
                          'create_lattice': True,
                          'eval_single_best': True,
                          'eval_best_in_lattice': True,
                          'search_parameters': {
                            'beam_pruning': 18.0,
                            'beam-pruning-limit': 100000,
                            'word-end-pruning': 0.75,
                            'word-end-pruning-limit': 15000
                          },
                          'best_path_algo': 'bellman-ford',  # options: bellman-ford, dijkstra
                          'fill_empty_segments': False,
                          'scorer': recog.Sclite,
                          'scorer_args': {'ref': create_corpora.stm_files['dev-other']},
                          'scorer_hyp_args': "hyp",
                          'rtf': 50,
                          'mem': 8,
                          'use_gpu': False,
                        }
        ##################################################
        :param cart_questions:
          - cart generation class
            - phoneme_path, max_leaves, min_obs
            - phonemes, steps, max_leaves, hmm_states
          - file path
        :param cart_lda_args:
          'initial_flow': 'mfcc+deriv+norm',  # is feature_flow from monophone training
          'context_flow': 'mfcc',
          'context_size': 9,
          'num_dim': 48,
          'num_iter': 2,
          'eigenvalue_args': {},
          'generalized_eigenvalue_args': {'all': {'verification_tolerance': 1e15}},
          'alignment': "train_mono",  # if using run function not needed
        ##################################################
        :param triphone_training_args:
          'feature_flow': 'mfcc+context+lda',
          'splits': 10,
          'accs_per_split': 2,
          'initial_alignment': "train_mono",  # if using run function not needed
        :param triphone_recognition_args:
        ##################################################
        :param sdm_tri_args:
          'feature_flow': "mfcc+context+lda",
          'alignment': "train_tri",   # if using run function not needed
        :param vtln_training_args:
          'feature_flow': {
                            'base_flow': 'uncached_mfcc',
                            'context_size': 9,
                            'lda_matrix': "{corpus_name}_{mono}"  # if using run function not needed
                          }
          'warp_mix': {
                        'splits': 8,
                        'accs_per_split': 2,
                        'feature_scorer': "estimate_mixtures_sdm.tri"  # if using run function not needed
                      }
          'train': {
                      'splits': 10,
                      'accs_per_split': 2,
                      'initial_alignment': "train_tri",  # if using run function not needed
                      'feature_flow': "mfcc+context+lda+vtln",
                    }
        :param vtln_recognition_args:
        ##################################################
        :param sat_training_args:
          'feature_cache': 'mfcc+context+lda',
          'cache_regex': '^mfcc.*$',
          'splits': 10,
          'accs_per_split': 2,
          'mixtures': "estimate_mixtures_sdm.tri",  # if using run function not needed
          'align_keep_values': {7: tk.gs.JOB_DEFAULT_KEEP_VALUE},
          'feature_flow': "mfcc+context+lda",
          'alignment': "train_tri",   # if using run function not needed
        :param sat_recognition_args:
        ##################################################
        :param sdm_vtln_args:
          'feature_flow': "mfcc+context+lda+vtln",
          'alignment': "train_vtln",  # if using run function not needed
        :param vtln_sat_training_args:
          'feature_cache': "mfcc+context+lda+vtln",
          'cache_regex': '^.*\\+vtln$',
          'mixtures': "estimate_mixtures_sdm.vtln"  # if using run function not needed
          'splits': 10,
          'accs_per_split': 2,
          'feature_flow': "mfcc+context+lda+vtln",
          'alignment': "train_vtln",  # if using run function not needed
        :param vtln_sat_recognition_args:
        ##################################################

        if necessary set:

        extra_config = sprint.SprintConfig()
        extra_config.allow_zero_weights = True
        {accumulate,split,align}_extra_args = {'extra_config': extra_config}

        '{accumulate,split,align}_extra_rqmt': {'mem': 10, 'time': 8},

        vtln align time = 8

        if not using the run function -> name and corpus almost always to be added

        """
        self.costa_args = costa_args
        self.am_args = am_args
        self.feature_extraction_args = feature_extraction_args
        self.linear_alignment_args = linear_alignment_args
        self.monophone_training_args = monophone_training_args
        self.monophone_recognition_args = monophone_recognition_args
        self.cart_questions = cart_questions
        self.cart_lda_args = cart_lda_args
        self.triphone_training_args = triphone_training_args
        self.triphone_recognition_args = triphone_recognition_args
        self.sdm_tri_args = sdm_tri_args
        self.vtln_training_args = vtln_training_args
        self.vtln_recognition_args = vtln_recognition_args
        self.sat_training_args = sat_training_args
        self.sat_recognition_args = sat_recognition_args
        self.sdm_vtln_args = sdm_vtln_args
        self.vtln_sat_training_args = vtln_sat_training_args
        self.vtln_sat_recognition_args = vtln_sat_recognition_args


class NnArgs:
    def __init__(
        self,
        am_args: dict = None,
        feature_extraction_args: dict = None,
        nn_ce_training_args: dict = None,
        nn_ce_count_recognition_args: dict = None,
        nn_ce_lstm_recognition_args: dict = None,
        nn_ce_trafo_recognition_args: dict = None,
        nn_ce_rescoring_args: dict = None,
        nn_smbr_training_args: dict = None,
        nn_smbr_count_recognition_args: dict = None,
        nn_smbr_lstm_recognition_args: dict = None,
        nn_smbr_trafo_recognition_args: dict = None,
        nn_smbr_rescoring_args: dict = None,
    ):
        """
        ##################################################
        :param am_args:
        {
          state_tying="monophone",
          states_per_phone=3,
          state_repetitions=1,
          across_word_model=True,
          early_recombination=False,
          tdp_scale=1.0,
          tdp_transition=(3.0, 0.0, 30.0, 0.0),  # loop, forward, skip, exit
          tdp_silence=(0.0, 3.0, "infinity", 20.0),
          tying_type="global",
          nonword_phones="",
          tdp_nonword=(0.0, 3.0, "infinity", 6.0)  # only used when tying_type = global-and-nonword
        }
        ##################################################
        :param feature_extraction_args:
          'mfcc':
          {
            'num_deriv': 2,
            'num_features': None,  # confusing name: number of max features, above number -> clipped
            'mfcc_options':
            {
              'warping_function': "mel",
              'filter_width': 268.258,  # 80
              'normalize': True,
              'normalization_options': None,
              'without_samples': False,
              'samples_options': {'audio_format': "wav",
                                  'dc_detection': True,},
              'cepstrum_options': {'normalize': False,
                                   'outputs': 16,
                                   'add_epsilon': False,},
              'fft_options': None,
            }
          }
          'gt':
          {
            'minfreq': 100,
            'maxfreq': 7500,
            'channels': 50,
            'warp_freqbreak': None,  # 3700
            'tempint_type': 'hanning',
            'tempint_shift': .01,
            'tempint_length': .025,
            'flush_before_gap': True,
            'do_specint': False,
            'specint_type': 'hanning',
            'specint_shift': 4,
            'specint_length': 9,
            'normalize': True,
            'preemphasis': True,
            'legacy_scaling': False,
            'without_samples': False,
            'samples_options': {'audio_format': "wav",
                                'dc_detection': True},
            'normalization_options': {},
          }
          'fb':
          {
            'warping_function': "mel",
            'filter_width': 80,
            'normalize': True,
            'normalization_options': None,
            'without_samples': False,
            'samples_options': {'audio_format': "wav",
                                'dc_detection': True},
            'fft_options': None,
            'apply_log': True,
            'add_epsilon': False,
          }
          'energy':
          {
              'without_samples': False,
              'samples_options': {'audio_format': "wav",
                          'dc_detection': True},
              'fft_options': {},
          }
        ##################################################

        ##################################################
        :param monophone_recognition_args:
          'eval_iter': [7, 8, 9, 10]
          'pronunciation_scales': [10.0, 11.0, 12.0, 13.0, 14.0]
          'lm_scales': [9.0, 9.25, 9.50, 9.75, 10.0, 10.25, 10.50]
          'recog_args': {
                          'feature_flow': dev_corpus_name,
                          'pronunciation_scale': pronunciation_scale,
                          'lm_scale': lm_scale,
                          'lm_lookahead': True,
                          'lookahead_options': None,
                          'create_lattice': True,
                          'eval_single_best': True,
                          'eval_best_in_lattice': True,
                          'search_parameters': {
                            'beam_pruning': 18.0,
                            'beam-pruning-limit': 100000,
                            'word-end-pruning': 0.75,
                            'word-end-pruning-limit': 15000
                          },
                          'best_path_algo': 'bellman-ford',  # options: bellman-ford, dijkstra
                          'fill_empty_segments': False,
                          'scorer': recog.Sclite,
                          'scorer_args': {'ref': create_corpora.stm_files['dev-other']},
                          'scorer_hyp_args': "hyp",
                          'rtf': 50,
                          'mem': 8,
                          'use_gpu': False,
                        }
        """
        self.am_args = am_args
        self.feature_extraction_args = feature_extraction_args
        self.nn_ce_training_args = nn_ce_training_args
        self.nn_ce_count_recognition_args = nn_ce_count_recognition_args
        self.nn_ce_lstm_recognition_args = nn_ce_lstm_recognition_args
        self.nn_ce_trafo_recognition_args = nn_ce_trafo_recognition_args
        self.nn_ce_rescoring_args = nn_ce_rescoring_args
        self.nn_smbr_training_args = nn_smbr_training_args
        self.nn_smbr_count_recognition_args = nn_smbr_count_recognition_args
        self.nn_smbr_lstm_recognition_args = nn_smbr_lstm_recognition_args
        self.nn_smbr_trafo_recognition_args = nn_smbr_trafo_recognition_args
        self.nn_smbr_rescoring_args = nn_smbr_rescoring_args
