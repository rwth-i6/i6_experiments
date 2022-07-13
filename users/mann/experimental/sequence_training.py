__all__ = ['add_bw_output', 'add_bw_layer']

import i6_core.rasr as rasr
SprintCommand = rasr.RasrCommand

def add_bw_output(csp, crnn_config, am_scale=1.0, ce_smoothing=0.0,
                  import_model=None, exp_average=0.001, 
                  prior_scale=1.0, tdp_scale=1.0,
                  extra_config=None, extra_post_config=None):

  # Prepare output layer to compute sequence loss
  if crnn_config['use_tensorflow']:
    crnn_config['network']['output']['loss_scale'] = ce_smoothing

    crnn_config['network']['accumulate_prior'] = {}
    crnn_config['network']['accumulate_prior']['class'] = "accumulate_mean"
    crnn_config['network']['accumulate_prior']['from'] = ['output']
    crnn_config['network']['accumulate_prior']["exp_average"] =  exp_average
    crnn_config['network']['accumulate_prior']["is_prob_distribution"] = True

    crnn_config['network']['combine_prior'] = {}
    crnn_config['network']['combine_prior']['class'] = 'combine'
    crnn_config['network']['combine_prior']['from'] = ['output', 'accumulate_prior']
    crnn_config['network']['combine_prior']['kind']  = 'eval'
    crnn_config['network']['combine_prior']['eval']  = "safe_log(source(0)) * am_scale - safe_log(source(1)) * prior_scale"
    crnn_config['network']['combine_prior']['eval_locals'] = {'am_scale': am_scale, 'prior_scale': prior_scale}

    crnn_config['network']['fast_bw'] = {}
    crnn_config['network']['fast_bw']['class'] = 'fast_bw'
    crnn_config['network']['fast_bw']['from']  = ['combine_prior']
    crnn_config['network']['fast_bw']['align_target'] = 'sprint'
    crnn_config['network']['fast_bw']['tdp_scale'] = tdp_scale
    crnn_config['network']['fast_bw']['sprint_opts'] = {
     "sprintExecPath":       SprintCommand.select_exe(csp.nn_trainer_exe, 'nn-trainer'),
     "sprintConfigStr":      "--config=fastbw.config",
     "sprintControlConfig":  {"verbose": True},
     "usePythonSegmentOrder": False,
     "numInstances": 1}

    crnn_config['network']['output_bw']               = {}
    crnn_config['network']['output_bw']['class']      = 'copy'
    crnn_config['network']['output_bw']['from']       = 'output'
    crnn_config['network']['output_bw']['loss_scale'] = 1 - ce_smoothing
    crnn_config['network']['output_bw']['loss']       = 'via_layer'
    crnn_config['network']['output_bw']['loss_opts']  = {"loss_wrt_to_act_in": "softmax", "align_layer": "fast_bw"}

  else: # Use Theano
    assert False, "Theano implementation of bw training not supportet yet."

  if 'chunking' in crnn_config:
   del crnn_config['chunking']
  if 'pretrain' in crnn_config and import_model is not None:
   del crnn_config['pretrain']

  # start training from existing model
  if import_model is not None:
   crnn_config['import_model_train_epoch1'] = str(import_model)[:-5 if crnn_config['use_tensorflow'] else None]

  # Create additional Sprint config file to compute losses
  mapping = { 'corpus': 'neural-network-trainer.corpus',
             'lexicon': [
               'neural-network-trainer.alignment-fsa-exporter.model-combination.lexicon'],
             'acoustic_model': [
               'neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model']
             }

  config, post_config = rasr.build_config_from_mapping(csp,mapping)
  post_config['*'].output_channel.file = 'fastbw.log'

  # Define action
  config.neural_network_trainer.action    = 'python-control'

  # neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder
  config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.allow_for_silence_repetitions                          = False
  config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.normalize_lemma_sequence_scores                        = True
  # neural_network_trainer.alignment_fsa_exporter.alignment-fsa-exporter
  config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.fix_allophone_context_at_word_boundaries         = True
  config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.transducer_builder_filter_out_invalid_allophones = True

  # additional config
  config._update(extra_config)
  post_config._update(extra_post_config)


  additional_sprint_config_files      = {'fastbw' : config}
  additional_sprint_post_config_files = {'fastbw' : post_config}


  return crnn_config, additional_sprint_config_files, additional_sprint_post_config_files


def add_bw_layer(csp, crnn_config, am_scale=1.0, ce_smoothing=0.0,
                 import_model=None, exp_average=0.001, 
                 prior_scale=1.0, tdp_scale=1.0):

  # Prepare output layer to compute sequence loss
  if crnn_config['use_tensorflow']:
    crnn_config['network']['output']['loss_scale'] = ce_smoothing

    crnn_config['network']['accumulate_prior'] = {}
    crnn_config['network']['accumulate_prior']['class'] = "accumulate_mean"
    crnn_config['network']['accumulate_prior']['from'] = ['output']
    crnn_config['network']['accumulate_prior']["exp_average"] =  exp_average
    crnn_config['network']['accumulate_prior']["is_prob_distribution"] = True

    crnn_config['network']['combine_prior'] = {}
    crnn_config['network']['combine_prior']['class'] = 'combine'
    crnn_config['network']['combine_prior']['from'] = ['output', 'accumulate_prior']
    crnn_config['network']['combine_prior']['kind']  = 'eval'
    crnn_config['network']['combine_prior']['eval']  = "safe_log(source(0)) * am_scale - safe_log(source(1)) * prior_scale"
    crnn_config['network']['combine_prior']['eval_locals'] = {'am_scale': am_scale, 'prior_scale': prior_scale}

    crnn_config['network']['fast_bw'] = {}
    crnn_config['network']['fast_bw']['class'] = 'fast_bw'
    crnn_config['network']['fast_bw']['from']  = ['combine_prior']
    crnn_config['network']['fast_bw']['align_target'] = 'sprint'
    crnn_config['network']['fast_bw']['tdp_scale'] = tdp_scale
    crnn_config['network']['fast_bw']['sprint_opts'] = {
     "sprintExecPath":       SprintCommand.select_exe(csp.nn_trainer_exe, 'nn-trainer'),
     "sprintConfigStr":      "--config=fastbw.config",
     "sprintControlConfig":  {"verbose": True},
     "usePythonSegmentOrder": False,
     "numInstances": 1}

    crnn_config['network']['output_bw']               = {}
    crnn_config['network']['output_bw']['class']      = 'copy'
    crnn_config['network']['output_bw']['from']       = 'output'
    crnn_config['network']['output_bw']['loss_scale'] = 1 - ce_smoothing
    crnn_config['network']['output_bw']['loss']       = 'via_layer'
    crnn_config['network']['output_bw']['loss_opts']  = {"loss_wrt_to_act_in": "softmax", "align_layer": "fast_bw"}

  else: # Use Theano
    assert False, "Theano implementation of bw training not supportet yet."

  if 'chunking' in crnn_config:
   del crnn_config['chunking']
  if 'pretrain' in crnn_config and import_model is not None:
   del crnn_config['pretrain']

  # start training from existing model
  if import_model is not None:
   crnn_config['import_model_train_epoch1'] = str(import_model)[:-5 if crnn_config['use_tensorflow'] else None]


def add_fastbw_configs(csp,
    extra_config=None,
    extra_post_config=None,
    acoustic_model_extra_config=None,
    fix_tdps_applicator = False,
  ):
  # Create additional Sprint config file to compute losses
  mapping = {
    'corpus': 'neural-network-trainer.corpus',
    'lexicon': [
      'neural-network-trainer.alignment-fsa-exporter.model-combination.lexicon'],
    'acoustic_model': [
      'neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model']
  }

  config, post_config = rasr.build_config_from_mapping(csp, mapping)
  post_config['*'].output_channel.file = 'fastbw.log'

  # Define action
  config.neural_network_trainer.action    = 'python-control'

  # neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder
  config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.allow_for_silence_repetitions                          = False
  config.neural_network_trainer.alignment_fsa_exporter.allophone_state_graph_builder.orthographic_parser.normalize_lemma_sequence_scores                        = True

  if fix_tdps_applicator:
    config["*"].fix_tdp_leaving_epsilon_arc = True
  # neural_network_trainer.alignment_fsa_exporter.alignment-fsa-exporter
  config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.fix_allophone_context_at_word_boundaries         = True
  config.neural_network_trainer.alignment_fsa_exporter.alignment_fsa_exporter.model_combination.acoustic_model.transducer_builder_filter_out_invalid_allophones = True

  # additional config
  print(type(acoustic_model_extra_config))
  assert isinstance(acoustic_model_extra_config, rasr.RasrConfig)
  config.neural_network_trainer.alignment_fsa_exporter.model_combination.acoustic_model._update(acoustic_model_extra_config)
  config._update(extra_config)
  post_config._update(extra_post_config)


  additional_sprint_config_files      = {'fastbw' : config}
  additional_sprint_post_config_files = {'fastbw' : post_config}


  return additional_sprint_config_files, additional_sprint_post_config_files
